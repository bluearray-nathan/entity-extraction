import streamlit as st
import requests
from bs4 import BeautifulSoup
from google.cloud import language_v1
from google.oauth2 import service_account
from google import genai
from google.genai import types
import pandas as pd
import json

# --- 1. CONFIGURATION & AUTH ---

# A. Google Cloud NLP API (Service Account)
if "gcp_service_account" in st.secrets:
    try:
        service_account_info = dict(st.secrets["gcp_service_account"])
        credentials = service_account.Credentials.from_service_account_info(service_account_info)
        nlp_client = language_v1.LanguageServiceClient(credentials=credentials)
    except Exception as e:
        st.error(f"Error loading GCP Credentials: {e}")
        st.stop()
else:
    st.error("GCP Credentials (gcp_service_account) not found in secrets.")
    st.stop()

# B. Google Gemini API (API Key)
client = None
if "gemini_api_key" in st.secrets:
    try:
        raw_secret = st.secrets["gemini_api_key"]
        api_key_string = None
        
        # LOGIC: Extract key string
        if isinstance(raw_secret, str):
            api_key_string = raw_secret
        else:
            secret_dict = dict(raw_secret)
            if "gemini_api_key" in secret_dict:
                api_key_string = secret_dict["gemini_api_key"]
            elif "api_key" in secret_dict:
                api_key_string = secret_dict["api_key"]
            else:
                for v in secret_dict.values():
                    if isinstance(v, str) and v.startswith("AIza"):
                        api_key_string = v
                        break
        
        if not api_key_string:
            st.error("❌ Error: Found [gemini_api_key] but no valid string key inside.")
            st.stop()

        client = genai.Client(api_key=api_key_string)
        
    except Exception as e:
        st.error(f"Error initializing Gemini: {e}")
        st.stop()
else:
    st.error("Gemini API Key (gemini_api_key) not found in secrets.")
    st.stop()

# --- 2. CORE FUNCTIONS ---

def scrape_body_text(url, exclude_selectors=None):
    """
    Scrapes visible text from a URL with stealth headers.
    Allows removing specific elements via CSS selectors (e.g. '.navbar').
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Referer': 'https://www.google.com/',
            'Upgrade-Insecure-Requests': '1'
        }
        
        session = requests.Session()
        response = session.get(url, headers=headers, timeout=15)
        
        if response.status_code == 403:
            return "ERROR: Access Denied (403). The website blocked the scraper."
        if response.status_code != 200:
            return f"ERROR: Status Code {response.status_code}"

        soup = BeautifulSoup(response.content, 'html.parser')
        
        # 1. Standard Cleanup (Always remove these)
        for element in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "iframe"]):
            element.extract()

        # 2. Custom User Exclusion (The new feature)
        if exclude_selectors:
            for selector in exclude_selectors:
                selector = selector.strip()
                if not selector: continue
                try:
                    # Select matches and remove them
                    found_elements = soup.select(selector)
                    for el in found_elements:
                        el.extract()
                except Exception as e:
                    # Ignore invalid selectors so the rest of the scrape still works
                    print(f"Invalid selector '{selector}': {e}")
            
        text = soup.get_text(separator=' ')
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        clean_text = ' '.join(chunk for chunk in chunks if chunk)
        
        return clean_text[:8000]
    except Exception as e:
        return f"ERROR: {str(e)}"

def analyze_entities(text):
    """Sends text to Google NLP API to get entities."""
    try:
        document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
        response = nlp_client.analyze_entities(request={'document': document})
        
        sorted_entities = sorted(response.entities, key=lambda x: x.salience, reverse=True)
        
        if not sorted_entities:
            return None, []
        
        main_entity = {
            "name": sorted_entities[0].name,
            "score": sorted_entities[0].salience
        }
        sub_entities = [e.name for e in sorted_entities[1:6]]
        
        return main_entity, sub_entities
    except Exception as e:
        st.error(f"🚨 Google NLP API Error: {e}")
        return None, []

def llm_audit_gemini(url, main_entity_data, sub_entities, model_name):
    """Asks Gemini to audit the entity alignment."""
    
    prompt = f"""
    You are a technical SEO Auditor.
    URL: {url}
    Main Entity: "{main_entity_data['name']}" (Score: {main_entity_data['score']:.2f})
    Sub-Entities: {", ".join(sub_entities)}
    
    Task: Audit content focus.
    
    RULE: If 'Main Entity' is generic (e.g. 'Member', 'Login', 'Home', Brand Name), ignore it and check if Sub-Entities match the URL topic.
    
    Return ONLY JSON:
    {{
        "verdict": "Pass" or "Review Needed",
        "reasoning": "Short explanation",
        "recommendation": "Actionable tip"
    }}
    """
    
    try:
        response = client.models.generate_content(
            model=model_name, 
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        return json.loads(response.text)
    except Exception as e:
        return {"verdict": "Error", "reasoning": str(e), "recommendation": "Check Model ID"}

# --- 3. STREAMLIT UI ---
st.set_page_config(page_title="Entity Auditor", layout="wide")

st.title("🔹 Google NLP + Gemini Entity Auditor")

# --- SIDEBAR SETTINGS ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model Selector
    available_models = ["gemini-1.5-flash"]
    try:
        if client:
            models = client.models.list()
            available_models = [m.name for m in models if "gemini" in m.name and "vision" not in m.name]
            available_models.sort(reverse=True)
    except:
        pass
    selected_model = st.selectbox("Select Gemini Model:", available_models, index=0)
    
    st.markdown("---")
    
    # NEW: Class Excluder Input
    st.subheader("🧹 Scraping Cleaner")
    st.info("Remove specific site elements (e.g. headers, cookie banners) to improve accuracy.")
    exclude_input = st.text_input(
        "Exclude CSS Selectors:", 
        placeholder=".navbar, #cookie-banner, .login-header",
        help="Enter classes (.) or IDs (#) separated by commas. These parts of the page will be deleted before analysis."
    )

urls_input = st.text_area("Enter URLs (one per line):", height=150)

if st.button("Run Audit", type="primary"):
    if not urls_input:
        st.warning("Please enter at least one URL.")
    else:
        urls = urls_input.strip().split('\n')
        
        # Parse user exclusion list
        exclude_list = [x.strip() for x in exclude_input.split(',')] if exclude_input else []
        
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, url in enumerate(urls):
            url = url.strip()
            if not url: continue
            
            status_text.text(f"Processing: {url}")
            
            # 1. Scrape (Passing the exclusion list)
            text = scrape_body_text(url, exclude_selectors=exclude_list)
            
            with st.expander(f"🕵️‍♂️ View Scraped Text for: {url}"):
                if text and "ERROR" in text:
                    st.error(text)
                else:
                    st.text(text[:1000] + "..." if text else "No text found")
            
            if not text or "ERROR" in text:
                results.append({"URL": url, "Status": "Scrape Blocked", "Verdict": "Fail", "Reasoning": text, "Action": "Check Headers"})
                continue
            
            # 2. NLP
            main_entity, sub_entities = analyze_entities(text)
            if not main_entity:
                results.append({"URL": url, "Status": "NLP Failed", "Verdict": "Fail", "Reasoning": "API Error", "Action": "Check API Console"})
                continue
            
            # 3. Gemini Audit
            audit = llm_audit_gemini(url, main_entity, sub_entities, selected_model)
            
            results.append({
                "URL": url,
                "Main Entity": f"{main_entity['name']}",
                "Salience": f"{main_entity['score']:.2f}",
                "Sub Entities": ", ".join(sub_entities),
                "Verdict": audit.get("verdict"),
                "Reasoning": audit.get("reasoning"),
                "Action": audit.get("recommendation")
            })
            
            progress_bar.progress((i + 1) / len(urls))
            
        status_text.text("Audit Complete!")
        progress_bar.empty()
        
        df = pd.DataFrame(results)
        if not df.empty:
            def highlight_verdict(val):
                color = 'red' if val in ["Review Needed", "Fail", "Error"] else 'green' if val == "Pass" else 'orange'
                return f'color: {color}; font-weight: bold'
            try:
                st.dataframe(df.style.map(highlight_verdict, subset=['Verdict']), use_container_width=True)
            except:
                st.dataframe(df, use_container_width=True)

