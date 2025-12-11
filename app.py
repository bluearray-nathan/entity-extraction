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
        # Force conversion to a standard Python dictionary to avoid "AttrDict" errors
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
if "gemini_api_key" in st.secrets:
    try:
        raw_secret = st.secrets["gemini_api_key"]
        api_key_string = None
        
        # LOGIC: Check if it's already a string (flat), or a dict (nested section)
        if isinstance(raw_secret, str):
            api_key_string = raw_secret
        else:
            # It's a dictionary/section. Let's look inside for the key.
            # We convert to a standard dict first to be safe.
            secret_dict = dict(raw_secret)
            
            # 1. Try exact match for the key name "gemini_api_key"
            if "gemini_api_key" in secret_dict:
                api_key_string = secret_dict["gemini_api_key"]
            # 2. Try 'api_key' (common variation)
            elif "api_key" in secret_dict:
                api_key_string = secret_dict["api_key"]
            # 3. Fallback: Search for ANY value starting with "AIza" (Google Keys always start with this)
            else:
                for v in secret_dict.values():
                    if isinstance(v, str) and v.startswith("AIza"):
                        api_key_string = v
                        break
        
        if not api_key_string:
            st.error("❌ Error: Found the [gemini_api_key] section, but could not find the actual key string inside.")
            st.write("Debug info - Keys found:", list(dict(raw_secret).keys()) if not isinstance(raw_secret, str) else "None")
            st.stop()

        # Initialize the Client with the cleaned string
        client = genai.Client(api_key=api_key_string)
        
    except Exception as e:
        st.error(f"Error initializing Gemini: {e}")
        st.stop()
else:
    st.error("Gemini API Key (gemini_api_key) not found in secrets.")
    st.stop()

# --- 2. CORE FUNCTIONS ---

def scrape_body_text(url):
    """Scrapes visible text from a URL with stealth headers."""
    try:
        # Mimic a real Chrome browser on Windows 10
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.google.com/',
            'Upgrade-Insecure-Requests': '1',
            'Connection': 'keep-alive'
        }
        
        # Use a Session to maintain cookies (helps with some blocks)
        session = requests.Session()
        response = session.get(url, headers=headers, timeout=15)
        
        # Check if we got blocked
        if response.status_code == 403:
            return "ERROR: Access Denied (403). The website likely blocked the scraper."
        if response.status_code != 200:
            return f"ERROR: Status Code {response.status_code}"

        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove non-content elements
        for element in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "iframe"]):
            element.extract()
            
        text = soup.get_text(separator=' ')
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        clean_text = ' '.join(chunk for chunk in chunks if chunk)
        
        return clean_text[:8000] # Limit to 8k chars to save tokens
    except Exception as e:
        return f"ERROR: {str(e)}"

def analyze_entities(text):
    """Sends text to Google NLP API to get entities."""
    try:
        document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
        response = nlp_client.analyze_entities(request={'document': document})
        
        # Sort entities by salience (importance)
        sorted_entities = sorted(response.entities, key=lambda x: x.salience, reverse=True)
        
        if not sorted_entities:
            return None, []
        
        # Main entity is the one with highest salience
        main_entity = {
            "name": sorted_entities[0].name,
            "score": sorted_entities[0].salience
        }
        
        # Get top 5 sub-entities
        sub_entities = [e.name for e in sorted_entities[1:6]]
        
        return main_entity, sub_entities
    except Exception as e:
        # --- DEBUG: Show the actual error on screen ---
        st.error(f"🚨 Google NLP API Error: {e}")
        return None, []

def llm_audit_gemini(url, main_entity_data, sub_entities):
    """Asks Gemini to audit the entity alignment using the new SDK."""
    
    prompt = f"""
    You are a technical SEO Auditor.
    
    I have a URL: {url}
    
    I ran Google NLP Entity Analysis on the body copy of this page.
    1. The Main Entity identified is: "{main_entity_data['name']}" (Salience Score: {main_entity_data['score']:.2f})
       *Note: A score closer to 1.0 means highly focused. A score below 0.10 often implies dilution.*
    2. The Sub-Entities are: {", ".join(sub_entities)}
    
    Your Task:
    Analyze if the content is focused on the correct topic based on the URL slug and the entities found.
    
    Return ONLY a JSON object with this exact structure:
    {{
        "verdict": "Pass" or "Review Needed",
        "reasoning": "A short, sharp explanation of why.",
        "recommendation": "One specific actionable recommendation."
    }}
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        return json.loads(response.text)
    except Exception as e:
        return {"verdict": "Error", "reasoning": str(e), "recommendation": "Check API"}

# --- 3. STREAMLIT UI ---
st.set_page_config(page_title="Entity Auditor", layout="wide")

st.title("🔹 Google NLP + Gemini Entity Auditor")
st.markdown("""
This tool scrapes a URL, runs it through **Google Cloud Natural Language API** to find the main entities, 
and then uses **Google Gemini** to determine if the page content is focused or diluted.
""")

with st.expander("ℹ️ How to interpret results"):
    st.info("""
    - **Salience Score:** 0.0 to 1.0. Higher is better. < 0.10 usually means the page has no clear focus.
    - **Verdict:** Generated by Gemini based on the relationship between the URL and the extracted entities.
    - **Note on RAC/Insurance:** These sites often block scrapers. Check the 'Debug' section if you see errors.
    """)

urls_input = st.text_area("Enter URLs (one per line):", height=150, placeholder="https://example.com/topic-a\nhttps://example.com/topic-b")

if st.button("Run Audit", type="primary"):
    if not urls_input:
        st.warning("Please enter at least one URL.")
    else:
        urls = urls_input.strip().split('\n')
        results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, url in enumerate(urls):
            url = url.strip()
            if not url: continue
            
            status_text.text(f"Processing: {url}")
            
            # 1. Scrape with Stealth
            text = scrape_body_text(url)
            
            # --- DEBUGGING VIEW ---
            # If the result is suspicious or an error, the user can now see why.
            with st.expander(f"🕵️‍♂️ View Scraped Text for: {url}"):
                if text and "ERROR" in text:
                    st.error(text)
                else:
                    st.text(text[:1000] + "..." if text else "No text found")
            
            if not text or "ERROR" in text:
                results.append({"URL": url, "Status": "Scrape Blocked", "Verdict": "Fail", "Reasoning": text, "Action": "Check Headers"})
                continue
            
            # 2. NLP API
            main_entity, sub_entities = analyze_entities(text)
            if not main_entity:
                results.append({"URL": url, "Status": "No Entities Found", "Verdict": "Fail", "Reasoning": "Google NLP found no topics. Text might be too short or blocked.", "Action": "Review Scraped Text"})
                continue
            
            # 3. Gemini Audit
            audit = llm_audit_gemini(url, main_entity, sub_entities)
            
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
        
        # Display Results
        df = pd.DataFrame(results)
        
        if not df.empty:
            # Stylize the dataframe to highlight "Review Needed"
            def highlight_verdict(val):
                color = 'red' if val in ["Review Needed", "Fail"] else 'green' if val == "Pass" else 'orange'
                return f'color: {color}; font-weight: bold'

            try:
                st.dataframe(df.style.map(highlight_verdict, subset=['Verdict']), use_container_width=True)
            except:
                st.dataframe(df, use_container_width=True)
        else:
            st.warning("No valid results generated.")


