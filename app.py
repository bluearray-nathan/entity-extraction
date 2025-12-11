import streamlit as st
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
# Note: webdriver_manager is removed to prevent version conflicts
from google.cloud import language_v1
from google.oauth2 import service_account
from google import genai
from google.genai import types
import pandas as pd
import json
import time

# --- 1. AUTHENTICATION & SETUP ---

# A. Google Cloud NLP (Service Account)
if "gcp_service_account" in st.secrets:
    try:
        service_account_info = dict(st.secrets["gcp_service_account"])
        credentials = service_account.Credentials.from_service_account_info(service_account_info)
        nlp_client = language_v1.LanguageServiceClient(credentials=credentials)
    except Exception as e:
        st.error(f"Error loading GCP Credentials: {e}")
        st.stop()
else:
    st.error("Missing 'gcp_service_account' in secrets.")
    st.stop()

# B. Gemini API
client = None
if "gemini_api_key" in st.secrets:
    try:
        raw_secret = st.secrets["gemini_api_key"]
        # Robust key extraction
        api_key_string = raw_secret if isinstance(raw_secret, str) else raw_secret.get("gemini_api_key") or raw_secret.get("api_key")
        
        if not api_key_string:
            st.error("Found gemini_api_key but no valid string inside.")
            st.stop()

        client = genai.Client(api_key=api_key_string)
    except Exception as e:
        st.error(f"Gemini Error: {e}")
        st.stop()
else:
    st.error("Missing 'gemini_api_key' in secrets.")
    st.stop()

# C. Selenium Browser Setup (Fixed for Cloud)
@st.cache_resource
def get_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    
    # CRITICAL FIX: Explicitly use the system-installed binary
    chrome_options.binary_location = "/usr/bin/chromium"

    # CRITICAL FIX: Explicitly use the system-installed driver
    service = Service("/usr/bin/chromedriver")
    
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

# --- 2. CORE FUNCTIONS ---

def scrape_with_selenium(url, exclude_phrases=None):
    """
    Uses Selenium to render JS, then scrapes text and cleans it.
    """
    try:
        driver = get_driver()
        driver.get(url)
        
        # Small wait for JS to settle
        time.sleep(2) 
        
        # Get the full text of the body
        full_text = driver.find_element(By.TAG_NAME, "body").text
        
        # The "Black Box" Cleaner
        if exclude_phrases:
            for phrase in exclude_phrases:
                phrase = phrase.strip()
                if phrase:
                    full_text = full_text.replace(phrase, "")
        
        return full_text.strip()
        
    except Exception as e:
        return f"ERROR: {str(e)}"

def analyze_entities(text):
    """Google NLP Entity Extraction"""
    try:
        # Cap at 100k chars for NLP API safety
        nlp_text = text[:100000] 
        
        document = language_v1.Document(content=nlp_text, type_=language_v1.Document.Type.PLAIN_TEXT)
        response = nlp_client.analyze_entities(request={'document': document})
        
        sorted_entities = sorted(response.entities, key=lambda x: x.salience, reverse=True)
        if not sorted_entities: return None, []
        
        main = {"name": sorted_entities[0].name, "score": sorted_entities[0].salience}
        subs = [e.name for e in sorted_entities[1:10]] # Grab top 10
        return main, subs
    except Exception as e:
        st.error(f"NLP API Error: {e}")
        return None, []

def llm_audit_gemini(url, main_entity, sub_entities, model_name):
    """Gemini Audit"""
    prompt = f"""
    You are a technical SEO Auditor.
    URL: {url}
    Main Entity Detected: "{main_entity['name']}" (Salience Score: {main_entity['score']:.2f})
    Sub-Entities Detected: {", ".join(sub_entities)}
    
    Task: Audit if the content actually focuses on the user's search intent.
    
    1. If Main Entity is generic (e.g. 'Home', 'Login', 'Cookies', Brand Name), ignore it.
    2. Look at the Sub-Entities. Do they match the likely topic of the URL?
    3. Is the content 'thin' or 'off-topic'?
    
    Return JSON: {{ "verdict": "Pass/Fail/Review", "reasoning": "...", "recommendation": "..." }}
    """
    try:
        response = client.models.generate_content(
            model=model_name, 
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        return json.loads(response.text)
    except Exception as e:
        return {"verdict": "Error", "reasoning": str(e), "recommendation": "Check Model"}

# --- 3. THE UI ---
st.set_page_config(page_title="Selenium Entity Auditor", layout="wide")
st.title("🚀 Selenium + Gemini Entity Auditor")

# SIDEBAR
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model Selection
    model_options = ["gemini-1.5-flash"]
    try:
        if client:
            m_list = client.models.list()
            model_options = [m.name for m in m_list if "gemini" in m.name and "vision" not in m.name]
            model_options.sort(reverse=True)
    except: pass
    selected_model = st.selectbox("Gemini Model", model_options)

    st.divider()
    
    # The Excluder Box
    st.subheader("🧹 Cleaner")
    st.info("Paste junk text here (headers, cookie banners) to remove it from analysis.")
    exclude_text = st.text_area("Phrases to Remove:", height=150)

# MAIN INPUT
urls_input = st.text_area("Enter URLs (one per line):", height=100)

if st.button("Run Audit", type="primary"):
    if not urls_input:
        st.warning("Enter a URL first.")
    else:
        urls = urls_input.strip().split('\n')
        # Prepare exclude list
        excludes = exclude_text.split('\n') if exclude_text else []
        
        results = []
        progress = st.progress(0)
        status = st.empty()
        
        for i, url in enumerate(urls):
            url = url.strip()
            if not url: continue
            
            status.text(f"Scraping with Selenium: {url}...")
            
            # 1. SCRAPE
            text = scrape_with_selenium(url, exclude_phrases=excludes)
            
            # Show preview
            with st.expander(f"📄 View Scraped Text ({len(text)} chars)"):
                st.text_area("Full Body Text", text, height=300)
            
            if "ERROR" in text or len(text) < 50:
                results.append({"URL": url, "Verdict": "Fail", "Reasoning": "Scrape failed or empty", "Action": "Check URL"})
                continue

            # 2. ANALYZE
            status.text(f"Analyzing entities for: {url}...")
            main_ent, sub_ents = analyze_entities(text)
            
            if not main_ent:
                results.append({"URL": url, "Verdict": "Error", "Reasoning": "No entities found", "Action": "Check content length"})
                continue
                
            # 3. AUDIT
            audit = llm_audit_gemini(url, main_ent, sub_ents, selected_model)
            
            results.append({
                "URL": url,
                "Main Entity": f"{main_ent['name']} ({main_ent['score']:.2f})",
                "Verdict": audit.get("verdict"),
                "Reasoning": audit.get("reasoning"),
                "Action": audit.get("recommendation")
            })
            
            progress.progress((i + 1) / len(urls))
            
        status.success("Done!")
        
        # Display Results
        df = pd.DataFrame(results)
        def color_verdict(val):
            color = 'red' if val in ['Fail', 'Error'] else 'green' if val == 'Pass' else 'orange'
            return f'color: {color}; font-weight: bold'
            
        st.dataframe(df.style.map(color_verdict, subset=['Verdict']), use_container_width=True)

