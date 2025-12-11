import streamlit as st
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from google.cloud import language_v1
from google.oauth2 import service_account
from google import genai
from google.genai import types
import pandas as pd
import json
import time

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Entity Interpreter", layout="wide")

# HARDCODED MODEL VERSION
ACTIVE_GEMINI_MODEL = "gemini-1.5-flash"

# --- 2. AUTHENTICATION ---

# A. Google Cloud NLP
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

# C. Selenium Browser Setup
@st.cache_resource
def get_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    
    # System binaries for Streamlit Cloud
    chrome_options.binary_location = "/usr/bin/chromium"
    service = Service("/usr/bin/chromedriver")
    
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

# --- 3. SESSION STATE ---
if 'results_df' not in st.session_state:
    st.session_state.results_df = None

# --- 4. CORE FUNCTIONS ---

def scrape_with_selenium(url, exclude_phrases=None):
    try:
        driver = get_driver()
        driver.get(url)
        time.sleep(2)
        
        full_text = driver.find_element(By.TAG_NAME, "body").text
        
        # CLEANING LOGIC (Comma Separated)
        if exclude_phrases:
            for phrase in exclude_phrases:
                phrase = phrase.strip()
                if phrase:
                    full_text = full_text.replace(phrase, "")
        
        return full_text.strip()
    except Exception as e:
        return f"ERROR: {str(e)}"

def deduplicate_entities(entities):
    """
    Merges entities that are functionally the same.
    """
    merged = {}
    
    for e in entities:
        # Normalize name
        clean_name = e.name.lower().rstrip('s')
        
        if clean_name in merged:
            merged[clean_name]['score'] += e.salience
            if e.name[0].isupper() and not merged[clean_name]['name'][0].isupper():
                merged[clean_name]['name'] = e.name
        else:
            merged[clean_name] = {
                'name': e.name,
                'score': e.salience
            }
            
    sorted_merged = sorted(merged.values(), key=lambda x: x['score'], reverse=True)
    return sorted_merged

def analyze_entities(text):
    try:
        nlp_text = text[:100000]
        document = language_v1.Document(content=nlp_text, type_=language_v1.Document.Type.PLAIN_TEXT)
        response = nlp_client.analyze_entities(request={'document': document})
        
        # 1. Deduplicate
        cleaned_entities = deduplicate_entities(response.entities)
        
        if not cleaned_entities: return None, []
        
        # 2. Extract
        main = {"name": cleaned_entities[0]['name'], "score": cleaned_entities[0]['score']}
        subs = [{"name": e['name'], "score": e['score']} for e in cleaned_entities[1:10]]
        
        return main, subs
    except Exception as e:
        st.error(f"NLP API Error: {e}")
        return None, []

def get_gemini_explanation(url, main_entity, text_sample):
    prompt = f"""
    You are an expert in Google's Natural Language Processing (NLP) API.
    
    Data:
    - URL: {url}
    - Main Entity Identified: "{main_entity['name']}" (Salience Score: {main_entity['score']:.2f})
    
    Here is the start of the webpage text:
    "{text_sample}..."
    
    Task: Explain WHY the NLP algorithm chose "{main_entity['name']}" as the most important concept.
    
    Analyze:
    1. Grammatical Subject: Is "{main_entity['name']}" the 'doer' in most sentences?
    2. Structural Prominence: Is it in headings or navigation links?
    3. Repetition/Lemmatization: Are there related words that grouped together?
    
    Output JSON ONLY:
    {{
        "context": "A 1-sentence definition of what this entity represents in this specific text.",
        "explanation": "2-3 bullet points explaining the technical reason (grammar/structure) for the high score."
    }}
    """
    try:
        response = client.models.generate_content(
            model=ACTIVE_GEMINI_MODEL, 
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        return json.loads(response.text)
    except Exception as e:
        return {"context": "Error", "explanation": str(e)}

# --- 5. THE UI ---
st.title("🧠 Entity Analysis Interpreter")

with st.sidebar:
    st.header("⚙️ Settings")
    st.info(f"Using Model: **{ACTIVE_GEMINI_MODEL}**")
    
    st.divider()
    st.subheader("🧹 Text to Remove")
    st.caption("Paste specific navigation text or headers here. Separate different phrases with a comma (,).")
    
    exclude_text = st.text_area(
        "Phrases to delete:", 
        height=150,
        placeholder="Breakdown Cover | RAC, Skip to content, Cookie Policy"
    )

urls_input = st.text_area("Enter URLs (one per line):", height=100)

# --- PROCESSING BLOCK ---
if st.button("Analyze & Explain", type="primary"):
    if not urls_input:
        st.warning("Enter a URL first.")
    else:
        urls = urls_input.strip().split('\n')
        
        raw_excludes = exclude_text.replace('\n', ',').split(',')
        excludes = [x.strip() for x in raw_excludes if x.strip()]
        
        results = []
        progress = st.progress(0)
        status = st.empty()
        
        for i, url in enumerate(urls):
            url = url.strip()
            if not url: continue
            
            status.text(f"Processing: {url}...")
            
            # 1. Scrape
            text = scrape_with_selenium(url, exclude_phrases=excludes)
            
            with st.expander(f"📄 Text Source: {url}"):
                st.text_area("Content Preview", text, height=200)

            if "ERROR" in text or len(text) < 50:
                results.append({"URL": url, "Main Entity": "Error", "Explanation": "Scrape Failed"})
                continue

            # 2. NLP Analysis
            main_ent, sub_ents = analyze_entities(text)
            
            if not main_ent:
                results.append({"URL": url, "Main Entity": "None", "Explanation": "No entities found"})
                continue
                
            # 3. Gemini Interpretation
            explanation_data = get_gemini_explanation(url, main_ent, text[:3000])
            
            formatted_subs = ", ".join([f"{s['name']} ({s['score']:.2f})" for s in sub_ents])
            
            # --- CRITICAL FIX: SANITIZE GEMINI OUTPUT ---
            # Ensure 'explanation' is always a String, never a List.
            raw_expl = explanation_data.get("explanation", "")
            if isinstance(raw_expl, list):
                clean_expl = "\n".join([str(item) for item in raw_expl])
            else:
                clean_expl = str(raw_expl)

            raw_ctx = explanation_data.get("context", "")
            clean_ctx = str(raw_ctx)
            # --------------------------------------------

            results.append({
                "URL": url,
                "Main Entity": f"{main_ent['name']} ({main_ent['score']:.2f})",
                "Sub Entities": formatted_subs,
                "Context": clean_ctx,
                "Why it was picked": clean_expl
            })
            
            progress.progress((i + 1) / len(urls))
            
        status.success("Done!")
        
        st.session_state.results_df = pd.DataFrame(results)

# --- RESULT DISPLAY BLOCK ---
if st.session_state.results_df is not None:
    st.divider()
    
    col1, col2 = st.columns([4, 1])
    
    with col2:
        csv = st.session_state.results_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="entity_analysis_results.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    st.dataframe(st.session_state.results_df, use_container_width=True)
