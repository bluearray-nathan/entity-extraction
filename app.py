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

# --- 1. AUTHENTICATION & SETUP ---

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

# --- 2. CORE FUNCTIONS ---

def scrape_with_selenium(url, exclude_phrases=None):
    try:
        driver = get_driver()
        driver.get(url)
        time.sleep(2)
        
        full_text = driver.find_element(By.TAG_NAME, "body").text
        
        if exclude_phrases:
            for phrase in exclude_phrases:
                phrase = phrase.strip()
                if phrase:
                    full_text = full_text.replace(phrase, "")
        
        return full_text.strip()
    except Exception as e:
        return f"ERROR: {str(e)}"

def analyze_entities(text):
    try:
        nlp_text = text[:100000]
        document = language_v1.Document(content=nlp_text, type_=language_v1.Document.Type.PLAIN_TEXT)
        response = nlp_client.analyze_entities(request={'document': document})
        
        sorted_entities = sorted(response.entities, key=lambda x: x.salience, reverse=True)
        if not sorted_entities: return None, []
        
        main = {"name": sorted_entities[0].name, "score": sorted_entities[0].salience}
        subs = [{"name": e.name, "score": e.salience} for e in sorted_entities[1:10]]
        
        return main, subs
    except Exception as e:
        st.error(f"NLP API Error: {e}")
        return None, []

def get_gemini_explanation(url, main_entity, text_sample, model_name):
    """
    Asks Gemini to explain WHY the NLP API picked this entity.
    """
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
            model=model_name, 
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        return json.loads(response.text)
    except Exception as e:
        return {"context": "Error", "explanation": str(e)}

# --- 3. THE UI ---
st.set_page_config(page_title="Entity Interpreter", layout="wide")
st.title("🧠 Entity Analysis Interpreter")

with st.sidebar:
    st.header("⚙️ Settings")
    model_options = ["gemini-1.5-flash"]
    try:
        if client:
            m_list = client.models.list()
            model_options = [m.name for m in m_list if "gemini" in m.name and "vision" not in m.name]
            model_options.sort(reverse=True)
    except: pass
    selected_model = st.selectbox("Gemini Model", model_options)
    st.divider()
    st.subheader("🧹 Cleaner")
    exclude_text = st.text_area("Phrases to Remove:", height=150)

urls_input = st.text_area("Enter URLs (one per line):", height=100)

if st.button("Analyze & Explain", type="primary"):
    if not urls_input:
        st.warning("Enter a URL first.")
    else:
        urls = urls_input.strip().split('\n')
        excludes = exclude_text.split('\n') if exclude_text else []
        
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
                
            # 3. Gemini Interpretation (Passing text context)
            explanation_data = get_gemini_explanation(url, main_ent, text[:3000], selected_model)
            
            formatted_subs = ", ".join([f"{s['name']} ({s['score']:.2f})" for s in sub_ents])
            
            results.append({
                "URL": url,
                "Main Entity": f"{main_ent['name']} ({main_ent['score']:.2f})",
                "Sub Entities": formatted_subs,
                "Context": explanation_data.get("context"),
                "Why it was picked": explanation_data.get("explanation")
            })
            
            progress.progress((i + 1) / len(urls))
            
        status.success("Done!")
        
        # --- DATA DISPLAY & EXPORT ---
        df = pd.DataFrame(results)
        
        # 1. Create columns for Table vs Download button
        col1, col2 = st.columns([4, 1])
        
        with col2:
            # Convert DF to CSV
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="entity_analysis_results.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        # 2. Display the styled table
        st.dataframe(df, use_container_width=True)
