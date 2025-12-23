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
st.set_page_config(page_title="Entity Optimizer", layout="wide")
ACTIVE_GEMINI_MODEL = "gemini-2.0-flash"

# --- 2. AUTHENTICATION ---
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

client = None
if "gemini_api_key" in st.secrets:
    try:
        raw_secret = st.secrets["gemini_api_key"]
        api_key_string = raw_secret if isinstance(raw_secret, str) else raw_secret.get("gemini_api_key") or raw_secret.get("api_key")
        client = genai.Client(api_key=api_key_string)
    except Exception as e:
        st.error(f"Gemini Error: {e}")
        st.stop()

# --- 3. BROWSER SETUP ---
@st.cache_resource
def get_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.binary_location = "/usr/bin/chromium"
    service = Service("/usr/bin/chromedriver")
    return webdriver.Chrome(service=service, options=chrome_options)

if 'results_df' not in st.session_state:
    st.session_state.results_df = None

# --- 4. CORE FUNCTIONS ---
def scrape_with_selenium(url, exclude_phrases=None):
    try:
        driver = get_driver()
        driver.get(url)
        time.sleep(3)
        full_text = driver.find_element(By.TAG_NAME, "body").text
        if exclude_phrases:
            for phrase in exclude_phrases:
                if phrase.strip(): full_text = full_text.replace(phrase.strip(), "")
        return full_text.strip()
    except Exception as e:
        return f"ERROR: {str(e)}"

def deduplicate_entities(entities):
    merged = {}
    for e in entities:
        clean_name = e.name.lower().rstrip('s')
        if clean_name in merged:
            merged[clean_name]['score'] += e.salience
        else:
            merged[clean_name] = {'name': e.name, 'score': e.salience}
    return sorted(merged.values(), key=lambda x: x['score'], reverse=True)

def analyze_entities(text):
    try:
        document = language_v1.Document(content=text[:100000], type_=language_v1.Document.Type.PLAIN_TEXT)
        response = nlp_client.analyze_entities(request={'document': document})
        cleaned = deduplicate_entities(response.entities)
        if not cleaned: return None, []
        return cleaned[0], cleaned[1:10]
    except Exception as e:
        return None, []

def get_gemini_optimization_advice(target_focus, main_entity, sub_entities, text_sample):
    # Format sub-entities for the prompt
    sub_list = ", ".join([f"{s['name']} ({s['score']:.2f})" for s in sub_entities])
    
    prompt = f"""
    You are an SEO Entity Specialist. 
    USER TARGET FOCUS: "{target_focus}"
    GOOGLE NLP RESULT: Main Entity is "{main_entity['name']}" (Salience: {main_entity['score']:.2f}).
    SUB-ENTITIES FOUND: {sub_list}
    
    TASK: 
    1. Compare the Target Focus to the NLP results. 
    2. Does Google "get it"? (Yes/No/Partial)
    3. Provide specific advice to align the page better with the Target Focus. 
    4. Give 2-3 concrete examples of copy changes (e.g., "Change [Existing Sentence] to [Proposed Sentence]").
    
    TEXT SAMPLE FOR CONTEXT: "{text_sample[:1500]}"
    
    OUTPUT JSON ONLY:
    {{
      "alignment_status": "Status here",
      "optimization_advice": "Your detailed analysis here",
      "actionable_examples": "Specific copy changes here"
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
        return {"alignment_status": "Error", "optimization_advice": str(e), "actionable_examples": "N/A"}

# --- 5. THE UI ---
st.title("🎯 Entity Alignment & Optimizer")

# SIDEBAR Settings
with st.sidebar:
    st.header("⚙️ Settings")
    exclude_text = st.text_area("Phrases to ignore (CSV):", placeholder="Login, Sign Up")
    st.divider()
    st.info("This tool compares what you WANT the page to be about vs. what Google NLP ACTUALLY sees.")

# MAIN INPUTS
target_focus = st.text_input("🎯 What is the main focus/topic of this page?", placeholder="e.g. Comprehensive Car Insurance for Seniors")

tab_auto, tab_manual = st.tabs(["🌐 Automatic (URL)", "📝 Manual (Paste Content)"])
input_items = []

with tab_auto:
    urls_input = st.text_area("Enter URLs (one per line):", height=100)
    if urls_input:
        for url in urls_input.strip().split('\n'):
            if url.strip(): input_items.append({"type": "url", "value": url.strip(), "label": url.strip()})

with tab_manual:
    m_label = st.text_input("Label:", placeholder="Competitor Page A")
    m_text = st.text_area("Paste text content:", height=200)
    if m_text: input_items.append({"type": "raw", "value": m_text, "label": m_label or "Manual Entry"})

# PROCESSING
if st.button("Analyze Alignment", type="primary"):
    if not target_focus:
        st.error("Please enter the 'Target Focus' so the AI knows what to compare against.")
    elif not input_items:
        st.warning("Please provide a URL or paste text content.")
    else:
        results = []
        progress = st.progress(0)
        
        raw_excludes = exclude_text.replace('\n', ',').split(',')
        excludes = [x.strip() for x in raw_excludes if x.strip()]
        
        for i, item in enumerate(input_items):
            # 1. Scrape/Get Text
            if item["type"] == "url":
                text = scrape_with_selenium(item["value"], excludes)
            else:
                text = item["value"]
                for p in excludes: 
                    if p: text = text.replace(p, "")
            
            if "ERROR" in text or len(text) < 50:
                results.append({"Source": item['label'], "Main Entity": "Error", "Alignment": "N/A", "Advice": "Check Source Text"})
                continue

            # 2. NLP Analysis
            main_ent, sub_ents = analyze_entities(text)
            if not main_ent:
                results.append({"Source": item['label'], "Main Entity": "None", "Alignment": "N/A", "Advice": "No Entities Found"})
                continue
            
            # 3. Gemini Advice
            advice = get_gemini_optimization_advice(target_focus, main_ent, sub_ents, text)
            
            results.append({
                "Source": item['label'],
                "Main Entity (Score)": f"{main_ent['name']} ({main_ent['score']:.2f})",
                "Target Alignment": advice.get("alignment_status"),
                "Optimization Advice": advice.get("optimization_advice"),
                "Actionable Examples": advice.get("actionable_examples")
            })
            progress.progress((i + 1) / len(input_items))
            
        st.session_state.results_df = pd.DataFrame(results)

# DISPLAY
if st.session_state.results_df is not None:
    st.divider()
    st.subheader("📊 Optimization Results")
    
    # Download Button
    csv = st.session_state.results_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Analysis", data=csv, file_name="entity_optimization.csv", mime="text/csv")
    
    # Custom Table Display
    st.table(st.session_state.results_df)

    # Individual Card View for better readability of long text
    for index, row in st.session_state.results_df.iterrows():
        with st.expander(f"🔍 Detailed View: {row['Source']}"):
            st.write(f"**Main Entity:** {row['Main Entity (Score)']}")
            st.write(f"**Alignment:** {row['Target Alignment']}")
            st.info(f"**Advice:** {row['Optimization Advice']}")
            st.success(f"**Examples:**\n{row['Actionable Examples']}")
