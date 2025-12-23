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
st.set_page_config(page_title="Entity Alignment Optimizer", layout="wide")
ACTIVE_GEMINI_MODEL = "gemini-2.5-flash"

# --- 2. AUTHENTICATION & INITIALIZATION ---
if 'manual_queue' not in st.session_state:
    st.session_state.manual_queue = []
if 'results_df' not in st.session_state:
    st.session_state.results_df = None

# Auth Logic
if "gcp_service_account" in st.secrets:
    try:
        service_account_info = dict(st.secrets["gcp_service_account"])
        credentials = service_account.Credentials.from_service_account_info(service_account_info)
        nlp_client = language_v1.LanguageServiceClient(credentials=credentials)
    except Exception as e:
        st.error(f"GCP Credentials Error: {e}"); st.stop()
else:
    st.error("Missing 'gcp_service_account' in secrets."); st.stop()

client = None
if "gemini_api_key" in st.secrets:
    try:
        raw_secret = st.secrets["gemini_api_key"]
        api_key_string = raw_secret if isinstance(raw_secret, str) else raw_secret.get("gemini_api_key") or raw_secret.get("api_key")
        client = genai.Client(api_key=api_key_string)
    except Exception as e:
        st.error(f"Gemini API Error: {e}"); st.stop()

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

# --- 4. CORE FUNCTIONS ---
def clean_output_text(text):
    """Removes markdown bolding (**) and ensures clean string output."""
    if isinstance(text, str):
        return text.replace("**", "").strip()
    return text

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
    sub_list = ", ".join([f"{s['name']} ({s['score']:.2f})" for s in sub_entities])
    prompt = f"""
    You are an SEO Entity Specialist. 
    TARGET FOCUS: "{target_focus}"
    GOOGLE NLP RESULT: Main Entity is "{main_entity['name']}" (Score: {main_entity['score']:.2f}).
    SUB-ENTITIES: {sub_list}
    
    TASK:
    1. Compare Target Focus vs NLP results. 
    2. Alignment Status: (Matched / Partial / Mismatched)
    3. Provide optimization advice and 2 specific copy changes.
    
    IMPORTANT: Do NOT use markdown bolding (**) in your response. 
    Output JSON ONLY.
    """
    try:
        response = client.models.generate_content(
            model=ACTIVE_GEMINI_MODEL, 
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        if response and response.text:
            return json.loads(response.text)
        return None
    except:
        return None

# --- 5. THE UI ---
st.title("🎯 Bulk Entity Optimizer (Clean View)")

with st.sidebar:
    st.header("⚙️ Global Settings")
    exclude_text = st.text_area("Phrases to ignore (CSV):", placeholder="Login, Get a Quote, Yell.com")
    st.divider()
    if st.button("🗑️ Reset All Data"):
        st.session_state.manual_queue = []
        st.session_state.results_df = None
        st.rerun()

tab_manual, tab_auto = st.tabs(["📝 Bulk Manual Queue", "🌐 Automatic (URL List)"])

with tab_manual:
    col1, col2 = st.columns(2)
    with col1:
        m_label = st.text_input("1. Page Label:", placeholder="e.g. Business Name")
    with col2:
        m_target = st.text_input("2. Target Focus:", placeholder="e.g. Roof Repairs")
    
    m_text = st.text_area("3. Paste content:", height=200, key="manual_input_area")
    
    if st.button("➕ Add to Queue"):
        if m_label and m_target and m_text:
            st.session_state.manual_queue.append({
                "label": m_label, "target": m_target, "content": m_text, "type": "raw"
            })
            st.success(f"Added {m_label}. Queue size: {len(st.session_state.manual_queue)}")
        else:
            st.warning("Complete all fields before adding.")

    if st.session_state.manual_queue:
        st.divider()
        st.write("**Current Queue List**")
        st.dataframe(pd.DataFrame(st.session_state.manual_queue)[["label", "target"]], use_container_width=True)

with tab_auto:
    auto_target = st.text_input("Global Target for URLs:", placeholder="e.g. Plumbers London")
    urls_input = st.text_area("Enter URLs (one per line):", height=150)

# --- PROCESSING ---
if st.button("🚀 Run Analysis on All Items", type="primary"):
    final_inputs = []
    if urls_input and auto_target:
        for url in urls_input.strip().split('\n'):
            if url.strip(): final_inputs.append({"type": "url", "value": url.strip(), "label": url.strip(), "target": auto_target})
    final_inputs.extend(st.session_state.manual_queue)

    if not final_inputs:
        st.warning("No pages to analyze.")
    else:
        results = []
        progress = st.progress(0)
        status_text = st.empty()
        
        raw_excludes = exclude_text.replace('\n', ',').split(',')
        excludes = [x.strip() for x in raw_excludes if x.strip()]
        
        for i, item in enumerate(final_inputs):
            status_text.text(f"Processing {i+1}/{len(final_inputs)}: {item['label']}")
            
            if item["type"] == "url":
                text = scrape_with_selenium(item["value"], excludes)
            else:
                text = item["content"]
                for p in excludes: 
                    if p: text = text.replace(p, "")
            
            if "ERROR" in text or len(text) < 50:
                results.append({"Source": item['label'], "Target": item['target'], "Main Entity": "Error", "Alignment": "N/A", "Advice": "Check source text."})
                continue

            main_ent, sub_ents = analyze_entities(text)
            if main_ent:
                advice = get_gemini_optimization_advice(item['target'], main_ent, sub_ents, text)
                if advice:
                    results.append({
                        "Source": item['label'],
                        "Target Focus": item['target'],
                        "Main Entity (Score)": f"{main_ent['name']} ({main_ent['score']:.2f})",
                        "Target Alignment": clean_output_text(advice.get("alignment_status")),
                        "Optimization Advice": clean_output_text(advice.get("optimization_advice")),
                        "Actionable Examples": clean_output_text(advice.get("actionable_examples"))
                    })
                else:
                    results.append({"Source": item['label'], "Target Focus": item['target'], "Main Entity (Score)": f"{main_ent['name']} ({main_ent['score']:.2f})", "Target Alignment": "API Error", "Optimization Advice": "Gemini failed to respond.", "Actionable Examples": "N/A"})
            else:
                results.append({"Source": item['label'], "Target Focus": item['target'], "Main Entity (Score)": "None", "Target Alignment": "N/A", "Optimization Advice": "No entities detected.", "Actionable Examples": "N/A"})
            
            progress.progress((i + 1) / len(final_inputs))
            
        st.session_state.results_df = pd.DataFrame(results)
        status_text.success(f"Successfully processed {len(final_inputs)} pages.")

# --- DISPLAY ---
if st.session_state.results_df is not None:
    st.divider()
    csv = st.session_state.results_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Clean CSV Report", data=csv, file_name="entity_analysis_clean.csv", mime="text/csv", use_container_width=True)
    
    # Render with st.dataframe for the cleanest possible tabular view
    st.dataframe(st.session_state.results_df, use_container_width=True, hide_index=True)
    
    # Render detailed view with plain text
    for _, row in st.session_state.results_df.iterrows():
        with st.expander(f"📋 Details: {row['Source']}"):
            st.text(f"Target Focus: {row['Target Focus']}")
            st.text(f"Main Entity: {row['Main Entity (Score)']}")
            st.write("---")
            st.text(f"Alignment: {row['Target Alignment']}")
            st.write(f"Advice: {row['Optimization Advice']}")
            st.write(f"Examples: {row['Actionable Examples']}")
