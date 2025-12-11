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
ACTIVE_GEMINI_MODEL = "gemini-1.5-flash"

# --- 2. AUTHENTICATION ---
# (Same authentication block as before - condensed for brevity)
# ... [Assuming the exact same Auth code as the previous file] ...

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

# --- 3. BROWSER SETUP ---
@st.cache_resource
def get_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    chrome_options.binary_location = "/usr/bin/chromium"
    service = Service("/usr/bin/chromedriver")
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

if 'results_df' not in st.session_state:
    st.session_state.results_df = None

# --- 4. CORE FUNCTIONS (Scrape, Dedup, Analyze, Explain) ---
# [Same functions as before]

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

def deduplicate_entities(entities):
    merged = {}
    for e in entities:
        clean_name = e.name.lower().rstrip('s')
        if clean_name in merged:
            merged[clean_name]['score'] += e.salience
            if e.name[0].isupper() and not merged[clean_name]['name'][0].isupper():
                merged[clean_name]['name'] = e.name
        else:
            merged[clean_name] = {'name': e.name, 'score': e.salience}
    return sorted(merged.values(), key=lambda x: x['score'], reverse=True)

def analyze_entities(text):
    try:
        nlp_text = text[:100000]
        document = language_v1.Document(content=nlp_text, type_=language_v1.Document.Type.PLAIN_TEXT)
        response = nlp_client.analyze_entities(request={'document': document})
        cleaned_entities = deduplicate_entities(response.entities)
        if not cleaned_entities: return None, []
        main = {"name": cleaned_entities[0]['name'], "score": cleaned_entities[0]['score']}
        subs = [{"name": e['name'], "score": e['score']} for e in cleaned_entities[1:10]]
        return main, subs
    except Exception as e:
        st.error(f"NLP API Error: {e}")
        return None, []

def get_gemini_explanation(url, main_entity, text_sample):
    prompt = f"""
    You are an expert in Google's Natural Language Processing (NLP) API.
    Data: URL: {url}, Main Entity: "{main_entity['name']}" (Score: {main_entity['score']:.2f}).
    Text Start: "{text_sample}..."
    Task: Explain WHY NLP chose "{main_entity['name']}".
    Analyze: Grammatical Subject, Structure, Repetition.
    Output JSON ONLY: {{ "context": "...", "explanation": "..." }}
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

# *** NEW: EDUCATIONAL CONTEXT BLOCK ***
with st.expander("ℹ️ Guide: How to understand these metrics", expanded=False):
    st.markdown("""
    ### 1. Main Entity & Salience Score
    The **Main Entity** is the single noun that Google believes is the "Subject" of your page. 
    The **Salience Score (0.0 - 1.0)** represents Google's confidence.
    
    * **High Score (> 0.20):** Google is very sure this page is about this topic.
    * **Low Score (< 0.05):** Google is confused. The topic is likely **fragmented** (e.g. "Car Cover" vs "Vehicle Cover") or **diluted** by other text.
    
    ### 2. Why is my score low?
    * **Adjective Fragmentation:** If you say *"Single Trip Cover"* and *"Annual Cover"*, Google sees two different things, not one big "Cover" entity.
    * **Grammar:** If you write *"You can buy cover"*, the subject is **You**. If you write *"Cover protects you"*, the subject is **Cover**.
    
    ### 3. Sub Entities
    These are the supporting concepts. For a page to rank well, Sub Entities should be **contextually relevant** (e.g. *breakdown, vehicle, policy*) rather than generic (e.g. *member, login, home*).
    """)

# SIDEBAR
with st.sidebar:
    st.header("⚙️ Settings")
    st.info(f"Using Model: **{ACTIVE_GEMINI_MODEL}**")
    st.divider()
    st.subheader("🧹 Text to Remove")
    st.caption("Paste specific navigation text or headers here (comma separated).")
    exclude_text = st.text_area("Phrases to delete:", height=150, placeholder="Breakdown Cover | RAC, Skip to content")

urls_input = st.text_area("Enter URLs (one per line):", height=100)

# PROCESSING
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
            
            # Scrape
            text = scrape_with_selenium(url, exclude_phrases=excludes)
            with st.expander(f"📄 Text Source: {url}"):
                st.text_area("Content Preview", text, height=200)

            if "ERROR" in text or len(text) < 50:
                results.append({"URL": url, "Main Entity": "Error", "Explanation": "Scrape Failed"})
                continue

            # NLP
            main_ent, sub_ents = analyze_entities(text)
            if not main_ent:
                results.append({"URL": url, "Main Entity": "None", "Explanation": "No entities found"})
                continue
                
            # Gemini
            explanation_data = get_gemini_explanation(url, main_ent, text[:3000])
            formatted_subs = ", ".join([f"{s['name']} ({s['score']:.2f})" for s in sub_ents])
            
            # Sanitize Output
            raw_expl = explanation_data.get("explanation", "")
            clean_expl = "\n".join([str(item) for item in raw_expl]) if isinstance(raw_expl, list) else str(raw_expl)
            
            results.append({
                "URL": url,
                "Main Entity": f"{main_ent['name']} ({main_ent['score']:.2f})",
                "Sub Entities": formatted_subs,
                "Context": str(explanation_data.get("context", "")),
                "Why it was picked": clean_expl
            })
            progress.progress((i + 1) / len(urls))
            
        status.success("Done!")
        st.session_state.results_df = pd.DataFrame(results)

# DISPLAY
if st.session_state.results_df is not None:
    st.divider()
    col1, col2 = st.columns([4, 1])
    with col2:
        csv = st.session_state.results_df.to_csv(index=False).encode('utf-8')
        st.download_button(label="📥 Download CSV", data=csv, file_name="entity_analysis.csv", mime="text/csv", use_container_width=True)
    st.dataframe(st.session_state.results_df, use_container_width=True)
