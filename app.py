import streamlit as st
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from google import genai
import json
import re

st.set_page_config(page_title="Semantic Keyword Clusterer", layout="wide")
st.title("🧠 Semantic Keyword Clustering & Intent Classifier")
st.markdown("Groups raw keywords into topical Pillar Page clusters and classifies search intent—runs **100% free locally** or with **Gemini AI Embeddings**.")

# Sidebar Controls
mode = st.sidebar.radio(
    "Clustering Engine",
    ["Free Local Engine (No API Key Needed)", "Gemini AI Embeddings (Deep Semantic)"],
    index=0
)

api_key = ""
if mode == "Gemini AI Embeddings (Deep Semantic)":
    api_key = st.sidebar.text_input("Enter Free Gemini API Key", type="password")
    st.sidebar.markdown("[Get a free key from Google AI Studio](https://aistudio.google.com/apikey)")

default_threshold = 0.72 if mode == "Free Local Engine (No API Key Needed)" else 0.65
distance_threshold = st.sidebar.slider(
    "Cluster Tightness (Lower = More Specific Clusters)", 
    min_value=0.30, 
    max_value=0.95, 
    value=default_threshold, 
    step=0.05
)

# Helper: Rule-based intent classifier for zero-API mode
def classify_intent_local(kw: str) -> str:
    kw_lower = kw.lower()
    transactional_terms = ["hire", "buy", "book", "quote", "consultant", "agency", "services", "for sale", "builders", "contractors", "company", "companies"]
    commercial_terms = ["best", "top", "cost", "price", "pricing", "vs", "review", "reviews", "comparison", "rates", "per sq ft"]
    
    if any(re.search(rf"\b{re.escape(t)}\b", kw_lower) for t in transactional_terms):
        return "Transactional"
    if any(re.search(rf"\b{re.escape(t)}\b", kw_lower) for t in commercial_terms):
        return "Commercial"
    return "Informational"

# Helper: Auto-name local clusters cleanly
def name_cluster_local(kw_list: list) -> str:
    stop_words = {"in", "for", "of", "the", "a", "an", "to", "how", "what", "is", "best", "top", "cost", "price", "guide"}
    words = []
    for kw in kw_list:
        for w in re.findall(r"[a-zA-Z0-9]+", kw.lower()):
            if w not in stop_words and len(w) > 2:
                words.append(w)
    if not words:
        return kw_list[0].title()
    top_terms = pd.Series(words).value_counts().head(3).index.tolist()
    return " ".join(top_terms).title() + " Hub"

# Sample Data (Preloaded with Real Estate & Technical SEO mix)
default_keywords = """building construction in bangalore
residential construction companies in bangalore
house construction cost per sq ft in bangalore
turnkey home construction bangalore
villa elevation design ideas
modern duplex house elevation
luxury residential architecture bangalore
technical seo audit checklist
how to fix crawl errors in search console
core web vitals optimization guide"""

keywords_input = st.text_area("Paste Keywords (One per line — minimum 2 keywords)", value=default_keywords, height=220)

if st.button("Cluster Keywords & Classify Intent", type="primary"):
    # Deduplicate while preserving order
    raw_lines = [k.strip() for k in keywords_input.split("\n") if k.strip()]
    keywords = list(dict.fromkeys(raw_lines))
    
    if len(keywords) < 2:
        st.warning("Please enter at least 2 unique keywords (one per line) to compare and cluster.")
        st.stop()

    if mode == "Free Local Engine (No API Key Needed)":
        with st.spinner("Vectorizing n-grams and clustering locally..."):
            # Word + character n-gram TF-IDF handles typos & partial matches locally
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), analyzer="char_wb", min_df=1)
            vectors = vectorizer.fit_transform(keywords).toarray()
            
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_threshold,
                metric="cosine",
                linkage="average"
            )
            labels = clustering.fit_predict(vectors)
            
            df = pd.DataFrame({"Keyword": keywords, "Cluster_ID": labels})
            cluster_names = df.groupby("Cluster_ID")["Keyword"].apply(list).apply(name_cluster_local).to_dict()
            
            df["Cluster Topic"] = df["Cluster_ID"].map(cluster_names)
            df["Search Intent"] = df["Keyword"].apply(classify_intent_local)
            
            final_df = df[["Keyword", "Cluster Topic", "Search Intent"]].sort_values(["Cluster Topic", "Search Intent"])
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Keywords", len(final_df))
            col2.metric("Topic Clusters Found", final_df["Cluster Topic"].nunique())
            col3.metric("API Credits Used", "0 (100% Local)")
            
            st.dataframe(final_df, use_container_width=True, hide_index=True)
            
            csv = final_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Clustered CSV", data=csv, file_name="clustered_keywords_local.csv", mime="text/csv")

    else:
        if not api_key:
            st.warning("Please paste your free Gemini API key in the sidebar, or switch to 'Free Local Engine' above.")
            st.stop()
            
        try:
            with st.spinner("Generating Gemini neural embeddings and naming Pillar Pages..."):
                client = genai.Client(api_key=api_key)
                
                embed_res = client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=keywords
                )
                vectors = [e.values for e in embed_res.embeddings]
                
                clustering = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=distance_threshold,
                    metric="cosine",
                    linkage="average"
                )
                labels = clustering.fit_predict(vectors)
                
                df = pd.DataFrame({"Keyword": keywords, "Cluster_ID": labels})
                grouped = df.groupby("Cluster_ID")["Keyword"].apply(list).to_dict()
                
                prompt = f"""
                Analyze these grouped SEO keyword clusters. For each Cluster_ID, return a JSON object mapping the ID (as a string) to an object with 'topic' (a concise Pillar Page title) and 'intent' (Informational, Commercial, or Transactional).
                Clusters: {json.dumps(grouped)}
                Return ONLY valid JSON.
                """
                
                llm_res = client.models.generate_content(
                    model="gemini-2.5-flash-lite",
                    contents=prompt
                )
                
                clean_json = llm_res.text.replace("```json", "").replace("```", "").strip()
                cluster_meta = json.loads(clean_json)
                
                df["Cluster Topic"] = df["Cluster_ID"].astype(str).map(lambda x: cluster_meta.get(x, {}).get("topic", "General Hub"))
                df["Search Intent"] = df["Cluster_ID"].astype(str).map(lambda x: cluster_meta.get(x, {}).get("intent", "Informational"))
                
                final_df = df[["Keyword", "Cluster Topic", "Search Intent"]].sort_values(["Cluster Topic", "Search Intent"])
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Keywords", len(final_df))
                col2.metric("AI Topic Clusters", final_df["Cluster Topic"].nunique())
                col3.metric("API Calls Used", "2 Batch Calls")
                
                st.dataframe(final_df, use_container_width=True, hide_index=True)
                
                csv = final_df.to_csv(index=False).encode("utf-8")
                st.download_button("Download Clustered CSV", data=csv, file_name="clustered_keywords_ai.csv", mime="text/csv")
                
        except Exception as e:
            st.error(f"API Error: {e}. Check that your Gemini API key is valid and active.")
