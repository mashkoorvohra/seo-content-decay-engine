import streamlit as st
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from google import genai
import json

st.set_page_config(page_title="Semantic Keyword Clusterer", layout="wide")
st.title("🧠 Semantic Keyword Clustering & Intent Classifier")
st.markdown("Groups raw keywords by true semantic meaning using Gemini Embeddings and Hierarchical Clustering.")

# Sidebar Controls
api_key = st.sidebar.text_input("Enter Free Gemini API Key", type="password")
st.sidebar.markdown("[Get a free key from Google AI Studio](https://aistudio.google.com/)")

demo_mode = st.sidebar.checkbox("Enable Instant Demo Mode (0 API Credits)", value=True)
distance_threshold = st.sidebar.slider("Cluster Tightness (Lower = More Clusters)", 0.3, 1.2, 0.75)

# Sample Data
default_keywords = """best technical seo tools
technical seo audit checklist
how to fix crawl errors
what is core web vitals
improve largest contentful paint
site speed optimization guide
enterprise seo agency pricing
hire technical seo consultant
seo audit services cost"""

keywords_input = st.text_area("Paste Keywords (One per line)", value=default_keywords, height=200)

if st.button("Cluster Keywords & Classify Intent"):
    keywords = [k.strip() for k in keywords_input.split("\n") if k.strip()]

    if demo_mode:
        st.info("Running in Instant Demo Mode (Zero API calls used). Uncheck 'Enable Instant Demo Mode' in the sidebar to run live with your API key.")
        demo_data = [
            {"Keyword": "best technical seo tools", "Cluster Topic": "Technical SEO Audits & Tools", "Search Intent": "Commercial"},
            {"Keyword": "technical seo audit checklist", "Cluster Topic": "Technical SEO Audits & Tools", "Search Intent": "Informational"},
            {"Keyword": "how to fix crawl errors", "Cluster Topic": "Technical SEO Audits & Tools", "Search Intent": "Informational"},
            {"Keyword": "what is core web vitals", "Cluster Topic": "Core Web Vitals & Speed", "Search Intent": "Informational"},
            {"Keyword": "improve largest contentful paint", "Cluster Topic": "Core Web Vitals & Speed", "Search Intent": "Informational"},
            {"Keyword": "site speed optimization guide", "Cluster Topic": "Core Web Vitals & Speed", "Search Intent": "Informational"},
            {"Keyword": "enterprise seo agency pricing", "Cluster Topic": "SEO Services & Hiring", "Search Intent": "Commercial"},
            {"Keyword": "hire technical seo consultant", "Cluster Topic": "SEO Services & Hiring", "Search Intent": "Transactional"},
            {"Keyword": "seo audit services cost", "Cluster Topic": "SEO Services & Hiring", "Search Intent": "Commercial"},
        ]
        st.dataframe(pd.DataFrame(demo_data), use_container_width=True)

    elif not api_key:
        st.warning("Please enter your free Gemini API key in the sidebar, or turn on Instant Demo Mode.")
    else:
        with st.spinner("Generating embeddings and clustering locally..."):
            client = genai.Client(api_key=api_key)

            # 1. Batch embed all keywords in ONE single API call
            embed_res = client.models.embed_content(
                model="text-embedding-004",
                contents=keywords
            )
            vectors = [e.values for e in embed_res.embeddings]

            # 2. Group locally using Scikit-Learn (Free CPU compute)
            clustering = AgglomerativeClustering(
                n_clusters=None, 
                distance_threshold=distance_threshold, 
                metric='cosine', 
                linkage='average'
            )
            labels = clustering.fit_predict(vectors)

            df = pd.DataFrame({"Keyword": keywords, "Cluster_ID": labels})

            # 3. Name all clusters in ONE single lightweight LLM call
            grouped = df.groupby("Cluster_ID")["Keyword"].apply(list).to_dict()
            prompt = f"""
            Analyze these grouped SEO keyword clusters. For each Cluster_ID, return a JSON object mapping the ID (as a string) to an object with 'topic' (a short Pillar Page name) and 'intent' (Informational, Commercial, or Transactional).
            Clusters: {json.dumps(grouped)}
            Return ONLY valid JSON.
            """

            llm_res = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=prompt
            )

            clean_json = llm_res.text.replace("```json", "").replace("```", "").strip()
            cluster_meta = json.loads(clean_json)

            df["Cluster Topic"] = df["Cluster_ID"].astype(str).map(lambda x: cluster_meta.get(x, {}).get("topic", "General"))
            df["Search Intent"] = df["Cluster_ID"].astype(str).map(lambda x: cluster_meta.get(x, {}).get("intent", "Informational"))

            final_df = df[["Keyword", "Cluster Topic", "Search Intent"]].sort_values("Cluster Topic")
            st.success(f"Grouped {len(keywords)} keywords into {df['Cluster_ID'].nunique()} semantic clusters using just 2 API calls!")
            st.dataframe(final_df, use_container_width=True)

            csv = final_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Clustered CSV", data=csv, file_name="clustered_keywords.csv", mime="text/csv")
