import streamlit as st
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
import litellm
import json
import re
import os

st.set_page_config(page_title="Universal AI Keyword Clusterer", layout="wide")
st.title("🧠 Provider-Agnostic Semantic Keyword Clusterer")
st.markdown("Cluster keywords into Pillar Pages and classify search intent using **0-API Local Mode**, **Anthropic Claude**, **OpenAI**, **Gemini**, or **Groq/OpenRouter**.")

# Sidebar: Provider & Model Configuration
st.sidebar.header("⚙️ AI Provider Settings")

provider = st.sidebar.selectbox(
    "Select AI Engine",
    [
        "Free Local Engine (No API Key Needed)",
        "Anthropic Claude (claude-sonnet-4-5 / haiku)",
        "Google Gemini (gemini-2.5-flash-lite)",
        "OpenAI (gpt-4o-mini)",
        "Groq / OpenRouter / Custom LiteLLM Model"
    ]
)

api_key = ""
custom_model = ""
embedding_mode = "Local TF-IDF (Free & Universal)"

if provider != "Free Local Engine (No API Key Needed)":
    api_key = st.sidebar.text_input("Enter Provider API Key", type="password")
    
    if provider == "Anthropic Claude (claude-sonnet-4-5 / haiku)":
        custom_model = st.sidebar.selectbox(
            "Claude Model",
            ["anthropic/claude-haiku-4-5-20251001", "anthropic/claude-sonnet-4-5-20250929"]
        )
        st.sidebar.caption("Note: Anthropic has no embedding endpoint, so vectors run locally while Claude names clusters & classifies intent.")
    elif provider == "Google Gemini (gemini-2.5-flash-lite)":
        custom_model = "gemini/gemini-2.5-flash-lite"
        embedding_mode = st.sidebar.radio("Vector Engine", ["Gemini Neural Embeddings", "Local TF-IDF (Free & Universal)"])
    elif provider == "OpenAI (gpt-4o-mini)":
        custom_model = "openai/gpt-4o-mini"
        embedding_mode = st.sidebar.radio("Vector Engine", ["OpenAI Neural Embeddings", "Local TF-IDF (Free & Universal)"])
    else:
        custom_model = st.sidebar.text_input(
            "LiteLLM Model String",
            value="groq/llama-3.3-70b-versatile",
            help="Examples: groq/llama-3.3-70b-versatile, openrouter/anthropic/claude-3.5-sonnet, deepseek/deepseek-chat"
        )

distance_threshold = st.sidebar.slider(
    "Cluster Tightness (Lower = More Specific Clusters)",
    min_value=0.30,
    max_value=0.95,
    value=0.72,
    step=0.05
)

# Local fallback helpers
def classify_intent_local(kw: str) -> str:
    kw_lower = kw.lower()
    transactional = ["hire", "buy", "book", "quote", "consultant", "agency", "services", "builders", "contractors", "company", "companies"]
    commercial = ["best", "top", "cost", "price", "pricing", "vs", "review", "reviews", "comparison", "rates", "per sq ft"]
    if any(re.search(rf"\b{re.escape(t)}\b", kw_lower) for t in transactional):
        return "Transactional"
    if any(re.search(rf"\b{re.escape(t)}\b", kw_lower) for t in commercial):
        return "Commercial"
    return "Informational"

def name_cluster_local(kw_list: list) -> str:
    stop_words = {"in", "for", "of", "the", "a", "an", "to", "how", "what", "is", "best", "top", "cost", "price", "guide"}
    words = [w for kw in kw_list for w in re.findall(r"[a-zA-Z0-9]+", kw.lower()) if w not in stop_words and len(w) > 2]
    if not words:
        return kw_list[0].title()
    top_terms = pd.Series(words).value_counts().head(3).index.tolist()
    return " ".join(top_terms).title() + " Hub"

# Main Input Area
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

keywords_input = st.text_area("Paste Keywords (One per line — minimum 2)", value=default_keywords, height=220)

if st.button("Cluster Keywords & Classify Intent", type="primary"):
    raw_lines = [k.strip() for k in keywords_input.split("\n") if k.strip()]
    keywords = list(dict.fromkeys(raw_lines))

    if len(keywords) < 2:
        st.warning("Please enter at least 2 unique keywords (one per line).")
        st.stop()

    if provider != "Free Local Engine (No API Key Needed)" and not api_key:
        st.warning(f"Please paste your API key for {provider}, or switch to 'Free Local Engine' in the sidebar.")
        st.stop()

    try:
        with st.spinner("Vectorizing keywords and clustering..."):
            # Step 1: Generate Vectors (Neural Embeddings if supported, otherwise Local TF-IDF)
            if embedding_mode == "Gemini Neural Embeddings":
                embed_res = litellm.embedding(
                    model="gemini/gemini-embedding-001",
                    input=keywords,
                    api_key=api_key
                )
                vectors = [item["embedding"] for item in embed_res.data]
                vector_label = "Gemini Neural Embeddings"
            elif embedding_mode == "OpenAI Neural Embeddings":
                embed_res = litellm.embedding(
                    model="openai/text-embedding-3-small",
                    input=keywords,
                    api_key=api_key
                )
                vectors = [item["embedding"] for item in embed_res.data]
                vector_label = "OpenAI Neural Embeddings"
            else:
                vectorizer = TfidfVectorizer(ngram_range=(1, 2), analyzer="char_wb", min_df=1)
                vectors = vectorizer.fit_transform(keywords).toarray()
                vector_label = "Local N-Gram TF-IDF"

            # Step 2: Hierarchical Clustering (100% local scikit-learn compute)
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_threshold,
                metric="cosine",
                linkage="average"
            )
            labels = clustering.fit_predict(vectors)
            df = pd.DataFrame({"Keyword": keywords, "Cluster_ID": labels})

            # Step 3: Name Clusters & Classify Intent (Universal LLM or Local Rules)
            if provider == "Free Local Engine (No API Key Needed)":
                cluster_names = df.groupby("Cluster_ID")["Keyword"].apply(list).apply(name_cluster_local).to_dict()
                df["Cluster Topic"] = df["Cluster_ID"].map(cluster_names)
                df["Search Intent"] = df["Keyword"].apply(classify_intent_local)
                engine_used = "Local Rule Engine (0 API Calls)"
            else:
                grouped = df.groupby("Cluster_ID")["Keyword"].apply(list).to_dict()
                prompt = f"""
                Analyze these grouped SEO keyword clusters. For each Cluster_ID, return a JSON object mapping the ID (as a string) to an object with:
                - 'topic': a concise, SEO-friendly Pillar Page title
                - 'intent': the primary search intent ('Informational', 'Commercial', or 'Transactional')

                Clusters: {json.dumps(grouped)}
                Return ONLY valid JSON with no markdown formatting.
                """

                response = litellm.completion(
                    model=custom_model,
                    messages=[{"role": "user", "content": prompt}],
                    api_key=api_key
                )

                raw_text = response.choices[0].message.content
                clean_json = raw_text.replace("```json", "").replace("```", "").strip()
                cluster_meta = json.loads(clean_json)

                df["Cluster Topic"] = df["Cluster_ID"].astype(str).map(lambda x: cluster_meta.get(x, {}).get("topic", "General Hub"))
                df["Search Intent"] = df["Cluster_ID"].astype(str).map(lambda x: cluster_meta.get(x, {}).get("intent", "Informational"))
                engine_used = custom_model

            # Display Results
            final_df = df[["Keyword", "Cluster Topic", "Search Intent"]].sort_values(["Cluster Topic", "Search Intent"])

            col1, col2, col3 = st.columns(3)
            col1.metric("Keywords Clustered", len(final_df))
            col2.metric("Pillar Clusters", final_df["Cluster Topic"].nunique())
            col3.metric("Active Pipeline", engine_used)

            st.caption(f"Vector Engine: **{vector_label}** | Clustering Algorithm: **Agglomerative Hierarchical (Cosine)**")
            st.dataframe(final_df, use_container_width=True, hide_index=True)

            csv = final_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Clustered CSV", data=csv, file_name="universal_seo_clusters.csv", mime="text/csv")

    except Exception as e:
        st.error(f"Execution Error: {e}")
