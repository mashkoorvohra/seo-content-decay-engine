provider, api_key, selected_model = render_ai_sidebar()
import streamlit as st
import pandas as pd
import litellm
import requests

st.set_page_config(page_title="SEO Content Decay Engine", layout="wide")
st.title("📉 SEO Content Decay Detector & AI Refresh Planner")
st.markdown("Identifies high-value pages losing organic clicks and generates **step-by-step content refresh briefs** using the latest AI models.")

@st.cache_data(ttl=3600)
def fetch_live_models(provider_name: str, key: str) -> list:
    defaults = {
        "Google Gemini": ["gemini-4-hydrogen", "gemini-4-carbon", "gemini-4-argon", "gemini-3.5-flash-lite"],
        "Anthropic Claude": ["claude-sonnet-5-5", "claude-haiku-5-5", "claude-opus-5-5", "claude-fable-5-2"],
        "OpenAI": ["gpt-6.2-astra", "gpt-6.2-sol", "gpt-6-luna", "gpt-5.4-mini"]
    }
    if not key:
        return defaults.get(provider_name, [])
    try:
        if provider_name == "Google Gemini":
            r = requests.get(f"https://generativelanguage.googleapis.com/v1beta/models?key={key}", timeout=6)
            models = [m["name"].replace("models/", "") for m in r.json().get("models", []) if "generateContent" in m.get("supportedGenerationMethods", [])]
            newest = [m for m in models if any(v in m for v in ["gemini-4", "gemini-3.8", "gemini-3.5", "gemini-3.1"])]
            return newest[::-1] if newest else defaults["Google Gemini"]
        elif provider_name == "Anthropic Claude":
            r = requests.get("https://api.anthropic.com/v1/models", headers={"x-api-key": key, "anthropic-version": "2023-06-01"}, timeout=6)
            models = [m["id"] for m in r.json().get("data", [])]
            return models if models else defaults["Anthropic Claude"]
        elif provider_name == "OpenAI":
            r = requests.get("https://api.openai.com/v1/models", headers={"Authorization": f"Bearer {key}"}, timeout=6)
            models = [m["id"] for m in r.json().get("data", []) if any(v in m["id"] for v in ["gpt-6", "gpt-5", "o3"])]
            return sorted(models, reverse=True) if models else defaults["OpenAI"]
    except Exception:
        pass
    return defaults.get(provider_name, [])

# Sidebar Settings
st.sidebar.header("⚙️ AI Refresh Strategist")
provider = st.sidebar.selectbox("AI Provider", ["Free Local Diagnosis Only", "Google Gemini", "Anthropic Claude", "OpenAI"])
api_key = ""
selected_model = ""

if provider != "Free Local Diagnosis Only":
    api_key = st.sidebar.text_input(f"Enter {provider} API Key", type="password")
    models_list = fetch_live_models(provider, api_key)
    selected_model = st.sidebar.selectbox("Select Latest Model", models_list)
    prefix_map = {"Google Gemini": "gemini/", "Anthropic Claude": "anthropic/", "OpenAI": "openai/"}
    if not selected_model.startswith(prefix_map[provider]):
        selected_model = prefix_map[provider] + selected_model

# Sample GSC Data or CSV Upload
uploaded_file = st.file_uploader("Upload GSC Pages Comparison CSV (Columns: Page, Previous Clicks, Current Clicks, Top Query)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.info("Showing sample Google Search Console 90-day comparison dataset. Upload your own GSC export CSV above anytime.")
    df = pd.DataFrame([
        {"Page": "/blog/home-construction-cost-bangalore/", "Top Query": "house construction cost per sq ft in bangalore", "Previous Clicks": 1420, "Current Clicks": 690},
        {"Page": "/blog/best-plotted-developments-sarjapur/", "Top Query": "plotted developments on sarjapur road", "Previous Clicks": 980, "Current Clicks": 510},
        {"Page": "/blog/modern-duplex-elevation-designs/", "Top Query": "modern duplex house elevation designs", "Previous Clicks": 840, "Current Clicks": 790},
        {"Page": "/services/turnkey-home-builders/", "Top Query": "turnkey residential builders bangalore", "Previous Clicks": 610, "Current Clicks": 240},
        {"Page": "/blog/vastu-tips-for-new-home/", "Top Query": "vastu shastra for east facing house", "Previous Clicks": 1150, "Current Clicks": 1280},
    ])

df["Click Change"] = df["Current Clicks"] - df["Previous Clicks"]
df["Decay %"] = ((df["Click Change"] / df["Previous Clicks"]) * 100).round(1)
df["Priority"] = df["Decay %"].apply(lambda x: "🔴 Critical Decay" if x <= -35 else ("🟠 Moderate Decay" if x <= -15 else "🟢 Healthy / Growing"))

decay_only = df[df["Decay %"] < -10].sort_values("Click Change")

c1, c2, c3 = st.columns(3)
c1.metric("Pages Audited", len(df))
c2.metric("Decaying Pages Flagged", len(decay_only))
c3.metric("Total Lost Clicks", int(decay_only["Click Change"].sum()))

st.dataframe(df.sort_values("Decay %"), use_container_width=True, hide_index=True)

if not decay_only.empty:
    st.subheader("🛠️ Generate Content Refresh Action Plan")
    target_page = st.selectbox("Pick a Decaying URL to Diagnose", decay_only["Page"].tolist())
    row = decay_only[decay_only["Page"] == target_page].iloc[0]

    if st.button("Generate Refresh Brief", type="primary"):
        if provider == "Free Local Diagnosis Only":
            st.success(f"**Local SEO Playbook for `{row['Page']}`** (Lost {abs(int(row['Click Change']))} clicks / {row['Decay %']}%)")
            st.markdown(f"""
            1. **Title & Freshness Overhaul:** Update the H1, Meta Title, and opening summary with current 2026 benchmarks for *"{row['Top Query']}"*.
            2. **AI Overview Direct-Answer Block:** Add a 45-word bold summary table right below the introduction answering *"{row['Top Query']}"* directly so LLMs cite your page.
            3. **Internal Link Injection:** Add 3–5 contextual internal links from your highest-traffic pages pointing to `{row['Page']}` using partial-match anchor text.
            4. **Schema Refresh:** Update `dateModified` in your `Article` / `FAQPage` JSON-LD schema and request re-indexing in Google Search Console.
            """)
        elif not api_key:
            st.warning(f"Please paste your {provider} API key in the sidebar, or select 'Free Local Diagnosis Only'.")
        else:
            with st.spinner(f"Building custom refresh brief with {selected_model}..."):
                prompt = f"""
                You are a Senior Technical SEO Strategist. This URL is experiencing severe organic traffic decay:
                - URL: {row['Page']}
                - Primary Keyword: {row['Top Query']}
                - Traffic Drop: From {row['Previous Clicks']} to {row['Current Clicks']} clicks ({row['Decay %']}%)

                Provide a concise, high-impact Content Refresh Brief containing:
                1. New click-worthy Title Tag & H1 (2026 updated).
                2. A 50-word 'AI Overview Citation Block' to win featured snippets and LLM citations.
                3. 3 missing semantic subtopics (H2s) competitors are likely covering.
                4. Internal linking anchor text suggestions.
                """
                resp = litellm.completion(model=selected_model, messages=[{"role": "user", "content": prompt}], api_key=api_key)
                st.markdown(resp.choices[0].message.content)
