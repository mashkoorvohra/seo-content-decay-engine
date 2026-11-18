import streamlit as st
import pandas as pd
import litellm
import requests
import io

st.set_page_config(page_title="SEO Content Decay Engine", layout="wide")
st.title("📉 Universal SEO Content Decay Detector & Refresh Planner")
st.markdown("Detects organic traffic decay across **any CSV, Excel (.xlsx), or pasted dataset**—runs **100% free locally** or with **any AI provider**.")

# Live Model Auto-Fetcher (Optional AI Mode)
@st.cache_data(ttl=3600)
def fetch_live_models(provider_name: str, key: str) -> list:
    defaults = {
        "Google Gemini": ["gemini-4-hydrogen", "gemini-4-carbon", "gemini-4-argon", "gemini-3.5-flash-lite"],
        "Anthropic Claude": ["claude-sonnet-5-5", "claude-haiku-5-5", "claude-opus-5-5"],
        "OpenAI": ["gpt-6.2-astra", "gpt-6.2-sol", "gpt-6-luna", "gpt-5.4-mini"]
    }
    if not key:
        return defaults.get(provider_name, [])
    try:
        if provider_name == "Google Gemini":
            r = requests.get(f"https://generativelanguage.googleapis.com/v1beta/models?key={key}", timeout=6)
            models = [m["name"].replace("models/", "") for m in r.json().get("models", []) if "generateContent" in m.get("supportedGenerationMethods", [])]
            newest = [m for m in models if any(v in m for v in ["gemini-4", "gemini-3.8", "gemini-3.5"])]
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

# Sidebar: Optional AI Strategist
st.sidebar.header("⚙️ Execution Mode")
provider = st.sidebar.selectbox("AI Strategist (Optional)", ["Free Local Playbook (0 API Credits)", "Google Gemini", "Anthropic Claude", "OpenAI"])
api_key = ""
selected_model = ""

if provider != "Free Local Playbook (0 API Credits)":
    api_key = st.sidebar.text_input(f"Enter {provider} API Key", type="password")
    models_list = fetch_live_models(provider, api_key)
    selected_model = st.sidebar.selectbox("Select Model", models_list)
    prefix_map = {"Google Gemini": "gemini/", "Anthropic Claude": "anthropic/", "OpenAI": "openai/"}
    if not selected_model.startswith(prefix_map[provider]):
        selected_model = prefix_map[provider] + selected_model

decay_threshold = st.sidebar.slider("Flag Decay Worse Than (%)", min_value=-80, max_value=-5, value=-15, step=5)

# 1. Universal Data Ingestion (Sample, Upload CSV/Excel, or Paste)
data_source = st.radio(
    "Choose Data Input Method:",
    ["Neutral Demo Dataset (Instant Preview)", "Upload CSV or Excel File (.csv / .xlsx)", "Paste Raw Data (Tab or Comma Separated)"],
    horizontal=True
)

raw_df = None

if data_source == "Neutral Demo Dataset (Instant Preview)":
    raw_df = pd.DataFrame([
        {"URL": "https://example.com/blog/cloud-security-checklist", "Primary Keyword": "cloud security checklist", "Previous Clicks": 3400, "Current Clicks": 1650},
        {"URL": "https://example.com/guides/enterprise-crm-pricing", "Primary Keyword": "enterprise crm pricing guide", "Previous Clicks": 2150, "Current Clicks": 1120},
        {"URL": "https://example.com/blog/remote-team-onboarding", "Primary Keyword": "remote employee onboarding steps", "Previous Clicks": 1890, "Current Clicks": 1740},
        {"URL": "https://example.com/products/ergonomic-standing-desk", "Primary Keyword": "best ergonomic standing desk", "Previous Clicks": 4200, "Current Clicks": 1980},
        {"URL": "https://example.com/blog/b2b-email-deliverability", "Primary Keyword": "improve b2b email deliverability", "Previous Clicks": 1520, "Current Clicks": 1810},
    ])
elif data_source == "Upload CSV or Excel File (.csv / .xlsx)":
    uploaded = st.file_uploader("Upload any GSC, Ahrefs, Semrush, or custom spreadsheet", type=["csv", "xlsx"])
    if uploaded:
        raw_df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
else:
    pasted = st.text_area(
        "Paste rows directly from Excel, Google Sheets, or CSV (include header row):",
        value="Page,Keyword,Prev_Clicks,Curr_Clicks\nhttps://site.com/page-a,seo audit tool,1200,540\nhttps://site.com/page-b,link building guide,900,850",
        height=150
    )
    if pasted.strip():
        sep = "\t" if "\t" in pasted.splitlines()[0] else ","
        raw_df = pd.read_csv(io.StringIO(pasted), sep=sep)

# 2. Smart Column Auto-Detection & Flexible Mapping
if raw_df is not None and not raw_df.empty:
    cols = list(raw_df.columns)

    def guess_col(keywords_list, fallback_idx=0):
        for c in cols:
            if any(k in str(c).lower() for k in keywords_list):
                return c
        return cols[min(fallback_idx, len(cols) - 1)]

    with st.expander("🔧 Column Mapping (Auto-Detected — Adjust Only If Needed)", expanded=(data_source != "Neutral Demo Dataset (Instant Preview)")):
        m1, m2, m3, m4 = st.columns(4)
        url_col = m1.selectbox("Page / URL Column", cols, index=cols.index(guess_col(["page", "url", "landing", "slug"], 0)))
        prev_col = m2.selectbox("Previous Clicks / Traffic", cols, index=cols.index(guess_col(["prev", "previous", "before", "last", "old", "clicks 1"], 1)))
        curr_col = m3.selectbox("Current Clicks / Traffic", cols, index=cols.index(guess_col(["curr", "current", "after", "new", "recent", "clicks 2"], min(2, len(cols)-1))))
        kw_options = ["(None — Extract from URL)"] + cols
        detected_kw = guess_col(["query", "keyword", "term"], -1)
        kw_default_idx = kw_options.index(detected_kw) if any(k in str(detected_kw).lower() for k in ["query", "keyword", "term"]) else 0
        kw_col = m4.selectbox("Target Keyword (Optional)", kw_options, index=kw_default_idx)

    # Clean numeric values safely (handles commas like "1,420" or strings)
    df = pd.DataFrame()
    df["Page"] = raw_df[url_col].astype(str)
    df["Previous Clicks"] = pd.to_numeric(raw_df[prev_col].astype(str).str.replace(",", ""), errors="coerce").fillna(0)
    df["Current Clicks"] = pd.to_numeric(raw_df[curr_col].astype(str).str.replace(",", ""), errors="coerce").fillna(0)

    if kw_col == "(None — Extract from URL)":
        df["Top Query"] = df["Page"].str.rstrip("/").str.split("/").str[-1].str.replace("-", " ").str.replace("_", " ")
    else:
        df["Top Query"] = raw_df[kw_col].astype(str)

    # Filter out zero-baseline rows to prevent division-by-zero
    df = df[df["Previous Clicks"] > 0].copy()
    df["Click Change"] = (df["Current Clicks"] - df["Previous Clicks"]).astype(int)
    df["Decay %"] = ((df["Click Change"] / df["Previous Clicks"]) * 100).round(1)
    df["Status"] = df["Decay %"].apply(lambda x: "🔴 Critical Decay" if x <= -35 else ("🟠 Moderate Decay" if x <= decay_threshold else "🟢 Stable / Growing"))

    flagged = df[df["Decay %"] <= decay_threshold].sort_values("Click Change")

    # KPI Summary
    k1, k2, k3 = st.columns(3)
    k1.metric("Valid URLs Analyzed", len(df))
    k2.metric("Decaying URLs Flagged", len(flagged))
    k3.metric("Net Lost Clicks", int(flagged["Click Change"].sum()) if not flagged.empty else 0)

    st.dataframe(df.sort_values("Decay %"), use_container_width=True, hide_index=True)

    st.download_button(
        "Download Full Decay Audit CSV",
        data=df.sort_values("Decay %").to_csv(index=False).encode("utf-8"),
        file_name="content_decay_audit.csv",
        mime="text/csv"
    )

    # 3. Actionable Refresh Plan
    if not flagged.empty:
        st.subheader("🛠️ Content Refresh Action Plan")
        chosen_url = st.selectbox("Select Flagged URL to Refresh:", flagged["Page"].tolist())
        row = flagged[flagged["Page"] == chosen_url].iloc[0]

        if st.button("Generate Refresh Playbook", type="primary"):
            if provider == "Free Local Playbook (0 API Credits)":
                st.success(f"**Free Local SEO Refresh Checklist for `{row['Page']}`** ({row['Decay %']}% traffic drop)")
                st.markdown(f"""
                * **Query Alignment (`{row['Top Query']}`):** Update the Title Tag, H1, and first 100 words to reflect current-year data and intent.
                * **AI Overview Citation Block:** Insert a 45-word direct summary table immediately under the H1 answering *"{row['Top Query']}"* for LLM snippet capture.
                * **Internal Link Equity Boost:** Add 3–5 contextual internal links from high-authority sibling pages pointing to `{row['Page']}`.
                * **Technical Freshness Signal:** Update `dateModified` in your JSON-LD schema and submit the URL via Google Search Console URL Inspection.
                """)
            elif not api_key:
                st.warning(f"Paste your {provider} API key in the sidebar, or keep 'Free Local Playbook' selected.")
            else:
                with st.spinner(f"Drafting custom refresh brief via {selected_model}..."):
                    prompt = f"Act as a Principal SEO Strategist. URL `{row['Page']}` (target query: `{row['Top Query']}`) dropped from {int(row['Previous Clicks'])} to {int(row['Current Clicks'])} clicks ({row['Decay %']}%). Give: 1) Updated Title & H1, 2) A 50-word AI Overview direct-answer block, 3) 3 missing semantic H2 subtopics, 4) Internal linking anchor suggestions."
                    resp = litellm.completion(model=selected_model, messages=[{"role": "user", "content": prompt}], api_key=api_key)
                    st.markdown(resp.choices[0].message.content)
