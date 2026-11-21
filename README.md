# 📉 Universal SEO Content Decay Detector & Refresh Planner

**Live App:** [https://seo-content-decay-engine.streamlit.app/](https://seo-content-decay-engine.streamlit.app/)
> **Install as an AI Agent Skill (Claude Code / Cursor / MCP):** Get the packaged agent skill for automated GSC click-drop triage and refresh briefs on [Agensi](https://agensi.io/skills/seo-content-decay-engine).

An interactive technical SEO tool that identifies high-value pages losing organic search clicks and generates prioritized content refresh playbooks. Built to run **100% free locally with zero API keys** or connect dynamically to **Google Gemini, Anthropic Claude, or OpenAI** using live-fetched model lists.

## Key Capabilities
- **Universal Data Ingestion:** Analyze neutral demo datasets, upload `.csv` or `.xlsx` exports (Google Search Console, Ahrefs, Semrush), or paste raw rows directly.
- **Smart Column Auto-Mapping:** Automatically detects URL, baseline clicks, current clicks, and target query columns regardless of export header naming.
- **Severity Triage & CSV Export:** Flags Critical (`≤ -35%`) and Moderate decay thresholds, calculates net lost clicks, and exports filtered audit tables.
- **Zero-Cost Local Playbooks + Multi-LLM Briefs:** Generates actionable on-page, internal linking, and schema refresh steps locally—or drafts custom AI briefs via `litellm`.

## Tech Stack
`Python` • `Streamlit` • `Pandas` • `OpenPyXL` • `LiteLLM`

## Run Locally
```bash
git clone [https://github.com/mashkoorvohra/seo-content-decay-engine.git](https://github.com/mashkoorvohra/seo-content-decay-engine.git)
cd seo-content-decay-engine
pip install -r requirements.txt
streamlit run app.py
