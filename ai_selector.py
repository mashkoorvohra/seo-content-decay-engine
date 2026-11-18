import streamlit as st
import requests

DEFAULT_MODELS = {
    "Google Gemini": [
        "gemini/gemini-3.5-flash-lite",
        "gemini/gemini-3.8-flash",
        "gemini/gemini-4-hydrogen",
        "gemini/gemini-4-carbon",
    ],
    "Anthropic Claude": [
        "anthropic/claude-haiku-5-5",
        "anthropic/claude-sonnet-5-5",
        "anthropic/claude-opus-5-5",
        "anthropic/claude-fable-5-2",
    ],
    "OpenAI": [
        "openai/gpt-6-luna",
        "openai/gpt-6-sol",
        "openai/gpt-6-astra",
        "openai/gpt-5.6-luna",
    ],
    "Groq / OpenRouter / Custom": [
        "groq/llama-3.3-70b-versatile",
        "openrouter/anthropic/claude-sonnet-5.5",
        "deepseek/deepseek-chat",
    ],
}

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_live_models(provider: str, api_key: str) -> list:
    """Queries provider APIs in real time to list the newest available models."""
    if not api_key:
        return DEFAULT_MODELS.get(provider, [])
    try:
        if provider == "Google Gemini":
            r = requests.get(
                f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}",
                timeout=6,
            )
            if r.status_code == 200:
                models = [
                    "gemini/" + m["name"].replace("models/", "")
                    for m in r.json().get("models", [])
                    if "generateContent" in m.get("supportedGenerationMethods", [])
                ]
                # Show newest versions (Gemini 4 / 3.8 / 3.5) at the top
                return sorted(models, reverse=True) or DEFAULT_MODELS[provider]

        elif provider == "Anthropic Claude":
            r = requests.get(
                "https://api.anthropic.com/v1/models",
                headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
                timeout=6,
            )
            if r.status_code == 200:
                models = ["anthropic/" + m["id"] for m in r.json().get("data", [])]
                return models or DEFAULT_MODELS[provider]

        elif provider == "OpenAI":
            r = requests.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=6,
            )
            if r.status_code == 200:
                models = [
                    "openai/" + m["id"]
                    for m in r.json().get("data", [])
                    if m["id"].startswith(("gpt-6", "gpt-5", "o3", "o4"))
                ]
                return sorted(models, reverse=True) or DEFAULT_MODELS[provider]
    except Exception:
        pass
    return DEFAULT_MODELS.get(provider, [])

def render_ai_sidebar():
    """Renders a universal, auto-updating AI provider & model selector."""
    st.sidebar.header("⚙️ AI Engine Configuration")
    provider = st.sidebar.selectbox(
        "AI Provider",
        ["Free Local Mode (0 API Credits)"] + list(DEFAULT_MODELS.keys()),
    )

    if provider == "Free Local Mode (0 API Credits)":
        return provider, None, None

    api_key = st.sidebar.text_input(f"{provider} API Key", type="password")
    available_models = fetch_live_models(provider, api_key)

    selected_model = st.sidebar.selectbox(
        "Select Model (Auto-Synced with Latest API)",
        available_models + ["Custom Model ID..."],
    )

    if selected_model == "Custom Model ID...":
        selected_model = st.sidebar.text_input(
            "Enter LiteLLM Model String",
            value=available_models[0] if available_models else "",
        )

    return provider, api_key, selected_model
