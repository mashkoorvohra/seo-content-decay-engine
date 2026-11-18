import streamlit as st
import requests

@st.cache_data(ttl=3600)
def fetch_live_models(provider_name: str, key: str) -> list:
    defaults = {
        "Google Gemini": ["gemini-4-hydrogen", "gemini-4-carbon", "gemini-4-argon", "gemini-3.5-flash-lite"],
        "Anthropic Claude": ["claude-sonnet-5-5", "claude-haiku-5-5", "claude-opus-5-5", "claude-fable-5-2"],
        "OpenAI": ["gpt-6.2-astra", "gpt-6.2-sol", "gpt-6-luna", "gpt-5.4-mini"],
        "Groq / OpenRouter": ["groq/llama-3.3-70b-versatile", "openrouter/anthropic/claude-sonnet-5-5"]
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

def render_ai_sidebar():
    st.sidebar.header("⚙️ AI Provider Settings")
    provider = st.sidebar.selectbox(
        "Select AI Engine",
        ["Free Local Engine (No API Key Needed)", "Google Gemini", "Anthropic Claude", "OpenAI", "Groq / OpenRouter"]
    )
    api_key = ""
    selected_model = ""

    if provider != "Free Local Engine (No API Key Needed)":
        api_key = st.sidebar.text_input(f"Enter {provider} API Key", type="password")
        models = fetch_live_models(provider, api_key)
        if provider == "Groq / OpenRouter":
            selected_model = st.sidebar.text_input("Model ID", value=models[0])
        else:
            selected_model = st.sidebar.selectbox("Select Model (Auto-Updated)", models)
            prefix_map = {"Google Gemini": "gemini/", "Anthropic Claude": "anthropic/", "OpenAI": "openai/"}
            if not selected_model.startswith(prefix_map[provider]):
                selected_model = prefix_map[provider] + selected_model

    return provider, api_key, selected_model
