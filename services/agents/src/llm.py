from __future__ import annotations

import os

from langchain_core.language_models import BaseChatModel


def create_chat_model(
    provider: str | None = None,
    model_name: str | None = None,
    *,
    temperature: float = 0.0,
) -> BaseChatModel:
    provider = (provider or os.getenv("LLM_PROVIDER", "openai")).lower()
    model_name = model_name or os.getenv("LLM_MODEL", "gpt-4o-mini")
    base_url = os.getenv("LLM_BASE_URL") or None

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_name,
            api_key=os.getenv("OPENAI_API_KEY") or None,
            base_url=base_url,
            temperature=temperature,
        )
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model_name,
            api_key=os.getenv("ANTHROPIC_API_KEY") or None,
            temperature=temperature,
        )
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=os.getenv("GEMINI_API_KEY") or None,
            temperature=temperature,
        )
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model_name,
            base_url=base_url or "http://localhost:11434",
            temperature=temperature,
        )
    if provider == "vllm":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_name,
            base_url=base_url or "http://localhost:8000/v1",
            api_key=os.getenv("VLLM_API_KEY", "EMPTY"),
            temperature=temperature,
        )
    raise ValueError("LLM_PROVIDER must be openai, anthropic, gemini, ollama, or vllm")
