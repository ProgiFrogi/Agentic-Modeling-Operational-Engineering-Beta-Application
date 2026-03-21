from __future__ import annotations

from langchain_openai import ChatOpenAI

from config.settings import get_settings


def get_chat_llm() -> ChatOpenAI:
    s = get_settings()
    kwargs = {"model": s.openai_model, "api_key": s.openai_api_key or None}
    if s.openai_base_url:
        kwargs["base_url"] = s.openai_base_url
    return ChatOpenAI(**kwargs)


def get_planner_llm() -> ChatOpenAI:
    s = get_settings()
    kwargs = {"model": s.openai_planner_model, "api_key": s.openai_api_key or None}
    if s.openai_base_url:
        kwargs["base_url"] = s.openai_base_url
    return ChatOpenAI(**kwargs)
