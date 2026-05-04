import os
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage


class LLMClient:
    def __init__(self) -> None:
        self.llm = None

    def invoke(self, messages: list[BaseMessage], error_message: str) -> str:
        try:
            response = self.get_model().invoke(messages)
        except Exception as exc:
            raise RuntimeError(f"{error_message}: {exc}") from exc

        return response.content.strip()

    def get_model(self):
        if self.llm is not None:
            return self.llm

        provider = os.getenv("LLM_PROVIDER", "openai").lower()
        model = os.getenv("LLM_MODEL", "gpt-5.5")
        api_key = os.getenv("LLM_API_KEY")
        base_url = optional_env("LLM_BASE_URL")

        if not api_key:
            raise RuntimeError("Не задан LLM_API_KEY. Добавьте ключ в .env или переменные окружения.")

        llm_kwargs: dict[str, Any] = {
            "model": model,
            "model_provider": provider,
            "api_key": api_key,
            "temperature": 1,
        }

        if base_url:
            llm_kwargs["base_url"] = base_url

        self.llm = init_chat_model(**llm_kwargs)
        return self.llm


def optional_env(name: str) -> str | None:
    value = os.getenv(name)
    if not value:
        return None

    value = value.strip()
    if not value or value.lower() == "none":
        return None

    return value
