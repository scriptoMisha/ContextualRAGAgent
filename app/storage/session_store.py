from typing import Any

from langchain_core.messages import BaseMessage


class SessionStore:
    def __init__(self) -> None:
        self.sessions: dict[str, dict[str, Any]] = {}

    def get(self, session_id: str) -> dict[str, Any]:
        return self.sessions.get(session_id, {"messages": [], "pending": None})

    def save(
        self,
        session_id: str,
        messages: list[BaseMessage],
        pending: dict[str, Any] | None,
    ) -> None:
        self.sessions[session_id] = {
            "messages": messages,
            "pending": pending,
        }
