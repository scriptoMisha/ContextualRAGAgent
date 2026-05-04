from typing import Any, Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage


class AgentState(TypedDict, total=False):
    message: str
    messages: list[BaseMessage]
    effective_question: str
    docs: list[Document]
    selected: dict[str, str]
    pending: dict[str, Any] | None
    decision: Literal["answerable", "ambiguous", "not_found"]
    answer: str
