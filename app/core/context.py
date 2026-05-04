from dataclasses import dataclass

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from app.services.llm import LLMClient
from app.storage.session_store import SessionStore


@dataclass
class AppContext:
    docs: list[Document]
    selectable_labels: list[tuple[str, str]]
    retriever: BM25Retriever
    llm: LLMClient
    sessions: SessionStore
