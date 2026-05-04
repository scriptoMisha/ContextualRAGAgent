from fastapi import FastAPI

from app.api.routes import build_router
from app.core.config import DATA_DIR
from app.core.context import AppContext
from app.rag.graph import build_graph
from app.rag.knowledge_base import get_selectable_labels, load_docs
from app.rag.nodes import AgentNodes
from app.rag.retriever import build_retriever
from app.services.llm import LLMClient
from app.storage.session_store import SessionStore


def create_app() -> FastAPI:
    context = build_context()
    nodes = AgentNodes(context)
    agent_graph = build_graph(nodes)

    app = FastAPI(title="Contextual RAG Agent")
    app.state.context = context
    app.state.agent_graph = agent_graph
    app.include_router(build_router(context, agent_graph))

    return app


def build_context() -> AppContext:
    docs = load_docs(DATA_DIR)
    return AppContext(
        docs=docs,
        selectable_labels=get_selectable_labels(docs),
        retriever=build_retriever(docs),
        llm=LLMClient(),
        sessions=SessionStore(),
    )
