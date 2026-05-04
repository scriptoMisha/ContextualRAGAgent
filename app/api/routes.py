import uuid

from fastapi import APIRouter, HTTPException
from langchain_core.messages import AIMessage, HumanMessage

from app.core.context import AppContext
from app.rag.state import AgentState
from app.schemas.chat import ChatRequest, ChatResponse


def build_router(context: AppContext, agent_graph) -> APIRouter:
    router = APIRouter()

    @router.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        session_id = request.session_id or str(uuid.uuid4())
        memory = context.sessions.get(session_id)
        messages = [
            *memory.get("messages", []),
            HumanMessage(content=request.message),
        ]

        state: AgentState = {
            "message": request.message,
            "messages": messages,
            "pending": memory.get("pending"),
        }

        try:
            result = agent_graph.invoke(state)
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        context.sessions.save(
            session_id=session_id,
            messages=[*messages, AIMessage(content=result["answer"])],
            pending=result.get("pending"),
        )

        return ChatResponse(answer=result["answer"], session_id=session_id)


    return router
