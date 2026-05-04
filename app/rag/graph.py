from langgraph.graph import END, START, StateGraph

from app.rag.nodes import AgentNodes
from app.rag.state import AgentState


def route_by_decision(state: AgentState) -> str:
    if state["decision"] == "ambiguous":
        return "ask_clarification"

    if state["decision"] == "not_found":
        return "answer_not_found"

    return "generate_answer"


def build_graph(nodes: AgentNodes):
    graph = StateGraph(AgentState)
    graph.add_node("resolve_pending_clarification", nodes.resolve_pending_clarification)
    graph.add_node("resolve_selected_context", nodes.resolve_selected_context)
    graph.add_node("retrieve_docs", nodes.retrieve_docs)
    graph.add_node("filter_by_selected_context", nodes.filter_by_selected_context)
    graph.add_node("classify_context", nodes.classify_context)
    graph.add_node("ask_clarification", nodes.ask_clarification)
    graph.add_node("answer_not_found", nodes.answer_not_found)
    graph.add_node("generate_answer", nodes.generate_answer)

    graph.add_edge(START, "resolve_pending_clarification")
    graph.add_edge("resolve_pending_clarification", "resolve_selected_context")
    graph.add_edge("resolve_selected_context", "retrieve_docs")
    graph.add_edge("retrieve_docs", "filter_by_selected_context")
    graph.add_edge("filter_by_selected_context", "classify_context")
    graph.add_conditional_edges(
        "classify_context",
        route_by_decision,
        {
            "ask_clarification": "ask_clarification",
            "answer_not_found": "answer_not_found",
            "generate_answer": "generate_answer",
        },
    )
    graph.add_edge("ask_clarification", END)
    graph.add_edge("answer_not_found", END)
    graph.add_edge("generate_answer", END)

    return graph.compile()
