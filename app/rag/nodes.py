from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from app.core.config import (
    ANSWER_HUMAN_PROMPT_TEMPLATE,
    ANSWER_SYSTEM_PROMPT,
    CLARIFICATION_HUMAN_PROMPT_TEMPLATE,
    CLARIFICATION_LLM_ERROR_MESSAGE,
    CLARIFICATION_SYSTEM_PROMPT,
    LLM_ERROR_MESSAGE,
    NOT_FOUND_ANSWER,
    SELECTED_CONTEXT_MESSAGE_LIMIT,
)
from app.core.context import AppContext
from app.rag.retriever import retrieve_relevant_docs
from app.rag.state import AgentState
from app.rag.text_processing import text_has_label


class AgentNodes:
    def __init__(self, context: AppContext) -> None:
        self.context = context

    def resolve_pending_clarification(self, state: AgentState) -> dict[str, Any]:
        pending = state.get("pending")
        if not pending:
            return {"effective_question": state["message"]}

        for option in pending["options"]:
            if text_has_label(state["message"], option):
                return {
                    "effective_question": pending["question"],
                    "pending": None,
                }

        return {"effective_question": state["message"]}

    def resolve_selected_context(self, state: AgentState) -> dict[str, Any]:
        recent_messages = state.get("messages", [])[-SELECTED_CONTEXT_MESSAGE_LIMIT:]

        for message in reversed(recent_messages):
            if not isinstance(message, HumanMessage):
                continue

            for group, label in self.context.selectable_labels:
                if text_has_label(str(message.content), label):
                    return {"selected": {group: label}}

        return {"selected": {}}

    def retrieve_docs(self, state: AgentState) -> dict[str, Any]:
        return {
            "docs": retrieve_relevant_docs(
                self.context.retriever,
                state["effective_question"],
            )
        }

    def filter_by_selected_context(self, state: AgentState) -> dict[str, Any]:
        selected = state.get("selected", {})
        return {
            "docs": [
                doc
                for doc in state.get("docs", [])
                if should_keep_doc(doc, selected)
            ]
        }

    def classify_context(self, state: AgentState) -> dict[str, Any]:
        docs = state.get("docs", [])
        if not docs:
            return {"decision": "not_found"}

        variants_by_group: dict[str, set[str]] = {}

        for doc in docs:
            group = doc.metadata.get("group")
            label = doc.metadata.get("label")

            if group != "general" and label:
                variants_by_group.setdefault(group, set()).add(label)

        for group, labels in variants_by_group.items():
            if len(labels) > 1:
                return {
                    "decision": "ambiguous",
                    "pending": {
                        "group": group,
                        "question": state["effective_question"],
                        "options": sorted(labels),
                    },
                }

        return {"decision": "answerable"}

    def ask_clarification(self, state: AgentState) -> dict[str, Any]:
        return {"answer": self.call_clarification_llm(state["pending"])}

    def answer_not_found(self, _: AgentState) -> dict[str, Any]:
        return {"answer": NOT_FOUND_ANSWER}

    def generate_answer(self, state: AgentState) -> dict[str, Any]:
        context = "\n\n".join(doc.page_content for doc in state.get("docs", []))
        question = state["effective_question"]
        messages = state.get("messages", [])

        return {"answer": self.call_answer_llm(context, question, messages)}

    def call_answer_llm(
        self,
        context: str,
        question: str,
        messages: list[BaseMessage],
    ) -> str:
        llm_messages = [
            SystemMessage(content=ANSWER_SYSTEM_PROMPT),
            *messages[-8:],
            HumanMessage(
                content=ANSWER_HUMAN_PROMPT_TEMPLATE.format(
                    context=context,
                    question=question,
                )
            ),
        ]

        return self.context.llm.invoke(llm_messages, LLM_ERROR_MESSAGE)

    def call_clarification_llm(self, pending: dict[str, Any]) -> str:
        labels = ", ".join(pending["options"])
        llm_messages = [
            SystemMessage(content=CLARIFICATION_SYSTEM_PROMPT),
            HumanMessage(
                content=CLARIFICATION_HUMAN_PROMPT_TEMPLATE.format(
                    group=pending["group"],
                    question=pending["question"],
                    labels=labels,
                )
            ),
        ]

        return self.context.llm.invoke(llm_messages, CLARIFICATION_LLM_ERROR_MESSAGE)


def should_keep_doc(doc: Document, selected: dict[str, str]) -> bool:
    group = doc.metadata.get("group")
    label = doc.metadata.get("label")
    return not (group in selected and label) or selected[group] == label
