from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from app.rag.text_processing import normalize_text


def build_retriever(docs: list[Document], k: int = 4) -> BM25Retriever:
    retriever = BM25Retriever.from_documents(
        documents=docs,
        preprocess_func=normalize_text,
    )
    retriever.k = k
    return retriever


def retrieve_relevant_docs(retriever: BM25Retriever, question: str) -> list[Document]:
    query_tokens = set(normalize_text(question))
    found_docs = retriever.invoke(question)

    return [
        doc
        for doc in found_docs
        if query_tokens & set(normalize_text(doc.page_content))
    ]
