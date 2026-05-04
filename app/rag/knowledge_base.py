from pathlib import Path

from langchain_core.documents import Document


DOC_SPECS = {
    "general_info.txt": {"group": "general", "label": None},
    "deadlines.txt": {"group": "general", "label": None},
    "benefits.txt": {"group": "general", "label": None},
    "germany_rules.txt": {"group": "country_rules", "label": "Германия"},
    "france_rules.txt": {"group": "country_rules", "label": "Франция"},
}


def load_docs(data_dir: Path) -> list[Document]:
    documents = []

    for filename, metadata in DOC_SPECS.items():
        path = data_dir / filename
        documents.append(
            Document(
                page_content=path.read_text(encoding="utf-8"),
                metadata={"source": filename, **metadata},
            )
        )

    return documents


def get_selectable_labels(docs: list[Document]) -> list[tuple[str, str]]:
    return [
        (doc.metadata["group"], doc.metadata["label"])
        for doc in docs
        if doc.metadata.get("group") != "general" and doc.metadata.get("label")
    ]
