"""
ChromaDB access.

Two persistent collections are used:
  * documents — PDF chunks (text + source/page metadata)
  * questions — generated multiple-choice questions

All ChromaDB details live here so the other modules stay simple.
"""

from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import settings
from app.llm import get_embeddings


def _open_collection(name: str) -> Chroma:
    return Chroma(
        collection_name=name,
        embedding_function=get_embeddings(),
        persist_directory=str(settings.chroma_dir),
        collection_metadata={"hnsw:space": "cosine"},  # relevance = 1 - cosine distance
    )


@lru_cache
def documents_store() -> Chroma:
    return _open_collection(settings.documents_collection)


@lru_cache
def questions_store() -> Chroma:
    return _open_collection(settings.questions_collection)


# ---------- documents ----------

def save_chunks(chunks: list[Document]) -> int:
    """
    Store chunks, replacing any earlier version of the same source file.

    New chunks are written first (upsert by id) and old leftovers removed
    afterwards, so if embedding fails (e.g. API rate limit) the previous
    version of the document is still there.
    """
    if not chunks:
        return 0
    source = chunks[0].metadata["source"]
    ids = [f"{c.metadata['source']}::chunk-{c.metadata['chunk_index']}" for c in chunks]
    documents_store().add_documents(chunks, ids=ids)
    delete_by_source(documents_store(), source, keep=set(ids))
    return len(chunks)


def get_chunks(source: str) -> list[Document]:
    """All chunks of one source, in their original order."""
    result = documents_store().get(where={"source": source})
    chunks = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(result["documents"], result["metadatas"])
    ]
    return sorted(chunks, key=lambda c: c.metadata["chunk_index"])


def list_sources() -> list[str]:
    """Names of all PDF files stored in the documents collection."""
    metadatas = documents_store().get(include=["metadatas"])["metadatas"]
    return sorted({meta["source"] for meta in metadatas})


def similarity_search(
    query: str, k: int, min_score: float = 0.0, source: str | None = None
) -> list[tuple[Document, float]]:
    """Return (chunk, relevance score 0..1) pairs scoring at least `min_score`, best first."""
    where = {"source": source} if source else None
    matches = documents_store().similarity_search_with_relevance_scores(query, k=k, filter=where)
    return [(doc, score) for doc, score in matches if score >= min_score]


# ---------- questions ----------

def save_questions(source: str, questions: list[dict]) -> None:
    """Replace all stored questions for a source with a new set (written first, then old ones removed)."""
    store = questions_store()
    ids = [f"{source}::q-{i}" for i in range(len(questions))]
    if questions:
        texts = [q["question"] for q in questions]
        metadatas = [{**_flatten_question(q), "source": source, "number": i + 1}
                     for i, q in enumerate(questions)]
        store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    delete_by_source(store, source, keep=set(ids))


def get_questions(source: str) -> list[dict]:
    """Stored questions for a source, in their original order."""
    result = questions_store().get(where={"source": source})
    questions = [_unflatten_question(meta) for meta in result["metadatas"]]
    return sorted(questions, key=lambda q: q["number"])


# ---------- helpers ----------

def delete_by_source(store: Chroma, source: str, keep: set[str] = frozenset()) -> None:
    """Delete every record of `source` except the ids in `keep`."""
    stale = [i for i in store.get(where={"source": source}, include=[])["ids"] if i not in keep]
    if stale:
        store.delete(ids=stale)


# ChromaDB metadata only accepts flat values, so the option list is stored
# as separate fields: option_1 ... option_4.

def _flatten_question(q: dict) -> dict:
    flat = {k: v for k, v in q.items() if k != "options"}
    for i, option in enumerate(q["options"], start=1):
        flat[f"option_{i}"] = option
    return flat


def _unflatten_question(meta: dict) -> dict:
    option_keys = sorted(k for k in meta if k.startswith("option_"))
    question = {k: v for k, v in meta.items() if not k.startswith("option_")}
    question["options"] = [meta[k] for k in option_keys]
    return question
