"""
Step 2 — Semantic search.

Embed the user's query and return the most similar chunks from ChromaDB,
together with their source page number and a relevance score.
"""

from dataclasses import dataclass

from app import vector_store
from app.config import settings

SAMPLE_QUERIES = [
    "What is the brain of the computer?",
    "Difference between RAM and ROM",
    "Which devices are used for output?",
    "How many bytes are in a kilobyte?",
    "What technology did the second generation of computers use?",
]


@dataclass
class SearchResult:
    text: str
    source: str
    page: int
    chunk_index: int
    score: float  # 0..1, higher is more relevant

    @property
    def score_percent(self) -> int:
        return round(self.score * 100)


def search(query: str, top_k: int | None = None, source: str | None = None) -> list[SearchResult]:
    query = query.strip()
    if not query:
        return []
    matches = vector_store.similarity_search(query, k=top_k or settings.search_top_k, source=source)
    return [
        SearchResult(
            text=doc.page_content,
            source=doc.metadata["source"],
            page=doc.metadata["page"],
            chunk_index=doc.metadata["chunk_index"],
            score=max(0.0, min(1.0, score)),
        )
        for doc, score in matches
    ]
