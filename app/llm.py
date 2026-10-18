"""
Factories for the language model and the embedding model.

Only this file knows which provider is used. To switch to another provider,
change these two functions — the rest of the application stays the same.
"""

from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings

from app.config import settings


class PrefixedEmbeddings(Embeddings):
    """
    Adds task prefixes before embedding. Models such as nomic-embed-text are
    trained with "search_query: " / "search_document: " prefixes; using them
    separates relevant from unrelated text much more clearly.
    """

    def __init__(self, base: Embeddings, query_prefix: str, document_prefix: str):
        self.base = base
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.base.embed_documents([self.document_prefix + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self.base.embed_query(self.query_prefix + text)


@lru_cache
def get_llm() -> ChatOllama:
    """Chat model used for question generation."""
    return ChatOllama(
        model=settings.ollama_llm_model,
        base_url=settings.ollama_base_url,
        temperature=settings.llm_temperature,
    )


@lru_cache
def get_embeddings() -> Embeddings:
    """Embedding model used to vectorise chunks and queries."""
    base = OllamaEmbeddings(
        model=settings.ollama_embedding_model,
        base_url=settings.ollama_base_url,
    )
    return PrefixedEmbeddings(base, settings.embed_query_prefix, settings.embed_document_prefix)
