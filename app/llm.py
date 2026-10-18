"""
Factories for the language model and the embedding model.

Only this file knows which provider is used. To switch to another provider,
change these two functions — the rest of the application stays the same.
"""

from functools import lru_cache

from langchain_ollama import ChatOllama, OllamaEmbeddings

from app.config import settings


@lru_cache
def get_llm() -> ChatOllama:
    """Chat model used for question generation."""
    return ChatOllama(
        model=settings.ollama_llm_model,
        base_url=settings.ollama_base_url,
        temperature=settings.llm_temperature,
    )


@lru_cache
def get_embeddings() -> OllamaEmbeddings:
    """Embedding model used to vectorise chunks and queries."""
    return OllamaEmbeddings(
        model=settings.ollama_embedding_model,
        base_url=settings.ollama_base_url,
    )
