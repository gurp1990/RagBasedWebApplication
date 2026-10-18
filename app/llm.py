"""
Factories for the language model and the embedding model (Google Gemini API).

Only this file knows which provider is used. To switch to another provider,
change these two functions — the rest of the application stays the same.

The API key is read from GOOGLE_API_KEY (put it in `.env`, never in code).
"""

from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from app.config import settings
from app.rate_limit import call_with_retry


class RetryingEmbeddings(Embeddings):
    """Wraps an embedding model so rate-limit errors are retried (see rate_limit.py)."""

    def __init__(self, base: Embeddings):
        self.base = base

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return call_with_retry(lambda: self.base.embed_documents(texts), "Embedding documents")

    def embed_query(self, text: str) -> list[float]:
        return call_with_retry(lambda: self.base.embed_query(text), "Embedding query")


def _api_key() -> str:
    if not settings.google_api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Add it to the .env file in the project root "
            "(get a key at https://aistudio.google.com/apikey)."
        )
    return settings.google_api_key


@lru_cache
def get_llm() -> ChatGoogleGenerativeAI:
    """
    Chat model used for question generation.

    The SDK's own retries are switched off (1 = a single attempt) because
    callers wrap each request in call_with_retry(), which also understands
    Google's suggested wait time and daily quotas.
    """
    return ChatGoogleGenerativeAI(
        model=settings.gemini_llm_model,
        google_api_key=_api_key(),
        temperature=settings.llm_temperature,
        max_retries=1,
    )


@lru_cache
def get_embeddings() -> Embeddings:
    """
    Embedding model used to vectorise chunks and queries.

    Gemini embeds chunks with task type RETRIEVAL_DOCUMENT and queries with
    RETRIEVAL_QUERY, which keeps relevant and unrelated text clearly apart.
    """
    return RetryingEmbeddings(
        GoogleGenerativeAIEmbeddings(
            model=settings.gemini_embedding_model,
            google_api_key=_api_key(),
            output_dimensionality=settings.embedding_dimensions,
        )
    )
