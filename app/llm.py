"""
Factories for the language model and the embedding model (Google Gemini API).

Only this file knows which provider is used. To switch to another provider,
change these functions — the rest of the application stays the same.

The API key is read from GOOGLE_API_KEY in `.env` (or the environment), never
from code. The models are created only when they are first used, so pages
that just read stored data keep working while no key is set.
"""

import os
from functools import lru_cache

from dotenv import dotenv_values
from langchain_core.embeddings import Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from app.config import PROJECT_ROOT, settings
from app.rate_limit import call_with_retry

API_KEY_URL = "https://aistudio.google.com/apikey"
PLACEHOLDER_KEYS = {"", "your-gemini-api-key-here"}  # the value shipped in .env.example


class MissingApiKeyError(RuntimeError):
    """No usable GOOGLE_API_KEY is configured."""

    def __init__(self):
        super().__init__(
            "Please provide a Gemini API key. Add GOOGLE_API_KEY=<your key> to the .env file "
            f"in the project folder. You can get a free key at {API_KEY_URL}"
        )


# ---------- API key ----------

def current_api_key() -> str | None:
    """
    The configured key, or None if it is missing or still the placeholder.
    `.env` is re-read on every call, so a newly added key works without a restart.
    """
    key = dotenv_values(PROJECT_ROOT / ".env").get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    key = key.strip().strip("'\"")
    return None if key in PLACEHOLDER_KEYS else key


def api_key_configured() -> bool:
    return current_api_key() is not None


def _api_key() -> str:
    key = current_api_key()
    if key is None:
        raise MissingApiKeyError()
    return key


# ---------- models ----------

def get_llm() -> ChatGoogleGenerativeAI:
    """Chat model used for question generation. Raises MissingApiKeyError without a key."""
    return _build_llm(_api_key())


@lru_cache
def _build_llm(api_key: str) -> ChatGoogleGenerativeAI:
    # The SDK's own retries are switched off (1 = a single attempt) because
    # callers wrap each request in call_with_retry(), which also understands
    # Google's suggested wait time and daily quotas.
    return ChatGoogleGenerativeAI(
        model=settings.gemini_llm_model,
        google_api_key=api_key,
        temperature=settings.llm_temperature,
        max_retries=1,
    )


@lru_cache
def _build_embedding_model(api_key: str) -> GoogleGenerativeAIEmbeddings:
    # Gemini embeds chunks with task type RETRIEVAL_DOCUMENT and queries with
    # RETRIEVAL_QUERY, which keeps relevant and unrelated text clearly apart.
    return GoogleGenerativeAIEmbeddings(
        model=settings.gemini_embedding_model,
        google_api_key=api_key,
        output_dimensionality=settings.embedding_dimensions,
    )


class GeminiEmbeddings(Embeddings):
    """
    Embedding model handed to ChromaDB. The Gemini client is created on first
    use (raising MissingApiKeyError if no key is set), and every call retries
    rate-limit errors (see rate_limit.py).
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        model = _build_embedding_model(_api_key())
        return call_with_retry(lambda: model.embed_documents(texts), "Embedding documents")

    def embed_query(self, text: str) -> list[float]:
        model = _build_embedding_model(_api_key())
        return call_with_retry(lambda: model.embed_query(text), "Embedding query")


@lru_cache
def get_embeddings() -> Embeddings:
    """Embedding model used to vectorise chunks and queries."""
    return GeminiEmbeddings()
