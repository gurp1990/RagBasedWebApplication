"""
Factories for the language model and the embedding model (Google Gemini API).

Only this file knows which provider is used. To switch to another provider,
change these two functions — the rest of the application stays the same.

The API key is read from GOOGLE_API_KEY (put it in `.env`, never in code).
"""

from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from app.config import settings


def _api_key() -> str:
    if not settings.google_api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Add it to the .env file in the project root "
            "(get a key at https://aistudio.google.com/apikey)."
        )
    return settings.google_api_key


@lru_cache
def get_llm() -> ChatGoogleGenerativeAI:
    """Chat model used for question generation."""
    return ChatGoogleGenerativeAI(
        model=settings.gemini_llm_model,
        google_api_key=_api_key(),
        temperature=settings.llm_temperature,
        max_retries=settings.llm_max_retries,
    )


@lru_cache
def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """
    Embedding model used to vectorise chunks and queries.

    Gemini embeds chunks with task type RETRIEVAL_DOCUMENT and queries with
    RETRIEVAL_QUERY, which keeps relevant and unrelated text clearly apart.
    """
    return GoogleGenerativeAIEmbeddings(
        model=settings.gemini_embedding_model,
        google_api_key=_api_key(),
        output_dimensionality=settings.embedding_dimensions,
    )
