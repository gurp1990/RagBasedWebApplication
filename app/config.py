"""
Central configuration.

Every setting can be overridden with an environment variable or a `.env` file
in the project root, so nothing needs to be changed in code.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class Settings:
    # --- LLM / embeddings (Google Gemini API) ---
    llm_provider: str = _env("LLM_MODEL", "Gemini")
    google_api_key: str = _env("GOOGLE_API_KEY", "")
    gemini_llm_model: str = _env("GEMINI_LLM_MODEL", "gemini-3.5-flash")
    gemini_embedding_model: str = _env("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
    embedding_dimensions: int = int(_env("EMBEDDING_DIMENSIONS", "768"))
    llm_temperature: float = float(_env("LLM_TEMPERATURE", "0.2"))
    llm_max_retries: int = int(_env("LLM_MAX_RETRIES", "6"))  # rides out API rate limits

    # --- Document processing ---
    chunk_size: int = int(_env("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(_env("CHUNK_OVERLAP", "80"))

    # --- ChromaDB ---
    chroma_dir: Path = PROJECT_ROOT / _env("CHROMA_DIR", "chroma_db_gemini")
    documents_collection: str = _env("DOCUMENTS_COLLECTION", "documents")
    questions_collection: str = _env("QUESTIONS_COLLECTION", "questions")

    # --- Retrieval / question generation ---
    search_top_k: int = int(_env("SEARCH_TOP_K", "4"))
    # Passages scoring below this (0..1) are treated as "not in the document".
    min_relevance_score: float = float(_env("MIN_RELEVANCE_SCORE", "0.59"))
    num_questions: int = int(_env("NUM_QUESTIONS", "5"))

    # --- Files ---
    samples_dir: Path = PROJECT_ROOT / _env("SAMPLES_DIR", "samples")
    upload_dir: Path = PROJECT_ROOT / _env("UPLOAD_DIR", "uploads")
    sample_outputs_dir: Path = PROJECT_ROOT / "sample_outputs"


settings = Settings()
