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
    # --- LLM / embeddings (Ollama) ---
    llm_provider: str = _env("LLM_MODEL", "Ollama")
    ollama_base_url: str = _env("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_llm_model: str = _env("OLLAMA_LLM_MODEL", "gemma3:4b")
    ollama_embedding_model: str = _env("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    llm_temperature: float = float(_env("LLM_TEMPERATURE", "0.2"))

    # --- Document processing ---
    chunk_size: int = int(_env("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(_env("CHUNK_OVERLAP", "80"))

    # --- ChromaDB ---
    chroma_dir: Path = PROJECT_ROOT / _env("CHROMA_DIR", "chroma_db")
    documents_collection: str = _env("DOCUMENTS_COLLECTION", "documents")
    questions_collection: str = _env("QUESTIONS_COLLECTION", "questions")

    # --- Retrieval / question generation ---
    search_top_k: int = int(_env("SEARCH_TOP_K", "4"))
    num_questions: int = int(_env("NUM_QUESTIONS", "5"))

    # --- Files ---
    samples_dir: Path = PROJECT_ROOT / _env("SAMPLES_DIR", "samples")
    upload_dir: Path = PROJECT_ROOT / _env("UPLOAD_DIR", "uploads")
    sample_outputs_dir: Path = PROJECT_ROOT / "sample_outputs"


settings = Settings()
