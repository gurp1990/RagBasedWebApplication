"""Ingestion pipeline: PDF file -> chunks -> ChromaDB."""

from pathlib import Path

from app import vector_store
from app.config import settings
from app.document_processor import process_pdf


def ingest_pdf(pdf_path: Path) -> int:
    """Process one PDF and store its chunks. Returns the number of chunks."""
    chunks = process_pdf(pdf_path)
    return vector_store.save_chunks(chunks)


def ingest_samples() -> dict[str, int]:
    """Ingest every PDF in the samples folder. Returns {file name: chunk count}."""
    return {pdf.name: ingest_pdf(pdf) for pdf in sorted(settings.samples_dir.glob("*.pdf"))}
