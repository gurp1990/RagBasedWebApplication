"""
Step 1 — Document processing.

Load a PDF, extract the text page by page and split it into overlapping
chunks. Every chunk keeps metadata (source file, page number, chunk index)
so search results and questions can point back to where they came from.
"""

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.config import settings


def clean_text(text: str) -> str:
    """Normalise whitespace and replace characters that often break in PDFs."""
    text = text.replace("\x7f", "•")  # bullet glyphs are often extracted as DEL
    lines = [" ".join(line.split()) for line in text.splitlines()]
    merged: list[str] = []
    for line in filter(None, lines):
        # A line starting in lower case continues the previous sentence (PDF line wrap).
        if merged and (line[0].islower() or line[0] == "("):
            merged[-1] += " " + line
        else:
            merged.append(line)
    return "\n".join(merged)


def load_pdf(pdf_path: Path) -> list[Document]:
    """Return one Document per PDF page, with the page number (1-based) in metadata."""
    reader = PdfReader(str(pdf_path))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:
            pages.append(
                Document(
                    page_content=text,
                    metadata={"source": pdf_path.name, "page": page_number},
                )
            )
    return pages


def split_into_chunks(pages: list[Document]) -> list[Document]:
    """Split pages into chunks. Page metadata is copied to every chunk."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index
    return chunks


def process_pdf(pdf_path: Path) -> list[Document]:
    """Load + chunk a PDF in one call."""
    pages = load_pdf(pdf_path)
    if not pages:
        raise ValueError(f"No extractable text found in {pdf_path.name}")
    return split_into_chunks(pages)
