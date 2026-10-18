"""
Web application (FastAPI + Jinja2 templates).

Pages
  GET  /                     home: documents, upload, ingest samples
  GET  /search?q=...         semantic search
  GET  /questions?source=... generated multiple-choice questions
  POST /upload               upload + ingest a PDF
  POST /ingest-samples       ingest every PDF in ./samples
  POST /questions/generate   generate and store questions for a document

JSON API
  GET  /api/search?q=...
  GET  /api/questions?source=...

Run:  uvicorn app.main:app --reload
"""

import logging
import shutil
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app import vector_store
from app.config import settings
from app.ingestion import ingest_pdf, ingest_samples
from app.question_generator import generate_questions
from app.semantic_search import SAMPLE_QUERIES, search

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

APP_DIR = Path(__file__).parent
WEB_LINKS = {"home": "/", "search": "/search", "questions": "/questions"}


@asynccontextmanager
async def lifespan(_: FastAPI):
    """On first start, ingest the sample PDFs so the app is usable immediately."""
    try:
        if not vector_store.list_sources():
            log.info("Empty database - ingesting sample PDFs: %s", ingest_samples())
    except Exception as error:
        log.error("Could not ingest samples (is Ollama running?): %s", error)
    yield


app = FastAPI(title="RAG PDF Question Generator", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
templates = Jinja2Templates(directory=APP_DIR / "templates")


def render(request: Request, template: str, **context):
    context.setdefault("links", WEB_LINKS)
    context.setdefault("message", request.query_params.get("message"))
    context.setdefault("error", request.query_params.get("error"))
    return templates.TemplateResponse(request, template, context)


def redirect(url: str, message: str = "", error: str = "") -> RedirectResponse:
    if message:
        url += ("&" if "?" in url else "?") + "message=" + quote(message)
    if error:
        url += ("&" if "?" in url else "?") + "error=" + quote(error)
    return RedirectResponse(url, status_code=303)


def default_source(source: str | None) -> str | None:
    sources = vector_store.list_sources()
    if source in sources:
        return source
    return sources[0] if sources else None


# ---------- pages ----------

@app.get("/")
def home(request: Request):
    documents = [
        {"name": s, "chunks": len(vector_store.get_chunks(s)),
         "questions": len(vector_store.get_questions(s))}
        for s in vector_store.list_sources()
    ]
    return render(request, "index.html", documents=documents, settings=settings)


@app.get("/search")
def search_page(request: Request, q: str = "", source: str = ""):
    searches = [{"query": q, "results": search(q, source=source or None)}] if q.strip() else []
    return render(
        request, "search.html",
        query=q, source=source, searches=searches,
        sources=vector_store.list_sources(), sample_queries=SAMPLE_QUERIES,
        min_score=settings.min_relevance_score,
    )


@app.get("/questions")
def questions_page(request: Request, source: str | None = None):
    source = default_source(source)
    questions = vector_store.get_questions(source) if source else []
    return render(
        request, "questions.html",
        source=source, questions=questions, sources=vector_store.list_sources(),
    )


# ---------- actions ----------

@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        return redirect("/", error="Please choose a PDF file.")
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    target = settings.upload_dir / Path(file.filename).name
    with target.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    try:
        count = ingest_pdf(target)
    except Exception as error:
        return redirect("/", error=f"Could not process {target.name}: {error}")
    return redirect("/", message=f"Ingested {target.name}: {count} chunks stored in ChromaDB.")


@app.post("/ingest-samples")
def ingest_sample_pdfs():
    try:
        result = ingest_samples()
    except Exception as error:
        return redirect("/", error=f"Ingestion failed: {error}")
    summary = ", ".join(f"{name} ({n} chunks)" for name, n in result.items()) or "no PDFs found"
    return redirect("/", message=f"Ingested: {summary}")


@app.post("/questions/generate")
def generate(source: str = Form(...)):
    url = f"/questions?source={quote(source)}"
    try:
        questions = generate_questions(source)
    except Exception as error:
        return redirect(url, error=f"Question generation failed: {error}")
    return redirect(url, message=f"Generated and stored {len(questions)} questions.")


# ---------- JSON API ----------

@app.get("/api/search")
def api_search(q: str, top_k: int | None = None, source: str | None = None,
               min_score: float | None = None):
    results = search(q, top_k=top_k, source=source, min_score=min_score)
    return {"query": q, "found": bool(results), "results": [asdict(r) for r in results]}


@app.get("/api/questions")
def api_questions(source: str | None = None):
    source = default_source(source)
    return {"source": source, "questions": vector_store.get_questions(source) if source else []}
