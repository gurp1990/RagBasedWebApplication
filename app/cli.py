"""
Command-line interface.

    python -m app.cli ingest [PDF ...]       # default: every PDF in ./samples
    python -m app.cli search "your query"    # semantic search demo
    python -m app.cli search                 # run the built-in sample queries
    python -m app.cli generate [SOURCE]      # generate + store 5 MCQs
    python -m app.cli export                 # write sample_outputs/*.html and *.json
"""

import argparse
import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from app import vector_store
from app.config import settings
from app.ingestion import ingest_pdf, ingest_samples
from app.question_generator import generate_questions
from app.semantic_search import SAMPLE_QUERIES, search

APP_DIR = Path(__file__).parent
EXPORT_LINKS = {"home": "index.html", "search": "semantic_search.html", "questions": "questions.html"}


def cmd_ingest(args) -> None:
    if args.pdfs:
        result = {Path(p).name: ingest_pdf(Path(p)) for p in args.pdfs}
    else:
        result = ingest_samples()
    for name, count in result.items():
        print(f"Ingested {name}: {count} chunks")


def cmd_search(args) -> None:
    for query in [args.query] if args.query else SAMPLE_QUERIES:
        print(f'\nQuery: "{query}"')
        results = search(query, top_k=args.top_k)
        if not results:
            print(f"  Not found in the document (no passage >= {settings.min_relevance_score:.2f}).")
        for i, r in enumerate(results, start=1):
            preview = r.text.replace("\n", " ")[:160]
            print(f"  {i}. [page {r.page} | score {r.score:.2f}] {preview}...")


def cmd_generate(args) -> None:
    source = args.source or _first_source()
    print(f"Generating questions for {source} with {settings.ollama_llm_model} ...")
    for i, q in enumerate(generate_questions(source), start=1):
        print(f"\nQ{i}. {q['question']}  (page {q['page']})")
        for letter, option in zip("ABCD", q["options"]):
            mark = "*" if option == q["correct_answer"] else " "
            print(f"   {mark} {letter}) {option}")
        print(f"   Explanation: {q['explanation']}")
    print("\nStored in ChromaDB collection:", settings.questions_collection)


def cmd_export(args) -> None:
    """Render the web pages to static HTML (CSS inlined) using the stored data."""
    source = args.source or _first_source()
    out = settings.sample_outputs_dir
    out.mkdir(exist_ok=True)

    questions = vector_store.get_questions(source)
    if not questions:
        print("No stored questions - generating them first ...")
        questions = generate_questions(source)
    searches = [{"query": q, "results": search(q)} for q in SAMPLE_QUERIES]
    documents = [{"name": s, "chunks": len(vector_store.get_chunks(s)),
                  "questions": len(vector_store.get_questions(s))} for s in vector_store.list_sources()]

    env = Environment(loader=FileSystemLoader(APP_DIR / "templates"), autoescape=select_autoescape())
    common = {
        "links": EXPORT_LINKS,
        "static_export": True,
        "inline_css": (APP_DIR / "static" / "style.css").read_text(encoding="utf-8"),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "model": settings.ollama_llm_model,
        "settings": settings,
        "min_score": settings.min_relevance_score,
    }
    pages = {
        "index.html": ("index.html", {"documents": documents}),
        "semantic_search.html": ("search.html", {"searches": searches}),
        "questions.html": ("questions.html", {"source": source, "questions": questions}),
    }
    for filename, (template, context) in pages.items():
        html = env.get_template(template).render(**common, **context)
        (out / filename).write_text(html, encoding="utf-8")

    (out / "semantic_search.json").write_text(json.dumps(
        [{"query": s["query"], "results": [asdict(r) for r in s["results"]]} for s in searches],
        indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "questions.json").write_text(json.dumps(
        {"source": source, "questions": questions}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Sample outputs written to {out}")


def _first_source() -> str:
    sources = vector_store.list_sources()
    if not sources:
        raise SystemExit("No documents in ChromaDB. Run: python -m app.cli ingest")
    return sources[0]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="RAG PDF assistant")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="load PDFs into ChromaDB")
    p.add_argument("pdfs", nargs="*", help="PDF files (default: samples folder)")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("search", help="semantic search")
    p.add_argument("query", nargs="?", help="query text (default: sample queries)")
    p.add_argument("--top-k", type=int, default=settings.search_top_k)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("generate", help="generate multiple-choice questions")
    p.add_argument("source", nargs="?", help="document file name (default: first stored)")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("export", help="write static sample output pages")
    p.add_argument("source", nargs="?", help="document file name (default: first stored)")
    p.set_defaults(func=cmd_export)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
