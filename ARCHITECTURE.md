# Architecture overview

The application is a small, fully local **Retrieval-Augmented Generation (RAG)** system.
A PDF is turned into searchable vector chunks in ChromaDB. Those chunks are used for
semantic search (retrieval) and as the only knowledge source for question generation.

```
                ┌──────────────────────── Web UI (FastAPI + Jinja2) ────────────────────────┐
                │   /  Documents          /search  Semantic Search      /questions  MCQs    │
                └───────┬──────────────────────────┬──────────────────────────┬─────────────┘
                        │ upload / ingest          │ query                    │ generate
                        ▼                          ▼                          ▼
 PDF ──► document_processor.py ──► ingestion.py   semantic_search.py     question_generator.py
         (pypdf: text per page,         │               │                  │  1 pick passages
          clean, chunk + metadata)      │               │                  │  2 LLM drafts MCQ (JSON)
                                        ▼               ▼                  │  3 quote check (code)
                              ┌───────── vector_store.py (ChromaDB) ──────┐│  4 blind-answer check (LLM)
                              │ collection "documents": chunks            │◄┘ 5 store questions
                              │ collection "questions": MCQs              │
                              └──────────────┬────────────────────────────┘
                                             │ embeddings / chat
                                             ▼
                                   llm.py  ──►  Ollama
                                   nomic-embed-text (embeddings), gemma3:4b (LLM)
```

## Modules (`app/`)

| File | Responsibility |
|---|---|
| `config.py` | All settings, read from environment variables / `.env`. |
| `llm.py` | Creates the Ollama chat model and embedding model. This is the only place to change to switch providers. |
| `document_processor.py` | Extracts text page by page with `pypdf`. Cleans it (bullets, wrapped lines). Splits it with LangChain's `RecursiveCharacterTextSplitter`. Every chunk keeps `source`, `page` and `chunk_index`. |
| `vector_store.py` | Everything ChromaDB: two persistent collections (`documents`, `questions`) using cosine distance. Saving, reading, similarity search and replacing data by source file. |
| `ingestion.py` | Pipeline: PDF → chunks → ChromaDB. Re-ingesting a file replaces its old chunks. |
| `semantic_search.py` | Embeds the query and returns the top-k chunks with page number and relevance score. Also holds the demo queries. |
| `prompts.py` | The two LLM prompts (generate, verify), kept apart from the logic so they are easy to tune. |
| `question_generator.py` | Generates, validates, verifies and stores the multiple-choice questions. |
| `main.py` | FastAPI web app: HTML pages + JSON API. |
| `cli.py` | Command-line tool: `ingest`, `search`, `generate`, `export` (static sample pages). |
| `templates/`, `static/` | Jinja2 HTML templates and CSS. The same templates render both the live pages and the static sample outputs. |

## Data flow

1. **Document processing** – `pypdf` extracts the text of each page. Cleaning replaces
   broken bullet glyphs and rejoins sentences that the PDF wrapped across lines. The text is
   split into ~500-character chunks with 80 characters of overlap. Each chunk is stored in
   ChromaDB with its page number, under an id like `computer_basics.pdf::chunk-3`.
2. **Semantic search** – the query is embedded with the same model (`nomic-embed-text`).
   Queries get the `search_query: ` prefix and chunks the `search_document: ` prefix, as the
   model expects (see `PrefixedEmbeddings` in `llm.py`). ChromaDB returns the nearest chunks
   by cosine similarity. Chunks scoring below `MIN_RELEVANCE_SCORE` (default 0.6) are
   dropped, so an off-topic query returns "Not found in the document" instead of
   unrelated passages. The UI shows each matching passage, its page number and a
   relevance score.
3. **Question generation** – the generator picks passages spread across the document and
   runs these steps for each one:
   * The LLM returns structured JSON (a Pydantic schema via `with_structured_output`):
     `source_quote`, `question`, `correct_answer`, three `wrong_answers` and `explanation`.
     Asking for the supporting sentence first keeps the small model anchored to the text.
   * **Code checks:** exactly 3 distinct wrong answers, no duplicate question, and the quote
     must really appear in the passage (≥ 80 % word overlap).
   * The code shuffles the four options, so the correct answer is always one of them.
   * **Blind answer check:** the LLM answers the finished question from the passage *without*
     seeing the intended answer. The question is kept only if both answers agree. This
     rejects ambiguous questions and questions with a wrong answer key.
   * A failed passage is retried (up to 3 times) and then replaced by another passage.

   The accepted questions are stored in the `questions` collection. The question text is
   embedded; the options, answer, explanation, page, quote and source passage are stored as
   metadata.

## Design choices

* **Everything runs locally:** Ollama serves both the LLM and the embeddings, and ChromaDB
  persists to `./chroma_db`. No API keys are needed.
* **LangChain is used where it helps:** text splitting, the Chroma and Ollama integrations,
  prompt templates and structured output. Plain Python handles everything else.
* **Easy to change:** settings are in `.env`, prompts in `prompts.py`, and providers in
  `llm.py`. All ChromaDB code is in `vector_store.py`.
* **Traceable answers:** every search result and every question keeps its page number and
  source passage, and the UI shows both.
