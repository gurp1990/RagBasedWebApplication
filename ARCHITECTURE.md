# Architecture overview

The application is a small **Retrieval-Augmented Generation (RAG)** system that uses the
**Google Gemini API** for embeddings and the LLM, with ChromaDB stored locally.
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
                                   llm.py  ──►  Google Gemini API (GOOGLE_API_KEY)
                                   gemini-embedding-2 (embeddings), gemini-3.5-flash (LLM)
```

Detailed diagrams, including the RAG function map and the database ER diagram, are in
[docs/workflow.html](docs/workflow.html). Open the file in a browser.

## Modules (`app/`)

| File | Responsibility |
|---|---|
| `config.py` | All settings, read from environment variables / `.env`. |
| `llm.py` | Creates the Gemini chat model and embedding model from `GOOGLE_API_KEY`. This is the only place to change to switch providers. |
| `rate_limit.py` | Detects Gemini `429 RESOURCE_EXHAUSTED` errors, waits and retries per-minute limits, and raises `RateLimitError` with a clear message. |
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
2. **Semantic search** – the query is embedded with the same model (`gemini-embedding-2`,
   768 dimensions). Gemini embeds queries with task type `RETRIEVAL_QUERY` and chunks with
   `RETRIEVAL_DOCUMENT`. ChromaDB returns the nearest chunks by cosine similarity. Chunks
   scoring below `MIN_RELEVANCE_SCORE` (default 0.59) are
   dropped, so an off-topic query returns "Not found in the document" instead of
   unrelated passages. The UI shows each matching passage, its page number and a
   relevance score.
3. **Question generation** – the generator picks passages spread across the document and
   runs these steps for each one:
   * The LLM returns structured JSON (a Pydantic schema via `with_structured_output`):
     `source_quote`, `question`, `correct_answer`, three `wrong_answers` and `explanation`.
     Asking for the supporting sentence first keeps the model anchored to the text.
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

* **Hosted models, local data:** Gemini serves both the LLM and the embeddings (no GPU or
  model download needed). ChromaDB persists locally to `./chroma_db_gemini`. The API key is
  read from `.env`, which Git ignores. PDF text and queries are sent to the Gemini API.
* **Rate limits:** every Gemini call (embeddings and LLM) goes through `call_with_retry()` in
  `rate_limit.py`. Per-minute `429` errors are retried after the delay Google suggests;
  daily limits stop at once. Callers get a `RateLimitError` with a user-facing message.
  Stored data is replaced only after a successful write: chunks are upserted before stale
  ones are deleted, and questions are saved only when a whole run succeeds.
* **LangChain is used where it helps:** text splitting, the Chroma and Gemini integrations,
  prompt templates and structured output. Plain Python handles everything else.
* **Easy to change:** settings are in `.env`, prompts in `prompts.py`, and providers in
  `llm.py`. All ChromaDB code is in `vector_store.py`.
* **Traceable answers:** every search result and every question keeps its page number and
  source passage, and the UI shows both.
