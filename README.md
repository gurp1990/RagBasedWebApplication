# RAG PDF Assistant — Semantic Search & Question Generation

A fully local **Retrieval-Augmented Generation** web application written in Python.

* **Document processing:** loads a PDF, extracts the text page by page, splits it into chunks
  with page-number metadata and stores them in **ChromaDB**.
* **Semantic search:** finds the passages most relevant to a question in your own words and
  shows each passage with its **source page**.
* **Question generation:** creates **5 multiple-choice questions** (4 options, the correct
  answer and an explanation) from the document. Each question is fact-checked against its
  source passage and then stored in ChromaDB.
* **Web UI:** HTML pages for documents, semantic search and an interactive quiz, plus a
  JSON API.

Stack: **Python · LangChain · ChromaDB · Ollama (`gemma3:4b` LLM, `nomic-embed-text` embeddings) · FastAPI**

See [ARCHITECTURE.md](ARCHITECTURE.md) for the architecture overview, and
[docs/workflow.html](docs/workflow.html) for the diagrams. Download that file and open it in a
browser, since GitHub shows HTML files as source code. It contains:

* **RAG at a glance:** where Retrieve, Augment and Generate happen in the app
* **RAG function map:** every function on the RAG path, in call order
* **Step-by-step workflows:** document processing, semantic search and question generation
* **Database ER diagram:** the ChromaDB collections `documents` and `questions` and their fields
* **Function index:** every file in `app/` and its main functions

---

## 1. Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.11+ | Tested with 3.11 |
| [Ollama](https://ollama.com) | Must be running locally (`http://localhost:11434`) |
| Ollama models | `ollama pull gemma3:4b` and `ollama pull nomic-embed-text` |

## 2. Setup

```bash
git clone <this-repo> RagBasedWebApplication
cd RagBasedWebApplication

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt

ollama pull gemma3:4b
ollama pull nomic-embed-text

# optional: change settings
copy .env.example .env      # Windows   (cp .env.example .env on macOS/Linux)
```

### Dependencies (`requirements.txt`)

| Package | Used for |
|---|---|
| `pypdf` | PDF text extraction |
| `langchain-core`, `langchain-text-splitters` | Documents, prompt templates, structured output, chunking |
| `langchain-ollama` | Ollama chat model + embeddings |
| `langchain-chroma`, `chromadb` | Vector store |
| `fastapi`, `uvicorn`, `jinja2`, `python-multipart` | Web application and HTML templates |
| `pydantic`, `python-dotenv` | Output schemas, configuration |

## 3. Configuration

All settings have defaults and can be overridden in `.env` (see [.env.example](.env.example)).

| Variable | Default | Meaning |
|---|---|---|
| `LLM_MODEL` | `Ollama` | LLM provider (label shown in the UI) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server |
| `OLLAMA_LLM_MODEL` | `gemma3:4b` | Chat model for question generation (e.g. `gemma3:12b`, `llama3.2:3b`) |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model |
| `LLM_TEMPERATURE` | `0.2` | Lower = more deterministic |
| `EMBED_QUERY_PREFIX` / `EMBED_DOCUMENT_PREFIX` | `search_query: ` / `search_document: ` | Task prefixes required by `nomic-embed-text`; set both empty for other embedding models |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `500` / `80` | Chunk length and overlap in characters |
| `CHROMA_DIR` | `chroma_db` | ChromaDB persistence folder |
| `DOCUMENTS_COLLECTION` / `QUESTIONS_COLLECTION` | `documents` / `questions` | ChromaDB collection names |
| `SEARCH_TOP_K` | `4` | Maximum number of search results |
| `MIN_RELEVANCE_SCORE` | `0.6` | Passages scoring below this are dropped; if none remain, the page shows "Not found in the document" |
| `NUM_QUESTIONS` | `5` | Questions generated per document |
| `SAMPLES_DIR` / `UPLOAD_DIR` | `samples` / `uploads` | PDF folders |

> If you change `OLLAMA_EMBEDDING_MODEL`, the embedding prefixes or the chunk settings,
> delete `chroma_db/` and ingest again so all vectors are created the same way.

**How the relevance threshold was chosen:** with the prefixes, the best score for 10
questions about the sample PDF was 0.64–0.76, and for 10 unrelated questions ("How do I
bake a cake?", "capital of India", …) it was 0.41–0.56. The default of 0.6 sits in that gap.
Raise it for stricter results; lower it if relevant passages are being missed.

## 4. Running

### Web application

```bash
uvicorn app.main:app --reload
```

Open <http://localhost:8000>. On the first start, the PDFs in `samples/` are ingested
automatically.

| Page | What it does |
|---|---|
| `/` Documents | Lists stored PDFs (chunks / questions), upload a new PDF, re-ingest samples |
| `/search` Semantic Search | Search box plus clickable sample queries; shows passages with page number and relevance score |
| `/questions` Questions | Generate or regenerate 5 MCQs (about 1–3 minutes with `gemma3:4b`). Click an option to check it, or use "Show all answers" to see the explanation, the quoted source sentence and the source passage |

JSON API: `GET /api/search?q=...&top_k=4`, `GET /api/questions?source=computer_basics.pdf`.
Interactive API docs: <http://localhost:8000/docs>.

### Command line

```bash
python -m app.cli ingest                     # ingest every PDF in samples/
python -m app.cli ingest path/to/file.pdf    # ingest a specific PDF
python -m app.cli search                     # run the built-in demo queries
python -m app.cli search "What is ROM?"      # run one query
python -m app.cli generate                   # generate + store 5 MCQs
python -m app.cli export                     # write sample_outputs/*.html and *.json
```

> ChromaDB's local persistent mode supports **one process at a time** for writes. If you
> run `ingest` or `generate` from the CLI while the web server is running, restart the
> server afterwards (or use the buttons in the web UI instead).

## 5. Sample outputs

Generated from `samples/computer_basics.pdf` and committed in [`sample_outputs/`](sample_outputs/).
They are static, self-contained HTML pages that open directly in a browser:

* [`sample_outputs/semantic_search.html`](sample_outputs/semantic_search.html): 5 demo queries with their matching passages and pages, plus 1 off-topic query that correctly returns "Not found in the document" ([JSON](sample_outputs/semantic_search.json))
* [`sample_outputs/questions.html`](sample_outputs/questions.html): the 5 generated questions as an interactive quiz ([JSON](sample_outputs/questions.json))
* [`sample_outputs/index.html`](sample_outputs/index.html): document overview

**Semantic search (top result per query)**

| Query | Results | Passage with the answer (page, score) |
|---|---|---|
| What is the brain of the computer? | 4 | #1: "CPU (Central Processing Unit): the 'brain' of the computer…" (p. 1, 0.76) |
| Difference between RAM and ROM | 3 | #1: "Memory: RAM is temporary (volatile) memory … ROM is permanent…" (p. 1, 0.75) |
| Which devices are used for output? | 3 | #1: "… Input Devices / Output Devices / Keyboard, Mouse / Monitor…" (p. 1, 0.64) |
| How many bytes are in a kilobyte? | 1 | #1: "Units of Memory … 1024 Bytes = 1 Kilobyte…" (p. 1, 0.76) |
| What technology did the second generation use? | 4 | #2: "… Second (1956–63): transistors…" (p. 1, 0.67) |
| How do I bake a chocolate cake? *(off-topic)* | 0 | **Not found in the document** |

**Generated question (example)**

```
Q2. What characteristic best describes RAM memory?   (page 1)
    A) start-up instructions
    B) permanent (non-volatile)
    C) solid state drive
  * D) temporary (volatile)
  Explanation: The text explicitly states that RAM is 'temporary (volatile) memory,'
  meaning it loses data when power is turned off. The other options describe ROM or storage.
  Source: "Memory: RAM is temporary (volatile) memory that loses data when power is off.
           ROM is permanent (non-volatile) memory that stores start-up instructions."
```

LLM output varies from run to run. Regenerating gives different (still verified) questions.

## 6. Project structure

```
RagBasedWebApplication/
├── app/
│   ├── config.py               settings (.env)
│   ├── llm.py                  Ollama LLM + embeddings
│   ├── document_processor.py   PDF -> cleaned pages -> chunks (+ page metadata)
│   ├── vector_store.py         ChromaDB collections: documents, questions
│   ├── ingestion.py            PDF -> ChromaDB pipeline
│   ├── semantic_search.py      query -> top-k passages
│   ├── prompts.py              LLM prompts
│   ├── question_generator.py   MCQ generation + verification + storage
│   ├── main.py                 FastAPI web app
│   ├── cli.py                  command-line interface
│   ├── templates/              Jinja2 HTML pages
│   └── static/style.css
├── samples/computer_basics.pdf
├── sample_outputs/             static HTML/JSON demo output
├── ARCHITECTURE.md
├── requirements.txt
└── .env.example
```

## 7. Customising

* **Another model:** set `OLLAMA_LLM_MODEL=gemma3:12b` in `.env`. A larger model gives better
  questions but is slower.
* **Another provider:** change the two functions in `app/llm.py`.
* **Better or different questions:** edit `app/prompts.py`.
* **Demo queries:** edit `SAMPLE_QUERIES` in `app/semantic_search.py`.

## Troubleshooting

* `Connection refused` / ingestion errors: start Ollama (`ollama serve`) and check that the
  models are pulled (`ollama list`).
* Fewer than 5 questions: some drafts failed verification on every attempt. Click
  *Regenerate*, or use a larger model.
* Tables in PDFs are extracted as plain lines (a `pypdf` limitation), so column structure is lost.
