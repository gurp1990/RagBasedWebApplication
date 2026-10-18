# RAG PDF Assistant — Semantic Search & Question Generation (Gemini)

A **Retrieval-Augmented Generation** web application written in Python, powered by the
**Google Gemini API**.

* **Document processing:** loads a PDF, extracts the text page by page, splits it into chunks
  with page-number metadata and stores them in **ChromaDB**.
* **Semantic search:** finds the passages most relevant to a question in your own words and
  shows each passage with its **source page**.
* **Question generation:** creates **5 multiple-choice questions** (4 options, the correct
  answer and an explanation) from the document. Each question is fact-checked against its
  source passage and then stored in ChromaDB.
* **Web UI:** HTML pages for documents, semantic search and an interactive quiz, plus a
  JSON API.

Stack: **Python · LangChain · ChromaDB · Google Gemini (`gemini-3.5-flash` LLM, `gemini-embedding-2` embeddings) · FastAPI**

> **Branch note:** this is the `gemini_online_implementation` branch. The `main` branch runs
> fully offline with Ollama instead.

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
| Google Gemini API key | Free to create at <https://aistudio.google.com/apikey> |
| Internet connection | Embeddings and question generation call the Gemini API |

No GPU and no local model download are needed: the models run on Google's servers.

**Privacy:** the text of every PDF you ingest, and every search query, is sent to the Gemini
API. Don't use this branch for confidential documents. Use the Ollama version on `main`
instead.

### Get a Gemini API key

1. Open <https://aistudio.google.com/apikey> and sign in with a Google account.
2. Click **Create API key** and copy it.
3. Keep it secret. It goes only into the `.env` file (see Setup), which Git ignores. Never
   put it in code, in the README or in a commit.

The app uses these two models:

| Model | Used for |
|---|---|
| `gemini-3.5-flash` | LLM that writes and checks the questions |
| `gemini-embedding-2` (768 dimensions) | Embeddings for chunks and search queries |

## 2. Setup

### Windows (10 / 11, PowerShell)

```powershell
git clone https://github.com/gurp1990/RagBasedWebApplication.git
cd RagBasedWebApplication
git switch gemini_online_implementation

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

copy .env.example .env
notepad .env          # set GOOGLE_API_KEY=<your key>, save and close
```

### Linux (Ubuntu, Debian, Fedora, …)

```bash
# Ubuntu/Debian: install Python and venv if needed
sudo apt install python3 python3-venv python3-pip

git clone https://github.com/gurp1990/RagBasedWebApplication.git
cd RagBasedWebApplication
git switch gemini_online_implementation

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
nano .env             # set GOOGLE_API_KEY=<your key>, save with Ctrl+O, exit with Ctrl+X
```

### Check that the key works

The command lists the Gemini models your key can use. An error message means the key is
wrong or not active yet.

```powershell
# Windows (PowerShell)
curl.exe -H "x-goog-api-key: <your key>" https://generativelanguage.googleapis.com/v1beta/models
```

```bash
# Linux
curl -H "x-goog-api-key: <your key>" https://generativelanguage.googleapis.com/v1beta/models
```

### Dependencies (`requirements.txt`)

| Package | Used for |
|---|---|
| `pypdf` | PDF text extraction |
| `langchain-core`, `langchain-text-splitters` | Documents, prompt templates, structured output, chunking |
| `langchain-google-genai` | Gemini chat model + Gemini embeddings |
| `langchain-chroma`, `chromadb` | Vector store |
| `fastapi`, `uvicorn`, `jinja2`, `python-multipart` | Web application and HTML templates |
| `pydantic`, `python-dotenv` | Output schemas, configuration |

## 3. Configuration

All settings have defaults and can be overridden in `.env` (see [.env.example](.env.example)).
Only `GOOGLE_API_KEY` is required.

| Variable | Default | Meaning |
|---|---|---|
| `GOOGLE_API_KEY` | *(none, required)* | Your Gemini API key |
| `LLM_MODEL` | `Gemini` | LLM provider (label shown in the UI) |
| `GEMINI_LLM_MODEL` | `gemini-3.5-flash` | Chat model for question generation (e.g. `gemini-3.5-flash-lite` for faster, cheaper runs) |
| `GEMINI_EMBEDDING_MODEL` | `gemini-embedding-2` | Embedding model |
| `EMBEDDING_DIMENSIONS` | `768` | Vector size: `768`, `1536` or `3072` |
| `LLM_TEMPERATURE` | `0.2` | Lower = more deterministic |
| `LLM_MAX_RETRIES` | `6` | Retries with back-off when the API rate limit is hit |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `500` / `80` | Chunk length and overlap in characters |
| `CHROMA_DIR` | `chroma_db_gemini` | ChromaDB persistence folder (separate from the Ollama database on `main`) |
| `DOCUMENTS_COLLECTION` / `QUESTIONS_COLLECTION` | `documents` / `questions` | ChromaDB collection names |
| `SEARCH_TOP_K` | `4` | Maximum number of search results |
| `MIN_RELEVANCE_SCORE` | `0.59` | Passages scoring below this are dropped; if none remain, the page shows "Not found in the document" |
| `NUM_QUESTIONS` | `5` | Questions generated per document |
| `SAMPLES_DIR` / `UPLOAD_DIR` | `samples` / `uploads` | PDF folders |

> If you change `GEMINI_EMBEDDING_MODEL`, `EMBEDDING_DIMENSIONS` or the chunk settings,
> delete `chroma_db_gemini/` and ingest again so all vectors are created the same way.

Gemini embeds chunks with task type `RETRIEVAL_DOCUMENT` and queries with `RETRIEVAL_QUERY`,
so search queries and passages are compared the way the model was trained for.

**How the relevance threshold was chosen:** I compared the embedding models on 10 questions
about the sample PDF and 10 unrelated questions ("How do I bake a cake?", "capital of India",
…). With `gemini-embedding-2` at 768 dimensions, the best score was 0.61–0.77 for related
questions and 0.48–0.57 for unrelated ones. That was the clearest gap of the models tested,
and the default of 0.59 sits inside it. Raise it for stricter results; lower it if relevant
passages are being missed.

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
| `/questions` Questions | Generate or regenerate 5 MCQs (about 1–2 minutes with `gemini-3.5-flash`). Click an option to check it, or use "Show all answers" to see the explanation, the quoted source sentence and the source passage |

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

Generated with Gemini from `samples/computer_basics.pdf` and committed in [`sample_outputs/`](sample_outputs/).
They are static, self-contained HTML pages that open directly in a browser:

* [`sample_outputs/semantic_search.html`](sample_outputs/semantic_search.html): 5 demo queries with their matching passages and pages, plus 1 off-topic query that correctly returns "Not found in the document" ([JSON](sample_outputs/semantic_search.json))
* [`sample_outputs/questions.html`](sample_outputs/questions.html): the 5 generated questions as an interactive quiz ([JSON](sample_outputs/questions.json))
* [`sample_outputs/index.html`](sample_outputs/index.html): document overview

**Semantic search (top result per query)**

| Query | Results | Passage with the answer (page, score) |
|---|---|---|
| What is the brain of the computer? | 4 | #1: "CPU (Central Processing Unit): the 'brain' of the computer…" (p. 1, 0.77) |
| Difference between RAM and ROM | 4 | #1: "Memory: RAM is temporary (volatile) memory … ROM is permanent…" (p. 1, 0.74) |
| Which devices are used for output? | 4 | #1: "… Input Devices / Output Devices / Keyboard, Mouse / Monitor…" (p. 1, 0.70) |
| How many bytes are in a kilobyte? | 4 | #1: "Units of Memory … 1024 Bytes = 1 Kilobyte…" (p. 1, 0.69) |
| What technology did the second generation use? | 4 | #1: "… Second (1956–63): transistors…" (p. 1, 0.68) |
| How do I bake a chocolate cake? *(off-topic)* | 0 | **Not found in the document** |

**Generated question (example)**

```
Q2. Which type of memory is permanent and stores start-up instructions?   (page 1)
  * A) ROM
    B) RAM
    C) HDD
    D) Motherboard
  Explanation: ROM is permanent (non-volatile) memory that stores start-up instructions,
  unlike RAM which is temporary.
  Source: "ROM is permanent (non-volatile) memory that stores start-up instructions."
```

LLM output varies from run to run. Regenerating gives different (still verified) questions.

## 6. Project structure

```
RagBasedWebApplication/
├── app/
│   ├── config.py               settings (.env)
│   ├── llm.py                  Gemini LLM + embeddings
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
├── docs/workflow.html          diagrams (RAG, workflows, ER diagram)
├── samples/computer_basics.pdf
├── sample_outputs/             static HTML/JSON demo output
├── ARCHITECTURE.md
├── requirements.txt
└── .env.example                copy to .env and add GOOGLE_API_KEY
```

## 7. Customising

* **Another model:** set `GEMINI_LLM_MODEL` in `.env`, for example `gemini-3.5-flash-lite`
  for faster and cheaper runs. To see which models your key can use, run the curl command
  under "Check that the key works".
* **Another provider:** change the two functions in `app/llm.py`.
* **Better or different questions:** edit `app/prompts.py`.
* **Demo queries:** edit `SAMPLE_QUERIES` in `app/semantic_search.py`.

## Troubleshooting

| Problem | Fix |
|---|---|
| `GOOGLE_API_KEY is not set` | Create `.env` from `.env.example` and add your key |
| `400 API key not valid` / `403 PERMISSION_DENIED` | Check the key in `.env`, with no quotes or spaces, or create a new key in AI Studio |
| `404 NOT_FOUND … model … is no longer available` | Google retired that model. Set `GEMINI_LLM_MODEL` to a current one from the model list |
| `429 RESOURCE_EXHAUSTED` | The free-tier rate limit was hit. The app retries automatically (`LLM_MAX_RETRIES`). If it still fails, wait a minute and click *Regenerate* |
| Search finds nothing after changing the embedding model | Delete `chroma_db_gemini/` and ingest again |
| Fewer than 5 questions | Some drafts failed verification on every attempt. Click *Regenerate* |
| Connection errors | Check your internet connection, and any proxy or firewall in front of `generativelanguage.googleapis.com` |

Tables in PDFs are extracted as plain lines (a `pypdf` limitation), so column structure is lost.
