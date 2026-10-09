# Smart RAG Lite — Backend

A FastAPI backend that answers questions **only from documents you upload**, then has
**an LLM (Groq) check the answer** against the source text before returning it.

Users sign up, log in, upload documents into a chat, and ask questions. Users, chats,
messages and document records are stored in **PostgreSQL**; document vectors are stored
in **ChromaDB** on disk.

## How a question is answered (the RAG pipeline)

Ingestion — `api/routes/uploads.py`:

```
Upload (PDF / DOCX / TXT / CSV)
  -> extract text (PyMuPDF, python-docx)                services/extraction.py
  -> split into 1000-char chunks, 150-char overlap     utils/text.py
  -> embed each chunk (sentence-transformers/all-MiniLM-L6-v2, 384 dims)          embeddings/encoder.py
  -> store vectors in ChromaDB (one collection/chat)   vectordb/chroma_client.py
  -> save document metadata in Postgres                services/documents_db.py
```

Question answering — a LangGraph state machine in `graph/workflow.py`:

```
retrieve -> grade --(relevant chunks found)--> generate -> verify -> answer
              |
              +--(none relevant, retries left)--> rewrite question -> retrieve (max 2x)
              +--(still none)--> web_search (only if TAVILY_API_KEY set) -> generate
```

| Step | File | What it does |
|---|---|---|
| retrieve | `graph/nodes/retriever.py` | Embeds the question, pulls the top 5 chunks from ChromaDB |
| grade | `graph/nodes/grader.py` | Groq LLM answers yes/no: is this chunk relevant? |
| rewrite | `graph/nodes/rewriter.py` | Groq LLM rewrites a question that found nothing |
| generate | `graph/nodes/generator.py` | Groq LLM writes the answer using **only** approved chunks |
| verify | `graph/nodes/verifier.py` | Groq LLM fact-checks the answer against the chunks → `grounded: true/false` |

If nothing relevant is found, the answer comes from the model's general knowledge and is
labelled `source: "ai_knowledge"` so it's never mistaken for a document answer.

All LLM calls go through one function: `complete()` in `services/llm.py`.

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/auth/signup` | Create account → JWT |
| POST | `/api/auth/login` | Log in → JWT |
| GET | `/api/auth/me` | Current user |
| GET | `/api/sessions` | List my chats |
| DELETE | `/api/sessions/{id}` | Delete chat + its files and vectors |
| GET | `/api/sessions/{id}/documents` | Documents in a chat |
| DELETE | `/api/sessions/{id}/documents/{doc_id}` | Remove one document |
| GET | `/api/sessions/{id}/messages` | Chat history |
| POST | `/api/uploads` | Upload files (creates a chat if `session_id` is omitted) |
| POST | `/api/chat` | Ask a question |
| GET | `/api/health` | Health check (used by Railway) |

Interactive docs: `http://localhost:8000/docs`

## Run locally

1. Install Python 3.13 and PostgreSQL, and create a database: `createdb smart_rag`
2. Install dependencies:
   ```
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. `cp .env.example .env` and fill in `GROQ_API_KEY` (from console.groq.com) and `DATABASE_URL`.
4. Start the server:
   ```
   uvicorn main:app --reload
   ```
   Tables are created automatically on first start. The first upload downloads the
   embedding model (~90 MB), so it takes longer than later uploads.

## Deploy on Railway

See the top-level `README.md` of this bundle for the full step-by-step guide.

## Notes

- **Groq rate limits:** one question makes about 7 LLM calls (5 grades + generate + verify), and up to ~18
  when the rewrite loop runs. `GROQ_MODEL` / `GROQ_FALLBACK_MODELS` switch models without code changes.
- **Pinned versions:** `requirements.txt` pins exact versions so a rebuild can't silently pull a
  breaking release (as happened with SQLAlchemy 2.1).
- **Database compatibility:** the tables are the same as the full Smart RAG project, so it can
  point at the same database and existing accounts keep working.
