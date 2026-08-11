# Zepto Support Assistant — Module 3

## Submission checklist

This folder contains the complete graded baseline for Module 3:

- 8 exact Zepto corpus documents in `docs/`
- Local `all-MiniLM-L6-v2` embeddings
- ChromaDB persistent vector collection
- LangGraph `StateGraph` with a `TypedDict` state and 3 nodes
- Keyword-based deterministic intent classifier
- Top-3 cosine retrieval
- Required offline mock mode (`MOCK_LLM=1`)
- Structured Pydantic output: `answer`, `sources`, `confidence`
- FastAPI `POST /ask`
- Dockerfile that runs the FastAPI service on port 7860
- Example API calls and architecture explanation

## Important grading setting

The graded baseline is the default:

```text
MOCK_LLM=1
```

Do **not** change this for the normal submission/demo. No API key, payment, network LLM call, or live cloud service is required for the graded path.

The optional real-LLM extension is activated only when:

```text
MOCK_LLM=0
```

and a valid `GROQ_API_KEY` is supplied. It is not required for the graded baseline.

---

## 1. Project structure

```text
support_assistant/
├── app.py
├── main.py
├── requirements.txt
├── Dockerfile
├── README.md
├── docs/
│   ├── doc_01.txt
│   ├── doc_02.txt
│   ├── doc_03.txt
│   ├── doc_04.txt
│   ├── doc_05.txt
│   ├── doc_06.txt
│   ├── doc_07.txt
│   └── doc_08.txt
└── chroma_db/                 # generated locally on first run
```

---

## 2. Architecture

```text
User query
    |
    v
FastAPI POST /ask
    |
    v
LangGraph StateGraph
    |
    v
classify_intent
    |
    +---- policy_question ----> retrieve_and_answer
    |                              |
    |                              +--> all-MiniLM-L6-v2 embedding
    |                              +--> ChromaDB cosine retrieval
    |                              +--> top 3 chunks
    |                              +--> MOCK_LLM=1 canned response
    |
    +---- general_question ----> direct_answer
                                   |
                                   +--> MOCK_LLM=1 fixed response
    |
    v
Pydantic validation
    |
    v
JSON: answer / sources / confidence
```

### Pipeline stages

1. **Ingestion** — `main.py` reads all eight files from `docs/`.
2. **Embedding** — each document is embedded locally with `all-MiniLM-L6-v2`.
3. **Indexing** — embeddings and document text are stored in the ChromaDB collection `zepto_support_corpus`.
4. **Retrieval** — policy questions embed the incoming query and retrieve the top three chunks using ChromaDB cosine distance.
5. **Generation** — in the required baseline, generation is deterministic and does not call an LLM. The answer is formed from the top retrieved chunk. General questions use a fixed canned response.
6. **Validation** — the final result is passed through the Pydantic `SupportResponse` schema containing `answer`, `sources`, and `confidence`.
7. **API** — FastAPI exposes `POST /ask`, and Uvicorn serves it locally.

### MOCK_LLM behavior

The graph's routing itself is independent of `MOCK_LLM`.

- `MOCK_LLM=1` (default): no LLM call. Classification uses the required keyword heuristic, retrieval still runs for policy questions, and both generation branches are deterministic.
- `MOCK_LLM=0`: optional real-LLM extension. The retrieval branch sends the retrieved context plus the structured prompt to Groq; the direct branch prompts the model without retrieval. The Pydantic schema is validated and the real-LLM path retries up to two additional times after validation failure.

The structured prompt contains all five requested skeleton components: **role, context, task, format, length**, plus an explicit negative constraint and a few-shot example.

---

## 3. Install and run locally

Open a terminal inside this `support_assistant` folder.

### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
$env:MOCK_LLM="1"
uvicorn app:app --host 127.0.0.1 --port 7860
```

### Windows CMD

```cmd
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
set MOCK_LLM=1
uvicorn app:app --host 127.0.0.1 --port 7860
```

The first startup downloads/caches the open-source embedding model if it is not already cached. No API key is needed.

Open:

```text
http://127.0.0.1:7860/docs
```

FastAPI Swagger UI can be used to call `POST /ask`.

---

## 4. Required example calls

### Example A — policy question routed to retrieval

Request:

```json
{
  "query": "How long do I have to report a damaged grocery item?"
}
```

The query contains `damaged` but the required classifier keyword list is intentionally based on the assignment's examples. To make a guaranteed policy route, use this example:

```json
{
  "query": "What is the return policy for damaged grocery items?"
}
```

Expected structure:

```json
{
  "answer": "Based on the retrieved context: doc_02 — Returns & Refunds: ...",
  "sources": [
    "doc_02",
    "...",
    "..."
  ],
  "confidence": 1.0
}
```

The exact second/third retrieved IDs can vary with embedding-library versions, but `doc_02` should be the relevant top result for the return-policy question.

### Example B — unrelated/general question

Request:

```json
{
  "query": "What is the capital of France?"
}
```

Expected:

```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

### PowerShell curl examples

```powershell
curl.exe -X POST "http://127.0.0.1:7860/ask" `
  -H "Content-Type: application/json" `
  -d '{\"query\":\"What is the return policy for damaged grocery items?\"}'
```

```powershell
curl.exe -X POST "http://127.0.0.1:7860/ask" `
  -H "Content-Type: application/json" `
  -d '{\"query\":\"What is the capital of France?\"}'
```

Save the two JSON responses in your submission/README evidence if your evaluator expects transcripts.

---

## 5. Docker

Build:

```powershell
docker build -t zepto-support-assistant .
```

Run:

```powershell
docker run --rm -p 7860:7860 -e MOCK_LLM=1 zepto-support-assistant
```

Then open:

```text
http://127.0.0.1:7860/docs
```

The container starts with:

```text
uvicorn app:app --host 0.0.0.0 --port 7860
```

The model is downloaded/cached during the first application startup if it is not already available.

---

## 6. Why this satisfies the acceptance criteria

- **8 documents**: all eight required corpus files are present under `docs/`.
- **Embeddings + ChromaDB**: `SentenceTransformer(all-MiniLM-L6-v2)` creates local embeddings and `PersistentClient` stores/querys them.
- **Structured prompt**: `PROMPT_TEMPLATE` explicitly contains role, context, task, format, length, negative constraint, and few-shot example.
- **LangGraph**: `StateGraph` uses `GraphState(TypedDict)` and has three nodes: `classify_intent`, `retrieve_and_answer`, `direct_answer`.
- **Conditional routing**: `classify_intent` routes policy questions to retrieval and general questions to direct answer.
- **Mock baseline**: `MOCK_LLM` defaults to `1`; no LLM API call is made.
- **Real retrieval in both modes**: retrieval uses the embedding model and ChromaDB independently of the LLM mode.
- **Top 3**: ChromaDB is queried with `n_results=3`.
- **Pydantic output**: `SupportResponse` validates `answer`, `sources`, and `confidence`.
- **Mock deterministic values**: policy source IDs are generated from retrieved chunks, general-question sources are empty, and confidence is `1.0`.
- **FastAPI**: `POST /ask` accepts `{"query": "..."}` and returns the validated response.
- **Docker**: `Dockerfile` builds the app and serves port `7860`.
- **README architecture**: this document describes ingestion → embedding → retrieval → generation and the MOCK_LLM branches.

---

## 7. Submission note

Submit the `support_assistant` folder inside the single project repository as:

```text
/support_assistant
```

Do not commit your `.venv/`, Python cache directories, API keys, or other secrets.

If the evaluator runs the required baseline, leave:

```text
MOCK_LLM=1
```

as the default.
