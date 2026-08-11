from fastapi import FastAPI
from main import QueryRequest, SupportResponse, answer_query

app = FastAPI(
    title="Zepto Support Assistant",
    version="1.0.0",
    description="Offline-mock RAG support assistant using LangGraph, ChromaDB and FastAPI.",
)


@app.get("/")
def root():
    return {"service": "Zepto Support Assistant", "status": "ok"}


@app.post("/ask", response_model=SupportResponse)
def ask(request: QueryRequest):
    return answer_query(request.query)
