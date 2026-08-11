import os
from pathlib import Path
from typing import TypedDict, Literal, List, Dict, Any

import chromadb
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END


BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "zepto_support_corpus"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MOCK_LLM = os.getenv("MOCK_LLM", "1")  # 1 = required graded baseline; 0 = optional real LLM


class SupportResponse(BaseModel):
    answer: str
    sources: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class QueryRequest(BaseModel):
    query: str


class GraphState(TypedDict, total=False):
    query: str
    intent: Literal["policy_question", "general_question"]
    retrieved: List[Dict[str, Any]]
    answer: str
    sources: List[str]
    confidence: float
    response: Dict[str, Any]


PROMPT_TEMPLATE = """
ROLE:
You are Zepto Support Assistant. Answer customer questions using only the supplied Zepto policy context.

CONTEXT:
{context}

TASK:
Answer the user's question accurately and concisely using the context above.

FORMAT:
Return JSON with exactly these fields:
- answer: string
- sources: list of chunk/document IDs used
- confidence: float from 0 to 1

LENGTH:
Keep the answer concise, normally 1-4 sentences.

NEGATIVE CONSTRAINT:
Do not answer using information not present in the provided context. If the context does not support the answer, clearly say that the provided context does not contain enough information.

FEW-SHOT EXAMPLE:
User: "How long do I have to report a damaged grocery item?"
Context: "doc_02 — Returns & Refunds: Grocery and perishable items may be reported for a return within 24 hours of delivery if damaged, spoiled, or incorrect."
Output:
{"answer":"Damaged, spoiled, or incorrect grocery/perishable items may be reported within 24 hours of delivery.","sources":["doc_02"],"confidence":1.0}

USER QUERY:
{query}
""".strip()


class ZeptoSupport:
    def __init__(self):
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.chroma.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._index_documents()

    def _index_documents(self):
        files = sorted(DOCS_DIR.glob("doc_*.txt"))
        if len(files) != 8:
            raise RuntimeError("Expected exactly 8 corpus documents in docs/.")

        existing = self.collection.get(include=[])
        existing_ids = set(existing.get("ids", []))

        new_files = [p for p in files if p.stem not in existing_ids]
        if not new_files:
            return

        texts = [p.read_text(encoding="utf-8").strip() for p in new_files]
        ids = [p.stem for p in new_files]
        embeddings = self.embedder.encode(
            texts, normalize_embeddings=True
        ).tolist()

        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=[{"document_id": x} for x in ids],
        )

    def classify_intent(self, state: GraphState) -> GraphState:
        q = state["query"].lower()
        keywords = [
            "delivery", "return", "refund", "membership",
            "tracking", "cancel", "gift card", "support hours"
        ]
        intent = "policy_question" if any(k in q for k in keywords) else "general_question"
        return {"intent": intent}

    def retrieve_and_answer(self, state: GraphState) -> GraphState:
        query = state["query"]
        q_embedding = self.embedder.encode(
            [query], normalize_embeddings=True
        ).tolist()

        result = self.collection.query(
            query_embeddings=q_embedding,
            n_results=3,
            include=["documents", "metadatas", "distances"],
        )

        retrieved = []
        for i in range(len(result["ids"][0])):
            doc_id = result["metadatas"][0][i]["document_id"]
            retrieved.append({
                "id": doc_id,
                "text": result["documents"][0][i],
                "distance": float(result["distances"][0][i]),
                "similarity": float(1.0 - result["distances"][0][i]),
            })

        if os.getenv("MOCK_LLM", "1") != "0":
            top = retrieved[0] if retrieved else {"id": "", "text": ""}
            snippet = top["text"][:200]
            answer = f"Based on the retrieved context: {snippet}"
            sources = [x["id"] for x in retrieved]
            confidence = 1.0
            return {
                "retrieved": retrieved,
                "answer": answer,
                "sources": sources,
                "confidence": confidence,
            }

        # Optional real-LLM path.
        answer, sources, confidence = self._real_llm_answer(query, retrieved)
        return {
            "retrieved": retrieved,
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
        }

    def direct_answer(self, state: GraphState) -> GraphState:
        if os.getenv("MOCK_LLM", "1") != "0":
            return {
                "answer": "I can only answer questions about Zepto policies right now.",
                "sources": [],
                "confidence": 1.0,
            }

        # Optional real-LLM path without retrieval.
        answer, sources, confidence = self._real_llm_answer(state["query"], [])
        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
        }

    def _real_llm_answer(self, query: str, retrieved: List[Dict[str, Any]]):
        try:
            from langchain_groq import ChatGroq
        except ImportError as exc:
            raise RuntimeError(
                "Optional real-LLM mode requires langchain-groq. "
                "Install requirements and set GROQ_API_KEY."
            ) from exc

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("Set GROQ_API_KEY when MOCK_LLM=0.")

        context = "\n\n".join(
            f"[{x['id']}] {x['text']}" for x in retrieved
        ) or "(No retrieved context; answer only if the model can do so from supplied information.)"

        prompt = PROMPT_TEMPLATE.format(
            context=context,
            query=query,
        )

        llm = ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            temperature=0,
            api_key=api_key,
        )

        last_error = None
        for attempt in range(3):  # initial attempt + 2 retries
            try:
                raw = llm.invoke(prompt).content
                import json
                parsed = json.loads(raw)
                validated = SupportResponse.model_validate(parsed)
                return (
                    validated.answer,
                    validated.sources,
                    validated.confidence,
                )
            except Exception as exc:
                last_error = exc
                prompt += (
                    "\nCORRECTION: Your previous output failed validation. "
                    "Return ONLY valid JSON with answer, sources, confidence."
                )

        raise RuntimeError(f"LLM response failed schema validation: {last_error}")

    def _route(self, state: GraphState) -> str:
        return state["intent"]

    def build_graph(self):
        workflow = StateGraph(GraphState)
        workflow.add_node("classify_intent", self.classify_intent)
        workflow.add_node("retrieve_and_answer", self.retrieve_and_answer)
        workflow.add_node("direct_answer", self.direct_answer)

        workflow.add_edge(START, "classify_intent")
        workflow.add_conditional_edges(
            "classify_intent",
            self._route,
            {
                "policy_question": "retrieve_and_answer",
                "general_question": "direct_answer",
            },
        )
        workflow.add_edge("retrieve_and_answer", END)
        workflow.add_edge("direct_answer", END)
        return workflow.compile()

    def ask(self, query: str) -> SupportResponse:
        graph = self.build_graph()
        state = graph.invoke({"query": query})
        result = SupportResponse(
            answer=state["answer"],
            sources=state.get("sources", []),
            confidence=state.get("confidence", 1.0),
        )
        return result


_service = None


def get_service():
    global _service
    if _service is None:
        _service = ZeptoSupport()
    return _service


def answer_query(query: str) -> dict:
    return get_service().ask(query).model_dump()
