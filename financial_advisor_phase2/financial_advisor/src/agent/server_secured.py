"""FastAPI server for the SECURED Financial Advisor Agent (Phase 2 — with Guard Node).

Runs on port 8001. The baseline agent continues to run on port 8000.
Both share the same Milvus collection (financial_docs).
"""

import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.agent.graph_secured import secured_agent
from src.rag.retriever import get_collection_stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Financial Advisor Agent — Secured (Guard Node)",
    version="2.0.0",
)


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    cited_documents: list[dict]
    injection_detected: bool
    guard_log: list[str]
    error: str | None = None


class IndexDocumentRequest(BaseModel):
    text: str
    doc_title: str
    doc_type: str


class IndexDocumentResponse(BaseModel):
    message: str
    doc_title: str
    doc_type: str
    num_chunks: int


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "financial-advisor-secured"}


@app.get("/stats")
def collection_stats():
    try:
        return get_collection_stats()
    except Exception as e:
        return {"error": str(e)}


@app.post("/query", response_model=QueryResponse)
def query_agent(request: QueryRequest):
    """Send a query to the secured financial advisor agent."""
    try:
        initial_state = {
            "user_query": request.query,
            "retrieved_chunks": [],
            "raw_response": "",
            "final_answer": "",
            "cited_documents": [],
            "error": None,
            # Guard fields — initialised as clean
            "injection_detected": False,
            "flagged_chunks": [],
            "guard_log": [],
        }

        result = secured_agent.invoke(initial_state)

        return QueryResponse(
            answer=result.get("final_answer", "No response generated."),
            cited_documents=result.get("cited_documents", []),
            injection_detected=result.get("injection_detected", False),
            guard_log=result.get("guard_log", []),
            error=result.get("error"),
        )
    except Exception as e:
        logger.error(f"Secured agent query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index-document", response_model=IndexDocumentResponse)
def index_document(request: IndexDocumentRequest):
    """Index a financial document (shared with baseline — same Milvus collection)."""
    try:
        from src.rag.indexer import chunk_text
        from src.rag.retriever import insert_chunks
        import uuid

        doc_id = str(uuid.uuid4())
        raw_chunks = chunk_text(request.text, chunk_size=256, overlap=30)

        chunks = [
            {
                "chunk_id": f"{doc_id}_{i}",
                "text": chunk,
                "doc_title": request.doc_title,
                "doc_type": request.doc_type,
                "doc_id": doc_id,
            }
            for i, chunk in enumerate(raw_chunks)
        ]

        insert_chunks(chunks)

        return IndexDocumentResponse(
            message="Document indexed successfully",
            doc_title=request.doc_title,
            doc_type=request.doc_type,
            num_chunks=len(chunks),
        )
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/reset-index")
def reset_index():
    """Drop and recreate the collection."""
    try:
        from src.rag.retriever import drop_collection, create_collection
        drop_collection()
        create_collection()
        return {"message": "Index reset successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
