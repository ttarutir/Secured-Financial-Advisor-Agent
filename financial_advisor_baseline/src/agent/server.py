"""FastAPI server for the baseline Financial Advisor Agent."""

import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.agent.graph import agent
from src.rag.retriever import get_collection_stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Financial Advisor Agent (Baseline — No Guard)", version="1.0.0")


class QueryRequest(BaseModel):
    query: str  # Free-form: includes client context + question


class QueryResponse(BaseModel):
    answer: str
    cited_documents: list[dict]
    error: str | None = None


class IndexDocumentRequest(BaseModel):
    text: str
    doc_title: str
    doc_type: str   # client_profile | firm_policy | market_data | (malicious)


class IndexDocumentResponse(BaseModel):
    message: str
    doc_title: str
    doc_type: str
    num_chunks: int


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "financial-advisor-baseline"}


@app.get("/stats")
def collection_stats():
    try:
        return get_collection_stats()
    except Exception as e:
        return {"error": str(e)}


@app.post("/query", response_model=QueryResponse)
def query_agent(request: QueryRequest):
    """Send a query to the financial advisor agent."""
    try:
        initial_state = {
            "user_query": request.query,
            "retrieved_chunks": [],
            "raw_response": "",
            "final_answer": "",
            "cited_documents": [],
            "error": None,
        }

        result = agent.invoke(initial_state)

        return QueryResponse(
            answer=result.get("final_answer", "No response generated."),
            cited_documents=result.get("cited_documents", []),
            error=result.get("error"),
        )
    except Exception as e:
        logger.error(f"Agent query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index-document", response_model=IndexDocumentResponse)
def index_document(request: IndexDocumentRequest):
    """Index a financial document (policy, profile, or test injection doc)."""
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
    """Drop and recreate the collection (useful between test runs)."""
    try:
        from src.rag.retriever import drop_collection, create_collection
        drop_collection()
        create_collection()
        return {"message": "Index reset successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
