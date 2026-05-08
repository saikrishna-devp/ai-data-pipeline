# api/main.py
# FastAPI REST API for the RAG pipeline

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from loguru import logger
from config.settings import get_settings
from pipeline.rag_pipeline import ask
from vectorstore.chroma_store import get_collection_stats, store_chunks
from pipeline.chunker import chunk_documents, generate_embeddings
from ingestion.wikipedia_ingestion import run_ingestion

settings = get_settings()

# ── App Setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI-Powered Data Pipeline API",
    description="RAG pipeline combining Data Engineering with AI",
    version="1.0.0"
)

# ── Request/Response Models ───────────────────────────────────────────────────
class QuestionRequest(BaseModel):
    question: str
    top_k: int = 3

class QuestionResponse(BaseModel):
    question: str
    answer: str
    sources: list
    chunks_used: int

class IngestResponse(BaseModel):
    status: str
    documents_ingested: int
    chunks_stored: int

class StatsResponse(BaseModel):
    total_chunks: int
    collection_name: str
    status: str

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model": settings.ollama_model}

@app.get("/stats", response_model=StatsResponse)
def get_stats():
    """Get vector store statistics."""
    count = get_collection_stats()
    return StatsResponse(
        total_chunks=count,
        collection_name=settings.chroma_collection,
        status="ready" if count > 0 else "empty"
    )

@app.post("/ask", response_model=QuestionResponse)
def ask_question(request: QuestionRequest):
    """Ask a question and get AI-generated answer."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        logger.info(f"API question: {request.question}")
        result = ask(request.question)
        return QuestionResponse(**result)
    except Exception as e:
        logger.error(f"API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest", response_model=IngestResponse)
def ingest_data():
    """Trigger data ingestion pipeline."""
    try:
        logger.info("API: starting ingestion...")
        docs   = run_ingestion()
        chunks = chunk_documents(docs)
        chunks = generate_embeddings(chunks)
        store_chunks(chunks)
        return IngestResponse(
            status="success",
            documents_ingested=len(docs),
            chunks_stored=len(chunks)
        )
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)