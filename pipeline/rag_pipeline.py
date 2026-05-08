# pipeline/rag_pipeline.py
# LangChain RAG Pipeline
# Retrieves relevant chunks from ChromaDB
# and generates answers using Ollama

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from loguru import logger
from config.settings import get_settings
from vectorstore.chroma_store import search, get_collection_stats
from pipeline.chunker import embedding_model

settings = get_settings()

# ── LLM Setup ─────────────────────────────────────────────────────────────────
llm = OllamaLLM(
    model=settings.ollama_model,
    base_url=settings.ollama_base_url,
    temperature=0.1
)

# ── Prompt Template ───────────────────────────────────────────────────────────
PROMPT_TEMPLATE = """You are an expert data engineering assistant.
Use the following context to answer the question accurately and concisely.
If the answer is not in the context, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer:"""

prompt = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

# ── RAG Pipeline ──────────────────────────────────────────────────────────────
def ask(question: str) -> dict:
    """
    Full RAG pipeline:
    1. Embed the question
    2. Retrieve relevant chunks from ChromaDB
    3. Build prompt with context
    4. Generate answer with Ollama
    5. Return answer with sources
    """
    logger.info(f"Question: {question}")

    # Step 1 - Embed question
    query_embedding = embedding_model.encode([question])[0].tolist()

    # Step 2 - Retrieve relevant chunks
    results = search(question, query_embedding, top_k=settings.retriever_top_k)

    if not results:
        return {
            "question": question,
            "answer":   "No relevant documents found. Please ingest data first.",
            "sources":  []
        }

    # Step 3 - Build context from chunks
    context = "\n\n".join([r["content"] for r in results])

    # Step 4 - Generate answer
    logger.info("Generating answer with Ollama...")
    final_prompt = prompt.format(context=context, question=question)
    answer = llm.invoke(final_prompt)

    # Step 5 - Build sources list
    sources = []
    for r in results:
        source = {
            "title":    r["metadata"]["title"],
            "source":   r["metadata"]["source"],
            "url":      r["metadata"].get("url", ""),
            "distance": round(r["distance"], 4)
        }
        if source not in sources:
            sources.append(source)

    logger.success(f"Answer generated ✅")

    return {
        "question": question,
        "answer":   answer,
        "sources":  sources,
        "chunks_used": len(results)
    }

# ── Test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Check data is loaded
    count = get_collection_stats()
    if count == 0:
        logger.warning("No data in ChromaDB! Running ingestion first...")
        from ingestion.wikipedia_ingestion import run_ingestion
        from pipeline.chunker import chunk_documents, generate_embeddings
        from vectorstore.chroma_store import store_chunks
        docs   = run_ingestion()
        chunks = chunk_documents(docs)
        chunks = generate_embeddings(chunks)
        store_chunks(chunks)

    # Test questions
    questions = [
        "What is Apache Kafka?",
        "How does RAG work?",
        "What is Apache Iceberg used for?"
    ]

    for question in questions:
        print("\n" + "="*60)
        result = ask(question)
        print(f"Q: {result['question']}")
        print(f"\nA: {result['answer']}")
        print(f"\nSources:")
        for s in result["sources"]:
            print(f"  - {s['title']} ({s['source']})")