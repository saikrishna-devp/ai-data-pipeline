# vectorstore/chroma_store.py
# Stores and retrieves document chunks from ChromaDB

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from loguru import logger
from config.settings import get_settings

settings = get_settings()

# ── ChromaDB Client ───────────────────────────────────────────────────────────
client = chromadb.HttpClient(
    host=settings.chroma_host,
    port=settings.chroma_port
)

def get_or_create_collection():
    """Get existing collection or create new one."""
    try:
        collection = client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"}
        )
        logger.success(f"Collection '{settings.chroma_collection}' ready ✅")
        return collection
    except Exception as e:
        logger.error(f"Failed to get collection: {e}")
        return None

# ── Store Chunks ──────────────────────────────────────────────────────────────
def store_chunks(chunks: list):
    """Store document chunks with embeddings in ChromaDB."""
    collection = get_or_create_collection()
    if not collection:
        return

    ids         = [chunk["id"] for chunk in chunks]
    embeddings  = [chunk["embedding"] for chunk in chunks]
    documents   = [chunk["content"] for chunk in chunks]
    metadatas   = [
        {
            "title":   chunk["title"],
            "source":  chunk["source"],
            "url":     chunk.get("url", ""),
            "doc_id":  chunk["doc_id"]
        }
        for chunk in chunks
    ]

    # Add in batches of 100
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch_ids        = ids[i:i+batch_size]
        batch_embeddings = embeddings[i:i+batch_size]
        batch_documents  = documents[i:i+batch_size]
        batch_metadatas  = metadatas[i:i+batch_size]

        collection.upsert(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_documents,
            metadatas=batch_metadatas
        )

    logger.success(f"Stored {len(chunks)} chunks in ChromaDB ✅")
    logger.info(f"Total documents in collection: {collection.count()}")

# ── Search ────────────────────────────────────────────────────────────────────
def search(query: str, query_embedding: list, top_k: int = None) -> list:
    """Search ChromaDB for most similar chunks."""
    collection = get_or_create_collection()
    if not collection:
        return []

    top_k = top_k or settings.retriever_top_k
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    for i in range(len(results["documents"][0])):
        chunk = {
            "content":  results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i]
        }
        chunks.append(chunk)

    logger.info(f"Found {len(chunks)} relevant chunks for query")
    return chunks

def get_collection_stats():
    """Get stats about the collection."""
    collection = get_or_create_collection()
    if collection:
        count = collection.count()
        logger.info(f"Collection '{settings.chroma_collection}' has {count} chunks")
        return count
    return 0

def clear_collection():
    """Clear all data from collection."""
    try:
        client.delete_collection(settings.chroma_collection)
        logger.info("Collection cleared")
    except Exception as e:
        logger.error(f"Clear failed: {e}")

# ── Test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from ingestion.wikipedia_ingestion import run_ingestion
    from pipeline.chunker import chunk_documents, generate_embeddings, embedding_model

    logger.info("Testing ChromaDB store...")

    # Ingest → chunk → embed → store
    docs   = run_ingestion()
    chunks = chunk_documents(docs)
    chunks = generate_embeddings(chunks)
    store_chunks(chunks)

    # Test search
    query = "What is Apache Kafka?"
    query_embedding = embedding_model.encode([query])[0].tolist()
    results = search(query, query_embedding, top_k=3)

    logger.success(f"Search results for: '{query}'")
    for i, r in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"  Title:   {r['metadata']['title']}")
        print(f"  Source:  {r['metadata']['source']}")
        print(f"  Content: {r['content'][:150]}...")
        print(f"  Distance:{r['distance']:.4f}")