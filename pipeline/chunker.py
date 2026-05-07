# pipeline/chunker.py
# Splits documents into chunks and generates embeddings

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from loguru import logger
from config.settings import get_settings

settings = get_settings()

# ── Load Embedding Model ──────────────────────────────────────────────────────
logger.info(f"Loading embedding model: {settings.embedding_model}")
embedding_model = SentenceTransformer(settings.embedding_model)
logger.success("Embedding model loaded ✅")

# ── Text Splitter ─────────────────────────────────────────────────────────────
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
    separators=["\n\n", "\n", ".", " "]
)

# ── Chunk Documents ───────────────────────────────────────────────────────────
def chunk_documents(documents: list) -> list:
    """Split documents into smaller chunks."""
    chunks = []
    for doc in documents:
        content = doc.get("content", "")
        if not content:
            continue
        splits = text_splitter.split_text(content)
        for i, split in enumerate(splits):
            chunk = {
                "id":          f"{doc['id']}_chunk_{i}",
                "doc_id":      doc["id"],
                "source":      doc["source"],
                "title":       doc["title"],
                "content":     split,
                "url":         doc.get("url", ""),
                "chunk_index": i,
                "total_chunks":len(splits)
            }
            chunks.append(chunk)
    logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks

# ── Generate Embeddings ───────────────────────────────────────────────────────
def generate_embeddings(chunks: list) -> list:
    """Generate vector embeddings for each chunk."""
    texts = [chunk["content"] for chunk in chunks]
    logger.info(f"Generating embeddings for {len(texts)} chunks...")
    embeddings = embedding_model.encode(texts, show_progress_bar=True)
    for i, chunk in enumerate(chunks):
        chunk["embedding"] = embeddings[i].tolist()
    logger.success(f"Generated {len(embeddings)} embeddings ✅")
    return chunks

# ── Test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from ingestion.wikipedia_ingestion import run_ingestion

    logger.info("Testing chunker + embeddings...")
    docs = run_ingestion()
    chunks = chunk_documents(docs)
    chunks_with_embeddings = generate_embeddings(chunks)

    logger.success(f"Total chunks: {len(chunks_with_embeddings)}")
    logger.info(f"Sample chunk:")
    logger.info(f"  Title: {chunks_with_embeddings[0]['title']}")
    logger.info(f"  Content: {chunks_with_embeddings[0]['content'][:100]}...")
    logger.info(f"  Embedding size: {len(chunks_with_embeddings[0]['embedding'])}")