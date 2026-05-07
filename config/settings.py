# config/settings.py
# Central configuration for AI-Powered Data Pipeline

import os
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # Ollama (Local LLM)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"

    # ChromaDB (Vector Database)
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_collection: str = "knowledge_base"

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"

    # Text Chunking
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Data Ingestion
    wikipedia_topics: list = [
        "Apache Kafka",
        "Apache Spark",
        "Data Engineering",
        "Machine Learning",
        "Vector Database",
        "Large Language Models",
        "Apache Iceberg",
        "Data Lakehouse"
    ]
    arxiv_max_results: int = 5

    # AWS S3
    aws_region: str = "us-east-1"
    s3_bucket: str = "iot-pipeline-saikrishna"
    s3_documents_prefix: str = "ai-pipeline/documents"

    # FastAPI
    api_host: str = "0.0.0.0"
    api_port: int = 8080

    # Dashboard
    dash_port: int = 8050

    # RAG
    retriever_top_k: int = 3

    class Config:
        env_file = ".env"

@lru_cache
def get_settings():
    return Settings()