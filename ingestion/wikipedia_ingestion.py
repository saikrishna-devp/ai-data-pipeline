# ingestion/wikipedia_ingestion.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import boto3
import wikipedia
import arxiv
from datetime import datetime, timezone
from loguru import logger
from config.settings import get_settings

settings = get_settings()

s3 = boto3.client("s3", region_name=settings.aws_region)

def utcnow():
    return datetime.now(timezone.utc)

def fetch_wikipedia_article(topic: str) -> dict:
    try:
        import requests
        url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + topic.replace(" ", "_")
        headers = {"User-Agent": "ai-data-pipeline/1.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        content = data.get("extract", "")
        if not content:
            logger.error(f"Wikipedia: no content for '{topic}'")
            return None
        doc = {
            "id":          f"wiki_{topic.replace(' ', '_')}",
            "source":      "wikipedia",
            "topic":       topic,
            "title":       data.get("title", topic),
            "content":     content,
            "url":         data.get("content_urls", {}).get("desktop", {}).get("page", ""),
            "ingested_at": utcnow().isoformat()
        }
        logger.success(f"Wikipedia: fetched '{topic}' ({len(content)} chars)")
        return doc
    except Exception as e:
        logger.error(f"Wikipedia failed for '{topic}': {e}")
        return None

def fetch_all_wikipedia() -> list:
    documents = []
    for topic in settings.wikipedia_topics:
        doc = fetch_wikipedia_article(topic)
        if doc:
            documents.append(doc)
    logger.info(f"Wikipedia: fetched {len(documents)} articles")
    return documents

def fetch_arxiv_papers(query: str = "data engineering RAG pipeline") -> list:
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=settings.arxiv_max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        documents = []
        for paper in client.results(search):
            doc = {
                "id":          f"arxiv_{paper.entry_id.split('/')[-1]}",
                "source":      "arxiv",
                "topic":       query,
                "title":       paper.title,
                "content":     f"{paper.title}\n\n{paper.summary}",
                "url":         paper.entry_id,
                "authors":     [a.name for a in paper.authors[:3]],
                "ingested_at": utcnow().isoformat()
            }
            documents.append(doc)
            logger.success(f"ArXiv: fetched '{paper.title[:50]}...'")
        logger.info(f"ArXiv: fetched {len(documents)} papers")
        return documents
    except Exception as e:
        logger.error(f"ArXiv failed: {e}")
        return []

def save_to_s3(documents: list, source: str):
    if not documents:
        return
    timestamp = utcnow().strftime("%Y%m%d_%H%M%S")
    key = f"{settings.s3_documents_prefix}/{source}/{source}_{timestamp}.json"
    data = json.dumps(documents, indent=2).encode("utf-8")
    s3.put_object(
        Bucket=settings.s3_bucket,
        Key=key,
        Body=data,
        ContentType="application/json"
    )
    logger.success(f"S3: saved {len(documents)} docs → s3://{settings.s3_bucket}/{key}")

def run_ingestion():
    logger.info("Starting data ingestion...")
    wiki_docs = fetch_all_wikipedia()
    save_to_s3(wiki_docs, "wikipedia")
    arxiv_docs = fetch_arxiv_papers("data engineering RAG pipeline")
    save_to_s3(arxiv_docs, "arxiv")
    all_docs = wiki_docs + arxiv_docs
    logger.success(f"Ingestion complete! Total: {len(all_docs)} documents")
    return all_docs

if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    docs = run_ingestion()
    print(f"\nSample document:")
    print(f"Title: {docs[0]['title']}")
    print(f"Source: {docs[0]['source']}")
    print(f"Content preview: {docs[0]['content'][:200]}...")