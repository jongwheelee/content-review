"""Generate embeddings for all content in the database.

This script generates embeddings for:
1. Data points (economic/financial metrics)
2. Filing facts (SEC GAAP data)
3. Research papers (NBER, Wikipedia)
4. News articles
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import List
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

from sentence_transformers import SentenceTransformer
from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert

from database.connection import db
from database.models import (
    DataPoint, DataSource, DataCategory, Embedding,
    FilingFact, Filing, Company,
    ResearchPaper, NewsArticle,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 32


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks for embedding."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # Try to break at sentence boundary
        if end < len(text):
            search_text = text[max(start, end - 200):end]
            for punct in [". ", ".\n", "! ", "? ", "\n"]:
                idx = search_text.rfind(punct)
                if idx != -1:
                    end = max(start, end - 200) + idx + len(punct)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks


async def embed_data_points():
    """Generate embeddings for data points."""
    logger.info("=== Embedding Data Points ===")

    db.initialize()
    model = SentenceTransformer(MODEL_NAME)
    total_embedded = 0

    async with db.async_session() as session:
        # Find data points without embeddings
        subquery = select(Embedding.source_id).where(
            Embedding.source_type == "data_point"
        )
        stmt = select(DataPoint).where(
            DataPoint.id.not_in(subquery)
        ).limit(2000)

        result = await session.execute(stmt)
        data_points = result.scalars().all()

        if not data_points:
            logger.info("  No data points to embed")
            return 0

        logger.info(f"  Found {len(data_points)} data points to embed")

        for i in range(0, len(data_points), BATCH_SIZE):
            batch = data_points[i:i + BATCH_SIZE]
            texts = []
            records = []

            for dp in batch:
                description = (
                    f"The {dp.metric_name} as reported by {dp.source.value} "
                    f"on {dp.date_recorded} was {dp.value} {dp.unit or ''}. "
                    f"Category: {dp.category.value}. "
                    f"Geographic scope: {dp.geographic_scope or 'Unknown'}."
                )

                texts.append(description)
                records.append({
                    "source_type": "data_point",
                    "source_id": dp.id,
                    "content_text": description,
                    "meta": {
                        "source": dp.source.value,
                        "category": dp.category.value,
                        "metric_name": dp.metric_name,
                    },
                })

            # Generate embeddings
            embeddings = model.encode(texts, show_progress_bar=False)

            for record, embedding in zip(records, embeddings):
                record["embedding"] = embedding.tolist()

            stmt = insert(Embedding).values(records)
            await session.execute(stmt)
            await session.commit()

            total_embedded += len(records)
            logger.info(f"  Embedded {len(records)} data points (batch {i // BATCH_SIZE + 1})")

    logger.info(f"Data points complete: {total_embedded} embeddings")
    return total_embedded


async def embed_filing_facts():
    """Generate embeddings for SEC filing facts."""
    logger.info("=== Embedding Filing Facts ===")

    db.initialize()
    model = SentenceTransformer(MODEL_NAME)
    total_embedded = 0

    async with db.async_session() as session:
        # Find filing facts without embeddings, joined with filings and companies
        subquery = select(Embedding.source_id).where(
            Embedding.source_type == "filing_fact"
        )

        stmt = (
            select(
                FilingFact.id,
                FilingFact.metric_name,
                FilingFact.value,
                FilingFact.unit,
                FilingFact.period_end,
                Filing.form_type,
                Company.ticker,
                Company.name.label("company_name"),
            )
            .join(Filing, FilingFact.filing_id == Filing.id)
            .join(Company, Filing.company_id == Company.id)
            .where(FilingFact.id.not_in(subquery))
            .limit(2000)
        )

        result = await session.execute(stmt)
        filing_facts = result.fetchall()

        if not filing_facts:
            logger.info("  No filing facts to embed")
            return 0

        logger.info(f"  Found {len(filing_facts)} filing facts to embed")

        for i in range(0, len(filing_facts), BATCH_SIZE):
            batch = filing_facts[i:i + BATCH_SIZE]
            texts = []
            records = []

            for ff in batch:
                description = (
                    f"{ff.company_name} ({ff.ticker}) reported {ff.metric_name} "
                    f"of {ff.value} {ff.unit or ''} for the period ending "
                    f"{ff.period_end} in their {ff.form_type} filing."
                )

                texts.append(description)
                records.append({
                    "source_type": "filing_fact",
                    "source_id": ff.id,
                    "content_text": description,
                    "meta": {
                        "ticker": ff.ticker,
                        "company_name": ff.company_name,
                        "metric_name": ff.metric_name,
                        "form_type": ff.form_type,
                    },
                })

            embeddings = model.encode(texts, show_progress_bar=False)

            for record, embedding in zip(records, embeddings):
                record["embedding"] = embedding.tolist()

            stmt = insert(Embedding).values(records)
            await session.execute(stmt)
            await session.commit()

            total_embedded += len(records)
            logger.info(f"  Embedded {len(records)} filing facts (batch {i // BATCH_SIZE + 1})")

    logger.info(f"Filing facts complete: {total_embedded} embeddings")
    return total_embedded


async def embed_research_papers():
    """Generate embeddings for research papers (NBER, Wikipedia)."""
    logger.info("=== Embedding Research Papers ===")

    db.initialize()
    model = SentenceTransformer(MODEL_NAME)
    total_embedded = 0

    async with db.async_session() as session:
        # Fetch unprocessed research papers
        stmt = select(ResearchPaper).where(
            ResearchPaper.processed == False
        ).limit(500)

        result = await session.execute(stmt)
        papers = result.scalars().all()

        if not papers:
            logger.info("  No research papers to embed")
            return 0

        logger.info(f"  Found {len(papers)} research papers to embed")

        for i in range(0, len(papers), BATCH_SIZE):
            batch = papers[i:i + BATCH_SIZE]
            texts = []
            records = []
            paper_ids = []

            for paper in batch:
                abstract = paper.abstract or ""
                # Combine title and abstract, truncate if needed
                text = f"{paper.title}. {abstract[:2000]}"

                texts.append(text)
                records.append({
                    "source_type": "research_paper",
                    "source_id": paper.id,
                    "content_text": text,
                    "meta": {
                        "paper_type": paper.paper_type,
                        "external_id": paper.external_id,
                        "title": paper.title,
                    },
                })
                paper_ids.append(paper.id)

            embeddings = model.encode(texts, show_progress_bar=False)

            for record, embedding in zip(records, embeddings):
                record["embedding"] = embedding.tolist()

            stmt = insert(Embedding).values(records)
            await session.execute(stmt)

            # Mark papers as processed
            for paper_id in paper_ids:
                paper = await session.get(ResearchPaper, paper_id)
                if paper:
                    paper.processed = True

            await session.commit()

            total_embedded += len(records)
            logger.info(f"  Embedded {len(records)} research papers (batch {i // BATCH_SIZE + 1})")

    logger.info(f"Research papers complete: {total_embedded} embeddings")
    return total_embedded


async def embed_news_articles():
    """Generate embeddings for news articles."""
    logger.info("=== Embedding News Articles ===")

    db.initialize()
    model = SentenceTransformer(MODEL_NAME)
    total_embedded = 0

    async with db.async_session() as session:
        # Find news articles without embeddings
        subquery = select(Embedding.source_id).where(
            Embedding.source_type == "news_article"
        )
        stmt = select(NewsArticle).where(
            NewsArticle.id.not_in(subquery)
        ).limit(500)

        result = await session.execute(stmt)
        articles = result.scalars().all()

        if not articles:
            logger.info("  No news articles to embed")
            return 0

        logger.info(f"  Found {len(articles)} news articles to embed")

        for i in range(0, len(articles), BATCH_SIZE):
            batch = articles[i:i + BATCH_SIZE]
            texts = []
            records = []

            for article in batch:
                # Combine headline and content summary
                content = article.content_summary or ""
                text = f"{article.headline}. {content[:1500]}"

                texts.append(text)
                records.append({
                    "source_type": "news_article",
                    "source_id": article.id,
                    "content_text": text,
                    "meta": {
                        "source_name": article.source_name,
                        "published_at": str(article.published_at) if article.published_at else None,
                        "tickers_mentioned": article.tickers_mentioned or [],
                        "topics": article.topics or [],
                    },
                })

            embeddings = model.encode(texts, show_progress_bar=False)

            for record, embedding in zip(records, embeddings):
                record["embedding"] = embedding.tolist()

            stmt = insert(Embedding).values(records)
            await session.execute(stmt)
            await session.commit()

            total_embedded += len(records)
            logger.info(f"  Embedded {len(records)} news articles (batch {i // BATCH_SIZE + 1})")

    logger.info(f"News articles complete: {total_embedded} embeddings")
    return total_embedded


async def run_all():
    """Run all embedding generation."""
    logger.info("=" * 60)
    logger.info("Starting embedding generation for all content")
    logger.info("=" * 60)

    start_time = datetime.now()

    results = {
        "data_points": await embed_data_points(),
        "filing_facts": await embed_filing_facts(),
        "research_papers": await embed_research_papers(),
        "news_articles": await embed_news_articles(),
    }

    total = sum(results.values())
    elapsed = datetime.now() - start_time

    logger.info("=" * 60)
    logger.info("Embedding generation complete!")
    logger.info(f"  Data points: {results['data_points']}")
    logger.info(f"  Filing facts: {results['filing_facts']}")
    logger.info(f"  Research papers: {results['research_papers']}")
    logger.info(f"  News articles: {results['news_articles']}")
    logger.info(f"  TOTAL: {total} embeddings")
    logger.info(f"  Time elapsed: {elapsed}")
    logger.info("=" * 60)

    return {"total": total, "by_type": results, "elapsed": str(elapsed)}


if __name__ == "__main__":
    asyncio.run(run_all())
