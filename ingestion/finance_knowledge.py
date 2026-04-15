"""Wikipedia finance knowledge ingestion module.

Fetches general finance knowledge from Wikipedia's public API.
"""

import asyncio
import logging
import re
from datetime import datetime
from typing import Any

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from database.connection import db
from database.models import ResearchPaper, IngestionLog, IngestionStatus, DataSource, Embedding

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 50 core finance topics to fetch
FINANCE_TOPICS = [
    "Inflation",
    "Interest rate",
    "Federal Reserve",
    "Yield curve",
    "Price-earnings ratio",
    "Market capitalization",
    "Bond",
    "Equity",
    "Monetary policy",
    "Fiscal policy",
    "GDP",
    "Recession",
    "Bear market",
    "Bull market",
    "Hedge fund",
    "Private equity",
    "IPO",
    "Dividend",
    "Short selling",
    "Margin call",
    "Quantitative easing",
    "Stagflation",
    "Deflation",
    "Credit default swap",
    "Collateralized debt obligation",
    "Federal funds rate",
    "Treasury bond",
    "Corporate bond",
    "Junk bond",
    "Asset allocation",
    "Portfolio diversification",
    "Beta",
    "Alpha",
    "Sharpe ratio",
    "Moving average",
    "RSI",
    "MACD",
    "Earnings per share",
    "Book value",
    "Enterprise value",
    "EBITDA",
    "Free cash flow",
    "Working capital",
    "Leverage",
    "Derivatives",
    "Options",
    "Futures",
    "Commodity",
    "Exchange rate",
    "Balance of payments",
]


def clean_html(text: str) -> str:
    """Remove HTML tags from text."""
    if not text:
        return ""
    soup = BeautifulSoup(text, 'html.parser')
    return soup.get_text(separator=' ', strip=True)


def extract_keywords(summary: str, sections: list[dict]) -> list[str]:
    """Extract finance-related keywords from article content."""
    keywords = []

    # Common finance terms to look for
    finance_terms = [
        "finance", "financial", "economics", "investment", "investing",
        "market", "trading", "stock", "bond", "equity", "debt",
        "interest", "inflation", "monetary", "fiscal", "policy",
        "portfolio", "asset", "liability", "capital", "liquidity",
        "risk", "return", "yield", "dividend", "valuation",
        "credit", "rating", "derivative", "option", "future",
        "hedge", "leverage", "margin", "volatility", "beta",
    ]

    text = f"{summary}".lower()
    for term in finance_terms:
        if term in text and term not in keywords:
            keywords.append(term)

    return keywords[:10]  # Limit to 10 keywords


async def fetch_wikipedia_summary(topic: str) -> dict[str, Any] | None:
    """Fetch Wikipedia article summary.

    Args:
        topic: Wikipedia article title

    Returns:
        Summary data dict or None
    """
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic}"

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                url,
                timeout=30.0,
                headers={
                    "User-Agent": "ContentReviewPipeline/1.0 (https://github.com/content-review-pipeline)",
                    "Accept": "application/json",
                }
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.warning(f"Error fetching summary for {topic}: {e}")
            return None


async def fetch_wikipedia_sections(topic: str) -> list[dict] | None:
    """Fetch Wikipedia article sections.

    Args:
        topic: Wikipedia article title

    Returns:
        List of section dicts or None
    """
    url = f"https://en.wikipedia.org/api/rest_v1/page/sections/{topic}"

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                url,
                timeout=30.0,
                headers={
                    "User-Agent": "ContentReviewPipeline/1.0 (https://github.com/content-review-pipeline)",
                    "Accept": "application/json",
                }
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.warning(f"Error fetching sections for {topic}: {e}")
            return None


async def upsert_research_paper(session, paper: dict) -> tuple[bool, int]:
    """Upsert a research paper into the database.

    Args:
        session: SQLAlchemy async session
        paper: Paper dict with metadata

    Returns:
        Tuple of (was_inserted, paper_id)
    """
    external_id = paper.get("external_id")
    if not external_id:
        return False, 0

    # Check for existing paper
    existing = await session.execute(
        select(ResearchPaper.id).where(ResearchPaper.external_id == external_id)
    )
    existing_id = existing.scalar_one_or_none()

    if existing_id:
        # Update existing paper
        paper_record = await session.get(ResearchPaper, existing_id)
        if paper_record:
            paper_record.title = paper.get("title", paper_record.title)
            paper_record.abstract = paper.get("abstract", paper_record.abstract)
            paper_record.keywords = paper.get("keywords", paper_record.keywords)
            paper_record.updated_at = datetime.now()
        return False, existing_id

    # Insert new paper
    stmt = insert(ResearchPaper).values(paper)
    await session.execute(stmt)

    # Get the inserted ID
    result = await session.execute(
        select(ResearchPaper.id).where(ResearchPaper.external_id == external_id)
    )
    paper_id = result.scalar_one()

    return True, paper_id


async def log_ingestion(
    session,
    source: DataSource,
    status: IngestionStatus,
    records_fetched: int = 0,
    errors: str | None = None,
):
    """Log ingestion run to ingestion_log table."""
    log_entry = IngestionLog(
        source=source,
        status=status,
        records_fetched=records_fetched,
        errors=errors,
    )
    session.add(log_entry)


async def run(topics: list[str] | None = None):
    """Fetch Wikipedia finance articles and upsert into database.

    This function:
    1. Fetches article summaries and sections from Wikipedia API
    2. Cleans HTML and extracts keywords
    3. Upserts articles into research_papers table
    4. Generates embeddings for new articles

    Args:
        topics: List of topics to fetch. Defaults to FINANCE_TOPICS.

    Returns:
        Dict with ingestion statistics
    """
    logger.info("Starting Wikipedia finance knowledge ingestion run")

    # Initialize database connection
    db.initialize()

    topics = topics or FINANCE_TOPICS

    total_fetched = 0
    total_inserted = 0
    total_embedded = 0
    errors = []
    papers_to_embed = []

    async with db.async_session() as session:
        # Log start
        await log_ingestion(
            session,
            DataSource.WIKIPEDIA,
            IngestionStatus.RUNNING,
        )
        await session.commit()

        for topic in topics:
            logger.info(f"Fetching: {topic}")

            try:
                # Fetch summary
                summary_data = await fetch_wikipedia_summary(topic)

                if not summary_data:
                    logger.warning(f"  No summary found for {topic}")
                    await asyncio.sleep(1)
                    continue

                # Skip disambiguation pages
                if summary_data.get("type") == "disambiguation":
                    logger.info(f"  Skipping disambiguation page: {topic}")
                    await asyncio.sleep(1)
                    continue

                total_fetched += 1

                # Fetch sections
                sections_data = await fetch_wikipedia_sections(topic)

                # Build abstract from summary and first 3 sections
                summary_text = clean_html(summary_data.get("extract", ""))

                section_texts = [summary_text]
                if sections_data:
                    for section in sections_data[:3]:
                        section_text = clean_html(section.get("text", ""))
                        if section_text:
                            section_texts.append(section_text)

                abstract = " ".join(section_texts)[:4000]  # Truncate if needed

                # Extract keywords
                keywords = extract_keywords(summary_text, sections_data or [])
                keywords = [topic] + keywords  # Add topic as first keyword

                # Build paper record
                external_id = f"wiki_{topic.lower().replace(' ', '_')}"
                paper = {
                    "external_id": external_id,
                    "title": summary_data.get("title", topic),
                    "authors": None,
                    "abstract": abstract,
                    "published_date": None,
                    "url": summary_data.get("content_urls", {}).get("desktop", {}).get("page", f"https://en.wikipedia.org/wiki/{topic}"),
                    "keywords": keywords,
                    "paper_type": "wikipedia",
                    "processed": False,
                }

                # Upsert paper
                inserted, paper_id = await upsert_research_paper(session, paper)

                if inserted:
                    total_inserted += 1
                    papers_to_embed.append((paper_id, paper["title"], abstract))
                    logger.info(f"  Inserted: {external_id}")
                else:
                    logger.info(f"  Already exists: {external_id}")

            except Exception as e:
                error_msg = f"Error processing {topic}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)

            # Rate limiting: 1 second delay between requests
            await asyncio.sleep(1)

            await session.commit()

        # Generate embeddings for new papers
        if papers_to_embed:
            logger.info(f"Generating embeddings for {len(papers_to_embed)} new articles...")
            try:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer('all-MiniLM-L6-v2')

                for paper_id, title, abstract in papers_to_embed:
                    # Combine title and abstract for embedding
                    text = f"{title}. {abstract}"

                    try:
                        embedding_vector = model.encode(text)

                        # Insert embedding into embeddings table
                        embedding_record = {
                            "source_type": "research_paper",
                            "source_id": paper_id,
                            "content_text": text,
                            "embedding": embedding_vector.tolist(),
                            "metadata_json": {"paper_type": "wikipedia"},
                        }
                        stmt = insert(Embedding).values(embedding_record)
                        await session.execute(stmt)

                        # Mark paper as processed
                        paper = await session.get(ResearchPaper, paper_id)
                        if paper:
                            paper.processed = True

                        total_embedded += 1
                    except Exception as e:
                        logger.error(f"Error generating embedding for paper {paper_id}: {e}")
                        errors.append(f"Embedding error for {paper_id}: {e}")

                await session.commit()
                logger.info(f"  Generated {total_embedded} embeddings")

            except Exception as e:
                error_msg = f"Error generating embeddings: {e}"
                logger.error(error_msg)
                errors.append(error_msg)

        # Log completion
        final_status = IngestionStatus.FAILED if errors else IngestionStatus.COMPLETED
        error_text = " | ".join(errors) if errors else None

        await log_ingestion(
            session,
            DataSource.WIKIPEDIA,
            final_status,
            records_fetched=total_fetched,
            errors=error_text,
        )
        await session.commit()

    logger.info(
        f"Wikipedia ingestion complete: "
        f"{total_fetched} articles fetched, "
        f"{total_inserted} inserted, "
        f"{total_embedded} embedded, "
        f"{len(errors)} errors"
    )

    return {
        "total_fetched": total_fetched,
        "total_inserted": total_inserted,
        "total_embedded": total_embedded,
        "errors": errors,
    }


if __name__ == "__main__":
    asyncio.run(run())
