"""NBER working papers ingestion module."""

import asyncio
import logging
import re
from datetime import datetime
from typing import Any

import feedparser
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


# NBER working papers page URL (RSS feed is down, scraping directly)
NBER_PAPERS_URL = "https://www.nber.org/papers-working"
NBER_RSS_URL = "https://www.nber.org/papers-working"  # Fallback to page URL for scraping

# Keywords for filtering papers (matches JEL codes and research areas)
RESEARCH_KEYWORDS = [
    "monetary policy",
    "inflation",
    "interest rate",
    "GDP",
    "recession",
    "fiscal policy",
    "labor market",
    "financial crisis",
    "asset pricing",
    "corporate finance",
    "banking",
    "Federal Reserve",
    "yield curve",
    "credit",
    "investment",
]

# Number of papers to fetch from feed
PAPERS_TO_FETCH = 50


def extract_nber_number(entry: dict) -> str | None:
    """Extract NBER working paper number from entry.

    Looks for patterns like "w31234", "No. 31234", etc.

    Args:
        entry: Feedparser entry dict

    Returns:
        NBER paper number (e.g., "w31234") or None
    """
    # Check title for paper number pattern
    title = entry.get("title", "")
    match = re.search(r"\b(?:No\.?\s*)?(\d{4,6})\b", title, re.IGNORECASE)
    if match:
        return f"w{match.group(1)}"

    # Check link for pattern
    link = entry.get("link", "")
    match = re.search(r"/papers/(\w{1,2}\d{4,6})", link)
    if match:
        return match.group(1)

    # Check summary for paper number
    summary = entry.get("summary", "")
    match = re.search(r"\b(?:Working Paper|No\.?)\s*(\w{1,2}\d{4,6})\b", summary, re.IGNORECASE)
    if match:
        paper_num = match.group(1)
        if not paper_num.startswith("w"):
            paper_num = f"w{paper_num}"
        return paper_num

    return None


def match_keywords(title: str, abstract: str) -> list[str]:
    """Match paper against research keywords.

    Args:
        title: Paper title
        abstract: Paper abstract

    Returns:
        List of matched keywords
    """
    text = f"{title} {abstract}".lower()
    matched = []

    for keyword in RESEARCH_KEYWORDS:
        if keyword.lower() in text:
            matched.append(keyword)

    return matched


def parse_authors(entry: dict) -> list[str]:
    """Parse authors from feed entry.

    Args:
        entry: Feedparser entry dict

    Returns:
        List of author names
    """
    authors = []

    # Try author field
    author_field = entry.get("author", "")
    if author_field:
        # Split by comma or "and"
        parts = re.split(r",\s*|\s+and\s+", author_field)
        authors.extend([p.strip() for p in parts if p.strip()])

    # Try author_detail field
    author_detail = entry.get("author_detail", {})
    if author_detail and author_detail.get("name"):
        name = author_detail["name"]
        if name not in authors:
            authors.append(name)

    # Try contributors
    contributors = entry.get("contributors", [])
    for contributor in contributors:
        name = contributor.get("name", "")
        if name and name not in authors:
            authors.append(name)

    return authors


def parse_published_date(entry: dict) -> datetime | None:
    """Parse published date from feed entry.

    Args:
        entry: Feedparser entry dict

    Returns:
        Parsed datetime or None
    """
    # Try published_parsed
    published_parsed = entry.get("published_parsed")
    if published_parsed:
        try:
            return datetime(*published_parsed[:6])
        except (TypeError, ValueError):
            pass

    # Try published string
    published = entry.get("published", "")
    if published:
        try:
            return datetime.strptime(published, "%a, %d %b %Y %H:%M:%S %z")
        except (ValueError, TypeError):
            pass

    # Try updated
    updated = entry.get("updated", "")
    if updated:
        try:
            return datetime.strptime(updated, "%a, %d %b %Y %H:%M:%S %z")
        except (ValueError, TypeError):
            pass

    return datetime.utcnow()


async def fetch_rss_feed(url: str, limit: int = 50) -> list[dict]:
    """Scrape NBER working papers page (RSS feed is down).

    Args:
        url: NBER working papers page URL
        limit: Maximum number of entries to return

    Returns:
        List of entry dicts with paper metadata
    """
    async with httpx.AsyncClient() as client:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
        response = await client.get(url, headers=headers, timeout=30.0)
        response.raise_for_status()

        # Parse HTML to extract paper entries
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        entries = []
        # Look for paper entries in the HTML structure
        # NBER working papers page lists papers with title, authors, and link
        paper_cards = soup.find_all('div', class_='paper-card') or soup.find_all('article') or soup.find_all('div', class_='paper')

        for card in paper_cards[:limit]:
            title_elem = card.find('h3') or card.find('h2') or card.find('a', class_='title')
            link_elem = card.find('a', href=True)
            authors_elem = card.find('div', class_='authors') or card.find('p', class_='authors')
            summary_elem = card.find('div', class_='abstract') or card.find('p', class_='summary')

            if title_elem or link_elem:
                entry = {
                    'title': title_elem.get_text(strip=True) if title_elem else 'Untitled',
                    'link': link_elem['href'] if link_elem else '',
                    'author': authors_elem.get_text(strip=True) if authors_elem else '',
                    'summary': summary_elem.get_text(strip=True) if summary_elem else '',
                }
                entries.append(entry)

        # Fallback: try to find any paper-like structure
        if not entries:
            # Try finding all links that look like paper links
            all_links = soup.find_all('a', href=re.compile(r'/papers/'))
            for link in all_links[:limit]:
                title = link.get_text(strip=True)
                if title and len(title) > 10:  # Filter out nav links
                    entry = {
                        'title': title,
                        'link': link['href'],
                        'author': '',
                        'summary': '',
                    }
                    entries.append(entry)

        return entries


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
            paper_record.updated_at = datetime.utcnow()
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


async def run(limit: int = PAPERS_TO_FETCH):
    """Fetch NBER working papers and upsert into database.

    This function:
    1. Fetches the NBER RSS feed
    2. Filters papers by research keywords
    3. Upserts matching papers into research_papers table
    4. Generates embeddings for new papers

    Args:
        limit: Maximum number of papers to fetch from feed

    Returns:
        Dict with ingestion statistics
    """
    logger.info("Starting NBER working papers ingestion run")

    # Initialize database connection
    db.initialize()

    total_fetched = 0
    total_matched = 0
    total_inserted = 0
    total_embedded = 0
    errors = []
    papers_to_embed = []

    async with db.async_session() as session:
        # Log start
        await log_ingestion(
            session,
            DataSource.NBER,
            IngestionStatus.RUNNING,
        )
        await session.commit()

        # Step 1: Fetch RSS feed
        logger.info(f"Fetching NBER RSS feed (limit: {limit} papers)...")
        try:
            entries = await fetch_rss_feed(NBER_RSS_URL, limit=limit)
            total_fetched = len(entries)
            logger.info(f"  Fetched {total_fetched} papers from feed")
        except httpx.HTTPError as e:
            error_msg = f"HTTP error fetching RSS feed: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
            await log_ingestion(
                session,
                DataSource.NBER,
                IngestionStatus.FAILED,
                errors=error_msg,
            )
            await session.commit()
            return {"error": error_msg}
        except Exception as e:
            error_msg = f"Error parsing RSS feed: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
            await log_ingestion(
                session,
                DataSource.NBER,
                IngestionStatus.FAILED,
                errors=error_msg,
            )
            await session.commit()
            return {"error": error_msg}

        # Step 2: Filter and process papers
        logger.info("Filtering papers by research keywords...")
        for entry in entries:
            title = entry.get("title", "")
            summary = entry.get("summary", "")
            abstract = summary[:2000] if summary else ""  # Truncate very long abstracts

            # Match keywords
            matched_keywords = match_keywords(title, abstract)

            if not matched_keywords:
                continue

            total_matched += 1

            # Extract metadata
            external_id = extract_nber_number(entry)
            if not external_id:
                # Generate ID from link
                link = entry.get("link", "")
                match = re.search(r"/papers/(\w+)", link)
                if match:
                    external_id = match.group(1)
                else:
                    external_id = f"nber_{len(title)}_{hash(title) % 10000}"

            authors = parse_authors(entry)
            published_date = parse_published_date(entry)
            link = entry.get("link", "")

            paper = {
                "external_id": external_id,
                "title": title,
                "authors": authors if authors else None,
                "abstract": abstract,
                "published_date": published_date.date() if published_date else None,
                "url": link,
                "keywords": matched_keywords,
                "paper_type": "nber",
                "processed": False,
            }

            # Upsert paper
            try:
                inserted, paper_id = await upsert_research_paper(session, paper)
                if inserted:
                    total_inserted += 1
                    papers_to_embed.append((paper_id, title, abstract))
                    logger.info(f"  Inserted: {external_id} - {title[:60]}...")
                else:
                    logger.debug(f"  Already exists: {external_id}")
            except Exception as e:
                error_msg = f"Error upserting paper {external_id}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)

        await session.commit()

        # Step 3: Generate embeddings for new papers
        if papers_to_embed:
            logger.info(f"Generating embeddings for {len(papers_to_embed)} new papers...")
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
                            "metadata_json": {"paper_type": "nber", "external_id": external_id},
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
            DataSource.NBER,
            final_status,
            records_fetched=total_matched,
            errors=error_text,
        )
        await session.commit()

    logger.info(
        f"NBER ingestion complete: "
        f"{total_fetched} papers fetched, "
        f"{total_matched} matched keywords, "
        f"{total_inserted} inserted, "
        f"{total_embedded} embedded, "
        f"{len(errors)} errors"
    )

    return {
        "total_fetched": total_fetched,
        "total_matched": total_matched,
        "total_inserted": total_inserted,
        "total_embedded": total_embedded,
        "errors": errors,
    }


if __name__ == "__main__":
    asyncio.run(run())
