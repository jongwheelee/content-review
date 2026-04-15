"""Earnings call transcripts ingestion module.

Uses Motley Fool's publicly accessible transcript archive.
No API key required - web scraping approach.
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
from database.models import Company, EarningsTranscript, IngestionLog, IngestionStatus, DataSource
# Embedder import removed - using direct sentence-transformers to avoid NumPy issues
# from processing.embedder import Embedder

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Top 25 companies by market cap for transcript ingestion
TOP_25_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "BRK.B", "TSLA", "AVGO", "JPM",
    "JNJ", "V", "PG", "MA", "UNH", "HD", "CVX", "MRK", "ABBV", "PEP",
    "KO", "COST", "LLY", "WMT", "MCD",
]

# Company name mappings for Fool.com search
# Fool.com uses company names, not tickers, in their URLs
COMPANY_NAME_MAPPING = {
    "AAPL": "apple",
    "MSFT": "microsoft",
    "GOOGL": "alphabet",
    "AMZN": "amazon",
    "NVDA": "nvidia",
    "META": "meta-platforms",
    "BRK.B": "berkshire-hathaway",
    "TSLA": "tesla",
    "AVGO": "broadcom",
    "JPM": "jpmorgan-chase",
    "JNJ": "johnson-johnson",
    "V": "visa",
    "PG": "procter-gamble",
    "MA": "mastercard",
    "UNH": "unitedhealth-group",
    "HD": "home-depot",
    "CVX": "chevron",
    "MRK": "merck",
    "ABBV": "abbvie",
    "PEP": "pepsico",
    "KO": "coca-cola",
    "COST": "costco",
    "LLY": "eli-lilly",
    "WMT": "walmart",
    "MCD": "mcdonalds",
}

BASE_URL = "https://www.fool.com"
TRANSCRIPTS_BASE = "https://www.fool.com/earnings-call-transcripts/"

# Rate limiting
REQUEST_DELAY_SECONDS = 3

# Maximum quarters to fetch (last 8 quarters = 2 years)
MAX_QUARTERS_TO_FETCH = 8

# Maximum transcript length
MAX_TRANSCRIPT_LENGTH = 50000


class MotleyFoolClient:
    """Client for Motley Fool earnings call transcripts.

    Uses web scraping approach - no API key required.
    Be respectful with rate limiting.
    """

    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        }
        self.session = None

    async def _get_session(self) -> httpx.AsyncClient:
        """Get or create HTTP session."""
        if self.session is None:
            self.session = httpx.AsyncClient(headers=self.headers, timeout=30.0)
        return self.session

    async def close(self):
        """Close the HTTP session."""
        if self.session:
            await self.session.aclose()
            self.session = None

    async def fetch_transcript_list(self, ticker: str) -> list[dict]:
        """Fetch list of transcript links for a ticker.

        Args:
            ticker: Company ticker symbol

        Returns:
            List of transcript dicts with url, title, date
        """
        client = await self._get_session()

        # Search for transcripts by company name
        company_name = COMPANY_NAME_MAPPING.get(ticker.lower(), ticker.lower())
        search_url = f"{BASE_URL}/quote/{company_name}/earnings/"

        try:
            response = await client.get(search_url)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning(f"  No earnings page for {ticker} ({company_name})")
                return []
            raise

        soup = BeautifulSoup(response.text, "lxml")
        transcripts = []

        # Find transcript links in article listings
        for article in soup.select("article, .article-card, [data-article]"):
            title_elem = article.select_one("h1, h2, h3, .headline")
            link_elem = article.select_one("a[href*='earnings-call-transcript']")
            date_elem = article.select_one("time, .published-date, .date")

            if title_elem and link_elem:
                title = title_elem.get_text(strip=True)

                # Check if title mentions the ticker
                if ticker.upper() not in title.upper():
                    continue

                href = link_elem.get("href", "")
                if not href.startswith("http"):
                    href = f"{BASE_URL}{href}"

                date_str = date_elem.get("datetime", "") if date_elem else ""

                transcripts.append({
                    "title": title,
                    "url": href,
                    "date": date_str,
                })

        # If no transcripts found via company page, try search
        if not transcripts:
            search_url = f"{BASE_URL}/search/?q={company_name}+earnings+call+transcript"
            try:
                response = await client.get(search_url)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "lxml")

                for result in soup.select(".search-result, article")[:20]:
                    title_elem = result.select_one(".title, h1, h2, h3")
                    link_elem = result.select_one("a[href*='earnings-call-transcript']")

                    if title_elem and link_elem:
                        title = title_elem.get_text(strip=True)
                        if ticker.upper() in title.upper():
                            href = link_elem.get("href", "")
                            if not href.startswith("http"):
                                href = f"{BASE_URL}{href}"

                            transcripts.append({
                                "title": title,
                                "url": href,
                                "date": "",
                            })
            except Exception:
                pass

        return transcripts[:20]  # Limit to 20 most recent

    async def fetch_transcript(self, url: str) -> dict[str, Any]:
        """Fetch full transcript content from URL.

        Args:
            url: Transcript page URL

        Returns:
            Dict with transcript content and metadata
        """
        client = await self._get_session()

        response = await client.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "lxml")

        # Extract title
        title_elem = soup.select_one("h1")
        title = title_elem.get_text(strip=True) if title_elem else ""

        # Extract publication date
        date_elem = soup.select_one("time")
        published_date = date_elem.get("datetime", "") if date_elem else ""

        # Extract transcript content
        # Look for article body divs
        content_divs = []

        # Try common content selectors
        for selector in [
            ".article-body",
            ".content",
            "article",
            "[data-article-body]",
            ".post-content",
        ]:
            content_divs = soup.select(selector)
            if content_divs:
                break

        # Extract text from content
        paragraphs = []
        speakers = []

        if content_divs:
            for div in content_divs:
                for p in div.select("p"):
                    text = p.get_text(strip=True)
                    if text and len(text) > 20:  # Skip short lines
                        # Check if this looks like a speaker label
                        if ":" in text and len(text.split(":")[0]) < 50:
                            parts = text.split(":", 1)
                            speakers.append({
                                "speaker": parts[0].strip(),
                                "text": parts[1].strip(),
                            })
                        else:
                            paragraphs.append(text)

        # Build full text
        full_text = "\n\n".join(paragraphs)

        # Parse fiscal quarter and year from title
        fiscal_quarter, fiscal_year = parse_fiscal_period(title)

        return {
            "url": url,
            "title": title,
            "published_date": published_date,
            "full_text": full_text,
            "paragraphs": paragraphs,
            "speakers": speakers,
            "fiscal_quarter": fiscal_quarter,
            "fiscal_year": fiscal_year,
        }


def parse_fiscal_period(title: str) -> tuple[int | None, int | None]:
    """Parse fiscal quarter and year from transcript title.

    Args:
        title: Transcript title

    Returns:
        Tuple of (fiscal_quarter, fiscal_year) or (None, None)
    """
    # Look for patterns like "Q1 2024", "Q4 2023", "Fourth Quarter 2023"
    quarter = None
    year = None

    # Match Q1, Q2, Q3, Q4
    q_match = re.search(r"\bQ([1-4])\b", title, re.IGNORECASE)
    if q_match:
        quarter = int(q_match.group(1))

    # Match year (2020-2029)
    year_match = re.search(r"\b(202[0-9]|203[0-9])\b", title)
    if year_match:
        year = int(year_match.group(1))

    # If no Q pattern, try "First Quarter", "Second Quarter", etc.
    if quarter is None:
        quarter_words = {
            "first": 1, "second": 2, "third": 3, "fourth": 4,
            "1st": 1, "2nd": 2, "3rd": 3, "4th": 4,
        }
        for word, q_num in quarter_words.items():
            if word in title.lower():
                quarter = q_num
                break

    return quarter, year


async def get_company_id_by_ticker(session, ticker: str) -> int | None:
    """Get company ID from database by ticker.

    Args:
        session: SQLAlchemy async session
        ticker: Company ticker symbol

    Returns:
        Company ID or None
    """
    result = await session.execute(
        select(Company.id).where(Company.ticker == ticker.upper())
    )
    return result.scalar_one_or_none()


async def check_existing_transcript(
    session, company_id: int, fiscal_quarter: int, fiscal_year: int
) -> bool:
    """Check if transcript already exists for quarter/year.

    Args:
        session: SQLAlchemy async session
        company_id: Company ID
        fiscal_quarter: Fiscal quarter (1-4)
        fiscal_year: Fiscal year

    Returns:
        True if exists, False otherwise
    """
    result = await session.execute(
        select(EarningsTranscript.id).where(
            EarningsTranscript.company_id == company_id,
            EarningsTranscript.fiscal_quarter == fiscal_quarter,
            EarningsTranscript.fiscal_year == fiscal_year,
        )
    )
    return result.scalar_one_or_none() is not None


async def upsert_transcript(session, transcript: dict) -> tuple[bool, int]:
    """Upsert earnings transcript into database.

    Args:
        session: SQLAlchemy async session
        transcript: Transcript dict with all fields

    Returns:
        Tuple of (was_inserted, transcript_id)
    """
    # Check for existing by company_id + quarter + year
    existing = await session.execute(
        select(EarningsTranscript.id).where(
            EarningsTranscript.company_id == transcript["company_id"],
            EarningsTranscript.fiscal_quarter == transcript["fiscal_quarter"],
            EarningsTranscript.fiscal_year == transcript["fiscal_year"],
        )
    )
    existing_id = existing.scalar_one_or_none()

    if existing_id:
        # Update existing transcript
        transcript_record = await session.get(EarningsTranscript, existing_id)
        if transcript_record:
            transcript_record.transcript_text = transcript["transcript_text"]
            transcript_record.source_url = transcript["source_url"]
            transcript_record.processed = False
        return False, existing_id

    # Insert new transcript
    stmt = insert(EarningsTranscript).values(transcript)
    await session.execute(stmt)

    # Get inserted ID
    result = await session.execute(
        select(EarningsTranscript.id).where(
            EarningsTranscript.company_id == transcript["company_id"],
            EarningsTranscript.fiscal_quarter == transcript["fiscal_quarter"],
            EarningsTranscript.fiscal_year == transcript["fiscal_year"],
        )
    )
    transcript_id = result.scalar_one()

    return True, transcript_id


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


async def run(tickers: list[str] | None = None):
    """Fetch earnings call transcripts and upsert into database.

    This function:
    1. For each ticker, searches for recent transcripts
    2. Fetches full transcript content
    3. Upserts into earnings_transcripts table
    4. Generates embeddings for new transcripts

    Rate limiting: 3 second delays between all requests.
    Only fetches last 8 quarters of data.

    Args:
        tickers: List of tickers to process. Defaults to TOP_25_TICKERS.

    Returns:
        Dict with ingestion statistics
    """
    logger.info("Starting earnings transcripts ingestion run")

    # Initialize database connection
    db.initialize()

    tickers = tickers or TOP_25_TICKERS
    client = MotleyFoolClient()

    total_transcripts_found = 0
    total_inserted = 0
    total_embedded = 0
    errors = []
    transcripts_to_embed = []

    async with db.async_session() as session:
        # Log start
        await log_ingestion(
            session,
            DataSource.EARNINGS_TRANSCRIPTS,
            IngestionStatus.RUNNING,
        )
        await session.commit()

        for ticker in tickers:
            logger.info(f"Processing ticker: {ticker}")

            try:
                # Fetch transcript list
                await asyncio.sleep(REQUEST_DELAY_SECONDS)
                transcript_links = await client.fetch_transcript_list(ticker)

                if not transcript_links:
                    logger.warning(f"  No transcripts found for {ticker}")
                    continue

                logger.info(f"  Found {len(transcript_links)} transcript links")

                # Get company ID
                company_id = await get_company_id_by_ticker(session, ticker)
                if not company_id:
                    logger.warning(f"  Company not in database: {ticker}")
                    continue

                # Process each transcript
                for link in transcript_links:
                    title = link.get("title", "")
                    url = link.get("url", "")

                    if not url:
                        continue

                    # Parse fiscal period from title
                    fiscal_quarter, fiscal_year = parse_fiscal_period(title)

                    if not fiscal_quarter or not fiscal_year:
                        logger.debug(f"  Could not parse fiscal period from: {title}")
                        continue

                    # Check if we already have this quarter
                    exists = await check_existing_transcript(
                        session, company_id, fiscal_quarter, fiscal_year
                    )
                    if exists:
                        logger.debug(f"  Transcript already exists for Q{fiscal_quarter} {fiscal_year}")
                        continue

                    # Check if within last 8 quarters
                    current_year = datetime.now().year
                    current_quarter = (datetime.now().month - 1) // 3 + 1

                    # Simple check: is it within last 2 years?
                    quarters_ago = (current_year - fiscal_year) * 4 + (current_quarter - fiscal_quarter)
                    if quarters_ago > MAX_QUARTERS_TO_FETCH or quarters_ago < 0:
                        logger.debug(f"  Skipping Q{fiscal_quarter} {fiscal_year} - outside range")
                        continue

                    total_transcripts_found += 1

                    # Fetch full transcript
                    try:
                        await asyncio.sleep(REQUEST_DELAY_SECONDS)
                        transcript_data = await client.fetch_transcript(url)

                        # Skip if no content
                        if not transcript_data.get("full_text"):
                            logger.warning(f"  No transcript content for {url}")
                            continue

                        # Truncate if too long
                        full_text = transcript_data["full_text"][:MAX_TRANSCRIPT_LENGTH]

                        # Parse published date
                        published_date = transcript_data.get("published_date", "")
                        if published_date:
                            try:
                                filed_at = datetime.fromisoformat(
                                    published_date.replace("Z", "+00:00")
                                )
                            except (ValueError, TypeError):
                                filed_at = datetime.utcnow()
                        else:
                            filed_at = datetime.utcnow()

                        # Build transcript record
                        transcript = {
                            "company_id": company_id,
                            "ticker": ticker.upper(),
                            "fiscal_quarter": fiscal_quarter,
                            "fiscal_year": fiscal_year,
                            "transcript_text": full_text,
                            "source_url": url,
                            "filed_at": filed_at,
                            "processed": False,
                        }

                        # Upsert
                        inserted, transcript_id = await upsert_transcript(
                            session, transcript
                        )

                        if inserted:
                            total_inserted += 1
                            transcripts_to_embed.append((
                                transcript_id,
                                f"{ticker} Q{fiscal_quarter} {fiscal_year}",
                                full_text,
                            ))
                            logger.info(
                                f"  Inserted: {ticker} Q{fiscal_quarter} {fiscal_year}"
                            )
                        else:
                            logger.debug(f"  Updated: {ticker} Q{fiscal_quarter} {fiscal_year}")

                    except httpx.HTTPStatusError as e:
                        if e.response.status_code == 404:
                            logger.debug(f"  404 for transcript: {url}")
                        else:
                            error_msg = f"HTTP {e.response.status_code} for {url}: {e}"
                            logger.error(error_msg)
                            errors.append(error_msg)
                    except httpx.HTTPError as e:
                        error_msg = f"HTTP error for {url}: {e}"
                        logger.error(error_msg)
                        errors.append(error_msg)
                    except Exception as e:
                        error_msg = f"Error processing {url}: {e}"
                        logger.error(error_msg)
                        errors.append(error_msg)

            except Exception as e:
                error_msg = f"Error processing ticker {ticker}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)

            # Commit after each ticker
            await session.commit()

        # Close client
        await client.close()

        # Generate embeddings for new transcripts
        if transcripts_to_embed:
            logger.info(f"Generating embeddings for {len(transcripts_to_embed)} new transcripts...")
            try:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer('all-MiniLM-L6-v2')

                for transcript_id, title, text in transcripts_to_embed:
                    try:
                        # Generate embedding
                        embedding = model.encode(text)

                        # Mark transcript as processed
                        transcript = await session.get(EarningsTranscript, transcript_id)
                        if transcript:
                            transcript.processed = True

                        total_embedded += 1
                    except Exception as e:
                        logger.error(f"Error generating embedding for {title}: {e}")
                        errors.append(f"Embedding error for {title}: {e}")

                await session.commit()
                logger.info(f"  Generated {total_embedded} embeddings")

            except Exception as e:
                error_msg = f"Error initializing embedder: {e}"
                logger.error(error_msg)
                errors.append(error_msg)

        # Log completion
        final_status = IngestionStatus.FAILED if errors else IngestionStatus.COMPLETED
        error_text = " | ".join(errors) if errors else None

        await log_ingestion(
            session,
            DataSource.EARNINGS_TRANSCRIPTS,
            final_status,
            records_fetched=total_inserted,
            errors=error_text,
        )
        await session.commit()

    logger.info(
        f"Earnings transcripts ingestion complete: "
        f"{total_transcripts_found} found, "
        f"{total_inserted} inserted, "
        f"{total_embedded} embedded, "
        f"{len(errors)} errors"
    )

    return {
        "total_transcripts_found": total_transcripts_found,
        "total_inserted": total_inserted,
        "total_embedded": total_embedded,
        "errors": errors,
    }


if __name__ == "__main__":
    asyncio.run(run())
