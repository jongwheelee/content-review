"""NewsAPI ingestion module for real-time financial news."""

import asyncio
import logging
import os
import re
from datetime import datetime, timedelta
from typing import Any

import httpx
from dotenv import load_dotenv
from sqlalchemy import select, func
from sqlalchemy.dialects.postgresql import insert

from database.connection import db
from database.models import NewsArticle, Company, IngestionLog, IngestionStatus, DataSource

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Topic-specific queries for financial news
FINANCIAL_QUERIES = [
    "Federal Reserve interest rates",
    "S&P 500 earnings",
    "inflation CPI",
    "GDP growth",
    "unemployment jobs report",
    "corporate earnings beat miss",
    "recession economic outlook",
]

# Topic tags for each query
QUERY_TOPICS = {
    "Federal Reserve interest rates": ["federal_reserve", "interest_rates", "monetary_policy"],
    "S&P 500 earnings": ["earnings", "sp500", "corporate"],
    "inflation CPI": ["inflation", "cpi", "economic_data"],
    "GDP growth": ["gdp", "economic_growth", "economic_data"],
    "unemployment jobs report": ["unemployment", "jobs", "labor_market"],
    "corporate earnings beat miss": ["earnings", "corporate", "guidance"],
    "recession economic outlook": ["recession", "economic_outlook", "forecast"],
}

# Rate limit tracking
DAILY_REQUEST_LIMIT = 100
DAILY_REQUEST_WARNING_THRESHOLD = 90


class NewsAPIClient:
    """Client for NewsAPI.org data ingestion.

    Rate limits (free tier):
    - 100 requests per day
    - Top headlines: 100 articles per request
    - Everything: 50 articles per request
    """

    BASE_URL = "https://newsapi.org/v2"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("NEWS_API_KEY")
        if not self.api_key:
            raise ValueError("NEWS_API_KEY environment variable is required")

    async def fetch_top_headlines(
        self,
        category: str = "business",
        country: str = "us",
        language: str = "en",
        page_size: int = 100,
    ) -> dict[str, Any]:
        """Fetch top headlines by category.

        Args:
            category: News category (business, technology, etc.)
            country: Country code (us, gb, etc.)
            language: Language code (en, es, etc.)
            page_size: Number of articles (max 100)

        Returns:
            JSON response from NewsAPI
        """
        async with httpx.AsyncClient() as client:
            params = {
                "apiKey": self.api_key,
                "category": category,
                "country": country,
                "language": language,
                "pageSize": page_size,
            }

            response = await client.get(
                f"{self.BASE_URL}/top-headlines",
                params=params,
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json()

    async def fetch_everything(
        self,
        query: str,
        from_date: str | None = None,
        to_date: str | None = None,
        language: str = "en",
        sort_by: str = "publishedAt",
        page_size: int = 50,
    ) -> dict[str, Any]:
        """Fetch articles matching a search query.

        Args:
            query: Search query string
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            language: Language code
            sort_by: Sort order (publishedAt, relevance, popularity)
            page_size: Number of articles (max 50)

        Returns:
            JSON response from NewsAPI
        """
        async with httpx.AsyncClient() as client:
            params = {
                "apiKey": self.api_key,
                "q": query,
                "language": language,
                "sortBy": sort_by,
                "pageSize": page_size,
            }

            if from_date:
                params["from"] = from_date
            if to_date:
                params["to"] = to_date

            response = await client.get(
                f"{self.BASE_URL}/everything",
                params=params,
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json()


def extract_tickers_from_title(title: str, known_tickers: set[str]) -> list[str]:
    """Extract potential ticker symbols from article title.

    Looks for uppercase 1-5 character words that match known tickers.

    Args:
        title: Article title
        known_tickers: Set of known ticker symbols from database

    Returns:
        List of matched ticker symbols
    """
    # Find all uppercase 1-5 character words
    potential_tickers = re.findall(r"\b[A-Z]{1,5}\b", title)

    # Filter to known tickers
    matched = [t for t in potential_tickers if t in known_tickers]

    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for ticker in matched:
        if ticker not in seen:
            seen.add(ticker)
            unique.append(ticker)

    return unique


def get_topics_for_query(query: str) -> list[str]:
    """Get topic tags for a search query."""
    return QUERY_TOPICS.get(query, ["general"])


async def get_known_tickers(session) -> set[str]:
    """Fetch all known tickers from the database."""
    result = await session.execute(select(Company.ticker).where(Company.ticker.isnot(None)))
    tickers = result.scalars().all()

    # Add common ETFs and indices that might not be in companies table
    common_tickers = {
        "SPY", "QQQ", "DIA", "IWM", "GLD", "TLT", "HYG", "LQD", "VNQ",
        "XLE", "XLF", "XLK", "XLV", "VIX", "DXY",
    }

    return set(tickers) | common_tickers


async def upsert_news_article(session, article: dict, known_tickers: set[str]) -> tuple[bool, int]:
    """Upsert a single news article.

    Args:
        session: SQLAlchemy async session
        article: Article dict from NewsAPI
        known_tickers: Set of known ticker symbols

    Returns:
        Tuple of (was_inserted, record_count)
    """
    # Extract metadata
    title = article.get("title", "")
    if not title:
        return False, 0

    # Skip if no URL (can't deduplicate)
    url = article.get("url", "")
    if not url:
        return False, 0

    # Extract tickers from title
    tickers_mentioned = extract_tickers_from_title(title, known_tickers)

    # Get content summary (first 500 chars)
    content = article.get("content", "") or ""
    content_summary = content[:500] if content else ""

    # Parse published date
    published_at_str = article.get("publishedAt")
    if published_at_str:
        try:
            published_at = datetime.fromisoformat(published_at_str.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            published_at = datetime.utcnow()
    else:
        published_at = datetime.utcnow()

    # Build record
    record = {
        "headline": title,
        "source_name": article.get("source", {}).get("name", ""),
        "url": url,
        "published_at": published_at,
        "content_summary": content_summary,
        "tickers_mentioned": tickers_mentioned if tickers_mentioned else None,
        "topics": None,  # Will be set by caller
        "sentiment_score": None,  # Could be added with sentiment analysis
        "raw_json": article,
    }

    # Check for existing URL (deduplication)
    existing = await session.execute(
        select(NewsArticle.id).where(NewsArticle.url == url)
    )
    existing_id = existing.scalar_one_or_none()

    if existing_id:
        # Skip - already exists
        return False, 0

    # Insert new article
    stmt = insert(NewsArticle).values(record)
    await session.execute(stmt)

    return True, 1


async def log_ingestion(
    session,
    source: DataSource,
    status: IngestionStatus,
    records_fetched: int = 0,
    errors: str | None = None,
    api_requests_used: int = 0,
):
    """Log ingestion run to ingestion_log table."""
    # Check current daily usage
    today = datetime.utcnow().date()
    result = await session.execute(
        select(func.sum(IngestionLog.records_fetched))
        .where(IngestionLog.source == source)
        .where(func.date(IngestionLog.ran_at) == today)
    )
    current_usage = result.scalar() or 0

    log_entry = IngestionLog(
        source=source,
        status=status,
        records_fetched=records_fetched,
        errors=errors,
    )
    session.add(log_entry)

    return current_usage + 1  # Return new usage count


async def check_daily_usage(session) -> int:
    """Check how many API requests have been used today."""
    today = datetime.utcnow().date()
    result = await session.execute(
        select(func.count(IngestionLog.id))
        .where(IngestionLog.source == DataSource.NEWS)
        .where(func.date(IngestionLog.ran_at) == today)
    )
    return result.scalar() or 0


async def run(
    fetch_headlines: bool = True,
    fetch_topic_queries: bool = False,
    queries: list[str] | None = None,
):
    """Fetch news from NewsAPI and upsert into database.

    This function supports two modes:
    1. Top headlines (every 6 hours) - uses 1 API request
    2. Topic-specific queries (daily) - uses 1 request per query

    Rate limit tracking:
    - Free tier: 100 requests/day
    - Stops if usage > 90 to avoid hitting limit

    Args:
        fetch_headlines: Whether to fetch top headlines
        fetch_topic_queries: Whether to fetch topic-specific articles
        queries: List of queries to fetch (defaults to FINANCIAL_QUERIES)

    Returns:
        Dict with ingestion statistics
    """
    logger.info("Starting NewsAPI ingestion run")

    # Initialize database connection
    db.initialize()

    client = NewsAPIClient()
    queries = queries or FINANCIAL_QUERIES

    total_articles = 0
    total_requests = 0
    errors = []
    usage_warning = False

    async with db.async_session() as session:
        # Check daily usage before starting
        daily_usage = await check_daily_usage(session)
        logger.info(f"Daily API requests used: {daily_usage}/{DAILY_REQUEST_LIMIT}")

        if daily_usage >= DAILY_REQUEST_WARNING_THRESHOLD:
            warning_msg = (
                f"Daily API usage ({daily_usage}) exceeds threshold ({DAILY_REQUEST_WARNING_THRESHOLD}). "
                f"Stopping to avoid hitting limit of {DAILY_REQUEST_LIMIT}."
            )
            logger.warning(warning_msg)

            await log_ingestion(
                session,
                DataSource.NEWS,
                IngestionStatus.FAILED,
                errors=warning_msg,
            )
            await session.commit()

            return {
                "error": warning_msg,
                "daily_usage": daily_usage,
                "articles_fetched": 0,
            }

        # Get known tickers for extraction
        known_tickers = await get_known_tickers(session)
        logger.info(f"Known tickers for matching: {len(known_tickers)}")

        # Log start
        await log_ingestion(
            session,
            DataSource.NEWS,
            IngestionStatus.RUNNING,
        )
        await session.commit()

        # Task 1: Fetch top headlines
        if fetch_headlines:
            logger.info("=== Fetching Top Headlines ===")

            try:
                response = await client.fetch_top_headlines(
                    category="business",
                    page_size=100,
                )

                total_requests += 1

                if response.get("status") == "ok":
                    articles = response.get("articles", [])
                    logger.info(f"  Received {len(articles)} articles")

                    for article in articles:
                        try:
                            inserted, count = await upsert_news_article(
                                session, article, known_tickers
                            )
                            if inserted:
                                total_articles += count
                        except Exception as e:
                            logger.error(f"    Error processing article: {e}")

                    await session.commit()
                    logger.info(f"  Inserted {total_articles} new articles")
                else:
                    error_msg = response.get("message", "Unknown error")
                    logger.error(f"  API error: {error_msg}")
                    errors.append(f"Headlines error: {error_msg}")

            except httpx.HTTPError as e:
                error_msg = f"HTTP error fetching headlines: {e}"
                logger.error(error_msg)
                errors.append(error_msg)

        # Task 2: Fetch topic-specific articles
        if fetch_topic_queries:
            logger.info("=== Fetching Topic-Specific Articles ===")

            # Calculate date range (last 24 hours for everything query)
            to_date = datetime.utcnow()
            from_date = to_date - timedelta(hours=24)
            from_date_str = from_date.strftime("%Y-%m-%dT%H:%M:%S")
            to_date_str = to_date.strftime("%Y-%m-%dT%H:%M:%S")

            for query in queries:
                # Check usage before each query
                daily_usage = await check_daily_usage(session) + total_requests

                if daily_usage >= DAILY_REQUEST_WARNING_THRESHOLD:
                    logger.warning(
                        f"Stopping topic queries - usage ({daily_usage}) near limit"
                    )
                    usage_warning = True
                    break

                logger.info(f"Fetching: {query}")
                topics = get_topics_for_query(query)

                try:
                    response = await client.fetch_everything(
                        query=query,
                        from_date=from_date_str,
                        to_date=to_date_str,
                        page_size=50,
                    )

                    total_requests += 1
                    query_articles = 0

                    if response.get("status") == "ok":
                        articles = response.get("articles", [])
                        logger.info(f"  Received {len(articles)} articles")

                        for article in articles:
                            try:
                                inserted, count = await upsert_news_article(
                                    session, article, known_tickers
                                )
                                if inserted:
                                    # Update topics for this article
                                    article_record = await session.execute(
                                        select(NewsArticle).where(NewsArticle.url == article.get("url"))
                                    )
                                    article_obj = article_record.scalar_one_or_none()
                                    if article_obj:
                                        article_obj.topics = topics
                                    query_articles += count
                            except Exception as e:
                                logger.error(f"    Error processing article: {e}")

                        await session.commit()
                        logger.info(f"  Inserted {query_articles} new articles for '{query}'")
                    else:
                        error_msg = response.get("message", "Unknown error")
                        logger.error(f"  API error: {error_msg}")
                        errors.append(f"Query '{query}' error: {error_msg}")

                except httpx.HTTPError as e:
                    error_msg = f"HTTP error for query '{query}': {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)

                # Small delay between queries (not rate limited, but be polite)
                await asyncio.sleep(1)

        # Log completion
        final_status = IngestionStatus.FAILED if errors else IngestionStatus.COMPLETED
        error_text = " | ".join(errors) if errors else None

        if usage_warning:
            error_text = (error_text + " | " if error_text else "") + "Stopped near daily limit"

        await log_ingestion(
            session,
            DataSource.NEWS,
            final_status,
            records_fetched=total_articles,
            errors=error_text,
        )
        await session.commit()

    logger.info(
        f"NewsAPI ingestion complete: "
        f"{total_articles} articles inserted, "
        f"{total_requests} API requests used"
    )

    return {
        "total_articles": total_articles,
        "total_requests": total_requests,
        "errors": errors,
        "usage_warning": usage_warning,
    }


if __name__ == "__main__":
    # Default: fetch headlines only (for frequent runs)
    # Use fetch_topic_queries=True for daily runs
    asyncio.run(run(fetch_headlines=True, fetch_topic_queries=False))
