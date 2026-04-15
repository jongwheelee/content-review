"""Alpha Vantage ingestion module for market data (ETFs)."""

import asyncio
import logging
import os
from datetime import datetime
from typing import Any

import httpx
from dotenv import load_dotenv
from sqlalchemy.dialects.postgresql import insert

from database.connection import db
from database.models import DataPoint, DataSource, DataCategory, IngestionLog, IngestionStatus

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Major ETFs representing broad market segments
# These are better than individual stocks for fact-checking macro claims
ETF_TICKERS = [
    "SPY",   # S&P 500 ETF (broad market)
    "QQQ",   # NASDAQ-100 ETF (tech-heavy)
    "DIA",   # Dow Jones Industrial Average ETF
    "IWM",   # Russell 2000 ETF (small-cap)
    "GLD",   # Gold ETF (commodities)
    "TLT",   # 20+ Year Treasury Bond ETF (long-term bonds)
    "HYG",   # High Yield Corporate Bond ETF
    "LQD",   # Investment Grade Corporate Bond ETF
    "VNQ",   # Real Estate ETF
    "XLE",   # Energy Sector ETF
    "XLF",   # Financial Sector ETF
    "XLK",   # Technology Sector ETF
    "XLV",   # Health Care Sector ETF
]

# ETF descriptions for reference
ETF_NAMES = {
    "SPY": "SPDR S&P 500 ETF Trust",
    "QQQ": "Invesco QQQ Trust (NASDAQ-100)",
    "DIA": "SPDR Dow Jones Industrial Average ETF",
    "IWM": "iShares Russell 2000 ETF",
    "GLD": "SPDR Gold Shares",
    "TLT": "iShares 20+ Year Treasury Bond ETF",
    "HYG": "iShares iBoxx $ High Yield Corporate Bond ETF",
    "LQD": "iShares iBoxx $ Investment Grade Corporate Bond ETF",
    "VNQ": "Vanguard Real Estate ETF",
    "XLE": "Energy Select Sector SPDR Fund",
    "XLF": "Financial Select Sector SPDR Fund",
    "XLK": "Technology Select Sector SPDR Fund",
    "XLV": "Health Care Select Sector SPDR Fund",
}


class AlphaVantageClient:
    """Client for Alpha Vantage API market data ingestion.

    Free tier limits:
    - 5 API calls per minute
    - 500 API calls per day
    """

    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY")
        if not self.api_key:
            raise ValueError("ALPHA_VANTAGE_API_KEY environment variable is required")

    async def fetch_daily(
        self, symbol: str, outputsize: str = "compact"
    ) -> dict[str, Any]:
        """Fetch daily time series for a symbol.

        Note: Uses TIME_SERIES_DAILY (free tier) instead of
        TIME_SERIES_DAILY_ADJUSTED (premium). Uses close price instead of adjusted close.

        Args:
            symbol: ETF ticker symbol
            outputsize: 'compact' (100 data points) or 'full' (20+ years)

        Returns:
            JSON response from Alpha Vantage API
        """
        async with httpx.AsyncClient() as client:
            params = {
                "function": "TIME_SERIES_DAILY",
                "symbol": symbol,
                "apikey": self.api_key,
                "outputsize": outputsize,
            }

            response = await client.get(
                self.BASE_URL, params=params, timeout=30.0
            )
            response.raise_for_status()
            return response.json()

    async def fetch_weekly_adjusted(self, symbol: str) -> dict[str, Any]:
        """Fetch weekly adjusted time series for a symbol.

        Args:
            symbol: ETF ticker symbol

        Returns:
            JSON response from Alpha Vantage API
        """
        async with httpx.AsyncClient() as client:
            params = {
                "function": "TIME_SERIES_WEEKLY_ADJUSTED",
                "symbol": symbol,
                "apikey": self.api_key,
            }

            response = await client.get(
                self.BASE_URL, params=params, timeout=30.0
            )
            response.raise_for_status()
            return response.json()

    async def fetch_quote(self, symbol: str) -> dict[str, Any]:
        """Fetch current quote for a symbol.

        Args:
            symbol: ETF ticker symbol

        Returns:
            JSON response with current quote
        """
        async with httpx.AsyncClient() as client:
            params = {
                "function": "GLOBAL_QUOTE",
                "symbol": symbol,
                "apikey": self.api_key,
            }

            response = await client.get(
                self.BASE_URL, params=params, timeout=30.0
            )
            response.raise_for_status()
            return response.json()


def parse_time_series(response: dict[str, Any], time_key: str, use_adjusted: bool = True) -> list[dict]:
    """Parse Alpha Vantage time series response.

    Args:
        response: Raw API response
        time_key: Key for time series data (e.g., 'Time Series (Daily)')
        use_adjusted: Use adjusted close (True) or regular close (False)

    Returns:
        List of dicts with date and value (close price)
    """
    observations = []

    time_series = response.get(time_key, {})

    for date, data in time_series.items():
        # Use adjusted close if available, otherwise use regular close
        if use_adjusted:
            close_value = data.get("5. adjusted close")
        else:
            close_value = data.get("4. close")

        # Skip missing values
        if close_value is None:
            continue

        try:
            value = float(close_value)
        except (ValueError, TypeError):
            continue

        observations.append({
            "date": date,
            "value": value,
            "open": float(data.get("1. open", 0) or 0),
            "high": float(data.get("2. high", 0) or 0),
            "low": float(data.get("3. low", 0) or 0),
            "close": float(data.get("4. close", 0) or 0),
            "volume": int(data.get("6. volume", 0) or 0),
        })

    return observations


def filter_last_n_years(observations: list[dict], years: int = 2) -> list[dict]:
    """Filter observations to last N years.

    Args:
        observations: List of observation dicts with 'date' key
        years: Number of years to keep

    Returns:
        Filtered list of observations
    """
    cutoff_date = datetime.now().replace(year=datetime.now().year - years)
    cutoff_str = cutoff_date.strftime("%Y-%m-%d")

    return [obs for obs in observations if obs["date"] >= cutoff_str]


async def upsert_data_points(
    session,
    ticker: str,
    observations: list[dict],
    category: DataCategory,
    raw_response: dict,
) -> int:
    """Upsert market data observations into data_points table.

    Args:
        session: SQLAlchemy async session
        ticker: ETF ticker symbol
        observations: List of observation dicts
        category: Data category (market_price or market_price_weekly)
        raw_response: Raw API response for storage

    Returns:
        Number of records upserted
    """
    if not observations:
        return 0

    records = []
    for obs in observations:
        # Build metadata JSON
        metadata = {
            "open": obs.get("open"),
            "high": obs.get("high"),
            "low": obs.get("low"),
            "close": obs.get("close"),
            "volume": obs.get("volume"),
            "etf_name": ETF_NAMES.get(ticker, ""),
            "data_type": "daily" if category == DataCategory.MARKET else "weekly",
        }

        records.append(
            {
                "source": DataSource.ALPHA_VANTAGE,
                "category": category,
                "metric_name": ticker,
                "value": obs["value"],  # adjusted close
                "unit": "USD",
                "date_recorded": datetime.strptime(obs["date"], "%Y-%m-%d").date(),
                "geographic_scope": "US Market",
                "confidence_score": 1.0,
                "raw_json": metadata,  # Store OHLCV in raw_json
            }
        )

    if not records:
        return 0

    # Use PostgreSQL INSERT ... ON CONFLICT for upsert
    stmt = insert(DataPoint).values(records)
    stmt = stmt.on_conflict_do_update(
        index_elements=["source", "metric_name", "date_recorded"],
        set_={
            "value": stmt.excluded.value,
            "raw_json": stmt.excluded.raw_json,
            "updated_at": datetime.utcnow(),
        },
    )

    await session.execute(stmt)
    return len(records)


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
    """Fetch market data for all ETFs and upsert into database.

    This is the main entry point for Alpha Vantage data ingestion.
    Fetches:
    1. Daily adjusted close prices for last 2 years
    2. Weekly adjusted prices for all available history

    Rate limiting: 13 second delays between calls (max 5/minute for free tier).

    Args:
        tickers: List of ETF tickers to fetch. Defaults to ETF_TICKERS.

    Returns:
        Dict with ingestion statistics
    """
    logger.info("Starting Alpha Vantage ingestion run")

    # Initialize database connection
    db.initialize()

    tickers = tickers or ETF_TICKERS
    client = AlphaVantageClient()

    total_daily_records = 0
    total_weekly_records = 0
    errors = []
    rate_limit_hits = 0

    async with db.async_session() as session:
        # Log start
        await log_ingestion(
            session,
            DataSource.ALPHA_VANTAGE,
            IngestionStatus.RUNNING,
        )
        await session.commit()

        for ticker in tickers:
            logger.info(
                f"Fetching ticker: {ticker} ({ETF_NAMES[ticker]})"
            )

            try:
                # Fetch daily prices (free tier: TIME_SERIES_DAILY)
                logger.info(f"  Fetching daily data for {ticker}...")
                daily_response = await client.fetch_daily(ticker)

                # Check for API errors
                if "Error Message" in daily_response:
                    error_msg = f"Alpha Vantage error for {ticker}: {daily_response['Error Message']}"
                    logger.error(error_msg)
                    errors.append(error_msg)
                    await asyncio.sleep(13)  # Still rate limit
                    continue

                if "Note" in daily_response:
                    # Rate limit message
                    note = daily_response["Note"]
                    if "rate limit" in note.lower() or "call" in note.lower():
                        logger.warning(f"Rate limit hit for {ticker}: {note}")
                        rate_limit_hits += 1
                        errors.append(f"Rate limit for {ticker}: {note}")
                        await asyncio.sleep(13)
                        continue

                # Parse and filter daily data (last 2 years)
                # Free tier uses regular close (not adjusted)
                daily_obs = parse_time_series(daily_response, "Time Series (Daily)", use_adjusted=False)
                daily_obs = filter_last_n_years(daily_obs, years=2)

                if daily_obs:
                    daily_count = await upsert_data_points(
                        session, ticker, daily_obs, DataCategory.MARKET, daily_response
                    )
                    total_daily_records += daily_count
                    logger.info(f"  Upserted {daily_count} daily records for {ticker}")

                # Rate limit delay
                await asyncio.sleep(13)

                # Fetch weekly adjusted prices
                logger.info(f"  Fetching weekly data for {ticker}...")
                weekly_response = await client.fetch_weekly_adjusted(ticker)

                # Check for API errors
                if "Error Message" in weekly_response:
                    logger.warning(f"Weekly data error for {ticker}: {weekly_response['Error Message']}")
                elif "Note" in weekly_response:
                    logger.warning(f"Weekly data rate limit for {ticker}")
                else:
                    # Parse weekly data (all available history)
                    weekly_obs = parse_time_series(weekly_response, "Weekly Adjusted Time Series")

                    if weekly_obs:
                        weekly_count = await upsert_data_points(
                            session, ticker, weekly_obs, DataCategory.MARKET, weekly_response
                        )
                        total_weekly_records += weekly_count
                        logger.info(f"  Upserted {weekly_count} weekly records for {ticker}")

            except httpx.HTTPStatusError as e:
                error_msg = f"HTTP {e.response.status_code} for {ticker}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
            except httpx.HTTPError as e:
                error_msg = f"HTTP error for {ticker}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
            except Exception as e:
                error_msg = f"Error processing {ticker}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)

            # Rate limit delay after each ticker (both daily and weekly calls)
            await asyncio.sleep(13)

            # Commit after each ticker
            await session.commit()

        # Log completion
        final_status = IngestionStatus.FAILED if errors else IngestionStatus.COMPLETED
        error_text = " | ".join(errors) if errors else None

        total_records = total_daily_records + total_weekly_records
        await log_ingestion(
            session,
            DataSource.ALPHA_VANTAGE,
            final_status,
            records_fetched=total_records,
            errors=error_text,
        )
        await session.commit()

    logger.info(
        f"Alpha Vantage ingestion complete: "
        f"{total_daily_records} daily + {total_weekly_records} weekly = {total_records} total records, "
        f"{len(errors)} errors, "
        f"{rate_limit_hits} rate limit hits"
    )

    return {
        "total_daily_records": total_daily_records,
        "total_weekly_records": total_weekly_records,
        "total_records": total_records,
        "errors": errors,
        "rate_limit_hits": rate_limit_hits,
    }


if __name__ == "__main__":
    asyncio.run(run())
