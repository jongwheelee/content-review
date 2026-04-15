"""Polygon.io ingestion module for market data and ticker details."""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Any

import httpx
from dotenv import load_dotenv
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from database.connection import db
from database.models import Company, MarketData, DataPoint, DataSource, DataCategory, IngestionLog, IngestionStatus

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Tickers for daily OHLCV bars
# ETFs + largest individual stocks by market cap
OHLCV_TICKERS = [
    # ETFs
    "SPY", "QQQ", "DIA", "IWM", "GLD", "TLT", "HYG", "LQD", "VNQ",
    "XLE", "XLF", "XLK", "XLV",
    # Mega-cap stocks
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL",
    "BRK.B", "JPM", "JNJ", "XOM", "UNH", "V",
]

# Tickers for options flow (most liquid)
OPTIONS_TICKERS = ["SPY", "QQQ"]


class PolygonClient:
    """Client for Polygon.io API data ingestion.

    Rate limits:
    - Free tier: 5 calls/minute
    - Paid tier: Unlimited (or much higher limits)
    """

    BASE_URL = "https://api.polygon.io"

    def __init__(self, api_key: str | None = None, tier: str | None = None):
        self.api_key = api_key or os.getenv("POLYGON_API_KEY")
        if not self.api_key:
            raise ValueError("POLYGON_API_KEY environment variable is required")

        self.tier = tier or os.getenv("POLYGON_TIER", "free")
        self.is_paid = self.tier.lower() == "paid"

    def _get_headers(self) -> dict[str, str]:
        """Get headers for Polygon API requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }

    async def fetch_aggregates(
        self,
        ticker: str,
        multiplier: int = 1,
        timespan: str = "day",
        from_date: str = "2024-01-01",
        to_date: str = None,
    ) -> list[dict]:
        """Fetch aggregate bars (OHLCV) for a ticker.

        Handles pagination via next_url cursor.

        Args:
            ticker: Ticker symbol
            multiplier: Size of the timespan (1 for daily)
            timespan: Timespan (minute, hour, day, week, month, quarter, year)
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD), defaults to today

        Returns:
            List of aggregate bar dicts
        """
        if to_date is None:
            to_date = datetime.now().strftime("%Y-%m-%d")

        all_results = []
        url = f"{self.BASE_URL}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"

        async with httpx.AsyncClient(headers=self._get_headers()) as client:
            while url:
                params = {
                    "adjusted": "true",
                    "sort": "asc",
                    "limit": 50000,
                }

                # If url contains query params, parse them
                if "?" in url and url.split("?")[0] != f"{self.BASE_URL}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}":
                    response = await client.get(url, timeout=30.0)
                else:
                    response = await client.get(url, params=params, timeout=30.0)

                response.raise_for_status()
                data = response.json()

                results = data.get("results", [])
                all_results.extend(results)

                # Check for pagination
                next_url = data.get("next_url")
                if next_url:
                    url = f"{self.BASE_URL}{next_url}"
                else:
                    url = None

        return all_results

    async def fetch_ticker_details(self, ticker: str) -> dict[str, Any]:
        """Fetch detailed information for a ticker.

        Args:
            ticker: Ticker symbol

        Returns:
            Ticker details dict
        """
        async with httpx.AsyncClient(headers=self._get_headers()) as client:
            response = await client.get(
                f"{self.BASE_URL}/v3/reference/tickers/{ticker}",
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json()

    async def fetch_options_snapshot(self, ticker: str) -> dict[str, Any]:
        """Fetch options snapshot for a ticker.

        Args:
            ticker: Ticker symbol

        Returns:
            Options snapshot data
        """
        async with httpx.AsyncClient(headers=self._get_headers()) as client:
            response = await client.get(
                f"{self.BASE_URL}/v3/snapshot/options/{ticker}",
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json()

    async def fetch_unusual_options(self) -> dict[str, Any]:
        """Fetch unusual options activity across all tickers.

        Returns:
            Unusual options activity data
        """
        async with httpx.AsyncClient(headers=self._get_headers()) as client:
            response = await client.get(
                f"{self.BASE_URL}/v3/snapshot/options/unusual",
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json()


def parse_aggregates(ticker: str, aggregates: list[dict]) -> list[dict]:
    """Parse Polygon aggregates into market data records.

    Args:
        ticker: Ticker symbol
        aggregates: List of aggregate bar dicts

    Returns:
        List of market data dicts
    """
    records = []

    for agg in aggregates:
        # Skip if no close price
        close = agg.get("c")
        if close is None:
            continue

        try:
            close = float(close)
        except (ValueError, TypeError):
            continue

        records.append({
            "ticker": ticker,
            "date_recorded": datetime.fromtimestamp(agg.get("t", 0) / 1000).date(),
            "open": float(agg.get("o", 0) or 0),
            "high": float(agg.get("h", 0) or 0),
            "low": float(agg.get("l", 0) or 0),
            "close": close,
            "volume": int(agg.get("v", 0) or 0),
            "adjusted_close": close,  # Using 'c' field as adjusted close
            "vwap": float(agg.get("vw", 0) or 0),
            "source": "POLYGON",
        })

    return records


async def upsert_market_data(session, records: list[dict]) -> int:
    """Upsert market data records into market_data table.

    Args:
        session: SQLAlchemy async session
        records: List of market data dicts

    Returns:
        Number of records upserted
    """
    if not records:
        return 0

    stmt = insert(MarketData).values(records)
    stmt = stmt.on_conflict_do_update(
        index_elements=["ticker", "date_recorded", "source"],
        set_={
            "open": stmt.excluded.open,
            "high": stmt.excluded.high,
            "low": stmt.excluded.low,
            "close": stmt.excluded.close,
            "volume": stmt.excluded.volume,
            "adjusted_close": stmt.excluded.adjusted_close,
            "vwap": stmt.excluded.vwap,
            "updated_at": datetime.utcnow(),
        },
    )

    await session.execute(stmt)
    return len(records)


async def upsert_company(session, ticker: str, details: dict) -> bool:
    """Upsert company details into companies table.

    Args:
        session: SQLAlchemy async session
        ticker: Ticker symbol
        details: Company details from Polygon API

    Returns:
        True if successful
    """
    # Map Polygon fields to Company model
    company_data = {
        "cik": details.get("cik", ""),
        "ticker": ticker,
        "name": details.get("name", details.get("ticker_name", "")),
        "sic_code": details.get("sic_code"),
        "sector": details.get("sic_code"),  # Polygon doesn't have direct sector
        "exchange": details.get("primary_exchange", ""),
        "market_cap": details.get("market_cap"),
    }

    # Remove empty cik (Polygon doesn't always provide CIK)
    if not company_data["cik"]:
        del company_data["cik"]

    stmt = insert(Company).values(company_data)
    stmt = stmt.on_conflict_do_update(
        index_elements=["ticker"],
        set_={
            "name": stmt.excluded.name,
            "market_cap": stmt.excluded.market_cap,
            "exchange": stmt.excluded.exchange,
            "updated_at": datetime.utcnow(),
        },
    )

    await session.execute(stmt)
    return True


async def upsert_options_flow(
    session, ticker: str, options_data: dict, raw_response: dict
) -> int:
    """Upsert options flow data into data_points table.

    Args:
        session: SQLAlchemy async session
        ticker: Ticker symbol
        options_data: Parsed options data
        raw_response: Raw API response

    Returns:
        Number of records upserted
    """
    records = []

    # Process each options contract
    for contract in options_data:
        records.append(
            {
                "source": DataSource.POLYGON,
                "category": DataCategory.MARKET,
                "metric_name": ticker,
                "value": contract.get("last_price", 0),
                "unit": "USD",
                "date_recorded": datetime.now().strftime("%Y-%m-%d"),
                "geographic_scope": "US Options Market",
                "confidence_score": 1.0,
                "raw_json": raw_response,
            }
        )

    if not records:
        return 0

    stmt = insert(DataPoint).values(records)
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


async def run(
    ohlcv_tickers: list[str] | None = None,
    options_tickers: list[str] | None = None,
):
    """Fetch market data from Polygon.io and upsert into database.

    This function performs three tasks:
    1. Fetch daily OHLCV bars (2 years) for all OHLCV_TICKERS
    2. Fetch ticker details and update companies table
    3. Fetch options flow snapshots for SPY and QQQ

    Rate limiting:
    - Free tier: 13 second delay between calls (5/minute limit)
    - Paid tier: No delay

    Args:
        ohlcv_tickers: List of tickers for OHLCV data. Defaults to OHLCV_TICKERS.
        options_tickers: List of tickers for options data. Defaults to OPTIONS_TICKERS.

    Returns:
        Dict with ingestion statistics
    """
    logger.info("Starting Polygon.io ingestion run")

    # Initialize database connection
    db.initialize()

    ohlcv_tickers = ohlcv_tickers or OHLCV_TICKERS
    options_tickers = options_tickers or OPTIONS_TICKERS

    client = PolygonClient()
    logger.info(f"Polygon tier: {'paid' if client.is_paid else 'free'}")

    # Calculate date range (2 years)
    to_date = datetime.now()
    from_date = to_date - timedelta(days=365 * 2)
    from_date_str = from_date.strftime("%Y-%m-%d")
    to_date_str = to_date.strftime("%Y-%m-%d")

    total_market_records = 0
    total_companies = 0
    total_options = 0
    errors = []

    async with db.async_session() as session:
        # Log start
        await log_ingestion(
            session,
            DataSource.POLYGON,
            IngestionStatus.RUNNING,
        )
        await session.commit()

        # Task 1: Fetch daily OHLCV bars
        logger.info("=== Fetching Daily OHLCV Bars ===")
        for ticker in ohlcv_tickers:
            logger.info(f"Fetching OHLCV for {ticker}...")

            try:
                aggregates = await client.fetch_aggregates(
                    ticker=ticker,
                    from_date=from_date_str,
                    to_date=to_date_str,
                )

                if aggregates:
                    records = parse_aggregates(ticker, aggregates)
                    count = await upsert_market_data(session, records)
                    total_market_records += count
                    logger.info(f"  Upserted {count} records for {ticker}")
                else:
                    logger.warning(f"  No data returned for {ticker}")

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    logger.warning(f"  404 for ticker {ticker} - may not exist on Polygon")
                else:
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

            # Rate limiting
            if not client.is_paid:
                await asyncio.sleep(13)

            await session.commit()

        # Task 2: Fetch ticker details
        logger.info("=== Fetching Ticker Details ===")
        for ticker in ohlcv_tickers:
            logger.info(f"Fetching details for {ticker}...")

            try:
                details = await client.fetch_ticker_details(ticker)

                if details.get("status") == "OK":
                    await upsert_company(session, ticker, details.get("results", {}))
                    total_companies += 1
                    logger.info(f"  Updated company info for {ticker}")
                else:
                    logger.warning(f"  API error for {ticker}: {details.get('error')}")

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    logger.warning(f"  404 for ticker {ticker}")
                else:
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

            # Rate limiting
            if not client.is_paid:
                await asyncio.sleep(13)

            await session.commit()

        # Task 3: Fetch options snapshots
        logger.info("=== Fetching Options Snapshots ===")
        for ticker in options_tickers:
            logger.info(f"Fetching options snapshot for {ticker}...")

            try:
                snapshot = await client.fetch_options_snapshot(ticker)

                if snapshot.get("status") == "OK":
                    results = snapshot.get("results", [])

                    # Get top 10 by unusual activity (simplified: just top 10)
                    top_options = results[:10] if results else []

                    if top_options:
                        count = await upsert_options_flow(
                            session, ticker, top_options, snapshot
                        )
                        total_options += count
                        logger.info(f"  Stored {count} options contracts for {ticker}")
                    else:
                        logger.warning(f"  No options data for {ticker}")
                else:
                    logger.warning(f"  API error for {ticker}: {snapshot.get('error')}")

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    logger.warning(f"  404 for options snapshot {ticker}")
                else:
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

            # Rate limiting
            if not client.is_paid:
                await asyncio.sleep(13)

            await session.commit()

        # Log completion
        final_status = IngestionStatus.FAILED if errors else IngestionStatus.COMPLETED
        error_text = " | ".join(errors) if errors else None

        total_records = total_market_records + total_companies + total_options
        await log_ingestion(
            session,
            DataSource.POLYGON,
            final_status,
            records_fetched=total_records,
            errors=error_text,
        )
        await session.commit()

    logger.info(
        f"Polygon.io ingestion complete: "
        f"{total_market_records} market records, "
        f"{total_companies} companies, "
        f"{total_options} options contracts, "
        f"{len(errors)} errors"
    )

    return {
        "total_market_records": total_market_records,
        "total_companies": total_companies,
        "total_options": total_options,
        "errors": errors,
    }


if __name__ == "__main__":
    asyncio.run(run())
