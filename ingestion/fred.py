"""FRED (Federal Reserve Economic Data) ingestion module."""

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
from database.models import DataPoint, DataSource, DataCategory, IngestionLog, IngestionStatus

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Series ID to human-readable name mapping
SERIES_NAMES = {
    "GDP": "Gross Domestic Product",
    "GDPC1": "Real Gross Domestic Product",
    "CPIAUCSL": "Consumer Price Index for All Urban Consumers: All Items",
    "CPILFESL": "Consumer Price Index for All Urban Consumers: All Items Less Food and Energy",
    "PCEPI": "Personal Consumption Expenditures Price Index",
    "FEDFUNDS": "Federal Funds Effective Rate",
    "DGS10": "10-Year Treasury Constant Maturity Rate",
    "DGS2": "2-Year Treasury Constant Maturity Rate",
    "T10Y2Y": "10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity",
    "UNRATE": "Unemployment Rate",
    "U6RATE": "U-6 Unemployment Rate",
    "PAYEMS": "All Employees: Total Nonfarm Payrolls",
    "RSAFS": "Retail and Food Services Sales",
    "HOUST": "Housing Starts: Total New Privately Owned Housing Units Started",
    "DEXUSEU": "U.S. / Euro Foreign Exchange Rate",
    "DEXJPUS": "Japan / U.S. Foreign Exchange Rate",
    "BAMLH0A0HYM2": "ICE BofA US High Yield Index Effective Yield",
    "VIXCLS": "CBOE Volatility Index (VIX)",
    "SP500": "S&P 500 Index",
    "NASDAQCOM": "NASDAQ Composite Index",
    "DCOILWTICO": "Crude Oil Prices: West Texas Intermediate (WTI)",
    "M2SL": "M2 Money Stock",
    "TOTALSL": "Total Nonrevolving Credit Owned and Securitized, Outstanding",
    "BOGMBASE": "Monetary Base",
    "WALCL": "Federal Reserve Assets: Total Assets",
}

# Series ID to category mapping
SERIES_CATEGORIES = {
    "GDP": DataCategory.ECONOMIC,
    "GDPC1": DataCategory.ECONOMIC,
    "CPIAUCSL": DataCategory.ECONOMIC,
    "CPILFESL": DataCategory.ECONOMIC,
    "PCEPI": DataCategory.ECONOMIC,
    "FEDFUNDS": DataCategory.ECONOMIC,
    "DGS10": DataCategory.FINANCIAL,
    "DGS2": DataCategory.FINANCIAL,
    "T10Y2Y": DataCategory.FINANCIAL,
    "UNRATE": DataCategory.ECONOMIC,
    "U6RATE": DataCategory.ECONOMIC,
    "PAYEMS": DataCategory.ECONOMIC,
    "RSAFS": DataCategory.ECONOMIC,
    "HOUST": DataCategory.ECONOMIC,
    "DEXUSEU": DataCategory.FINANCIAL,
    "DEXJPUS": DataCategory.FINANCIAL,
    "BAMLH0A0HYM2": DataCategory.FINANCIAL,
    "VIXCLS": DataCategory.MARKET,
    "SP500": DataCategory.MARKET,
    "NASDAQCOM": DataCategory.MARKET,
    "DCOILWTICO": DataCategory.MARKET,
    "M2SL": DataCategory.ECONOMIC,
    "TOTALSL": DataCategory.FINANCIAL,
    "BOGMBASE": DataCategory.ECONOMIC,
    "WALCL": DataCategory.ECONOMIC,
}

# Unit mappings for each series
SERIES_UNITS = {
    "GDP": "Billions of Dollars",
    "GDPC1": "Billions of Chained 2017 Dollars",
    "CPIAUCSL": "Index 1982-1984=100",
    "CPILFESL": "Index 1982-1984=100",
    "PCEPI": "Index 2017=100",
    "FEDFUNDS": "Percent",
    "DGS10": "Percent",
    "DGS2": "Percent",
    "T10Y2Y": "Percent",
    "UNRATE": "Percent",
    "U6RATE": "Percent",
    "PAYEMS": "Thousands of Persons",
    "RSAFS": "Millions of Dollars",
    "HOUST": "Thousands of Units",
    "DEXUSEU": "U.S. Dollars per Euro",
    "DEXJPUS": "Japanese Yen per U.S. Dollar",
    "BAMLH0A0HYM2": "Percent",
    "VIXCLS": "Index",
    "SP500": "Index",
    "NASDAQCOM": "Index",
    "DCOILWTICO": "Dollars per Barrel",
    "M2SL": "Billions of Dollars",
    "TOTALSL": "Millions of Dollars",
    "BOGMBASE": "Millions of Dollars",
    "WALCL": "Millions of Dollars",
}


class FredClient:
    """Client for FRED API data ingestion."""

    BASE_URL = "https://api.stlouisfed.org/fred"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("FRED_API_KEY")
        if not self.api_key:
            raise ValueError("FRED_API_KEY environment variable is required")

    async def fetch_series(
        self,
        series_id: str,
        observation_start: str | None = None,
        observation_end: str | None = None,
    ) -> dict[str, Any]:
        """Fetch economic data series from FRED.

        Uses default frequency for each series (no frequency parameter).

        Args:
            series_id: FRED series ID (e.g., 'GDP', 'CPIAUCSL')
            observation_start: Start date in YYYY-MM-DD format
            observation_end: End date in YYYY-MM-DD format

        Returns:
            JSON response from FRED API
        """
        async with httpx.AsyncClient() as client:
            params = {
                "series_id": series_id,
                "api_key": self.api_key,
                "file_type": "json",
            }

            if observation_start:
                params["observation_start"] = observation_start
            if observation_end:
                params["observation_end"] = observation_end

            response = await client.get(
                f"{self.BASE_URL}/series/observations",
                params=params,
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json()


async def upsert_data_points(
    session, series_id: str, observations: list[dict], raw_response: dict
) -> int:
    """Upsert observations into data_points table.

    Args:
        session: SQLAlchemy async session
        series_id: FRED series ID
        observations: List of observation dicts with 'date' and 'value'
        raw_response: Raw API response for storage

    Returns:
        Number of records upserted
    """
    if not observations:
        return 0

    category = SERIES_CATEGORIES.get(series_id, DataCategory.ECONOMIC)
    unit = SERIES_UNITS.get(series_id, "Unknown")

    records = []
    for obs in observations:
        # Skip missing values (FRED returns '.' for missing data)
        if obs.get("value") == "." or obs.get("value") is None:
            continue

        try:
            value = float(obs["value"])
        except (ValueError, TypeError):
            continue

        records.append(
            {
                "source": DataSource.FRED,
                "category": category,
                "metric_name": series_id,
                "value": value,
                "unit": unit,
                "date_recorded": datetime.strptime(obs["date"], "%Y-%m-%d").date(),
                "geographic_scope": "United States",
                "confidence_score": 1.0,
                "raw_json": raw_response,
            }
        )

    if not records:
        return 0

    # Use PostgreSQL INSERT ... ON CONFLICT for upsert
    # Conflict on (source, metric_name, date_recorded)
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


async def run():
    """Fetch all FRED series and upsert into database.

    This is the main entry point for FRED data ingestion.
    Fetches the last 10 years of data for each series with
    rate limiting (0.5s delay between requests).
    """
    logger.info("Starting FRED ingestion run")

    # Initialize database connection
    db.initialize()

    # Calculate date range (last 10 years)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 10)
    observation_start = start_date.strftime("%Y-%m-%d")
    observation_end = end_date.strftime("%Y-%m-%d")

    client = FredClient()
    total_records = 0
    errors = []

    async with db.async_session() as session:
        # Log start
        await log_ingestion(
            session,
            DataSource.FRED,
            IngestionStatus.RUNNING,
        )
        await session.commit()

        for series_id in SERIES_NAMES.keys():
            try:
                logger.info(f"Fetching series: {series_id} ({SERIES_NAMES[series_id]})")

                # Fetch data from FRED
                response = await client.fetch_series(
                    series_id=series_id,
                    observation_start=observation_start,
                    observation_end=observation_end,
                )

                # Extract observations
                observations = response.get("observations", [])
                observation_count = len(observations)

                if observation_count == 0:
                    logger.warning(f"No observations found for {series_id}")
                    continue

                # Upsert into database
                records_upserted = await upsert_data_points(
                    session, series_id, observations, response
                )
                total_records += records_upserted
                logger.info(
                    f"  Upserted {records_upserted} records for {series_id} "
                    f"(fetched {observation_count} observations)"
                )

                # Rate limiting: wait 0.5s between requests
                await asyncio.sleep(0.5)

            except httpx.HTTPError as e:
                error_msg = f"HTTP error fetching {series_id}: {e}"
                logger.warning(error_msg)
                errors.append(error_msg)
                await session.rollback()
            except Exception as e:
                error_msg = f"Error processing {series_id}: {e}"
                logger.warning(error_msg)
                errors.append(error_msg)
                await session.rollback()

        # Log completion
        final_status = IngestionStatus.FAILED if errors else IngestionStatus.COMPLETED
        error_text = " | ".join(errors) if errors else None

        await log_ingestion(
            session,
            DataSource.FRED,
            final_status,
            records_fetched=total_records,
            errors=error_text,
        )
        await session.commit()

    logger.info(
        f"FRED ingestion complete: {total_records} total records upserted, "
        f"{len(errors)} errors"
    )

    return {"total_records": total_records, "errors": errors}


if __name__ == "__main__":
    asyncio.run(run())
