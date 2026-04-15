"""Bureau of Labor Statistics (BLS) ingestion module."""

import asyncio
import logging
import os
from datetime import datetime
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


# BLS Series ID to human-readable name mapping
SERIES_NAMES = {
    "CES0000000001": "All Employees: Total Nonfarm Payrolls",
    "LNS14000000": "Unemployment Rate",
    "CUUR0000SA0": "Consumer Price Index for All Urban Consumers: All Items",
    "CUUR0000SA0L1E": "Consumer Price Index: All Items Less Food and Energy",
    "CES0500000003": "Average Hourly Earnings of All Private Sector Employees",
    "PRS85006092": "Nonfarm Business Sector: Labor Productivity",
    "WPUFD4": "Producer Price Index by Commodity: Finished Goods",
}

# Unit mappings for each series
SERIES_UNITS = {
    "CES0000000001": "Thousands of Persons",
    "LNS14000000": "Percent",
    "CUUR0000SA0": "Index 1982-1984=100",
    "CUUR0000SA0L1E": "Index 1982-1984=100",
    "CES0500000003": "Dollars per Hour",
    "PRS85006092": "Index 2012=100",
    "WPUFD4": "Index 1982=100",
}


class BLSClient:
    """Client for BLS Public Data API v2 ingestion.

    The BLS API uses POST requests and returns JSON.
    No API key required for basic use (limited to 25 requests/second).
    With API key: 500 requests/second.
    """

    BASE_URL = "https://api.bls.gov/publicAPI/v2"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("BLS_API_KEY")

    def _get_headers(self) -> dict[str, str]:
        """Get headers for BLS API requests."""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "ContentReview/1.0",
        }
        if self.api_key:
            headers["Authorization"] = self.api_key
        return headers

    async def fetch_series(
        self,
        series_ids: list[str],
        start_year: str,
        end_year: str,
        calculations: bool = True,
        annual_average: bool = False,
    ) -> dict[str, Any]:
        """Fetch data for specified series IDs.

        Args:
            series_ids: List of BLS series IDs to fetch
            start_year: Start year (YYYY format)
            end_year: End year (YYYY format)
            calculations: Include percent changes and calculations
            annual_average: Include annual averages

        Returns:
            JSON response from BLS API
        """
        payload = {
            "seriesid": series_ids,
            "startyear": start_year,
            "endyear": end_year,
            "calculations": "true" if calculations else "false",
            "annualaverage": "true" if annual_average else "false",
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.BASE_URL}/timeseries/data/",
                json=payload,
                headers=self._get_headers(),
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json()

    async def fetch_series_single(
        self,
        series_id: str,
        start_year: str,
        end_year: str,
    ) -> dict[str, Any]:
        """Fetch a single series (more efficient for one series).

        Args:
            series_id: BLS series ID
            start_year: Start year (YYYY format)
            end_year: End year (YYYY format)

        Returns:
            JSON response from BLS API
        """
        payload = {
            "seriesid": [series_id],
            "startyear": start_year,
            "endyear": end_year,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.BASE_URL}/timeseries/data/",
                json=payload,
                headers=self._get_headers(),
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json()


def parse_bls_response(response: dict[str, Any]) -> list[dict]:
    """Parse BLS API response into list of observations.

    Args:
        response: Raw BLS API response

    Returns:
        List of dicts with series_id, date, value, footnote
    """
    observations = []

    results = response.get("Results", {})
    series_list = results.get("series", [])

    for series_data in series_list:
        series_id = series_data.get("seriesID")
        data_points = series_data.get("data", [])

        for data in data_points:
            year = data.get("year")
            period = data.get("period")
            value = data.get("value")
            footnote = data.get("footnotes", [{}])[0].get("text", "")

            # Skip if missing required fields
            if not all([year, period, value]):
                continue

            # Convert period (M01-M12) to month number
            month = period[1:] if period.startswith("M") else "01"

            # Create date string (BLS monthly data is for the month)
            date_str = f"{year}-{month.zfill(2)}-01"

            observations.append({
                "series_id": series_id,
                "date": date_str,
                "value": value,
                "footnote": footnote,
                "year": year,
                "period": period,
            })

    return observations


async def upsert_data_points(
    session, series_id: str, observations: list[dict], raw_response: dict
) -> int:
    """Upsert observations into data_points table.

    Args:
        session: SQLAlchemy async session
        series_id: BLS series ID
        observations: List of observation dicts
        raw_response: Raw API response for storage

    Returns:
        Number of records upserted
    """
    if not observations:
        return 0

    unit = SERIES_UNITS.get(series_id, "Unknown")

    records = []
    for obs in observations:
        # Skip missing or invalid values
        try:
            value = float(obs["value"])
        except (ValueError, TypeError, KeyError):
            continue

        records.append(
            {
                "source": DataSource.BLS,
                "category": DataCategory.ECONOMIC,
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


async def run(series_ids: list[str] | None = None):
    """Fetch all BLS series and upsert into database.

    This is the main entry point for BLS data ingestion.
    Fetches 5 years of monthly data for each series with
    rate limiting (0.5s delay between requests).

    Args:
        series_ids: List of series IDs to fetch. Defaults to all configured series.

    Returns:
        Dict with ingestion statistics
    """
    logger.info("Starting BLS ingestion run")

    # Initialize database connection
    db.initialize()

    # Calculate date range (5 years)
    current_year = datetime.now().year
    start_year = str(current_year - 5)
    end_year = str(current_year)

    series_ids = series_ids or list(SERIES_NAMES.keys())
    client = BLSClient()

    total_records = 0
    errors = []

    async with db.async_session() as session:
        # Log start
        await log_ingestion(
            session,
            DataSource.BLS,
            IngestionStatus.RUNNING,
        )
        await session.commit()

        # Fetch each series individually for better error handling
        for series_id in series_ids:
            try:
                logger.info(
                    f"Fetching series: {series_id} ({SERIES_NAMES[series_id]})"
                )

                # Fetch data from BLS
                response = await client.fetch_series_single(
                    series_id=series_id,
                    start_year=start_year,
                    end_year=end_year,
                )

                # Check for API errors in response
                status = response.get("status", "")
                if status != "REQUEST_SUCCEEDED":
                    message = response.get("message", "Unknown error")
                    logger.warning(f"BLS API returned status '{status}': {message}")
                    if status == "REQUEST_FAILED":
                        errors.append(f"BLS API error for {series_id}: {message}")
                        continue

                # Parse observations
                observations = parse_bls_response(response)
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
                logger.error(error_msg)
                errors.append(error_msg)
            except Exception as e:
                error_msg = f"Error processing {series_id}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)

        # Log completion
        final_status = IngestionStatus.FAILED if errors else IngestionStatus.COMPLETED
        error_text = " | ".join(errors) if errors else None

        await log_ingestion(
            session,
            DataSource.BLS,
            final_status,
            records_fetched=total_records,
            errors=error_text,
        )
        await session.commit()

    logger.info(
        f"BLS ingestion complete: {total_records} total records upserted, "
        f"{len(errors)} errors"
    )

    return {"total_records": total_records, "errors": errors}


if __name__ == "__main__":
    asyncio.run(run())
