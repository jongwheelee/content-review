"""SEC EDGAR ingestion module for filings and company facts."""

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
from database.models import Company, Filing, FilingFact, IngestionLog, IngestionStatus, DataSource

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Top 50 S&P 500 companies by market cap (as of 2024)
# This is a curated list for demonstration purposes
TOP_50_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "BRK.B", "TSLA", "AVGO", "JPM",
    "JNJ", "V", "PG", "MA", "UNH", "HD", "CVX", "MRK", "ABBV", "PEP",
    "KO", "COST", "LLY", "WMT", "MCD", "CSCO", "TMO", "ACN", "ADBE", "NKE",
    "TXN", "ABT", "VZ", "DIS", "CRM", "INTC", "NEE", "PM", "UPS", "RTX",
    "QCOM", "HON", "LOW", "ORCL", "MS", "IBM", "AMD", "GS", "CAT", "BA"
]

# GAAP facts to extract from company facts API
GAAP_FACTS = [
    "Revenues",
    "NetIncomeLoss",
    "EarningsPerShareBasic",
    "Assets",
    "Liabilities",
    "StockholdersEquity",
    "OperatingIncomeLoss",
    "CommonStockSharesOutstanding",
    "CashAndCashEquivalentsAtCarryingValue",
]

# Form types to fetch
FORM_TYPES = ["10-K", "10-Q"]


class SECEdgarClient:
    """Client for SEC EDGAR API data ingestion.

    Follows SEC EDGAR API requirements:
    - User-Agent header with company information
    - Rate limiting (1 second between requests)
    """

    BASE_URL = "https://www.sec.gov"
    DATA_URL = "https://data.sec.gov"
    FILES_URL = "https://www.sec.gov/files"

    def __init__(self, user_agent: str | None = None):
        self.user_agent = user_agent or os.getenv("SEC_USER_AGENT")
        if not self.user_agent:
            raise ValueError(
                "SEC_USER_AGENT environment variable is required. "
                "See https://www.sec.gov/os/webmaster-faq#code-support"
            )

        self.headers = {
            "User-Agent": f"ContentReview {self.user_agent}",
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate",
        }

    async def fetch_company_tickers(self) -> dict[str, Any]:
        """Fetch the full company ticker JSON from SEC.

        Returns:
            Dict mapping CIK to ticker/name information
        """
        async with httpx.AsyncClient(headers=self.headers) as client:
            response = await client.get(
                f"{self.FILES_URL}/company_tickers.json",
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json()

    async def fetch_company_submissions(self, cik: str) -> dict[str, Any]:
        """Fetch company filings from submissions API.

        Args:
            cik: Company CIK (will be zero-padded to 10 digits)

        Returns:
            Company submissions data with recent filings
        """
        cik_padded = cik.zfill(10)

        async with httpx.AsyncClient(headers=self.headers) as client:
            response = await client.get(
                f"{self.DATA_URL}/submissions/CIK{cik_padded}.json",
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json()

    async def fetch_company_facts(self, cik: str) -> dict[str, Any]:
        """Fetch XBRL company facts from API.

        Args:
            cik: Company CIK (will be zero-padded to 10 digits)

        Returns:
            Company facts with all XBRL tags and values
        """
        cik_padded = cik.zfill(10)

        async with httpx.AsyncClient(headers=self.headers) as client:
            response = await client.get(
                f"{self.DATA_URL}/api/xbrl/companyfacts/CIK{cik_padded}.json",
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json()


def parse_company_tickers(data: dict[str, Any]) -> list[dict]:
    """Parse company tickers JSON into list of company dicts.

    Args:
        data: Raw JSON from company_tickers.json

    Returns:
        List of dicts with cik, ticker, name
    """
    companies = []
    for _, company in data.items():
        companies.append({
            "cik": str(company["cik_str"]).zfill(10),
            "ticker": company.get("ticker", ""),
            "name": company.get("title", ""),
        })
    return companies


def pad_cik(cik: str) -> str:
    """Pad CIK to 10 digits with leading zeros."""
    return str(cik).zfill(10)


async def upsert_companies(session, companies: list[dict]) -> int:
    """Upsert companies into the companies table in batches.

    Args:
        session: SQLAlchemy async session
        companies: List of company dicts

    Returns:
        Number of records upserted
    """
    if not companies:
        return 0

    # Deduplicate by CIK (keep last occurrence)
    unique_companies = {}
    for company in companies:
        unique_companies[company["cik"]] = company
    companies = list(unique_companies.values())

    # Batch inserts to avoid exceeding PostgreSQL's 32767 parameter limit
    # Each company has 4 fields, so max 8191 companies per batch (8191 * 4 = 32764)
    batch_size = 500
    total_upserted = 0

    for i in range(0, len(companies), batch_size):
        batch = companies[i:i + batch_size]
        stmt = insert(Company).values(batch)
        stmt = stmt.on_conflict_do_update(
            index_elements=["cik"],
            set_={
                "ticker": stmt.excluded.ticker,
                "name": stmt.excluded.name,
            },
        )
        await session.execute(stmt)
        total_upserted += len(batch)

    return total_upserted


async def upsert_filing(
    session, company_id: int, form_type: str, filing_data: dict
) -> int | None:
    """Upsert a single filing into the filings table.

    Args:
        session: SQLAlchemy async session
        company_id: Company ID from companies table
        form_type: Form type (10-K, 10-Q, etc.)
        filing_data: Filing metadata from SEC API

    Returns:
        Filing ID if successful, None otherwise
    """
    # Parse dates - SEC API returns YYYY-MM-DD format strings
    report_date_str = filing_data.get("reportDate", "")
    file_date_str = filing_data.get("fileDate", "")

    # Skip filings with missing required dates
    if not report_date_str or not file_date_str:
        logger.warning(f"Skipping filing with missing dates: {form_type}")
        return None

    try:
        period_of_report = datetime.strptime(report_date_str, "%Y-%m-%d").date()
        filed_at = datetime.strptime(file_date_str, "%Y-%m-%d")
    except (ValueError, TypeError) as e:
        logger.warning(f"Error parsing filing dates: {e}")
        return None

    accession_number = filing_data.get("accessionNumber", "").replace("-", "")

    # Skip if no accession number
    if not accession_number:
        logger.warning(f"Skipping filing with missing accession number: {form_type}")
        return None

    filing = {
        "company_id": company_id,
        "form_type": form_type,
        "period_of_report": period_of_report,
        "filed_at": filed_at,
        "accession_number": accession_number,
        "filing_url": filing_data.get("fileNumber"),
        "processed": False,
    }

    stmt = insert(Filing).values(filing)
    stmt = stmt.on_conflict_do_update(
        index_elements=["accession_number"],
        set_={
            "processed": False,
        },
    )

    result = await session.execute(stmt)
    await session.flush()

    # Get the filing ID
    filing_id = await session.execute(
        select(Filing.id).where(Filing.accession_number == filing["accession_number"])
    )
    filing_id = filing_id.scalar_one_or_none()

    return filing_id


async def upsert_filing_facts(session, filing_id: int, facts: list[dict]) -> int:
    """Upsert filing facts in batch.

    Args:
        session: SQLAlchemy async session
        filing_id: Filing ID from filings table
        facts: List of fact dicts with metric_name, value, unit, periods

    Returns:
        Number of facts inserted
    """
    if not facts:
        return 0

    # Batch inserts to avoid exceeding PostgreSQL's 32767 parameter limit
    # Each fact has 7 fields, so max ~4681 facts per batch
    batch_size = 1000
    total_inserted = 0

    for i in range(0, len(facts), batch_size):
        batch = facts[i:i + batch_size]
        records = []
        for fact in batch:
            records.append({
                "filing_id": filing_id,
                "metric_name": fact["metric_name"],
                "value": fact.get("value"),
                "unit": fact.get("unit"),
                "period_start": fact.get("period_start"),
                "period_end": fact.get("period_end"),
                "context_label": fact.get("context_label"),
            })

        stmt = insert(FilingFact).values(records)
        stmt = stmt.on_conflict_do_nothing(
            index_elements=["filing_id", "metric_name", "period_end"],
        )
        await session.execute(stmt)
        total_inserted += len(records)

    return total_inserted


def extract_gaap_facts(company_facts: dict[str, Any]) -> list[dict]:
    """Extract GAAP facts from company facts response.

    Args:
        company_facts: Raw response from companyfacts API

    Returns:
        List of fact dicts ready for insertion
    """
    facts = []
    us_gaap = company_facts.get("facts", {}).get("us-gaap", {})

    for fact_name in GAAP_FACTS:
        if fact_name not in us_gaap:
            continue

        fact_data = us_gaap[fact_name]
        units = fact_data.get("units", {})

        # Get the primary unit (usually USD or shares)
        for unit_label, values in units.items():
            for value in values:
                # Extract period information
                period_start_str = value.get("start")
                period_end_str = value.get("end")

                # Skip if no value or period
                if value.get("val") is None or not period_end_str:
                    continue

                # Convert date strings to date objects
                try:
                    period_start = datetime.strptime(period_start_str, "%Y-%m-%d").date() if period_start_str else None
                    period_end = datetime.strptime(period_end_str, "%Y-%m-%d").date() if period_end_str else None
                except (ValueError, TypeError):
                    continue

                facts.append({
                    "metric_name": fact_name,
                    "value": value.get("val"),
                    "unit": unit_label,
                    "period_start": period_start,
                    "period_end": period_end,
                    "context_label": str(value.get("fy", "")) if value.get("fy") is not None else "",
                })

    return facts


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
    """Main ingestion function for SEC EDGAR data.

    Steps:
    1. Fetch all company tickers and upsert to companies table
    2. For each ticker in the provided list (or TOP_50_TICKERS):
       a. Fetch recent 10-K and 10-Q filings
       b. Upsert filing metadata
       c. For unprocessed 10-K filings, fetch company facts
       d. Extract and store GAAP facts
       e. Mark filing as processed

    Args:
        tickers: List of tickers to process. Defaults to TOP_50_TICKERS.

    Returns:
        Dict with ingestion statistics
    """
    logger.info("Starting SEC EDGAR ingestion run")

    # Initialize database connection
    db.initialize()

    tickers = tickers or TOP_50_TICKERS
    client = SECEdgarClient()

    total_companies = 0
    total_filings = 0
    total_facts = 0
    errors = []

    # Build ticker to CIK mapping
    ticker_to_cik = {}
    cik_to_company_id = {}

    async with db.async_session() as session:
        # Log start
        await log_ingestion(
            session,
            DataSource.SEC_EDGAR,
            IngestionStatus.RUNNING,
        )
        await session.commit()

        # Step 1: Fetch all company tickers
        logger.info("Fetching company tickers from SEC...")
        try:
            tickers_data = await client.fetch_company_tickers()
            companies = parse_company_tickers(tickers_data)

            total_companies = await upsert_companies(session, companies)
            logger.info(f"Upserted {total_companies} companies")

            # Build mapping for quick lookup
            for company in companies:
                ticker_to_cik[company["ticker"]] = company["cik"]

        except httpx.HTTPError as e:
            error_msg = f"Failed to fetch company tickers: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
            await log_ingestion(
                session,
                DataSource.SEC_EDGAR,
                IngestionStatus.FAILED,
                errors=error_msg,
            )
            await session.commit()
            return {"error": error_msg}

        await session.commit()

        # Step 2: Process each ticker
        for ticker in tickers:
            logger.info(f"Processing ticker: {ticker}")

            cik = ticker_to_cik.get(ticker)
            if not cik:
                logger.warning(f"CIK not found for ticker: {ticker}")
                continue

            try:
                # Fetch company submissions
                await asyncio.sleep(1)  # Rate limiting
                submissions = await client.fetch_company_submissions(cik)

                # Get company ID from database
                company_result = await session.execute(
                    select(Company.id).where(Company.cik == cik)
                )
                company_id = company_result.scalar_one_or_none()

                if not company_id:
                    logger.warning(f"Company not in database: {ticker} (CIK: {cik})")
                    continue

                # Process recent filings (10-K and 10-Q)
                recent_filings = submissions.get("filings", {}).get("recent", {})
                form_types = recent_filings.get("form", [])  # SEC API uses "form" not "formType"
                accession_numbers = recent_filings.get("accessionNumber", [])
                report_dates = recent_filings.get("reportDate", [])
                file_dates = recent_filings.get("filingDate", [])  # SEC API uses "filingDate" not "fileDate"

                filings_to_process = []
                for i, form_type in enumerate(form_types):
                    if form_type in FORM_TYPES:
                        filings_to_process.append({
                            "form_type": form_type,
                            "accession_number": accession_numbers[i] if i < len(accession_numbers) else "",
                            "report_date": report_dates[i] if i < len(report_dates) else "",
                            "file_date": file_dates[i] if i < len(file_dates) else "",
                        })

                # Limit to most recent 10 filings per form type
                filings_to_process = filings_to_process[:20]

                for filing_data in filings_to_process:
                    try:
                        filing_id = await upsert_filing(
                            session, company_id, filing_data["form_type"], {
                                "reportDate": filing_data["report_date"],
                                "fileDate": filing_data["file_date"],
                                "accessionNumber": filing_data["accession_number"],
                            }
                        )

                        if filing_id:
                            total_filings += 1

                        # Step 3: For unprocessed 10-K filings, fetch company facts
                        if filing_data["form_type"] == "10-K":
                            logger.info(f"  Fetching company facts for {ticker} 10-K...")

                            await asyncio.sleep(1)  # Rate limiting
                            company_facts = await client.fetch_company_facts(cik)

                            facts = extract_gaap_facts(company_facts)

                            if facts:
                                facts_inserted = await upsert_filing_facts(session, filing_id, facts)
                                total_facts += facts_inserted

                            # Mark filing as processed
                            filing = await session.get(Filing, filing_id)
                            if filing:
                                filing.processed = True

                            logger.info(f"  Extracted {len(facts)} facts for {ticker}")

                    except Exception as e:
                        logger.error(f"    Error processing filing: {e}")
                        errors.append(f"Error processing {ticker} filing: {e}")

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    logger.warning(f"404 for ticker {ticker} (CIK: {cik}) - company may not file with SEC")
                else:
                    error_msg = f"HTTP error for {ticker}: {e}"
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

            # Commit after each ticker to avoid large transactions
            await session.commit()

        # Log completion
        final_status = IngestionStatus.FAILED if errors else IngestionStatus.COMPLETED
        error_text = " | ".join(errors) if errors else None

        await log_ingestion(
            session,
            DataSource.SEC_EDGAR,
            final_status,
            records_fetched=total_filings + total_facts,
            errors=error_text,
        )
        await session.commit()

    logger.info(
        f"SEC EDGAR ingestion complete: "
        f"{total_companies} companies, "
        f"{total_filings} filings, "
        f"{total_facts} facts, "
        f"{len(errors)} errors"
    )

    return {
        "total_companies": total_companies,
        "total_filings": total_filings,
        "total_facts": total_facts,
        "errors": errors,
    }


if __name__ == "__main__":
    asyncio.run(run())
