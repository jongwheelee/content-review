"""Claim verification module for financial fact-checking pipeline.

This module verifies extracted claims against the database by finding
supporting or contradicting evidence.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

import numpy as np
from sqlalchemy import select, func, text
from sqlalchemy.orm import Session

from database.connection import db
from database.models import (
    DataPoint,
    FilingFact,
    Filing,
    Company,
    ResearchPaper,
    EarningsTranscript,
    NewsArticle,
    Embedding,
    DataSource,
)
from processing.embedder import Embedder

from .claim_extractor import Claim


class VerificationStatus(str, Enum):
    """Detailed internal verification status for reasoning."""

    VERIFIED = "VERIFIED"
    CONTRADICTED = "CONTRADICTED"
    UNVERIFIABLE = "UNVERIFIABLE"
    PARTIALLY_CORRECT = "PARTIALLY_CORRECT"
    OUTDATED = "OUTDATED"


class UserFacingStatus(str, Enum):
    """Simplified user-facing status categories."""

    LIKELY_FINE = "LIKELY_FINE"  # VERIFIED claims
    SHOULD_VERIFY = "SHOULD_VERIFY"  # UNVERIFIABLE, PARTIALLY_CORRECT
    POTENTIALLY_MISLEADING = "POTENTIALLY_MISLEADING"  # CONTRADICTED, OUTDATED


def get_user_facing_status(status: VerificationStatus) -> UserFacingStatus:
    """Map detailed internal status to user-facing category.

    Args:
        status: Internal VerificationStatus

    Returns:
        UserFacingStatus category
    """
    if status == VerificationStatus.VERIFIED:
        return UserFacingStatus.LIKELY_FINE
    elif status in (VerificationStatus.CONTRADICTED, VerificationStatus.OUTDATED):
        return UserFacingStatus.POTENTIALLY_MISLEADING
    else:  # UNVERIFIABLE, PARTIALLY_CORRECT
        return UserFacingStatus.SHOULD_VERIFY


@dataclass
class Evidence:
    """Evidence found for a claim."""

    source: str
    source_type: str  # data_point, filing_fact, research_paper, news_article
    content: str
    value: Optional[float]
    date: Optional[str]
    relevance_score: float


@dataclass
class VerificationResult:
    """Result of verifying a claim."""

    claim: Claim
    status: VerificationStatus
    confidence: float
    evidence: list[Evidence]
    correction: Optional[str]
    explanation: str
    user_facing_status: Optional[UserFacingStatus] = None

    def __post_init__(self):
        """Set user-facing status after initialization."""
        if self.user_facing_status is None:
            self.user_facing_status = get_user_facing_status(self.status)


# Mapping from common terms to actual FRED/database metric names
# Keys are checked as substrings (case-insensitive) against extracted entities
METRIC_NAME_MAPPING = {
    "interest rate": ["FEDFUNDS", "DGS10", "DGS2"],  # Federal funds rate, 10Y/2Y treasury
    "federal reserve": ["FEDFUNDS"],
    "federal funds": ["FEDFUNDS"],
    "inflation": ["CPILFESL", "CPIAUCSL"],  # Core CPI, All-items CPI
    "inflation rate": ["CPILFESL", "CPIAUCSL"],
    "cpi": ["CPILFESL", "CPIAUCSL"],
    "unemployment": ["UNRATE"],
    "unemployment rate": ["UNRATE"],
    "gdp": ["GDP", "GDPC1"],
    "gdp growth": ["GDP", "GDPC1"],
    "revenue": [],  # Handled by company filings
    "earnings": [],  # Handled by company filings
}


def _extract_metric_name(entities: list[str]) -> Optional[str]:
    """Extract metric name from entities list.

    Returns either a direct metric name or a mapped database metric name.
    """
    # First, check if any entity maps to known database metrics
    for entity in entities:
        entity_lower = entity.lower()
        for term, metrics in METRIC_NAME_MAPPING.items():
            if term in entity_lower:
                # Return the primary (first) metric name for this term
                if metrics:
                    return metrics[0]
                # If no mapping, fall through to return the entity itself
                break

    # Look for common metric patterns
    metric_patterns = [
        r"(?:GDP|inflation|unemployment|interest rate|revenue|earnings|profit|margin)",
        r"(?:CPI|PPI|PEG|EPS|EBITDA)",
    ]
    for entity in entities:
        for pattern in metric_patterns:
            if re.search(pattern, entity, re.IGNORECASE):
                # Check if this entity has a mapping
                entity_lower = entity.lower()
                for term, metrics in METRIC_NAME_MAPPING.items():
                    if term in entity_lower and metrics:
                        return metrics[0]
                return entity

    # Return first entity that doesn't look like a ticker
    for entity in entities:
        if not re.match(r"^[A-Z]{1,5}$", entity):
            return entity
    return None


def _extract_ticker(entities: list[str]) -> Optional[str]:
    """Extract ticker symbol from entities list."""
    for entity in entities:
        if re.match(r"^[A-Z]{1,5}$", entity):
            return entity
    return None


def _parse_time_reference(
    time_ref: Optional[str],
    content_date: Optional[str] = None,
) -> tuple[Optional[datetime], Optional[datetime]]:
    """Parse time reference string into datetime range.

    Args:
        time_ref: Time reference string (e.g., "2024", "last month", "Q3 2024")
        content_date: When the source content was written (ISO date string)

    Returns:
        Tuple of (start_date, end_date) for querying data.
        - For absolute refs like "2024": returns (2024-01-01, 2024-12-31)
        - For relative refs like "last month": resolves relative to content_date
        - If content_date is None and ref is relative: returns (None, None) to get latest
    """
    if not time_ref:
        return (None, None)

    # Parse content_date as anchor for relative references
    anchor_date = None
    if content_date:
        try:
            anchor_date = datetime.fromisoformat(content_date.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            # Try simple YYYY-MM-DD format
            try:
                anchor_date = datetime.strptime(content_date, "%Y-%m-%d")
            except (ValueError, TypeError):
                anchor_date = None

    # Handle "Q1 2024", "Q2 2023" format (absolute)
    quarter_match = re.match(r"Q(\d)\s+(\d{4})", time_ref)
    if quarter_match:
        quarter = int(quarter_match.group(1))
        year = int(quarter_match.group(2))
        start_month = (quarter - 1) * 3 + 1
        end_month = start_month + 2
        start = datetime(year, start_month, 1)
        end = datetime(year, end_month, 28)
        return (start, end)

    # Handle "2023", "2024" format (absolute year)
    year_match = re.match(r"^(\d{4})$", time_ref)
    if year_match:
        year = int(year_match.group(1))
        return (datetime(year, 1, 1), datetime(year, 12, 31))

    # Handle relative references - resolve relative to content_date
    time_ref_lower = time_ref.lower()

    if "last month" in time_ref_lower:
        if anchor_date:
            # Go to previous month
            if anchor_date.month == 1:
                start = datetime(anchor_date.year - 1, 12, 1)
                end = datetime(anchor_date.year - 1, 12, 31)
            else:
                start = datetime(anchor_date.year, anchor_date.month - 1, 1)
                end = datetime(anchor_date.year, anchor_date.month - 1, 28)
            return (start, end)
        else:
            # No anchor - get recent data (last 3 months)
            recent = datetime.now() - timedelta(days=90)
            return (recent, None)  # end=None means "no upper bound"

    if "last quarter" in time_ref_lower:
        if anchor_date:
            # Go to previous quarter
            current_quarter = (anchor_date.month - 1) // 3
            prev_quarter = (current_quarter - 1) % 4
            year = anchor_date.year if current_quarter > 0 else anchor_date.year - 1
            start = datetime(year, prev_quarter * 3 + 1, 1)
            end = datetime(year, prev_quarter * 3 + 3, 31)
            return (start, end)
        else:
            recent = datetime.now() - timedelta(days=120)
            return (recent, None)

    if "past 6 months" in time_ref_lower or "last 6 months" in time_ref_lower:
        if anchor_date:
            six_months_ago = anchor_date - timedelta(days=180)
            return (six_months_ago, anchor_date)
        else:
            recent = datetime.now() - timedelta(days=200)
            return (recent, None)

    if "past year" in time_ref_lower or "last year" in time_ref_lower:
        if anchor_date:
            year_ago = anchor_date - timedelta(days=365)
            return (year_ago, anchor_date)
        else:
            recent = datetime.now() - timedelta(days=400)
            return (recent, None)

    # Try to extract any 4-digit year from the reference
    year_match = re.search(r"(\d{4})", time_ref)
    if year_match:
        year = int(year_match.group(1))
        return (datetime(year, 1, 1), datetime(year, 12, 31))

    return (None, None)


def _get_metric_names_for_search(extracted_metric: str) -> list[str]:
    """Get list of metric names to search for.

    Args:
        extracted_metric: The metric name extracted from claim

    Returns:
        List of metric names to try in database lookup
    """
    # Always include the extracted metric
    metrics = [extracted_metric]

    # Add related metrics from mapping
    for term, mapped_metrics in METRIC_NAME_MAPPING.items():
        if extracted_metric in mapped_metrics:
            # Add sibling metrics (e.g., if FEDFUNDS, also try DGS10, DGS2)
            metrics.extend([m for m in mapped_metrics if m != extracted_metric])
            break

    return list(dict.fromkeys(metrics))  # Remove duplicates, preserve order


def _calculate_inflation_rate(conn, metric_name: str, date_recorded: datetime) -> Optional[float]:
    """Calculate YoY inflation rate for a given CPI value.

    Args:
        conn: Database connection
        metric_name: The CPI metric name (e.g., CPIAUCSL)
        date_recorded: The date of the current CPI value

    Returns:
        YoY inflation rate as percentage, or None if prior year data unavailable
    """
    stmt = text("""
        SELECT value
        FROM data_points
        WHERE metric_name = :metric
        AND date_recorded = :prior_date
    """)
    # Get value from 12 months prior
    prior_date = date_recorded.replace(year=date_recorded.year - 1)
    result = conn.execute(stmt, {"metric": metric_name, "prior_date": prior_date})
    row = result.fetchone()

    if row and row.value:
        current_cpi = float(date_recorded) if isinstance(date_recorded, (int, float)) else None
        # Re-fetch current value
        curr_stmt = text("""
            SELECT value FROM data_points
            WHERE metric_name = :metric AND date_recorded = :date
        """)
        curr_result = conn.execute(curr_stmt, {"metric": metric_name, "date": date_recorded})
        curr_row = curr_result.fetchone()
        if curr_row and curr_row.value and row.value:
            return ((float(curr_row.value) - float(row.value)) / float(row.value)) * 100
    return None


def lookup_numeric(claim: Claim) -> list[Evidence]:
    """Lookup numeric claims in data_points table.

    For STATISTIC, RATE, MACRO_FACT claims:
    - Extract metric name from entities
    - Parse time reference from claim (using content_date as anchor)
    - Query data_points for values in the relevant time range
    - Calculate derived metrics (e.g., YoY inflation) when appropriate

    Time resolution:
    - Absolute refs ("2024", "Q3 2024"): query that exact period
    - Relative refs ("last month"): resolve relative to content_date
    - No time ref: return most recent data
    """
    from datetime import date

    db.initialize()
    extracted_metric = _extract_metric_name(claim.entities)
    if not extracted_metric:
        return []

    # Parse time reference into (start_date, end_date) tuple
    start_date, end_date = _parse_time_reference(claim.time_reference, claim.content_date)

    # Get all metric names to search
    metric_names = _get_metric_names_for_search(extracted_metric)
    evidence_list = []

    with db.sync_engine.connect() as conn:
        for metric_name in metric_names:
            if len(evidence_list) >= 5:
                break

            # For CPI metrics, always calculate YoY inflation rate
            if metric_name in ("CPIAUCSL", "CPILFESL"):
                # Need to fetch 13+ months prior data for YoY calculation
                # LAG(12) needs 13 months of data to produce a YoY value for the first month
                from datetime import timedelta
                cte_start_date = start_date - timedelta(days=400) if start_date else datetime(2020, 1, 1)
                # Extend end_date slightly to ensure we capture the target month's data
                cte_end_date = end_date + timedelta(days=31) if end_date else None

                inflation_stmt = text("""
                    WITH cpi_data AS (
                        SELECT
                            date_recorded,
                            value as cpi_current,
                            LAG(value, 12) OVER (ORDER BY date_recorded) as cpi_year_ago
                        FROM data_points
                        WHERE metric_name = :metric
                        AND date_recorded >= :cte_start_date
                        AND (:cte_end_date IS NULL OR date_recorded < :cte_end_date)
                        ORDER BY date_recorded DESC
                    )
                    SELECT
                        date_recorded,
                        cpi_current,
                        cpi_year_ago,
                        ROUND(((cpi_current - cpi_year_ago) / cpi_year_ago * 100)::numeric, 2) as yoy_inflation_pct
                    FROM cpi_data
                    WHERE cpi_year_ago IS NOT NULL
                    AND date_recorded >= :start_date
                    AND (:end_date IS NULL OR date_recorded < :end_date)
                    ORDER BY date_recorded DESC
                    LIMIT :limit
                """)
                result = conn.execute(inflation_stmt, {
                    "metric": metric_name,
                    "start_date": start_date,
                    "end_date": end_date,
                    "cte_start_date": cte_start_date,
                    "cte_end_date": cte_end_date,
                    "limit": 5 - len(evidence_list)
                })
                for row in result:
                    evidence_list.append(Evidence(
                        source=f"FRED - {metric_name} (YoY calculation)",
                        source_type="data_point",
                        content=f"Inflation rate (YoY): {row.yoy_inflation_pct}% as of {row.date_recorded} "
                                f"(CPI: {row.cpi_current:.2f}, prior year: {row.cpi_year_ago:.2f})",
                        value=float(row.yoy_inflation_pct),
                        date=str(row.date_recorded),
                        relevance_score=1.0,
                    ))
                continue  # Move to next metric

            # For other metrics, filter by date range if specified
            if start_date is not None and end_date is not None:
                # Absolute date range - query within that range
                exact_stmt = text("""
                    SELECT metric_name, value, unit, date_recorded, source
                    FROM data_points
                    WHERE metric_name ILIKE :metric
                    AND date_recorded >= :start_date
                    AND date_recorded <= :end_date
                    ORDER BY date_recorded DESC
                    LIMIT :limit
                """)
                remaining = 5 - len(evidence_list)
                result = conn.execute(exact_stmt, {
                    "metric": metric_name,
                    "start_date": start_date,
                    "end_date": end_date,
                    "limit": remaining
                })
            elif start_date is not None and end_date is None:
                # Relative date (e.g., "last 3 months") - get data since start_date
                exact_stmt = text("""
                    SELECT metric_name, value, unit, date_recorded, source
                    FROM data_points
                    WHERE metric_name ILIKE :metric
                    AND date_recorded >= :start_date
                    ORDER BY date_recorded DESC
                    LIMIT :limit
                """)
                remaining = 5 - len(evidence_list)
                result = conn.execute(exact_stmt, {
                    "metric": metric_name,
                    "start_date": start_date,
                    "limit": remaining
                })
            else:
                # No time reference - get latest data
                exact_stmt = text("""
                    SELECT metric_name, value, unit, date_recorded, source
                    FROM data_points
                    WHERE metric_name ILIKE :metric
                    ORDER BY date_recorded DESC
                    LIMIT :limit
                """)
                remaining = 5 - len(evidence_list)
                result = conn.execute(exact_stmt, {"metric": metric_name, "limit": remaining})

            rows = result.fetchall()

            for row in rows:
                evidence_list.append(Evidence(
                    source=f"{row.source} - {row.metric_name}",
                    source_type="data_point",
                    content=f"{row.metric_name}: {row.value} {row.unit or ''}",
                    value=float(row.value) if row.value else None,
                    date=str(row.date_recorded) if row.date_recorded else None,
                    relevance_score=1.0,
                ))

            # Fuzzy match (partial) if still need more
            if len(evidence_list) < 5:
                if start_date is not None and end_date is not None:
                    fuzzy_stmt = text("""
                        SELECT metric_name, value, unit, date_recorded, source
                        FROM data_points
                        WHERE metric_name ILIKE :fuzzy_metric
                        AND date_recorded >= :start_date
                        AND date_recorded <= :end_date
                        ORDER BY date_recorded DESC
                        LIMIT :remaining
                    """)
                    remaining = 5 - len(evidence_list)
                    result = conn.execute(fuzzy_stmt, {
                        "fuzzy_metric": f"%{metric_name}%",
                        "start_date": start_date,
                        "end_date": end_date,
                        "remaining": remaining
                    })
                elif start_date is not None and end_date is None:
                    fuzzy_stmt = text("""
                        SELECT metric_name, value, unit, date_recorded, source
                        FROM data_points
                        WHERE metric_name ILIKE :fuzzy_metric
                        AND date_recorded >= :start_date
                        ORDER BY date_recorded DESC
                        LIMIT :remaining
                    """)
                    remaining = 5 - len(evidence_list)
                    result = conn.execute(fuzzy_stmt, {
                        "fuzzy_metric": f"%{metric_name}%",
                        "start_date": start_date,
                        "remaining": remaining
                    })
                else:
                    fuzzy_stmt = text("""
                        SELECT metric_name, value, unit, date_recorded, source
                        FROM data_points
                        WHERE metric_name ILIKE :fuzzy_metric
                        ORDER BY date_recorded DESC
                        LIMIT :remaining
                    """)
                    remaining = 5 - len(evidence_list)
                    result = conn.execute(fuzzy_stmt, {
                        "fuzzy_metric": f"%{metric_name}%",
                        "remaining": remaining
                    })
                rows = result.fetchall()

                for row in rows:
                    evidence_list.append(Evidence(
                        source=f"{row.source} - {row.metric_name}",
                        source_type="data_point",
                        content=f"{row.metric_name}: {row.value} {row.unit or ''}",
                        value=float(row.value) if row.value else None,
                        date=str(row.date_recorded) if row.date_recorded else None,
                        relevance_score=0.8,
                    ))

    return evidence_list[:5]


def lookup_company(claim: Claim) -> list[Evidence]:
    """Lookup company facts from SEC filings.

    For COMPANY_FACT claims:
    - Extract ticker from entities
    - Query filing_facts joined with companies and filings
    - Match on metric_name and period near time_reference
    """
    db.initialize()
    ticker = _extract_ticker(claim.entities)
    if not ticker:
        return []

    metric_name = _extract_metric_name(claim.entities)
    time_ref = _parse_time_reference(claim.time_reference)

    evidence_list = []

    with db.sync_engine.connect() as conn:
        # Base query for filing facts
        stmt = text("""
            SELECT
                c.ticker,
                c.name AS company_name,
                ff.metric_name,
                ff.value,
                ff.unit,
                ff.period_end,
                f.form_type
            FROM filing_facts ff
            JOIN filings f ON ff.filing_id = f.id
            JOIN companies c ON f.company_id = c.id
            WHERE c.ticker = :ticker
            AND ff.metric_name ILIKE :metric
            ORDER BY ff.period_end DESC
            LIMIT 10
        """)

        result = conn.execute(stmt, {
            "ticker": ticker,
            "metric": f"%{metric_name}%" if metric_name else "%"
        })
        rows = result.fetchall()

        for row in rows:
            # Calculate relevance based on time proximity
            relevance = 1.0
            if time_ref and row.period_end:
                period_date = row.period_end
                days_diff = abs((period_date - time_ref.date()).days) if hasattr(period_date, 'date') else abs((period_date - time_ref).days)
                if days_diff > 365:
                    relevance = 0.5
                elif days_diff > 180:
                    relevance = 0.7

            evidence_list.append(Evidence(
                source=f"{row.company_name} ({row.ticker}) - {row.form_type}",
                source_type="filing_fact",
                content=f"{row.metric_name}: {row.value} {row.unit or ''} (period ending {row.period_end})",
                value=float(row.value) if row.value else None,
                date=str(row.period_end) if row.period_end else None,
                relevance_score=relevance,
            ))

    return evidence_list[:10]


def lookup_semantic(claim: Claim) -> list[Evidence]:
    """Semantic search across embeddings table.

    For CONCEPT_DEFINITION, CAUSAL_CLAIM, COMPARATIVE_CLAIM:
    - Generate embedding using all-MiniLM-L6-v2
    - Query embeddings using cosine similarity
    - Return top 5 semantically similar content
    """
    db.initialize()
    embedder = Embedder()

    # Generate embedding for the claim
    query_embedding = embedder.embed(claim.normalized_text)
    embedding_array = query_embedding.tolist()

    evidence_list = []

    with db.sync_engine.connect() as conn:
        # Cosine similarity search using pgvector
        # Note: Use CAST(?::vector) or literal ::vector to avoid SQLAlchemy bind param conflict
        stmt = text("""
            SELECT
                content_text,
                meta,
                source_type,
                1 - (embedding <=> CAST(:query_vector AS vector)) as similarity
            FROM embeddings
            WHERE 1 - (embedding <=> CAST(:query_vector AS vector)) > 0.3
            ORDER BY similarity DESC
            LIMIT 5
        """)

        result = conn.execute(stmt, {"query_vector": embedding_array})
        rows = result.fetchall()

        for row in rows:
            source_type = row.source_type
            metadata = row.meta or {}

            # Build source description from metadata
            if source_type == "research_paper":
                source = metadata.get("title", "Research Paper")
            elif source_type == "earnings_transcript":
                source = f"{metadata.get('ticker', '')} Q{metadata.get('fiscal_quarter', '')}{metadata.get('fiscal_year', '')}"
            elif source_type == "data_point":
                source = f"{metadata.get('metric_name', 'Data Point')} ({metadata.get('source', '')})"
            elif source_type == "filing_fact":
                source = f"{metadata.get('company_name', '')} ({metadata.get('ticker', '')})"
            else:
                source = source_type

            evidence_list.append(Evidence(
                source=source,
                source_type=source_type,
                content=row.content_text,
                value=None,
                date=metadata.get("date_recorded") or metadata.get("published_date"),
                relevance_score=float(row.similarity),
            ))

    return evidence_list


def lookup_news(claim: Claim) -> list[Evidence]:
    """Lookup recent news articles for contradictions or corroboration.

    For any claim type - check news from last 30 days:
    - Query news_articles where published_at > NOW() - 30 days
    - Match on tickers_mentioned overlapping with claim.entities
    - Return relevant recent articles
    """
    db.initialize()
    ticker = _extract_ticker(claim.entities)

    evidence_list = []

    with db.sync_engine.connect() as conn:
        # Query recent news articles
        stmt = text("""
            SELECT
                headline,
                source_name,
                content_summary,
                published_at,
                tickers_mentioned,
                sentiment_score
            FROM news_articles
            WHERE published_at > NOW() - INTERVAL '30 days'
            AND (:ticker IS NULL OR tickers_mentioned && ARRAY[:ticker]::text[])
            ORDER BY published_at DESC
            LIMIT 10
        """)

        result = conn.execute(stmt, {"ticker": ticker})
        rows = result.fetchall()

        for row in rows:
            # Calculate relevance based on sentiment (potential contradiction indicator)
            relevance = 0.7
            if row.sentiment_score is not None:
                # Negative sentiment might indicate contradiction
                if row.sentiment_score < -0.3:
                    relevance = 0.85

            evidence_list.append(Evidence(
                source=f"{row.source_name or 'News'}",
                source_type="news_article",
                content=f"{row.headline}. {row.content_summary or ''}",
                value=None,
                date=str(row.published_at) if row.published_at else None,
                relevance_score=relevance,
            ))

    return evidence_list[:10]


def route_claim(claim: Claim) -> list[Evidence]:
    """Route claim to appropriate lookup functions based on claim_type."""
    all_evidence = []

    if claim.claim_type in ["STATISTIC", "RATE", "MACRO_FACT"]:
        all_evidence.extend(lookup_numeric(claim))
        all_evidence.extend(lookup_news(claim))

    elif claim.claim_type == "COMPANY_FACT":
        all_evidence.extend(lookup_company(claim))
        all_evidence.extend(lookup_news(claim))

    elif claim.claim_type == "CONCEPT_DEFINITION":
        all_evidence.extend(lookup_semantic(claim))

    elif claim.claim_type in ["CAUSAL_CLAIM", "COMPARATIVE_CLAIM"]:
        all_evidence.extend(lookup_semantic(claim))
        all_evidence.extend(lookup_numeric(claim))

    elif claim.claim_type == "PREDICTION":
        all_evidence.extend(lookup_news(claim))
        all_evidence.extend(lookup_semantic(claim))

    else:
        # Unknown claim type - try semantic search
        all_evidence.extend(lookup_semantic(claim))

    return all_evidence


def verify_claim(claim: Claim) -> VerificationResult:
    """Verify a single claim against the database.

    Args:
        claim: Claim object to verify

    Returns:
        VerificationResult with status, evidence, and explanation
    """
    evidence = route_claim(claim)

    if not evidence:
        return VerificationResult(
            claim=claim,
            status=VerificationStatus.UNVERIFIABLE,
            confidence=0.0,
            evidence=[],
            correction=None,
            explanation="No relevant evidence found in database for this claim.",
        )

    # Analyze evidence to determine verification status
    supporting = [e for e in evidence if e.relevance_score >= 0.7]
    contradicting = [e for e in evidence if e.relevance_score < 0.5]

    # Simple heuristic for status determination
    if len(supporting) >= 3 and not contradicting:
        status = VerificationStatus.VERIFIED
        confidence = min(0.95, sum(e.relevance_score for e in supporting) / len(supporting))
        explanation = f"Claim is supported by {len(supporting)} high-relevance evidence sources."

    elif contradicting and len(contradicting) > len(supporting):
        status = VerificationStatus.CONTRADICTED
        confidence = sum(1 - e.relevance_score for e in contradicting) / len(contradicting)
        explanation = f"Claim is contradicted by {len(contradicting)} evidence sources."

    elif supporting and contradicting:
        status = VerificationStatus.PARTIALLY_CORRECT
        confidence = 0.6
        explanation = "Evidence is mixed - some sources support while others contradict the claim."

    elif supporting:
        # Check if evidence is outdated
        oldest_evidence = min(supporting, key=lambda e: e.date or "")
        if oldest_evidence.date:
            try:
                evidence_date = datetime.fromisoformat(oldest_evidence.date.replace("Z", "+00:00"))
                if (datetime.now() - evidence_date).days > 365:
                    status = VerificationStatus.OUTDATED
                    confidence = 0.5
                    explanation = "Supporting evidence exists but may be outdated (>1 year old)."
                else:
                    status = VerificationStatus.VERIFIED
                    confidence = 0.75
                    explanation = f"Claim is supported by {len(supporting)} evidence sources."
            except (ValueError, TypeError):
                status = VerificationStatus.VERIFIED
                confidence = 0.75
                explanation = f"Claim is supported by {len(supporting)} evidence sources."
        else:
            status = VerificationStatus.VERIFIED
            confidence = 0.75
            explanation = f"Claim is supported by {len(supporting)} evidence sources."

    else:
        status = VerificationStatus.UNVERIFIABLE
        confidence = 0.3
        explanation = "Insufficient high-quality evidence to verify this claim."

    # Generate correction if contradicted
    correction = None
    if status == VerificationStatus.CONTRADICTED and contradicting:
        best_evidence = max(contradicting, key=lambda e: 1 - e.relevance_score)
        if best_evidence.value is not None:
            correction = f"According to {best_evidence.source}, the value is {best_evidence.value}"

    return VerificationResult(
        claim=claim,
        status=status,
        confidence=confidence,
        evidence=evidence,
        correction=correction,
        explanation=explanation,
    )
