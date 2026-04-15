"""SQLAlchemy ORM models for Content Review Pipeline."""

from datetime import datetime
from enum import Enum
from typing import Optional, List

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    Date,
    Numeric,
    Boolean,
    Enum as SQLEnum,
    JSON,
    Index,
    ForeignKey,
    ARRAY,
)
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP, VARCHAR
from sqlalchemy.orm import declarative_base, relationship
from pgvector.sqlalchemy import Vector

Base = declarative_base()


# ============================================================================
# ENUMS
# ============================================================================


class DataSource(str, Enum):
    """Supported data sources."""

    FRED = "FRED"
    SEC_EDGAR = "SEC_EDGAR"
    ALPHA_VANTAGE = "ALPHA_VANTAGE"
    POLYGON = "POLYGON"
    BLS = "BLS"
    NEWS = "NEWS"
    NBER = "NBER"
    EARNINGS_TRANSCRIPTS = "EARNINGS_TRANSCRIPTS"
    WIKIPEDIA = "WIKIPEDIA"


class DataCategory(str, Enum):
    """Data categories."""

    ECONOMIC = "economic"
    FINANCIAL = "financial"
    MARKET = "market"
    CORPORATE = "corporate"
    RESEARCH = "research"


class IngestionStatus(str, Enum):
    """Job execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class PaperType(str, Enum):
    """Research paper types."""

    NBER = "nber"
    SSRN = "ssrn"
    OTHER = "other"
    WIKIPEDIA = "wikipedia"


# PostgreSQL Enum types for SQLAlchemy (use values, not names)
DataSourceEnum = SQLEnum(DataSource, name="data_source", values_callable=lambda x: [e.value for e in x])
DataCategoryEnum = SQLEnum(DataCategory, name="data_category", values_callable=lambda x: [e.value for e in x])
IngestionStatusEnum = SQLEnum(IngestionStatus, name="ingestion_status", values_callable=lambda x: [e.value for e in x])
PaperTypeEnum = SQLEnum(PaperType, name="paper_type", values_callable=lambda x: [e.value for e in x])


# ============================================================================
# CORE TABLES
# ============================================================================


class Company(Base):
    """Reference table for publicly traded companies."""

    __tablename__ = "companies"

    id = Column(Integer, primary_key=True)
    cik = Column(VARCHAR(10), unique=True, nullable=False)
    ticker = Column(VARCHAR(10), unique=True)
    name = Column(VARCHAR(255), nullable=False)
    sic_code = Column(VARCHAR(4))
    sector = Column(VARCHAR(100))
    exchange = Column(VARCHAR(50))
    market_cap = Column(Numeric(20, 2))
    created_at = Column(TIMESTAMP(timezone=True), default=datetime.utcnow)

    __table_args__ = (
        Index("idx_companies_ticker", "ticker"),
        Index("idx_companies_sector", "sector"),
    )

    def __repr__(self) -> str:
        return f"<Company(ticker={self.ticker}, name={self.name})>"


class MarketData(Base):
    """Daily OHLCV market data from various sources."""

    __tablename__ = "market_data"

    id = Column(Integer, primary_key=True)
    ticker = Column(VARCHAR(10), nullable=False)
    date_recorded = Column(Date, nullable=False)
    open = Column(Numeric(18, 6))
    high = Column(Numeric(18, 6))
    low = Column(Numeric(18, 6))
    close = Column(Numeric(18, 6))
    volume = Column(Integer)
    adjusted_close = Column(Numeric(18, 6))
    vwap = Column(Numeric(18, 6))
    source = Column(VARCHAR(50))
    created_at = Column(TIMESTAMP(timezone=True), default=datetime.utcnow)

    __table_args__ = (
        Index("idx_market_data_ticker_date", "ticker", "date_recorded"),
        Index("idx_market_data_date", "date_recorded"),
    )

    def __repr__(self) -> str:
        return f"<MarketData(ticker={self.ticker}, date={self.date_recorded})>"


class DataPoint(Base):
    """Economic and financial metrics from multiple sources."""

    __tablename__ = "data_points"

    id = Column(Integer, primary_key=True)
    source = Column(DataSourceEnum, nullable=False)
    category = Column(DataCategoryEnum, nullable=False)
    metric_name = Column(VARCHAR(255), nullable=False)
    value = Column(Numeric)
    unit = Column(VARCHAR(50))
    date_recorded = Column(Date, nullable=False)
    geographic_scope = Column(VARCHAR(100))
    confidence_score = Column(Numeric(3, 2))
    raw_json = Column(JSONB, nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), default=datetime.utcnow)
    updated_at = Column(TIMESTAMP(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("idx_data_points_metric_date", "metric_name", "date_recorded"),
        Index("idx_data_points_source", "source"),
        Index("idx_data_points_category", "category"),
        Index("idx_data_points_raw_json", "raw_json", postgresql_using="gin"),
    )

    def __repr__(self) -> str:
        return f"<DataPoint(metric={self.metric_name}, source={self.source})>"


# ============================================================================
# SEC FILINGS
# ============================================================================


class Filing(Base):
    """SEC EDGAR filing metadata."""

    __tablename__ = "filings"

    id = Column(Integer, primary_key=True)
    company_id = Column(Integer, ForeignKey("companies.id", ondelete="CASCADE"), nullable=False)
    form_type = Column(VARCHAR(20), nullable=False)
    period_of_report = Column(Date, nullable=False)
    filed_at = Column(TIMESTAMP(timezone=True), nullable=False)
    accession_number = Column(VARCHAR(50), unique=True, nullable=False)
    filing_url = Column(Text)
    processed = Column(Boolean, default=False)
    created_at = Column(TIMESTAMP(timezone=True), default=datetime.utcnow)

    company = relationship("Company")

    __table_args__ = (
        Index("idx_filings_company_id", "company_id"),
        Index("idx_filings_form_type", "form_type"),
        Index("idx_filings_period_of_report", "period_of_report"),
        Index("idx_filings_filed_at", "filed_at"),
        Index("idx_filings_processed", "processed"),
    )

    def __repr__(self) -> str:
        return f"<Filing(form={self.form_type}, company_id={self.company_id})>"


class FilingFact(Base):
    """Extracted numeric facts from SEC filings."""

    __tablename__ = "filing_facts"

    id = Column(Integer, primary_key=True)
    filing_id = Column(Integer, ForeignKey("filings.id", ondelete="CASCADE"), nullable=False)
    metric_name = Column(VARCHAR(255), nullable=False)
    value = Column(Numeric)
    unit = Column(VARCHAR(50))
    period_start = Column(Date)
    period_end = Column(Date)
    context_label = Column(VARCHAR(255))

    __table_args__ = (
        Index("idx_filing_facts_filing_id", "filing_id"),
        Index("idx_filing_facts_metric_name", "metric_name"),
        Index("idx_filing_facts_period", "period_start", "period_end"),
    )

    def __repr__(self) -> str:
        return f"<FilingFact(metric={self.metric_name}, value={self.value})>"


# ============================================================================
# RESEARCH PAPERS
# ============================================================================


class ResearchPaper(Base):
    """Academic research papers from NBER, SSRN, etc."""

    __tablename__ = "research_papers"

    id = Column(Integer, primary_key=True)
    external_id = Column(VARCHAR(100), unique=True)
    title = Column(Text, nullable=False)
    authors = Column(ARRAY(String))
    abstract = Column(Text)
    published_date = Column(Date)
    url = Column(Text)
    keywords = Column(ARRAY(String))
    paper_type = Column(PaperTypeEnum, default="other")
    processed = Column(Boolean, default=False)
    created_at = Column(TIMESTAMP(timezone=True), default=datetime.utcnow)

    __table_args__ = (
        Index("idx_research_papers_published_date", "published_date"),
        Index("idx_research_papers_paper_type", "paper_type"),
        Index("idx_research_papers_processed", "processed"),
        Index("idx_research_papers_keywords", "keywords", postgresql_using="gin"),
    )

    def __repr__(self) -> str:
        return f"<ResearchPaper(title={self.title[:50]}...)>"


# ============================================================================
# EARNINGS TRANSCRIPTS
# ============================================================================


class EarningsTranscript(Base):
    """Earnings call transcripts by company and quarter."""

    __tablename__ = "earnings_transcripts"

    id = Column(Integer, primary_key=True)
    company_id = Column(Integer, ForeignKey("companies.id", ondelete="CASCADE"), nullable=False)
    ticker = Column(VARCHAR(10), nullable=False)
    fiscal_quarter = Column(Integer, nullable=False)
    fiscal_year = Column(Integer, nullable=False)
    transcript_text = Column(Text, nullable=False)
    source_url = Column(Text)
    filed_at = Column(TIMESTAMP(timezone=True), nullable=False)
    processed = Column(Boolean, default=False)
    created_at = Column(TIMESTAMP(timezone=True), default=datetime.utcnow)

    company = relationship("Company")

    __table_args__ = (
        Index("idx_earnings_transcripts_company_id", "company_id"),
        Index("idx_earnings_transcripts_ticker", "ticker"),
        Index("idx_earnings_transcripts_fiscal", "fiscal_year", "fiscal_quarter"),
        Index("idx_earnings_transcripts_filed_at", "filed_at"),
        Index("idx_earnings_transcripts_processed", "processed"),
    )

    def __repr__(self) -> str:
        return f"<EarningsTranscript(ticker={self.ticker}, Q{self.fiscal_quarter}{self.fiscal_year})>"


# ============================================================================
# NEWS ARTICLES
# ============================================================================


class NewsArticle(Base):
    """Financial news articles with sentiment analysis."""

    __tablename__ = "news_articles"

    id = Column(Integer, primary_key=True)
    headline = Column(Text, nullable=False)
    source_name = Column(VARCHAR(255))
    url = Column(Text, unique=True)
    published_at = Column(TIMESTAMP(timezone=True), nullable=False)
    content_summary = Column(Text)
    tickers_mentioned = Column(ARRAY(String))
    topics = Column(ARRAY(String))
    sentiment_score = Column(Numeric(3, 2))
    raw_json = Column(JSONB)
    created_at = Column(TIMESTAMP(timezone=True), default=datetime.utcnow)

    __table_args__ = (
        Index("idx_news_articles_published_at", "published_at"),
        Index("idx_news_articles_topics", "topics", postgresql_using="gin"),
        Index("idx_news_articles_tickers", "tickers_mentioned", postgresql_using="gin"),
        Index("idx_news_articles_source", "source_name"),
    )

    def __repr__(self) -> str:
        return f"<NewsArticle(headline={self.headline[:50]}...)>"


# ============================================================================
# EMBEDDINGS
# ============================================================================


class Embedding(Base):
    """Vector embeddings for semantic search across all content."""

    __tablename__ = "embeddings"

    id = Column(Integer, primary_key=True)
    source_type = Column(VARCHAR(50), nullable=False)
    source_id = Column(Integer, nullable=False)
    content_text = Column(Text, nullable=False)
    embedding = Column(Vector(384))
    meta = Column(JSONB)
    created_at = Column(TIMESTAMP(timezone=True), default=datetime.utcnow)

    __table_args__ = (
        Index("idx_embeddings_source", "source_type", "source_id"),
        Index("idx_embeddings_meta", "meta", postgresql_using="gin"),
        Index(
            "idx_embeddings_embedding",
            "embedding",
            postgresql_using="ivfflat",
            postgresql_ops={"embedding": "vector_cosine_ops"},
            postgresql_with={"lists": 100},
        ),
    )

    def __repr__(self) -> str:
        return f"<Embedding(source_type={self.source_type}, source_id={self.source_id})>"


# ============================================================================
# INGESTION LOGGING
# ============================================================================


class IngestionLog(Base):
    """Audit log for data ingestion jobs."""

    __tablename__ = "ingestion_log"

    id = Column(Integer, primary_key=True)
    source = Column(DataSourceEnum, nullable=False)
    status = Column(IngestionStatusEnum, nullable=False)
    records_fetched = Column(Integer, default=0)
    errors = Column(Text)
    ran_at = Column(TIMESTAMP(timezone=True), default=datetime.utcnow)

    __table_args__ = (
        Index("idx_ingestion_log_source", "source"),
        Index("idx_ingestion_log_status", "status"),
        Index("idx_ingestion_log_ran_at", "ran_at"),
    )

    def __repr__(self) -> str:
        return f"<IngestionLog(source={self.source}, status={self.status})>"
