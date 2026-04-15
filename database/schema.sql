-- Financial Fact-Checking Database Schema
-- PostgreSQL with pgvector extension

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================================
-- ENUMS
-- ============================================================================

CREATE TYPE data_source AS ENUM (
    'FRED',
    'SEC_EDGAR',
    'ALPHA_VANTAGE',
    'POLYGON',
    'BLS',
    'NEWS',
    'NBER',
    'EARNINGS_TRANSCRIPTS'
);

CREATE TYPE data_category AS ENUM (
    'economic',
    'financial',
    'market',
    'corporate',
    'research'
);

CREATE TYPE ingestion_status AS ENUM (
    'pending',
    'running',
    'completed',
    'failed'
);

CREATE TYPE paper_type AS ENUM (
    'nber',
    'ssrn',
    'other'
);

-- ============================================================================
-- CORE TABLES
-- ============================================================================

-- Companies reference table
CREATE TABLE companies (
    id SERIAL PRIMARY KEY,
    cik VARCHAR(10) UNIQUE NOT NULL,
    ticker VARCHAR(10) UNIQUE,
    name VARCHAR(255) NOT NULL,
    sic_code VARCHAR(4),
    sector VARCHAR(100),
    exchange VARCHAR(50),
    market_cap NUMERIC(20, 2),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_companies_ticker ON companies(ticker);
CREATE INDEX idx_companies_sector ON companies(sector);

-- Market data (replaces generic data_points for price data)
CREATE TABLE market_data (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    date_recorded DATE NOT NULL,
    open NUMERIC(18, 6),
    high NUMERIC(18, 6),
    low NUMERIC(18, 6),
    close NUMERIC(18, 6),
    volume BIGINT,
    adjusted_close NUMERIC(18, 6),
    vwap NUMERIC(18, 6),
    source VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker, date_recorded, source)
);

CREATE INDEX idx_market_data_ticker_date ON market_data(ticker, date_recorded);
CREATE INDEX idx_market_data_date ON market_data(date_recorded);

-- Economic and financial data points
CREATE TABLE data_points (
    id SERIAL PRIMARY KEY,
    source data_source NOT NULL,
    category data_category NOT NULL,
    metric_name VARCHAR(255) NOT NULL,
    value NUMERIC,
    unit VARCHAR(50),
    date_recorded DATE NOT NULL,
    geographic_scope VARCHAR(100),
    confidence_score NUMERIC(3, 2) CHECK (confidence_score >= 0 AND confidence_score <= 1),
    raw_json JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_data_points_metric_date ON data_points(metric_name, date_recorded);
CREATE INDEX idx_data_points_source ON data_points(source);
CREATE INDEX idx_data_points_category ON data_points(category);
CREATE INDEX idx_data_points_raw_json ON data_points USING GIN (raw_json);

-- ============================================================================
-- SEC FILINGS
-- ============================================================================

CREATE TABLE filings (
    id SERIAL PRIMARY KEY,
    company_id INTEGER NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    form_type VARCHAR(20) NOT NULL,
    period_of_report DATE NOT NULL,
    filed_at TIMESTAMPTZ NOT NULL,
    accession_number VARCHAR(50) UNIQUE NOT NULL,
    filing_url TEXT,
    processed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_filings_company_id ON filings(company_id);
CREATE INDEX idx_filings_form_type ON filings(form_type);
CREATE INDEX idx_filings_period_of_report ON filings(period_of_report);
CREATE INDEX idx_filings_filed_at ON filings(filed_at);
CREATE INDEX idx_filings_processed ON filings(processed);

-- Individual facts extracted from filings
CREATE TABLE filing_facts (
    id SERIAL PRIMARY KEY,
    filing_id INTEGER NOT NULL REFERENCES filings(id) ON DELETE CASCADE,
    metric_name VARCHAR(255) NOT NULL,
    value NUMERIC,
    unit VARCHAR(50),
    period_start DATE,
    period_end DATE,
    context_label VARCHAR(255)
);

CREATE INDEX idx_filing_facts_filing_id ON filing_facts(filing_id);
CREATE INDEX idx_filing_facts_metric_name ON filing_facts(metric_name);
CREATE INDEX idx_filing_facts_period ON filing_facts(period_start, period_end);

-- ============================================================================
-- RESEARCH PAPERS (Tier 2)
-- ============================================================================

CREATE TABLE research_papers (
    id SERIAL PRIMARY KEY,
    external_id VARCHAR(100) UNIQUE,
    title TEXT NOT NULL,
    authors TEXT[],
    abstract TEXT,
    published_date DATE,
    url TEXT,
    keywords TEXT[],
    paper_type paper_type DEFAULT 'other',
    processed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_research_papers_published_date ON research_papers(published_date);
CREATE INDEX idx_research_papers_paper_type ON research_papers(paper_type);
CREATE INDEX idx_research_papers_processed ON research_papers(processed);
CREATE INDEX idx_research_papers_keywords ON research_papers USING GIN (keywords);

-- ============================================================================
-- EARNINGS TRANSCRIPTS (Tier 2)
-- ============================================================================

CREATE TABLE earnings_transcripts (
    id SERIAL PRIMARY KEY,
    company_id INTEGER NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    ticker VARCHAR(10) NOT NULL,
    fiscal_quarter INTEGER NOT NULL CHECK (fiscal_quarter BETWEEN 1 AND 4),
    fiscal_year INTEGER NOT NULL,
    transcript_text TEXT NOT NULL,
    source_url TEXT,
    filed_at TIMESTAMPTZ NOT NULL,
    processed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(company_id, fiscal_quarter, fiscal_year)
);

CREATE INDEX idx_earnings_transcripts_company_id ON earnings_transcripts(company_id);
CREATE INDEX idx_earnings_transcripts_ticker ON earnings_transcripts(ticker);
CREATE INDEX idx_earnings_transcripts_fiscal ON earnings_transcripts(fiscal_year, fiscal_quarter);
CREATE INDEX idx_earnings_transcripts_filed_at ON earnings_transcripts(filed_at);
CREATE INDEX idx_earnings_transcripts_processed ON earnings_transcripts(processed);

-- ============================================================================
-- NEWS ARTICLES (Tier 3)
-- ============================================================================

CREATE TABLE news_articles (
    id SERIAL PRIMARY KEY,
    headline TEXT NOT NULL,
    source_name VARCHAR(255),
    url TEXT UNIQUE,
    published_at TIMESTAMPTZ NOT NULL,
    content_summary TEXT,
    tickers_mentioned TEXT[],
    topics TEXT[],
    sentiment_score NUMERIC(3, 2) CHECK (sentiment_score >= -1 AND sentiment_score <= 1),
    raw_json JSONB,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_news_articles_published_at ON news_articles(published_at);
CREATE INDEX idx_news_articles_topics ON news_articles USING GIN (topics);
CREATE INDEX idx_news_articles_tickers ON news_articles USING GIN (tickers_mentioned);
CREATE INDEX idx_news_articles_source ON news_articles(source_name);

-- ============================================================================
-- EMBEDDINGS (Semantic Search)
-- ============================================================================

CREATE TABLE embeddings (
    id SERIAL PRIMARY KEY,
    source_type VARCHAR(50) NOT NULL,
    source_id INTEGER NOT NULL,
    content_text TEXT NOT NULL,
    embedding vector(384),
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_embeddings_source ON embeddings(source_type, source_id);
CREATE INDEX idx_embeddings_metadata ON embeddings USING GIN (metadata);
CREATE INDEX idx_embeddings_embedding ON embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- ============================================================================
-- INGESTION LOGGING
-- ============================================================================

CREATE TABLE ingestion_log (
    id SERIAL PRIMARY KEY,
    source data_source NOT NULL,
    status ingestion_status NOT NULL,
    records_fetched INTEGER DEFAULT 0,
    errors TEXT,
    ran_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_ingestion_log_source ON ingestion_log(source);
CREATE INDEX idx_ingestion_log_status ON ingestion_log(status);
CREATE INDEX idx_ingestion_log_ran_at ON ingestion_log(ran_at);

-- ============================================================================
-- HELPER FUNCTIONS & TRIGGERS
-- ============================================================================

-- Auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_data_points_updated_at
    BEFORE UPDATE ON data_points
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- VIEWS
-- ============================================================================

-- Latest market data per ticker
CREATE VIEW v_latest_market_data AS
SELECT DISTINCT ON (ticker)
    ticker,
    date_recorded,
    open,
    high,
    low,
    close,
    volume,
    adjusted_close,
    vwap,
    source
FROM market_data
ORDER BY ticker, date_recorded DESC;

-- Recent filings by company
CREATE VIEW v_recent_filings AS
SELECT
    c.ticker,
    c.name AS company_name,
    f.form_type,
    f.period_of_report,
    f.filed_at,
    f.accession_number
FROM filings f
JOIN companies c ON f.company_id = c.id
ORDER BY f.filed_at DESC
LIMIT 100;

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON TABLE companies IS 'Reference table for publicly traded companies';
COMMENT ON TABLE market_data IS 'Daily OHLCV market data from various sources';
COMMENT ON TABLE data_points IS 'Economic and financial metrics from multiple sources';
COMMENT ON TABLE filings IS 'SEC EDGAR filing metadata';
COMMENT ON TABLE filing_facts IS 'Extracted numeric facts from SEC filings';
COMMENT ON TABLE research_papers IS 'Academic research papers from NBER, SSRN, etc.';
COMMENT ON TABLE earnings_transcripts IS 'Earnings call transcripts by company and quarter';
COMMENT ON TABLE news_articles IS 'Financial news articles with sentiment analysis';
COMMENT ON TABLE embeddings IS 'Vector embeddings for semantic search across all content';
COMMENT ON TABLE ingestion_log IS 'Audit log for data ingestion jobs';
