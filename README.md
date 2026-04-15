# Content Review Pipeline

A FastAPI-based fact-checking and summarization pipeline for financial content.

## Features

- **Claim extraction**: Automatically extract verifiable financial claims from newsletters, articles, or drafts
- **Fact verification**: Check claims against authoritative sources (FRED, SEC EDGAR, BLS) using SQL database with pgvector semantic search
- **Verdict generation**: Claude-powered verdicts with explanations, corrections, and confidence scores
- **ELI5 summarizer**: Generate corrected summaries with content quality feedback
- **Temporal anchoring**: Relative time references (e.g., "last month") resolve relative to content date
- **Web UI**: FastAPI backend with HTMX-powered async updates

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

1. Copy `.env.example` to `.env`
2. Fill in your API keys and database credentials

```bash
cp .env.example .env
```

## Usage

### Start the web application

```bash
uvicorn main:app --reload
```

Then visit `http://localhost:8000` in your browser.

### Run fact-checking via CLI

```bash
python -m analysis.fact_checker "Your content here"
```

## Project Structure

```
content-review-pipeline/
├── analysis/           # Fact-checking and summarization
│   ├── claim_extractor.py    # Extract verifiable claims
│   ├── claim_verifier.py     # Lookup evidence with temporal anchoring
│   ├── verdict_generator.py  # Generate verdicts with Claude
│   ├── fact_checker.py       # Orchestrate complete pipeline
│   └── summarizer.py         # Generate corrected summaries
├── ingestion/          # Data source connectors
│   ├── fred.py         # Federal Reserve Economic Data
│   ├── sec_edgar.py    # SEC filings
│   ├── alpha_vantage.py # Market data
│   ├── polygon.py      # Stock market data
│   ├── bls.py          # Bureau of Labor Statistics
│   ├── news.py         # News API
│   ├── nber.py         # National Bureau of Economic Research
│   └── earnings_transcripts.py  # Earnings call transcripts
├── processing/         # Data transformation
│   ├── cleaner.py      # Data cleaning utilities
│   └── embedder.py     # Text embedding generation
├── database/           # Database layer
│   ├── schema.sql      # Raw SQL schema
│   ├── models.py       # SQLAlchemy ORM models
│   └── connection.py   # Database connection management
├── scheduler/          # Scheduled jobs
│   └── jobs.py         # APScheduler job definitions
├── templates/          # Jinja2 HTML templates
├── main.py             # FastAPI application
├── requirements.txt
├── .env.example
└── README.md
```

## Key Concepts

### Temporal Anchoring

When content references relative time (e.g., "last month", "last quarter"), the system resolves these references relative to when the content was written, not the current date.

Example:
- Content dated `2024-06-15` says "inflation last month"
- System queries May 2024 data (not current month)
- Returns: 3.39% YoY inflation for May 2024

### Metric Name Mapping

Common terms are mapped to database metric names:
- "inflation" → CPILFESL, CPIAUCSL
- "interest rate" → FEDFUNDS, DGS10, DGS2
- "unemployment" → UNRATE
- "GDP" → GDP, GDPC1

### YoY Inflation Calculation

Uses SQL `LAG(12)` window function to calculate year-over-year inflation from CPI data.

## Development

```bash
# Run tests
pytest

# Format code
black .

# Lint
flake8 .
```
