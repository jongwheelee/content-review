# Content Review Pipeline

A FastAPI-based data ingestion and processing pipeline for financial and economic data sources.

## Features

- **Multi-source data ingestion**: FRED, SEC EDGAR, Alpha Vantage, Polygon.io, BLS, News APIs, NBER, earnings transcripts
- **Async database operations**: SQLAlchemy ORM with asyncpg driver and pgvector support
- **Automated scheduling**: APScheduler for periodic data collection jobs
- **Text embeddings**: sentence-transformers integration for semantic search

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

### Start the API server

```bash
uvicorn main:app --reload
```

### Run scheduler jobs

```bash
python -m scheduler.jobs
```

## Project Structure

```
content-review-pipeline/
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
├── requirements.txt
├── .env.example
└── README.md
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/ingest/{source}` | Trigger manual ingestion |
| GET | `/data/search` | Search ingested content |
| GET | `/jobs/status` | Get scheduler job status |

## Development

```bash
# Run tests
pytest

# Format code
black .

# Lint
flake8 .
```
