# Fact-Check Pipeline Web App

A local web application for fact-checking financial content with HTMX-powered async updates.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Anthropic API key
export ANTHROPIC_API_KEY=your_api_key_here

# 3. Run the server
python main.py

# 4. Open browser
open http://localhost:8000
```

## Features

- **Home Page** (`/`) - Paste content and submit for analysis
- **Results Page** (`/analysis/<id>`) - View fact-check results with live status updates
- **History Page** (`/history`) - Browse past analyses from current session

## Tech Stack

- **Backend**: FastAPI + Jinja2 templates
- **Frontend**: HTMX for async updates, no build step required
- **Theme**: Bloomberg Terminal-inspired (dark, monospace, sharp edges)

## Project Structure

```
content-review-pipeline/
├── main.py                      # FastAPI web application
├── analysis/
│   ├── claim_extractor.py       # Extract claims using Claude
│   ├── claim_verifier.py        # Verify claims against database
│   ├── verdict_generator.py     # Generate verdicts with Claude
│   ├── fact_checker.py          # Orchestrate fact-checking
│   └── summarizer.py            # Generate corrected summaries
├── templates/
│   ├── base.html                # Base template with CSS theme
│   ├── home.html                # Content input form
│   ├── results.html             # Results display with tabs
│   ├── history.html             # Past analyses
│   ├── error.html               # Error page
│   └── partials/
│       ├── _status.html         # Job status indicator
│       ├── _summary_content.html
│       ├── _factcheck_content.html
│       └── _feedback_content.html
└── static/                      # Static assets (if needed)
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Home page |
| GET | `/history` | History page |
| GET | `/analysis/{id}` | Results page |
| POST | `/api/analyze` | Submit content for analysis |
| GET | `/api/status/{id}` | Get job status |
| GET | `/api/results/{id}` | Get full results JSON |
| GET | `/partials/status/{id}` | HTMX status partial |

## Quality Score Calculation

Weighted average of three dimensions:

| Dimension | Weight | Based On |
|-----------|--------|----------|
| Factual Accuracy | 50% | % of claims verified as correct |
| Clarity | 25% | Tone + ambiguity issues |
| Completeness | 25% | Structure + length issues |

## Content Quality Checks

- **Tone**: Sensationalist language, overconfidence
- **Ambiguity**: Vague time references, missing baselines
- **Structure**: Unsupported causal claims
- **Length**: Verbose sentences, imbalanced coverage

## Design Notes

- No rounded corners (sharp, professional aesthetic)
- Monospace fonts for data and code
- Color coding: green=verified, red=contradicted, yellow=unverifiable
- Orange accent color for primary actions
- Dark theme optimized for long reading sessions
