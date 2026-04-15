"""FastAPI web application for Fact-Check Pipeline.

Provides a web UI for submitting content, running fact-checks,
and viewing results with HTMX-powered async updates.
"""

import asyncio
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Request, Form

# Load environment variables from .env file (override existing)
load_dotenv(override=True)
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from analysis.fact_checker import run_fact_check, FactCheckReport
from analysis.summarizer import generate_summary, FactCheckedSummary


# ============================================================================
# APP SETUP
# ============================================================================

app = FastAPI(
    title="Fact-Check Pipeline",
    description="Financial content fact-checking and summarization",
    version="1.0.0",
)

# Setup templates and static files
BASE_DIR = Path(__file__).parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

TEMPLATES_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)
(STATIC_DIR / "css").mkdir(exist_ok=True)
(STATIC_DIR / "js").mkdir(exist_ok=True)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ============================================================================
# IN-MEMORY STORAGE (replace with SQLite later)
# ============================================================================

class AnalysisJob(BaseModel):
    """Represents a fact-check job."""
    id: str
    content: str
    submitted_at: datetime
    completed_at: Optional[datetime] = None
    status: str = "pending"  # pending, running, completed, failed
    summary: Optional[FactCheckedSummary] = None
    error: Optional[str] = None


# In-memory job store
jobs: dict[str, AnalysisJob] = {}


# ============================================================================
# ROUTES - PAGES
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with content input form."""
    return templates.TemplateResponse(
        request=request,
        name="home.html",
        context={"title": "Fact-Check Pipeline"}
    )


@app.get("/analysis/{job_id}", response_class=HTMLResponse)
async def analysis_result(request: Request, job_id: str):
    """Results page for a specific analysis job."""
    job = jobs.get(job_id)
    if not job:
        return templates.TemplateResponse(
            request=request,
            name="error.html",
            context={
                "title": "Not Found",
                "error": f"Analysis job {job_id} not found",
            }
        )

    return templates.TemplateResponse(
        request=request,
        name="results.html",
        context={
            "title": f"Analysis: {job_id[:8]}",
            "job": job,
            "summary": job.summary if job else None,
        }
    )


@app.get("/history", response_class=HTMLResponse)
async def history(request: Request):
    """History page showing past analyses."""
    # Sort by submitted_at, most recent first
    sorted_jobs = sorted(
        jobs.values(),
        key=lambda j: j.submitted_at,
        reverse=True,
    )
    return templates.TemplateResponse(
        request=request,
        name="history.html",
        context={
            "title": "History",
            "jobs": sorted_jobs,
        }
    )


# ============================================================================
# ROUTES - API
# ============================================================================

@app.post("/api/analyze")
async def submit_analysis(content: str = Form(...), api_key: str = Form(None)):
    """Submit content for fact-checking analysis."""
    job_id = str(uuid.uuid4())
    job = AnalysisJob(
        id=job_id,
        content=content,
        submitted_at=datetime.now(),
    )
    jobs[job_id] = job

    # Start background processing with API key
    asyncio.create_task(process_analysis(job_id, api_key))

    return JSONResponse({
        "job_id": job_id,
        "status": "pending",
        "redirect": f"/analysis/{job_id}",
    })


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    """Get status of an analysis job."""
    job = jobs.get(job_id)
    if not job:
        return JSONResponse(
            {"error": "Job not found"},
            status_code=404,
        )

    return JSONResponse({
        "job_id": job.id,
        "status": job.status,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "has_summary": job.summary is not None,
        "error": job.error,
    })


@app.get("/api/results/{job_id}")
async def get_results(job_id: str):
    """Get full results of a completed analysis."""
    job = jobs.get(job_id)
    if not job:
        return JSONResponse(
            {"error": "Job not found"},
            status_code=404,
        )

    if job.status != "completed":
        return JSONResponse(
            {"error": f"Job not completed (status: {job.status})"},
            status_code=400,
        )

    # Serialize summary
    summary_data = None
    if job.summary:
        summary_data = {
            "sections": [
                {
                    "topic": s.topic,
                    "original_narrative": s.original_narrative,
                    "corrected_narrative": s.corrected_narrative,
                    "has_corrections": s.has_corrections,
                    "claims": [
                        {
                            "original_text": c.claim.original_text,
                            "claim_type": c.claim.claim_type.value,
                            "status": c.status.value,
                            "user_facing_status": c.user_facing_status.value,
                            "confidence": c.confidence,
                            "explanation": c.explanation,
                            "correction": c.correction,
                            "evidence_count": len(c.evidence),
                        }
                        for c in s.claims
                    ],
                }
                for s in job.summary.sections
            ],
            "unverifiable_claims": [
                {
                    "original_text": c.claim.original_text,
                    "explanation": c.explanation,
                    "correction": c.correction,
                }
                for c in job.summary.unverifiable_claims
            ],
            "feedback": [
                {
                    "category": f.category,
                    "issue": f.issue,
                    "location": f.location,
                    "suggestion": f.suggestion,
                }
                for f in job.summary.feedback
            ],
            "quality_score": {
                "overall": job.summary.quality_score.overall,
                "factual_accuracy": job.summary.quality_score.factual_accuracy,
                "clarity": job.summary.quality_score.clarity,
                "completeness": job.summary.quality_score.completeness,
                "issue_counts": job.summary.quality_score.issue_counts,
            },
        }

    return JSONResponse({
        "job_id": job.id,
        "content": job.content,
        "summary": summary_data,
    })


# ============================================================================
# HTMX PARTIALS
# ============================================================================

@app.get("/partials/status/{job_id}")
async def partial_status(job_id: str):
    """HTMX partial: Job status indicator."""
    job = jobs.get(job_id)
    if not job:
        return HTMLResponse("<span class='error'>Job not found</span>")

    status_classes = {
        "pending": "status-pending",
        "running": "status-running",
        "completed": "status-completed",
        "failed": "status-failed",
    }

    status_icons = {
        "pending": "⏳",
        "running": "🔄",
        "completed": "✅",
        "failed": "❌",
    }

    status_text = f"{status_icons.get(job.status, '')} {job.status.title()}"
    if job.completed_at:
        status_text += f" at {job.completed_at.strftime('%H:%M:%S')}"

    return HTMLResponse(f"""
        <span class="{status_classes.get(job.status, '')}" style="font-family: var(--font-mono); font-size: 13px;">
            {status_text}
        </span>
    """)


@app.get("/partials/results/{job_id}")
async def partial_results(request: Request, job_id: str):
    """HTMX partial: Results content when job completes."""
    job = jobs.get(job_id)
    if not job or job.status != "completed":
        return HTMLResponse("<em>Results not ready yet...</em>")

    return templates.TemplateResponse(
        request=request,
        name="partials/_results_content.html",
        context={
            "job": job,
            "summary": job.summary,
        }
    )


# ============================================================================
# BACKGROUND PROCESSING
# ============================================================================

async def process_analysis(job_id: str, api_key: str = None):
    """Process an analysis job in the background."""
    job = jobs.get(job_id)
    if not job:
        return

    job.status = "running"

    try:
        # Use provided API key or fall back to environment variable
        if not api_key:
            import os
            api_key = os.getenv("ANTHROPIC_API_KEY")

        if not api_key:
            raise ValueError(
                "Anthropic API key not set. "
                "Please enter your API key on the home page or set "
                "the ANTHROPIC_API_KEY environment variable."
            )

        # Run fact-checking
        report = await run_fact_check(job.content, api_key)

        # Generate summary
        summary = generate_summary(job.content, report)

        # Store results
        job.summary = summary
        job.status = "completed"
        job.completed_at = datetime.now()

    except Exception as e:
        job.status = "failed"
        job.error = str(e)
        job.completed_at = datetime.now()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
