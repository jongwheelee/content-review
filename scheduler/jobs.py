"""Scheduled jobs for Content Review Pipeline using APScheduler."""

import asyncio
import logging
import os
from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from dotenv import load_dotenv

from database.connection import db
from database.models import IngestionLog, IngestionStatus, DataSource

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JobScheduler:
    """Manages scheduled data ingestion jobs using AsyncIOScheduler."""

    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self._initialized = False

    def initialize(self):
        """Initialize the scheduler and register all jobs."""
        if self._initialized:
            return

        # Initialize database connection
        db.initialize()

        # Register all scheduled jobs
        self._register_jobs()

        self._initialized = True
        logger.info("Job scheduler initialized")

    def _register_jobs(self):
        """Register all scheduled jobs with their triggers."""

        # 1. FRED economic data — daily at 6:00 AM UTC
        self.scheduler.add_job(
            self.refresh_fred,
            CronTrigger(hour=6, minute=0, timezone="UTC"),
            id="refresh_fred",
            name="FRED Economic Data",
        )

        # 2. BLS data — 1st of month at 7:00 AM UTC
        self.scheduler.add_job(
            self.refresh_bls,
            CronTrigger(day=1, hour=7, minute=0, timezone="UTC"),
            id="refresh_bls",
            name="BLS Monthly Data",
        )

        # 3. Alpha Vantage market data — weekdays at 6:30 AM UTC
        self.scheduler.add_job(
            self.refresh_alpha_vantage,
            CronTrigger(day_of_week="mon-fri", hour=6, minute=30, timezone="UTC"),
            id="refresh_alpha_vantage",
            name="Alpha Vantage Market Data",
        )

        # 4. Polygon market data — weekdays at 6:45 AM UTC (15 min after Alpha Vantage)
        self.scheduler.add_job(
            self.refresh_polygon,
            CronTrigger(day_of_week="mon-fri", hour=6, minute=45, timezone="UTC"),
            id="refresh_polygon",
            name="Polygon Market Data",
        )

        # 5. SEC EDGAR filings — Sundays at 8:00 AM UTC
        self.scheduler.add_job(
            self.refresh_edgar,
            CronTrigger(day_of_week="sun", hour=8, minute=0, timezone="UTC"),
            id="refresh_edgar",
            name="SEC EDGAR Filings",
        )

        # 7. News headlines — every 6 hours (0, 6, 12, 18 UTC)
        self.scheduler.add_job(
            self.refresh_news_headlines,
            CronTrigger(hour="*/6", minute=0, timezone="UTC"),
            id="refresh_news_headlines",
            name="News Headlines",
        )

        # 8. News topic queries — daily at 7:30 AM UTC
        self.scheduler.add_job(
            self.refresh_news_topics,
            CronTrigger(hour=7, minute=30, timezone="UTC"),
            id="refresh_news_topics",
            name="News Topic Queries",
        )

        # 9. NBER working papers — Sundays at 9:00 AM UTC
        self.scheduler.add_job(
            self.refresh_nber,
            CronTrigger(day_of_week="sun", hour=9, minute=0, timezone="UTC"),
            id="refresh_nber",
            name="NBER Working Papers",
        )

        # 10. Earnings transcripts — Sundays at 10:00 AM UTC
        self.scheduler.add_job(
            self.refresh_transcripts,
            CronTrigger(day_of_week="sun", hour=10, minute=0, timezone="UTC"),
            id="refresh_transcripts",
            name="Earnings Transcripts",
        )

        # 11. Run embedder — daily at 11:00 AM UTC (after all sources refresh)
        self.scheduler.add_job(
            self.run_embedder,
            CronTrigger(hour=11, minute=0, timezone="UTC"),
            id="run_embedder",
            name="Generate Embeddings",
        )

    async def _log_job_start(self, source: DataSource) -> int:
        """Log job start to ingestion_log table.

        Args:
            source: Data source enum value

        Returns:
            Log entry ID
        """
        async with db.async_session() as session:
            log_entry = IngestionLog(
                source=source,
                status=IngestionStatus.RUNNING,
                records_fetched=0,
                errors=None,
            )
            session.add(log_entry)
            await session.flush()
            return log_entry.id

    async def _log_job_complete(
        self,
        log_id: int,
        source: DataSource,
        status: IngestionStatus,
        records_fetched: int = 0,
        errors: str | None = None,
    ):
        """Log job completion to ingestion_log table.

        Args:
            log_id: Log entry ID to update
            source: Data source enum value
            status: Job status
            records_fetched: Number of records processed
            errors: Error message if any
        """
        async with db.async_session() as session:
            log_entry = await session.get(IngestionLog, log_id)
            if log_entry:
                log_entry.status = status
                log_entry.records_fetched = records_fetched
                log_entry.errors = errors

    async def refresh_fred(self):
        """Refresh FRED economic data.

        Scheduled: Daily at 6:00 AM UTC
        """
        logger.info("=== Starting FRED refresh ===")
        log_id = await self._log_job_start(DataSource.FRED)

        try:
            from ingestion.fred import run as run_fred

            result = await run_fred()
            total_records = result.get("total_records", 0)
            errors = result.get("errors", [])

            await self._log_job_complete(
                log_id,
                DataSource.FRED,
                IngestionStatus.COMPLETED if not errors else IngestionStatus.FAILED,
                records_fetched=total_records,
                errors=" | ".join(errors) if errors else None,
            )
            logger.info(f"FRED refresh complete: {total_records} records")

        except Exception as e:
            logger.error(f"FRED refresh failed: {e}")
            await self._log_job_complete(
                log_id,
                DataSource.FRED,
                IngestionStatus.FAILED,
                errors=str(e),
            )
            raise

    async def refresh_bls(self):
        """Refresh BLS economic data.

        Scheduled: 1st of month at 7:00 AM UTC
        """
        logger.info("=== Starting BLS refresh ===")
        log_id = await self._log_job_start(DataSource.BLS)

        try:
            from ingestion.bls import run as run_bls

            result = await run_bls()
            total_records = result.get("total_records", 0)
            errors = result.get("errors", [])

            await self._log_job_complete(
                log_id,
                DataSource.BLS,
                IngestionStatus.COMPLETED if not errors else IngestionStatus.FAILED,
                records_fetched=total_records,
                errors=" | ".join(errors) if errors else None,
            )
            logger.info(f"BLS refresh complete: {total_records} records")

        except Exception as e:
            logger.error(f"BLS refresh failed: {e}")
            await self._log_job_complete(
                log_id,
                DataSource.BLS,
                IngestionStatus.FAILED,
                errors=str(e),
            )
            raise

    async def refresh_alpha_vantage(self):
        """Refresh Alpha Vantage market data.

        Scheduled: Weekdays at 6:30 AM UTC
        """
        logger.info("=== Starting Alpha Vantage refresh ===")
        log_id = await self._log_job_start(DataSource.ALPHA_VANTAGE)

        try:
            from ingestion.alpha_vantage import run as run_alpha_vantage

            result = await run_alpha_vantage()
            total_records = (
                result.get("total_daily_records", 0)
                + result.get("total_weekly_records", 0)
            )
            errors = result.get("errors", [])

            await self._log_job_complete(
                log_id,
                DataSource.ALPHA_VANTAGE,
                IngestionStatus.COMPLETED if not errors else IngestionStatus.FAILED,
                records_fetched=total_records,
                errors=" | ".join(errors) if errors else None,
            )
            logger.info(f"Alpha Vantage refresh complete: {total_records} records")

        except Exception as e:
            logger.error(f"Alpha Vantage refresh failed: {e}")
            await self._log_job_complete(
                log_id,
                DataSource.ALPHA_VANTAGE,
                IngestionStatus.FAILED,
                errors=str(e),
            )
            raise

    async def refresh_polygon(self):
        """Refresh Polygon.io market data.

        Scheduled: Weekdays at 6:45 AM UTC
        """
        logger.info("=== Starting Polygon refresh ===")
        log_id = await self._log_job_start(DataSource.POLYGON)

        try:
            from ingestion.polygon import run as run_polygon

            result = await run_polygon()
            total_records = (
                result.get("total_market_records", 0)
                + result.get("total_companies", 0)
                + result.get("total_options", 0)
            )
            errors = result.get("errors", [])

            await self._log_job_complete(
                log_id,
                DataSource.POLYGON,
                IngestionStatus.COMPLETED if not errors else IngestionStatus.FAILED,
                records_fetched=total_records,
                errors=" | ".join(errors) if errors else None,
            )
            logger.info(f"Polygon refresh complete: {total_records} records")

        except Exception as e:
            logger.error(f"Polygon refresh failed: {e}")
            await self._log_job_complete(
                log_id,
                DataSource.POLYGON,
                IngestionStatus.FAILED,
                errors=str(e),
            )
            raise

    async def refresh_edgar(self):
        """Refresh SEC EDGAR filings.

        Scheduled: Sundays at 8:00 AM UTC
        """
        logger.info("=== Starting SEC EDGAR refresh ===")
        log_id = await self._log_job_start(DataSource.SEC_EDGAR)

        try:
            from ingestion.sec_edgar import run as run_edgar

            result = await run_edgar()
            total_records = result.get("total_filings", 0) + result.get("total_facts", 0)
            errors = result.get("errors", [])

            await self._log_job_complete(
                log_id,
                DataSource.SEC_EDGAR,
                IngestionStatus.COMPLETED if not errors else IngestionStatus.FAILED,
                records_fetched=total_records,
                errors=" | ".join(errors) if errors else None,
            )
            logger.info(f"SEC EDGAR refresh complete: {total_records} records")

        except Exception as e:
            logger.error(f"SEC EDGAR refresh failed: {e}")
            await self._log_job_complete(
                log_id,
                DataSource.SEC_EDGAR,
                IngestionStatus.FAILED,
                errors=str(e),
            )
            raise

    async def refresh_news_headlines(self):
        """Refresh news headlines.

        Scheduled: Every 6 hours (0, 6, 12, 18 UTC)
        """
        logger.info("=== Starting news headlines refresh ===")
        log_id = await self._log_job_start(DataSource.NEWS)

        try:
            from ingestion.news import run as run_news

            # Fetch headlines only for frequent runs
            result = await run_news(fetch_headlines=True, fetch_topic_queries=False)
            total_articles = result.get("total_articles", 0)
            errors = result.get("errors", [])

            await self._log_job_complete(
                log_id,
                DataSource.NEWS,
                IngestionStatus.COMPLETED if not errors else IngestionStatus.FAILED,
                records_fetched=total_articles,
                errors=" | ".join(errors) if errors else None,
            )
            logger.info(f"News headlines refresh complete: {total_articles} articles")

        except Exception as e:
            logger.error(f"News headlines refresh failed: {e}")
            await self._log_job_complete(
                log_id,
                DataSource.NEWS,
                IngestionStatus.FAILED,
                errors=str(e),
            )
            raise

    async def refresh_news_topics(self):
        """Refresh news with topic-specific queries.

        Scheduled: Daily at 7:30 AM UTC
        """
        logger.info("=== Starting news topics refresh ===")
        log_id = await self._log_job_start(DataSource.NEWS)

        try:
            from ingestion.news import run as run_news

            # Fetch topic queries for daily run
            result = await run_news(fetch_headlines=False, fetch_topic_queries=True)
            total_articles = result.get("total_articles", 0)
            errors = result.get("errors", [])

            await self._log_job_complete(
                log_id,
                DataSource.NEWS,
                IngestionStatus.COMPLETED if not errors else IngestionStatus.FAILED,
                records_fetched=total_articles,
                errors=" | ".join(errors) if errors else None,
            )
            logger.info(f"News topics refresh complete: {total_articles} articles")

        except Exception as e:
            logger.error(f"News topics refresh failed: {e}")
            await self._log_job_complete(
                log_id,
                DataSource.NEWS,
                IngestionStatus.FAILED,
                errors=str(e),
            )
            raise

    async def refresh_nber(self):
        """Refresh NBER working papers.

        Scheduled: Sundays at 9:00 AM UTC
        """
        logger.info("=== Starting NBER refresh ===")
        log_id = await self._log_job_start(DataSource.NBER)

        try:
            from ingestion.nber import run as run_nber

            result = await run_nber()
            total_papers = result.get("total_matched", 0)
            errors = result.get("errors", [])

            await self._log_job_complete(
                log_id,
                DataSource.NBER,
                IngestionStatus.COMPLETED if not errors else IngestionStatus.FAILED,
                records_fetched=total_papers,
                errors=" | ".join(errors) if errors else None,
            )
            logger.info(f"NBER refresh complete: {total_papers} papers")

        except Exception as e:
            logger.error(f"NBER refresh failed: {e}")
            await self._log_job_complete(
                log_id,
                DataSource.NBER,
                IngestionStatus.FAILED,
                errors=str(e),
            )
            raise

    async def refresh_transcripts(self):
        """Refresh earnings call transcripts.

        Scheduled: Sundays at 10:00 AM UTC
        """
        logger.info("=== Starting earnings transcripts refresh ===")
        log_id = await self._log_job_start(DataSource.EARNINGS_TRANSCRIPTS)

        try:
            from ingestion.earnings_transcripts import run as run_transcripts

            result = await run_transcripts()
            total_transcripts = result.get("total_inserted", 0)
            errors = result.get("errors", [])

            await self._log_job_complete(
                log_id,
                DataSource.EARNINGS_TRANSCRIPTS,
                IngestionStatus.COMPLETED if not errors else IngestionStatus.FAILED,
                records_fetched=total_transcripts,
                errors=" | ".join(errors) if errors else None,
            )
            logger.info(f"Earnings transcripts refresh complete: {total_transcripts} transcripts")

        except Exception as e:
            logger.error(f"Earnings transcripts refresh failed: {e}")
            await self._log_job_complete(
                log_id,
                DataSource.EARNINGS_TRANSCRIPTS,
                IngestionStatus.FAILED,
                errors=str(e),
            )
            raise

    async def run_embedder(self):
        """Generate embeddings for all unembedded content.

        Scheduled: Daily at 11:00 AM UTC (after all source refreshes)
        """
        logger.info("=== Starting embedding generation ===")
        # Use FRED as proxy source for embedder logging
        log_id = await self._log_job_start(DataSource.FRED)

        try:
            from processing.embedder import run_all as run_embedder_all

            result = await run_embedder_all()
            total_embeddings = result.get("total", 0)
            errors = result.get("errors", [])

            await self._log_job_complete(
                log_id,
                DataSource.FRED,
                IngestionStatus.COMPLETED if not errors else IngestionStatus.FAILED,
                records_fetched=total_embeddings,
                errors=" | ".join(errors) if errors else None,
            )
            logger.info(f"Embedding generation complete: {total_embeddings} embeddings")

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            await self._log_job_complete(
                log_id,
                DataSource.FRED,
                IngestionStatus.FAILED,
                errors=str(e),
            )
            raise

    def start(self):
        """Start the scheduler."""
        self.initialize()
        self.scheduler.start()
        logger.info("Scheduler started")

    def shutdown(self, wait: bool = True):
        """Shutdown the scheduler.

        Args:
            wait: Whether to wait for pending jobs to complete
        """
        self.scheduler.shutdown(wait=wait)
        logger.info("Scheduler shutdown")

    def get_job_status(self) -> list[dict]:
        """Get status of all scheduled jobs.

        Returns:
            List of job status dicts with id, name, next_run, trigger
        """
        return [
            {
                "id": job.id,
                "name": job.name,
                "next_run": job.next_run_time.isoformat()
                if job.next_run_time
                else None,
                "trigger": str(job.trigger),
            }
            for job in self.scheduler.get_jobs()
        ]

    def print_schedule(self):
        """Print all scheduled jobs and their next run times."""
        logger.info("\n" + "=" * 60)
        logger.info("SCHEDULED JOBS")
        logger.info("=" * 60)

        jobs = self.scheduler.get_jobs()
        for job in jobs:
            next_run = job.next_run_time
            next_run_str = next_run.strftime("%Y-%m-%d %H:%M:%S UTC") if next_run else "N/A"
            logger.info(f"  {job.name}")
            logger.info(f"    ID: {job.id}")
            logger.info(f"    Trigger: {job.trigger}")
            logger.info(f"    Next run: {next_run_str}")
            logger.info("")

        logger.info("=" * 60)


# Global scheduler instance
scheduler = JobScheduler()


async def keep_alive():
    """Keep the event loop alive for the scheduler."""
    while True:
        await asyncio.sleep(1)


def main():
    """Main entry point - starts scheduler and keeps it running."""
    # Initialize and start scheduler
    scheduler.initialize()
    scheduler.start()

    # Print schedule on startup
    scheduler.print_schedule()

    logger.info("Scheduler is running. Press Ctrl+C to stop.")

    # Keep the event loop alive
    try:
        asyncio.run(keep_alive())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutting down scheduler...")
        scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped")


if __name__ == "__main__":
    main()
