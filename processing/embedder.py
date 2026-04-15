"""Text embedding generation using sentence-transformers.

Generates vector embeddings for semantic search across all data sources.
Uses the all-MiniLM-L6-v2 model (384 dimensions).
"""

import asyncio
import logging
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy import select, func
from sqlalchemy.dialects.postgresql import insert
from dotenv import load_dotenv

from database.connection import db

load_dotenv()
from database.models import (
    DataPoint,
    DataSource,
    DataCategory,
    Embedding,
    FilingFact,
    Filing,
    Company,
    ResearchPaper,
    EarningsTranscript,
    IngestionLog,
    IngestionStatus,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Model configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
BATCH_SIZE = 64

# Chunking configuration for transcripts
CHUNK_SIZE = 1000  # characters
CHUNK_OVERLAP = 200  # characters


class Embedder:
    """Generate text embeddings using sentence-transformers."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the embedding model."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for a single text.

        Args:
            text: Input text to embed.

        Returns:
            Numpy array of shape (384,).
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding

    def embed_batch(self, texts: list[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to embed.
            batch_size: Batch size for inference.

        Returns:
            Numpy array of shape (n_texts, 384).
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        return embeddings


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks.

    Args:
        text: Input text to chunk.
        chunk_size: Maximum chunk size in characters.
        overlap: Overlap between chunks in characters.

    Returns:
        List of text chunks.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # Try to break at sentence boundary in the last 200 chars
        if end < len(text):
            search_text = text[max(start, end - 200):end]
            for punct in [". ", ".\n", "! ", "? ", "\n"]:
                idx = search_text.rfind(punct)
                if idx != -1:
                    end = max(start, end - 200) + idx + len(punct)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks


async def log_ingestion(
    session,
    source: DataSource,
    status: IngestionStatus,
    records_fetched: int = 0,
    errors: str | None = None,
):
    """Log embedding generation to ingestion_log table."""
    log_entry = IngestionLog(
        source=source,
        status=status,
        records_fetched=records_fetched,
        errors=errors,
    )
    session.add(log_entry)


async def embed_data_points() -> int:
    """Generate embeddings for data_points not yet embedded.

    Fetches data_points where source_type='data_point' doesn't exist in embeddings.
    Creates description using template:
    "The {metric_name} as reported by {source} on {date_recorded} was {value} {unit}.
    Category: {category}. Geographic scope: {geographic_scope}."

    Returns:
        Number of embeddings generated.
    """
    logger.info("Embedding data points...")

    db.initialize()
    embedder = Embedder()
    total_embedded = 0
    errors = []

    async with db.async_session() as session:
        # Fetch data_points not yet embedded
        # Subquery to find data_points without embeddings
        subquery = select(Embedding.source_id).where(
            Embedding.source_type == "data_point"
        )
        stmt = select(DataPoint).where(
            DataPoint.id.not_in(subquery)
        ).limit(1000)

        result = await session.execute(stmt)
        data_points = result.scalars().all()

        if not data_points:
            logger.info("  No data points to embed")
            return 0

        logger.info(f"  Found {len(data_points)} data points to embed")

        # Process in batches
        for i in range(0, len(data_points), BATCH_SIZE):
            batch = data_points[i:i + BATCH_SIZE]
            texts = []
            records = []

            for dp in batch:
                # Build description from template
                description = (
                    f"The {dp.metric_name} as reported by {dp.source.value} "
                    f"on {dp.date_recorded} was {dp.value} {dp.unit or ''}. "
                    f"Category: {dp.category.value}. "
                    f"Geographic scope: {dp.geographic_scope or 'Unknown'}."
                )

                # Build metadata
                metadata = {
                    "source": dp.source.value,
                    "category": dp.category.value,
                    "date_recorded": str(dp.date_recorded),
                    "value": float(dp.value) if dp.value else None,
                    "metric_name": dp.metric_name,
                }

                texts.append(description)
                records.append({
                    "source_type": "data_point",
                    "source_id": dp.id,
                    "content_text": description,
                    "metadata": metadata,
                })

            # Generate embeddings
            try:
                embeddings = embedder.embed_batch(texts)

                # Insert embeddings
                for record, embedding in zip(records, embeddings):
                    record["embedding"] = embedding.tolist()

                stmt = insert(Embedding).values(records)
                await session.execute(stmt)
                await session.commit()

                total_embedded += len(records)
                logger.info(f"  Embedded {len(records)} data points (batch {i // BATCH_SIZE + 1})")

            except Exception as e:
                error_msg = f"Error embedding batch {i // BATCH_SIZE + 1}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)

    # Log completion
    async with db.async_session() as session:
        await log_ingestion(
            session,
            DataSource.FRED,  # Using FRED as proxy for data_points
            IngestionStatus.COMPLETED if not errors else IngestionStatus.FAILED,
            records_fetched=total_embedded,
            errors=" | ".join(errors) if errors else None,
        )
        await session.commit()

    logger.info(f"Data points embedding complete: {total_embedded} embeddings generated")
    return total_embedded


async def embed_filing_facts() -> int:
    """Generate embeddings for filing_facts not yet embedded.

    Fetches filing_facts joined with filings and companies.
    Creates description using template:
    "{company_name} ({ticker}) reported {metric_name} of {value} {unit}
    for the period ending {period_end} in their {form_type} filing."

    Returns:
        Number of embeddings generated.
    """
    logger.info("Embedding filing facts...")

    db.initialize()
    embedder = Embedder()
    total_embedded = 0
    errors = []

    async with db.async_session() as session:
        # Fetch filing_facts not yet embedded, joined with filings and companies
        subquery = select(Embedding.source_id).where(
            Embedding.source_type == "filing_fact"
        )

        stmt = (
            select(
                FilingFact.id,
                FilingFact.metric_name,
                FilingFact.value,
                FilingFact.unit,
                FilingFact.period_end,
                Filing.form_type,
                Company.ticker,
                Company.name.label("company_name"),
            )
            .join(Filing, FilingFact.filing_id == Filing.id)
            .join(Company, Filing.company_id == Company.id)
            .where(FilingFact.id.not_in(subquery))
            .limit(1000)
        )

        result = await session.execute(stmt)
        filing_facts = result.fetchall()

        if not filing_facts:
            logger.info("  No filing facts to embed")
            return 0

        logger.info(f"  Found {len(filing_facts)} filing facts to embed")

        # Process in batches
        for i in range(0, len(filing_facts), BATCH_SIZE):
            batch = filing_facts[i:i + BATCH_SIZE]
            texts = []
            records = []

            for ff in batch:
                # Build description from template
                description = (
                    f"{ff.company_name} ({ff.ticker}) reported {ff.metric_name} "
                    f"of {ff.value} {ff.unit or ''} for the period ending "
                    f"{ff.period_end} in their {ff.form_type} filing."
                )

                # Build metadata
                metadata = {
                    "ticker": ff.ticker,
                    "company_name": ff.company_name,
                    "metric_name": ff.metric_name,
                    "period_end": str(ff.period_end) if ff.period_end else None,
                    "form_type": ff.form_type,
                }

                texts.append(description)
                records.append({
                    "source_type": "filing_fact",
                    "source_id": ff.id,
                    "content_text": description,
                    "metadata": metadata,
                })

            # Generate embeddings
            try:
                embeddings = embedder.embed_batch(texts)

                # Insert embeddings
                for record, embedding in zip(records, embeddings):
                    record["embedding"] = embedding.tolist()

                stmt = insert(Embedding).values(records)
                await session.execute(stmt)
                await session.commit()

                total_embedded += len(records)
                logger.info(f"  Embedded {len(records)} filing facts (batch {i // BATCH_SIZE + 1})")

            except Exception as e:
                error_msg = f"Error embedding batch {i // BATCH_SIZE + 1}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)

    # Log completion
    async with db.async_session() as session:
        await log_ingestion(
            session,
            DataSource.SEC_EDGAR,
            IngestionStatus.COMPLETED if not errors else IngestionStatus.FAILED,
            records_fetched=total_embedded,
            errors=" | ".join(errors) if errors else None,
        )
        await session.commit()

    logger.info(f"Filing facts embedding complete: {total_embedded} embeddings generated")
    return total_embedded


async def embed_research_papers() -> int:
    """Generate embeddings for research_papers where processed=false.

    Description = title + ". " + abstract (truncated to 512 chars)
    metadata = {external_id, title, authors, published_date, paper_type, keywords}

    After embedding, sets processed=true.

    Returns:
        Number of embeddings generated.
    """
    logger.info("Embedding research papers...")

    db.initialize()
    embedder = Embedder()
    total_embedded = 0
    errors = []

    async with db.async_session() as session:
        # Fetch research_papers where processed=false
        stmt = select(ResearchPaper).where(
            ResearchPaper.processed == False
        ).limit(500)

        result = await session.execute(stmt)
        papers = result.scalars().all()

        if not papers:
            logger.info("  No research papers to embed")
            return 0

        logger.info(f"  Found {len(papers)} research papers to embed")

        # Process in batches
        papers_to_update = []

        for i in range(0, len(papers), BATCH_SIZE):
            batch = papers[i:i + BATCH_SIZE]
            texts = []
            records = []

            for paper in batch:
                # Build description: title + abstract (truncated)
                abstract = paper.abstract or ""
                description = f"{paper.title}. {abstract[:512]}"

                # Build metadata
                metadata = {
                    "external_id": paper.external_id,
                    "title": paper.title,
                    "authors": paper.authors if paper.authors else [],
                    "published_date": str(paper.published_date) if paper.published_date else None,
                    "paper_type": paper.paper_type,
                    "keywords": paper.keywords if paper.keywords else [],
                }

                texts.append(description)
                records.append({
                    "source_type": "research_paper",
                    "source_id": paper.id,
                    "content_text": description,
                    "metadata": metadata,
                })
                papers_to_update.append(paper.id)

            # Generate embeddings
            try:
                embeddings = embedder.embed_batch(texts)

                # Insert embeddings
                for record, embedding in zip(records, embeddings):
                    record["embedding"] = embedding.tolist()

                stmt = insert(Embedding).values(records)
                await session.execute(stmt)

                # Mark papers as processed
                for paper_id in papers_to_update[-len(batch):]:
                    paper = await session.get(ResearchPaper, paper_id)
                    if paper:
                        paper.processed = True

                await session.commit()

                total_embedded += len(records)
                logger.info(f"  Embedded {len(records)} research papers (batch {i // BATCH_SIZE + 1})")

            except Exception as e:
                error_msg = f"Error embedding batch {i // BATCH_SIZE + 1}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)

    # Log completion
    async with db.async_session() as session:
        await log_ingestion(
            session,
            DataSource.NBER,
            IngestionStatus.COMPLETED if not errors else IngestionStatus.FAILED,
            records_fetched=total_embedded,
            errors=" | ".join(errors) if errors else None,
        )
        await session.commit()

    logger.info(f"Research papers embedding complete: {total_embedded} embeddings generated")
    return total_embedded


async def embed_transcripts() -> int:
    """Generate embeddings for earnings_transcripts where processed=false.

    Chunks transcript_text into 1000-char overlapping chunks (200 char overlap).
    Embeds each chunk separately — one row in embeddings per chunk.
    metadata = {ticker, fiscal_quarter, fiscal_year, chunk_index}

    After all chunks embedded, sets processed=true.

    Returns:
        Number of embeddings generated (may be more than transcripts due to chunking).
    """
    logger.info("Embedding earnings transcripts...")

    db.initialize()
    embedder = Embedder()
    total_embedded = 0
    total_chunks = 0
    errors = []

    async with db.async_session() as session:
        # Fetch earnings_transcripts where processed=false
        stmt = select(EarningsTranscript).where(
            EarningsTranscript.processed == False
        ).limit(100)

        result = await session.execute(stmt)
        transcripts = result.scalars().all()

        if not transcripts:
            logger.info("  No earnings transcripts to embed")
            return 0

        logger.info(f"  Found {len(transcripts)} transcripts to embed")

        # Process transcripts one at a time (each may produce multiple chunks)
        for idx, transcript in enumerate(transcripts):
            try:
                # Chunk the transcript text
                chunks = chunk_text(
                    transcript.transcript_text,
                    chunk_size=CHUNK_SIZE,
                    overlap=CHUNK_OVERLAP,
                )

                if not chunks:
                    logger.warning(f"  No chunks generated for transcript {transcript.id}")
                    continue

                total_chunks += len(chunks)

                # Build records for all chunks
                records = []
                for chunk_idx, chunk in enumerate(chunks):
                    metadata = {
                        "ticker": transcript.ticker,
                        "fiscal_quarter": transcript.fiscal_quarter,
                        "fiscal_year": transcript.fiscal_year,
                        "chunk_index": chunk_idx,
                        "total_chunks": len(chunks),
                    }

                    records.append({
                        "source_type": "earnings_transcript",
                        "source_id": transcript.id,
                        "content_text": chunk,
                        "metadata": metadata,
                    })

                # Generate embeddings for all chunks
                texts = [r["content_text"] for r in records]
                embeddings = embedder.embed_batch(texts)

                # Add embeddings to records
                for record, embedding in zip(records, embeddings):
                    record["embedding"] = embedding.tolist()

                # Insert all chunk embeddings
                stmt = insert(Embedding).values(records)
                await session.execute(stmt)

                # Mark transcript as processed
                transcript.processed = True
                await session.commit()

                total_embedded += len(records)
                logger.info(
                    f"  Embedded transcript {transcript.id} "
                    f"({transcript.ticker} Q{transcript.fiscal_quarter} {transcript.fiscal_year}): "
                    f"{len(chunks)} chunks"
                )

            except Exception as e:
                error_msg = f"Error embedding transcript {transcript.id}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)

    # Log completion
    async with db.async_session() as session:
        await log_ingestion(
            session,
            DataSource.EARNINGS_TRANSCRIPTS,
            IngestionStatus.COMPLETED if not errors else IngestionStatus.FAILED,
            records_fetched=total_embedded,
            errors=" | ".join(errors) if errors else None,
        )
        await session.commit()

    logger.info(
        f"Earnings transcripts embedding complete: "
        f"{total_embedded} embeddings from {total_chunks} chunks"
    )
    return total_embedded


async def run_all() -> dict[str, int]:
    """Run all embedding functions sequentially.

    Calls:
    1. embed_data_points()
    2. embed_filing_facts()
    3. embed_research_papers()
    4. embed_transcripts()

    Returns:
        Dict with counts for each embedding type.
    """
    logger.info("=== Starting full embedding generation ===")

    results = {}
    errors = []

    try:
        # 1. Embed data points
        results["data_points"] = await embed_data_points()
    except Exception as e:
        logger.error(f"Error embedding data points: {e}")
        errors.append(f"data_points: {e}")
        results["data_points"] = 0

    try:
        # 2. Embed filing facts
        results["filing_facts"] = await embed_filing_facts()
    except Exception as e:
        logger.error(f"Error embedding filing facts: {e}")
        errors.append(f"filing_facts: {e}")
        results["filing_facts"] = 0

    try:
        # 3. Embed research papers
        results["research_papers"] = await embed_research_papers()
    except Exception as e:
        logger.error(f"Error embedding research papers: {e}")
        errors.append(f"research_papers: {e}")
        results["research_papers"] = 0

    try:
        # 4. Embed transcripts
        results["transcripts"] = await embed_transcripts()
    except Exception as e:
        logger.error(f"Error embedding transcripts: {e}")
        errors.append(f"transcripts: {e}")
        results["transcripts"] = 0

    total = sum(results.values())

    logger.info(
        f"=== Embedding generation complete ===\n"
        f"  Data points: {results['data_points']}\n"
        f"  Filing facts: {results['filing_facts']}\n"
        f"  Research papers: {results['research_papers']}\n"
        f"  Transcripts: {results['transcripts']}\n"
        f"  Total: {total} embeddings"
    )

    if errors:
        logger.warning(f"Errors: {errors}")

    return {
        "total": total,
        "by_type": results,
        "errors": errors,
    }


if __name__ == "__main__":
    asyncio.run(run_all())
