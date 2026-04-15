"""Claim extraction module for financial fact-checking pipeline.

This module extracts verifiable financial claims from content such as
newsletters, articles, or drafts for subsequent fact-checking against
the database.
"""

import json
import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import httpx


class ClaimType(str, Enum):
    """Types of verifiable financial claims."""

    STATISTIC = "STATISTIC"  # Numeric claims: "inflation is at 3.2%"
    RATE = "RATE"  # Interest rates, yields, growth rates
    COMPANY_FACT = "COMPANY_FACT"  # Earnings, revenue, market cap claims
    MACRO_FACT = "MACRO_FACT"  # GDP, employment, economic conditions
    CONCEPT_DEFINITION = "CONCEPT_DEFINITION"  # Definitions of finance terms
    CAUSAL_CLAIM = "CAUSAL_CLAIM"  # X caused Y, X leads to Y
    COMPARATIVE_CLAIM = "COMPARATIVE_CLAIM"  # X is higher/lower/better than Y
    PREDICTION = "PREDICTION"  # Forward-looking statements with numbers


@dataclass
class Claim:
    """A verifiable financial claim extracted from content."""

    claim_id: str
    original_text: str
    claim_type: ClaimType
    normalized_text: str
    entities: list[str]
    time_reference: Optional[str]
    confidence: float
    content_date: Optional[str] = None  # When the source content was written

    def __init__(
        self,
        claim_id: str,
        original_text: str,
        claim_type: ClaimType,
        normalized_text: str,
        entities: list[str],
        time_reference: Optional[str],
        confidence: float,
        content_date: Optional[str] = None,
    ):
        self.claim_id = claim_id
        self.original_text = original_text
        self.claim_type = claim_type
        self.normalized_text = normalized_text
        self.entities = entities
        self.time_reference = time_reference
        self.confidence = confidence
        self.content_date = content_date


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using simple heuristics."""
    # Split on sentence-ending punctuation followed by space or end
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    # Filter out empty strings and very short fragments
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 20]


def extract_claims(
    content: str,
    api_key: str,
    confidence_threshold: float = 0.6,
    content_date: Optional[str] = None,
) -> list[Claim]:
    """Extract verifiable financial claims from content.

    Args:
        content: The financial text to analyze (newsletter, article, draft, etc.)
        api_key: Anthropic API key for Claude
        confidence_threshold: Minimum confidence score to include a claim (default 0.6)
        content_date: When the content was written/published (e.g., "2024-06-15").
                      Used to resolve relative time references like "last month".

    Returns:
        List of Claim objects with confidence >= threshold
    """
    sentences = split_into_sentences(content)
    claims = []

    # Process sentences in batches of 10
    batch_size = 10
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        batch_claims = _process_sentence_batch(
            batch, api_key, confidence_threshold, content_date
        )
        claims.extend(batch_claims)

    return claims


def _process_sentence_batch(
    sentences: list[str],
    api_key: str,
    confidence_threshold: float,
    content_date: Optional[str] = None,
) -> list[Claim]:
    """Process a batch of sentences through Claude API.

    Args:
        sentences: List of sentences to process
        api_key: Anthropic API key
        confidence_threshold: Minimum confidence to include claim
        content_date: When the source content was written (for resolving relative times)
    """
    system_prompt = (
        "You are a financial fact-checking assistant. Extract verifiable "
        "financial claims from text. Return JSON only."
    )

    # Build user prompt with all sentences in the batch
    user_prompt_parts = []
    for idx, sentence in enumerate(sentences):
        user_prompt_parts.append(
            f"Sentence {idx + 1}: {sentence}\n\n"
            f"Analyze this sentence and extract any verifiable financial claims.\n"
            f"Return a JSON object with:\n"
            f"- has_claim: boolean\n"
            f"- claim_type: one of [STATISTIC, RATE, COMPANY_FACT, MACRO_FACT, "
            f"CONCEPT_DEFINITION, CAUSAL_CLAIM, COMPARATIVE_CLAIM, PREDICTION]\n"
            f"- normalized_text: cleaned claim for database lookup\n"
            f"- entities: list of relevant tickers, metrics, or institutions\n"
            f"- time_reference: time period referenced (e.g., '2024', 'Q3 2024', "
            f"'last month', 'past 6 months'). Be specific - if the sentence says "
            f"'in 2024', use '2024'. If it says 'last month', use 'last month'.\n"
            f"- confidence: 0.0-1.0\n"
        )

    user_prompt = "\n---\n".join(user_prompt_parts)

    # Add content date context if available
    date_context = ""
    if content_date:
        date_context = f"\n\nThe source content was published on: {content_date}\n"
        date_context += "Use this date to interpret relative time references like 'last month' or 'recently'."

    user_prompt += date_context
    user_prompt += (
        "\n\nReturn your response as a JSON array where each element corresponds "
        "to a sentence in order. Example:\n"
        "[\n"
        '  {"has_claim": true, "claim_type": "STATISTIC", "normalized_text": "...", "entities": [...], "time_reference": "Q3 2024", "confidence": 0.9},\n'
        '  {"has_claim": false, "claim_type": null, "normalized_text": null, "entities": [], "time_reference": null, "confidence": 0.0}\n'
        "]"
    )

    try:
        response = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 4096,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
            },
            timeout=60.0,
        )
        response.raise_for_status()
        result = response.json()
        content_text = result["content"][0]["text"]

        # Extract JSON from markdown code blocks if present
        import re
        json_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', content_text, re.DOTALL)
        if json_match:
            content_text = json_match.group(1)
        else:
            # Try to find JSON array directly
            json_match = re.search(r'\[.*\]', content_text, re.DOTALL)
            if json_match:
                content_text = json_match.group(0)

        # Parse JSON response
        batch_results = json.loads(content_text)

        claims = []
        for idx, result in enumerate(batch_results):
            if idx >= len(sentences):
                break

            if result.get("has_claim") and result.get("confidence", 0) >= confidence_threshold:
                claim = Claim(
                    claim_id=str(uuid.uuid4()),
                    original_text=sentences[idx],
                    claim_type=ClaimType(result["claim_type"]),
                    normalized_text=result["normalized_text"],
                    entities=result.get("entities", []),
                    time_reference=result.get("time_reference"),
                    confidence=float(result["confidence"]),
                    content_date=content_date,
                )
                claims.append(claim)

        return claims

    except httpx.HTTPStatusError as e:
        # Log error with response details
        error_detail = f"HTTP {e.response.status_code}"
        if e.response.status_code == 401:
            error_detail = "Invalid API key"
        elif e.response.status_code == 404:
            error_detail = f"Model not found - check model name"
        print(f"Error processing batch: {error_detail}")
        print(f"Response: {e.response.text[:200] if e.response.text else 'No details'}")
        return []
    except (httpx.RequestError, json.JSONDecodeError, KeyError) as e:
        # Log error but continue processing other batches
        print(f"Error processing batch: {e}")
        return []
