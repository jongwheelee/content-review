"""Verdict generation module for financial fact-checking pipeline.

This module uses Claude to produce human-readable verdicts with
explanations and corrections for verified claims.
"""

import json
import time
from typing import Optional

import httpx

from .claim_extractor import Claim
from .claim_verifier import (
    Evidence,
    VerificationResult,
    VerificationStatus,
    UserFacingStatus,
    get_user_facing_status,
)


class VerdictGenerationError(Exception):
    """Error during verdict generation."""

    pass


def format_evidence(evidence: list[Evidence]) -> str:
    """Format evidence into a clean numbered list for the prompt.

    Args:
        evidence: List of Evidence objects

    Returns:
        Formatted string with numbered evidence items
    """
    if not evidence:
        return "No evidence available."

    lines = []
    for idx, ev in enumerate(evidence, 1):
        date_str = f" ({ev.date})" if ev.date else ""
        value_str = f" [Value: {ev.value}]" if ev.value is not None else ""
        lines.append(
            f"{idx}. [{ev.source_type.upper()}] {ev.source}{date_str}: "
            f"{ev.content}{value_str} (relevance: {ev.relevance_score:.2f})"
        )

    return "\n".join(lines)


def generate_verdict(
    claim: Claim,
    evidence: list[Evidence],
    api_key: str,
    max_retries: int = 3,
) -> VerificationResult:
    """Generate a human-readable verdict for a claim using Claude.

    Args:
        claim: The claim to verify
        evidence: List of evidence found for the claim
        api_key: Anthropic API key
        max_retries: Maximum number of retry attempts (default 3)

    Returns:
        VerificationResult with status, confidence, explanation, correction, and severity

    Raises:
        VerdictGenerationError: If all retry attempts fail
    """
    # Handle empty evidence case
    if not evidence:
        return VerificationResult(
            claim=claim,
            status=VerificationStatus.UNVERIFIABLE,
            confidence=0.0,
            evidence=[],
            correction=(
                "Consider citing a primary source such as FRED, SEC EDGAR, or BLS "
                "for this type of claim."
            ),
            explanation="No matching data found in our database for this claim.",
        )

    system_prompt = (
        "You are a rigorous financial fact-checker with access to authoritative "
        "data sources. Your job is to verify financial claims and provide clear, "
        "actionable corrections. Be precise, cite specific numbers, and explain "
        "WHY something is wrong, not just that it is wrong. Never make up data — "
        "only use the evidence provided."
    )

    formatted_evidence = format_evidence(evidence)

    user_prompt = f"""Verify this financial claim using the evidence provided.

CLAIM: {claim.original_text}
CLAIM TYPE: {claim.claim_type.value}

EVIDENCE FROM DATABASE:
{formatted_evidence}

Return a JSON object with:
- status: VERIFIED | CONTRADICTED | PARTIALLY_CORRECT | OUTDATED | UNVERIFIABLE
- confidence: 0.0-1.0
- explanation: detailed explanation of why the claim is correct/incorrect
  (2-3 sentences, cite specific numbers and dates from evidence)
- correction: if status is not VERIFIED, provide the corrected version
  of the claim with accurate numbers. If UNVERIFIABLE, suggest what
  source would be needed to verify it. null if VERIFIED.
- severity: LOW | MEDIUM | HIGH
  (HIGH = factually wrong with specific numbers,
   MEDIUM = outdated or partially correct,
   LOW = minor imprecision or unverifiable)
- sources_used: list of source names from evidence"""

    # Retry logic with exponential backoff
    last_error = None
    for attempt in range(max_retries):
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
                    "max_tokens": 1024,
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
            json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', content_text, re.DOTALL)
            if json_match:
                content_text = json_match.group(1)
            else:
                # Try to find JSON object directly
                json_match = re.search(r'\{.*\}', content_text, re.DOTALL)
                if json_match:
                    content_text = json_match.group(0)

            # Parse JSON response
            verdict_data = json.loads(content_text)

            # Map string status to VerificationStatus enum
            status_str = verdict_data.get("status", "UNVERIFIABLE")
            try:
                status = VerificationStatus(status_str)
            except ValueError:
                status = VerificationStatus.UNVERIFIABLE

            # Store severity from Claude for sorting
            severity = verdict_data.get("severity", "LOW")

            return VerificationResult(
                claim=claim,
                status=status,
                confidence=float(verdict_data.get("confidence", 0.5)),
                evidence=evidence,
                correction=verdict_data.get("correction"),
                explanation=verdict_data.get(
                    "explanation", "Unable to generate explanation."
                ),
                user_facing_status=get_user_facing_status(status),
            )

        except httpx.HTTPStatusError as e:
            last_error = e
            if e.response.status_code == 429:
                # Rate limit - use longer backoff
                backoff = (2 ** attempt) * 2
            else:
                backoff = 2 ** attempt

        except (httpx.RequestError, json.JSONDecodeError, KeyError) as e:
            last_error = e
            backoff = 2 ** attempt

        # Wait before retry (exponential backoff)
        if attempt < max_retries - 1:
            time.sleep(backoff)

    # All retries exhausted
    raise VerdictGenerationError(
        f"Failed to generate verdict after {max_retries} attempts. Last error: {last_error}"
    )
