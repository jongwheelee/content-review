"""Main fact-checking orchestrator for financial content.

This module ties together claim extraction, verification, and verdict
generation to produce complete fact-check reports.
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from .claim_extractor import Claim, extract_claims
from .claim_verifier import (
    VerificationResult,
    VerificationStatus,
    UserFacingStatus,
    get_user_facing_status,
    route_claim,
)
from .verdict_generator import format_evidence, generate_verdict


@dataclass
class FactCheckReport:
    """Complete fact-check report for analyzed content."""

    content_id: str
    original_content: str
    checked_at: datetime
    total_claims: int
    likely_fine_count: int
    should_verify_count: int
    potentially_misleading_count: int
    results: list[VerificationResult]
    overall_accuracy_score: float

    def __init__(
        self,
        content_id: str,
        original_content: str,
        checked_at: datetime,
        total_claims: int,
        likely_fine_count: int,
        should_verify_count: int,
        potentially_misleading_count: int,
        results: list[VerificationResult],
        overall_accuracy_score: float,
    ):
        self.content_id = content_id
        self.original_content = original_content
        self.checked_at = checked_at
        self.total_claims = total_claims
        self.likely_fine_count = likely_fine_count
        self.should_verify_count = should_verify_count
        self.potentially_misleading_count = potentially_misleading_count
        self.results = results
        self.overall_accuracy_score = overall_accuracy_score


async def _process_claim_with_semaphore(
    claim: Claim,
    api_key: str,
    semaphore: asyncio.Semaphore,
) -> VerificationResult:
    """Process a single claim with semaphore-limited concurrency."""
    async with semaphore:
        # Get evidence for the claim
        evidence = route_claim(claim)
        # Generate verdict using Claude
        return generate_verdict(claim, evidence, api_key)


async def run_fact_check(
    content: str,
    api_key: str,
    confidence_threshold: float = 0.6,
    max_concurrent_verdicts: int = 5,
    content_date: Optional[str] = None,
) -> FactCheckReport:
    """Run complete fact-check pipeline on content.

    Args:
        content: The financial text to fact-check
        api_key: Anthropic API key
        confidence_threshold: Minimum confidence for claim extraction
        max_concurrent_verdicts: Max simultaneous Claude API calls (default 5)
        content_date: When the content was written/published (ISO format: "YYYY-MM-DD").
                      Used to resolve relative time references like "last month".
                      If not provided, relative references use current date.

    Returns:
        FactCheckReport with all results
    """
    # Step 1: Extract claims
    print(f"Step 1: Extracting claims from content...")
    claims = extract_claims(content, api_key, confidence_threshold, content_date)
    print(f"  Found {len(claims)} verifiable claims")

    # Step 2 & 3: Get evidence and generate verdicts concurrently
    print(f"Step 2-3: Verifying claims and generating verdicts...")

    # Create semaphore to limit concurrent API calls
    semaphore = asyncio.Semaphore(max_concurrent_verdicts)

    # Process all claims concurrently (with rate limiting)
    tasks = [
        _process_claim_with_semaphore(claim, api_key, semaphore)
        for claim in claims
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle any exceptions from gather
    processed_results = []
    for result in results:
        if isinstance(result, Exception):
            print(f"  Warning: Error processing claim: {result}")
        else:
            processed_results.append(result)

    # Step 4: Assemble report
    print(f"Step 4: Assembling report...")

    # Count by user-facing status
    likely_fine_count = sum(
        1 for r in processed_results
        if r.user_facing_status == UserFacingStatus.LIKELY_FINE
    )
    should_verify_count = sum(
        1 for r in processed_results
        if r.user_facing_status == UserFacingStatus.SHOULD_VERIFY
    )
    potentially_misleading_count = sum(
        1 for r in processed_results
        if r.user_facing_status == UserFacingStatus.POTENTIALLY_MISLEADING
    )

    # Sort by user-facing status (POTENTIALLY_MISLEADING first, then SHOULD_VERIFY, then LIKELY_FINE)
    status_order = {
        UserFacingStatus.POTENTIALLY_MISLEADING: 0,
        UserFacingStatus.SHOULD_VERIFY: 1,
        UserFacingStatus.LIKELY_FINE: 2,
    }
    processed_results.sort(
        key=lambda r: status_order.get(r.user_facing_status, 3)
    )

    # Calculate overall accuracy (proportion of claims that are likely fine)
    total_claims = len(processed_results)
    overall_accuracy_score = (
        likely_fine_count / total_claims if total_claims > 0 else 0.0
    )

    report = FactCheckReport(
        content_id=str(uuid.uuid4()),
        original_content=content,
        checked_at=datetime.now(),
        total_claims=total_claims,
        likely_fine_count=likely_fine_count,
        should_verify_count=should_verify_count,
        potentially_misleading_count=potentially_misleading_count,
        results=processed_results,
        overall_accuracy_score=overall_accuracy_score,
    )

    print(f"  Report complete: {likely_fine_count}/{total_claims} likely fine")
    return report


def format_report(report: FactCheckReport) -> str:
    """Format fact-check report as markdown.

    Uses user-facing status categories for section headers while
    showing detailed reasoning in explanations.

    Args:
        report: FactCheckReport to format

    Returns:
        Formatted markdown string
    """
    lines = [
        "# Fact-Check Report",
        f"Checked at: {report.checked_at.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Overall accuracy: {report.overall_accuracy_score:.1%} "
        f"({report.likely_fine_count}/{report.total_claims} claims likely fine)",
        "",
    ]

    # Potentially Misleading or Inaccurate (CONTRADICTED, OUTDATED)
    misleading = [
        r for r in report.results
        if r.user_facing_status == UserFacingStatus.POTENTIALLY_MISLEADING
    ]
    if misleading:
        lines.append("## ⚠️ Potentially Misleading or Inaccurate")
        lines.append("")
        for result in misleading:
            # Show detailed internal status in explanation
            internal_status = result.status.value
            lines.append(f"### Claim: {result.claim.original_text}")
            lines.append(f"**Assessment**: Potentially misleading ({internal_status})")
            lines.append(f"**Confidence**: {result.confidence:.1%}")
            lines.append(f"**Explanation**: {result.explanation}")
            if result.correction:
                lines.append(f"**Correction**: {result.correction}")
            lines.append(f"**Evidence**: {len(result.evidence)} sources found")
            lines.append("")

    # Should Verify (UNVERIFIABLE, PARTIALLY_CORRECT)
    should_verify = [
        r for r in report.results
        if r.user_facing_status == UserFacingStatus.SHOULD_VERIFY
    ]
    if should_verify:
        lines.append("## ❓ Should Verify")
        lines.append("")
        for result in should_verify:
            internal_status = result.status.value
            lines.append(f"### Claim: {result.claim.original_text}")
            lines.append(f"**Assessment**: Should verify ({internal_status})")
            lines.append(f"**Confidence**: {result.confidence:.1%}")
            lines.append(f"**Explanation**: {result.explanation}")
            if result.correction:
                lines.append(f"**Suggested Action**: {result.correction}")
            lines.append("")

    # Likely Fine (VERIFIED)
    likely_fine = [
        r for r in report.results
        if r.user_facing_status == UserFacingStatus.LIKELY_FINE
    ]
    if likely_fine:
        lines.append("## ✓ Likely Fine")
        lines.append("")
        for result in likely_fine:
            lines.append(f"### Claim: {result.claim.original_text}")
            lines.append(f"**Assessment**: Likely fine")
            lines.append(f"**Confidence**: {result.confidence:.1%}")
            lines.append(f"**Explanation**: {result.explanation}")
            lines.append(f"**Evidence**: {len(result.evidence)} sources found")
            lines.append("")

    return "\n".join(lines)


async def main():
    """CLI entry point for fact-checking."""
    import sys
    import os

    # Get content from stdin or command line argument
    if len(sys.argv) > 1:
        content = sys.argv[1]
    else:
        content = sys.stdin.read()

    if not content.strip():
        print("Error: No content provided. Pass text as argument or via stdin.")
        sys.exit(1)

    # Get API key from environment
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        sys.exit(1)

    # Run fact check
    report = await run_fact_check(content, api_key)

    # Print formatted report
    print("\n" + "=" * 60 + "\n")
    print(format_report(report))


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
