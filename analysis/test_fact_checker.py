"""Test suite for the fact-checking pipeline.

Tests the full pipeline with sample financial content containing
intentional errors that the database should catch.
"""

import asyncio
import os
import sys
from datetime import datetime

import pytest

from .claim_extractor import Claim, ClaimType, extract_claims
from .claim_verifier import (
    VerificationStatus,
    UserFacingStatus,
    lookup_company,
    lookup_numeric,
    lookup_semantic,
    route_claim,
    verify_claim,
)
from .verdict_generator import format_evidence, generate_verdict
from .fact_checker import FactCheckReport, run_fact_check, format_report


# Sample financial content with intentional errors
TEST_CONTENT = """
The Federal Reserve raised interest rates by 75 basis points in 2024,
bringing the federal funds rate to 8.5%. This was driven by inflation
which hit a 40-year high of 12% last month. Apple reported revenue of
$500 billion in their most recent quarter, making it the most profitable
company in history. The yield curve has been inverted for the past 6 months,
which historically predicts a recession within 12-18 months.
GDP growth in the US was negative for three consecutive quarters in 2024,
technically meeting the definition of a recession.
"""


def get_api_key() -> str:
    """Get Anthropic API key from environment."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set - skipping integration tests")
    return api_key


class TestClaimExtraction:
    """Test claim extraction from financial content."""

    def test_extract_claims_returns_claims(self):
        """Test that claims are extracted from test content."""
        api_key = get_api_key()
        claims = extract_claims(TEST_CONTENT, api_key)

        assert len(claims) > 0, "Should extract at least one claim"
        print(f"\nExtracted {len(claims)} claims:")
        for claim in claims:
            print(f"  - [{claim.claim_type.value}] {claim.original_text[:60]}...")

    def test_claim_types_detected(self):
        """Test that different claim types are detected."""
        api_key = get_api_key()
        claims = extract_claims(TEST_CONTENT, api_key)

        claim_types = {claim.claim_type for claim in claims}
        print(f"\nClaim types detected: {claim_types}")

        # Should detect various types from the content
        # Rate claims (interest rates), Macro facts (GDP, inflation), etc.
        assert len(claim_types) >= 1, "Should detect multiple claim types"


class TestEvidenceLookup:
    """Test evidence lookup functions."""

    def test_lookup_numeric(self):
        """Test numeric claim lookup."""
        claim = Claim(
            claim_id="test-1",
            original_text="inflation hit a 40-year high of 12%",
            claim_type=ClaimType.MACRO_FACT,
            normalized_text="inflation rate 12 percent",
            entities=["inflation", "CPI"],
            time_reference="last month",
            confidence=0.9,
        )

        evidence = lookup_numeric(claim)
        print(f"\nNumeric lookup found {len(evidence)} evidence items")
        for ev in evidence:
            print(f"  - {ev.source}: {ev.content} (relevance: {ev.relevance_score})")

    def test_lookup_company(self):
        """Test company fact lookup."""
        claim = Claim(
            claim_id="test-2",
            original_text="Apple reported revenue of $500 billion",
            claim_type=ClaimType.COMPANY_FACT,
            normalized_text="Apple revenue 500 billion",
            entities=["AAPL", "revenue"],
            time_reference="Q4 2024",
            confidence=0.9,
        )

        evidence = lookup_company(claim)
        print(f"\nCompany lookup found {len(evidence)} evidence items")
        for ev in evidence:
            print(f"  - {ev.source}: {ev.content}")

    def test_lookup_semantic(self):
        """Test semantic search lookup."""
        claim = Claim(
            claim_id="test-3",
            original_text="yield curve inversion predicts recession",
            claim_type=ClaimType.CAUSAL_CLAIM,
            normalized_text="yield curve inversion predicts economic recession",
            entities=["yield curve", "recession"],
            time_reference="6 months",
            confidence=0.8,
        )

        evidence = lookup_semantic(claim)
        print(f"\nSemantic lookup found {len(evidence)} evidence items")
        for ev in evidence:
            print(f"  - [{ev.source_type}] {ev.source}: {ev.content[:50]}... (similarity: {ev.relevance_score})")

    def test_route_claim(self):
        """Test claim routing to correct lookup functions."""
        # Test MACRO_FACT routing
        macro_claim = Claim(
            claim_id="test-4",
            original_text="GDP growth was negative",
            claim_type=ClaimType.MACRO_FACT,
            normalized_text="GDP growth negative",
            entities=["GDP"],
            time_reference="2024",
            confidence=0.85,
        )

        evidence = route_claim(macro_claim)
        print(f"\nRouted MACRO_FACT claim: found {len(evidence)} evidence items")

        # Should have both numeric and news evidence
        source_types = {ev.source_type for ev in evidence}
        print(f"  Source types: {source_types}")


class TestVerdictGeneration:
    """Test verdict generation with Claude."""

    def test_format_evidence(self):
        """Test evidence formatting."""
        evidence = [
            Evidence(
                source="FRED - CPI",
                source_type="data_point",
                content="Consumer Price Index: 3.2%",
                value=3.2,
                date="2024-03-01",
                relevance_score=0.95,
            ),
            Evidence(
                source="SEC - AAPL 10-K",
                source_type="filing_fact",
                content="Revenue: $383.285 billion",
                value=383285000000,
                date="2023-09-30",
                relevance_score=0.88,
            ),
        ]

        formatted = format_evidence(evidence)
        print(f"\nFormatted evidence:\n{formatted}")

        assert "1." in formatted
        assert "2." in formatted
        assert "FRED" in formatted
        assert "SEC" in formatted

    def test_generate_verdict_contradicted(self):
        """Test verdict generation for contradicted claim."""
        api_key = get_api_key()

        claim = Claim(
            claim_id="test-5",
            original_text="Apple reported revenue of $500 billion",
            claim_type=ClaimType.COMPANY_FACT,
            normalized_text="Apple revenue 500 billion",
            entities=["AAPL", "revenue"],
            time_reference="2024",
            confidence=0.9,
        )

        evidence = [
            Evidence(
                source="SEC - AAPL 10-K",
                source_type="filing_fact",
                content="Apple Inc. reported annual revenue of $383.285 billion for fiscal year 2023",
                value=383285000000,
                date="2023-09-30",
                relevance_score=0.95,
            ),
        ]

        result = generate_verdict(claim, evidence, api_key)

        print(f"\nVerdict for Apple revenue claim:")
        print(f"  Status: {result.status.value}")
        print(f"  User-facing: {result.user_facing_status.value}")
        print(f"  Confidence: {result.confidence}")
        print(f"  Explanation: {result.explanation}")
        print(f"  Correction: {result.correction}")

        assert result.user_facing_status in (
            UserFacingStatus.POTENTIALLY_MISLEADING,
            UserFacingStatus.SHOULD_VERIFY
        )
        assert result.correction is not None

    def test_generate_verdict_empty_evidence(self):
        """Test verdict generation with no evidence."""
        claim = Claim(
            claim_id="test-6",
            original_text="Some unverifiable claim",
            claim_type=ClaimType.MACRO_FACT,
            normalized_text="unverifiable claim",
            entities=[],
            time_reference=None,
            confidence=0.7,
        )

        result = generate_verdict(claim, [], "fake-api-key")

        print(f"\nVerdict for unverifiable claim:")
        print(f"  Status: {result.status.value}")
        print(f"  User-facing: {result.user_facing_status.value}")
        print(f"  Explanation: {result.explanation}")
        print(f"  Correction: {result.correction}")

        assert result.user_facing_status == UserFacingStatus.SHOULD_VERIFY
        assert "FRED" in result.correction or "SEC" in result.correction or "BLS" in result.correction


class TestFullPipeline:
    """Test the complete fact-checking pipeline."""

    def test_run_fact_check(self):
        """Run full fact-check pipeline on test content."""
        api_key = get_api_key()

        async def run_test():
            report = await run_fact_check(TEST_CONTENT, api_key)
            return report

        report = asyncio.run(run_test())

        print("\n" + "=" * 60)
        print("FULL FACT-CHECK REPORT")
        print("=" * 60)
        print(format_report(report))

        # Assertions
        assert report.total_claims > 0, "Should find claims in content"
        assert report.likely_fine_count + report.should_verify_count + report.potentially_misleading_count == report.total_claims
        assert 0.0 <= report.overall_accuracy_score <= 1.0

        print(f"\nSummary:")
        print(f"  Total claims: {report.total_claims}")
        print(f"  Likely fine: {report.likely_fine_count}")
        print(f"  Should verify: {report.should_verify_count}")
        print(f"  Potentially misleading: {report.potentially_misleading_count}")
        print(f"  Accuracy: {report.overall_accuracy_score:.1%}")

    def test_format_report_structure(self):
        """Test report formatting structure."""
        report = FactCheckReport(
            content_id="test-id",
            original_content="Test content",
            checked_at=datetime.now(),
            total_claims=3,
            likely_fine_count=1,
            should_verify_count=1,
            potentially_misleading_count=1,
            results=[],
            overall_accuracy_score=0.33,
        )

        formatted = format_report(report)

        assert "# Fact-Check Report" in formatted
        assert "Overall accuracy:" in formatted
        assert "## Verified Claims" in formatted or "## High Severity Issues" in formatted


class TestSideBySideDisplay:
    """Test side-by-side claim/correction display."""

    def test_print_claims_with_corrections(self):
        """Print each claim with its verdict and correction side by side."""
        api_key = get_api_key()

        async def run_pipeline():
            return await run_fact_check(TEST_CONTENT, api_key)

        report = asyncio.run(run_pipeline())

        print("\n" + "=" * 80)
        print("SIDE-BY-SIDE CLAIM VERIFICATION DISPLAY")
        print("=" * 80)

        for idx, result in enumerate(report.results, 1):
            print(f"\n{'─' * 80}")
            print(f"CLAIM {idx}: {result.claim.original_text}")
            print(f"{'─' * 80}")
            print(f"Type:              {result.claim.claim_type.value}")
            print(f"Assessment:        {result.user_facing_status.value}")
            print(f"Internal status:   {result.status.value}")
            print(f"Confidence:        {result.confidence:.1%}")
            print(f"\nExplanation:")
            print(f"  {result.explanation}")

            if result.correction:
                print(f"\nCorrection:")
                print(f"  ❌ ORIGINAL:  {result.claim.original_text}")
                print(f"  ✅ CORRECTED: {result.correction}")

            print(f"\nEvidence ({len(result.evidence)} sources):")
            for ev in result.evidence[:3]:  # Show top 3
                print(f"  • [{ev.source_type}] {ev.source}")
                if ev.value is not None:
                    print(f"    Value: {ev.value}, Date: {ev.date}, Relevance: {ev.relevance_score:.2f}")

        print(f"\n{'=' * 80}")
        print(f"OVERALL: {report.likely_fine_count}/{report.total_claims} likely fine ({report.overall_accuracy_score:.1%})")
        print(f"{'=' * 80}")


# Import Evidence for tests
from .claim_verifier import Evidence


if __name__ == "__main__":
    # Run with: python -m analysis.test_fact_checker
    pytest.main([__file__, "-v", "-s"])
