"""ELI5 summarizer with fact-checking and content quality feedback.

This module takes original content and a fact-check report, then produces:
1. A corrected summary with explicit [original → corrected] notation
2. Content quality feedback (length, tone, ambiguity, structure)
3. An overall quality score (weighted average)
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from .claim_extractor import Claim, ClaimType
from .claim_verifier import VerificationResult, VerificationStatus, UserFacingStatus
from .fact_checker import FactCheckReport


# ============================================================================
# TOPIC GROUPING
# ============================================================================

TOPIC_KEYWORDS = {
    "Monetary Policy & Inflation": [
        "fed", "federal reserve", "interest rate", "rates", "inflation",
        "cpi", "basis points", "bps", "federal funds", "monetary policy",
        "jerome powell", "central bank", "tightening", "easing"
    ],
    "Corporate Performance": [
        "revenue", "earnings", "profit", "margin", "company", "corp",
        "apple", "microsoft", "google", "amazon", "tesla", "meta",
        "ticker", "stock", "market cap", "valuation", "10-k", "10-q",
        "sec filing", "quarterly", "fiscal"
    ],
    "Economic Indicators": [
        "gdp", "recession", "yield curve", "unemployment", "jobs",
        "employment", "economic", "economy", "growth", "contraction",
        "expansion", "leading indicator", "lagging indicator"
    ],
}


def determine_topic(claim: Claim) -> str:
    """Determine which topic group a claim belongs to."""
    text_lower = (claim.original_text + " " + claim.normalized_text).lower()

    for topic, keywords in TOPIC_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                return topic

    return "Other Claims"


def group_claims_by_topic(results: list[VerificationResult]) -> dict[str, list[VerificationResult]]:
    """Group verification results by topic."""
    grouped = {}

    for result in results:
        topic = determine_topic(result.claim)
        if topic not in grouped:
            grouped[topic] = []
        grouped[topic].append(result)

    return grouped


# ============================================================================
# NARRATIVE BUILDING
# ============================================================================

@dataclass
class SummarySection:
    """A section of the fact-checked summary."""
    topic: str
    original_narrative: str
    corrected_narrative: str
    claims: list[VerificationResult]
    has_corrections: bool


def build_section_narratives(claims: list[VerificationResult]) -> tuple[str, str]:
    """Build original and corrected narratives for a section.

    Returns:
        Tuple of (original_narrative, corrected_narrative)
    """
    original_parts = []
    corrected_parts = []

    for result in claims:
        original = result.claim.original_text

        if result.user_facing_status == UserFacingStatus.LIKELY_FINE:
            # Claim is verified - no correction needed
            original_parts.append(original)
            corrected_parts.append(original)
        elif result.user_facing_status == UserFacingStatus.POTENTIALLY_MISLEADING:
            # Claim is contradicted or outdated - show correction
            if result.correction:
                original_parts.append(original)
                corrected_parts.append(f"{original} → **{result.correction}**")
            else:
                original_parts.append(original)
                corrected_parts.append(f"{original} → **{result.explanation}**")
        else:  # SHOULD_VERIFY
            # Claim is unverifiable or partially correct - flag it
            original_parts.append(original)
            if result.correction:
                corrected_parts.append(f"{original} → ? {result.correction}")
            else:
                corrected_parts.append(f"{original} → ? {result.explanation}")

    original_narrative = " ".join(original_parts)
    corrected_narrative = " ".join(corrected_parts)

    return original_narrative, corrected_narrative


# ============================================================================
# CONTENT QUALITY ANALYSIS
# ============================================================================

@dataclass
class ContentFeedback:
    """Feedback on non-factual content quality."""
    category: str  # "length", "tone", "ambiguity", "structure"
    issue: str
    location: str
    suggestion: str


@dataclass
class QualityScore:
    """Overall content quality score."""
    overall: int  # 1-10
    factual_accuracy: int  # 1-10
    clarity: int  # 1-10
    completeness: int  # 1-10
    issue_counts: dict[str, int] = field(default_factory=dict)


# Tone patterns to flag (extreme cases only)
SENSATIONALIST_PATTERNS = [
    r"\b(disaster|catastrophe|crash|collapse|implosion|meltdown)\b",
    r"\b(miracle|miraculous|unprecedented|shocking|stunning)\b",
    r"\b(will definitely|will certainly|guaranteed|inevitable)\b",
    r"(?i)\bcrash imminent\b",
    r"(?i)\bdisaster looms\b",
    r"(?i)\bperfect storm\b",
    r"(?i)\bbloodbath\b",
    r"(?i)\barmageddon\b",
]

# Ambiguity patterns
VAGUE_TIME_PATTERNS = [
    r"\b(recently|lately|soon|shortly|in the near future)\b",
    r"\b(last month|last week|last quarter)\b(?!.*\d{4})",
    r"\b(soon|shortly)\b",
]

MISSING_BASELINE_PATTERNS = [
    r"\b(increased|decreased|rose|fell|grew|dropped)\b\s+(\d+)%",
    r"\b(higher|lower|better|worse)\b\s+than\s+(?!specific)",
]

# Structural issues
MISSING_CONTEXT_PATTERNS = [
    r"\b(therefore|thus|hence|consequently)\b.*?(caused|led to|resulted in)",
]


def analyze_content_quality(content: str, report: FactCheckReport) -> list[ContentFeedback]:
    """Analyze content for non-factual quality issues.

    Checks:
    - Length (verbose sections)
    - Tone (extreme sensationalism/overconfidence)
    - Ambiguity (vague time, missing baselines)
    - Structure (missing context, logical gaps)
    """
    feedback = []

    # Split into sentences for location tracking
    sentences = re.split(r"(?<=[.!?])\s+", content)

    # === TONE ANALYSIS ===
    for pattern in SENSATIONALIST_PATTERNS:
        matches = list(re.finditer(pattern, content, re.IGNORECASE))
        if matches:
            # Find which sentence contains the match
            match = matches[0]
            for idx, sentence in enumerate(sentences):
                if match.group(0) in sentence:
                    feedback.append(ContentFeedback(
                        category="tone",
                        issue=f"Sensationalist/overconfident language: '{match.group(0)}'",
                        location=f"Sentence {idx + 1}",
                        suggestion="Use more measured language (e.g., 'may cause' instead of 'will cause')",
                    ))
                    break

    # === AMBIGUITY ANALYSIS ===
    # Vague time references
    for pattern in VAGUE_TIME_PATTERNS:
        matches = list(re.finditer(pattern, content, re.IGNORECASE))
        for match in matches:
            for idx, sentence in enumerate(sentences):
                if match.group(0) in sentence:
                    feedback.append(ContentFeedback(
                        category="ambiguity",
                        issue=f"Vague time reference: '{match.group(0)}'",
                        location=f"Sentence {idx + 1}",
                        suggestion="Specify exact date or period (e.g., 'March 2024' instead of 'last month')",
                    ))
                    break

    # Missing baselines for percentages
    for pattern in MISSING_BASELINE_PATTERNS:
        matches = list(re.finditer(pattern, content, re.IGNORECASE))
        for match in matches:
            # Check if baseline is mentioned nearby
            context = content[max(0, match.start() - 50):match.end() + 50]
            if not any(term in context.lower() for term in ["vs", "versus", "compared to", "from", "prior", "previous"]):
                for idx, sentence in enumerate(sentences):
                    if match.group(0) in sentence:
                        feedback.append(ContentFeedback(
                            category="ambiguity",
                            issue=f"Percentage change without baseline: '{match.group(0)}'",
                            location=f"Sentence {idx + 1}",
                            suggestion="Add comparison baseline (e.g., 'up 15% vs. Q4 2023')",
                        ))
                        break

    # === STRUCTURE ANALYSIS ===
    # Check for causal claims without supporting evidence
    for pattern in MISSING_CONTEXT_PATTERNS:
        matches = list(re.finditer(pattern, content, re.IGNORECASE))
        for match in matches:
            for idx, sentence in enumerate(sentences):
                if match.group(0) in sentence:
                    feedback.append(ContentFeedback(
                        category="structure",
                        issue=f"Causal claim may lack supporting context: '{match.group(0)[:50]}...'",
                        location=f"Sentence {idx + 1}",
                        suggestion="Add supporting data or acknowledge other potential factors",
                    ))
                    break

    # === LENGTH ANALYSIS ===
    # Check for overly long sentences
    for idx, sentence in enumerate(sentences):
        word_count = len(sentence.split())
        if word_count > 40:
            feedback.append(ContentFeedback(
                category="length",
                issue=f"Sentence is verbose ({word_count} words)",
                location=f"Sentence {idx + 1}",
                suggestion="Consider splitting into 2-3 shorter sentences",
            ))

    # Check for imbalanced content (too much focus on one topic)
    topic_counts = {}
    for sentence in sentences:
        for topic, keywords in TOPIC_KEYWORDS.items():
            if any(kw in sentence.lower() for kw in keywords):
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
                break

    if topic_counts:
        max_topic = max(topic_counts, key=topic_counts.get)
        max_count = topic_counts[max_topic]
        total = sum(topic_counts.values())
        if max_count > total * 0.7:  # More than 70% focus on one topic
            feedback.append(ContentFeedback(
                category="length",
                issue=f"Content heavily focused on {max_topic} ({max_count}/{total} sentences)",
                location="Overall structure",
                suggestion="Consider balancing coverage across topics",
            ))

    return feedback


# ============================================================================
# QUALITY SCORE CALCULATION
# ============================================================================

def calculate_quality_score(
    report: FactCheckReport,
    feedback: list[ContentFeedback]
) -> QualityScore:
    """Calculate overall content quality score.

    Weights:
    - Factual accuracy: 50% (most important for fact-checking)
    - Clarity: 25% (tone + ambiguity issues)
    - Completeness: 25% (structure + length issues)
    """
    # Factual accuracy score (1-10)
    if report.total_claims == 0:
        factual_score = 10  # No claims to verify = no factual errors
    else:
        factual_ratio = report.likely_fine_count / report.total_claims
        factual_score = max(1, round(factual_ratio * 10))

    # Count issues by category
    issue_counts = {"tone": 0, "ambiguity": 0, "structure": 0, "length": 0}
    for fb in feedback:
        if fb.category in issue_counts:
            issue_counts[fb.category] += 1

    # Clarity score (1-10) - based on tone + ambiguity
    clarity_issues = issue_counts["tone"] + issue_counts["ambiguity"]
    clarity_score = max(1, 10 - (clarity_issues * 1.5))  # -1.5 per issue

    # Completeness score (1-10) - based on structure + length
    completeness_issues = issue_counts["structure"] + issue_counts["length"]
    completeness_score = max(1, 10 - (completeness_issues * 1.5))

    # Weighted average
    overall = round(
        factual_score * 0.50 +
        clarity_score * 0.25 +
        completeness_score * 0.25
    )

    return QualityScore(
        overall=overall,
        factual_accuracy=factual_score,
        clarity=clarity_score,
        completeness=completeness_score,
        issue_counts=issue_counts,
    )


# ============================================================================
# SUMMARY GENERATION
# ============================================================================

@dataclass
class FactCheckedSummary:
    """Complete fact-checked summary with feedback."""
    sections: list[SummarySection]
    unverifiable_claims: list[VerificationResult]
    feedback: list[ContentFeedback]
    quality_score: QualityScore


def generate_summary(
    content: str,
    report: FactCheckReport
) -> FactCheckedSummary:
    """Generate a fact-checked summary with content quality feedback.

    Args:
        content: Original financial content
        report: Fact-check report from run_fact_check()

    Returns:
        FactCheckedSummary with corrected narratives and feedback
    """
    # Group claims by topic
    grouped = group_claims_by_topic(report.results)

    # Build summary sections
    sections = []
    for topic, claims in grouped.items():
        original, corrected = build_section_narratives(claims)
        has_corrections = any(
            r.user_facing_status != UserFacingStatus.LIKELY_FINE
            for r in claims
        )
        sections.append(SummarySection(
            topic=topic,
            original_narrative=original,
            corrected_narrative=corrected,
            claims=claims,
            has_corrections=has_corrections,
        ))

    # Separate unverifiable claims
    unverifiable = [
        r for r in report.results
        if r.user_facing_status == UserFacingStatus.SHOULD_VERIFY
    ]

    # Analyze content quality
    feedback = analyze_content_quality(content, report)

    # Calculate quality score
    quality_score = calculate_quality_score(report, feedback)

    return FactCheckedSummary(
        sections=sections,
        unverifiable_claims=unverifiable,
        feedback=feedback,
        quality_score=quality_score,
    )


# ============================================================================
# FORMATTING
# ============================================================================

def format_summary(summary: FactCheckedSummary) -> str:
    """Format the fact-checked summary as markdown.

    Output structure:
    1. Summary with corrections
    2. Unverifiable claims
    3. Content quality feedback
    4. Overall quality score
    """
    lines = []

    # === FACT-CHECKED SUMMARY ===
    lines.append("# Fact-Checked Summary")
    lines.append("")

    for section in summary.sections:
        lines.append(f"## {section.topic}")
        lines.append("")

        if section.has_corrections:
            lines.append("**Corrections:**")
            lines.append("")

        # Show each claim with its status
        for result in section.claims:
            status_icon = {
                UserFacingStatus.LIKELY_FINE: "✓",
                UserFacingStatus.SHOULD_VERIFY: "?",
                UserFacingStatus.POTENTIALLY_MISLEADING: "⚠️",
            }.get(result.user_facing_status, "•")

            lines.append(f"- {status_icon} **{result.claim.original_text}**")

            if result.user_facing_status != UserFacingStatus.LIKELY_FINE:
                if result.correction:
                    lines.append(f"  - **Correction**: {result.correction}")
                else:
                    lines.append(f"  - **Note**: {result.explanation}")

            lines.append(f"  - *Evidence*: {len(result.evidence)} sources")
            lines.append("")

        lines.append("")

    # === UNVERIFIABLE CLAIMS ===
    if summary.unverifiable_claims:
        lines.append("## Unverifiable Claims")
        lines.append("")
        lines.append("The following claims could not be verified against our database.")
        lines.append("")

        for result in summary.unverifiable_claims:
            lines.append(f"- **{result.claim.original_text}**")
            lines.append(f"  - {result.explanation}")
            if result.correction:
                lines.append(f"  - *Suggested source*: {result.correction}")
            lines.append("")

    # === CONTENT QUALITY FEEDBACK ===
    if summary.feedback:
        lines.append("---")
        lines.append("")
        lines.append("## Content Quality Feedback")
        lines.append("")

        # Group feedback by category
        feedback_by_category = {}
        for fb in summary.feedback:
            if fb.category not in feedback_by_category:
                feedback_by_category[fb.category] = []
            feedback_by_category[fb.category].append(fb)

        category_titles = {
            "tone": "Tone",
            "ambiguity": "Ambiguity",
            "structure": "Structure",
            "length": "Length",
        }

        for category, items in feedback_by_category.items():
            lines.append(f"### {category_titles.get(category, category)}")
            lines.append("")

            for fb in items:
                lines.append(f"- **{fb.issue}** ({fb.location})")
                lines.append(f"  - *Suggestion*: {fb.suggestion}")
                lines.append("")

    # === QUALITY SCORE ===
    lines.append("---")
    lines.append("")
    lines.append("## Overall Quality Score")
    lines.append("")
    lines.append(f"**{summary.quality_score.overall}/10**")
    lines.append("")
    lines.append("| Dimension | Score |")
    lines.append("|-----------|-------|")
    lines.append(f"| Factual Accuracy | {summary.quality_score.factual_accuracy}/10 |")
    lines.append(f"| Clarity | {summary.quality_score.clarity}/10 |")
    lines.append(f"| Completeness | {summary.quality_score.completeness}/10 |")
    lines.append("")

    # Issue breakdown
    if any(summary.quality_score.issue_counts.values()):
        lines.append("**Issues Found:**")
        lines.append("")
        for category, count in summary.quality_score.issue_counts.items():
            if count > 0:
                lines.append(f"- {category.title()}: {count} issue(s)")
        lines.append("")

    return "\n".join(lines)


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

async def run_summarizer(content: str, api_key: str) -> str:
    """Run the complete fact-checking and summarization pipeline.

    Args:
        content: Financial content to fact-check and summarize
        api_key: Anthropic API key

    Returns:
        Formatted summary as markdown string
    """
    from .fact_checker import run_fact_check

    # Run fact-checking
    report = await run_fact_check(content, api_key)

    # Generate summary
    summary = generate_summary(content, report)

    # Format and return
    return format_summary(summary)


if __name__ == "__main__":
    import asyncio
    import sys
    import os

    # Get content from stdin or command line
    if len(sys.argv) > 1:
        content = " ".join(sys.argv[1:])
    else:
        content = sys.stdin.read()

    if not content.strip():
        print("Error: No content provided.", file=sys.stderr)
        sys.exit(1)

    # Get API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    # Run and print
    result = asyncio.run(run_summarizer(content, api_key))
    print(result)
