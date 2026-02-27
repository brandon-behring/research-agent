"""Synthesis Writer node -- produces the final structured research report.

Uses Sonnet (reasoning model) to synthesize all gathered evidence into
a coherent report with citations, confidence assessments, and identified gaps.
This is the most expensive LLM call -- justified because final output quality matters.

Uses ``with_structured_output()`` for guaranteed report structure.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from research_agent.config import AgentConfig
from research_agent.exceptions import SynthesisError
from research_agent.llm import create_llm
from research_agent.state import NodeUpdate, ResearchState

logger = logging.getLogger(__name__)

# Maximum search results included in the synthesis context window.
_MAX_SEARCH_RESULTS_IN_CONTEXT = 15

# Maximum citation entries per source included in the synthesis context.
_MAX_CITATIONS_IN_CONTEXT = 5

SYSTEM_PROMPT = """You are a research synthesis expert. Given the gathered evidence from a
multi-agent research system, produce a structured report.

Your report must include:
1. **Executive Summary** -- 2-3 sentence overview of findings
2. **Key Findings** -- Main discoveries, each with:
   - A confidence level (high/medium/low) reflecting how well-supported the finding is
   - A count of how many distinct sources support it (source_count)
   - Supporting citations in the text
3. **Concept Map** -- How the key concepts relate to each other (prose description)
4. **Concept Map (Mermaid)** -- Mermaid graph syntax showing relationships:
   ```
   graph TD
     A[Concept A] -->|relationship| B[Concept B]
   ```
   Use descriptive edge labels. Include 3-8 key concepts.
5. **Citation Landscape** -- Seminal papers, recent developments, citation chains
6. **Methodological Considerations** -- Assumptions, limitations, verification approaches
7. **Gaps & Limitations** -- What the KB didn't cover, where evidence is thin
8. **Next Research Questions** -- 2-4 follow-up questions suggested by gaps/limitations
9. **Confidence Assessment** -- How reliable are these findings? (high/medium/low with reasoning)

Finding confidence guidelines:
- HIGH: Multiple high-scoring sources (>=0.8) directly support this finding
- MEDIUM: Some sources support it but evidence is indirect or scores are moderate
- LOW: Few sources, low scores, or finding is inferred rather than directly stated

Citation format: [Author (Year)] with source IDs in footnotes.
Be honest about gaps -- if few results were found, say so clearly.
Use the Evidence Quality Metadata section to calibrate your confidence assessment.
If evidence is sparse or scores are low, acknowledge this explicitly.
Do NOT hallucinate papers or results not in the evidence provided."""


class Finding(BaseModel):
    """A single research finding with confidence metadata."""

    text: str = Field(description="The finding text with supporting citations")
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence level for this specific finding"
    )
    source_count: int = Field(
        default=0,
        description="Number of sources supporting this finding",
    )

    def to_markdown(self) -> str:
        """Render finding as markdown with confidence tag."""
        tag = self.confidence.upper()
        sources = f" ({self.source_count} sources)" if self.source_count else ""
        return f"[{tag}]{sources} {self.text}"


class SynthesisReport(BaseModel):
    """Structured synthesis report from the LLM."""

    executive_summary: str = Field(description="2-3 sentence overview of findings")
    key_findings: list[Finding] = Field(
        description="Main discoveries, each with confidence level and source count",
    )
    concept_map: str = Field(description="How key concepts relate to each other")
    concept_map_mermaid: str = Field(
        default="",
        description=(
            "Mermaid graph syntax showing concept relationships. "
            "Use 'graph TD' with labeled edges. Example: "
            "graph TD\\n  A[DML] -->|uses| B[Cross-fitting]"
        ),
    )
    citation_landscape: str = Field(description="Seminal papers, recent developments, chains")
    methodological_considerations: str = Field(
        description="Assumptions, limitations, verification approaches"
    )
    gaps_limitations: str = Field(description="What the KB didn't cover, thin evidence areas")
    next_questions: list[str] = Field(
        default_factory=list,
        description="Follow-up research questions suggested by gaps and limitations",
    )
    confidence_level: Literal["high", "medium", "low"] = Field(
        description="Overall reliability of findings"
    )
    confidence_reasoning: str = Field(description="Why this confidence level was assigned")

    def to_markdown(self) -> str:
        """Render the report as a markdown document."""
        findings = "\n".join(f"- {f.to_markdown()}" for f in self.key_findings)

        parts = [
            f"# Research Report\n\n"
            f"## Executive Summary\n\n{self.executive_summary}\n\n"
            f"## Key Findings\n\n{findings}\n\n"
            f"## Concept Map\n\n{self.concept_map}",
        ]

        if self.concept_map_mermaid:
            parts.append(f"\n\n```mermaid\n{self.concept_map_mermaid}\n```")

        parts.append(
            f"\n\n## Citation Landscape\n\n{self.citation_landscape}\n\n"
            f"## Methodological Considerations\n\n{self.methodological_considerations}\n\n"
            f"## Gaps & Limitations\n\n{self.gaps_limitations}"
        )

        if self.next_questions:
            questions = "\n".join(f"- {q}" for q in self.next_questions)
            parts.append(f"\n\n## Next Research Questions\n\n{questions}")

        parts.append(
            f"\n\n## Confidence Assessment\n\n"
            f"**Level**: {self.confidence_level}\n\n"
            f"{self.confidence_reasoning}\n"
        )

        return "".join(parts)


# Stopwords excluded from term extraction (common English + academic filler).
_STOPWORDS = frozenset(
    {
        "the",
        "this",
        "that",
        "these",
        "those",
        "with",
        "from",
        "into",
        "through",
        "about",
        "which",
        "where",
        "when",
        "while",
        "their",
        "there",
        "also",
        "been",
        "have",
        "more",
        "some",
        "such",
        "than",
        "they",
        "were",
        "what",
        "will",
        "each",
        "other",
        "between",
        "under",
        "using",
        "based",
        "provides",
        "approach",
        "however",
        "both",
        "does",
        "well",
        "show",
        "used",
        "many",
        "most",
        "over",
        "only",
        "very",
        "after",
        "before",
        "should",
        "could",
        "would",
        "being",
        "given",
    }
)

# Minimum term length for meaningful matching.
_MIN_TERM_LENGTH = 5

# Minimum number of terms from a finding that must match a search result.
_MIN_TERM_OVERLAP = 2


def _extract_terms(text: str) -> list[str]:
    """Extract significant terms from text for evidence matching.

    Filters out short words and stopwords to focus on domain-specific content.

    Args:
        text: Source text to extract terms from.

    Returns:
        List of unique lowercase terms with length >= _MIN_TERM_LENGTH,
        excluding stopwords.
    """
    words = re.findall(r"[a-zA-Z]+", text.lower())
    seen: set[str] = set()
    terms: list[str] = []
    for w in words:
        if len(w) >= _MIN_TERM_LENGTH and w not in _STOPWORDS and w not in seen:
            seen.add(w)
            terms.append(w)
    return terms


def _count_matching_sources(
    finding_terms: list[str],
    state: ResearchState,
) -> tuple[int, float]:
    """Count search results that mention enough terms from a finding.

    Args:
        finding_terms: Significant terms extracted from finding text.
        state: ResearchState with search_results.

    Returns:
        Tuple of (matching_count, avg_score_of_matches).
        avg_score is 0.0 if matching_count is 0.
    """
    if not finding_terms or not state.search_results:
        return 0, 0.0

    matching_scores: list[float] = []
    for r in state.search_results:
        haystack = f"{r.title} {r.content}".lower()
        overlap = sum(1 for t in finding_terms if t in haystack)
        if overlap >= min(_MIN_TERM_OVERLAP, len(finding_terms)):
            matching_scores.append(r.score)

    if not matching_scores:
        return 0, 0.0
    return len(matching_scores), sum(matching_scores) / len(matching_scores)


def _validate_findings(
    findings: list[Finding],
    state: ResearchState,
) -> list[Finding]:
    """Re-score findings against actual evidence in ResearchState.

    For each finding:
    1. Count how many search results mention key terms from the finding text
    2. Cap source_count to actual matching sources (LLM can't claim more than exist)
    3. Downgrade confidence if evidence is thin:
       - 0 matching sources → force LOW
       - 1-2 matching sources → cap at MEDIUM
       - 3+ matching sources with avg score >= 0.7 → allow HIGH

    Args:
        findings: LLM-generated findings with self-assessed confidence.
        state: Complete research state with search_results.

    Returns:
        New list of Finding objects with validated source_count and confidence.
    """
    validated: list[Finding] = []
    for finding in findings:
        terms = _extract_terms(finding.text)
        matching_count, avg_score = _count_matching_sources(terms, state)

        # Cap source_count to reality
        validated_source_count = min(finding.source_count, matching_count)

        # Determine maximum allowed confidence
        if matching_count == 0:
            max_confidence: Literal["high", "medium", "low"] = "low"
        elif matching_count <= 2:
            max_confidence = "medium"
        elif avg_score >= 0.7:
            max_confidence = "high"
        else:
            max_confidence = "medium"

        # Apply ceiling: can only downgrade, never upgrade
        confidence_order: dict[str, int] = {"low": 0, "medium": 1, "high": 2}
        if confidence_order.get(finding.confidence, 0) > confidence_order[max_confidence]:
            validated_confidence = max_confidence
        else:
            validated_confidence = finding.confidence

        validated.append(
            Finding(
                text=finding.text,
                confidence=validated_confidence,
                source_count=validated_source_count,
            )
        )

    return validated


# Citation patterns: [Author (Year)] or (Author, Year) or (Author Year)
_CITATION_PATTERNS = [
    re.compile(r"\[([^\]]+?)\s*\((\d{4})\)\]"),  # [Author (2018)]
    re.compile(r"\(([^)]+?),\s*(\d{4})\)"),  # (Author, 2018)
    re.compile(r"\(([^)]+?)\s+(\d{4})\)"),  # (Author 2018)
]


def _check_citation_grounding(
    report: SynthesisReport,
    state: ResearchState,
) -> list[str]:
    """Find citation-like patterns in the report not backed by evidence.

    Scans for patterns like [Author (Year)] or (Author, Year) and checks
    each against known authors/years from search_results and source_details.

    Args:
        report: The synthesis report to check.
        state: ResearchState with search_results and source_details.

    Returns:
        List of warning strings for ungrounded citations (may be empty).
    """
    # Build known (author_fragment, year) pairs from evidence
    known_authors: list[tuple[str, str]] = []
    for r in state.search_results:
        if r.authors and r.year:
            known_authors.append((r.authors.lower(), r.year))
    for s in state.source_details:
        authors = s.get("authors", "")
        year = str(s.get("year", ""))
        if authors and year:
            known_authors.append((authors.lower(), year))

    # Extract all text from report for scanning
    report_text = report.to_markdown()

    # Find all citation-like patterns
    found_citations: list[tuple[str, str]] = []
    for pattern in _CITATION_PATTERNS:
        for match in pattern.finditer(report_text):
            author_part = match.group(1).strip()
            year_part = match.group(2).strip()
            found_citations.append((author_part, year_part))

    # Deduplicate
    seen: set[tuple[str, str]] = set()
    unique_citations: list[tuple[str, str]] = []
    for cite in found_citations:
        key = (cite[0].lower(), cite[1])
        if key not in seen:
            seen.add(key)
            unique_citations.append(cite)

    # Check each citation against known evidence
    warnings: list[str] = []
    for author_part, year in unique_citations:
        author_lower = author_part.lower()
        grounded = False
        for known_author, known_year in known_authors:
            if known_year == year:
                # Check if any fragment of the cited author appears in known authors
                # Split on common separators to handle "Chernozhukov et al."
                fragments = re.split(r"[\s,]+", author_lower)
                significant = [f for f in fragments if len(f) >= 3 and f not in ("et", "al", "al.")]
                if any(frag in known_author for frag in significant):
                    grounded = True
                    break
        if not grounded:
            warnings.append(f"'{author_part} ({year})' not found in evidence")

    return warnings


class EvidenceMetadata(BaseModel):
    """Structured evidence quality summary attached to the report.

    Provides machine-readable quality signals so consumers can assess
    report reliability programmatically, not just via prose.
    """

    total_sources: int = Field(default=0, description="Total search results found")
    high_score_sources: int = Field(default=0, description="Sources with score >= 0.8")
    avg_score: float = Field(default=0.0, description="Mean relevance score across all sources")
    year_range: str = Field(default="", description="e.g., '2018-2023'")
    concepts_explored: int = Field(default=0)
    methods_audited: int = Field(default=0)
    grounding_warnings: list[str] = Field(default_factory=list)
    validation_applied: bool = Field(default=True)


def _build_evidence_metadata(
    state: ResearchState,
    grounding_warnings: list[str],
) -> EvidenceMetadata:
    """Build structured evidence metadata from pipeline state.

    Args:
        state: Complete research state.
        grounding_warnings: Warnings from citation grounding check.

    Returns:
        EvidenceMetadata with computed quality signals.
    """
    total = len(state.search_results)
    high = sum(1 for r in state.search_results if r.score >= 0.8)
    avg = sum(r.score for r in state.search_results) / total if total else 0.0

    year_range = ""
    years = [int(r.year) for r in state.search_results if r.year.isdigit()]
    if years:
        year_range = f"{min(years)}-{max(years)}" if len(years) > 1 else str(years[0])

    return EvidenceMetadata(
        total_sources=total,
        high_score_sources=high,
        avg_score=round(avg, 3),
        year_range=year_range,
        concepts_explored=len(state.concepts),
        methods_audited=len(state.assumption_audits),
        grounding_warnings=grounding_warnings,
        validation_applied=True,
    )


def _build_evidence_context(state: ResearchState) -> str:
    """Compile all gathered evidence into a single context string.

    Args:
        state: Current state with all agent outputs.

    Returns:
        Formatted evidence string for the synthesis LLM.
    """
    sections: list[str] = []

    # Original query and plan
    sections.append(f"## Research Question\n{state.query}")
    sections.append(f"## Planning Rationale\n{state.planning_rationale}")

    # Node summaries -- high-level signals from earlier analysis
    summaries = []
    if state.search_summary:
        summaries.append(f"- **Literature**: {state.search_summary}")
    if state.concept_map_summary:
        summaries.append(f"- **Concepts**: {state.concept_map_summary}")
    if state.citation_summary:
        summaries.append(f"- **Citations**: {state.citation_summary}")
    if state.assumption_summary:
        summaries.append(f"- **Assumptions**: {state.assumption_summary}")
    if summaries:
        sections.append("## Analysis Overview\n" + "\n".join(summaries))

    # Search results
    if state.search_results:
        search_lines = [f"## Literature Search ({len(state.search_results)} results)"]
        for i, r in enumerate(state.search_results[:_MAX_SEARCH_RESULTS_IN_CONTEXT], 1):
            search_lines.append(
                f"\n### Result {i}: {r.title}\n"
                f"Authors: {r.authors}\nYear: {r.year}\n"
                f"Source ID: {r.source_id}\nScore: {r.score}\n\n{r.content}"
            )
        sections.append("\n".join(search_lines))

    # Concepts
    if state.concepts:
        concept_lines = [f"## Knowledge Graph Concepts ({len(state.concepts)})"]
        for c in state.concepts:
            if c.neighborhood_summary:
                concept_lines.append(f"\n### {c.name}\n{c.neighborhood_summary}")
            elif c.description:
                concept_lines.append(f"\n### {c.name} ({c.concept_type})\n{c.description}")
        sections.append("\n".join(concept_lines))

    # Similar concepts (embedding-based discovery)
    if state.similar_concepts:
        sim_lines = [f"### Embedding-Similar Concepts ({len(state.similar_concepts)})"]
        for sc in state.similar_concepts:
            source = sc.get("source_concept", "?")
            similarity = sc.get("similarity", 0)
            ctype = sc.get("concept_type", "")
            type_str = f" [{ctype}]" if ctype else ""
            sim_lines.append(
                f"- {sc.get('name', '?')} ({similarity:.0%} similar to {source}){type_str}"
            )
        # Append to concepts section if it exists, otherwise standalone
        if state.concepts:
            sections[-1] += "\n\n" + "\n".join(sim_lines)
        else:
            sections.append("\n".join(sim_lines))

    # Citations
    if state.citations:
        cite_lines = [f"## Citation Networks ({len(state.citations)} sources analyzed)"]
        for cit in state.citations:
            cite_lines.append(f"\n### {cit.source_title}")
            if cit.citing:
                cite_lines.append(f"Citing ({len(cit.citing)}):")
                for ref in cit.citing[:_MAX_CITATIONS_IN_CONTEXT]:
                    cite_lines.append(f"  - {ref.get('title', 'Unknown')} ({ref.get('year', '?')})")
            if cit.cited_by:
                cite_lines.append(f"Cited by ({len(cit.cited_by)}):")
                for ref in cit.cited_by[:_MAX_CITATIONS_IN_CONTEXT]:
                    cite_lines.append(f"  - {ref.get('title', 'Unknown')} ({ref.get('year', '?')})")
            if cit.similar_papers:
                cite_lines.append(f"Similar papers ({len(cit.similar_papers)}):")
                for p in cit.similar_papers[:_MAX_CITATIONS_IN_CONTEXT]:
                    cite_lines.append(
                        f"  - {p.get('title', 'Unknown')} ({p.get('coupling_pct', 0):.0f}% overlap)"
                    )
        sections.append("\n".join(cite_lines))

    # Cross-domain bridges
    if state.cross_domain_matches:
        xd_lines = [f"## Cross-Domain Bridges ({len(state.cross_domain_matches)})"]
        for m in state.cross_domain_matches:
            src = m.get("source", "?")
            tgt = m.get("target", "?")
            src_dom = m.get("source_domain", "?")
            tgt_dom = m.get("target_domain", "?")
            link = m.get("link_type", "?")
            xd_lines.append(f"- {src} ({src_dom}) \u2194 {tgt} ({tgt_dom}) [{link}]")
        sections.append("\n".join(xd_lines))

    # Source details (enriched metadata from get_source)
    if state.source_details:
        detail_lines = [f"## Source Details ({len(state.source_details)} enriched)"]
        for s in state.source_details:
            title = s.get("title", "Unknown")
            detail_lines.append(f"\n### {title}")
            if s.get("authors"):
                detail_lines.append(f"Authors: {s['authors']}")
            if s.get("year"):
                detail_lines.append(f"Year: {s['year']}")
            if s.get("type"):
                detail_lines.append(f"Type: {s['type']}")
            if s.get("doi"):
                detail_lines.append(f"DOI: {s['doi']}")
        sections.append("\n".join(detail_lines))

    # Assumptions
    if state.assumption_audits:
        audit_lines = [f"## Assumption Audits ({len(state.assumption_audits)} methods)"]
        for a in state.assumption_audits:
            audit_lines.append(f"\n### {a.method_name}\n{a.raw_output}")
        sections.append("\n".join(audit_lines))

    # Conceptual connections
    if state.connection_explanations:
        conn_lines = [f"## Conceptual Connections ({len(state.connection_explanations)})"]
        for conn in state.connection_explanations:
            conn_lines.append(f"\n### {conn['concept_a']} → {conn['concept_b']}")
            path_len = conn.get("path_length", 0)
            explanation = conn.get("path_explanation", "")
            conn_lines.append(f"Path ({path_len} hops): {explanation}")
            for step in conn.get("path", []):
                evidence = step.get("evidence", [])
                ev_str = f" ({len(evidence)} evidence chunks)" if evidence else ""
                conn_lines.append(
                    f"- {step.get('concept_name', '?')} [{step.get('concept_type', '')}]{ev_str}"
                )
        sections.append("\n".join(conn_lines))

    # Evidence quality metadata -- helps LLM calibrate confidence
    meta_lines = ["## Evidence Quality Metadata"]

    # KB corpus context (from pre-pipeline stats)
    if state.kb_stats_summary:
        meta_lines.append(f"- KB corpus: {state.kb_stats_summary}")
    if state.kb_domains:
        meta_lines.append(f"- KB domains: {', '.join(state.kb_domains)}")

    # Search coverage
    total_results = len(state.search_results)
    if total_results == 0:
        meta_lines.append("- **WARNING: No search results found** -- report is based on KB gaps")
    elif total_results < 3:
        meta_lines.append(f"- **Sparse evidence**: only {total_results} results found")
    else:
        avg_score = sum(r.score for r in state.search_results) / total_results
        meta_lines.append(f"- Search: {total_results} results, avg score {avg_score:.2f}")

    # Score distribution
    if state.search_results:
        high = sum(1 for r in state.search_results if r.score >= 0.8)
        med = sum(1 for r in state.search_results if 0.5 <= r.score < 0.8)
        low = sum(1 for r in state.search_results if r.score < 0.5)
        meta_lines.append(
            f"- Score distribution: {high} high (>=0.8), {med} medium, {low} low (<0.5)"
        )

    # Year recency
    if state.search_results:
        years = [int(r.year) for r in state.search_results if r.year.isdigit()]
        if years:
            meta_lines.append(f"- Year range: {min(years)}-{max(years)}")

    # Concept coverage
    total_concepts = len(state.concepts)
    concepts_with_detail = sum(1 for c in state.concepts if c.description)
    if total_concepts > 0:
        meta_lines.append(
            f"- Concepts: {total_concepts} explored, {concepts_with_detail} with full detail"
        )

    # Assumption coverage
    total_audits = len(state.assumption_audits)
    audits_with_assumptions = sum(1 for a in state.assumption_audits if a.assumptions)
    if total_audits > 0:
        meta_lines.append(
            f"- Assumptions: {total_audits} methods audited, "
            f"{audits_with_assumptions} with structured assumptions"
        )

    sections.append("\n".join(meta_lines))

    return "\n\n---\n\n".join(sections)


async def synthesis_writer(state: ResearchState, config: AgentConfig) -> NodeUpdate:
    """Synthesize all gathered evidence into a structured report.

    Args:
        state: Complete state with all agent outputs populated.
        config: Agent configuration (model selection).

    Returns:
        NodeUpdate with ``report`` and ``confidence_assessment``.

    Raises:
        SynthesisError: If synthesis fails after timeout or LLM error.
    """
    logger.info("Synthesizing research report")

    evidence = _build_evidence_context(state)
    logger.info("Evidence context: %d chars", len(evidence))

    llm = create_llm(
        config.models.synthesis,
        max_tokens=4096,
        temperature=0.1,
    ).with_structured_output(SynthesisReport)

    timeout_s = config.synthesis_timeout
    try:
        async with asyncio.timeout(timeout_s):
            result = await llm.ainvoke(
                [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(
                        content=f"Synthesize the following research evidence:\n\n{evidence}"
                    ),
                ]
            )
            if not isinstance(result, SynthesisReport):
                raise SynthesisError(f"Expected SynthesisReport, got {type(result).__name__}")
    except TimeoutError as e:
        raise SynthesisError(f"Synthesis timed out after {timeout_s}s") from e
    except Exception as e:
        raise SynthesisError(f"Synthesis failed: {e}") from e

    # Post-LLM validation: enforce evidence constraints
    result.key_findings = _validate_findings(result.key_findings, state)

    grounding_warnings = _check_citation_grounding(result, state)
    if grounding_warnings:
        result.confidence_reasoning += "\n\n**Grounding warnings**: " + "; ".join(
            grounding_warnings
        )

    evidence_metadata = _build_evidence_metadata(state, grounding_warnings)

    report_md = result.to_markdown()

    # Append evidence metadata to the report markdown
    meta_lines = [
        "\n\n## Evidence Metadata\n",
        f"- **Total sources**: {evidence_metadata.total_sources}",
        f"- **High-score sources** (>=0.8): {evidence_metadata.high_score_sources}",
        f"- **Average score**: {evidence_metadata.avg_score:.3f}",
    ]
    if evidence_metadata.year_range:
        meta_lines.append(f"- **Year range**: {evidence_metadata.year_range}")
    meta_lines.append(f"- **Concepts explored**: {evidence_metadata.concepts_explored}")
    meta_lines.append(f"- **Methods audited**: {evidence_metadata.methods_audited}")
    if evidence_metadata.grounding_warnings:
        meta_lines.append(f"- **Grounding warnings**: {len(evidence_metadata.grounding_warnings)}")
    meta_lines.append(f"- **Validation applied**: {evidence_metadata.validation_applied}")
    report_md += "\n".join(meta_lines) + "\n"

    confidence = f"**{result.confidence_level}**: {result.confidence_reasoning}"

    logger.info("Report generated: %d chars (validated)", len(report_md))

    return NodeUpdate(
        report=report_md,
        confidence_assessment=confidence,
        evidence_metadata=evidence_metadata.model_dump(),
        current_node="synthesis_writer",
    )
