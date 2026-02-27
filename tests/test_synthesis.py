"""Tests for synthesis evidence context building.

Validates that _build_evidence_context() includes:
1. Node summary fields (search, concept, citation, assumption)
2. Evidence quality metadata (coverage, score distribution, year range)
"""

from __future__ import annotations

from research_agent.nodes.synthesis import _build_evidence_context
from research_agent.state import (
    AssumptionAudit,
    ConceptInfo,
    ResearchState,
    SearchResult,
)


class TestSummaryFields:
    """Tests for the Analysis Overview section from node summaries."""

    def test_includes_search_summary(self) -> None:
        """search_summary appears in Analysis Overview."""
        state = ResearchState(
            query="test",
            planning_rationale="plan",
            search_summary="Found 5 unique results across 3 queries.",
        )
        ctx = _build_evidence_context(state)
        assert "Analysis Overview" in ctx
        assert "**Literature**: Found 5 unique results" in ctx

    def test_includes_all_four_summaries(self) -> None:
        """All 4 summary fields appear when populated."""
        state = ResearchState(
            query="test",
            planning_rationale="plan",
            search_summary="search info",
            concept_map_summary="concept info",
            citation_summary="citation info",
            assumption_summary="assumption info",
        )
        ctx = _build_evidence_context(state)
        assert "**Literature**: search info" in ctx
        assert "**Concepts**: concept info" in ctx
        assert "**Citations**: citation info" in ctx
        assert "**Assumptions**: assumption info" in ctx

    def test_omits_summary_section_when_all_empty(self) -> None:
        """No Analysis Overview when all summaries are empty."""
        state = ResearchState(query="test", planning_rationale="plan")
        ctx = _build_evidence_context(state)
        assert "Analysis Overview" not in ctx

    def test_partial_summaries_included(self) -> None:
        """Only non-empty summaries appear in Analysis Overview."""
        state = ResearchState(
            query="test",
            planning_rationale="plan",
            search_summary="search",
            concept_map_summary="",
            citation_summary="citations",
        )
        ctx = _build_evidence_context(state)
        assert "**Literature**: search" in ctx
        assert "**Citations**: citations" in ctx
        assert "**Concepts**" not in ctx


class TestEvidenceQualityMetadata:
    """Tests for the Evidence Quality Metadata section."""

    def test_no_results_warning(self) -> None:
        """Zero search results triggers WARNING."""
        state = ResearchState(query="test", planning_rationale="plan")
        ctx = _build_evidence_context(state)
        assert "WARNING: No search results found" in ctx

    def test_sparse_evidence(self) -> None:
        """1-2 results triggers sparse evidence warning."""
        state = ResearchState(
            query="test",
            planning_rationale="plan",
            search_results=[
                SearchResult(title="A", content="", source_id="s1", score=0.8),
            ],
        )
        ctx = _build_evidence_context(state)
        assert "Sparse evidence" in ctx
        assert "only 1 results found" in ctx

    def test_normal_coverage_shows_avg_score(self) -> None:
        """3+ results shows count and average score."""
        state = ResearchState(
            query="test",
            planning_rationale="plan",
            search_results=[
                SearchResult(title="A", content="", source_id="s1", score=0.9),
                SearchResult(title="B", content="", source_id="s2", score=0.6),
                SearchResult(title="C", content="", source_id="s3", score=0.3),
            ],
        )
        ctx = _build_evidence_context(state)
        assert "3 results" in ctx
        assert "avg score 0.60" in ctx

    def test_score_distribution(self) -> None:
        """Score distribution buckets: high (>=0.8), medium, low (<0.5)."""
        state = ResearchState(
            query="test",
            planning_rationale="plan",
            search_results=[
                SearchResult(title="A", content="", source_id="s1", score=0.9),
                SearchResult(title="B", content="", source_id="s2", score=0.6),
                SearchResult(title="C", content="", source_id="s3", score=0.3),
            ],
        )
        ctx = _build_evidence_context(state)
        assert "1 high" in ctx
        assert "1 medium" in ctx
        assert "1 low" in ctx

    def test_year_range(self) -> None:
        """Year range extracted from search results."""
        state = ResearchState(
            query="test",
            planning_rationale="plan",
            search_results=[
                SearchResult(title="A", content="", source_id="s1", score=0.8, year="2018"),
                SearchResult(title="B", content="", source_id="s2", score=0.7, year="2023"),
                SearchResult(title="C", content="", source_id="s3", score=0.6, year="2020"),
            ],
        )
        ctx = _build_evidence_context(state)
        assert "2018" in ctx
        assert "2023" in ctx

    def test_year_range_skips_non_numeric(self) -> None:
        """Non-numeric years are silently skipped."""
        state = ResearchState(
            query="test",
            planning_rationale="plan",
            search_results=[
                SearchResult(title="A", content="", source_id="s1", score=0.8, year="2020"),
                SearchResult(title="B", content="", source_id="s2", score=0.7, year=""),
                SearchResult(title="C", content="", source_id="s3", score=0.6, year="n/a"),
            ],
        )
        ctx = _build_evidence_context(state)
        assert "2020" in ctx

    def test_concept_coverage(self) -> None:
        """Concept count and detail count reported."""
        state = ResearchState(
            query="test",
            planning_rationale="plan",
            concepts=[
                ConceptInfo(concept_id="c1", name="DML", description="A method"),
                ConceptInfo(concept_id="c2", name="IV", description=""),
            ],
        )
        ctx = _build_evidence_context(state)
        assert "2 explored" in ctx
        assert "1 with full detail" in ctx

    def test_assumption_coverage(self) -> None:
        """Assumption audit count and structured count reported."""
        state = ResearchState(
            query="test",
            planning_rationale="plan",
            assumption_audits=[
                AssumptionAudit(
                    method_name="DML",
                    assumptions=[{"name": "unconf"}],
                    raw_output="...",
                ),
                AssumptionAudit(method_name="IV", assumptions=[], raw_output="failed"),
            ],
        )
        ctx = _build_evidence_context(state)
        assert "2 methods audited" in ctx
        assert "1 with structured assumptions" in ctx

    def test_no_concepts_omits_concept_line(self) -> None:
        """No concept metadata when concepts list is empty."""
        state = ResearchState(query="test", planning_rationale="plan")
        ctx = _build_evidence_context(state)
        assert "Concepts:" not in ctx
        # But metadata section header is always present
        assert "Evidence Quality Metadata" in ctx

    def test_no_audits_omits_assumption_line(self) -> None:
        """No assumption metadata when audits list is empty."""
        state = ResearchState(query="test", planning_rationale="plan")
        ctx = _build_evidence_context(state)
        assert "Assumptions:" not in ctx
