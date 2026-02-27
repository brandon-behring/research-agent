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


class TestConceptualConnections:
    """Tests for the Conceptual Connections section in evidence context."""

    def test_includes_connections_when_present(self) -> None:
        """Conceptual Connections section included when explanations exist."""
        state = ResearchState(
            query="test",
            planning_rationale="plan",
            connection_explanations=[
                {
                    "concept_a": "DML",
                    "concept_b": "cross-fitting",
                    "path_length": 1,
                    "path_explanation": "DML directly uses cross-fitting.",
                    "path": [
                        {
                            "concept_name": "DML",
                            "concept_type": "METHOD",
                            "evidence": [{"text": "...", "source": "..."}],
                        },
                        {
                            "concept_name": "cross-fitting",
                            "concept_type": "METHOD",
                            "evidence": [],
                        },
                    ],
                }
            ],
        )
        ctx = _build_evidence_context(state)
        assert "Conceptual Connections (1)" in ctx
        assert "DML → cross-fitting" in ctx
        assert "1 hops" in ctx
        assert "DML [METHOD] (1 evidence chunks)" in ctx
        assert "cross-fitting [METHOD]" in ctx

    def test_omits_connections_when_empty(self) -> None:
        """No Conceptual Connections section when list is empty."""
        state = ResearchState(query="test", planning_rationale="plan")
        ctx = _build_evidence_context(state)
        assert "Conceptual Connections" not in ctx

    def test_multiple_connections(self) -> None:
        """Multiple connections all rendered."""
        state = ResearchState(
            query="test",
            planning_rationale="plan",
            connection_explanations=[
                {
                    "concept_a": "A",
                    "concept_b": "B",
                    "path_length": 2,
                    "path_explanation": "A connects to B via C.",
                    "path": [],
                },
                {
                    "concept_a": "X",
                    "concept_b": "Y",
                    "path_length": 1,
                    "path_explanation": "X relates to Y.",
                    "path": [],
                },
            ],
        )
        ctx = _build_evidence_context(state)
        assert "Conceptual Connections (2)" in ctx
        assert "A → B" in ctx
        assert "X → Y" in ctx


class TestSimilarConceptsSection:
    """Tests for similar concepts in evidence context."""

    def test_includes_similar_concepts(self) -> None:
        """Similar concepts section appears when populated."""
        state = ResearchState(
            query="test",
            planning_rationale="plan",
            similar_concepts=[
                {"name": "TMLE", "similarity": 0.87, "source_concept": "DML"},
                {"name": "Cross-fitting", "similarity": 0.85, "source_concept": "DML"},
            ],
        )
        ctx = _build_evidence_context(state)
        assert "Embedding-Similar Concepts (2)" in ctx
        assert "TMLE" in ctx
        assert "87%" in ctx

    def test_omits_similar_concepts_when_empty(self) -> None:
        """No similar concepts section when list is empty."""
        state = ResearchState(query="test", planning_rationale="plan")
        ctx = _build_evidence_context(state)
        assert "Embedding-Similar" not in ctx


class TestCrossDomainSection:
    """Tests for cross-domain bridges in evidence context."""

    def test_includes_cross_domain_matches(self) -> None:
        """Cross-domain section appears when populated."""
        state = ResearchState(
            query="test",
            planning_rationale="plan",
            cross_domain_matches=[
                {
                    "source": "IV",
                    "target": "Granger Causality",
                    "source_domain": "causal_inference",
                    "target_domain": "time_series",
                    "link_type": "ANALOGOUS",
                },
            ],
        )
        ctx = _build_evidence_context(state)
        assert "Cross-Domain Bridges (1)" in ctx
        assert "IV (causal_inference)" in ctx
        assert "Granger Causality (time_series)" in ctx

    def test_omits_cross_domain_when_empty(self) -> None:
        """No cross-domain section when list is empty."""
        state = ResearchState(query="test", planning_rationale="plan")
        ctx = _build_evidence_context(state)
        assert "Cross-Domain" not in ctx


class TestSourceDetailsSection:
    """Tests for source details in evidence context."""

    def test_includes_source_details(self) -> None:
        """Source details section appears when source_details populated."""
        state = ResearchState(
            query="test",
            planning_rationale="plan",
            source_details=[
                {
                    "title": "Double Machine Learning",
                    "authors": "Chernozhukov et al.",
                    "year": "2018",
                    "type": "Paper",
                    "doi": "10.1111/ectj.12097",
                }
            ],
        )
        ctx = _build_evidence_context(state)
        assert "Source Details (1 enriched)" in ctx
        assert "Double Machine Learning" in ctx
        assert "Chernozhukov" in ctx
        assert "10.1111" in ctx

    def test_omits_source_details_when_empty(self) -> None:
        """No source details section when list is empty."""
        state = ResearchState(query="test", planning_rationale="plan")
        ctx = _build_evidence_context(state)
        assert "Source Details" not in ctx


class TestKBContextInMetadata:
    """Tests for KB context in evidence quality metadata."""

    def test_includes_kb_stats(self) -> None:
        """kb_stats_summary appears in evidence metadata."""
        state = ResearchState(
            query="test",
            planning_rationale="plan",
            kb_stats_summary="495 sources, 226K chunks",
        )
        ctx = _build_evidence_context(state)
        assert "KB corpus: 495 sources" in ctx

    def test_includes_kb_domains(self) -> None:
        """kb_domains appear in evidence metadata."""
        state = ResearchState(
            query="test",
            planning_rationale="plan",
            kb_domains=["causal_inference", "time_series"],
        )
        ctx = _build_evidence_context(state)
        assert "KB domains: causal_inference, time_series" in ctx

    def test_omits_kb_context_when_empty(self) -> None:
        """No KB context lines when fields are empty."""
        state = ResearchState(query="test", planning_rationale="plan")
        ctx = _build_evidence_context(state)
        assert "KB corpus" not in ctx
        assert "KB domains" not in ctx
