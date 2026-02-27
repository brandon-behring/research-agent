"""Tests for synthesis evidence context building.

Validates that _build_evidence_context() includes:
1. Node summary fields (search, concept, citation, assumption)
2. Evidence quality metadata (coverage, score distribution, year range)
"""

from __future__ import annotations

from research_agent.nodes.synthesis import (
    EvidenceMetadata,
    Finding,
    SynthesisReport,
    _build_evidence_context,
    _build_evidence_metadata,
    _check_citation_grounding,
    _count_matching_sources,
    _extract_terms,
    _validate_findings,
)
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


class TestFindingModel:
    """Tests for the Finding model and rendering."""

    def test_finding_to_markdown_high_with_sources(self) -> None:
        """HIGH finding renders with source count."""
        f = Finding(text="DML uses cross-fitting", confidence="high", source_count=3)
        assert f.to_markdown() == "[HIGH] (3 sources) DML uses cross-fitting"

    def test_finding_to_markdown_low_no_sources(self) -> None:
        """LOW finding without source count omits parenthetical."""
        f = Finding(text="Weak evidence", confidence="low")
        assert f.to_markdown() == "[LOW] Weak evidence"

    def test_finding_to_markdown_medium(self) -> None:
        """MEDIUM finding renders correctly."""
        f = Finding(text="Moderate support", confidence="medium", source_count=1)
        assert f.to_markdown() == "[MEDIUM] (1 sources) Moderate support"


class TestSynthesisReportMarkdown:
    """Tests for SynthesisReport.to_markdown() with Finding objects."""

    def test_findings_rendered_with_confidence(self) -> None:
        """Key findings rendered with confidence tags in markdown."""
        report = SynthesisReport(
            executive_summary="Summary.",
            key_findings=[
                Finding(text="Finding A", confidence="high", source_count=2),
                Finding(text="Finding B", confidence="low"),
            ],
            concept_map="A -> B",
            citation_landscape="Cite.",
            methodological_considerations="Method.",
            gaps_limitations="Gap.",
            confidence_level="medium",
            confidence_reasoning="Reasoning.",
        )
        md = report.to_markdown()
        assert "[HIGH] (2 sources) Finding A" in md
        assert "[LOW] Finding B" in md
        assert "## Key Findings" in md

    def test_mermaid_concept_map_rendered(self) -> None:
        """concept_map_mermaid renders as fenced code block."""
        report = SynthesisReport(
            executive_summary="S",
            key_findings=[Finding(text="F", confidence="low")],
            concept_map="A -> B",
            concept_map_mermaid="graph TD\n  A[DML] -->|uses| B[Cross-fitting]",
            citation_landscape="C",
            methodological_considerations="M",
            gaps_limitations="G",
            confidence_level="low",
            confidence_reasoning="R",
        )
        md = report.to_markdown()
        assert "```mermaid" in md
        assert "graph TD" in md
        assert "A[DML] -->|uses| B[Cross-fitting]" in md

    def test_no_mermaid_when_empty(self) -> None:
        """Empty concept_map_mermaid omits fenced block."""
        report = SynthesisReport(
            executive_summary="S",
            key_findings=[Finding(text="F", confidence="low")],
            concept_map="A -> B",
            citation_landscape="C",
            methodological_considerations="M",
            gaps_limitations="G",
            confidence_level="low",
            confidence_reasoning="R",
        )
        md = report.to_markdown()
        assert "```mermaid" not in md

    def test_next_questions_rendered(self) -> None:
        """next_questions renders as bulleted list."""
        report = SynthesisReport(
            executive_summary="S",
            key_findings=[Finding(text="F", confidence="low")],
            concept_map="A -> B",
            citation_landscape="C",
            methodological_considerations="M",
            gaps_limitations="G",
            next_questions=["What about X?", "How does Y work?"],
            confidence_level="low",
            confidence_reasoning="R",
        )
        md = report.to_markdown()
        assert "## Next Research Questions" in md
        assert "- What about X?" in md
        assert "- How does Y work?" in md

    def test_no_next_questions_when_empty(self) -> None:
        """Empty next_questions omits section."""
        report = SynthesisReport(
            executive_summary="S",
            key_findings=[Finding(text="F", confidence="low")],
            concept_map="A -> B",
            citation_landscape="C",
            methodological_considerations="M",
            gaps_limitations="G",
            confidence_level="low",
            confidence_reasoning="R",
        )
        md = report.to_markdown()
        assert "Next Research Questions" not in md


# ── Validation tests (Phase 8) ───────────────────────────────────────


def _make_search_results(
    n: int = 3, base_score: float = 0.8, content: str = "double machine learning cross-fitting"
) -> list[SearchResult]:
    """Helper to create search results for validation tests."""
    return [
        SearchResult(
            title=f"Paper {i}",
            content=content,
            source_id=f"s{i}",
            score=base_score + (i * 0.01),
            authors=f"Author{i} et al.",
            year=str(2018 + i),
        )
        for i in range(n)
    ]


class TestExtractTerms:
    """Tests for _extract_terms."""

    def test_extracts_long_words(self) -> None:
        """Words >= 5 chars are extracted."""
        terms = _extract_terms("The double machine learning method")
        assert "double" in terms
        assert "machine" in terms
        assert "learning" in terms
        assert "method" in terms

    def test_filters_short_words(self) -> None:
        """Words < 5 chars excluded."""
        terms = _extract_terms("DML is a good method for this task")
        assert "good" not in terms
        assert "this" not in terms

    def test_filters_stopwords(self) -> None:
        """Stopwords excluded even if long enough."""
        terms = _extract_terms("using these other approaches between models")
        assert "using" not in terms
        assert "these" not in terms
        assert "other" not in terms
        assert "between" not in terms
        assert "approaches" in terms
        assert "models" in terms

    def test_deduplicates(self) -> None:
        """Duplicate terms appear only once."""
        terms = _extract_terms("learning and learning again learning")
        assert terms.count("learning") == 1

    def test_empty_text(self) -> None:
        """Empty text returns empty list."""
        assert _extract_terms("") == []


class TestCountMatchingSources:
    """Tests for _count_matching_sources."""

    def test_matches_sources_with_overlapping_terms(self) -> None:
        """Sources with >= 2 matching terms count as matches."""
        results = _make_search_results(3, content="double machine learning cross-fitting")
        state = ResearchState(query="test", search_results=results)
        terms = ["double", "machine", "learning"]
        count, avg = _count_matching_sources(terms, state)
        assert count == 3
        assert avg > 0.0

    def test_no_matches_returns_zero(self) -> None:
        """Sources with no overlapping terms return 0."""
        results = _make_search_results(3, content="completely unrelated content here")
        state = ResearchState(query="test", search_results=results)
        terms = ["quantum", "physics", "entanglement"]
        count, avg = _count_matching_sources(terms, state)
        assert count == 0
        assert avg == 0.0

    def test_empty_search_results(self) -> None:
        """Empty search_results returns 0."""
        state = ResearchState(query="test")
        count, avg = _count_matching_sources(["term1", "term2"], state)
        assert count == 0

    def test_empty_terms(self) -> None:
        """Empty terms returns 0."""
        results = _make_search_results(3)
        state = ResearchState(query="test", search_results=results)
        count, avg = _count_matching_sources([], state)
        assert count == 0

    def test_single_term_finding_matches_with_one_overlap(self) -> None:
        """A finding with only 1 significant term matches on 1 overlap."""
        results = _make_search_results(1, content="estimation using propensity")
        state = ResearchState(query="test", search_results=results)
        # Only one term — min overlap is min(2, 1) = 1
        count, _ = _count_matching_sources(["estimation"], state)
        assert count == 1


class TestValidateFindings:
    """Tests for _validate_findings."""

    def test_inflated_source_count_gets_capped(self) -> None:
        """LLM-claimed source_count is capped to actual matching sources."""
        results = _make_search_results(2, content="double machine learning estimation")
        state = ResearchState(query="test", search_results=results)
        findings = [
            Finding(
                text="Double machine learning provides robust estimation",
                confidence="high",
                source_count=10,
            )
        ]
        validated = _validate_findings(findings, state)
        assert validated[0].source_count <= 2

    def test_zero_matches_forces_low(self) -> None:
        """Finding with 0 matching sources → forced LOW."""
        results = _make_search_results(3, content="completely unrelated topic here")
        state = ResearchState(query="test", search_results=results)
        findings = [
            Finding(
                text="Quantum entanglement drives decoherence",
                confidence="high",
                source_count=5,
            )
        ]
        validated = _validate_findings(findings, state)
        assert validated[0].confidence == "low"
        assert validated[0].source_count == 0

    def test_one_to_two_matches_caps_medium(self) -> None:
        """Finding with 1-2 matching sources → capped MEDIUM."""
        results = _make_search_results(2, content="double machine learning estimation")
        state = ResearchState(query="test", search_results=results)
        findings = [
            Finding(
                text="Double machine learning provides estimation",
                confidence="high",
                source_count=5,
            )
        ]
        validated = _validate_findings(findings, state)
        assert validated[0].confidence == "medium"

    def test_three_plus_high_score_allows_high(self) -> None:
        """Finding with 3+ high-score matches → HIGH allowed."""
        results = _make_search_results(4, base_score=0.85, content="double machine learning")
        state = ResearchState(query="test", search_results=results)
        findings = [
            Finding(
                text="Double machine learning is well-supported",
                confidence="high",
                source_count=4,
            )
        ]
        validated = _validate_findings(findings, state)
        assert validated[0].confidence == "high"

    def test_three_plus_low_score_caps_medium(self) -> None:
        """Finding with 3+ matches but low avg score → capped MEDIUM."""
        results = _make_search_results(4, base_score=0.3, content="double machine learning")
        state = ResearchState(query="test", search_results=results)
        findings = [
            Finding(
                text="Double machine learning is well-supported",
                confidence="high",
                source_count=4,
            )
        ]
        validated = _validate_findings(findings, state)
        assert validated[0].confidence == "medium"

    def test_empty_search_results_all_low(self) -> None:
        """Empty search_results → all findings forced LOW."""
        state = ResearchState(query="test")
        findings = [
            Finding(text="Some finding about methods", confidence="high", source_count=3),
            Finding(text="Another finding here today", confidence="medium", source_count=1),
        ]
        validated = _validate_findings(findings, state)
        assert all(f.confidence == "low" for f in validated)
        assert all(f.source_count == 0 for f in validated)

    def test_short_finding_text_handled_gracefully(self) -> None:
        """Very short finding text with few terms still works."""
        results = _make_search_results(1, content="short text match")
        state = ResearchState(query="test", search_results=results)
        findings = [Finding(text="short", confidence="medium", source_count=1)]
        validated = _validate_findings(findings, state)
        # Should not crash, graceful handling
        assert len(validated) == 1

    def test_low_confidence_not_upgraded(self) -> None:
        """Validation never upgrades confidence, only downgrades."""
        results = _make_search_results(5, base_score=0.9, content="double machine learning")
        state = ResearchState(query="test", search_results=results)
        findings = [
            Finding(
                text="Double machine learning provides estimation",
                confidence="low",
                source_count=1,
            )
        ]
        validated = _validate_findings(findings, state)
        assert validated[0].confidence == "low"

    def test_multiple_findings_validated_independently(self) -> None:
        """Each finding validated against evidence independently."""
        results = _make_search_results(
            3, base_score=0.85, content="double machine learning cross-fitting"
        )
        state = ResearchState(query="test", search_results=results)
        findings = [
            Finding(
                text="Double machine learning uses cross-fitting",
                confidence="high",
                source_count=3,
            ),
            Finding(
                text="Quantum physics drives entanglement",
                confidence="high",
                source_count=5,
            ),
        ]
        validated = _validate_findings(findings, state)
        assert validated[0].confidence == "high"
        assert validated[1].confidence == "low"


class TestCheckCitationGrounding:
    """Tests for _check_citation_grounding."""

    def test_grounded_citation_no_warnings(self) -> None:
        """Citation matching search_results author+year → no warning."""
        state = ResearchState(
            query="test",
            search_results=[
                SearchResult(
                    title="DML Paper",
                    content="Content",
                    source_id="s1",
                    score=0.9,
                    authors="Chernozhukov et al.",
                    year="2018",
                ),
            ],
        )
        report = SynthesisReport(
            executive_summary="Chernozhukov et al. [Chernozhukov (2018)] showed DML works.",
            key_findings=[Finding(text="F", confidence="low")],
            concept_map="C",
            citation_landscape="Cite",
            methodological_considerations="M",
            gaps_limitations="G",
            confidence_level="medium",
            confidence_reasoning="R",
        )
        warnings = _check_citation_grounding(report, state)
        assert len(warnings) == 0

    def test_fabricated_author_generates_warning(self) -> None:
        """Citation with unknown author+year → warning."""
        state = ResearchState(
            query="test",
            search_results=[
                SearchResult(
                    title="Paper",
                    content="C",
                    source_id="s1",
                    score=0.9,
                    authors="Smith et al.",
                    year="2020",
                ),
            ],
        )
        report = SynthesisReport(
            executive_summary="As shown by [Fabricated (2025)] in their work.",
            key_findings=[Finding(text="F", confidence="low")],
            concept_map="C",
            citation_landscape="Cite",
            methodological_considerations="M",
            gaps_limitations="G",
            confidence_level="medium",
            confidence_reasoning="R",
        )
        warnings = _check_citation_grounding(report, state)
        assert len(warnings) == 1
        assert "Fabricated" in warnings[0]

    def test_wrong_year_generates_warning(self) -> None:
        """Author from evidence but wrong year → warning."""
        state = ResearchState(
            query="test",
            search_results=[
                SearchResult(
                    title="Paper",
                    content="C",
                    source_id="s1",
                    score=0.9,
                    authors="Smith et al.",
                    year="2020",
                ),
            ],
        )
        report = SynthesisReport(
            executive_summary="As shown by [Smith (2025)] in their work.",
            key_findings=[Finding(text="F", confidence="low")],
            concept_map="C",
            citation_landscape="Cite",
            methodological_considerations="M",
            gaps_limitations="G",
            confidence_level="medium",
            confidence_reasoning="R",
        )
        warnings = _check_citation_grounding(report, state)
        assert len(warnings) >= 1

    def test_mixed_grounded_and_ungrounded(self) -> None:
        """Mix of grounded and ungrounded → only ungrounded warned."""
        state = ResearchState(
            query="test",
            search_results=[
                SearchResult(
                    title="Paper",
                    content="C",
                    source_id="s1",
                    score=0.9,
                    authors="Smith et al.",
                    year="2020",
                ),
            ],
        )
        report = SynthesisReport(
            executive_summary="[Smith (2020)] showed X. [Jones (2019)] showed Y.",
            key_findings=[Finding(text="F", confidence="low")],
            concept_map="C",
            citation_landscape="Cite",
            methodological_considerations="M",
            gaps_limitations="G",
            confidence_level="medium",
            confidence_reasoning="R",
        )
        warnings = _check_citation_grounding(report, state)
        assert len(warnings) == 1
        assert "Jones" in warnings[0]

    def test_no_citations_in_report_no_warnings(self) -> None:
        """Report without citation patterns → empty warnings."""
        state = ResearchState(
            query="test",
            search_results=_make_search_results(2),
        )
        report = SynthesisReport(
            executive_summary="This is a plain summary without citations.",
            key_findings=[Finding(text="F", confidence="low")],
            concept_map="C",
            citation_landscape="Cite",
            methodological_considerations="M",
            gaps_limitations="G",
            confidence_level="medium",
            confidence_reasoning="R",
        )
        warnings = _check_citation_grounding(report, state)
        assert len(warnings) == 0

    def test_source_details_provide_author_coverage(self) -> None:
        """Authors from source_details also ground citations."""
        state = ResearchState(
            query="test",
            search_results=[],
            source_details=[
                {"authors": "DetailAuthor et al.", "year": "2021"},
            ],
        )
        report = SynthesisReport(
            executive_summary="[DetailAuthor (2021)] showed interesting results.",
            key_findings=[Finding(text="F", confidence="low")],
            concept_map="C",
            citation_landscape="Cite",
            methodological_considerations="M",
            gaps_limitations="G",
            confidence_level="medium",
            confidence_reasoning="R",
        )
        warnings = _check_citation_grounding(report, state)
        assert len(warnings) == 0

    def test_parenthetical_citation_format(self) -> None:
        """(Author, Year) format is also detected."""
        state = ResearchState(
            query="test",
            search_results=[],
        )
        report = SynthesisReport(
            executive_summary="This was shown (FakeAuthor, 2023) in their study.",
            key_findings=[Finding(text="F", confidence="low")],
            concept_map="C",
            citation_landscape="Cite",
            methodological_considerations="M",
            gaps_limitations="G",
            confidence_level="medium",
            confidence_reasoning="R",
        )
        warnings = _check_citation_grounding(report, state)
        assert len(warnings) >= 1
        assert "FakeAuthor" in warnings[0]


class TestEvidenceMetadata:
    """Tests for EvidenceMetadata and _build_evidence_metadata."""

    def test_metadata_from_populated_state(self) -> None:
        """Metadata correctly computed from state with results."""
        results = _make_search_results(5, base_score=0.7)
        state = ResearchState(
            query="test",
            search_results=results,
            concepts=[
                ConceptInfo(concept_id="c1", name="DML"),
                ConceptInfo(concept_id="c2", name="IV"),
            ],
            assumption_audits=[
                AssumptionAudit(method_name="DML", raw_output="..."),
            ],
        )
        meta = _build_evidence_metadata(state, ["warning 1"])
        assert meta.total_sources == 5
        assert meta.high_score_sources >= 0
        assert meta.avg_score > 0.0
        assert meta.concepts_explored == 2
        assert meta.methods_audited == 1
        assert meta.grounding_warnings == ["warning 1"]
        assert meta.validation_applied is True

    def test_metadata_from_empty_state(self) -> None:
        """Empty state → sensible defaults."""
        state = ResearchState(query="test")
        meta = _build_evidence_metadata(state, [])
        assert meta.total_sources == 0
        assert meta.high_score_sources == 0
        assert meta.avg_score == 0.0
        assert meta.year_range == ""
        assert meta.concepts_explored == 0
        assert meta.methods_audited == 0
        assert meta.grounding_warnings == []

    def test_year_range_computed(self) -> None:
        """Year range extracted from search results."""
        results = [
            SearchResult(title="A", content="C", source_id="s1", score=0.8, year="2018"),
            SearchResult(title="B", content="C", source_id="s2", score=0.7, year="2023"),
        ]
        state = ResearchState(query="test", search_results=results)
        meta = _build_evidence_metadata(state, [])
        assert meta.year_range == "2018-2023"

    def test_single_year(self) -> None:
        """Single year result → just that year."""
        results = [
            SearchResult(title="A", content="C", source_id="s1", score=0.8, year="2020"),
        ]
        state = ResearchState(query="test", search_results=results)
        meta = _build_evidence_metadata(state, [])
        assert meta.year_range == "2020"

    def test_high_score_sources_counted(self) -> None:
        """Sources with score >= 0.8 counted correctly."""
        results = [
            SearchResult(title="A", content="C", source_id="s1", score=0.9),
            SearchResult(title="B", content="C", source_id="s2", score=0.85),
            SearchResult(title="C", content="C", source_id="s3", score=0.5),
        ]
        state = ResearchState(query="test", search_results=results)
        meta = _build_evidence_metadata(state, [])
        assert meta.high_score_sources == 2

    def test_metadata_model_dump(self) -> None:
        """EvidenceMetadata serializes to dict correctly."""
        meta = EvidenceMetadata(
            total_sources=3,
            high_score_sources=2,
            avg_score=0.85,
            year_range="2018-2023",
            concepts_explored=5,
            methods_audited=2,
            grounding_warnings=["w1"],
            validation_applied=True,
        )
        d = meta.model_dump()
        assert d["total_sources"] == 3
        assert d["grounding_warnings"] == ["w1"]
