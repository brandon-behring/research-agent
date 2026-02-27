"""Unit tests for synthesis validation functions -- no API key required.

Tests _validate_findings and _check_citation_grounding independently
from synthesis_writer, using real ResearchState objects with known evidence.
"""

from __future__ import annotations

from research_agent.nodes.synthesis import (
    Finding,
    SynthesisReport,
    _build_evidence_metadata,
    _check_citation_grounding,
    _validate_findings,
)
from research_agent.state import (
    AssumptionAudit,
    ConceptInfo,
    ResearchState,
    SearchResult,
)


def _dml_state() -> ResearchState:
    """Build a DML-focused state for validation testing."""
    return ResearchState(
        query="What are the assumptions of double machine learning?",
        search_results=[
            SearchResult(
                title="Double/Debiased Machine Learning",
                content="DML provides treatment effect estimation using cross-fitting "
                "to avoid overfitting bias. Relies on unconfoundedness.",
                source_id="s1",
                score=0.92,
                authors="Chernozhukov et al.",
                year="2018",
            ),
            SearchResult(
                title="Debiased ML of CATEs",
                content="Extends double machine learning to conditional average "
                "treatment effects under unconfoundedness assumption.",
                source_id="s2",
                score=0.87,
                authors="Semenova and Chernozhukov",
                year="2021",
            ),
            SearchResult(
                title="Cross-fitting in high dimensions",
                content="Cross-fitting prevents overfitting in double machine "
                "learning estimation of treatment effects.",
                source_id="s3",
                score=0.80,
                authors="Newey and Robins",
                year="2018",
            ),
        ],
        concepts=[
            ConceptInfo(
                concept_id="c1",
                name="double machine learning",
                concept_type="METHOD",
                description="Framework for treatment effect estimation.",
            ),
        ],
        assumption_audits=[
            AssumptionAudit(method_name="DML", raw_output="Requires unconfoundedness."),
        ],
    )


class TestValidateFindingsIntegration:
    """Integration-style tests for _validate_findings with realistic state."""

    def test_grounded_finding_keeps_high(self) -> None:
        """Finding well-supported by 3 DML sources retains HIGH confidence."""
        state = _dml_state()
        findings = [
            Finding(
                text="Double machine learning uses cross-fitting to avoid overfitting bias",
                confidence="high",
                source_count=3,
            )
        ]
        validated = _validate_findings(findings, state)
        assert validated[0].confidence == "high"
        assert validated[0].source_count == 3

    def test_ungrounded_finding_forced_low(self) -> None:
        """Finding about unrelated topic forced to LOW."""
        state = _dml_state()
        findings = [
            Finding(
                text="Quantum entanglement drives decoherence patterns",
                confidence="high",
                source_count=5,
            )
        ]
        validated = _validate_findings(findings, state)
        assert validated[0].confidence == "low"
        assert validated[0].source_count == 0

    def test_source_count_capped_to_reality(self) -> None:
        """LLM claims 10 sources but only 3 exist → capped to actual count."""
        state = _dml_state()
        findings = [
            Finding(
                text="Double machine learning treatment effects",
                confidence="high",
                source_count=10,
            )
        ]
        validated = _validate_findings(findings, state)
        assert validated[0].source_count <= len(state.search_results)


class TestCitationGroundingIntegration:
    """Integration-style tests for _check_citation_grounding with realistic state."""

    def test_chernozhukov_2018_grounded(self) -> None:
        """Chernozhukov (2018) is in search_results → no warning."""
        state = _dml_state()
        report = SynthesisReport(
            executive_summary="[Chernozhukov (2018)] established the DML framework.",
            key_findings=[Finding(text="F", confidence="low")],
            concept_map="C",
            citation_landscape="Cite",
            methodological_considerations="M",
            gaps_limitations="G",
            confidence_level="medium",
            confidence_reasoning="R",
        )
        warnings = _check_citation_grounding(report, state)
        grounded_warnings = [w for w in warnings if "Chernozhukov" in w]
        assert len(grounded_warnings) == 0

    def test_fabricated_citation_warned(self) -> None:
        """Fabricated author not in evidence → warning generated."""
        state = _dml_state()
        report = SynthesisReport(
            executive_summary="[FakeAuthor (2025)] showed novel results.",
            key_findings=[Finding(text="F", confidence="low")],
            concept_map="C",
            citation_landscape="Cite",
            methodological_considerations="M",
            gaps_limitations="G",
            confidence_level="medium",
            confidence_reasoning="R",
        )
        warnings = _check_citation_grounding(report, state)
        assert any("FakeAuthor" in w for w in warnings)


class TestEvidenceMetadataIntegration:
    """Integration-style tests for _build_evidence_metadata."""

    def test_dml_state_metadata(self) -> None:
        """Metadata from DML state has correct counts."""
        state = _dml_state()
        meta = _build_evidence_metadata(state, [])
        assert meta.total_sources == 3
        assert meta.high_score_sources >= 1
        assert meta.concepts_explored == 1
        assert meta.methods_audited == 1
        assert meta.year_range == "2018-2021"

    def test_metadata_with_warnings(self) -> None:
        """Grounding warnings included in metadata."""
        state = _dml_state()
        warnings = ["'Fake (2025)' not found in evidence"]
        meta = _build_evidence_metadata(state, warnings)
        assert meta.grounding_warnings == warnings
        assert meta.validation_applied is True
