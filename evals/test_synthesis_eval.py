"""Eval: Synthesis report quality with LLM-as-judge.

Tests that the synthesis node produces coherent, grounded reports.
Uses Haiku as judge to grade Sonnet output for cost efficiency.

Run: ``pytest evals/test_synthesis_eval.py -m eval --timeout=120``
"""

from __future__ import annotations

import pytest

from evals.conftest import load_golden_case, requires_api_key
from evals.judges import grade_synthesis
from evals.metrics import EvalResult, grounding_score
from research_agent.config import AgentConfig, ModelConfig
from research_agent.nodes.synthesis import (
    SynthesisReport,
    _build_evidence_context,
    synthesis_writer,
)
from research_agent.state import (
    AssumptionAudit,
    ConceptInfo,
    ResearchState,
    SearchResult,
    SubTask,
)


def _build_test_state(query: str) -> ResearchState:
    """Build a realistic state for synthesis testing."""
    return ResearchState(
        query=query,
        sub_tasks=[
            SubTask(
                description="Find foundational papers on DML",
                search_queries=["double machine learning"],
                concepts_to_explore=["double machine learning"],
                methods_to_audit=["DML"],
            ),
        ],
        planning_rationale="Decomposed into DML-focused subtask.",
        search_results=[
            SearchResult(
                title="Double/Debiased Machine Learning",
                content="DML provides a framework for treatment effect estimation "
                "using cross-fitting to avoid overfitting bias.",
                source_id="src-001-dml",
                score=0.92,
                authors="Chernozhukov et al. (2018)",
                year="2018",
            ),
            SearchResult(
                title="Debiased ML of CATEs",
                content="Extends DML to conditional average treatment effects "
                "under unconfoundedness.",
                source_id="src-002-cate",
                score=0.87,
                authors="Semenova and Chernozhukov (2021)",
                year="2021",
            ),
        ],
        search_summary="Found 2 results.",
        concepts=[
            ConceptInfo(
                concept_id="concept-dml-001",
                name="double machine learning",
                concept_type="METHOD",
                description="Framework for treatment effect estimation.",
            ),
        ],
        concept_map_summary="1 concept explored.",
        assumption_audits=[
            AssumptionAudit(
                method_name="DML",
                raw_output="Assumptions: unconfoundedness, overlap, Neyman orthogonality.",
            ),
        ],
        assumption_summary="Audited 1 method: DML.",
    )


@requires_api_key
@pytest.mark.eval
async def test_synthesis_with_judge() -> None:
    """Full synthesis + LLM-as-judge grading on DML case."""
    state = _build_test_state("What are the assumptions of double machine learning?")

    config = AgentConfig(
        models=ModelConfig(synthesis="claude-haiku-4-5-20251001"),
        max_search_results=5,
        max_concepts=3,
        max_citations=3,
    )

    result = await synthesis_writer(state, config)
    report = result["report"]

    # Build evidence for judge
    evidence = _build_evidence_context(state)

    # Grade with LLM judge
    verdict = await grade_synthesis(report, evidence)

    eval_result = EvalResult(
        case_name="dml_assumptions",
        dimension="synthesis_judge",
        scores={
            "completeness": float(verdict.completeness),
            "grounding": float(verdict.grounding),
            "gap_honesty": float(verdict.gap_honesty),
            "coherence": float(verdict.coherence),
            "average": verdict.average_score,
        },
    )

    # Minimum thresholds
    errors = []
    if verdict.completeness < 3:
        errors.append(f"Low completeness: {verdict.completeness}")
    if verdict.grounding < 3:
        errors.append(f"Low grounding: {verdict.grounding}")
    if verdict.gap_honesty < 3:
        errors.append(f"Low gap_honesty: {verdict.gap_honesty}")

    if errors:
        eval_result.passed = False
        eval_result.errors = errors
        pytest.fail(f"Synthesis judge eval failed: {'; '.join(errors)}")


@pytest.mark.eval
def test_grounding_score_on_report() -> None:
    """Grounding score computed on a synthetic report."""
    report = (
        "# Research Report\n\n"
        "Chernozhukov et al. (2018) introduced DML.\n"
        "Key assumptions include unconfoundedness and overlap.\n"
        "Cross-fitting eliminates overfitting bias."
    )
    evidence_terms = ["Chernozhukov", "DML", "unconfoundedness", "cross-fitting", "overlap"]

    score = grounding_score(report, evidence_terms)
    assert score >= 0.8, f"Grounding score too low: {score}"


@pytest.mark.eval
def test_synthesis_report_structure() -> None:
    """SynthesisReport.to_markdown() produces expected sections."""
    report = SynthesisReport(
        executive_summary="DML is a causal inference framework.",
        key_findings=["Finding 1", "Finding 2"],
        concept_map="DML -> cross-fitting",
        citation_landscape="Chernozhukov (2018) is foundational.",
        methodological_considerations="Requires unconfoundedness.",
        gaps_limitations="Limited finite-sample coverage.",
        confidence_level="medium",
        confidence_reasoning="Good coverage but some gaps.",
    )

    md = report.to_markdown()
    assert "## Executive Summary" in md
    assert "## Key Findings" in md
    assert "## Concept Map" in md
    assert "## Citation Landscape" in md
    assert "## Methodological Considerations" in md
    assert "## Gaps & Limitations" in md
    assert "## Confidence Assessment" in md
    assert "**Level**: medium" in md


@requires_api_key
@pytest.mark.eval
async def test_synthesis_mentions_dml_terms() -> None:
    """Synthesis report mentions expected terms from DML golden case.

    Only tests DML case since the test state has DML-specific evidence.
    """
    case = load_golden_case("dml_assumptions")
    state = _build_test_state(case["query"])

    config = AgentConfig(
        models=ModelConfig(synthesis="claude-haiku-4-5-20251001"),
        max_search_results=5,
        max_concepts=3,
        max_citations=3,
    )

    result = await synthesis_writer(state, config)
    report = result["report"]

    must_mention: list[str] = case["synthesis"]["must_mention"]
    report_lower = report.lower()

    missing = [term for term in must_mention if term.lower() not in report_lower]
    if missing:
        pytest.fail(f"Report missing expected terms: {missing}")
