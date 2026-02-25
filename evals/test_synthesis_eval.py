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


def _build_dml_state() -> ResearchState:
    """Build a realistic state for DML synthesis testing."""
    return ResearchState(
        query="What are the assumptions of double machine learning?",
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
        search_summary="Found 2 results on DML.",
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


def _build_iv_rdd_state() -> ResearchState:
    """Build a realistic state for IV vs RDD comparison synthesis testing."""
    return ResearchState(
        query="Compare instrumental variables and regression discontinuity design",
        sub_tasks=[
            SubTask(
                description="Find foundational papers on instrumental variables",
                search_queries=["instrumental variables"],
                concepts_to_explore=["instrumental variables"],
                methods_to_audit=["IV"],
            ),
            SubTask(
                description="Find foundational papers on regression discontinuity",
                search_queries=["regression discontinuity design"],
                concepts_to_explore=["regression discontinuity"],
                methods_to_audit=["RDD"],
            ),
        ],
        planning_rationale="Decomposed into IV and RDD subtasks for comparison.",
        search_results=[
            SearchResult(
                title="Identification and Estimation of Local Average Treatment Effects",
                content="Instrumental variables exploit exogenous variation to identify "
                "causal effects under the exclusion restriction. The LATE framework "
                "identifies effects for compliers when instruments are binary.",
                source_id="src-010-iv",
                score=0.94,
                authors="Angrist and Imbens (1994)",
                year="1994",
            ),
            SearchResult(
                title="Regression-Discontinuity Analysis",
                content="Regression discontinuity designs exploit sharp cutoffs in "
                "treatment assignment. Units just above and below the threshold are "
                "compared as quasi-experimental groups.",
                source_id="src-011-rdd",
                score=0.91,
                authors="Thistlethwaite and Campbell (1960)",
                year="1960",
            ),
            SearchResult(
                title="Practical Guide to RDD",
                content="Bandwidth selection is critical: too narrow loses power, "
                "too wide introduces bias. McCrary density tests check manipulation.",
                source_id="src-012-rdd-guide",
                score=0.85,
                authors="Imbens and Lemieux (2008)",
                year="2008",
            ),
        ],
        search_summary="Found 3 results on IV and RDD.",
        concepts=[
            ConceptInfo(
                concept_id="concept-iv-001",
                name="instrumental variables",
                concept_type="METHOD",
                description="Exploits exogenous instruments for causal identification.",
            ),
            ConceptInfo(
                concept_id="concept-rdd-001",
                name="regression discontinuity",
                concept_type="METHOD",
                description="Exploits treatment assignment cutoffs for causal inference.",
            ),
        ],
        concept_map_summary="2 concepts explored: IV, RDD.",
        assumption_audits=[
            AssumptionAudit(
                method_name="IV",
                raw_output="Assumptions: exclusion restriction, relevance, monotonicity (LATE).",
            ),
            AssumptionAudit(
                method_name="RDD",
                raw_output="Assumptions: continuity at cutoff, no manipulation, "
                "bandwidth selection.",
            ),
        ],
        assumption_summary="Audited 2 methods: IV, RDD.",
    )


def _build_causal_forests_state() -> ResearchState:
    """Build a realistic state for causal forests synthesis testing."""
    return ResearchState(
        query="How do causal forests estimate heterogeneous treatment effects?",
        sub_tasks=[
            SubTask(
                description="Find papers on causal forest methodology",
                search_queries=["causal forest", "heterogeneous treatment effects"],
                concepts_to_explore=["causal forest", "heterogeneous treatment effects"],
                methods_to_audit=[],
            ),
            SubTask(
                description="Understand estimation and inference procedures",
                search_queries=["generalized random forest estimation"],
                concepts_to_explore=["honest estimation"],
                methods_to_audit=[],
            ),
        ],
        planning_rationale="Decomposed into methodology and estimation subtasks.",
        search_results=[
            SearchResult(
                title="Estimation and Inference of Heterogeneous Treatment Effects "
                "using Random Forests",
                content="Causal forests adapt random forests for treatment effect "
                "heterogeneity. Honesty — using separate samples for tree construction "
                "and estimation — enables valid inference.",
                source_id="src-020-cf",
                score=0.95,
                authors="Wager and Athey (2018)",
                year="2018",
            ),
            SearchResult(
                title="Generalized Random Forests",
                content="GRF extends causal forests to a general framework for "
                "heterogeneous estimation. Local moment conditions are solved "
                "using forest-based adaptive weighting.",
                source_id="src-021-grf",
                score=0.90,
                authors="Athey, Tibshirani, and Wager (2019)",
                year="2019",
            ),
        ],
        search_summary="Found 2 results on causal forests.",
        concepts=[
            ConceptInfo(
                concept_id="concept-cf-001",
                name="causal forest",
                concept_type="METHOD",
                description="Random forest adapted for heterogeneous treatment effect estimation.",
            ),
            ConceptInfo(
                concept_id="concept-hte-001",
                name="heterogeneous treatment effects",
                concept_type="CONCEPT",
                description="Treatment effects that vary across subpopulations.",
            ),
        ],
        concept_map_summary="2 concepts explored: causal forest, HTE.",
        assumption_audits=[],
        assumption_summary="No methods to audit.",
    )


_JUDGE_CASES = [
    ("dml_assumptions", _build_dml_state),
    ("iv_vs_rdd", _build_iv_rdd_state),
    ("causal_forests", _build_causal_forests_state),
]


@requires_api_key
@pytest.mark.eval
@pytest.mark.parametrize("case_name,build_state", _JUDGE_CASES, ids=[c[0] for c in _JUDGE_CASES])
async def test_synthesis_with_judge(case_name: str, build_state) -> None:
    """Full synthesis + LLM-as-judge grading on a golden case."""
    state = build_state()

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
        case_name=case_name,
        dimension="synthesis_judge",
        scores={
            "completeness": float(verdict.completeness),
            "grounding": float(verdict.grounding),
            "gap_honesty": float(verdict.gap_honesty),
            "coherence": float(verdict.coherence),
            "average": verdict.average_score,
        },
    )

    # Minimum thresholds (all dimensions >= 3)
    errors = []
    if verdict.completeness < 3:
        errors.append(f"Low completeness: {verdict.completeness}")
    if verdict.grounding < 3:
        errors.append(f"Low grounding: {verdict.grounding}")
    if verdict.gap_honesty < 3:
        errors.append(f"Low gap_honesty: {verdict.gap_honesty}")
    if verdict.coherence < 3:
        errors.append(f"Low coherence: {verdict.coherence}")

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
    state = _build_dml_state()

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
