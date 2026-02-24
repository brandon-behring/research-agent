"""Eval: Query planner decomposition quality.

Tests that the planner produces reasonable sub-task decompositions
for golden test cases. Checks concept recall, method recall, and
minimum task/query counts.

Run: ``pytest evals/test_query_planner_eval.py -m eval --timeout=120``
"""

from __future__ import annotations

from typing import Any

import pytest

from evals.conftest import load_golden_case, requires_api_key
from evals.metrics import EvalResult, concept_recall, method_recall
from research_agent.config import AgentConfig, ModelConfig
from research_agent.nodes.query_planner import query_planner
from research_agent.state import ResearchState

CASES = ["dml_assumptions", "iv_vs_rdd", "causal_forests"]


@requires_api_key
@pytest.mark.eval
@pytest.mark.parametrize("case_name", CASES)
async def test_planner_decomposition(case_name: str) -> None:
    """Planner produces sub-tasks meeting golden case expectations.

    Checks:
        - At least ``min_tasks`` sub-tasks created
        - Expected concepts appear (concept recall >= 0.5)
        - Expected methods appear (method recall >= 0.5)
        - At least ``min_search_queries`` total queries
    """
    case = load_golden_case(case_name)
    expectations: dict[str, Any] = case["planner"]

    config = AgentConfig(
        models=ModelConfig(planning="claude-haiku-4-5-20251001"),
        max_search_results=5,
        max_concepts=3,
        max_citations=3,
    )
    state = ResearchState(query=case["query"])

    result = await query_planner(state, config)
    sub_tasks = result["sub_tasks"]

    # Collect all concepts and methods from sub-tasks
    all_concepts = []
    all_methods = []
    total_queries = 0
    for task in sub_tasks:
        all_concepts.extend(task.concepts_to_explore)
        all_methods.extend(task.methods_to_audit)
        total_queries += len(task.search_queries)

    # Compute metrics
    c_recall = concept_recall(expectations["expected_concepts"], all_concepts)
    m_recall = method_recall(expectations["expected_methods"], all_methods)

    eval_result = EvalResult(
        case_name=case_name,
        dimension="planner",
        scores={
            "concept_recall": c_recall,
            "method_recall": m_recall,
            "num_tasks": float(len(sub_tasks)),
            "num_queries": float(total_queries),
        },
    )

    # Assertions
    errors = []
    if len(sub_tasks) < expectations["min_tasks"]:
        errors.append(f"Too few tasks: {len(sub_tasks)} < {expectations['min_tasks']}")
    if c_recall < 0.5:
        errors.append(f"Low concept recall: {c_recall:.2f}")
    if m_recall < 0.5:
        errors.append(f"Low method recall: {m_recall:.2f}")
    if total_queries < expectations["min_search_queries"]:
        errors.append(f"Too few queries: {total_queries} < {expectations['min_search_queries']}")

    if errors:
        eval_result.passed = False
        eval_result.errors = errors
        pytest.fail(f"Planner eval failed for {case_name}: {'; '.join(errors)}")
