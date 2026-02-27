"""Shared test fixtures for research-agent tests.

Provides mocked MCP client and standard test configurations.
All MCP responses are realistic JSON matching research-kb's output_format='json' schema.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from research_agent.config import AgentConfig, MCPConfig, ModelConfig
from research_agent.mcp_client import ResearchKBClient
from research_agent.state import ResearchState, SubTask


@pytest.fixture
def test_config() -> AgentConfig:
    """Standard test configuration with fast models."""
    return AgentConfig(
        models=ModelConfig(
            planning="claude-haiku-4-5-20251001",
            synthesis="claude-haiku-4-5-20251001",  # Use haiku for speed in tests
        ),
        mcp=MCPConfig(transport="stdio", research_kb_path="/fake/path"),
        max_search_results=5,
        max_concepts=3,
        max_citations=5,
    )


@pytest.fixture
def mock_mcp() -> ResearchKBClient:
    """Mocked MCP client with realistic research-kb JSON responses."""
    client = AsyncMock(spec=ResearchKBClient)

    # --- Search response ---
    client.search.return_value = json.dumps(
        {
            "query": "double machine learning",
            "expanded_query": "double machine learning DML debiased",
            "execution_time_ms": 245,
            "result_count": 3,
            "results": [
                {
                    "rank": 1,
                    "title": (
                        "Double/Debiased Machine Learning for Treatment and Structural Parameters"
                    ),
                    "authors": "Chernozhukov et al. (2018) [Paper]",
                    "year": 2018,
                    "source_id": "src-001-dml",
                    "chunk_id": "chk-001",
                    "content": (
                        "Double machine learning (DML) provides a general"
                        " framework for estimating\n"
                        "treatment effects using machine learning methods"
                        " while maintaining valid\n"
                        "statistical inference. The key insight is"
                        " cross-fitting..."
                    ),
                    "scores": {
                        "combined": 0.892,
                        "fts": 0.756,
                        "vector": 0.891,
                        "graph": 0.823,
                        "citation": 0.0,
                    },
                },
                {
                    "rank": 2,
                    "title": "Debiased Machine Learning of Conditional Average Treatment Effects",
                    "authors": "Semenova and Chernozhukov (2021) [Paper]",
                    "year": 2021,
                    "source_id": "src-002-cate",
                    "chunk_id": "chk-002",
                    "content": (
                        "We propose debiased machine learning estimators for conditional average\n"
                        "treatment effects (CATEs)..."
                    ),
                    "scores": {
                        "combined": 0.834,
                        "fts": 0.612,
                        "vector": 0.845,
                        "graph": 0.790,
                        "citation": 0.0,
                    },
                },
                {
                    "rank": 3,
                    "title": "Cross-fitting and Double Machine Learning",
                    "authors": "Newey and Robins (2018) [Paper]",
                    "year": 2018,
                    "source_id": "src-003-crossfit",
                    "chunk_id": "chk-003",
                    "content": (
                        "We analyze the properties of cross-fitting in the context of\n"
                        "semiparametric estimation..."
                    ),
                    "scores": {
                        "combined": 0.781,
                        "fts": 0.580,
                        "vector": 0.802,
                        "graph": 0.710,
                        "citation": 0.0,
                    },
                },
            ],
        }
    )

    # --- Fast search response ---
    client.fast_search.return_value = json.dumps(
        {
            "query": "DML assumptions",
            "execution_time_ms": 42,
            "result_count": 2,
            "results": [
                {
                    "rank": 1,
                    "title": (
                        "Double/Debiased Machine Learning for Treatment and Structural Parameters"
                    ),
                    "authors": "Chernozhukov et al. (2018)",
                    "year": 2018,
                    "source_id": "src-001-dml",
                    "chunk_id": "chk-010",
                    "content": "The key assumptions are unconfoundedness and overlap...",
                    "scores": {"combined": 0.845},
                },
                {
                    "rank": 2,
                    "title": "Assumption Lean Inference",
                    "authors": "Vansteelandt and Dukes (2022)",
                    "year": 2022,
                    "source_id": "src-004-lean",
                    "chunk_id": "chk-020",
                    "content": "Discusses relaxing parametric assumptions in causal inference...",
                    "scores": {"combined": 0.712},
                },
            ],
        }
    )

    # --- Concept responses ---
    client.get_concept.return_value = json.dumps(
        {
            "concept_id": "concept-dml-001",
            "name": "Double Machine Learning",
            "concept_type": "METHOD",
            "definition": (
                "A framework for estimating treatment effects that uses cross-fitting\n"
                "to avoid overfitting bias in nuisance parameter estimation."
            ),
            "relationships": [
                {"type": "REQUIRES", "target_id": "concept-unconf-001"},
                {"type": "REQUIRES", "target_id": "concept-overlap-001"},
                {"type": "USES", "target_id": "concept-crossfit-001"},
                {"type": "ADDRESSES", "target_id": "concept-regbias-001"},
                {"type": "RELATED_TO", "target_id": "concept-iv-001"},
            ],
        }
    )

    client.graph_neighborhood.return_value = json.dumps(
        {
            "center": {
                "id": "concept-dml-001",
                "name": "double machine learning",
                "type": "METHOD",
            },
            "nodes": [
                {"name": "Unconfoundedness", "type": "ASSUMPTION"},
                {"name": "Overlap condition", "type": "ASSUMPTION"},
                {"name": "Cross-fitting", "type": "METHOD"},
                {"name": "Regularization bias", "type": "PROBLEM"},
                {"name": "Instrumental variables", "type": "METHOD"},
                {"name": "Neyman orthogonality", "type": "THEOREM"},
                {"name": "CATE estimation", "type": "METHOD"},
                {"name": "Propensity score", "type": "METHOD"},
            ],
            "edges": [
                {"source": "concept-dml-001", "target": "concept-unconf-001", "type": "REQUIRES"},
                {"source": "concept-dml-001", "target": "concept-overlap-001", "type": "REQUIRES"},
                {"source": "concept-dml-001", "target": "concept-crossfit-001", "type": "USES"},
                {"source": "concept-dml-001", "target": "concept-regbias-001", "type": "ADDRESSES"},
                {"source": "concept-dml-001", "target": "concept-iv-001", "type": "RELATED_TO"},
                {"source": "concept-dml-001", "target": "concept-neyman-001", "type": "RELATED_TO"},
                {"source": "concept-dml-001", "target": "concept-cate-001", "type": "RELATED_TO"},
                {"source": "concept-dml-001", "target": "concept-ps-001", "type": "RELATED_TO"},
                {"source": "concept-crossfit-001", "target": "concept-dml-001", "type": "USES"},
                {"source": "concept-cate-001", "target": "concept-dml-001", "type": "USES"},
                {"source": "concept-neyman-001", "target": "concept-dml-001", "type": "REQUIRES"},
                {"source": "concept-ps-001", "target": "concept-dml-001", "type": "RELATED_TO"},
            ],
            "relationship_type_counts": {
                "REQUIRES": 2,
                "USES": 2,
                "ADDRESSES": 1,
                "RELATED_TO": 3,
            },
        }
    )

    # --- Citation responses ---
    client.citation_network.return_value = json.dumps(
        {
            "source_id": "src-001-dml",
            "title": "Double/Debiased Machine Learning",
            "citing": [
                {
                    "title": "Debiased Machine Learning of CATEs",
                    "year": 2021,
                    "source_id": "src-002-cate",
                    "authors": "Semenova and Chernozhukov",
                },
                {
                    "title": "Automatic Debiased Machine Learning",
                    "year": 2022,
                    "source_id": "src-005-auto",
                    "authors": "Chernozhukov et al.",
                },
                {
                    "title": "DML for Difference-in-Differences",
                    "year": 2023,
                    "source_id": "src-006-dml-did",
                    "authors": "Chang (2023)",
                },
            ],
            "cited_by": [
                {
                    "title": "Estimation and Inference of Heterogeneous Treatment Effects",
                    "year": 2017,
                    "source_id": "src-007-hte",
                    "authors": "Athey and Imbens",
                },
                {
                    "title": "High-Dimensional Methods and Inference",
                    "year": 2014,
                    "source_id": "src-008-hdm",
                    "authors": "Belloni, Chernozhukov, Hansen",
                },
            ],
        }
    )

    client.biblio_coupling.return_value = json.dumps(
        {
            "source_id": "src-001-dml",
            "title": "Double/Debiased Machine Learning",
            "similar": [
                {
                    "title": "Debiased Machine Learning of CATEs",
                    "year": 2021,
                    "source_id": "src-002-cate",
                    "authors": "Semenova and Chernozhukov",
                    "coupling_strength": 0.452,
                    "shared_references": 8,
                },
                {
                    "title": "Automatic Debiased Machine Learning",
                    "year": 2022,
                    "source_id": "src-005-auto",
                    "authors": "Chernozhukov et al.",
                    "coupling_strength": 0.387,
                    "shared_references": 6,
                },
            ],
        }
    )

    # --- Assumption audit response ---
    client.audit_assumptions.return_value = json.dumps(
        {
            "method": "Double Machine Learning (DML)",
            "aliases": ["DML", "debiased ML", "double ML"],
            "method_id": "concept-dml-001",
            "definition": "Framework for treatment effect estimation using ML nuisance estimation.",
            "source": "graph",
            "assumptions": [
                {
                    "name": "Unconfoundedness",
                    "importance": "CRITICAL",
                    "formal_statement": "Y(t) \u22a5 T | X",
                    "plain_english": "All confounders are observed and included in X.",
                    "violation_consequence": "Treatment effect estimates are biased.",
                    "verification_approaches": ["Sensitivity analysis", "placebo tests"],
                    "citation": "Chernozhukov et al. (2018), Section 2",
                    "concept_id": "concept-unconf-001",
                    "relationship": "REQUIRES",
                },
                {
                    "name": "Overlap (Positivity)",
                    "importance": "CRITICAL",
                    "formal_statement": "0 < P(T=1|X) < 1",
                    "plain_english": "Every unit has positive probability of treatment.",
                    "violation_consequence": "Extreme propensity scores, unstable estimates.",
                    "verification_approaches": ["Check propensity score distribution"],
                    "citation": "Chernozhukov et al. (2018), Assumption 3.1",
                    "concept_id": "concept-overlap-001",
                    "relationship": "REQUIRES",
                },
                {
                    "name": "Neyman Orthogonality",
                    "importance": "STANDARD",
                    "formal_statement": "\u2202\u03b8 E[\u03c8(W; \u03b8\u2080, \u03b7\u2080)] = 0",
                    "plain_english": "Score function is insensitive to nuisance estimation error.",
                    "violation_consequence": "\u221an convergence may not hold.",
                    "verification_approaches": ["Verify moment condition structure"],
                    "citation": "Chernozhukov et al. (2018), Definition 2.1",
                    "concept_id": "concept-neyman-001",
                    "relationship": "REQUIRES",
                },
            ],
            "code_docstring_snippet": (
                "Assumptions:\n"
                "    [CRITICAL] - unconfoundedness: All confounders"
                " observed and included in X\n"
                "    [CRITICAL] - overlap: Every unit has positive"
                " probability of treatment\n"
                "    - neyman_orthogonality: Score function"
                " insensitive to nuisance estimation error"
            ),
        }
    )

    # --- Get source response (markdown) ---
    client.get_source.return_value = (
        "## Double/Debiased Machine Learning for Treatment and Structural Parameters\n\n"
        "**Authors:** Chernozhukov, Chetverikov, Demirer, Duflo, Hansen, Newey, Robins\n"
        "**Year:** 2018\n"
        "**Type:** Paper\n"
        "**Source ID:** `src-001-dml`\n"
        "**DOI:** 10.1111/ectj.12097\n\n"
        "### Abstract\n"
        "We revisit the classic semiparametric problem of inference on a low-dimensional\n"
        "parameter in the presence of high-dimensional nuisance parameters.\n"
    )

    # --- Find similar concepts response (markdown) ---
    client.find_similar_concepts.return_value = (
        "## Similar Concepts to Double Machine Learning\n\n"
        "| Concept | Similarity | Type |\n"
        "|---------|-----------|------|\n"
        "| Debiased Machine Learning | 0.95 | METHOD |\n"
        "| Targeted Learning (TMLE) | 0.87 | METHOD |\n"
        "| Cross-fitting | 0.85 | METHOD |\n"
    )

    # --- Cross-domain concepts response (markdown) ---
    client.cross_domain_concepts.return_value = (
        "## Cross-Domain Bridges: causal_inference → time_series\n\n"
        "| Source Concept | Target Concept | Link Type | Similarity |\n"
        "|---------------|---------------|-----------|------------|\n"
        "| Instrumental Variables | Granger Causality | ANALOGOUS | 0.88 |\n"
        "| Treatment Effect | Impulse Response | ANALOGOUS | 0.86 |\n"
    )

    # --- List domains response (markdown) ---
    client.list_domains.return_value = (
        "## Available Domains\n\n"
        "| Domain | Sources | Concepts |\n"
        "|--------|---------|----------|\n"
        "| causal_inference | 312 | 145 |\n"
        "| time_series | 98 | 52 |\n"
        "| rag_llm | 45 | 28 |\n"
        "| statistical_methodology | 40 | 21 |\n"
    )

    # --- Stats response (markdown) ---
    client.stats.return_value = (
        "## Knowledge Base Statistics\n\n"
        "- **Sources:** 495\n"
        "- **Chunks:** 226,432\n"
        "- **Concepts:** 246\n"
        "- **Relationships:** 1,847\n"
        "- **Domains:** 4\n"
    )

    # --- Explain connection response ---
    client.explain_connection.return_value = json.dumps(
        {
            "concept_a": "double machine learning",
            "concept_b": "cross-fitting",
            "path_length": 1,
            "path_explanation": "DML directly uses cross-fitting for sample splitting.",
            "path": [
                {
                    "concept_name": "double machine learning",
                    "concept_type": "METHOD",
                    "evidence": [
                        {
                            "text": "DML uses cross-fitting to partition samples.",
                            "source": "Chernozhukov et al. (2018)",
                        },
                    ],
                },
                {
                    "concept_name": "cross-fitting",
                    "concept_type": "METHOD",
                    "evidence": [
                        {
                            "text": "Cross-fitting prevents overfitting in nuisance estimation.",
                            "source": "Newey and Robins (2018)",
                        },
                    ],
                },
            ],
        }
    )

    return client


@pytest.fixture
def sample_state() -> ResearchState:
    """Sample state for testing individual nodes."""
    return ResearchState(
        query="What are the assumptions of double machine learning?",
        sub_tasks=[
            SubTask(
                description="Find foundational papers on DML",
                search_queries=["double machine learning Chernozhukov", "DML cross-fitting"],
                concepts_to_explore=["double machine learning", "cross-fitting"],
                methods_to_audit=["DML"],
            ),
            SubTask(
                description="Understand identification assumptions",
                search_queries=["unconfoundedness assumption causal inference"],
                concepts_to_explore=["unconfoundedness"],
                methods_to_audit=[],
            ),
        ],
    )
