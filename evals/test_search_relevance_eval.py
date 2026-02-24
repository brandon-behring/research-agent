"""Eval: Search result parsing quality.

Tests the search parsing pipeline against captured MCP output.
Validates that parsers extract structured data correctly from
realistic markdown responses.

Run: ``pytest evals/test_search_relevance_eval.py -m eval``
"""

from __future__ import annotations

import pytest

from evals.metrics import EvalResult
from research_agent.nodes.literature_search import _parse_search_results

# Realistic captured MCP output for parsing validation
CAPTURED_SEARCH_OUTPUT = """## Search Results for: double machine learning assumptions
*Query expanded to: double machine learning DML debiased assumptions*

**Found 4 results** (in 312ms)

### 1. Double/Debiased Machine Learning for Treatment and Structural Parameters
Chernozhukov, Chetverikov, Demirer, Duflo, Hansen, Newey, Robins (2018) [Paper]
*Section 2.1: Framework and Assumptions*

> Double machine learning (DML) provides a general framework for estimating
> treatment effects using machine learning methods while maintaining valid
> statistical inference. The key assumptions are unconfoundedness and overlap.

*Score: 0.921 | FTS: 0.812 | Vector: 0.923 | Graph: 0.876*
*Source ID: `src-001-dml` | Chunk ID: `chk-001-s2`*

### 2. Debiased Machine Learning of Conditional Average Treatment Effects
Semenova and Chernozhukov (2021) [Paper]

> We propose debiased machine learning estimators for conditional average
> treatment effects (CATEs) under unconfoundedness.

*Score: 0.867 | FTS: 0.690 | Vector: 0.878 | Graph: 0.823*
*Source ID: `src-002-cate` | Chunk ID: `chk-002-s1`*

### 3. Cross-fitting and the Efficiency of Double Machine Learning
Newey and Robins (2018) [Paper]

> We analyze the properties of cross-fitting in semiparametric estimation
> and show it eliminates overfitting bias in nuisance estimation.

*Score: 0.812 | FTS: 0.610 | Vector: 0.834 | Graph: 0.743*
*Source ID: `src-003-crossfit` | Chunk ID: `chk-003-s1`*

### 4. Assumption-Lean Inference with Generalized DML
Vansteelandt and Dukes (2022) [Paper]

> Discusses relaxing parametric assumptions in the DML framework.

*Score: 0.756 | FTS: 0.540 | Vector: 0.789 | Graph: 0.680*
*Source ID: `src-004-lean` | Chunk ID: `chk-004-s1`*
"""


@pytest.mark.eval
class TestSearchParsing:
    """Test search result parsing against captured MCP output."""

    def test_parses_all_results(self) -> None:
        """Parses all 4 results from captured output."""
        results = _parse_search_results(CAPTURED_SEARCH_OUTPUT)
        assert len(results) == 4

    def test_extracts_titles(self) -> None:
        """Titles correctly extracted from ### headers."""
        results = _parse_search_results(CAPTURED_SEARCH_OUTPUT)
        assert results[0].title.startswith("Double/Debiased Machine Learning")
        assert "Conditional Average Treatment" in results[1].title

    def test_extracts_source_ids(self) -> None:
        """Source IDs correctly extracted from backtick-delimited text."""
        results = _parse_search_results(CAPTURED_SEARCH_OUTPUT)
        assert results[0].source_id == "src-001-dml"
        assert results[1].source_id == "src-002-cate"
        assert results[2].source_id == "src-003-crossfit"
        assert results[3].source_id == "src-004-lean"

    def test_scores_in_valid_range(self) -> None:
        """All scores between 0 and 1."""
        results = _parse_search_results(CAPTURED_SEARCH_OUTPUT)
        for r in results:
            assert 0.0 <= r.score <= 1.0, f"Invalid score {r.score} for {r.title}"

    def test_scores_descending_in_source(self) -> None:
        """Scores are in descending order as provided by research-kb."""
        results = _parse_search_results(CAPTURED_SEARCH_OUTPUT)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_extracts_content(self) -> None:
        """Content lines extracted from > blockquotes."""
        results = _parse_search_results(CAPTURED_SEARCH_OUTPUT)
        # Content should be from the > blockquote lines
        assert "double machine learning" in results[0].content.lower()
        assert len(results[0].content) > 20  # Non-trivial content

    def test_extracts_authors(self) -> None:
        """Author lines extracted."""
        results = _parse_search_results(CAPTURED_SEARCH_OUTPUT)
        assert "Chernozhukov" in results[0].authors

    def test_extracts_year(self) -> None:
        """Publication year extracted from author line."""
        results = _parse_search_results(CAPTURED_SEARCH_OUTPUT)
        assert results[0].year == "2018"
        assert results[1].year == "2021"

    def test_extracts_chunk_ids(self) -> None:
        """Chunk IDs extracted."""
        results = _parse_search_results(CAPTURED_SEARCH_OUTPUT)
        assert results[0].chunk_id == "chk-001-s2"

    def test_empty_input_returns_empty(self) -> None:
        """Empty markdown returns empty list."""
        assert _parse_search_results("") == []
        assert _parse_search_results("   ") == []

    def test_no_results_section(self) -> None:
        """Markdown without ### headers returns empty."""
        assert _parse_search_results("## Search Results\nNo results found.") == []

    def test_metrics_on_captured_output(self) -> None:
        """Compute eval metrics on parsing quality."""
        results = _parse_search_results(CAPTURED_SEARCH_OUTPUT)

        expected_sources = {"src-001-dml", "src-002-cate", "src-003-crossfit", "src-004-lean"}
        actual_sources = {r.source_id for r in results}
        precision = (
            len(expected_sources & actual_sources) / len(actual_sources) if actual_sources else 0
        )

        eval_result = EvalResult(
            case_name="captured_search",
            dimension="search_parsing",
            scores={
                "source_precision": precision,
                "num_results": float(len(results)),
                "avg_score": sum(r.score for r in results) / len(results) if results else 0,
            },
        )
        assert eval_result.scores["source_precision"] == 1.0
