"""Unit tests for eval metrics -- no API key required.

Tests concept_recall, method_recall, grounding_score, EvalResult, write_eval_results.
"""

from __future__ import annotations

import csv
import os
import tempfile

from evals.metrics import (
    EvalResult,
    concept_recall,
    grounding_score,
    method_recall,
    print_eval_summary,
    write_eval_results,
)


class TestConceptRecall:
    """Tests for concept_recall metric."""

    def test_all_found(self) -> None:
        """All expected concepts found → 1.0."""
        result = concept_recall(["DML", "IV"], ["DML", "IV", "RDD"])
        assert result == 1.0

    def test_none_found(self) -> None:
        """No expected concepts found → 0.0."""
        result = concept_recall(["DML", "IV"], ["RDD", "DiD"])
        assert result == 0.0

    def test_partial_found(self) -> None:
        """Some expected found → fraction."""
        result = concept_recall(["DML", "IV", "RDD"], ["DML"])
        assert abs(result - 1 / 3) < 0.01

    def test_empty_expected_returns_one(self) -> None:
        """Empty expected → 1.0 (vacuously true)."""
        assert concept_recall([], ["DML"]) == 1.0

    def test_empty_actual_returns_zero(self) -> None:
        """Empty actual → 0.0."""
        assert concept_recall(["DML"], []) == 0.0

    def test_alias_matching(self) -> None:
        """Alias dict enables matching alternate names."""
        expected = [{"name": "double machine learning", "aliases": ["DML"]}]
        actual = ["DML"]
        assert concept_recall(expected, actual) == 1.0

    def test_bidirectional_substring_matching(self) -> None:
        """Substring matching works both directions."""
        # "DML" is substring of "DML estimation"
        assert concept_recall(["DML"], ["DML estimation"]) == 1.0

    def test_case_insensitive(self) -> None:
        """Matching is case-insensitive."""
        assert concept_recall(["dml"], ["DML"]) == 1.0


class TestMethodRecall:
    """Tests for method_recall metric."""

    def test_all_methods_found(self) -> None:
        """All expected methods found → 1.0."""
        result = method_recall(["IV", "DML"], ["IV", "DML", "RDD"])
        assert result == 1.0

    def test_alias_matching(self) -> None:
        """Method aliases work like concept aliases."""
        expected = [{"name": "instrumental variables", "aliases": ["IV"]}]
        actual = ["IV"]
        assert method_recall(expected, actual) == 1.0

    def test_empty_expected_returns_one(self) -> None:
        """Empty expected → 1.0."""
        assert method_recall([], ["IV"]) == 1.0


class TestGroundingScore:
    """Tests for grounding_score metric."""

    def test_all_terms_found(self) -> None:
        """All evidence terms in report → 1.0."""
        report = "Chernozhukov introduced DML with cross-fitting."
        terms = ["Chernozhukov", "DML", "cross-fitting"]
        assert grounding_score(report, terms) == 1.0

    def test_no_terms_found(self) -> None:
        """No evidence terms in report → 0.0."""
        report = "This report discusses quantum physics."
        terms = ["Chernozhukov", "DML"]
        assert grounding_score(report, terms) == 0.0

    def test_partial_terms(self) -> None:
        """Some terms found → fraction."""
        report = "DML is a framework."
        terms = ["DML", "cross-fitting", "Chernozhukov"]
        assert abs(grounding_score(report, terms) - 1 / 3) < 0.01

    def test_empty_terms_returns_one(self) -> None:
        """Empty terms → 1.0 (vacuously true)."""
        assert grounding_score("anything", []) == 1.0

    def test_case_insensitive(self) -> None:
        """Matching is case-insensitive."""
        assert grounding_score("dml method", ["DML"]) == 1.0


class TestEvalResult:
    """Tests for EvalResult dataclass."""

    def test_default_values(self) -> None:
        """EvalResult has sensible defaults."""
        r = EvalResult(case_name="test", dimension="synthesis")
        assert r.passed is True
        assert r.errors == []
        assert r.scores == {}
        assert r.timestamp  # non-empty

    def test_with_scores_and_errors(self) -> None:
        """EvalResult stores scores and errors."""
        r = EvalResult(
            case_name="dml",
            dimension="synthesis_judge",
            scores={"completeness": 4.0, "grounding": 3.0},
            passed=False,
            errors=["Low grounding"],
        )
        assert r.scores["completeness"] == 4.0
        assert not r.passed


class TestWriteEvalResults:
    """Tests for write_eval_results CSV export."""

    def test_csv_round_trip(self) -> None:
        """Write results to CSV and verify structure."""
        results = [
            EvalResult(
                case_name="dml",
                dimension="synthesis",
                scores={"completeness": 4.0},
                passed=True,
                timestamp="2025-01-01T00:00:00",
            ),
            EvalResult(
                case_name="iv",
                dimension="planner",
                scores={"concept_recall": 0.8},
                passed=False,
                errors=["Low recall"],
                timestamp="2025-01-01T00:00:01",
            ),
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = f.name
        try:
            write_eval_results(results, path)
            with open(path) as f:
                reader = csv.reader(f)
                rows = list(reader)
            assert rows[0] == ["timestamp", "case_name", "dimension", "passed", "scores", "errors"]
            assert len(rows) == 3  # header + 2 results
            assert rows[1][1] == "dml"
            assert rows[2][3] == "False"
            assert "Low recall" in rows[2][5]
        finally:
            os.unlink(path)


class TestPrintEvalSummary:
    """Tests for print_eval_summary."""

    def test_summary_format(self) -> None:
        """Summary includes pass/fail counts and scores."""
        results = [
            EvalResult(
                case_name="dml",
                dimension="synthesis",
                scores={"completeness": 4.0},
                passed=True,
            ),
            EvalResult(
                case_name="iv",
                dimension="planner",
                scores={"recall": 0.5},
                passed=False,
                errors=["Low recall"],
            ),
        ]
        summary = print_eval_summary(results)
        assert "Passed: 1/2" in summary
        assert "[PASS] dml" in summary
        assert "[FAIL] iv" in summary
        assert "Low recall" in summary
