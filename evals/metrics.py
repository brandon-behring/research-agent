"""Eval metrics -- concept recall, method recall, grounding score, CSV export."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


def _any_match(candidates: list[str], actual_lower: list[str], actual_joined: str) -> bool:
    """Check if any candidate matches any actual value.

    Uses bidirectional substring matching to handle abbreviations:
    - "DML" matches "double machine learning" (substring of actual)
    - "double machine learning" matches "DML" (actual is substring of candidate)

    Args:
        candidates: Lowercased candidate strings (name + aliases).
        actual_lower: Lowercased actual values.
        actual_joined: Space-joined actual_lower for substring search.

    Returns:
        True if any candidate matches any actual value.
    """
    actual_set = set(actual_lower)
    for c in candidates:
        # Exact match
        if c in actual_set:
            return True
        # Candidate is substring of the joined actual string
        if c in actual_joined:
            return True
        # Bidirectional substring: actual is substring of candidate
        if any(c in a or a in c for a in actual_lower):
            return True
    return False


def _extract_candidates(entry: str | dict[str, Any]) -> list[str]:
    """Extract lowercased candidate strings from a plain string or alias dict.

    Args:
        entry: Either a plain string or {"name": "...", "aliases": [...]}.

    Returns:
        List of lowercased candidate strings.
    """
    if isinstance(entry, str):
        return [entry.lower()]
    name = entry.get("name", "")
    aliases = entry.get("aliases", [])
    return [name.lower()] + [a.lower() for a in aliases]


def concept_recall(expected: list[str | dict[str, Any]], actual: list[str]) -> float:
    """Fraction of expected concepts found in actual (alias-aware).

    Supports both simple strings and dicts with aliases::

        expected = ["DML"]                                          # simple
        expected = [{"name": "DML", "aliases": ["double ML"]}]     # with aliases

    Args:
        expected: Concepts that should appear (strings or alias dicts).
        actual: Concepts that were found.

    Returns:
        Recall score in [0.0, 1.0]. Returns 1.0 if expected is empty.

    Example:
        >>> concept_recall(
        ...     [{"name": "double machine learning", "aliases": ["DML"]}],
        ...     ["DML", "overlap"],
        ... )
        1.0
    """
    if not expected:
        return 1.0
    actual_lower = [c.lower() for c in actual]
    actual_joined = " ".join(actual_lower)
    found = sum(
        1
        for entry in expected
        if _any_match(_extract_candidates(entry), actual_lower, actual_joined)
    )
    return found / len(expected)


def method_recall(expected: list[str | dict[str, Any]], actual: list[str]) -> float:
    """Fraction of expected methods found in actual (alias-aware, substring-aware).

    Same alias API as concept_recall for consistency.

    Args:
        expected: Methods that should appear (strings or alias dicts).
        actual: Methods that were found.

    Returns:
        Recall score in [0.0, 1.0]. Returns 1.0 if expected is empty.
    """
    if not expected:
        return 1.0
    actual_lower = [m.lower() for m in actual]
    actual_joined = " ".join(actual_lower)
    found = sum(
        1
        for entry in expected
        if _any_match(_extract_candidates(entry), actual_lower, actual_joined)
    )
    return found / len(expected)


def grounding_score(report: str, evidence_terms: list[str]) -> float:
    """Fraction of evidence terms mentioned in the report (case-insensitive).

    A rough proxy for how well-grounded the report is in the evidence.

    Args:
        report: The generated report text.
        evidence_terms: Key terms from the evidence (author names, methods, concepts).

    Returns:
        Score in [0.0, 1.0]. Returns 1.0 if evidence_terms is empty.
    """
    if not evidence_terms:
        return 1.0
    report_lower = report.lower()
    found = sum(1 for t in evidence_terms if t.lower() in report_lower)
    return found / len(evidence_terms)


@dataclass
class EvalResult:
    """Result from a single evaluation run.

    Attributes:
        case_name: Name of the golden case.
        dimension: What was evaluated (planner, search, synthesis).
        scores: Dict of metric_name -> score.
        passed: Whether all assertions passed.
        errors: Any assertion or runtime errors.
        timestamp: When the eval ran.
    """

    case_name: str
    dimension: str
    scores: dict[str, float] = field(default_factory=dict)
    passed: bool = True
    errors: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


def write_eval_results(results: list[EvalResult], path: str) -> None:
    """Write eval results to CSV for tracking over time.

    Args:
        path: Output CSV file path.
        results: List of EvalResult to write.
    """
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "case_name", "dimension", "passed", "scores", "errors"])
        for r in results:
            writer.writerow(
                [
                    r.timestamp,
                    r.case_name,
                    r.dimension,
                    r.passed,
                    str(r.scores),
                    "; ".join(r.errors),
                ]
            )


def print_eval_summary(results: list[EvalResult]) -> str:
    """Format eval results as a readable summary.

    Args:
        results: List of EvalResult.

    Returns:
        Formatted summary string.
    """
    buf = io.StringIO()
    buf.write("\n=== Eval Summary ===\n")
    passed = sum(1 for r in results if r.passed)
    buf.write(f"Passed: {passed}/{len(results)}\n\n")

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        buf.write(f"[{status}] {r.case_name} / {r.dimension}\n")
        for metric, score in r.scores.items():
            buf.write(f"  {metric}: {score:.3f}\n")
        if r.errors:
            for err in r.errors:
                buf.write(f"  ERROR: {err}\n")
        buf.write("\n")

    return buf.getvalue()
