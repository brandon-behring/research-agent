"""Unit tests for eval judges -- no API key required.

Tests _extract_json parsing, JudgeVerdict model, and grade_synthesis structure.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from evals.judges import JudgeVerdict, _extract_json


class TestExtractJson:
    """Tests for _extract_json helper."""

    def test_plain_json(self) -> None:
        """Plain JSON without fences parses correctly."""
        raw = '{"completeness": 4, "grounding": 3}'
        result = _extract_json(raw)
        assert result["completeness"] == 4
        assert result["grounding"] == 3

    def test_json_with_markdown_fences(self) -> None:
        """JSON wrapped in ```json fences parses correctly."""
        raw = '```json\n{"completeness": 5, "grounding": 4}\n```'
        result = _extract_json(raw)
        assert result["completeness"] == 5

    def test_json_with_bare_fences(self) -> None:
        """JSON wrapped in bare ``` fences parses correctly."""
        raw = '```\n{"key": "value"}\n```'
        result = _extract_json(raw)
        assert result["key"] == "value"

    def test_malformed_json_raises(self) -> None:
        """Malformed JSON raises json.JSONDecodeError."""
        with pytest.raises(json.JSONDecodeError):
            _extract_json("not json at all")

    def test_json_with_leading_whitespace(self) -> None:
        """JSON with leading/trailing whitespace parses correctly."""
        raw = '  \n  {"key": 1}  \n  '
        result = _extract_json(raw)
        assert result["key"] == 1


class TestJudgeVerdict:
    """Tests for JudgeVerdict model."""

    def test_average_score_calculation(self) -> None:
        """average_score computes mean of 4 dimensions."""
        v = JudgeVerdict(
            completeness=4,
            completeness_reason="Good",
            grounding=3,
            grounding_reason="Decent",
            gap_honesty=5,
            gap_honesty_reason="Excellent",
            coherence=4,
            coherence_reason="Clear",
            overall_assessment="Solid report",
        )
        assert v.average_score == 4.0

    def test_average_score_non_integer(self) -> None:
        """average_score returns float even for non-round averages."""
        v = JudgeVerdict(
            completeness=3,
            completeness_reason="OK",
            grounding=4,
            grounding_reason="Good",
            gap_honesty=3,
            gap_honesty_reason="OK",
            coherence=4,
            coherence_reason="Good",
            overall_assessment="OK report",
        )
        assert v.average_score == 3.5

    def test_scores_validated_range(self) -> None:
        """Scores outside 1-5 range are rejected."""
        with pytest.raises(ValidationError):
            JudgeVerdict(
                completeness=6,
                completeness_reason="Too high",
                grounding=3,
                grounding_reason="OK",
                gap_honesty=3,
                gap_honesty_reason="OK",
                coherence=3,
                coherence_reason="OK",
                overall_assessment="Invalid",
            )

    def test_scores_below_range_rejected(self) -> None:
        """Scores below 1 are rejected."""
        with pytest.raises(ValidationError):
            JudgeVerdict(
                completeness=0,
                completeness_reason="Too low",
                grounding=3,
                grounding_reason="OK",
                gap_honesty=3,
                gap_honesty_reason="OK",
                coherence=3,
                coherence_reason="OK",
                overall_assessment="Invalid",
            )

    def test_from_dict(self) -> None:
        """JudgeVerdict can be constructed from dict (simulating LLM parse)."""
        data = {
            "completeness": 4,
            "completeness_reason": "Good coverage",
            "grounding": 5,
            "grounding_reason": "Well-sourced",
            "gap_honesty": 3,
            "gap_honesty_reason": "Some gaps missed",
            "coherence": 4,
            "coherence_reason": "Well-organized",
            "overall_assessment": "Strong report",
        }
        v = JudgeVerdict(**data)
        assert v.completeness == 4
        assert v.average_score == 4.0
