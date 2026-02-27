"""Tests for the shared parsing utility."""

from __future__ import annotations

import json

from research_agent.parsing import parse_json_first


class TestParseJsonFirst:
    """Tests for parse_json_first()."""

    def test_json_path_succeeds(self) -> None:
        """Uses json_parser when it succeeds."""
        result = parse_json_first(
            '{"x": 1}',
            json_parser=lambda s: json.loads(s)["x"],
            markdown_parser=lambda s: 99,
            context="test",
        )
        assert result == 1

    def test_falls_back_to_markdown(self) -> None:
        """Falls back to markdown_parser on JSONDecodeError."""
        result = parse_json_first(
            "not json",
            json_parser=lambda s: json.loads(s),
            markdown_parser=lambda s: "fallback",
            context="test",
        )
        assert result == "fallback"

    def test_falls_back_on_key_error(self) -> None:
        """Falls back on KeyError from json_parser."""

        def bad_parser(s: str) -> str:
            raise KeyError("missing")

        result = parse_json_first(
            '{"x": 1}',
            json_parser=bad_parser,
            markdown_parser=lambda s: "fallback",
            context="test",
        )
        assert result == "fallback"

    def test_falls_back_on_type_error(self) -> None:
        """Falls back on TypeError from json_parser."""

        def bad_parser(s: str) -> str:
            raise TypeError("bad type")

        result = parse_json_first(
            '{"x": 1}',
            json_parser=bad_parser,
            markdown_parser=lambda s: "fallback",
        )
        assert result == "fallback"

    def test_default_context_label(self) -> None:
        """Uses 'response' as default context label in log message."""
        # Just verifying no crash when context is omitted
        result = parse_json_first(
            "not json",
            json_parser=lambda s: json.loads(s),
            markdown_parser=lambda s: "ok",
        )
        assert result == "ok"
