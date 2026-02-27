"""Shared parsing utility for JSON-first with markdown fallback.

All MCP responses are requested as JSON (output_format='json').
This module provides the generic fallback pattern used by all node parsers.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def parse_json_first(
    raw: str,
    json_parser: Callable[[str], T],
    markdown_parser: Callable[[str], T],
    *,
    context: str = "",
) -> T:
    """Parse MCP response with JSON-first strategy and markdown fallback.

    Args:
        raw: Raw response string from research-kb.
        json_parser: Function that parses JSON string -> T. May raise
            json.JSONDecodeError, KeyError, or TypeError.
        markdown_parser: Fallback function that parses markdown string -> T.
        context: Label for log messages (e.g., "search results", "citation network").

    Returns:
        Parsed result from whichever parser succeeds.
    """
    try:
        return json_parser(raw)
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning(
            "JSON parse failed for %s, falling back to markdown: %s",
            context or "response",
            exc,
        )
        return markdown_parser(raw)
