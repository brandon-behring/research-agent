"""Eval fixtures -- skip if no API key, shared golden case loading."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

EVALS_DIR = Path(__file__).parent
CASES_DIR = EVALS_DIR / "cases"

# Skip all evals if no API key
requires_api_key = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set -- skipping eval tests",
)


def load_golden_case(name: str) -> dict[str, Any]:
    """Load a golden test case from evals/cases/.

    Args:
        name: Case filename without extension (e.g., 'dml_assumptions').

    Returns:
        Parsed JSON dict with query, planner expectations, synthesis expectations.

    Raises:
        FileNotFoundError: If the case file doesn't exist.
    """
    path = CASES_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Golden case not found: {path}")
    return json.loads(path.read_text())


@pytest.fixture
def dml_case() -> dict[str, Any]:
    """Golden case: DML assumptions."""
    return load_golden_case("dml_assumptions")


@pytest.fixture
def iv_case() -> dict[str, Any]:
    """Golden case: IV vs RDD comparison."""
    return load_golden_case("iv_vs_rdd")


@pytest.fixture
def causal_forests_case() -> dict[str, Any]:
    """Golden case: Causal forests."""
    return load_golden_case("causal_forests")
