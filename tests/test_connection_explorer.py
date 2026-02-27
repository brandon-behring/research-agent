"""Tests for the connection_explorer node.

Tests parsing, deduplication, timeout handling, and graceful degradation
for the explain_connection concept path tracing.
"""

from __future__ import annotations

import json

import pytest

from research_agent.config import AgentConfig
from research_agent.mcp_client import ResearchKBClient
from research_agent.nodes.connection_explorer import (
    _parse_connection_response,
    connection_explorer,
)
from research_agent.state import ResearchState, SubTask


class TestParseConnectionResponse:
    """Tests for _parse_connection_response."""

    def test_parses_valid_json(self) -> None:
        """Parses well-formed explain_connection response."""
        raw = json.dumps(
            {
                "concept_a": "DML",
                "concept_b": "cross-fitting",
                "path_length": 1,
                "path_explanation": "DML uses cross-fitting.",
                "path": [
                    {
                        "concept_name": "DML",
                        "concept_type": "METHOD",
                        "evidence": [{"text": "...", "source": "..."}],
                    },
                ],
            }
        )
        result = _parse_connection_response(raw)
        assert result["concept_a"] == "DML"
        assert result["concept_b"] == "cross-fitting"
        assert result["path_length"] == 1
        assert len(result["path"]) == 1

    def test_handles_invalid_json(self) -> None:
        """Returns empty dict on invalid JSON."""
        result = _parse_connection_response("not json {{{")
        assert result == {}

    def test_handles_empty_string(self) -> None:
        """Returns empty dict on empty input."""
        result = _parse_connection_response("")
        assert result == {}

    def test_handles_partial_json(self) -> None:
        """Returns dict with defaults for missing fields."""
        raw = json.dumps({"concept_a": "A", "concept_b": "B"})
        result = _parse_connection_response(raw)
        assert result["concept_a"] == "A"
        assert result["path_length"] == 0
        assert result["path"] == []


class TestConnectionExplorer:
    """Tests for the connection_explorer node."""

    @pytest.mark.asyncio
    async def test_explores_planned_connections(
        self, test_config: AgentConfig, mock_mcp: ResearchKBClient
    ) -> None:
        """Calls explain_connection for each pair in sub-tasks."""
        state = ResearchState(
            query="test",
            sub_tasks=[
                SubTask(
                    description="t",
                    connections_to_explain=[["DML", "cross-fitting"]],
                ),
            ],
        )
        result = await connection_explorer(state, test_config, mock_mcp)

        assert "connection_explanations" in result
        assert len(result["connection_explanations"]) == 1
        assert mock_mcp.explain_connection.call_count == 1

    @pytest.mark.asyncio
    async def test_deduplicates_pairs(
        self, test_config: AgentConfig, mock_mcp: ResearchKBClient
    ) -> None:
        """Deduplicates concept pairs across sub-tasks."""
        state = ResearchState(
            query="test",
            sub_tasks=[
                SubTask(
                    description="t1",
                    connections_to_explain=[["DML", "cross-fitting"]],
                ),
                SubTask(
                    description="t2",
                    connections_to_explain=[["cross-fitting", "DML"]],  # same pair reversed
                ),
            ],
        )
        result = await connection_explorer(state, test_config, mock_mcp)

        # Should deduplicate to 1 call
        assert mock_mcp.explain_connection.call_count == 1
        assert len(result["connection_explanations"]) == 1

    @pytest.mark.asyncio
    async def test_handles_no_connections(
        self, test_config: AgentConfig, mock_mcp: ResearchKBClient
    ) -> None:
        """Returns empty when no connections_to_explain."""
        state = ResearchState(
            query="test",
            sub_tasks=[SubTask(description="t")],
        )
        result = await connection_explorer(state, test_config, mock_mcp)

        assert result["connection_explanations"] == []
        assert mock_mcp.explain_connection.call_count == 0

    @pytest.mark.asyncio
    async def test_graceful_degradation_on_failure(
        self, test_config: AgentConfig, mock_mcp: ResearchKBClient
    ) -> None:
        """Failed explain_connection calls are skipped, not fatal."""
        mock_mcp.explain_connection.side_effect = RuntimeError("Not connected")

        state = ResearchState(
            query="test",
            sub_tasks=[
                SubTask(
                    description="t",
                    connections_to_explain=[["A", "B"], ["C", "D"]],
                ),
            ],
        )
        result = await connection_explorer(state, test_config, mock_mcp)

        # Both failed → empty results, but no exception
        assert result["connection_explanations"] == []

    @pytest.mark.asyncio
    async def test_skips_malformed_pairs(
        self, test_config: AgentConfig, mock_mcp: ResearchKBClient
    ) -> None:
        """Pairs with fewer than 2 elements are ignored."""
        state = ResearchState(
            query="test",
            sub_tasks=[
                SubTask(
                    description="t",
                    connections_to_explain=[["only_one"], ["A", "B"]],
                ),
            ],
        )
        await connection_explorer(state, test_config, mock_mcp)

        # Only the valid pair should be explored
        assert mock_mcp.explain_connection.call_count == 1

    @pytest.mark.asyncio
    async def test_uses_graph_only_mode(
        self, test_config: AgentConfig, mock_mcp: ResearchKBClient
    ) -> None:
        """Calls explain_connection with use_llm=False (graph-only)."""
        state = ResearchState(
            query="test",
            sub_tasks=[
                SubTask(
                    description="t",
                    connections_to_explain=[["DML", "cross-fitting"]],
                ),
            ],
        )
        await connection_explorer(state, test_config, mock_mcp)

        call_kwargs = mock_mcp.explain_connection.call_args
        assert call_kwargs.kwargs.get("use_llm") is False
        assert call_kwargs.kwargs.get("style") == "research"

    @pytest.mark.asyncio
    async def test_multiple_pairs_explored(
        self, test_config: AgentConfig, mock_mcp: ResearchKBClient
    ) -> None:
        """Multiple unique pairs are all explored."""
        state = ResearchState(
            query="test",
            sub_tasks=[
                SubTask(
                    description="t",
                    connections_to_explain=[
                        ["DML", "cross-fitting"],
                        ["IV", "LATE"],
                        ["DiD", "parallel trends"],
                    ],
                ),
            ],
        )
        result = await connection_explorer(state, test_config, mock_mcp)

        assert mock_mcp.explain_connection.call_count == 3
        assert len(result["connection_explanations"]) == 3
