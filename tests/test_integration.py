"""End-to-end integration tests with live MCP server and Anthropic API.

These tests require:
    - RESEARCH_KB_PATH pointing to a running research-kb repo
    - ANTHROPIC_API_KEY set in the environment

All models are set to Haiku for cost (~$0.02/run). These tests validate
wiring — that the full pipeline produces a report — not output quality
(that's what evals/ is for).

Run:
    RESEARCH_KB_PATH=~/Claude/research-kb ANTHROPIC_API_KEY=sk-... \\
        pytest tests/test_integration.py -m integration -v
"""

from __future__ import annotations

import os

import pytest

from research_agent.config import AgentConfig, MCPConfig, ModelConfig
from research_agent.graph import StreamEvent, run_research, stream_research

# Skip entire module if required env vars are not set
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("RESEARCH_KB_PATH"),
        reason="RESEARCH_KB_PATH not set (live MCP server required)",
    ),
    pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY not set",
    ),
]


@pytest.fixture
def integration_config() -> AgentConfig:
    """Config for integration tests — all Haiku, small limits."""
    return AgentConfig(
        models=ModelConfig(
            planning="claude-haiku-4-5-20251001",
            synthesis="claude-haiku-4-5-20251001",
        ),
        mcp=MCPConfig(
            transport="stdio",
            research_kb_path=os.environ.get("RESEARCH_KB_PATH", ""),
        ),
        max_search_results=5,
        max_concepts=3,
        max_citations=3,
    )


class TestBlockingIntegration:
    """Integration tests using the blocking run_research() path."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(120)
    async def test_full_pipeline_produces_report(self, integration_config: AgentConfig) -> None:
        """Full pipeline produces a non-empty report with populated intermediate state."""
        result = await run_research(
            "What are the assumptions of double machine learning?",
            integration_config,
        )

        # Report should be non-empty
        assert result.get("report"), "Expected non-empty report"
        assert len(result["report"]) > 100, "Report suspiciously short"

        # Intermediate state should be populated
        assert len(result.get("sub_tasks", [])) > 0, "Expected sub-tasks from planner"
        assert len(result.get("search_results", [])) > 0, "Expected search results"

    @pytest.mark.asyncio
    @pytest.mark.timeout(120)
    async def test_sparse_topic_completes_gracefully(self, integration_config: AgentConfig) -> None:
        """Off-topic query completes without crashing (may produce thin report)."""
        result = await run_research(
            "What are the best pizza toppings for a research seminar?",
            integration_config,
        )

        # Should complete without error and produce some report
        assert "report" in result
        # Report may be thin but should exist
        assert isinstance(result["report"], str)


class TestStreamingIntegration:
    """Integration tests using the streaming stream_research() path."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(120)
    async def test_streaming_yields_events(self, integration_config: AgentConfig) -> None:
        """Streaming pipeline yields expected event sequence."""
        events: list[StreamEvent] = []
        async for event in stream_research(
            "What are the assumptions of double machine learning?",
            integration_config,
        ):
            events.append(event)

        # Should have node_end events for at least the 5 core nodes
        node_end_events = [e for e in events if e.event_type == "node_end"]
        assert len(node_end_events) >= 5, (
            f"Expected >= 5 node_end events, got {len(node_end_events)}"
        )

        # Should have exactly one report_chunk
        report_events = [e for e in events if e.event_type == "report_chunk"]
        assert len(report_events) == 1, f"Expected 1 report_chunk, got {len(report_events)}"
        assert len(report_events[0].data) > 100, "Report content suspiciously short"

        # Should end with complete event
        assert events[-1].event_type == "complete"
