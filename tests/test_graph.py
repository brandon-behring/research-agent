"""End-to-end tests for the research analysis graph.

Tests the full pipeline with mocked MCP client and mocked LLM responses.
Validates that:
1. All nodes execute in correct order
2. State flows through the graph correctly
3. Conditional routing works (assumption auditor skip)
4. Final report is generated
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from research_agent.config import AgentConfig, MCPConfig, ModelConfig
from research_agent.graph import _should_audit_assumptions, build_graph
from research_agent.state import ResearchState, SubTask


@pytest.fixture
def e2e_config() -> AgentConfig:
    """Config for end-to-end tests."""
    return AgentConfig(
        models=ModelConfig(
            planning="claude-haiku-4-5-20251001",
            synthesis="claude-haiku-4-5-20251001",
            analysis="claude-haiku-4-5-20251001",
        ),
        mcp=MCPConfig(transport="stdio", research_kb_path="/fake"),
        max_search_results=3,
        max_concepts=2,
        max_citations=3,
    )


class TestConditionalRouting:
    """Tests for graph conditional edges."""

    def test_routes_to_auditor_when_methods_present(self) -> None:
        """Routes to assumption_auditor if methods were identified."""
        state = ResearchState(
            query="test",
            sub_tasks=[SubTask(description="t", methods_to_audit=["DML"])],
        )
        assert _should_audit_assumptions(state) == "assumption_auditor"

    def test_routes_to_synthesis_when_no_methods(self) -> None:
        """Skips assumption_auditor when no methods identified."""
        state = ResearchState(
            query="test",
            sub_tasks=[SubTask(description="t", methods_to_audit=[])],
        )
        assert _should_audit_assumptions(state) == "synthesis"

    def test_routes_to_synthesis_with_empty_subtasks(self) -> None:
        """Handles empty sub_tasks list."""
        state = ResearchState(query="test", sub_tasks=[])
        assert _should_audit_assumptions(state) == "synthesis"


class TestEndToEnd:
    """End-to-end graph execution with mocked dependencies."""

    @pytest.mark.asyncio
    async def test_full_pipeline_with_methods(
        self, e2e_config: AgentConfig, mock_mcp: AsyncMock
    ) -> None:
        """Full pipeline including assumption auditor."""
        # Mock LLM responses for query_planner and synthesis
        mock_llm_response = MagicMock()
        mock_llm_response.content = """[
            {
                "description": "Find DML papers",
                "search_queries": ["double machine learning"],
                "concepts_to_explore": ["double machine learning"],
                "methods_to_audit": ["DML"]
            }
        ]"""

        mock_synthesis_response = MagicMock()
        mock_synthesis_response.content = "# Research Report\n\nThis is the synthesis."

        with patch("research_agent.nodes.query_planner.ChatAnthropic") as mock_planner_cls, \
             patch("research_agent.nodes.synthesis.ChatAnthropic") as mock_synth_cls:

            # Setup planner mock
            mock_planner = AsyncMock()
            mock_planner.ainvoke.return_value = mock_llm_response
            mock_planner_cls.return_value = mock_planner

            # Setup synthesis mock
            mock_synth = AsyncMock()
            mock_synth.ainvoke.return_value = mock_synthesis_response
            mock_synth_cls.return_value = mock_synth

            graph = build_graph(e2e_config, mock_mcp)
            initial = ResearchState(
                query="What are the assumptions of double machine learning?"
            )
            result = await graph.ainvoke(initial)

        # Verify all nodes executed
        assert result["current_node"] == "synthesis_writer"
        assert "report" in result
        assert len(result["report"]) > 0

        # Verify MCP calls were made
        assert mock_mcp.search.call_count >= 1
        assert mock_mcp.graph_neighborhood.call_count >= 1
        assert mock_mcp.audit_assumptions.call_count >= 1

    @pytest.mark.asyncio
    async def test_pipeline_skips_auditor(
        self, e2e_config: AgentConfig, mock_mcp: AsyncMock
    ) -> None:
        """Pipeline skips assumption auditor when no methods identified."""
        mock_llm_response = MagicMock()
        mock_llm_response.content = """[
            {
                "description": "Explore RAG architectures",
                "search_queries": ["RAG retrieval augmented generation"],
                "concepts_to_explore": ["retrieval augmented generation"],
                "methods_to_audit": []
            }
        ]"""

        mock_synthesis_response = MagicMock()
        mock_synthesis_response.content = "# RAG Report\n\nFindings about RAG."

        with patch("research_agent.nodes.query_planner.ChatAnthropic") as mock_planner_cls, \
             patch("research_agent.nodes.synthesis.ChatAnthropic") as mock_synth_cls:

            mock_planner = AsyncMock()
            mock_planner.ainvoke.return_value = mock_llm_response
            mock_planner_cls.return_value = mock_planner

            mock_synth = AsyncMock()
            mock_synth.ainvoke.return_value = mock_synthesis_response
            mock_synth_cls.return_value = mock_synth

            graph = build_graph(e2e_config, mock_mcp)
            result = await graph.ainvoke(ResearchState(query="How does RAG work?"))

        # Assumption auditor should NOT have been called
        assert mock_mcp.audit_assumptions.call_count == 0
        assert result["report"] is not None


class TestStateDataclasses:
    """Test state dataclass defaults and creation."""

    def test_default_state_creation(self) -> None:
        """ResearchState can be created with minimal args."""
        state = ResearchState(query="test")
        assert state.query == "test"
        assert state.sub_tasks == []
        assert state.search_results == []
        assert state.report == ""

    def test_subtask_defaults(self) -> None:
        """SubTask has correct defaults."""
        task = SubTask(description="test")
        assert task.search_queries == []
        assert task.concepts_to_explore == []
        assert task.methods_to_audit == []
