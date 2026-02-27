"""End-to-end tests for the research analysis graph.

Tests the full pipeline with mocked MCP client and mocked LLM responses.
Validates that:
1. All nodes execute in correct order
2. State flows through the graph correctly
3. Conditional routing works (assumption auditor skip)
4. Final report is generated
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from research_agent.config import AgentConfig, MCPConfig, ModelConfig
from research_agent.exceptions import NodeTimeoutError, PlannerError
from research_agent.graph import (
    _CRITICAL_NODES,
    StreamEvent,
    _fetch_kb_context,
    _make_resilient_node,
    _parse_domain_list,
    _parse_stats_summary,
    _passthrough,
    _should_audit_assumptions,
    _summarize_update,
    build_graph,
    run_research,
    stream_research,
)
from research_agent.nodes.query_planner import PlannerOutput
from research_agent.nodes.synthesis import Finding, SynthesisReport
from research_agent.state import ResearchState, SubTask


@pytest.fixture
def e2e_config() -> AgentConfig:
    """Config for end-to-end tests."""
    return AgentConfig(
        models=ModelConfig(
            planning="claude-haiku-4-5-20251001",
            synthesis="claude-haiku-4-5-20251001",
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

    def test_routes_to_connection_explorer_when_no_methods(self) -> None:
        """Skips assumption_auditor → goes to connection_explorer when no methods."""
        state = ResearchState(
            query="test",
            sub_tasks=[SubTask(description="t", methods_to_audit=[])],
        )
        assert _should_audit_assumptions(state) == "connection_explorer"

    def test_routes_to_auditor_when_discovered_methods_present(self) -> None:
        """Routes to assumption_auditor when discovered_methods exist."""
        state = ResearchState(
            query="test",
            sub_tasks=[SubTask(description="t", methods_to_audit=[])],
            discovered_methods=["Unconfoundedness"],
        )
        assert _should_audit_assumptions(state) == "assumption_auditor"

    def test_routes_to_connection_explorer_with_empty_subtasks(self) -> None:
        """Handles empty sub_tasks list → connection_explorer."""
        state = ResearchState(query="test", sub_tasks=[])
        assert _should_audit_assumptions(state) == "connection_explorer"


class TestParseDomainList:
    """Tests for _parse_domain_list markdown parser."""

    def test_parses_markdown_table(self) -> None:
        """Extracts domain names from markdown table."""
        raw = (
            "## Available Domains\n\n"
            "| Domain | Sources | Concepts |\n"
            "|--------|---------|----------|\n"
            "| causal_inference | 312 | 145 |\n"
            "| time_series | 98 | 52 |\n"
        )
        assert _parse_domain_list(raw) == ["causal_inference", "time_series"]

    def test_returns_empty_on_no_table(self) -> None:
        """Returns empty list when no table found."""
        assert _parse_domain_list("No table here") == []

    def test_skips_header_row(self) -> None:
        """Skips the 'Domain' header row."""
        raw = "| Domain | Count |\n|---|---|\n| stats | 10 |\n"
        assert _parse_domain_list(raw) == ["stats"]

    def test_handles_empty_input(self) -> None:
        """Handles empty string gracefully."""
        assert _parse_domain_list("") == []


class TestParseStatsSummary:
    """Tests for _parse_stats_summary markdown parser."""

    def test_parses_sources_and_chunks(self) -> None:
        """Extracts sources and chunks from bullet list."""
        raw = "## Knowledge Base Statistics\n\n- **Sources:** 495\n- **Chunks:** 226,432\n"
        assert _parse_stats_summary(raw) == "495 sources, 226,432 chunks"

    def test_partial_data_sources_only(self) -> None:
        """Returns just sources when chunks missing."""
        raw = "- **Sources:** 100\n"
        assert _parse_stats_summary(raw) == "100 sources"

    def test_empty_on_no_data(self) -> None:
        """Returns empty string when no stats found."""
        assert _parse_stats_summary("No stats here") == ""

    def test_handles_empty_input(self) -> None:
        """Handles empty string gracefully."""
        assert _parse_stats_summary("") == ""


class TestFetchKBContext:
    """Tests for _fetch_kb_context pre-pipeline helper."""

    @pytest.mark.asyncio
    async def test_returns_domains_and_stats(self, mock_mcp: AsyncMock) -> None:
        """Successfully fetches domains and stats from MCP."""
        domains, stats = await _fetch_kb_context(mock_mcp)
        assert "causal_inference" in domains
        assert "time_series" in domains
        assert "sources" in stats

    @pytest.mark.asyncio
    async def test_graceful_on_list_domains_failure(self, mock_mcp: AsyncMock) -> None:
        """Returns empty domains when list_domains fails."""
        from research_agent.exceptions import MCPToolError

        mock_mcp.list_domains.side_effect = MCPToolError("list_domains", "timeout")
        domains, stats = await _fetch_kb_context(mock_mcp)
        assert domains == []
        assert "sources" in stats  # stats still works

    @pytest.mark.asyncio
    async def test_graceful_on_stats_failure(self, mock_mcp: AsyncMock) -> None:
        """Returns empty stats when stats fails."""
        from research_agent.exceptions import MCPToolError

        mock_mcp.stats.side_effect = MCPToolError("stats", "timeout")
        domains, stats = await _fetch_kb_context(mock_mcp)
        assert len(domains) > 0  # domains still works
        assert stats == ""

    @pytest.mark.asyncio
    async def test_graceful_on_both_failures(self, mock_mcp: AsyncMock) -> None:
        """Returns empty values when both calls fail."""
        mock_mcp.list_domains.side_effect = RuntimeError("down")
        mock_mcp.stats.side_effect = RuntimeError("down")
        domains, stats = await _fetch_kb_context(mock_mcp)
        assert domains == []
        assert stats == ""


class TestEndToEnd:
    """End-to-end graph execution with mocked dependencies."""

    @pytest.mark.asyncio
    async def test_full_pipeline_with_methods(
        self, e2e_config: AgentConfig, mock_mcp: AsyncMock
    ) -> None:
        """Full pipeline including assumption auditor."""
        # Structured output models — with_structured_output returns these directly
        planner_output = PlannerOutput(
            sub_tasks=[
                SubTask(
                    description="Find DML papers",
                    search_queries=["double machine learning"],
                    concepts_to_explore=["double machine learning"],
                    methods_to_audit=["DML"],
                )
            ],
            rationale="Decomposing DML query into foundational search.",
        )

        synthesis_output = SynthesisReport(
            executive_summary="DML provides a framework for causal inference.",
            key_findings=[
                Finding(text="DML uses cross-fitting", confidence="high", source_count=3),
                Finding(text="Requires unconfoundedness", confidence="medium", source_count=1),
            ],
            concept_map="DML -> cross-fitting -> unconfoundedness",
            citation_landscape="Chernozhukov et al. (2018) is foundational.",
            methodological_considerations="Overlap assumption is critical.",
            gaps_limitations="Limited coverage of finite-sample properties.",
            confidence_level="medium",
            confidence_reasoning="Good coverage of core concepts, some gaps.",
        )

        with (
            patch("research_agent.nodes.query_planner.create_llm") as mock_planner_cls,
            patch("research_agent.nodes.synthesis.create_llm") as mock_synth_cls,
        ):
            # Setup planner mock — with_structured_output returns model directly
            mock_planner = AsyncMock()
            mock_planner.ainvoke.return_value = planner_output
            mock_planner_cls.return_value.with_structured_output.return_value = mock_planner

            # Setup synthesis mock
            mock_synth = AsyncMock()
            mock_synth.ainvoke.return_value = synthesis_output
            mock_synth_cls.return_value.with_structured_output.return_value = mock_synth

            graph = build_graph(e2e_config, mock_mcp)
            initial = ResearchState(query="What are the assumptions of double machine learning?")
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
        planner_output = PlannerOutput(
            sub_tasks=[
                SubTask(
                    description="Explore RAG architectures",
                    search_queries=["RAG retrieval augmented generation"],
                    concepts_to_explore=["retrieval augmented generation"],
                    methods_to_audit=[],
                )
            ],
            rationale="RAG-focused query, no statistical methods.",
        )

        synthesis_output = SynthesisReport(
            executive_summary="RAG combines retrieval with generation.",
            key_findings=[
                Finding(text="Dense retrieval outperforms sparse", confidence="high"),
            ],
            concept_map="RAG -> retrieval -> generation",
            citation_landscape="Lewis et al. (2020) introduced RAG.",
            methodological_considerations="Chunk size affects quality.",
            gaps_limitations="Limited coverage in this KB.",
            confidence_level="low",
            confidence_reasoning="Few sources available on RAG.",
        )

        # Override graph_neighborhood to return NO ASSUMPTION/THEOREM neighbors
        # so auto-discovery doesn't trigger the auditor
        import json

        mock_mcp.graph_neighborhood.return_value = json.dumps(
            {
                "center": {"id": "concept-rag-001", "name": "RAG", "type": "METHOD"},
                "nodes": [
                    {"name": "Dense retrieval", "type": "METHOD"},
                    {"name": "Chunk size", "type": "PARAMETER"},
                ],
                "edges": [],
                "relationship_type_counts": {},
            }
        )

        with (
            patch("research_agent.nodes.query_planner.create_llm") as mock_planner_cls,
            patch("research_agent.nodes.synthesis.create_llm") as mock_synth_cls,
        ):
            mock_planner = AsyncMock()
            mock_planner.ainvoke.return_value = planner_output
            mock_planner_cls.return_value.with_structured_output.return_value = mock_planner

            mock_synth = AsyncMock()
            mock_synth.ainvoke.return_value = synthesis_output
            mock_synth_cls.return_value.with_structured_output.return_value = mock_synth

            graph = build_graph(e2e_config, mock_mcp)
            result = await graph.ainvoke(ResearchState(query="How does RAG work?"))

        # Assumption auditor should NOT have been called (no planner methods,
        # no ASSUMPTION/THEOREM neighbors in graph)
        assert mock_mcp.audit_assumptions.call_count == 0
        assert result["report"] is not None


class TestStateModels:
    """Test state Pydantic model defaults and creation."""

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


class TestPassthrough:
    """Tests for the analysis_join passthrough node."""

    def test_returns_node_update(self) -> None:
        """Passthrough returns NodeUpdate with current_node set."""
        state = ResearchState(query="test")
        result = _passthrough(state)
        assert result["current_node"] == "analysis_join"


class TestSummarizeUpdate:
    """Tests for the _summarize_update helper."""

    def test_query_planner_summary(self) -> None:
        """Summarizes query planner output."""
        update = {"sub_tasks": [{"description": "a"}, {"description": "b"}]}
        result = _summarize_update("query_planner", update)
        assert "2 sub-tasks" in result

    def test_literature_search_summary(self) -> None:
        """Summarizes literature search output."""
        update = {"search_results": [1, 2, 3]}
        result = _summarize_update("literature_search", update)
        assert "3 search results" in result

    def test_synthesis_summary(self) -> None:
        """Summarizes synthesis output."""
        update = {"report": "A" * 500}
        result = _summarize_update("synthesis", update)
        assert "500 chars" in result

    def test_analysis_join_summary(self) -> None:
        """Summarizes analysis_join passthrough."""
        result = _summarize_update("analysis_join", {})
        assert "parallel join" in result.lower()

    def test_unknown_node(self) -> None:
        """Unknown node gets generic summary."""
        result = _summarize_update("unknown_node", {})
        assert "Completed unknown_node" in result


class TestStreamResearch:
    """Tests for stream_research() async generator."""

    @pytest.mark.asyncio
    async def test_yields_node_events(self, e2e_config: AgentConfig, mock_mcp: AsyncMock) -> None:
        """Stream yields node_end events for each pipeline node."""
        planner_output = PlannerOutput(
            sub_tasks=[
                SubTask(
                    description="Find DML papers",
                    search_queries=["double machine learning"],
                    concepts_to_explore=["double machine learning"],
                    methods_to_audit=["DML"],
                )
            ],
            rationale="Decomposing DML query.",
        )

        synthesis_output = SynthesisReport(
            executive_summary="DML provides a framework.",
            key_findings=[
                Finding(text="DML uses cross-fitting", confidence="high", source_count=2),
            ],
            concept_map="DML -> cross-fitting",
            citation_landscape="Chernozhukov et al.",
            methodological_considerations="Overlap is critical.",
            gaps_limitations="Limited coverage.",
            confidence_level="medium",
            confidence_reasoning="Good coverage.",
        )

        with (
            patch("research_agent.nodes.query_planner.create_llm") as mock_planner_cls,
            patch("research_agent.nodes.synthesis.create_llm") as mock_synth_cls,
            patch("research_agent.mcp_client.ResearchKBClient.__aenter__", return_value=mock_mcp),
            patch("research_agent.mcp_client.ResearchKBClient.__aexit__", return_value=None),
        ):
            mock_planner = AsyncMock()
            mock_planner.ainvoke.return_value = planner_output
            mock_planner_cls.return_value.with_structured_output.return_value = mock_planner

            mock_synth = AsyncMock()
            mock_synth.ainvoke.return_value = synthesis_output
            mock_synth_cls.return_value.with_structured_output.return_value = mock_synth

            events: list[StreamEvent] = []
            async for event in stream_research("DML assumptions", e2e_config):
                events.append(event)

        # Verify node_end events for all pipeline nodes
        node_end_events = [e for e in events if e.event_type == "node_end"]
        node_names = [e.node_name for e in node_end_events]
        assert "query_planner" in node_names
        assert "literature_search" in node_names
        assert "concept_explorer" in node_names
        assert "citation_analyzer" in node_names
        assert "synthesis" in node_names
        # With DML methods, assumption_auditor should also fire
        assert "assumption_auditor" in node_names

    @pytest.mark.asyncio
    async def test_yields_report_chunk(self, e2e_config: AgentConfig, mock_mcp: AsyncMock) -> None:
        """Stream yields report_chunk event with report content."""
        planner_output = PlannerOutput(
            sub_tasks=[
                SubTask(
                    description="Find DML papers",
                    search_queries=["DML"],
                    concepts_to_explore=["DML"],
                    methods_to_audit=["DML"],
                )
            ],
            rationale="Test.",
        )

        synthesis_output = SynthesisReport(
            executive_summary="Test report content.",
            key_findings=[Finding(text="Finding 1", confidence="medium")],
            concept_map="A -> B",
            citation_landscape="Cite.",
            methodological_considerations="Method.",
            gaps_limitations="Gap.",
            confidence_level="low",
            confidence_reasoning="Test.",
        )

        with (
            patch("research_agent.nodes.query_planner.create_llm") as mock_planner_cls,
            patch("research_agent.nodes.synthesis.create_llm") as mock_synth_cls,
            patch("research_agent.mcp_client.ResearchKBClient.__aenter__", return_value=mock_mcp),
            patch("research_agent.mcp_client.ResearchKBClient.__aexit__", return_value=None),
        ):
            mock_planner = AsyncMock()
            mock_planner.ainvoke.return_value = planner_output
            mock_planner_cls.return_value.with_structured_output.return_value = mock_planner

            mock_synth = AsyncMock()
            mock_synth.ainvoke.return_value = synthesis_output
            mock_synth_cls.return_value.with_structured_output.return_value = mock_synth

            events: list[StreamEvent] = []
            async for event in stream_research("DML", e2e_config):
                events.append(event)

        report_events = [e for e in events if e.event_type == "report_chunk"]
        assert len(report_events) == 1
        assert len(report_events[0].data) > 0

    @pytest.mark.asyncio
    async def test_yields_complete_event(
        self, e2e_config: AgentConfig, mock_mcp: AsyncMock
    ) -> None:
        """Stream yields terminal complete event."""
        planner_output = PlannerOutput(
            sub_tasks=[
                SubTask(
                    description="Find papers",
                    search_queries=["test"],
                    concepts_to_explore=["test"],
                    methods_to_audit=[],
                )
            ],
            rationale="Test.",
        )

        synthesis_output = SynthesisReport(
            executive_summary="Summary.",
            key_findings=[Finding(text="F", confidence="low")],
            concept_map="A",
            citation_landscape="C",
            methodological_considerations="M",
            gaps_limitations="G",
            confidence_level="low",
            confidence_reasoning="Low.",
        )

        with (
            patch("research_agent.nodes.query_planner.create_llm") as mock_planner_cls,
            patch("research_agent.nodes.synthesis.create_llm") as mock_synth_cls,
            patch("research_agent.mcp_client.ResearchKBClient.__aenter__", return_value=mock_mcp),
            patch("research_agent.mcp_client.ResearchKBClient.__aexit__", return_value=None),
        ):
            mock_planner = AsyncMock()
            mock_planner.ainvoke.return_value = planner_output
            mock_planner_cls.return_value.with_structured_output.return_value = mock_planner

            mock_synth = AsyncMock()
            mock_synth.ainvoke.return_value = synthesis_output
            mock_synth_cls.return_value.with_structured_output.return_value = mock_synth

            events: list[StreamEvent] = []
            async for event in stream_research("test", e2e_config):
                events.append(event)

        # Last event should be complete
        assert events[-1].event_type == "complete"
        assert "chars" in events[-1].data

    @pytest.mark.asyncio
    async def test_empty_query_raises_planner_error(self) -> None:
        """Empty query raises PlannerError."""
        with pytest.raises(PlannerError, match="empty"):
            async for _ in stream_research(""):
                pass

    @pytest.mark.asyncio
    async def test_whitespace_query_raises_planner_error(self) -> None:
        """Whitespace-only query raises PlannerError."""
        with pytest.raises(PlannerError, match="empty"):
            async for _ in stream_research("  \t  "):
                pass

    @pytest.mark.asyncio
    async def test_stream_yields_all_event_types(self) -> None:
        """stream_research yields node_end, report_chunk, complete events."""

        async def _fake_astream(*args, **kwargs):
            yield {
                "query_planner": {
                    "sub_tasks": [{"description": "t1"}],
                    "node_duration_ms": 100,
                },
            }
            yield {
                "synthesis": {
                    "report": "# Test Report",
                    "node_duration_ms": 200,
                },
            }

        with (
            patch("research_agent.graph.ResearchKBClient") as mock_cls,
            patch(
                "research_agent.graph._fetch_kb_context",
                return_value=([], ""),
            ),
            patch("research_agent.graph.build_graph") as mock_build,
        ):
            mock_mcp = AsyncMock()
            mock_cls.return_value.__aenter__ = AsyncMock(
                return_value=mock_mcp,
            )
            mock_cls.return_value.__aexit__ = AsyncMock(
                return_value=None,
            )
            mock_graph = AsyncMock()
            mock_graph.astream = _fake_astream
            mock_build.return_value = mock_graph

            events = []
            async for event in stream_research("What is DML?"):
                events.append(event)

            types = [e.event_type for e in events]
            assert "node_end" in types
            assert "report_chunk" in types
            assert "complete" in types
            complete = [e for e in events if e.event_type == "complete"][0]
            assert complete.duration_ms is not None
            assert complete.duration_ms >= 0


class TestResilientNode:
    """Tests for _make_resilient_node timeout and error handling."""

    @pytest.mark.asyncio
    async def test_successful_node_passes_through(self) -> None:
        """Successful node returns its result unchanged."""
        from research_agent.state import NodeUpdate

        async def good_node(state: ResearchState) -> NodeUpdate:
            return NodeUpdate(current_node="test", search_summary="ok")

        wrapped = _make_resilient_node("concept_explorer", good_node, 5)
        state = ResearchState(query="test")
        result = await wrapped(state)
        assert result["search_summary"] == "ok"
        assert result["current_node"] == "test"

    @pytest.mark.asyncio
    async def test_noncritical_timeout_returns_empty(self) -> None:
        """Non-critical node timeout returns empty NodeUpdate."""
        import asyncio

        from research_agent.state import NodeUpdate

        async def slow_node(state: ResearchState) -> NodeUpdate:
            await asyncio.sleep(10)
            return NodeUpdate(current_node="never_reached")

        wrapped = _make_resilient_node("concept_explorer", slow_node, 0)
        state = ResearchState(query="test")
        result = await wrapped(state)
        assert result["current_node"] == "concept_explorer"

    @pytest.mark.asyncio
    async def test_critical_timeout_raises_node_timeout_error(self) -> None:
        """Critical node timeout raises NodeTimeoutError."""
        import asyncio

        from research_agent.state import NodeUpdate

        async def slow_node(state: ResearchState) -> NodeUpdate:
            await asyncio.sleep(10)
            return NodeUpdate(current_node="never_reached")

        wrapped = _make_resilient_node("query_planner", slow_node, 0)
        state = ResearchState(query="test")
        with pytest.raises(NodeTimeoutError, match="query_planner"):
            await wrapped(state)

    @pytest.mark.asyncio
    async def test_noncritical_error_returns_empty(self) -> None:
        """Non-critical node error returns empty NodeUpdate."""
        from research_agent.state import NodeUpdate

        async def bad_node(state: ResearchState) -> NodeUpdate:
            raise RuntimeError("boom")

        wrapped = _make_resilient_node("citation_analyzer", bad_node, 5)
        state = ResearchState(query="test")
        result = await wrapped(state)
        assert result["current_node"] == "citation_analyzer"

    @pytest.mark.asyncio
    async def test_critical_error_propagates(self) -> None:
        """Critical node error propagates unchanged."""
        from research_agent.state import NodeUpdate

        async def bad_node(state: ResearchState) -> NodeUpdate:
            raise PlannerError("decomposition failed")

        wrapped = _make_resilient_node("query_planner", bad_node, 5)
        state = ResearchState(query="test")
        with pytest.raises(PlannerError, match="decomposition failed"):
            await wrapped(state)

    @pytest.mark.asyncio
    async def test_synthesis_is_critical(self) -> None:
        """Synthesis node propagates errors (critical)."""
        from research_agent.exceptions import SynthesisError
        from research_agent.state import NodeUpdate

        async def bad_node(state: ResearchState) -> NodeUpdate:
            raise SynthesisError("synthesis failed")

        wrapped = _make_resilient_node("synthesis", bad_node, 5)
        state = ResearchState(query="test")
        with pytest.raises(SynthesisError):
            await wrapped(state)

    def test_critical_nodes_set(self) -> None:
        """Verify _CRITICAL_NODES contains expected nodes."""
        assert {
            "query_planner",
            "literature_search",
            "synthesis",
        } == _CRITICAL_NODES

    @pytest.mark.asyncio
    async def test_wrapper_preserves_function_name(self) -> None:
        """Wrapped function has descriptive __name__."""
        from research_agent.state import NodeUpdate

        async def my_node(state: ResearchState) -> NodeUpdate:
            return NodeUpdate(current_node="test")

        wrapped = _make_resilient_node("concept_explorer", my_node, 5)
        assert "concept_explorer" in wrapped.__name__

    @pytest.mark.asyncio
    async def test_wrapper_injects_duration_ms(self) -> None:
        """Wrapped function injects node_duration_ms into result."""
        from research_agent.state import NodeUpdate

        async def fast_node(state: ResearchState) -> NodeUpdate:
            return NodeUpdate(current_node="test")

        wrapped = _make_resilient_node("concept_explorer", fast_node, 5)
        state = ResearchState(query="test")
        result = await wrapped(state)
        assert "node_duration_ms" in result
        assert isinstance(result["node_duration_ms"], int)
        assert result["node_duration_ms"] >= 0

    @pytest.mark.asyncio
    async def test_node_timeout_error_not_caught_by_generic_handler(
        self,
    ) -> None:
        """NodeTimeoutError propagates through except Exception handler."""
        from research_agent.state import NodeUpdate

        async def raises_nte(state: ResearchState) -> NodeUpdate:
            raise NodeTimeoutError("inner", 10)

        # Wrap a non-critical node that would normally catch exceptions
        wrapped = _make_resilient_node("concept_explorer", raises_nte, 5)
        state = ResearchState(query="test")
        # NodeTimeoutError is re-raised even for non-critical nodes
        with pytest.raises(NodeTimeoutError, match="inner"):
            await wrapped(state)


class TestStreamEventTiming:
    """Tests for StreamEvent timing fields."""

    def test_default_timing_fields(self) -> None:
        """StreamEvent defaults: duration_ms=None, timestamp=0.0."""
        event = StreamEvent(event_type="node_end", node_name="test", data="ok")
        assert event.duration_ms is None
        assert event.timestamp == 0.0

    def test_explicit_timing_fields(self) -> None:
        """StreamEvent accepts explicit timing values."""
        event = StreamEvent(
            event_type="node_end",
            node_name="test",
            data="ok",
            duration_ms=1500,
            timestamp=12345.678,
        )
        assert event.duration_ms == 1500
        assert event.timestamp == 12345.678


class TestRunResearch:
    """Tests for run_research() entry point."""

    @pytest.mark.asyncio
    async def test_empty_query_raises_planner_error(self) -> None:
        """Empty query raises PlannerError before touching MCP."""
        with pytest.raises(PlannerError, match="empty"):
            await run_research("")

    @pytest.mark.asyncio
    async def test_whitespace_query_raises_planner_error(self) -> None:
        """Whitespace-only query raises PlannerError."""
        with pytest.raises(PlannerError, match="empty"):
            await run_research("   ")

    @pytest.mark.asyncio
    async def test_none_config_uses_default(self) -> None:
        """Passing config=None creates a default AgentConfig."""
        # Patch MCP to avoid real connection — just verify entry path
        with (
            patch(
                "research_agent.graph.ResearchKBClient",
            ) as mock_cls,
            patch(
                "research_agent.graph._fetch_kb_context",
                return_value=([], ""),
            ),
            patch(
                "research_agent.graph.build_graph",
            ) as mock_build,
        ):
            mock_mcp = AsyncMock()
            mock_cls.return_value.__aenter__ = AsyncMock(
                return_value=mock_mcp,
            )
            mock_cls.return_value.__aexit__ = AsyncMock(
                return_value=None,
            )
            mock_graph = AsyncMock()
            mock_graph.ainvoke.return_value = {"report": "test"}
            mock_build.return_value = mock_graph

            result = await run_research("What is DML?")
            assert result["report"] == "test"
            mock_build.assert_called_once()
