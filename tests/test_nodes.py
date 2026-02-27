"""Unit tests for individual agent nodes.

Each node is tested with mocked MCP client to verify:
1. Correct MCP tool calls are made
2. Output state updates are well-formed
3. Error handling (MCP failures) doesn't crash the pipeline
"""

from __future__ import annotations

import pytest

from research_agent.config import AgentConfig
from research_agent.mcp_client import ResearchKBClient
from research_agent.nodes.assumption_auditor import assumption_auditor
from research_agent.nodes.citation_analyzer import citation_analyzer
from research_agent.nodes.concept_explorer import concept_explorer
from research_agent.nodes.literature_search import literature_search
from research_agent.state import ResearchState, SearchResult, SubTask


class TestLiteratureSearch:
    """Tests for the literature search node."""

    @pytest.mark.asyncio
    async def test_executes_all_queries(
        self, sample_state: ResearchState, test_config: AgentConfig, mock_mcp: ResearchKBClient
    ) -> None:
        """Runs search for every query in every sub-task."""
        result = await literature_search(sample_state, test_config, mock_mcp)

        # 3 queries across 2 sub-tasks
        assert mock_mcp.search.call_count == 3
        assert "search_results" in result
        assert len(result["search_results"]) > 0

    @pytest.mark.asyncio
    async def test_deduplicates_results(
        self, sample_state: ResearchState, test_config: AgentConfig, mock_mcp: ResearchKBClient
    ) -> None:
        """Deduplicates search results by source_id."""
        result = await literature_search(sample_state, test_config, mock_mcp)

        source_ids = [r.source_id for r in result["search_results"] if r.source_id]
        assert len(source_ids) == len(set(source_ids)), "Duplicate source_ids found"

    @pytest.mark.asyncio
    async def test_falls_back_to_fast_search(
        self, sample_state: ResearchState, test_config: AgentConfig, mock_mcp: ResearchKBClient
    ) -> None:
        """Falls back to fast_search when main search fails."""
        mock_mcp.search.side_effect = RuntimeError("MCP timeout")

        result = await literature_search(sample_state, test_config, mock_mcp)

        assert mock_mcp.fast_search.call_count > 0
        assert "search_results" in result

    @pytest.mark.asyncio
    async def test_results_sorted_by_score(
        self, sample_state: ResearchState, test_config: AgentConfig, mock_mcp: ResearchKBClient
    ) -> None:
        """Results are sorted by score descending."""
        result = await literature_search(sample_state, test_config, mock_mcp)

        results = result["search_results"]
        if len(results) >= 2:
            scores = [r.score for r in results]
            assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_produces_summary(
        self, sample_state: ResearchState, test_config: AgentConfig, mock_mcp: ResearchKBClient
    ) -> None:
        """Generates a search summary."""
        result = await literature_search(sample_state, test_config, mock_mcp)
        assert "search_summary" in result
        assert "results" in result["search_summary"].lower()

    @pytest.mark.asyncio
    async def test_gather_exception_degrades_gracefully(
        self, test_config: AgentConfig, mock_mcp: ResearchKBClient
    ) -> None:
        """OSError from gather doesn't crash the node — degrades gracefully."""
        mock_mcp.search.side_effect = OSError("Connection reset")
        mock_mcp.fast_search.side_effect = OSError("Connection reset")

        state = ResearchState(
            query="test",
            sub_tasks=[
                SubTask(description="t", search_queries=["q1"]),
            ],
        )
        result = await literature_search(state, test_config, mock_mcp)

        assert "search_results" in result
        assert "search_summary" in result

    @pytest.mark.asyncio
    async def test_threads_domain_and_context_to_search(
        self, test_config: AgentConfig, mock_mcp: ResearchKBClient
    ) -> None:
        """Passes sub-task search_domain and search_context to MCP search calls."""
        state = ResearchState(
            query="test",
            sub_tasks=[
                SubTask(
                    description="Causal task",
                    search_queries=["DML assumptions"],
                    search_domain="causal_inference",
                    search_context="auditing",
                ),
            ],
        )
        await literature_search(state, test_config, mock_mcp)

        # Verify domain and context_type were passed to MCP
        call_kwargs = mock_mcp.search.call_args
        assert call_kwargs.kwargs.get("domain") == "causal_inference"
        assert call_kwargs.kwargs.get("context_type") == "auditing"

    @pytest.mark.asyncio
    async def test_empty_domain_passes_none(
        self, test_config: AgentConfig, mock_mcp: ResearchKBClient
    ) -> None:
        """Empty search_domain passes None to MCP (no domain filter)."""
        state = ResearchState(
            query="test",
            sub_tasks=[
                SubTask(
                    description="Cross-domain",
                    search_queries=["general query"],
                    search_domain="",
                    search_context="balanced",
                ),
            ],
        )
        await literature_search(state, test_config, mock_mcp)

        call_kwargs = mock_mcp.search.call_args
        assert call_kwargs.kwargs.get("domain") is None

    @pytest.mark.asyncio
    async def test_sparse_result_triggers_fast_search_supplement(
        self, test_config: AgentConfig, mock_mcp: ResearchKBClient
    ) -> None:
        """When primary search returns < _MIN_SPARSE_THRESHOLD, fast_search supplements."""
        # Return only 1 result from primary search (below threshold of 2)
        mock_mcp.search.return_value = """### 1. Only Paper
*Score: 0.9*
*Source ID: `src-only`*
"""
        state = ResearchState(
            query="test",
            sub_tasks=[SubTask(description="t", search_queries=["sparse query"])],
        )
        result = await literature_search(state, test_config, mock_mcp)

        # fast_search should have been called to supplement
        assert mock_mcp.fast_search.call_count >= 1
        assert "search_results" in result


class TestConceptExplorer:
    """Tests for the concept explorer node."""

    @pytest.mark.asyncio
    async def test_explores_planned_concepts(
        self, sample_state: ResearchState, test_config: AgentConfig, mock_mcp: ResearchKBClient
    ) -> None:
        """Explores concepts listed in sub-tasks."""
        result = await concept_explorer(sample_state, test_config, mock_mcp)

        assert "concepts" in result
        assert len(result["concepts"]) > 0
        # Should have called graph_neighborhood for each unique concept
        assert mock_mcp.graph_neighborhood.call_count >= 1

    @pytest.mark.asyncio
    async def test_deduplicates_concepts(
        self, test_config: AgentConfig, mock_mcp: ResearchKBClient
    ) -> None:
        """Doesn't explore the same concept twice."""
        state = ResearchState(
            query="test",
            sub_tasks=[
                SubTask(
                    description="t1",
                    concepts_to_explore=["DML", "dml", "DML"],  # Duplicates
                ),
            ],
        )
        await concept_explorer(state, test_config, mock_mcp)

        # Should only call once for "DML" despite 3 entries
        assert mock_mcp.graph_neighborhood.call_count == 1

    @pytest.mark.asyncio
    async def test_handles_exploration_failure(
        self, sample_state: ResearchState, test_config: AgentConfig, mock_mcp: ResearchKBClient
    ) -> None:
        """Gracefully handles MCP failures."""
        mock_mcp.graph_neighborhood.side_effect = RuntimeError("Concept not found")

        result = await concept_explorer(sample_state, test_config, mock_mcp)

        # Should return empty but not crash
        assert "concepts" in result
        assert "concept_map_summary" in result

    @pytest.mark.asyncio
    async def test_gather_exception_degrades_gracefully(
        self, test_config: AgentConfig, mock_mcp: ResearchKBClient
    ) -> None:
        """OSError from gather doesn't crash the node — degrades gracefully."""
        mock_mcp.graph_neighborhood.side_effect = OSError("Connection reset")

        state = ResearchState(
            query="test",
            sub_tasks=[
                SubTask(description="t", concepts_to_explore=["DML"]),
            ],
        )
        result = await concept_explorer(state, test_config, mock_mcp)

        assert "concepts" in result
        assert len(result["concepts"]) == 0

    @pytest.mark.asyncio
    async def test_get_concept_enriches_concept_info(
        self, test_config: AgentConfig, mock_mcp: ResearchKBClient
    ) -> None:
        """get_concept enriches ConceptInfo with type, description, and relationships."""
        state = ResearchState(
            query="test",
            sub_tasks=[
                SubTask(description="t", concepts_to_explore=["double machine learning"]),
            ],
        )
        result = await concept_explorer(state, test_config, mock_mcp)

        assert len(result["concepts"]) == 1
        concept = result["concepts"][0]
        # Detail fields populated from get_concept response
        assert concept.concept_type == "METHOD"
        assert "cross-fitting" in concept.description
        assert len(concept.relationships) == 5
        assert concept.relationships[0]["type"] == "REQUIRES"
        # Neighborhood summary still present
        assert "Graph Neighborhood" in concept.neighborhood_summary
        # Name enriched from detail
        assert concept.name == "Double Machine Learning"

    @pytest.mark.asyncio
    async def test_get_concept_failure_falls_back_to_neighborhood_only(
        self, test_config: AgentConfig, mock_mcp: ResearchKBClient
    ) -> None:
        """When get_concept fails, falls back to neighborhood-only ConceptInfo."""
        mock_mcp.get_concept.side_effect = RuntimeError("Concept not found in KB")

        state = ResearchState(
            query="test",
            sub_tasks=[
                SubTask(description="t", concepts_to_explore=["double machine learning"]),
            ],
        )
        result = await concept_explorer(state, test_config, mock_mcp)

        assert len(result["concepts"]) == 1
        concept = result["concepts"][0]
        # Neighborhood still works
        assert concept.concept_id == "concept-dml-001"
        assert "Graph Neighborhood" in concept.neighborhood_summary
        # Detail fields empty (fallback)
        assert concept.concept_type == ""
        assert concept.description == ""
        assert concept.relationships == []
        # Name is the original input (not enriched)
        assert concept.name == "double machine learning"


class TestCitationAnalyzer:
    """Tests for the citation analyzer node."""

    @pytest.mark.asyncio
    async def test_analyzes_top_results(
        self, test_config: AgentConfig, mock_mcp: ResearchKBClient
    ) -> None:
        """Analyzes citation networks for top search results."""
        state = ResearchState(
            query="test",
            search_results=[
                SearchResult(
                    title="Paper A",
                    content="...",
                    source_id="src-001",
                    score=0.9,
                ),
                SearchResult(
                    title="Paper B",
                    content="...",
                    source_id="src-002",
                    score=0.8,
                ),
            ],
        )

        result = await citation_analyzer(state, test_config, mock_mcp)

        assert "citations" in result
        assert len(result["citations"]) == 2
        assert mock_mcp.citation_network.call_count == 2
        assert mock_mcp.biblio_coupling.call_count == 2

    @pytest.mark.asyncio
    async def test_skips_results_without_source_id(
        self, test_config: AgentConfig, mock_mcp: ResearchKBClient
    ) -> None:
        """Skips search results that don't have source IDs."""
        state = ResearchState(
            query="test",
            search_results=[
                SearchResult(title="No ID", content="...", source_id="", score=0.9),
            ],
        )

        result = await citation_analyzer(state, test_config, mock_mcp)

        assert len(result["citations"]) == 0
        assert mock_mcp.citation_network.call_count == 0

    @pytest.mark.asyncio
    async def test_deduplicates_source_ids(
        self, test_config: AgentConfig, mock_mcp: ResearchKBClient
    ) -> None:
        """Doesn't analyze the same source twice."""
        state = ResearchState(
            query="test",
            search_results=[
                SearchResult(title="A", content="", source_id="src-001", score=0.9),
                SearchResult(title="A dup", content="", source_id="src-001", score=0.8),
            ],
        )

        result = await citation_analyzer(state, test_config, mock_mcp)

        assert len(result["citations"]) == 1

    @pytest.mark.asyncio
    async def test_partial_failure_doesnt_block_other_sources(
        self, test_config: AgentConfig, mock_mcp: ResearchKBClient
    ) -> None:
        """If citation_network fails for one source, other sources still complete."""
        call_count = 0

        async def _network_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if kwargs.get("source_id") == "src-fail":
                raise RuntimeError("Network timeout")
            return mock_mcp.citation_network.return_value

        mock_mcp.citation_network.side_effect = _network_side_effect

        state = ResearchState(
            query="test",
            search_results=[
                SearchResult(title="Fails", content="", source_id="src-fail", score=0.9),
                SearchResult(title="Works", content="", source_id="src-ok", score=0.8),
            ],
        )

        result = await citation_analyzer(state, test_config, mock_mcp)

        # Both sources should be analyzed (parallel, not short-circuit)
        assert len(result["citations"]) == 2
        # The failing source has empty citing/cited_by
        fail_cit = next(c for c in result["citations"] if c.source_id == "src-fail")
        assert fail_cit.citing == []
        # The working source has populated citations
        ok_cit = next(c for c in result["citations"] if c.source_id == "src-ok")
        assert len(ok_cit.citing) > 0


class TestAssumptionAuditor:
    """Tests for the assumption auditor node."""

    @pytest.mark.asyncio
    async def test_audits_identified_methods(
        self, sample_state: ResearchState, test_config: AgentConfig, mock_mcp: ResearchKBClient
    ) -> None:
        """Audits all methods from sub-tasks."""
        result = await assumption_auditor(sample_state, test_config, mock_mcp)

        assert "assumption_audits" in result
        assert len(result["assumption_audits"]) == 1  # Only "DML"
        assert result["assumption_audits"][0].method_name == "DML"

    @pytest.mark.asyncio
    async def test_deduplicates_methods(
        self, test_config: AgentConfig, mock_mcp: ResearchKBClient
    ) -> None:
        """Doesn't audit the same method twice."""
        state = ResearchState(
            query="test",
            sub_tasks=[
                SubTask(description="t1", methods_to_audit=["DML", "dml"]),
                SubTask(description="t2", methods_to_audit=["DML"]),
            ],
        )

        await assumption_auditor(state, test_config, mock_mcp)

        assert mock_mcp.audit_assumptions.call_count == 1

    @pytest.mark.asyncio
    async def test_no_methods_returns_empty(
        self, test_config: AgentConfig, mock_mcp: ResearchKBClient
    ) -> None:
        """Returns empty when no methods identified."""
        state = ResearchState(
            query="test",
            sub_tasks=[SubTask(description="t1", methods_to_audit=[])],
        )

        result = await assumption_auditor(state, test_config, mock_mcp)

        assert len(result["assumption_audits"]) == 0
        assert "No statistical methods" in result["assumption_summary"]

    @pytest.mark.asyncio
    async def test_handles_audit_failure(
        self, sample_state: ResearchState, test_config: AgentConfig, mock_mcp: ResearchKBClient
    ) -> None:
        """Records failure without crashing."""
        mock_mcp.audit_assumptions.side_effect = RuntimeError("Method not found")

        result = await assumption_auditor(sample_state, test_config, mock_mcp)

        assert len(result["assumption_audits"]) == 1
        assert "failed" in result["assumption_audits"][0].raw_output.lower()
