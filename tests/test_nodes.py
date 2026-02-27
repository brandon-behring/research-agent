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
from research_agent.nodes.citation_analyzer import (
    _parse_source_detail,
    citation_analyzer,
)
from research_agent.nodes.concept_explorer import concept_explorer
from research_agent.nodes.literature_search import literature_search
from research_agent.nodes.query_planner import _build_system_prompt
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

    @pytest.mark.asyncio
    async def test_auto_discovers_assumption_theorem_neighbors(
        self, test_config: AgentConfig, mock_mcp: ResearchKBClient
    ) -> None:
        """Auto-discovers ASSUMPTION/THEOREM type neighbors as methods."""
        state = ResearchState(
            query="test",
            sub_tasks=[
                SubTask(description="t", concepts_to_explore=["double machine learning"]),
            ],
        )
        result = await concept_explorer(state, test_config, mock_mcp)

        # The mock graph_neighborhood returns Unconfoundedness (ASSUMPTION),
        # Overlap condition (ASSUMPTION), and Neyman orthogonality (THEOREM)
        assert "discovered_methods" in result
        discovered = result["discovered_methods"]
        assert len(discovered) == 3  # capped at _MAX_DISCOVERED_METHODS=3
        names_lower = [m.lower() for m in discovered]
        assert "unconfoundedness" in names_lower
        assert "overlap condition" in names_lower
        assert "neyman orthogonality" in names_lower

    @pytest.mark.asyncio
    async def test_discovery_caps_at_max(
        self, test_config: AgentConfig, mock_mcp: ResearchKBClient
    ) -> None:
        """Auto-discovery respects _MAX_DISCOVERED_METHODS cap."""
        import json

        mock_mcp.graph_neighborhood.return_value = json.dumps(
            {
                "center": {"id": "c1", "name": "test", "type": "METHOD"},
                "nodes": [{"name": f"Assumption {i}", "type": "ASSUMPTION"} for i in range(10)],
                "edges": [],
                "relationship_type_counts": {},
            }
        )
        state = ResearchState(
            query="test",
            sub_tasks=[SubTask(description="t", concepts_to_explore=["test"])],
        )
        result = await concept_explorer(state, test_config, mock_mcp)

        # Should be capped at 3
        assert len(result["discovered_methods"]) == 3

    @pytest.mark.asyncio
    async def test_no_discovery_when_no_assumption_neighbors(
        self, test_config: AgentConfig, mock_mcp: ResearchKBClient
    ) -> None:
        """No discovered_methods when graph has no ASSUMPTION/THEOREM neighbors."""
        import json

        mock_mcp.graph_neighborhood.return_value = json.dumps(
            {
                "center": {"id": "c1", "name": "RAG", "type": "METHOD"},
                "nodes": [
                    {"name": "Dense retrieval", "type": "METHOD"},
                    {"name": "Chunk size", "type": "PARAMETER"},
                ],
                "edges": [],
                "relationship_type_counts": {},
            }
        )
        state = ResearchState(
            query="test",
            sub_tasks=[SubTask(description="t", concepts_to_explore=["RAG"])],
        )
        result = await concept_explorer(state, test_config, mock_mcp)

        assert result["discovered_methods"] == []

    @pytest.mark.asyncio
    async def test_discovery_deduplicates_across_concepts(
        self, test_config: AgentConfig, mock_mcp: ResearchKBClient
    ) -> None:
        """Duplicate ASSUMPTION names across concepts are deduplicated."""
        # Both concepts return the same ASSUMPTION neighbor
        state = ResearchState(
            query="test",
            sub_tasks=[
                SubTask(
                    description="t",
                    concepts_to_explore=["DML", "cross-fitting"],
                ),
            ],
        )
        # max_concepts=3 in test_config, both will be explored
        result = await concept_explorer(state, test_config, mock_mcp)

        # "Unconfoundedness" appears in both neighborhoods but should be deduped
        names_lower = [m.lower() for m in result["discovered_methods"]]
        assert names_lower.count("unconfoundedness") == 1


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

    @pytest.mark.asyncio
    async def test_enriches_sources_with_get_source(
        self, test_config: AgentConfig, mock_mcp: ResearchKBClient
    ) -> None:
        """Enriches analyzed sources with full metadata via get_source."""
        state = ResearchState(
            query="test",
            search_results=[
                SearchResult(title="Paper A", content="...", source_id="src-001", score=0.9),
            ],
        )
        result = await citation_analyzer(state, test_config, mock_mcp)
        assert "source_details" in result
        assert len(result["source_details"]) == 1
        assert result["source_details"][0]["title"] is not None
        mock_mcp.get_source.assert_awaited()

    @pytest.mark.asyncio
    async def test_enrichment_graceful_on_failure(
        self, test_config: AgentConfig, mock_mcp: ResearchKBClient
    ) -> None:
        """Source enrichment failure doesn't block citation analysis."""
        mock_mcp.get_source.side_effect = RuntimeError("source not found")
        state = ResearchState(
            query="test",
            search_results=[
                SearchResult(title="Paper A", content="...", source_id="src-001", score=0.9),
            ],
        )
        result = await citation_analyzer(state, test_config, mock_mcp)
        assert len(result["citations"]) == 1  # Citations still work
        assert result["source_details"] == []  # Enrichment gracefully empty


class TestParseSourceDetail:
    """Tests for _parse_source_detail markdown parser."""

    def test_parses_full_metadata(self) -> None:
        """Extracts all fields from well-formed markdown."""
        raw = (
            "## Double Machine Learning\n\n"
            "**Authors:** Chernozhukov et al.\n"
            "**Year:** 2018\n"
            "**Type:** Paper\n"
            "**Source ID:** `src-001-dml`\n"
            "**DOI:** 10.1111/ectj.12097\n"
        )
        detail = _parse_source_detail(raw)
        assert detail["title"] == "Double Machine Learning"
        assert detail["authors"] == "Chernozhukov et al."
        assert detail["year"] == "2018"
        assert detail["type"] == "Paper"
        assert detail["source_id"] == "src-001-dml"
        assert detail["doi"] == "10.1111/ectj.12097"

    def test_handles_partial_metadata(self) -> None:
        """Handles response with only some fields."""
        raw = "## Some Title\n**Year:** 2020\n"
        detail = _parse_source_detail(raw)
        assert detail["title"] == "Some Title"
        assert detail["year"] == "2020"
        assert "authors" not in detail

    def test_returns_empty_on_no_structure(self) -> None:
        """Returns empty dict on unrecognizable input."""
        assert _parse_source_detail("just plain text") == {}


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
    async def test_merges_discovered_methods(
        self, test_config: AgentConfig, mock_mcp: ResearchKBClient
    ) -> None:
        """Includes auto-discovered methods in auditing."""
        state = ResearchState(
            query="test",
            sub_tasks=[SubTask(description="t", methods_to_audit=["DML"])],
            discovered_methods=["Unconfoundedness", "Overlap condition"],
        )
        result = await assumption_auditor(state, test_config, mock_mcp)

        # Should audit 3 unique methods: DML + 2 discovered
        assert len(result["assumption_audits"]) == 3
        assert mock_mcp.audit_assumptions.call_count == 3

    @pytest.mark.asyncio
    async def test_deduplicates_discovered_against_planner(
        self, test_config: AgentConfig, mock_mcp: ResearchKBClient
    ) -> None:
        """Discovered methods that overlap with planner methods are deduped."""
        state = ResearchState(
            query="test",
            sub_tasks=[SubTask(description="t", methods_to_audit=["DML"])],
            discovered_methods=["dml", "Overlap"],  # "dml" duplicates "DML"
        )
        result = await assumption_auditor(state, test_config, mock_mcp)

        # Should audit 2 unique methods: DML + Overlap
        assert len(result["assumption_audits"]) == 2
        assert mock_mcp.audit_assumptions.call_count == 2

    @pytest.mark.asyncio
    async def test_passes_domain_when_all_subtasks_agree(
        self, test_config: AgentConfig, mock_mcp: ResearchKBClient
    ) -> None:
        """Passes inferred domain to MCP when all sub-tasks share same domain."""
        state = ResearchState(
            query="test",
            sub_tasks=[
                SubTask(
                    description="t1",
                    methods_to_audit=["DML"],
                    search_domain="causal_inference",
                ),
                SubTask(
                    description="t2",
                    methods_to_audit=[],
                    search_domain="causal_inference",
                ),
            ],
        )
        await assumption_auditor(state, test_config, mock_mcp)

        call_kwargs = mock_mcp.audit_assumptions.call_args
        assert call_kwargs.kwargs.get("domain") == "causal_inference"
        assert call_kwargs.kwargs.get("scope") == "applied"

    @pytest.mark.asyncio
    async def test_no_domain_when_subtasks_disagree(
        self, test_config: AgentConfig, mock_mcp: ResearchKBClient
    ) -> None:
        """No domain passed when sub-tasks have different domains."""
        state = ResearchState(
            query="test",
            sub_tasks=[
                SubTask(
                    description="t1",
                    methods_to_audit=["DML"],
                    search_domain="causal_inference",
                ),
                SubTask(
                    description="t2",
                    methods_to_audit=[],
                    search_domain="time_series",
                ),
            ],
        )
        await assumption_auditor(state, test_config, mock_mcp)

        call_kwargs = mock_mcp.audit_assumptions.call_args
        assert call_kwargs.kwargs.get("domain") is None
        assert call_kwargs.kwargs.get("scope") == "general"

    @pytest.mark.asyncio
    async def test_handles_audit_failure(
        self, sample_state: ResearchState, test_config: AgentConfig, mock_mcp: ResearchKBClient
    ) -> None:
        """Records failure without crashing."""
        mock_mcp.audit_assumptions.side_effect = RuntimeError("Method not found")

        result = await assumption_auditor(sample_state, test_config, mock_mcp)

        assert len(result["assumption_audits"]) == 1
        assert "failed" in result["assumption_audits"][0].raw_output.lower()


class TestPlannerSystemPrompt:
    """Tests for dynamic planner system prompt building."""

    def test_includes_domains_when_available(self) -> None:
        """Prompt includes KB domains when pre-pipeline populated them."""
        state = ResearchState(
            query="test",
            kb_domains=["causal_inference", "time_series"],
            kb_stats_summary="495 sources, 226K chunks",
        )
        prompt = _build_system_prompt(state)
        assert "causal_inference" in prompt
        assert "time_series" in prompt
        assert "495 sources" in prompt

    def test_falls_back_to_default_description(self) -> None:
        """Prompt uses default KB description when no context available."""
        state = ResearchState(query="test")
        prompt = _build_system_prompt(state)
        assert "causal inference" in prompt.lower()
        assert "Available KB domains" not in prompt

    def test_includes_stats_without_domains(self) -> None:
        """Prompt includes stats even if domains list is empty."""
        state = ResearchState(query="test", kb_stats_summary="100 sources, 50K chunks")
        prompt = _build_system_prompt(state)
        assert "100 sources" in prompt
