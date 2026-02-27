"""Tests for Pydantic model validation.

Tests score bounds, empty query stripping, frozen immutability,
serialization roundtrip, and ValidationError on bad input.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from research_agent.state import (
    CitationInfo,
    ConceptInfo,
    ResearchState,
    SearchResult,
    SubTask,
)


class TestSubTask:
    """Tests for SubTask model."""

    def test_creates_with_description_only(self) -> None:
        """SubTask requires only description."""
        task = SubTask(description="test task")
        assert task.description == "test task"
        assert task.search_queries == []
        assert task.concepts_to_explore == []
        assert task.methods_to_audit == []

    def test_strips_empty_queries(self) -> None:
        """Empty and whitespace-only queries are filtered out."""
        task = SubTask(
            description="test",
            search_queries=["valid", "", "  ", "also valid"],
        )
        assert task.search_queries == ["valid", "also valid"]

    def test_strips_whitespace_from_queries(self) -> None:
        """Leading/trailing whitespace stripped from queries."""
        task = SubTask(
            description="test",
            search_queries=["  padded  ", "clean"],
        )
        assert task.search_queries == ["padded", "clean"]

    def test_frozen_immutability(self) -> None:
        """SubTask fields cannot be modified after creation."""
        task = SubTask(description="test")
        with pytest.raises(ValidationError):
            task.description = "changed"

    def test_search_domain_defaults_empty(self) -> None:
        """search_domain defaults to empty string (cross-domain)."""
        task = SubTask(description="test")
        assert task.search_domain == ""

    def test_search_context_defaults_balanced(self) -> None:
        """search_context defaults to 'balanced'."""
        task = SubTask(description="test")
        assert task.search_context == "balanced"

    def test_search_context_accepts_valid_values(self) -> None:
        """search_context accepts building, auditing, balanced, empty."""
        for ctx in ("balanced", "building", "auditing", ""):
            task = SubTask(description="test", search_context=ctx)
            assert task.search_context == ctx

    def test_search_context_rejects_invalid(self) -> None:
        """search_context rejects unknown values."""
        with pytest.raises(ValidationError, match="search_context"):
            SubTask(description="test", search_context="invalid")

    def test_connections_to_explain_defaults_empty(self) -> None:
        """connections_to_explain defaults to empty list."""
        task = SubTask(description="test")
        assert task.connections_to_explain == []

    def test_connections_to_explain_accepts_pairs(self) -> None:
        """connections_to_explain accepts list of concept pairs."""
        task = SubTask(
            description="test",
            connections_to_explain=[["DML", "cross-fitting"], ["IV", "LATE"]],
        )
        assert len(task.connections_to_explain) == 2
        assert task.connections_to_explain[0] == ["DML", "cross-fitting"]

    def test_serialization_roundtrip(self) -> None:
        """SubTask survives JSON serialization/deserialization."""
        task = SubTask(
            description="Find DML papers",
            search_queries=["double machine learning"],
            concepts_to_explore=["DML"],
            methods_to_audit=["DML"],
            search_domain="causal_inference",
            search_context="auditing",
            connections_to_explain=[["DML", "cross-fitting"]],
        )
        data = task.model_dump()
        reconstructed = SubTask(**data)
        assert reconstructed == task


class TestSearchResult:
    """Tests for SearchResult model."""

    def test_score_lower_bound(self) -> None:
        """Score must be >= 0.0."""
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            SearchResult(title="test", content="c", source_id="s", score=-0.1)

    def test_score_upper_bound(self) -> None:
        """Score must be <= 1.0."""
        with pytest.raises(ValidationError, match="less than or equal to 1"):
            SearchResult(title="test", content="c", source_id="s", score=1.5)

    def test_valid_score_range(self) -> None:
        """Score within [0, 1] is accepted."""
        r = SearchResult(title="t", content="c", source_id="s", score=0.85)
        assert r.score == 0.85

    def test_boundary_scores(self) -> None:
        """Boundary values 0.0 and 1.0 are valid."""
        r0 = SearchResult(title="t", content="c", source_id="s", score=0.0)
        r1 = SearchResult(title="t", content="c", source_id="s", score=1.0)
        assert r0.score == 0.0
        assert r1.score == 1.0

    def test_frozen_immutability(self) -> None:
        """SearchResult fields cannot be modified."""
        r = SearchResult(title="t", content="c", source_id="s", score=0.5)
        with pytest.raises(ValidationError):
            r.title = "changed"

    def test_default_optional_fields(self) -> None:
        """Optional fields default to empty strings."""
        r = SearchResult(title="t", content="c", source_id="s", score=0.5)
        assert r.authors == ""
        assert r.year == ""
        assert r.chunk_id == ""


class TestConceptInfo:
    """Tests for ConceptInfo model."""

    def test_creates_with_required_fields(self) -> None:
        """ConceptInfo requires concept_id and name."""
        c = ConceptInfo(concept_id="cid-001", name="DML")
        assert c.concept_id == "cid-001"
        assert c.name == "DML"

    def test_frozen_immutability(self) -> None:
        """ConceptInfo is frozen."""
        c = ConceptInfo(concept_id="cid", name="test")
        with pytest.raises(ValidationError):
            c.name = "changed"

    def test_relationships_default_empty(self) -> None:
        """Relationships default to empty list."""
        c = ConceptInfo(concept_id="cid", name="test")
        assert c.relationships == []


class TestCitationInfo:
    """Tests for CitationInfo model."""

    def test_creates_with_required_fields(self) -> None:
        """CitationInfo requires source_id and source_title."""
        c = CitationInfo(source_id="src-001", source_title="Paper A")
        assert c.source_id == "src-001"

    def test_default_lists(self) -> None:
        """Citing/cited_by/similar_papers default to empty."""
        c = CitationInfo(source_id="src", source_title="t")
        assert c.citing == []
        assert c.cited_by == []
        assert c.similar_papers == []


class TestResearchState:
    """Tests for ResearchState model."""

    def test_creates_with_defaults(self) -> None:
        """ResearchState can be created with all defaults."""
        s = ResearchState()
        assert s.query == ""
        assert s.sub_tasks == []
        assert s.report == ""

    def test_creates_with_query(self) -> None:
        """ResearchState accepts query."""
        s = ResearchState(query="What is DML?")
        assert s.query == "What is DML?"

    def test_mutable_state(self) -> None:
        """ResearchState is NOT frozen (LangGraph needs mutation)."""
        s = ResearchState(query="test")
        s.query = "changed"
        assert s.query == "changed"

    def test_discovered_methods_defaults_empty(self) -> None:
        """discovered_methods defaults to empty list."""
        s = ResearchState()
        assert s.discovered_methods == []

    def test_connection_explanations_defaults_empty(self) -> None:
        """connection_explanations defaults to empty list."""
        s = ResearchState()
        assert s.connection_explanations == []

    def test_serialization_roundtrip(self) -> None:
        """Full state survives serialization."""
        s = ResearchState(
            query="test",
            sub_tasks=[SubTask(description="t1")],
            report="# Report",
            discovered_methods=["DML", "IV"],
            connection_explanations=[{"concept_a": "A", "concept_b": "B"}],
        )
        data = s.model_dump()
        reconstructed = ResearchState(**data)
        assert reconstructed.query == s.query
        assert len(reconstructed.sub_tasks) == 1
        assert reconstructed.discovered_methods == ["DML", "IV"]
        assert len(reconstructed.connection_explanations) == 1
