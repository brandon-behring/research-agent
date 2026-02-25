"""State schema for the research analysis graph.

Pydantic v2 models with frozen immutability for sub-models.
ResearchState is mutable (LangGraph reconstructs via ``schema(**dict)``).
Each node returns a partial dict — LangGraph merges updates into state automatically.
"""

from __future__ import annotations

from typing import Any, TypedDict

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ── Immutable sub-models ─────────────────────────────────────────────


class SubTask(BaseModel):
    """A decomposed research sub-task from the query planner."""

    model_config = ConfigDict(frozen=True)

    description: str = Field(description="What to investigate")
    search_queries: list[str] = Field(
        default_factory=list,
        description="1-3 specific search queries for the knowledge base",
    )
    concepts_to_explore: list[str] = Field(
        default_factory=list,
        description="Key concepts to look up in the knowledge graph",
    )
    methods_to_audit: list[str] = Field(
        default_factory=list,
        description="Statistical methods whose assumptions should be checked",
    )

    @field_validator("search_queries", mode="before")
    @classmethod
    def strip_empty_queries(cls, v: list[str]) -> list[str]:
        """Remove empty or whitespace-only search queries."""
        return [q.strip() for q in v if q and q.strip()]


class SearchResult(BaseModel):
    """A single search result from research-kb."""

    model_config = ConfigDict(frozen=True)

    title: str = Field(description="Document title")
    content: str = Field(description="Matched content snippet")
    source_id: str = Field(description="Unique source identifier")
    score: float = Field(ge=0.0, le=1.0, description="Relevance score [0, 1]")
    authors: str = Field(default="", description="Author attribution line")
    year: str = Field(default="", description="Publication year")
    chunk_id: str = Field(default="", description="Chunk-level identifier")


class ConceptInfo(BaseModel):
    """A concept from the knowledge graph."""

    model_config = ConfigDict(frozen=True)

    concept_id: str = Field(description="UUID of the concept")
    name: str = Field(description="Human-readable concept name")
    concept_type: str = Field(default="", description="E.g., METHOD, ASSUMPTION, THEOREM")
    description: str = Field(default="", description="Concept description")
    relationships: list[dict[str, str]] = Field(
        default_factory=list,
        description="Edges: type + target_id",
    )
    neighborhood_summary: str = Field(
        default="",
        description="Graph neighborhood markdown from research-kb",
    )


class CitationInfo(BaseModel):
    """Citation network information for a source."""

    model_config = ConfigDict(frozen=True)

    source_id: str = Field(description="UUID of the analyzed source")
    source_title: str = Field(description="Title of the analyzed source")
    citing: list[dict[str, str]] = Field(
        default_factory=list,
        description="Papers that cite this source",
    )
    cited_by: list[dict[str, str]] = Field(
        default_factory=list,
        description="Papers cited by this source",
    )
    similar_papers: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Bibliographically similar papers",
    )


class AssumptionAudit(BaseModel):
    """Assumption audit result for a statistical method."""

    model_config = ConfigDict(frozen=True)

    method_name: str = Field(description="Statistical method name")
    assumptions: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Structured assumption details",
    )
    raw_output: str = Field(default="", description="Raw markdown from research-kb")


# ── Mutable graph state ──────────────────────────────────────────────


class ResearchState(BaseModel):
    """Complete state flowing through the research analysis graph.

    Design:
        - All fields have defaults -> graph can start at any node for testing
        - Mutable (no frozen) because LangGraph reconstructs via schema(**dict)
        - Typed for IDE support and validation
    """

    # --- Input ---
    query: str = Field(default="", description="Original research question")

    # --- Query Planner output ---
    sub_tasks: list[SubTask] = Field(default_factory=list)
    planning_rationale: str = ""

    # --- Literature Search output ---
    search_results: list[SearchResult] = Field(default_factory=list)
    search_summary: str = ""

    # --- Concept Explorer output ---
    concepts: list[ConceptInfo] = Field(default_factory=list)
    concept_map_summary: str = ""

    # --- Citation Analyzer output ---
    citations: list[CitationInfo] = Field(default_factory=list)
    citation_summary: str = ""

    # --- Assumption Auditor output ---
    assumption_audits: list[AssumptionAudit] = Field(default_factory=list)
    assumption_summary: str = ""

    # --- Synthesis output ---
    report: str = ""
    confidence_assessment: str = ""

    # --- Metadata ---
    current_node: str = ""


# ── Typed node return ────────────────────────────────────────────────


class NodeUpdate(TypedDict, total=False):
    """Typed dict for node return values. All fields optional (partial updates)."""

    sub_tasks: list[SubTask]
    planning_rationale: str
    search_results: list[SearchResult]
    search_summary: str
    concepts: list[ConceptInfo]
    concept_map_summary: str
    citations: list[CitationInfo]
    citation_summary: str
    assumption_audits: list[AssumptionAudit]
    assumption_summary: str
    report: str
    confidence_assessment: str
    current_node: str
