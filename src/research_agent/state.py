"""State schema for the research analysis graph.

Uses TypedDict (LangGraph convention) with Annotated reducers for list fields.
Each node returns a partial dict — LangGraph merges updates into state automatically.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SubTask:
    """A decomposed research sub-task from the query planner."""

    description: str
    search_queries: list[str] = field(default_factory=list)
    concepts_to_explore: list[str] = field(default_factory=list)
    methods_to_audit: list[str] = field(default_factory=list)


@dataclass
class SearchResult:
    """A single search result from research-kb."""

    title: str
    content: str
    source_id: str
    score: float
    authors: str = ""
    year: str = ""
    chunk_id: str = ""


@dataclass
class ConceptInfo:
    """A concept from the knowledge graph."""

    concept_id: str
    name: str
    concept_type: str = ""
    description: str = ""
    relationships: list[dict[str, str]] = field(default_factory=list)
    neighborhood_summary: str = ""


@dataclass
class CitationInfo:
    """Citation network information for a source."""

    source_id: str
    source_title: str
    citing: list[dict[str, str]] = field(default_factory=list)
    cited_by: list[dict[str, str]] = field(default_factory=list)
    similar_papers: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class AssumptionAudit:
    """Assumption audit result for a statistical method."""

    method_name: str
    assumptions: list[dict[str, Any]] = field(default_factory=list)
    raw_output: str = ""


@dataclass
class ResearchState:
    """Complete state flowing through the research analysis graph.

    Design:
        - All fields have defaults → graph can start at any node for testing
        - Immutable update pattern → each node returns dict of changes
        - Typed for IDE support and validation
    """

    # --- Input ---
    query: str = ""

    # --- Query Planner output ---
    sub_tasks: list[SubTask] = field(default_factory=list)
    planning_rationale: str = ""

    # --- Literature Search output ---
    search_results: list[SearchResult] = field(default_factory=list)
    search_summary: str = ""

    # --- Concept Explorer output ---
    concepts: list[ConceptInfo] = field(default_factory=list)
    concept_map_summary: str = ""

    # --- Citation Analyzer output ---
    citations: list[CitationInfo] = field(default_factory=list)
    citation_summary: str = ""

    # --- Assumption Auditor output ---
    assumption_audits: list[AssumptionAudit] = field(default_factory=list)
    assumption_summary: str = ""

    # --- Synthesis output ---
    report: str = ""
    confidence_assessment: str = ""

    # --- Metadata ---
    errors: list[str] = field(default_factory=list)
    current_node: str = ""
