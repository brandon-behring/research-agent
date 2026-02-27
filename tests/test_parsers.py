"""Tests for MCP response parsers (JSON-first with markdown fallback).

Tests cover:
    - JSON parsers: valid input, empty/missing fields, edge cases
    - Markdown parsers: backward compatibility (renamed with Markdown suffix)
    - Fallback behavior: JSON failure triggers markdown fallback
"""

from __future__ import annotations

import json

import pytest

from research_agent.nodes.citation_analyzer import (
    _parse_biblio_coupling,
    _parse_biblio_coupling_json,
    _parse_biblio_coupling_markdown,
    _parse_citation_network,
    _parse_citation_network_json,
    _parse_citation_network_markdown,
)
from research_agent.nodes.concept_explorer import (
    _build_neighborhood_summary,
    _parse_concept_detail,
    _parse_concept_detail_json,
    _parse_concept_detail_markdown,
)
from research_agent.nodes.literature_search import (
    _parse_search_results,
    _parse_search_results_json,
    _parse_search_results_markdown,
)

# ═══════════════════════════════════════════════════════════════════════
# Search Result Parsers
# ═══════════════════════════════════════════════════════════════════════


class TestSearchResultParserJSON:
    """Tests for _parse_search_results_json."""

    def test_valid_json(self) -> None:
        """Parses valid JSON search results correctly."""
        raw = json.dumps(
            {
                "results": [
                    {
                        "title": "Test Paper",
                        "content": "Content here",
                        "source_id": "src-001",
                        "authors": "Author (2024)",
                        "year": 2024,
                        "chunk_id": "chk-001",
                        "scores": {"combined": 0.85, "fts": 0.7, "vector": 0.9},
                    }
                ]
            }
        )
        results = _parse_search_results_json(raw)
        assert len(results) == 1
        assert results[0].title == "Test Paper"
        assert results[0].score == 0.85
        assert results[0].source_id == "src-001"
        assert results[0].year == "2024"  # int → str conversion

    def test_empty_results(self) -> None:
        """Empty results array returns empty list."""
        raw = json.dumps({"results": []})
        assert _parse_search_results_json(raw) == []

    def test_missing_fields_use_defaults(self) -> None:
        """Missing fields get default values."""
        raw = json.dumps({"results": [{"title": "Minimal"}]})
        results = _parse_search_results_json(raw)
        assert len(results) == 1
        assert results[0].title == "Minimal"
        assert results[0].content == ""
        assert results[0].source_id == ""
        assert results[0].score == 0.0
        assert results[0].year == ""

    def test_score_clamping_above_one(self) -> None:
        """Scores above 1.0 are clamped to 1.0."""
        raw = json.dumps({"results": [{"scores": {"combined": 1.5}}]})
        results = _parse_search_results_json(raw)
        assert results[0].score == 1.0

    def test_score_clamping_below_zero(self) -> None:
        """Negative scores are clamped to 0.0."""
        raw = json.dumps({"results": [{"scores": {"combined": -0.5}}]})
        results = _parse_search_results_json(raw)
        assert results[0].score == 0.0

    def test_scores_not_dict_defaults_to_zero(self) -> None:
        """Non-dict scores field defaults to 0.0."""
        raw = json.dumps({"results": [{"scores": 0.85}]})
        results = _parse_search_results_json(raw)
        assert results[0].score == 0.0

    def test_multiple_results(self) -> None:
        """Multiple results are all parsed."""
        raw = json.dumps(
            {
                "results": [
                    {"title": "A", "scores": {"combined": 0.9}},
                    {"title": "B", "scores": {"combined": 0.8}},
                    {"title": "C", "scores": {"combined": 0.7}},
                ]
            }
        )
        results = _parse_search_results_json(raw)
        assert len(results) == 3
        assert results[0].title == "A"
        assert results[2].title == "C"

    def test_invalid_json_raises(self) -> None:
        """Invalid JSON raises JSONDecodeError."""
        with pytest.raises(json.JSONDecodeError):
            _parse_search_results_json("not json")


class TestSearchResultParserMarkdown:
    """Tests for _parse_search_results_markdown (backward compat)."""

    def test_empty_string(self) -> None:
        """Empty input returns empty list."""
        assert _parse_search_results_markdown("") == []

    def test_whitespace_only(self) -> None:
        """Whitespace-only input returns empty list."""
        assert _parse_search_results_markdown("   \n\n  ") == []

    def test_no_result_headers(self) -> None:
        """Markdown without ### N. headers returns empty."""
        assert _parse_search_results_markdown("## Results\nSome text") == []

    def test_single_result(self) -> None:
        """Parses a single result correctly."""
        md = """### 1. Test Paper
Author (2024) [Paper]

> Content here

*Score: 0.85*
*Source ID: `src-001` | Chunk ID: `chk-001`*
"""
        results = _parse_search_results_markdown(md)
        assert len(results) == 1
        assert results[0].title == "Test Paper"
        assert results[0].score == 0.85
        assert results[0].source_id == "src-001"

    def test_missing_score(self) -> None:
        """Result without score gets default 0.0."""
        md = """### 1. No Score Paper
Author (2024) [Paper]

> Content

*Source ID: `src-001`*
"""
        results = _parse_search_results_markdown(md)
        assert len(results) == 1
        assert results[0].score == 0.0

    def test_missing_source_id(self) -> None:
        """Result without source_id gets empty string."""
        md = """### 1. No ID Paper
Author (2024) [Paper]

> Content

*Score: 0.5*
"""
        results = _parse_search_results_markdown(md)
        assert len(results) == 1
        assert results[0].source_id == ""

    @pytest.mark.parametrize(
        "score_str,expected",
        [
            ("0.0", 0.0),
            ("1.0", 1.0),
            ("0.5", 0.5),
            ("0.999", 0.999),
        ],
    )
    def test_valid_scores(self, score_str: str, expected: float) -> None:
        """Various valid score formats are parsed correctly."""
        md = f"""### 1. Paper
*Score: {score_str}*
*Source ID: `src-001`*
"""
        results = _parse_search_results_markdown(md)
        assert len(results) == 1
        assert abs(results[0].score - expected) < 0.001

    def test_multiple_results(self) -> None:
        """Multiple results are all parsed."""
        md = """### 1. Paper A
*Score: 0.9*
*Source ID: `src-a`*

### 2. Paper B
*Score: 0.8*
*Source ID: `src-b`*

### 3. Paper C
*Score: 0.7*
*Source ID: `src-c`*
"""
        results = _parse_search_results_markdown(md)
        assert len(results) == 3


class TestSearchResultParserFallback:
    """Tests for the unified _parse_search_results (JSON → markdown fallback)."""

    def test_empty_string(self) -> None:
        """Empty input returns empty list."""
        assert _parse_search_results("") == []

    def test_whitespace_only(self) -> None:
        """Whitespace returns empty list."""
        assert _parse_search_results("   \n\n  ") == []

    def test_json_input_succeeds(self) -> None:
        """Valid JSON is parsed via JSON path."""
        raw = json.dumps(
            {"results": [{"title": "Paper", "scores": {"combined": 0.9}, "source_id": "src-1"}]}
        )
        results = _parse_search_results(raw)
        assert len(results) == 1
        assert results[0].title == "Paper"

    def test_markdown_fallback_on_invalid_json(self) -> None:
        """Falls back to markdown parser when JSON fails."""
        md = """### 1. Fallback Paper
*Score: 0.75*
*Source ID: `src-fb`*
"""
        results = _parse_search_results(md)
        assert len(results) == 1
        assert results[0].title == "Fallback Paper"
        assert results[0].score == 0.75


# ═══════════════════════════════════════════════════════════════════════
# Citation Network Parsers
# ═══════════════════════════════════════════════════════════════════════


class TestCitationNetworkParserJSON:
    """Tests for _parse_citation_network_json."""

    def test_valid_json(self) -> None:
        """Parses both citing and cited_by correctly."""
        raw = json.dumps(
            {
                "citing": [
                    {"title": "Paper A", "year": 2024, "source_id": "src-a"},
                    {"title": "Paper B", "year": 2023, "source_id": "src-b"},
                ],
                "cited_by": [
                    {"title": "Paper C", "year": 2020, "source_id": "src-c"},
                ],
            }
        )
        citing, cited_by = _parse_citation_network_json(raw)
        assert len(citing) == 2
        assert len(cited_by) == 1
        assert citing[0]["title"] == "Paper A"
        assert citing[0]["year"] == "2024"  # int → str
        assert cited_by[0]["source_id"] == "src-c"

    def test_empty_arrays(self) -> None:
        """Empty citing/cited_by arrays return empty lists."""
        raw = json.dumps({"citing": [], "cited_by": []})
        citing, cited_by = _parse_citation_network_json(raw)
        assert citing == []
        assert cited_by == []

    def test_missing_arrays_default_empty(self) -> None:
        """Missing keys default to empty lists."""
        raw = json.dumps({})
        citing, cited_by = _parse_citation_network_json(raw)
        assert citing == []
        assert cited_by == []

    def test_invalid_json_raises(self) -> None:
        """Invalid JSON raises JSONDecodeError."""
        with pytest.raises(json.JSONDecodeError):
            _parse_citation_network_json("not json")


class TestCitationNetworkParserMarkdown:
    """Tests for _parse_citation_network_markdown (backward compat)."""

    def test_empty_string(self) -> None:
        """Empty input returns empty lists."""
        citing, cited_by = _parse_citation_network_markdown("")
        assert citing == []
        assert cited_by == []

    def test_whitespace_only(self) -> None:
        """Whitespace returns empty lists."""
        citing, cited_by = _parse_citation_network_markdown("  \n  ")
        assert citing == []
        assert cited_by == []

    def test_no_sections(self) -> None:
        """Markdown without section headers returns empty."""
        citing, cited_by = _parse_citation_network_markdown("## Citation Network\nSome text")
        assert citing == []
        assert cited_by == []

    def test_citing_section(self) -> None:
        """Parses citing section correctly."""
        md = """### Citing This Source (1)
- **Paper A** (2024)
  - ID: `src-a`
"""
        citing, cited_by = _parse_citation_network_markdown(md)
        assert len(citing) == 1
        assert citing[0]["title"] == "Paper A"
        assert citing[0]["year"] == "2024"

    def test_both_sections(self) -> None:
        """Parses both citing and cited-by."""
        md = """### Citing This Source (1)
- **Paper A** (2024)

### Cited By This Source (1)
- **Paper B** (2020)
"""
        citing, cited_by = _parse_citation_network_markdown(md)
        assert len(citing) == 1
        assert len(cited_by) == 1

    def test_multiline_extracts_source_id(self) -> None:
        """Extracts source_id from continuation lines (research-kb canonical format)."""
        md = """### Citing This Source (2)
*Papers that built on this work*

- **Debiased ML of CATEs** (2021)
  - Semenova and Chernozhukov
  - ID: `src-002-cate`

- **Automatic Debiased ML** (2022)
  - Chernozhukov et al.
  - ID: `src-005-auto`

### Cited By This Source (1)
*Foundations and context*

- **High-Dimensional Methods** (2014)
  - Belloni, Chernozhukov, Hansen
  - ID: `src-008-hdm`
"""
        citing, cited_by = _parse_citation_network_markdown(md)
        assert len(citing) == 2
        assert citing[0]["source_id"] == "src-002-cate"
        assert citing[1]["source_id"] == "src-005-auto"
        assert len(cited_by) == 1
        assert cited_by[0]["source_id"] == "src-008-hdm"

    def test_inline_id_still_works(self) -> None:
        """Inline ID on the title line still works (backward compat)."""
        md = """### Citing This Source (1)
- **Paper A** (2024) - ID: `src-inline`
"""
        citing, _ = _parse_citation_network_markdown(md)
        assert len(citing) == 1
        assert citing[0]["source_id"] == "src-inline"


class TestCitationNetworkParserFallback:
    """Tests for the unified _parse_citation_network fallback."""

    def test_empty_string(self) -> None:
        """Empty input returns empty lists."""
        citing, cited_by = _parse_citation_network("")
        assert citing == []
        assert cited_by == []

    def test_json_input_succeeds(self) -> None:
        """Valid JSON is parsed via JSON path."""
        raw = json.dumps(
            {
                "citing": [{"title": "A", "year": 2024, "source_id": "src-a"}],
                "cited_by": [],
            }
        )
        citing, cited_by = _parse_citation_network(raw)
        assert len(citing) == 1
        assert citing[0]["title"] == "A"

    def test_markdown_fallback(self) -> None:
        """Falls back to markdown when JSON fails."""
        md = """### Citing This Source (1)
- **Fallback Paper** (2024)
  - ID: `src-fb`
"""
        citing, _ = _parse_citation_network(md)
        assert len(citing) == 1
        assert citing[0]["title"] == "Fallback Paper"


# ═══════════════════════════════════════════════════════════════════════
# Bibliographic Coupling Parsers
# ═══════════════════════════════════════════════════════════════════════


class TestBiblioCouplingParserJSON:
    """Tests for _parse_biblio_coupling_json."""

    def test_valid_json(self) -> None:
        """Parses coupling_strength → coupling_pct conversion correctly."""
        raw = json.dumps(
            {
                "similar": [
                    {
                        "title": "Paper A",
                        "year": 2021,
                        "source_id": "src-a",
                        "coupling_strength": 0.452,
                        "shared_references": 8,
                    }
                ]
            }
        )
        papers = _parse_biblio_coupling_json(raw)
        assert len(papers) == 1
        assert papers[0]["title"] == "Paper A"
        assert papers[0]["year"] == "2021"  # int → str
        assert abs(papers[0]["coupling_pct"] - 45.2) < 0.01  # 0.452 * 100
        assert papers[0]["shared_references"] == 8
        assert papers[0]["source_id"] == "src-a"

    def test_empty_similar(self) -> None:
        """Empty similar array returns empty list."""
        raw = json.dumps({"similar": []})
        assert _parse_biblio_coupling_json(raw) == []

    def test_missing_optional_fields(self) -> None:
        """Only title/year present when optional fields absent."""
        raw = json.dumps({"similar": [{"title": "Minimal", "year": 2020}]})
        papers = _parse_biblio_coupling_json(raw)
        assert len(papers) == 1
        assert papers[0]["title"] == "Minimal"
        assert "coupling_pct" not in papers[0]
        assert "source_id" not in papers[0]

    def test_zero_coupling_strength(self) -> None:
        """Zero coupling strength produces 0.0 coupling_pct."""
        raw = json.dumps({"similar": [{"title": "X", "coupling_strength": 0.0}]})
        papers = _parse_biblio_coupling_json(raw)
        assert papers[0]["coupling_pct"] == 0.0

    def test_invalid_json_raises(self) -> None:
        """Invalid JSON raises JSONDecodeError."""
        with pytest.raises(json.JSONDecodeError):
            _parse_biblio_coupling_json("not json")


class TestBiblioCouplingParserMarkdown:
    """Tests for _parse_biblio_coupling_markdown (backward compat)."""

    def test_empty_string(self) -> None:
        """Empty input returns empty list."""
        assert _parse_biblio_coupling_markdown("") == []

    def test_whitespace_only(self) -> None:
        """Whitespace returns empty list."""
        assert _parse_biblio_coupling_markdown("  \n  ") == []

    def test_parses_inline_coupling_scores(self) -> None:
        """Inline coupling and ID on the title line still works."""
        md = "- **Paper A** (2024) - Coupling: **45.2%** (8 shared refs) - ID: `src-a`\n"
        papers = _parse_biblio_coupling_markdown(md)
        assert len(papers) == 1
        assert papers[0]["coupling_pct"] == 45.2
        assert papers[0]["source_id"] == "src-a"

    def test_parses_multiline_without_coupling(self) -> None:
        """Multi-line entries still extract title/year."""
        md = """- **Paper B** (2023)
  - Some other info
"""
        papers = _parse_biblio_coupling_markdown(md)
        assert len(papers) == 1
        assert papers[0]["title"] == "Paper B"
        assert papers[0]["year"] == "2023"

    def test_multiline_extracts_coupling_and_id(self) -> None:
        """Extracts coupling_pct and source_id from continuation lines."""
        md = (
            "- **Debiased ML of CATEs** (2021)\n"
            "  - Semenova and Chernozhukov\n"
            "  - Coupling: **45.2%** (8 shared refs)\n"
            "  - ID: `src-002-cate`\n"
            "\n"
            "- **Automatic Debiased ML** (2022)\n"
            "  - Chernozhukov et al.\n"
            "  - Coupling: **38.7%** (6 shared refs)\n"
            "  - ID: `src-005-auto`\n"
        )
        papers = _parse_biblio_coupling_markdown(md)
        assert len(papers) == 2
        assert papers[0]["coupling_pct"] == 45.2
        assert papers[0]["source_id"] == "src-002-cate"
        assert papers[1]["coupling_pct"] == 38.7
        assert papers[1]["source_id"] == "src-005-auto"


class TestBiblioCouplingParserFallback:
    """Tests for the unified _parse_biblio_coupling fallback."""

    def test_empty_string(self) -> None:
        """Empty input returns empty list."""
        assert _parse_biblio_coupling("") == []

    def test_json_input_succeeds(self) -> None:
        """Valid JSON is parsed via JSON path."""
        raw = json.dumps(
            {"similar": [{"title": "A", "coupling_strength": 0.5, "source_id": "src-a"}]}
        )
        papers = _parse_biblio_coupling(raw)
        assert len(papers) == 1
        assert abs(papers[0]["coupling_pct"] - 50.0) < 0.01

    def test_markdown_fallback(self) -> None:
        """Falls back to markdown when JSON fails."""
        md = "- **Fallback** (2024) - Coupling: **30.0%** - ID: `src-fb`\n"
        papers = _parse_biblio_coupling(md)
        assert len(papers) == 1
        assert papers[0]["coupling_pct"] == 30.0


# ═══════════════════════════════════════════════════════════════════════
# Concept Detail Parsers
# ═══════════════════════════════════════════════════════════════════════


class TestConceptDetailParserJSON:
    """Tests for _parse_concept_detail_json."""

    def test_valid_json(self) -> None:
        """Parses full concept detail correctly."""
        raw = json.dumps(
            {
                "concept_id": "concept-dml-001",
                "name": "Double Machine Learning",
                "concept_type": "METHOD",
                "definition": "A framework for estimating treatment effects.",
                "relationships": [
                    {"type": "REQUIRES", "target_id": "concept-unconf-001"},
                    {"type": "USES", "target_id": "concept-crossfit-001"},
                ],
            }
        )
        result = _parse_concept_detail_json(raw)
        assert result is not None
        assert result.concept_id == "concept-dml-001"
        assert result.name == "Double Machine Learning"
        assert result.concept_type == "METHOD"
        assert result.description == "A framework for estimating treatment effects."
        assert len(result.relationships) == 2

    def test_definition_maps_to_description(self) -> None:
        """JSON key 'definition' maps to ConceptInfo.description."""
        raw = json.dumps({"definition": "The description text."})
        result = _parse_concept_detail_json(raw)
        assert result is not None
        assert result.description == "The description text."

    def test_missing_fields_use_defaults(self) -> None:
        """Missing fields get defaults."""
        raw = json.dumps({"name": "Minimal"})
        result = _parse_concept_detail_json(raw)
        assert result is not None
        assert result.name == "Minimal"
        assert result.concept_id == ""
        assert result.description == ""
        assert result.relationships == []

    def test_invalid_json_raises(self) -> None:
        """Invalid JSON raises JSONDecodeError."""
        with pytest.raises(json.JSONDecodeError):
            _parse_concept_detail_json("not json")


class TestConceptDetailParserMarkdown:
    """Tests for _parse_concept_detail_markdown (backward compat)."""

    def test_empty_string(self) -> None:
        """Empty input returns None."""
        assert _parse_concept_detail_markdown("") is None

    def test_whitespace_only(self) -> None:
        """Whitespace returns None."""
        assert _parse_concept_detail_markdown("  \n  ") is None

    def test_minimal_concept(self) -> None:
        """Parses minimal concept with just name and ID."""
        md = "## DML\n*Type: METHOD | ID: `concept-dml-001`*\n"
        result = _parse_concept_detail_markdown(md)
        assert result is not None
        assert result.name == "DML"
        assert result.concept_id == "concept-dml-001"

    def test_full_concept(self) -> None:
        """Parses complete concept with all fields."""
        md = """## Double Machine Learning
**Type:** METHOD
**ID:** `concept-dml-001`

### Description
A framework for estimating treatment effects.

### Relationships (2 total)
- REQUIRES \u2192 `concept-unconf-001`
- USES \u2192 `concept-crossfit-001`
"""
        result = _parse_concept_detail_markdown(md)
        assert result is not None
        assert result.name == "Double Machine Learning"
        assert result.concept_type == "METHOD"
        assert "treatment effects" in result.description
        assert len(result.relationships) == 2

    def test_no_id_or_name_returns_none(self) -> None:
        """Returns None if neither ID nor name can be extracted."""
        assert _parse_concept_detail_markdown("Some random text") is None


class TestConceptDetailParserFallback:
    """Tests for the unified _parse_concept_detail fallback."""

    def test_empty_string(self) -> None:
        """Empty input returns None."""
        assert _parse_concept_detail("") is None

    def test_json_input_succeeds(self) -> None:
        """Valid JSON is parsed via JSON path."""
        raw = json.dumps(
            {
                "concept_id": "cid-1",
                "name": "Test",
                "concept_type": "METHOD",
                "definition": "A test concept.",
            }
        )
        result = _parse_concept_detail(raw)
        assert result is not None
        assert result.name == "Test"

    def test_markdown_fallback(self) -> None:
        """Falls back to markdown when JSON fails."""
        md = "## Fallback Concept\n**ID:** `cid-fb`\n"
        result = _parse_concept_detail(md)
        assert result is not None
        assert result.name == "Fallback Concept"


# ═══════════════════════════════════════════════════════════════════════
# Neighborhood Summary Builder
# ═══════════════════════════════════════════════════════════════════════


class TestBuildNeighborhoodSummary:
    """Tests for _build_neighborhood_summary."""

    def test_full_data(self) -> None:
        """Builds formatted summary with all fields."""
        data = {
            "center": {"id": "cid-1", "name": "DML", "type": "METHOD"},
            "nodes": [
                {"name": "Unconfoundedness", "type": "ASSUMPTION"},
                {"name": "Cross-fitting", "type": "METHOD"},
            ],
            "edges": [{"source": "a", "target": "b", "type": "REQUIRES"}],
            "relationship_type_counts": {"REQUIRES": 1},
        }
        summary = _build_neighborhood_summary(data)
        assert "## Graph Neighborhood: DML" in summary
        assert "ID: `cid-1`" in summary
        assert "2 connected concepts, 1 relationships" in summary
        assert "- Unconfoundedness [ASSUMPTION]" in summary
        assert "- REQUIRES: 1" in summary

    def test_empty_data(self) -> None:
        """Handles empty/minimal data gracefully."""
        data: dict = {"center": {}, "nodes": [], "edges": []}
        summary = _build_neighborhood_summary(data)
        assert "## Graph Neighborhood: Unknown" in summary
        assert "0 connected concepts" in summary

    def test_no_type_counts(self) -> None:
        """Omits relationships section when type_counts is empty."""
        data = {
            "center": {"name": "X"},
            "nodes": [{"name": "Y", "type": "T"}],
            "edges": [],
        }
        summary = _build_neighborhood_summary(data)
        assert "### Relationships" not in summary
