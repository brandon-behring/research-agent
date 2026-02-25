"""Tests for MCP markdown parsers.

Parametrized tests for malformed input: empty strings, missing headers,
partial results, non-numeric scores. Validates defensive parsing behavior.
"""

from __future__ import annotations

import pytest

from research_agent.nodes.citation_analyzer import _parse_biblio_coupling, _parse_citation_network
from research_agent.nodes.concept_explorer import _parse_concept_detail
from research_agent.nodes.literature_search import _parse_search_results


class TestSearchResultParser:
    """Tests for _parse_search_results."""

    def test_empty_string(self) -> None:
        """Empty input returns empty list."""
        assert _parse_search_results("") == []

    def test_whitespace_only(self) -> None:
        """Whitespace-only input returns empty list."""
        assert _parse_search_results("   \n\n  ") == []

    def test_no_result_headers(self) -> None:
        """Markdown without ### N. headers returns empty."""
        assert _parse_search_results("## Results\nSome text") == []

    def test_single_result(self) -> None:
        """Parses a single result correctly."""
        md = """### 1. Test Paper
Author (2024) [Paper]

> Content here

*Score: 0.85*
*Source ID: `src-001` | Chunk ID: `chk-001`*
"""
        results = _parse_search_results(md)
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
        results = _parse_search_results(md)
        assert len(results) == 1
        assert results[0].score == 0.0

    def test_missing_source_id(self) -> None:
        """Result without source_id gets empty string."""
        md = """### 1. No ID Paper
Author (2024) [Paper]

> Content

*Score: 0.5*
"""
        results = _parse_search_results(md)
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
        results = _parse_search_results(md)
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
        results = _parse_search_results(md)
        assert len(results) == 3


class TestCitationNetworkParser:
    """Tests for _parse_citation_network."""

    def test_empty_string(self) -> None:
        """Empty input returns empty lists."""
        citing, cited_by = _parse_citation_network("")
        assert citing == []
        assert cited_by == []

    def test_whitespace_only(self) -> None:
        """Whitespace returns empty lists."""
        citing, cited_by = _parse_citation_network("  \n  ")
        assert citing == []
        assert cited_by == []

    def test_no_sections(self) -> None:
        """Markdown without section headers returns empty."""
        citing, cited_by = _parse_citation_network("## Citation Network\nSome text")
        assert citing == []
        assert cited_by == []

    def test_citing_section(self) -> None:
        """Parses citing section correctly."""
        md = """### Citing This Source (1)
- **Paper A** (2024)
  - ID: `src-a`
"""
        citing, cited_by = _parse_citation_network(md)
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
        citing, cited_by = _parse_citation_network(md)
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
        citing, cited_by = _parse_citation_network(md)
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
        citing, _ = _parse_citation_network(md)
        assert len(citing) == 1
        assert citing[0]["source_id"] == "src-inline"


class TestBiblioCouplingParser:
    """Tests for _parse_biblio_coupling."""

    def test_empty_string(self) -> None:
        """Empty input returns empty list."""
        assert _parse_biblio_coupling("") == []

    def test_whitespace_only(self) -> None:
        """Whitespace returns empty list."""
        assert _parse_biblio_coupling("  \n  ") == []

    def test_parses_inline_coupling_scores(self) -> None:
        """Inline coupling and ID on the title line still works."""
        md = "- **Paper A** (2024) - Coupling: **45.2%** (8 shared refs) - ID: `src-a`\n"
        papers = _parse_biblio_coupling(md)
        assert len(papers) == 1
        assert papers[0]["coupling_pct"] == 45.2
        assert papers[0]["source_id"] == "src-a"

    def test_parses_multiline_without_coupling(self) -> None:
        """Multi-line entries still extract title/year."""
        md = """- **Paper B** (2023)
  - Some other info
"""
        papers = _parse_biblio_coupling(md)
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
        papers = _parse_biblio_coupling(md)
        assert len(papers) == 2
        assert papers[0]["coupling_pct"] == 45.2
        assert papers[0]["source_id"] == "src-002-cate"
        assert papers[1]["coupling_pct"] == 38.7
        assert papers[1]["source_id"] == "src-005-auto"


class TestConceptDetailParser:
    """Tests for _parse_concept_detail."""

    def test_empty_string(self) -> None:
        """Empty input returns None."""
        assert _parse_concept_detail("") is None

    def test_whitespace_only(self) -> None:
        """Whitespace returns None."""
        assert _parse_concept_detail("  \n  ") is None

    def test_minimal_concept(self) -> None:
        """Parses minimal concept with just name and ID."""
        md = "## DML\n*Type: METHOD | ID: `concept-dml-001`*\n"
        result = _parse_concept_detail(md)
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
        result = _parse_concept_detail(md)
        assert result is not None
        assert result.name == "Double Machine Learning"
        assert result.concept_type == "METHOD"
        assert "treatment effects" in result.description
        assert len(result.relationships) == 2

    def test_no_id_or_name_returns_none(self) -> None:
        """Returns None if neither ID nor name can be extracted."""
        assert _parse_concept_detail("Some random text") is None
