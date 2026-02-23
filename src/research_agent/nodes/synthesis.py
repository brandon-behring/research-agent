"""Synthesis Writer node — produces the final structured research report.

Uses Sonnet (reasoning model) to synthesize all gathered evidence into
a coherent report with citations, confidence assessments, and identified gaps.
This is the most expensive LLM call — justified because final output quality matters.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from research_agent.config import AgentConfig
from research_agent.state import ResearchState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a research synthesis expert. Given the gathered evidence from a
multi-agent research system, produce a structured report.

Your report must include:
1. **Executive Summary** — 2-3 sentence overview of findings
2. **Key Findings** — Main discoveries, each with supporting citations
3. **Concept Map** — How the key concepts relate to each other
4. **Citation Landscape** — Seminal papers, recent developments, citation chains
5. **Methodological Considerations** — Assumptions, limitations, verification approaches
6. **Gaps & Limitations** — What the KB didn't cover, where evidence is thin
7. **Confidence Assessment** — How reliable are these findings? (high/medium/low with reasoning)

Citation format: [Author (Year)] with source IDs in footnotes.
Be honest about gaps — if few results were found, say so clearly.
Do NOT hallucinate papers or results not in the evidence provided."""


def _build_evidence_context(state: ResearchState) -> str:
    """Compile all gathered evidence into a single context string.

    Args:
        state: Current state with all agent outputs.

    Returns:
        Formatted evidence string for the synthesis LLM.
    """
    sections: list[str] = []

    # Original query and plan
    sections.append(f"## Research Question\n{state.query}")
    sections.append(f"## Planning Rationale\n{state.planning_rationale}")

    # Search results
    if state.search_results:
        search_lines = [f"## Literature Search ({len(state.search_results)} results)"]
        for i, r in enumerate(state.search_results[:15], 1):
            search_lines.append(
                f"\n### Result {i}: {r.title}\n"
                f"Authors: {r.authors}\nYear: {r.year}\n"
                f"Source ID: {r.source_id}\nScore: {r.score}\n\n{r.content}"
            )
        sections.append("\n".join(search_lines))

    # Concepts
    if state.concepts:
        concept_lines = [f"## Knowledge Graph Concepts ({len(state.concepts)})"]
        for c in state.concepts:
            if c.neighborhood_summary:
                concept_lines.append(f"\n### {c.name}\n{c.neighborhood_summary}")
            elif c.description:
                concept_lines.append(
                    f"\n### {c.name} ({c.concept_type})\n{c.description}"
                )
        sections.append("\n".join(concept_lines))

    # Citations
    if state.citations:
        cite_lines = [f"## Citation Networks ({len(state.citations)} sources analyzed)"]
        for c in state.citations:
            cite_lines.append(f"\n### {c.source_title}")
            if c.citing:
                cite_lines.append(f"Citing ({len(c.citing)}):")
                for ref in c.citing[:5]:
                    cite_lines.append(f"  - {ref.get('title', 'Unknown')} ({ref.get('year', '?')})")
            if c.cited_by:
                cite_lines.append(f"Cited by ({len(c.cited_by)}):")
                for ref in c.cited_by[:5]:
                    cite_lines.append(f"  - {ref.get('title', 'Unknown')} ({ref.get('year', '?')})")
            if c.similar_papers:
                cite_lines.append(f"Similar papers ({len(c.similar_papers)}):")
                for p in c.similar_papers[:5]:
                    cite_lines.append(
                        f"  - {p.get('title', 'Unknown')} "
                        f"({p.get('coupling_pct', 0):.0f}% overlap)"
                    )
        sections.append("\n".join(cite_lines))

    # Assumptions
    if state.assumption_audits:
        audit_lines = [f"## Assumption Audits ({len(state.assumption_audits)} methods)"]
        for a in state.assumption_audits:
            audit_lines.append(f"\n### {a.method_name}\n{a.raw_output}")
        sections.append("\n".join(audit_lines))

    return "\n\n---\n\n".join(sections)


async def synthesis_writer(state: ResearchState, config: AgentConfig) -> dict[str, Any]:
    """Synthesize all gathered evidence into a structured report.

    Args:
        state: Complete state with all agent outputs populated.
        config: Agent configuration (model selection).

    Returns:
        Dict with 'report' and 'confidence_assessment' updates.
    """
    logger.info("Synthesizing research report")

    evidence = _build_evidence_context(state)
    logger.info("Evidence context: %d chars", len(evidence))

    llm = ChatAnthropic(
        model=config.models.synthesis,
        max_tokens=4096,
        temperature=0.1,
    )

    response = await llm.ainvoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Synthesize the following research evidence:\n\n{evidence}"),
    ])

    content = response.content
    if isinstance(content, list):
        content = content[0].get("text", "") if content else ""

    # Extract confidence assessment from the report
    confidence = "See Confidence Assessment section in report."
    if "Confidence Assessment" in content:
        conf_idx = content.index("Confidence Assessment")
        confidence = content[conf_idx: conf_idx + 500]

    logger.info("Report generated: %d chars", len(content))

    return {
        "report": content,
        "confidence_assessment": confidence,
        "current_node": "synthesis_writer",
    }
