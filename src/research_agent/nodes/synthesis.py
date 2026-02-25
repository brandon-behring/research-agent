"""Synthesis Writer node -- produces the final structured research report.

Uses Sonnet (reasoning model) to synthesize all gathered evidence into
a coherent report with citations, confidence assessments, and identified gaps.
This is the most expensive LLM call -- justified because final output quality matters.

Uses ``with_structured_output()`` for guaranteed report structure.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from research_agent.config import AgentConfig
from research_agent.exceptions import SynthesisError
from research_agent.state import NodeUpdate, ResearchState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a research synthesis expert. Given the gathered evidence from a
multi-agent research system, produce a structured report.

Your report must include:
1. **Executive Summary** -- 2-3 sentence overview of findings
2. **Key Findings** -- Main discoveries, each with supporting citations
3. **Concept Map** -- How the key concepts relate to each other
4. **Citation Landscape** -- Seminal papers, recent developments, citation chains
5. **Methodological Considerations** -- Assumptions, limitations, verification approaches
6. **Gaps & Limitations** -- What the KB didn't cover, where evidence is thin
7. **Confidence Assessment** -- How reliable are these findings? (high/medium/low with reasoning)

Citation format: [Author (Year)] with source IDs in footnotes.
Be honest about gaps -- if few results were found, say so clearly.
Do NOT hallucinate papers or results not in the evidence provided."""


class SynthesisReport(BaseModel):
    """Structured synthesis report from the LLM."""

    executive_summary: str = Field(description="2-3 sentence overview of findings")
    key_findings: list[str] = Field(description="Main discoveries with supporting citations")
    concept_map: str = Field(description="How key concepts relate to each other")
    citation_landscape: str = Field(description="Seminal papers, recent developments, chains")
    methodological_considerations: str = Field(
        description="Assumptions, limitations, verification approaches"
    )
    gaps_limitations: str = Field(description="What the KB didn't cover, thin evidence areas")
    confidence_level: Literal["high", "medium", "low"] = Field(
        description="Overall reliability of findings"
    )
    confidence_reasoning: str = Field(description="Why this confidence level was assigned")

    def to_markdown(self) -> str:
        """Render the report as a markdown document."""
        findings = "\n".join(f"- {f}" for f in self.key_findings)
        return (
            f"# Research Report\n\n"
            f"## Executive Summary\n\n{self.executive_summary}\n\n"
            f"## Key Findings\n\n{findings}\n\n"
            f"## Concept Map\n\n{self.concept_map}\n\n"
            f"## Citation Landscape\n\n{self.citation_landscape}\n\n"
            f"## Methodological Considerations\n\n{self.methodological_considerations}\n\n"
            f"## Gaps & Limitations\n\n{self.gaps_limitations}\n\n"
            f"## Confidence Assessment\n\n"
            f"**Level**: {self.confidence_level}\n\n"
            f"{self.confidence_reasoning}\n"
        )


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
                concept_lines.append(f"\n### {c.name} ({c.concept_type})\n{c.description}")
        sections.append("\n".join(concept_lines))

    # Citations
    if state.citations:
        cite_lines = [f"## Citation Networks ({len(state.citations)} sources analyzed)"]
        for cit in state.citations:
            cite_lines.append(f"\n### {cit.source_title}")
            if cit.citing:
                cite_lines.append(f"Citing ({len(cit.citing)}):")
                for ref in cit.citing[:5]:
                    cite_lines.append(f"  - {ref.get('title', 'Unknown')} ({ref.get('year', '?')})")
            if cit.cited_by:
                cite_lines.append(f"Cited by ({len(cit.cited_by)}):")
                for ref in cit.cited_by[:5]:
                    cite_lines.append(f"  - {ref.get('title', 'Unknown')} ({ref.get('year', '?')})")
            if cit.similar_papers:
                cite_lines.append(f"Similar papers ({len(cit.similar_papers)}):")
                for p in cit.similar_papers[:5]:
                    cite_lines.append(
                        f"  - {p.get('title', 'Unknown')} ({p.get('coupling_pct', 0):.0f}% overlap)"
                    )
        sections.append("\n".join(cite_lines))

    # Assumptions
    if state.assumption_audits:
        audit_lines = [f"## Assumption Audits ({len(state.assumption_audits)} methods)"]
        for a in state.assumption_audits:
            audit_lines.append(f"\n### {a.method_name}\n{a.raw_output}")
        sections.append("\n".join(audit_lines))

    return "\n\n---\n\n".join(sections)


async def synthesis_writer(state: ResearchState, config: AgentConfig) -> NodeUpdate:
    """Synthesize all gathered evidence into a structured report.

    Args:
        state: Complete state with all agent outputs populated.
        config: Agent configuration (model selection).

    Returns:
        NodeUpdate with ``report`` and ``confidence_assessment``.

    Raises:
        SynthesisError: If synthesis fails after timeout or LLM error.
    """
    logger.info("Synthesizing research report")

    evidence = _build_evidence_context(state)
    logger.info("Evidence context: %d chars", len(evidence))

    llm = ChatAnthropic(
        model=config.models.synthesis,  # type: ignore[call-arg]
        max_tokens=4096,
        temperature=0.1,
    ).with_structured_output(SynthesisReport)

    timeout_s = config.synthesis_timeout
    try:
        async with asyncio.timeout(timeout_s):
            result: SynthesisReport = await llm.ainvoke(  # type: ignore[assignment]
                [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(
                        content=f"Synthesize the following research evidence:\n\n{evidence}"
                    ),
                ]
            )
    except TimeoutError as e:
        raise SynthesisError(f"Synthesis timed out after {timeout_s}s") from e
    except Exception as e:
        raise SynthesisError(f"Synthesis failed: {e}") from e

    report_md = result.to_markdown()
    confidence = f"**{result.confidence_level}**: {result.confidence_reasoning}"

    logger.info("Report generated: %d chars", len(report_md))

    return NodeUpdate(
        report=report_md,
        confidence_assessment=confidence,
        current_node="synthesis_writer",
    )
