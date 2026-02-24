"""LLM-as-judge grading functions for synthesis evaluation.

Uses Haiku (cheap) to grade Sonnet output -- cost-efficient eval pattern.
"""

from __future__ import annotations

import logging

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for research synthesis reports.
Given a research report and the evidence it was based on, grade the report on these dimensions:

1. **completeness** (1-5): Does the report cover the key aspects of the research question?
2. **grounding** (1-5): Are claims supported by the provided evidence? No hallucinated papers?
3. **gap_honesty** (1-5): Does the report honestly acknowledge limitations and gaps?
4. **coherence** (1-5): Is the report well-organized and logically structured?

Provide a brief justification for each score."""


class JudgeVerdict(BaseModel):
    """Structured grading output from the LLM judge."""

    completeness: int = Field(ge=1, le=5, description="Coverage of key aspects (1-5)")
    completeness_reason: str = Field(description="Justification for completeness score")
    grounding: int = Field(ge=1, le=5, description="Evidence support for claims (1-5)")
    grounding_reason: str = Field(description="Justification for grounding score")
    gap_honesty: int = Field(ge=1, le=5, description="Honest acknowledgment of gaps (1-5)")
    gap_honesty_reason: str = Field(description="Justification for gap_honesty score")
    coherence: int = Field(ge=1, le=5, description="Organization and logical flow (1-5)")
    coherence_reason: str = Field(description="Justification for coherence score")
    overall_assessment: str = Field(description="Brief overall assessment")

    @property
    def average_score(self) -> float:
        """Average across all 4 dimensions."""
        return (self.completeness + self.grounding + self.gap_honesty + self.coherence) / 4.0


async def grade_synthesis(
    report: str,
    evidence: str,
    model: str = "claude-haiku-4-5-20251001",
) -> JudgeVerdict:
    """Grade a synthesis report using LLM-as-judge.

    Args:
        report: The generated research report.
        evidence: The evidence context the report was based on.
        model: Model to use for grading (default: Haiku for cost efficiency).

    Returns:
        JudgeVerdict with scores and justifications.

    Raises:
        Exception: If the LLM call fails.
    """
    llm = ChatAnthropic(
        model=model,
        max_tokens=1024,
        temperature=0.0,
    ).with_structured_output(JudgeVerdict)

    prompt = (
        f"## Report to Grade\n\n{report}\n\n"
        f"## Evidence Provided\n\n{evidence[:5000]}\n\n"
        "Grade the report on completeness, grounding, gap_honesty, and coherence."
    )

    verdict: JudgeVerdict = await llm.ainvoke([
        SystemMessage(content=JUDGE_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])

    logger.info(
        "Judge scores: completeness=%d grounding=%d gap_honesty=%d coherence=%d (avg=%.1f)",
        verdict.completeness,
        verdict.grounding,
        verdict.gap_honesty,
        verdict.coherence,
        verdict.average_score,
    )

    return verdict
