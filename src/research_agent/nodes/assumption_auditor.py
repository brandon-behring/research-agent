"""Assumption Auditor node -- audits statistical method assumptions.

For each method identified by the planner, calls research-kb's assumption
auditing tool to get formal statements, violation consequences, and
verification approaches. Critical for causal inference research where
identification depends on assumptions.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from research_agent.config import AgentConfig
from research_agent.mcp_client import ResearchKBClient
from research_agent.state import AssumptionAudit, NodeUpdate, ResearchState

logger = logging.getLogger(__name__)

# Timeout for all assumption audit calls combined.
_AUDIT_TIMEOUT_SECONDS = 45


def _format_audit_for_synthesis(data: dict[str, Any]) -> str:
    """Format parsed JSON audit data into readable markdown for synthesis context.

    The ``synthesis.py:143`` feeds ``a.raw_output`` directly to the LLM,
    so this must produce well-structured, readable text.

    Args:
        data: Parsed JSON dict from research_kb_audit_assumptions.

    Returns:
        Formatted markdown string for synthesis consumption.
    """
    lines = [f"## Assumptions for: {data.get('method', 'Unknown')}"]

    for a in data.get("assumptions", []):
        lines.append(f"\n**{a.get('name', '?')}** [{a.get('importance', '')}]")
        if a.get("formal_statement"):
            lines.append(f"  - Formal: {a['formal_statement']}")
        if a.get("plain_english"):
            lines.append(f"  - Plain English: {a['plain_english']}")
        if a.get("violation_consequence"):
            lines.append(f"  - If violated: {a['violation_consequence']}")
        if a.get("verification_approaches"):
            approaches = ", ".join(a["verification_approaches"])
            lines.append(f"  - Verify: {approaches}")

    if data.get("code_docstring_snippet"):
        lines.append(
            f"\n### Code Docstring Snippet\n```python\n{data['code_docstring_snippet']}\n```"
        )

    return "\n".join(lines)


async def assumption_auditor(
    state: ResearchState,
    config: AgentConfig,
    mcp: ResearchKBClient,
) -> NodeUpdate:
    """Audit assumptions for all methods identified by the planner.

    Args:
        state: Current state with sub_tasks containing methods_to_audit.
        config: Agent configuration.
        mcp: Connected MCP client.

    Returns:
        NodeUpdate with ``assumption_audits`` and ``assumption_summary``.
    """
    logger.info("Starting assumption audits")

    audited_methods: set[str] = set()

    # Collect and deduplicate methods from sub-tasks + auto-discovered
    unique_methods: list[str] = []
    for task in state.sub_tasks:
        for method in task.methods_to_audit:
            method_lower = method.lower()
            if method_lower not in audited_methods:
                audited_methods.add(method_lower)
                unique_methods.append(method)
    # Merge auto-discovered methods from concept_explorer
    for method in state.discovered_methods:
        method_lower = method.lower()
        if method_lower not in audited_methods:
            audited_methods.add(method_lower)
            unique_methods.append(method)
            logger.info("Including auto-discovered method: %s", method)

    # Infer domain: if all sub-tasks share the same search_domain, use it
    domains = {t.search_domain for t in state.sub_tasks if t.search_domain}
    inferred_domain = domains.pop() if len(domains) == 1 else None
    audit_scope = "applied" if inferred_domain else "general"

    audits: list[AssumptionAudit] = []

    try:
        async with asyncio.timeout(_AUDIT_TIMEOUT_SECONDS):
            # Fan out all audit calls concurrently
            raw_results = await asyncio.gather(
                *[
                    mcp.audit_assumptions(
                        method_name=m,
                        include_docstring=True,
                        domain=inferred_domain,
                        scope=audit_scope,
                    )
                    for m in unique_methods
                ],
                return_exceptions=True,
            )

            for method, raw in zip(unique_methods, raw_results, strict=True):
                if isinstance(raw, BaseException):
                    logger.warning("Assumption audit failed for '%s': %s", method, raw)
                    audits.append(
                        AssumptionAudit(
                            method_name=method,
                            raw_output=f"Audit failed: {raw}",
                        )
                    )
                else:
                    assumptions: list[dict[str, Any]] = []
                    raw_output = raw
                    try:
                        parsed = json.loads(raw)
                        assumptions = parsed.get("assumptions", [])
                        raw_output = _format_audit_for_synthesis(parsed)
                    except (json.JSONDecodeError, KeyError, TypeError) as exc:
                        logger.warning("JSON parse failed for audit '%s': %s", method, exc)
                    audits.append(
                        AssumptionAudit(
                            method_name=method,
                            assumptions=assumptions,
                            raw_output=raw_output,
                        )
                    )
                    logger.info("Audited assumptions for: %s", method)

    except TimeoutError:
        logger.warning("Assumption auditing timed out with %d audits", len(audits))

    if not audits:
        summary = "No statistical methods identified for assumption auditing."
    else:
        summary = f"Audited {len(audits)} methods: {', '.join(audited_methods)}."

    logger.info(summary)

    return NodeUpdate(
        assumption_audits=audits,
        assumption_summary=summary,
        current_node="assumption_auditor",
    )
