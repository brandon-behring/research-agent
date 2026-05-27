---
name: research-agent-archived
description: research-agent (LangGraph 7-node pipeline) is to be archived under the synthesis-kb migration. It was never used in real work; capabilities re-home as synthesis-kb MCP tools.
metadata:
  node_type: memory
  type: project
  originSessionId: 1b78bb6c-1471-47e2-8fd6-4d21b96dec31
---

**Status (2026-05-21)**: Slated for archival at M6 of the synthesis-kb build (see [[synthesis-kb-planned]]).

## Decision

Archive (not delete, not repurpose, not keep-as-is). Reasons:
- 4,811 LOC + 451 tests is real intellectual artifact, useful even unmaintained — parallel-fan-out pattern and per-node resilience worth preserving as documented prior art
- Deletion is irreversible
- Repurposing as a thin orchestrator risks a half-finished system
- Keep-as-is means maintaining a system the user does not use

## Migration map for research-agent's nodes

| node | new home |
|------|----------|
| `query_planner` | research_toolkit's `/research-plan` (exists) |
| `literature_search` | research-kb's `research_kb_search` (exists) |
| `concept_explorer` | `synthesis_kb_concept_neighborhood` + `synthesis_kb_find_similar_concepts` |
| `citation_analyzer` | research-kb's `research_kb_citation_network` + `synthesis_kb_biblio_coupling` |
| `assumption_auditor` | research-kb's `research_kb_audit_assumptions` + new synthesis-side audit |
| `connection_explorer` | `synthesis_kb_explain_bridge` |
| `synthesis` | research_toolkit's `/research-synthesize` (to be added) |

## Why it was unused

LangGraph orchestration (parallel fan-out, conditional routing, streaming) was designed for unattended-batch research. The actual use case turned out to be **interactive, modular, audit-heavy research** — better served by skills than by sealed pipelines. The architecture matched the intended problem but not the actual problem.

## On archival

At M6: add one-line README pointing to synthesis-kb. No code changes. Tests remain runnable for reference.
