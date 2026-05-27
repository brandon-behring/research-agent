---
name: synthesis-kb-planned
description: A new sibling repo to research-kb is planned (synthesis-kb) that indexes synthesized dossiers; research-agent is to be archived. Authoritative design doc lives at ~/.claude/plans/i-want-to-think-foamy-summit.md.
metadata:
  node_type: memory
  type: project
  originSessionId: 1b78bb6c-1471-47e2-8fd6-4d21b96dec31
---

**Status (2026-05-21)**: Architecture approved, not yet built. M0 begins next.

## Target four-repo state

| Repo | Layer | Role |
|------|-------|------|
| `research-kb` | Primary literature | Untouched — source of truth for what literature says |
| `synthesis-kb` (new) | Personal synthesis | Indexes dossiers (claims, evidence, bridges, extracted concepts); MCP + Streamlit dashboard; mirrors research-kb's stack (Postgres + pgvector + Kuzu + FastMCP) |
| `research_toolkit` | Dossier production | Builds dossiers; produces TWO outputs per project (cache → research-kb, dossier → synthesis-kb) |
| `research-agent` | Legacy | Archive — README pointer to synthesis-kb, no maintenance (see [[research-agent-archived]]) |

## Build sequence

M0 (cache export side) → M1 (synthesis-kb scaffold + ingest one dossier) → M2 (search + biblio coupling) → M3 (concept graph + bridges) → M4 (Streamlit dashboard) → M5 (consistency audit) → M6 (retire research-agent).

M1 is the discipline check: abandon cheaply if ingesting one dossier and querying it does not feel useful.

## Key design decisions

- **Epistemic separation enforced at storage layer** — see [[feedback-epistemic-separation]]
- **Files canonical, DB derived** — `claim_graph.jsonl`, `evidence_ledger.yml`, `bib_ledger.yml` are inputs; concepts, bridges, embeddings are DB-only derived data
- **Re-ingestion: upsert by stable IDs** + soft delete via `status='stale'`. Claim history lives in git on canonical files.
- **Three-identifier evidence model** — display (bibkey), truth (cache_id + sha256 + byte offset), retrieval (research-kb UUIDs, re-resolvable)
- **Wrapper command** `/dossier-publish <project>` chains the four-step ingestion pipeline

## Future-revisit items (§13 of plan)

Bibkey resolution strategy, dossier-audit findings flow, cross-pipeline dedup, concept-name disambiguation across domains, multi-machine sync, book/publication integration, MCP HTTP transport, prompt versioning, Tier-2/3 escalation paths. Each has a trigger criterion specified.

## brandon-behring.dev integration

Long-term goal: public synthesis-map at `brandon-behring.dev/knowledge` consuming `synthesis_kb_export_graph` output via build-time sync. Tracking issue filed on `brandon-behring/brandon-behring.dev`. See [[reference-brandon-behring-dev]].

## Authoritative reference

Full design at `~/.claude/plans/i-want-to-think-foamy-summit.md` — that file is the canonical source; this memory is the lookup pointer.
