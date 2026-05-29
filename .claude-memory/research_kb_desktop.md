---
name: research-kb-desktop
description: research-kb is LIVE & mature on the DESKTOP and the cache→KB ingest is DONE (2026-05-29: 3,446 sources / 1.74M chunks / 36 domains, agents+ml_security now live). Next primary-side work = the KG/concept layer. Handoff doc names the gotchas.
metadata:
  node_type: memory
  type: project
---

research-kb's deployed instance runs on the **DESKTOP** and is **live & mature**. Verified 2026-05-29 via MCP: **3,446 sources · 1,736,651 chunks (100% embedded) · 94,169 citations · 36 domains**. Stack (docker): `research-kb-postgres` :5432, `research-kb-grobid` :8070, grafana, prometheus; embed daemon `BAAI/bge-large-en-v1.5` (1024-dim) at `/tmp/research_kb_embed.sock`. Code at `~/Claude/research-kb`. *(Machine-explicit, git-synced: deployed instance on desktop; laptop (Mac) holds corpus + toolkit.)*

**Cache→KB ingest: DONE (2026-05-29).** The laptop's `research_cache` (near-disjoint from the prior corpus) was ingested via the new `scripts/ingest_cache.py`: **1,246 sources / 53,825 chunks** — arXiv (633, via real-PDF fetch→GROBID TEI, since cached `/abs` HTML is abstract-only), journal HTML (161, trafilatura), cache PDFs (61, GROBID), frontier web (391, tagged `source_class=web` + short half-life). New domains **`agents` (205 src)** + **`ml_security` (78)** are live & searchable. Commits: research-kb `5c2278d` (ingester) + `5427669` (#18 fix). The #18 `citation_network` MCP bug (formatters did attr-access on dict items) is fixed.

**Next primary-side work = the KG / concept-extraction layer** (the "eventually" north-star — cross-domain bridges). Only 4/36 domains have concepts; the 2 new + 30 others = 0. Gated on DISK (root 96% full; prune `research-kb/fixtures` 114G + `backups` 56G first) + an embedding/LLM-tier budget call (Anthropic Haiku ≈ $117/~4h full corpus vs Ollama ≈ 930h CPU). Suggested first pass: the 2 new frontier domains.

**Gotchas (full list in the handoff doc):** 8 GB GPU → Docling/dispatcher OOMs, use the TEI/trafilatura path + `embed_missing.py`; MCP server caches imports → `/mcp` reconnect after enum/tool changes; `chunks.domain_id` defaults to causal_inference if unset; `/var/tmp` shared with sandboxed shell, `/tmp` not.

**Full handoff:** `research-agent/docs/plans/active/2026-05-29_session-handoff.md`. Related: [[kb-north-star]], [[synthesis-kb-planned]], [[user-runpod-deploy]].
