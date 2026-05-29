# Session Handoff — 2026-05-29 (post cache→research-kb ingest)

> For the next session continuing the **deferred items**. Full session detail + decisions:
> `~/.claude/plans/check-out-remote-ancient-stonebraker.md` (not git-synced — laptop/desktop local).

## TL;DR
The **cache→research-kb ingest is DONE** (1,246 sources, 100% embedded, citation graph built).
research-kb is **live & mature on the DESKTOP**: 3,446 sources / 1.74M chunks / 36 domains.
New domains `agents` + `ml_security` are live & searchable. The #18 citation-network MCP bug is fixed.
Remaining work is the **KG/concept layer**, **dependabot**, and small housekeeping — all below.

## What was done this session (commits)
- research-agent `f662712` — reconcile memory/design to live KB reality (docs-only)
- research-kb `5c2278d` — `scripts/ingest_cache.py` cache-aware ingester (+1,246 src / +53,825 chunks)
- research-kb `5427669` — fix #18 citation formatters (dict access); issue #18 CLOSED
- All pushed to both remotes (github.com + github-dev).

## Live KB state (desktop, verified 2026-05-29)
- **3,446 sources / 1,736,651 chunks (100% embedded) / 94,169 citations / 36 domains.**
- New: `agents` (205 src / 2,625 chunks), `ml_security` (78 src / 645). Concepts still only in 4 domains
  (causal_inference, rag_llm, time_series, deep_learning); the other 32 incl. the 2 new = 0 concepts.
- Stack (docker): `research-kb-postgres` :5432, `research-kb-grobid` :8070, grafana, prometheus.
  Embed daemon: `BAAI/bge-large-en-v1.5` (1024-dim) at `/tmp/research_kb_embed.sock`.

## ⚠️ CRITICAL environment gotchas (cost real time to rediscover)
1. **GPU = 8 GB RTX 2070 SUPER.** Docling/`PDFDispatcher.ingest_pdf` **OOMs** while the embed daemon is
   resident. DO NOT use the dispatcher for batches. Use the TEI/trafilatura path (`ingest_cache.py`) +
   `scripts/embed_missing.py --batch 64` for embedding backfill (no Docling on GPU).
2. **MCP server caches imports** (stdio process). Enum/tool-code changes need a **`/mcp` reconnect** to
   take effect — the #18 fix + `SourceType.WEB` are committed but the *running* server needs a reconnect.
3. **`/var/tmp` is shared with the sandboxed Bash; `/tmp` is NOT** (and `/tmp` is tmpfs → wiped on reboot).
   Stage scratch to `/var/tmp`. The `gh` tool can't read files via `--body-file` (sandbox) → pipe via stdin.
4. **`chunks.domain_id` DEFAULTS to `causal_inference`** in `ChunkStore.batch_create` if omitted — always
   pass `domain_id`. `ingest_cache.py` does + runs a reconcile UPDATE as a safety net.
5. **Disk: root 96% full (~36 G free).** ~170 G reclaimable: `research-kb/fixtures` (114 G) + `backups`
   (56 G). **Prune before the KG phase** (concepts table is ~4.2 G for ~4 domains).
6. **arXiv staging** `/var/tmp/arxiv_pdf_staging/` (694 PDFs+TEIs) may be cleared on reboot; re-fetch via
   `/var/tmp/fetch_arxiv_569.py` (resumable, polite ≤1 req/3s) if the arxiv path needs re-running.

## The reusable ingester
`research-kb/scripts/ingest_cache.py --source {web,arxiv,journal,pdf,rest,all} [--dry-run] [--limit N]
[--skip-embedding] [--build-citations]`. Idempotent on `file_hash` (cache blob sha256). Reads
`~/Claude/research_cache` manifests; tags `source_class` + a 3-date/half-life temporal model on
`sources.metadata`; arxiv path reads the staged GROBID TEIs.

---

## DEFERRED ITEMS (next-session work, priority order)

### 1. KG / concept-extraction layer — the north-star "eventually", biggest
**Goal:** concepts + relationships → cross-domain bridges (the "connections-first" purpose). Only 4/36
domains have concepts; the new frontier domains (`agents`, `ml_security`) + 30 others have 0.
- **Tool:** `research-kb/scripts/extract_concepts.py --domain <D> --backend {ollama,anthropic}` (LLM-based;
  `packages/extraction/`). No cross-domain batch orchestration — needs a loop over domains.
- **Gates:** (a) DISK — prune fixtures/backups first; (b) embedding-tier/budget decision: Ollama
  llama3.1:8b @concurrency-1 ≈ **930 h CPU** (impractical) vs Anthropic Haiku ≈ **$117 / ~4 h** full corpus.
- **Suggested first pass:** the 2 new domains (`agents` 205 src + `ml_security` 78) — high-value frontier
  content, small enough to validate the KG pipeline + cost before the full 36-domain run.
- This is the synthesis-kb/KG goal — see `[[kb-north-star]]`, `[[synthesis-kb-planned]]`, and the locked
  design `docs/plans/active/2026-05-27_kb-system-design-all-domain.md`.

### 2. Phase C — dependabot (research-agent repo)
5 branches pushed, **0 PRs raised**: `langchain-anthropic` 0.3→1.4 (HIGH — `src/research_agent/llm.py`,
0→1 major), `tenacity` 8→9 (MED — `mcp_client.py` retry decorators); `pydantic`/`pytest-cov`/`ruff`
low-risk. Review the two majors' code paths; `gh pr create` per branch (or close). Investigate why no
PRs auto-raised (`.github/dependabot.yml` / GitHub App perms).

### 3. Housekeeping
- **`/mcp` reconnect** to activate the #18 fix + `SourceType.WEB` in the running server (one-time).
- **Two push-remotes** on both repos (`origin` → github.com + github-dev) — confirm intentional or drop one.
- **Laptop-only docs** absent from the corpus bootstrap: `~/Claude/research_corpus_2026-05-28_DESKTOP_HANDOFF.md`
  + `~/.claude/plans/dig-deep-and-understand-sparkling-cloud.md` — sync from the laptop if needed.

### 4. Quality follow-ups (optional)
- **Journal tier (161 ingested)** is mostly DOI **abstract pages**, not full text (tagged
  `source_class=journal`). They aid discovery but aren't full primaries — consider PDF-fetch for the
  high-value ones, or a quality re-pass.
- **arXiv citation enrichment** via OpenAlex/Semantic Scholar API (by arXiv ID) for richer edges beyond
  GROBID-extracted references.
- Deliberately **excluded** from ingest: 57 crossref-API JSON + 9 `example.com` placeholders (not documents).

## Key file pointers
- Ingester: `research-kb/scripts/ingest_cache.py` · backfill: `research-kb/scripts/embed_missing.py`
- arXiv fetch: `/var/tmp/fetch_arxiv_569.py` (resumable)
- Locked KB design: `research-agent/docs/plans/active/2026-05-27_kb-system-design-all-domain.md`
- Full session log: `~/.claude/plans/check-out-remote-ancient-stonebraker.md`
