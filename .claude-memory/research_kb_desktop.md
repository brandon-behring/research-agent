---
name: research-kb-desktop
description: research-kb's deployed instance is LIVE & mature on the DESKTOP (verified 2026-05-28: 2,200 sources / 1.68M chunks / 36 domains); the open primary-side work is closing the cache→KB ingestion gap (1,668 of 1,676 cache docs not yet in) + making the cache RAG-ready
metadata:
  node_type: memory
  type: project
---

research-kb's deployed instance runs on the **DESKTOP** and is **live & mature** (verified 2026-05-28 via MCP health/stats): **2,200 sources · 1,682,826 chunks · 311,591 concepts · 745,336 relationships · 55,503 citations · 36 domains** (concepts in only 4: causal_inference, rag_llm, time_series, deep_learning; the other 32 are vector/FTS-only). Stack running: `research-kb-postgres` (:5432), `research-kb-grobid` (:8070), grafana, prometheus. Code (Postgres + pgvector + Kuzu + FastMCP) is at `~/Claude/research-kb`; embedding = `BAAI/bge-large-en-v1.5` (1024-dim) via a SentenceTransformer daemon. *(Machine-explicit because this memory is git-synced: the DEPLOYED, queryable instance is on the desktop; the laptop (Mac) holds the corpus + toolkit.)*

**Why:** the deploy prerequisite the earlier note worried about is already satisfied — the primary layer is up and broadly populated (built from arXiv/textbook ingestion, NOT the laptop's research_cache). "Primaries-first" is no longer blocked on deployment; it is now about closing the cache→KB gap and making the cache ingestible.

**How to apply (open primary-side work, 2026-05-28):**
1. The laptop's `research_cache` is **near-disjoint** from the live KB — only 8 of 1,676 cache sha256s are in `sources.file_hash`; the other **1,668 are a real ingestion gap**.
2. The cache is **NOT RAG-ready**: 81% of `text/` is raw-HTML-polluted and arXiv entries are abstract *pages*, not papers. First step = make it ingestible — for arXiv, **fetch real PDFs (`arxiv.org/pdf/<id>`) → GROBID** (validated 2026-05-28: 5/5 papers, clean full-text + 610 refs, file_hash dedup confirmed via Mamba); for other web, re-extract clean prose.
3. `ingest_corpus.py` is a **hardcoded PDF manifest, not a cache reader** — a new `scripts/ingest_cache.py` is needed (reuse `SourceStore.create` / `ChunkStore.create` / `EmbeddingClient.embed`; cache sha256 = `file_hash` → idempotent, so the 8 present auto-skip).
4. Disk: root **96% full (~36 G free)** — cache ingest (~0.4 G) fits, but concept-extraction across 32 domains would need cleanup/expansion first.

Related: [[synthesis-kb-planned]], [[user-runpod-deploy]], [[kb-north-star]].
