---
name: research-kb-desktop
description: research-kb's queryable KB code is on the desktop, not this laptop — deploy it there (pointed at ~/Claude/research_cache) before any primary-side KB work
metadata:
  node_type: memory
  type: project
---

research-kb's query/ingestion layer (Postgres + pgvector + Kuzu + FastMCP — the stack the design doc cites with `schema.sql` line numbers) is **not on this laptop**. Present here are only its local data dir (`~/Claude/research-kb/{inbox,composed}/`), the central 842-doc primary cache (`~/Claude/research_cache/`), and a RunPod GPU embedding spec (`runpod-deploy/examples/research-kb/pdf_embed_gpu.yaml`). The real KB lives on the **desktop**.

**Why:** the unified-KB build is primaries-first (user's call, 2026-05-27), so research-kb must be queryable before synthesis-kb evidence can resolve to primary sources.

**How to apply:** on the desktop, locate/clone + deploy research-kb and point its ingestion at `~/Claude/research_cache` (842 primary docs). Until then, all primary-side milestones (design-doc M0+) are blocked. Confirm its actual deployed schema + MCP tool surface there before finalizing ingestion code. Related: [[synthesis-kb-planned]], [[user-runpod-deploy]].
