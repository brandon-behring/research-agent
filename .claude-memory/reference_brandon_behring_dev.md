---
name: reference-brandon-behring-dev
description: "brandon-behring.dev is the user's personal portfolio site (Astro v6 + Cloudflare Workers). Planned consumer of synthesis-kb graph export at /knowledge or /synthesis-map route. Tracking issue filed on the repo."
metadata:
  node_type: memory
  type: reference
  originSessionId: 1b78bb6c-1471-47e2-8fd6-4d21b96dec31
---

**Repo**: `~/Claude/brandon-behring.dev` — GitHub: `brandon-behring/brandon-behring.dev`

**Stack**: Astro v6 + MDX, deployed via Cloudflare Workers (`wrangler`). Static-first; no runtime DB.

**Planned synthesis-kb integration** (long-term, post-M6 of synthesis-kb build):
- New Astro route (proposed `/knowledge` or `/synthesis-map`)
- Client-side Cytoscape.js force-directed graph
- Build-time sync script pulls `synthesis_kb_export_graph` Cytoscape JSON → commits to repo → renders statically
- Privacy filter: `--public` flag emits only `visibility: public` dossiers (default private)
- No Cloudflare Worker proxy to synthesis-kb needed — keeps the site static

**Related**:
- Tracking issue: filed on `brandon-behring/brandon-behring.dev` (per plan §12.1, executed 2026-05-21)
- synthesis-kb design: [[synthesis-kb-planned]]
- Plan file: `~/.claude/plans/i-want-to-think-foamy-summit.md` §5.4
