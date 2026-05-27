---
name: feedback-epistemic-separation
description: User requires strong epistemic separation between primary-literature RAG (research-kb) and synthesis RAG (synthesis-kb). Never conflate primary sources with synthesized claims in retrieval or storage.
metadata:
  node_type: memory
  type: feedback
  originSessionId: 1b78bb6c-1471-47e2-8fd6-4d21b96dec31
---

User requires the storage and retrieval layers to enforce a sharp distinction between **matters of fact** (primary literature in `research-kb`) and **matters of synthesis** (user's own claims in the planned `synthesis-kb`).

**Why**: A search for "what Bertsekas said about policy gradients" must not return the user's dossier's claim about Bertsekas. The citation direction is asymmetric: dossiers cite primaries, never vice versa. Mixing them at the storage layer corrupts both ranking and provenance.

**How to apply**:
- Never propose extending research-kb's `sources` table with `source_type='dossier'`. The user rejected this explicitly.
- Synthesis content goes in `synthesis-kb` (separate repo, separate Postgres, separate MCP server).
- Cross-KB references travel through explicit URIs / FKs, not through shared tables.
- When designing queries or schemas that touch synthesized content, ask: "could a primary-literature search ever surface this row?" If yes, the design violates the separation.

Related: [[synthesis-kb-planned]] enforces this at the storage layer.
