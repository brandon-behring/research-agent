---
name: user-runpod-deploy
description: User maintains the runpod-deploy PyPI package — config-driven RunPod GPU orchestration. Candidate Tier-2 escalation path for synthesis-kb extractor / consistency audit when local Ollama quality is insufficient and cost estimates justify GPU pods.
metadata:
  node_type: memory
  type: user
  originSessionId: 1b78bb6c-1471-47e2-8fd6-4d21b96dec31
---

User owns and maintains `runpod-deploy` (`~/Claude/runpod-deploy`, published on PyPI), a config-driven RunPod GPU orchestration package. Consumer repos own their job configs and project commands; runpod-deploy owns the RunPod mechanics.

**Relevance for synthesis-kb**: when concept extraction (Q9.3) or consistency audit (Q9.1) requires more capacity than local Ollama llama3.1:8b provides, runpod-deploy is the natural escalation path — user-owned tooling, clean integration.

**When to suggest runpod-deploy**:
- Concept extraction quality on synthesized text is insufficient (Tier-1 → Tier-2)
- Consistency audit LLM-call volume exceeds Anthropic API budget
- Bridge precomputation on a larger corpus needs GPU acceleration

**When NOT to suggest**:
- Small workloads where local Ollama or Claude API is cheaper end-to-end
- Latency-sensitive interactive queries (pod spin-up time matters)

Reference: README at `~/Claude/runpod-deploy/README.md`. Smoke example at `examples/smoke/a4000_smoke.yaml`.
