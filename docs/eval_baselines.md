# Eval Baselines

LLM-as-judge evaluation scores for the synthesis pipeline. In production, Haiku judges Sonnet output. Eval tests use Haiku for both synthesis and judging (cost optimization, ~$0.10 per full run).

## Golden Cases

Three golden cases test different research question types:

| Case | Query | What it tests |
|------|-------|---------------|
| `dml_assumptions` | "What are the assumptions of double machine learning?" | Single-method deep dive, assumption extraction |
| `iv_vs_rdd` | "Compare instrumental variables and regression discontinuity designs" | Multi-method comparison, tradeoff analysis |
| `causal_forests` | "How do causal forests estimate heterogeneous treatment effects?" | Algorithm explanation, concept graph traversal |

## Baseline Scores

Scores are 1-5 per dimension, graded by Haiku judge. Eval uses Haiku for synthesis (cost optimization); production scores may differ with Sonnet synthesis.

| Case | Completeness | Grounding | Gap Honesty | Coherence | Avg |
|------|-------------|-----------|-------------|-----------|-----|
| dml_assumptions | 3 | 4 | 5 | 4 | 4.00 |
| iv_vs_rdd | 4 | 5 | 5 | 4 | 4.50 |
| causal_forests | 4 | 5 | 5 | 5 | 4.75 |

**Recorded**: 2026-02-25 (Haiku synthesis + Haiku judge). All 3 cases pass minimum thresholds (>=3 on all dimensions).

## Reproduction

```bash
# Prerequisites: research-kb running locally, API key set
export RESEARCH_KB_PATH=~/Claude/research-kb
export ANTHROPIC_API_KEY=sk-...

# Run all 3 golden cases (~2 min, ~$0.10)
pytest evals/test_synthesis_eval.py -m eval --timeout=120 -v

# Run single case (e.g., DML)
pytest "evals/test_synthesis_eval.py::test_synthesis_with_judge[dml_assumptions]" -m eval --timeout=120 -v
```

## Scoring Dimensions

| Dimension | What it measures | Minimum threshold |
|-----------|-----------------|-------------------|
| **Completeness** | Covers key aspects of the research question | 3/5 |
| **Grounding** | Claims supported by provided evidence, no hallucinated papers | 3/5 |
| **Gap Honesty** | Honestly acknowledges limitations and gaps in coverage | 3/5 |
| **Coherence** | Well-organized, logically structured report | 3/5 |

## Pipeline Timing

| Metric | Before (sequential) | After (parallel gather) |
|--------|-------------------|----------------------|
| Full pipeline | ~191s | ~210s (local stdio) |
| Literature search | ~45s | ~60s (semaphore=3) |
| Concept explorer | ~15s | ~10s |
| Citation analyzer | ~30s | ~25s |
| Assumption auditor | ~10s | ~8s |

**Finding**: Local stdio transport shows no net speedup because research-kb's connection pool (max 10) becomes the bottleneck when queries run concurrently — each search uses multiple DB operations (BM25 + vector + graph + citation scoring). The `asyncio.Semaphore(3)` in literature search prevents pool exhaustion but limits parallelism.

**Where parallelism helps**: Concept explorer, citation analyzer, and assumption auditor all show modest improvements since their MCP calls are lighter-weight. The architecture is designed for HTTP/remote transport where network latency dominates and the connection pool is not shared — expect 2-3x speedup in that deployment mode.

## Judge Implementation

- **Judge model**: `claude-haiku-4-5-20251001` (cost-efficient)
- **Synthesis model**: `claude-sonnet-4-5-20250514` (production), `claude-haiku-4-5-20251001` (eval tests)
- **Structured output**: Pydantic `JudgeVerdict` via JSON prompting (more reliable than tool_use for multi-field schemas)
- **Source**: `evals/judges.py`
