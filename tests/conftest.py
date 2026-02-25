"""Shared test fixtures for research-agent tests.

Provides mocked MCP client and standard test configurations.
All MCP responses are realistic markdown matching research-kb's actual output format.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from research_agent.config import AgentConfig, MCPConfig, ModelConfig
from research_agent.mcp_client import ResearchKBClient
from research_agent.state import ResearchState, SubTask


@pytest.fixture
def test_config() -> AgentConfig:
    """Standard test configuration with fast models."""
    return AgentConfig(
        models=ModelConfig(
            planning="claude-haiku-4-5-20251001",
            synthesis="claude-haiku-4-5-20251001",  # Use haiku for speed in tests
        ),
        mcp=MCPConfig(transport="stdio", research_kb_path="/fake/path"),
        max_search_results=5,
        max_concepts=3,
        max_citations=5,
    )


@pytest.fixture
def mock_mcp() -> ResearchKBClient:
    """Mocked MCP client with realistic research-kb responses."""
    client = AsyncMock(spec=ResearchKBClient)

    # --- Search response ---
    client.search.return_value = """## Search Results for: double machine learning
*Query expanded to: double machine learning DML debiased*

**Found 3 results** (in 245ms)

### 1. Double/Debiased Machine Learning for Treatment and Structural Parameters
Chernozhukov et al. (2018) [Paper]
*Section 2.1: Framework*

> Double machine learning (DML) provides a general framework for estimating
> treatment effects using machine learning methods while maintaining valid
> statistical inference. The key insight is cross-fitting...

*Score: 0.892 | FTS: 0.756 | Vector: 0.891 | Graph: 0.823*
*Source ID: `src-001-dml` | Chunk ID: `chk-001`*

### 2. Debiased Machine Learning of Conditional Average Treatment Effects
Semenova and Chernozhukov (2021) [Paper]

> We propose debiased machine learning estimators for conditional average
> treatment effects (CATEs)...

*Score: 0.834 | FTS: 0.612 | Vector: 0.845 | Graph: 0.790*
*Source ID: `src-002-cate` | Chunk ID: `chk-002`*

### 3. Cross-fitting and Double Machine Learning
Newey and Robins (2018) [Paper]

> We analyze the properties of cross-fitting in the context of
> semiparametric estimation...

*Score: 0.781 | FTS: 0.580 | Vector: 0.802 | Graph: 0.710*
*Source ID: `src-003-crossfit` | Chunk ID: `chk-003`*
"""

    # --- Fast search response ---
    client.fast_search.return_value = """## Search Results for: DML assumptions

**Found 2 results** (in 42ms)

### 1. Double/Debiased Machine Learning for Treatment and Structural Parameters
Chernozhukov et al. (2018)

> The key assumptions are unconfoundedness and overlap...

*Score: 0.845*
*Source ID: `src-001-dml` | Chunk ID: `chk-010`*

### 2. Assumption Lean Inference
Vansteelandt and Dukes (2022)

> Discusses relaxing parametric assumptions in causal inference...

*Score: 0.712*
*Source ID: `src-004-lean` | Chunk ID: `chk-020`*
"""

    # --- Concept responses ---
    client.get_concept.return_value = """## Double Machine Learning
**Type:** METHOD
**ID:** `concept-dml-001`

### Description
A framework for estimating treatment effects that uses cross-fitting
to avoid overfitting bias in nuisance parameter estimation.

### Relationships (5 total)
- REQUIRES \u2192 `concept-unconf-001`
- REQUIRES \u2192 `concept-overlap-001`
- USES \u2192 `concept-crossfit-001`
- ADDRESSES \u2192 `concept-regbias-001`
- RELATED_TO \u2192 `concept-iv-001`
"""

    client.graph_neighborhood.return_value = """## Graph Neighborhood: double machine learning
*Type: METHOD | ID: `concept-dml-001`*

**8 connected concepts, 12 relationships**

### Connected Concepts
- Unconfoundedness [ASSUMPTION]
- Overlap condition [ASSUMPTION]
- Cross-fitting [METHOD]
- Regularization bias [PROBLEM]
- Instrumental variables [METHOD]
- Neyman orthogonality [THEOREM]
- CATE estimation [METHOD]
- Propensity score [METHOD]

### Relationships
- REQUIRES: 2
- USES: 2
- ADDRESSES: 1
- RELATED_TO: 3
"""

    # --- Citation responses ---
    client.citation_network.return_value = """## Citation Network: Double/Debiased Machine Learning
*Source ID: `src-001-dml`*

### Citing This Source (3)
*Papers that built on this work*

- **Debiased Machine Learning of CATEs** (2021)
  - Semenova and Chernozhukov
  - ID: `src-002-cate`

- **Automatic Debiased Machine Learning** (2022)
  - Chernozhukov et al.
  - ID: `src-005-auto`

- **DML for Difference-in-Differences** (2023)
  - Chang (2023)
  - ID: `src-006-dml-did`

### Cited By This Source (2)
*Foundations and context*

- **Estimation and Inference of Heterogeneous Treatment Effects** (2017)
  - Athey and Imbens
  - ID: `src-007-hte`

- **High-Dimensional Methods and Inference** (2014)
  - Belloni, Chernozhukov, Hansen
  - ID: `src-008-hdm`
"""

    client.biblio_coupling.return_value = (
        "## Bibliographically Similar: Double/Debiased ML\n"
        "*Source ID: `src-001-dml`*\n"
        "\n"
        "**2 similar sources** by shared references\n"
        "\n"
        "- **Debiased Machine Learning of CATEs** (2021)\n"
        "  - Semenova and Chernozhukov\n"
        "  - Coupling: **45.2%** (8 shared refs)\n"
        "  - ID: `src-002-cate`\n"
        "\n"
        "- **Automatic Debiased Machine Learning** (2022)\n"
        "  - Chernozhukov et al.\n"
        "  - Coupling: **38.7%** (6 shared refs)\n"
        "  - ID: `src-005-auto`\n"
    )

    # --- Assumption audit response ---
    client.audit_assumptions.return_value = """## Assumptions for: Double Machine Learning (DML)
**Aliases**: DML, debiased ML, double ML
**Method ID**: `concept-dml-001`

**Definition**: Framework for treatment effect estimation using ML nuisance estimation.

**Source**: graph

### Required Assumptions (3 found)

#### Critical (identification fails if violated)

**1. Unconfoundedness** [CRITICAL]
   - **Formal**: `Y(t) \u22a5 T | X`
   - **Plain English**: All confounders are observed and included in X.
   - **If violated**: Treatment effect estimates are biased.
   - **Verify**: Sensitivity analysis, placebo tests
   - **Citation**: Chernozhukov et al. (2018), Section 2
   - **Concept ID**: `concept-unconf-001`
   - **Relationship**: REQUIRES

**2. Overlap (Positivity)** [CRITICAL]
   - **Formal**: `0 < P(T=1|X) < 1`
   - **Plain English**: Every unit has positive probability of treatment.
   - **If violated**: Extreme propensity scores, unstable estimates.
   - **Verify**: Check propensity score distribution
   - **Citation**: Chernozhukov et al. (2018), Assumption 3.1
   - **Concept ID**: `concept-overlap-001`
   - **Relationship**: REQUIRES

#### Standard

**3. Neyman Orthogonality** [STANDARD]
   - **Formal**: `\u2202\u03b8 E[\u03c8(W; \u03b8\u2080, \u03b7\u2080)] = 0`
   - **Plain English**: Score function is insensitive to nuisance estimation error.
   - **If violated**: \u221an convergence may not hold.
   - **Verify**: Verify moment condition structure
   - **Citation**: Chernozhukov et al. (2018), Definition 2.1
   - **Concept ID**: `concept-neyman-001`
   - **Relationship**: REQUIRES

### Code Docstring Snippet

```python
Assumptions:
    [CRITICAL] - unconfoundedness: All confounders observed and included in X
    [CRITICAL] - overlap: Every unit has positive probability of treatment
    - neyman_orthogonality: Score function insensitive to nuisance estimation error
```
"""

    return client


@pytest.fixture
def sample_state() -> ResearchState:
    """Sample state for testing individual nodes."""
    return ResearchState(
        query="What are the assumptions of double machine learning?",
        sub_tasks=[
            SubTask(
                description="Find foundational papers on DML",
                search_queries=["double machine learning Chernozhukov", "DML cross-fitting"],
                concepts_to_explore=["double machine learning", "cross-fitting"],
                methods_to_audit=["DML"],
            ),
            SubTask(
                description="Understand identification assumptions",
                search_queries=["unconfoundedness assumption causal inference"],
                concepts_to_explore=["unconfoundedness"],
                methods_to_audit=[],
            ),
        ],
    )
