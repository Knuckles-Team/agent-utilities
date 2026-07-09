---
name: agent-utilities-development
skill_type: skill
description: >-
  How to develop for agent-utilities. Covers concept lifecycle, DSTDD design
  phase, testing patterns, coding conventions, and the 1:1:1 traceability rule.
tags: [development, concepts, testing, dstdd, traceability]
---

# Developing for agent-utilities

## 🏷️ Concept Lifecycle

All new features MUST follow this lifecycle:

```
New Feature Request
       │
       ▼
  ┌────────────────────┐
  │ KG Analogy Search  │  ← Does a similar concept already exist?
  │ (similarity ≥ 0.7) │
  └────────┬───────────┘
           │
    ┌──────┴──────┐
    │             │
  EXTEND      PROPOSE
    │             │
    ▼             ▼
  Augment    NewConceptProposal
  existing   (requires C4 diagram,
  concept     pillar assignment,
              pipeline phase)
    │             │
    └──────┬──────┘
           │
           ▼
  .specify/design/<feature>/design.md
           │
           ▼
  SDDManager.validate_design()
           │
           ▼
  .specify/specs/<feature>/spec.md
```

### Concept ID Format
- `ORCH-1.X` — Orchestration pillar
- `KG-2.X` — Knowledge Graph pillar
- `AHE-3.X` — Agentic Harness Engineering pillar
- `ECO-4.X` — Ecosystem & Peripherals pillar
- `OS-5.X` — Agent OS Infrastructure pillar

### 1:1:1 Traceability Rule

Every concept MUST have all three:
1. **Code**: `CONCEPT:X.Y` tag in module docstring
2. **Test**: `CONCEPT:X.Y` tag in test file docstring
3. **Docs**: Dedicated page in `docs/pillars/<pillar>/`

## 🧪 Testing Patterns

### Test File Layout
```
tests/
├── unit/
│   ├── core/              # OS-5.x tests
│   ├── graph/             # AU-ORCH.planning.orchestration-overview tests
│   ├── knowledge_graph/   # AU-KG.compute.kg-x tests
│   └── harness/           # AU-AHE.optimization.telemetry-optimization tests
├── integration/           # Cross-pillar integration tests
└── conftest.py            # Shared fixtures
```

### Test Conventions
- Test files: `test_<module_name>.py`
- Concept tag in docstring: `"""Tests for CONCEPT:AU-KG.compute.spectral-cluster-navigator — Topological Analysis."""`
- Use `@pytest.mark.parametrize` for edge cases
- Mock external services (LLM, MCP servers) in unit tests
- Use `graph_fixture()` from conftest for KG tests
- Target >80% coverage per concept

### Common Fixtures
```python
# conftest.py provides:
@pytest.fixture
def mock_engine():
    """IntelligenceGraphEngine with in-memory backend."""

@pytest.fixture
def sample_graph():
    """Pre-populated NetworkX graph with test nodes."""

@pytest.fixture
def mock_agent():
    """Pydantic AI agent with mocked model."""
```

## 📝 Coding Conventions

### Module Docstrings
Every module MUST start with:
```python
"""Module description.

CONCEPT:X.Y — Concept Name

Detailed explanation of what this module does.
"""
```

### Pydantic Models
- All KG models extend `RegistryNode` (from `models/knowledge_graph.py`)
- Use `RegistryNodeType` enum for type field
- Use `RegistryEdgeType` enum for edges
- All fields have `Field(description=...)` annotations

### Agent Prompts
- Agent prompts live in `agent_utilities/prompts/<name>.json`
- Format: `{task, type, metadata, identity, instructions, tools, version}`
- Tools list references MCP server names or tool function names
- Workflow SKILL.md frontmatter references prompt by name via `agent: <name>`

### OWL Ontology Conventions
- All ontologies use `@prefix : <http://knuckles.team/kg#>` namespace
- Aligned to BFO (Basic Formal Ontology) upper ontology
- Domain-specific files: `ontology_<domain>.ttl`
- Import core: `owl:imports <http://knuckles.team/kg>`
