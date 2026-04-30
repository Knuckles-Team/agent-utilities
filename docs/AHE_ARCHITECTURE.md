# Agentic Harness Engineering (AHE) — Architecture

> CONCEPT:AU-012 — Agentic Harness Engineering

## Overview

AHE is a closed-loop optimization framework where an **Evolve Agent**
iteratively improves the agent harness — its tools, middleware, memory,
skills, sub-agents, and system prompt — guided by three pillars of
structured observability.

## Hybrid State Model

| State Layer | Implementation | Storage |
|---|---|---|
| **Epistemic** (what the agent knows) | `IntelligenceGraphEngine` + MAGMA views | `knowledge_graph.db` |
| **Normative** (what the agent is allowed to do) | Component files (prompts, middleware, tools) | Filesystem + git |
| **Causal** (what caused improvement) | Change Manifests | `.specify/manifests/` + KG |

## AHE Evolution Loop

```mermaid
graph LR
    A[Langfuse Traces] --> B[Automated Distillation]
    B --> C[Summaries & Clusters]
    C --> D[Failure Taxonomies]
    D --> E[Layered Evidence Corpus]
    E --> F[Evolve Agent Decisions]

    B -.-> G[langfuse-agent API]
    C -.-> H[RLM Summarizer]
    D -.-> I[KG Semantic Clustering]
    E -.-> J[Versioned Files + KG Nodes]
```

## Component Types

AHE decomposes the harness into 7 independently editable component types:

```mermaid
graph TD
    subgraph "AHE Component Types"
        SP[System Prompt<br>prompt_builder.py<br>structured_prompts.py]
        TD[Tool Description<br>tool_filtering.py<br>SKILL.md frontmatter]
        TI[Tool Implementation<br>tools/*.py<br>mcp_server.py]
        MW[Middleware<br>middlewares.py<br>guardrails.py<br>tool_guard.py]
        SK[Skills<br>universal-skills/]
        SA[Sub-Agents<br>graph/steps/<br>HSM specialist nodes]
        LM[Long-Term Memory<br>knowledge_graph/<br>MemoryNode]
    end

    subgraph "Observability Pillars"
        CO[Component Observability<br>File-level diffs + git]
        EO[Experience Observability<br>TraceDistiller → EvidenceCorpus]
        DO[Decision Observability<br>ChangeManifest + VerificationResult]
    end

    SP --> CO
    TD --> CO
    TI --> CO
    MW --> CO
    SK --> CO
    SA --> CO
    LM --> CO
```

## Constraint Hierarchy

Constraints escalate through 4 enforcement levels when violations are detected:

```mermaid
graph LR
    P[PROMPT<br>Level 1: Advisory] --> TD2[TOOL_DESCRIPTION<br>Level 2: Descriptive]
    TD2 --> M[MIDDLEWARE<br>Level 3: Blocking]
    M --> TI2[TOOL_IMPLEMENTATION<br>Level 4: Hardcoded]

    style P fill:#4caf50,color:#fff
    style TD2 fill:#ff9800,color:#fff
    style M fill:#f44336,color:#fff
    style TI2 fill:#9c27b0,color:#fff
```

When a constraint is violated at the prompt level, the `ConstraintEngine`
auto-escalates it to middleware-level enforcement after the escalation
threshold is reached. This ensures the agent cannot repeatedly "forget"
important constraints.

## Package Structure

```
agent_utilities/harness/
├── __init__.py              # Package exports (CONCEPT:AU-012)
├── manifest.py              # ComponentType, ComponentEdit, ChangeManifest
├── evidence_corpus.py       # EvidenceLayer, EvidenceEntry, EvidenceCorpus
├── component_registry.py    # HarnessComponentRegistry
├── trace_backend.py         # TraceBackend ABC + Langfuse/OTel/File backends
├── trace_distiller.py       # TraceDistiller pipeline
├── evolve_agent.py          # EvolveAgent (lightweight + full modes)
├── verifier.py              # ManifestVerifier + auto-revert
└── constraint_engine.py     # ConstraintLevel, ConstraintEngine
```

## Integration Points

- **SDD Pipeline**: Manifests stored in `.specify/manifests/` alongside specs/plans
- **Knowledge Graph**: `ChangeManifest`, `ComponentEditRecord`, `EvidenceRecord`, `ConstraintState` node types
- **RLM**: `TraceDistiller` uses RLM for deep failure analysis on massive trace data
- **Graph Steps**: `evolve_step` registered as a specialist node in the HSM
- **Langfuse Agent**: Direct API import via `from langfuse_agent.langfuse_api import LangfuseAPI`
