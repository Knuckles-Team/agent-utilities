# Agentic Harness Engineering (AHE) — Architecture

> CONCEPT:AU-AHE.harness.harness-evolution — Agentic Harness Engineering

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
    A[AU-OS.governance.wasm-micro-agent-sandbox: Langfuse Traces] --> B[AHE-3.1: Automated Distillation]
    B --> C["KG-2.6: Summaries & Clusters"]
    C --> D[ORCH-1.21: Failure Taxonomies]
    D --> E[KG-2.6: Layered Evidence Corpus]
    E --> F[ORCH-1.1: Evolve Agent Decisions]

    B -.-> G[AU-OS.governance.wasm-micro-agent-sandbox: langfuse-agent API]
    C -.-> H[KG-2.6: RLM Summarizer]
    D -.-> I[KG-2.0: KG Semantic Clustering]
    E -.-> J[KG-2.0: Versioned Files + KG Nodes]
```

## Component Types

AHE decomposes the harness into 7 independently editable component types:

```mermaid
graph TD
    subgraph "AHE Component Types"
        SP["System Prompt<br>prompting/builder.py<br>prompting/structured.py"]
        TD["Tool Description<br>tool_filtering.py<br>SKILL.md frontmatter"]
        TI["Tool Implementation<br>tools/*.py<br>mcp_server.py"]
        MW["AU-OS.governance.reactive-multi-axis-budget: Middleware<br>middlewares.py<br>guardrails.py<br>tool_guard.py"]
        SK["Skills<br>universal-skills/"]
        SA["Sub-Agents<br>graph/steps/<br>HSM specialist nodes"]
        LM["Long-Term Memory<br>knowledge_graph/<br>MemoryNode"]
    end

    subgraph "AU-OS.governance.wasm-micro-agent-sandbox: Observability Pillars"
        CO["AU-OS.governance.wasm-micro-agent-sandbox: Component Observability<br>File-level diffs + git"]
        EO["AU-OS.governance.wasm-micro-agent-sandbox: Experience Observability<br>TraceDistiller → EvidenceCorpus"]
        DO["AU-OS.governance.wasm-micro-agent-sandbox: Decision Observability<br>ChangeManifest + VerificationResult"]
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
    P["PROMPT<br>Level 1: Advisory"] --> TD2["TOOL_DESCRIPTION<br>Level 2: Descriptive"]
    TD2 --> M["MIDDLEWARE<br>Level 3: Blocking"]
    M --> TI2["TOOL_IMPLEMENTATION<br>Level 4: Hardcoded"]

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
├── __init__.py              # Package exports (CONCEPT:AU-AHE.harness.harness-evolution)
├── manifest.py              # ComponentType, ComponentEdit, ChangeManifest
├── evidence_corpus.py       # EvidenceLayer, EvidenceEntry, EvidenceCorpus
├── component_registry.py    # HarnessComponentRegistry
├── trace_backend.py         # TraceBackend ABC + Langfuse/OTel/File backends
├── evolve_agent.py          # EvolveAgent (lightweight + full modes)
├── verifier.py              # ManifestVerifier + auto-revert
└── constraint_engine.py     # ConstraintLevel, ConstraintEngine
```

## Integration Points

- **SDD Pipeline**: Manifests stored in `.specify/manifests/` alongside specs/plans
- **Knowledge Graph**: `ChangeManifest`, `ComponentEditRecord`, `EvidenceRecord`, `ConstraintState` node types
- **RLM**: `TraceDistiller` (`knowledge_graph/adaptation/trace_distiller.py`) uses RLM for deep failure analysis on massive trace data
- **Langfuse Agent**: Direct API import via `from langfuse_agent.api_client import LangfuseApi` (see `harness/trace_backend.py` `LangfuseTraceBackend`)
