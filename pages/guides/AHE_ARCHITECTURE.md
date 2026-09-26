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

<div class="admonition architecture" markdown>
<p class="admonition-title">Traces distill up to Evolve Agent decisions, each stage backed by an integration</p>

Langfuse traces feed automated distillation (backed by the langfuse-agent
API), producing summaries & clusters (backed by an RLM summarizer), which
build failure taxonomies (backed by KG semantic clustering), which build a
layered evidence corpus (backed by versioned files + KG nodes), which
finally drives Evolve Agent decisions.
</div>

## Component Types

AHE decomposes the harness into 7 independently editable component types:

<div class="admonition architecture" markdown>
<p class="admonition-title">Seven component types, all observed at the file level</p>

The 7 independently editable component types — System Prompt
(`prompting/builder.py`, `prompting/structured.py`), Tool Description
(`tool_filtering.py`, `SKILL.md` frontmatter), Tool Implementation
(`tools/*.py`, `mcp_server.py`), Middleware (`middlewares.py`,
`guardrails.py`, `tool_guard.py`), Skills (`universal-skills/`), Sub-Agents
(`graph/steps/`, HSM specialist nodes), and Long-Term Memory
(`knowledge_graph/`, `MemoryNode`) — all feed Component Observability
(file-level diffs + git), the first of three observability pillars.
The other two pillars, Experience Observability (`TraceDistiller` →
`EvidenceCorpus`) and Decision Observability (`ChangeManifest` +
`VerificationResult`), observe the harness's runtime behavior rather than
its files directly.
</div>

## Constraint Hierarchy

Constraints escalate through 4 enforcement levels when violations are detected:

<div class="admonition architecture" markdown>
<p class="admonition-title">Four enforcement levels, escalating on repeat violation</p>

Constraints escalate through four levels: Level 1 PROMPT (advisory), Level
2 TOOL_DESCRIPTION (descriptive), Level 3 MIDDLEWARE (blocking), Level 4
TOOL_IMPLEMENTATION (hardcoded).
</div>

When a constraint is violated at the prompt level, the `ConstraintEngine`
auto-escalates it to middleware-level enforcement after the escalation
threshold is reached. This ensures the agent cannot repeatedly "forget"
important constraints.

## Package Structure

- `agent_utilities/harness/`
    - `__init__.py` — package exports (CONCEPT:AU-AHE.harness.harness-evolution)
    - `manifest.py` — `ComponentType`, `ComponentEdit`, `ChangeManifest`
    - `evidence_corpus.py` — `EvidenceLayer`, `EvidenceEntry`, `EvidenceCorpus`
    - `component_registry.py` — `HarnessComponentRegistry`
    - `trace_backend.py` — `TraceBackend` ABC + Langfuse/OTel/File backends
    - `evolve_agent.py` — `EvolveAgent` (lightweight + full modes)
    - `verifier.py` — `ManifestVerifier` + auto-revert
    - `constraint_engine.py` — `ConstraintLevel`, `ConstraintEngine`

## Integration Points

- **SDD Pipeline**: Manifests stored in `.specify/manifests/` alongside specs/plans
- **Knowledge Graph**: `ChangeManifest`, `ComponentEditRecord`, `EvidenceRecord`, `ConstraintState` node types
- **RLM**: `TraceDistiller` (`knowledge_graph/adaptation/trace_distiller.py`) uses RLM for deep failure analysis on massive trace data
- **Langfuse Agent**: Direct API import via `from langfuse_agent.api_client import LangfuseApi` (see `harness/trace_backend.py` `LangfuseTraceBackend`)
