# DSTDD Pipeline: Design-Spec-Test Driven Development (CONCEPT:AU-ORCH.execution.execution-budget-caps)

## Overview
The DSTDD (Design-Spec-Test Driven Development) pipeline is the formalized workflow that prevents architectural bloat. It ensures that all new features, external integrations, or research ideas are woven natively into the existing Knowledge Graph and 5-Pillar topology.

## The "Extend Before Invent" Mandate
> "New functionality MUST first be expressed as an extension, augmentation, or composition of an existing pillar/concept before a new CONCEPT: tag or domain is introduced. The Knowledge Graph is the arbiter."

## Workflow
1. **Design Phase**:
   - An agent reads the research paper or codebase to be ingested.
   - It uses `universal-skills` (`skill-graph-builder` + `c4-architecture`) to parse the context.
   - The KG analogy engine (`kg_analogize`) maps the new ideas onto the existing 5 pillars.
   - A design artifact (Mermaid C4 + pillar interconnection) is generated in `.specify/design/`.
2. **Spec Phase**:
   - The design is decomposed into actionable specs inside `.specify/specs/`.
   - The spec explicitly references the existing pillars being extended.
3. **Test Phase**:
   - Auto-generate TDD tests.
   - Validate against the 15-phase Intelligence Graph Pipeline.

## Five-Pillar Interwoven Context
<div class="admonition architecture" markdown>
<p class="admonition-title">Five pillars, interwoven around the developer/agent</p>

A developer or agent (via Antigravity IDE, Claude, or a direct API)
submits tasks to **ORCH-1.0 Graph Orchestration** (router -> planner ->
dispatcher, a 15-phase pipeline), which queries/ingests **KG-2.0
Knowledge Graph** (NetworkX + LadybugDB, the single source of truth) —
bidirectionally, since the KG also provides ontological routing back to
orchestration. The KG feeds **AHE-3.0 Agentic Harness** (self-model,
TeamConfig, evolution), which promotes proven coalitions to **ECO-4.0
Ecosystem** (MCP, A2A, universal-skills). Ecosystem uses **OS-5.0 Agent
OS Kernel** (auth, guardrails, lifecycle) for execution safety, and the
kernel persists execution traces and telemetry back to the Knowledge
Graph.
</div>
