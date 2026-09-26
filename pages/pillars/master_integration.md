# Ecosystem Integration & Concept Wiring

> **Single source of truth** for all CONCEPT: tags interconnection.

This document serves as the master blueprint for the `agent-utilities` OS Kernel. It illustrates precisely how the 5 foundational pillars interact across execution boundaries to form a continuous, resilient intelligence graph.

---

## 1. The Five Pillars Overview

- **ORCH (Orchestration Engine)**: The cognitive router, HTN planner, and task dispatcher.
- **KG (Knowledge Graph)**: The active epistemic state, tiered memory, and semantic search engine.
- **AHE (Agentic Harness)**: The continuous evaluation, evolution, curriculum, and task detection engine.
- **ECO (Ecosystem Peripherals)**: External integrations, A2A consensus, and MCP tool factories.
- **OS (Agent OS Kernel)**: The guardrails, security policies, paths, and cognitive scheduler.

---

## 2. Master Wiring Diagram

<div class="admonition architecture" markdown>
<p class="admonition-title">Pillar 1 — Orchestration Engine (ORCH)</p>

Intelligence Graph Core (ORCH-1.0) feeds the HTN Planning Pipeline
(ORCH-1.1), which feeds Agent Orchestrator (ORCH-1.0). The orchestrator
feeds the Capability Wiring Engine (ORCH-1.4), which feeds Specialist
Routing & Discovery (ORCH-1.2); Execution Safety & State (ORCH-1.3) also
feeds the orchestrator. The DSTDD Pipeline feeds back into Intelligence
Graph Core.
</div>

<div class="admonition architecture" markdown>
<p class="admonition-title">Pillar 2 — Knowledge Graph (KG)</p>

Active Knowledge Graph (KG-2.0) feeds Ontology & Epistemics (KG-2.2);
Graph Integrity & Retrieval (KG-2.3) and Research Intelligence both feed
back into KG-2.0. Tiered Memory & Context (KG-2.1) feeds Memory Stability;
Multi-Domain Architecture (KG-2.7) feeds Domain: Finance (KG-2.6);
Inductive Knowledge (KG-2.4) feeds Topological Analysis (KG-2.5).
</div>

<div class="admonition architecture" markdown>
<p class="admonition-title">Pillar 3 — Agentic Harness (AHE)</p>

Agentic Harness Core (AHE-3.0) feeds the Continuous Evaluation Engine
(AHE-3.1), which feeds both the Agentic Evolution Engine (AHE-3.2) and Team
& Synergy Optimization (AHE-3.3); Backtest & Curriculum also feeds
Continuous Evaluation. The Evolution Engine feeds Distributed Agentic
Evolution (AHE-3.4). Heavy Thinking & Background Intelligence and
KG-Native Task Detection both feed back into Agentic Harness Core.
</div>

<div class="admonition architecture" markdown>
<p class="admonition-title">Pillar 4 — Ecosystem Peripherals (ECO)</p>

Tool Interface & MCP Factory (ECO-4.0) feeds MCP Live Discovery; the Agent
Toolkit Ingestor and the KG MCP Server & Execution component both feed back
into ECO-4.0. Market Data KG Node Models feeds the A2A Network & Consensus
component (ECO-4.1).
</div>

<div class="admonition architecture" markdown>
<p class="admonition-title">Pillar 5 — Agent OS Kernel (OS)</p>

Agent OS Kernel & XDG Paths (OS-5.0) feeds both Security & Auth (OS-5.1)
and Resource Scheduling (OS-5.2). Security & Auth feeds Guardrails & Safety,
which feeds Telemetry & Observability.
</div>

<div class="admonition architecture" markdown>
<p class="admonition-title">Cross-pillar execution edges</p>

- ORCH's Capability Wiring Engine wires discovered tools into ECO's Tool
  Interface & MCP Factory.
- ORCH's Intelligence Graph Core retrieves templates/memory from KG's
  Graph Integrity & Retrieval.
- ORCH's Specialist Routing & Discovery reads KG's Active Knowledge Graph.
- ORCH's HTN Planning Pipeline records memory contexts into KG's Tiered
  Memory & Context.
- ORCH's Agent Orchestrator tracks state & fallbacks via Execution Safety
  & State.
- AHE's Continuous Evaluation Engine updates its self-model in KG's Active
  Knowledge Graph.
- AHE's Agentic Evolution Engine generates new skill topologies for ECO's
  Agent Toolkit Ingestor.
- AHE's Team & Synergy Optimization forms coalitions via KG's Ontology &
  Epistemics.
- ECO's MCP Live Discovery populates callable resources into KG's Active
  Knowledge Graph.
- ECO's KG MCP Server & Execution exposes KG logic as tools through OS's
  Security & Auth.
- ECO's Market Data KG Node Models injects financial signals into KG's
  Domain: Finance.
- OS's Security & Auth validates tool requests from ECO's Tool Interface
  & MCP Factory.
- OS's Guardrails & Safety emits execution faults to AHE's Continuous
  Evaluation Engine.
- OS's Telemetry & Observability stores traces into KG's Active Knowledge
  Graph.
- OS's Resource Scheduling preempts heavy planning in ORCH's Agent
  Orchestrator.
</div>

---

## 3. Consolidation Key (v2.0)

> **Note:** The canonical, machine-checked concept registry now lives in
> [`registry/concepts.yaml`](https://github.com/Knuckles-Team/agent-utilities/blob/main/registry/concepts.yaml) (single source of truth, regenerated via
> `scripts/build_concepts_yaml.py` and enforced by `scripts/check_concepts.py`).
> The current registry tracks **70 concepts across 12 pillars**; the historical
> merge log below records how the earlier sprawling layout was first pruned and
> may use concept IDs that have since been renumbered in `concepts.yaml`.

To achieve maximum system stability and clean 1:1:1 traceability, the legacy conceptual layout was pruned and synthesized down to a compact concept set:
* **Legacy ORCH Consolidation**:
  * `ORCH-1.0` -> Merged into `ORCH-1.3` (Execution Safety & State).
  * `AU-ORCH.planning.legal-automation-roadmap` -> Merged into `ORCH-1.0` (Agent Orchestrator).
  * `ORCH-1.14` & `ORCH-1.17` -> Merged into `ORCH-1.2` (Specialist Routing & Discovery).
  * `ORCH-1.15` & `ORCH-1.16` -> Merged into `ORCH-1.1` (HTN Planning Pipeline).
  * `ORCH-1.18`, `ORCH-1.19`, `AU-ORCH.execution.service-registry-initialization` -> Merged into `ORCH-1.4` (Capability Wiring Engine).
* **Legacy KG Consolidation**:
  * `KG-2.7` (External Graph Federation) -> Eliminated due to collision with multi-domain structure.
  * `KG-2.3` (Dynamic AR-Graph) -> Merged into `KG-2.2` (Ontology & Epistemics).
  * `KG-2.6` (Time-Series Weighted Graph) -> Merged into `KG-2.6` (Domain: Finance).
* **Legacy AHE Consolidation**:
  * `AHE-3.4` (Distributed Agentic Evolution) -> Merged into `AHE-3.2` (Agentic Evolution Engine).
  * `AU-AHE.harness.concept-2` (Distributed Agent State Manager) -> Displaced by `ORCH-1.3` (Execution Safety & State).
* **Legacy ECO Consolidation**:
  * `AU-ECO.toolkit.journey-map-milestones` (Terminal Agent Launcher) -> Merged into `ECO-4.0` (Tool Interface & MCP Factory).
  * `AU-ECO.mcp.toolkit-live-discovery` (Agent Hook Installer) -> Merged into `ECO-4.0` (Tool Interface & MCP Factory).
  * `AU-OS.deployment.infra-orchestration`, `AU-OS.deployment.blueprint-library`, `AU-ECO.bus.pluggable-queue-backend` (Quant ecosystem) -> Synthesized into `AU-ECO.ui.company-infrastructure-orchestration` (Market Data Connectors).
