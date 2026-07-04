# Comparative Analysis: OpenHands → agent-utilities (KG-grounded SWE platform)

**Date:** 2026-06-13
**Sources pinned:**
- OpenHands @ `/home/apps/workspace/open-source-libraries/OpenHands` (cloned HEAD)
- agent-utilities @ `main` `2a712d7` (worktree `feat/openhands-parity-swe-platform`)
**Ledger:** [`innovation_ledger_openhands.json`](./innovation_ledger_openhands.json) (10 rows, all `verified`)

## Executive summary

OpenHands is a heavyweight, SWE-specific agent platform: a `CodeActAgent` running an
edit→run→test loop inside a **persistent sandboxed dev workspace**, streaming a typed
**action/observation** event stream to a React UI, with **git issue/PR resolvers** across six
platforms and a published **SWE-bench (~77.6%)** score.

agent-utilities is a broader, knowledge-centric self-evolving framework. It already has the
hard parts OpenHands lacks — an epistemic-graph **world model** with **live multi-language code
ingestion** (`Code`/`Test`/`Feature` nodes + `calls`/`covers`/`dependsOn` edges), an OWL/RDF
Palantir-parity ontology, a **golden-loop self-evolution** engine, RLM sandbox tiering, and a
fail-closed `ActionPolicy`. What it lacks is the **SWE surface**: a CodeAct agent, a persistent
dev-workspace runtime, a SWE-bench harness, and git resolvers.

**Thesis:** we don't copy OpenHands' architecture — we project the SWE loop onto ours. Every
code action becomes **KG provenance grounded to the symbols it mutated**; grounding is **graph
queries**, not context-stuffing; and **SWE-bench failures feed the golden loop** so the agent
*self-improves*. That is how we surpass, not merely match.

## Gap matrix

| # | OpenHands capability | agent-utilities today | Verdict | New concept |
|---|---|---|---|---|
| 1 | Persistent dev workspace (stateful shell/files/tests/ports) | RLM sandbox = stateless snippet-against-namespace | **GAP** | AU-OS.scaling.bridge-developer-workspace-mutating |
| 2 | Typed action/observation event stream | RLM `SandboxResult` = stdout+vars only | **GAP** | ORCH-1.46 |
| 3 | Linear event history | — (opportunity to surpass via KG provenance) | **SURPASS** | KG-2.64 |
| 4 | Context-stuffing + condenser grounding | live code ontology, not yet exposed as agent tools | **SURPASS** | KG-2.65 |
| 5 | CodeActAgent edit→run→test loop | no swe/codeact agent or mode | **GAP** | ORCH-1.47 |
| 6 | SWE-bench benchmark | only LongMemEval | **GAP** | AHE-3.22 |
| 7 | (none — static score) | golden loop exists, unused for SWE | **SURPASS** | AHE-3.23 |
| 8 | Git issue/PR resolvers (6 platforms) + webhooks | MCP connectors only, no resolver | **GAP** | AU-ECO.connector.git-task-resolver |
| 9 | Browser action tier (playwright/browsergym) | agent-browser skill, not in-loop | **GAP** | AU-ECO.toolkit.browser-agent-tier |
| 10 | React action/observation stream UI | agent-webui exists, no SWE view | **GAP** | AU-OS.scaling.kg-provenance-panel-data |

## Where we already win (do not rebuild — leverage)

- **World-model memory:** epistemic-graph KG vs. OpenHands' linear history + summarizing condenser.
- **Code ontology is live:** `pipeline/phases/parse.py` → `Code/Test/Feature` + `calls/covers`;
  `ontology_software.ttl`. This is the substrate for KG-2.65 grounding — already built.
- **Self-evolution:** `adaptation/failure_analyzer.py` + `research/golden_loop.py` +
  AU-AHE.assimilation.research-auto-merge governed merge. SWE-bench just needs to feed it (AHE-3.23).
- **Security:** fail-closed `ActionPolicy` (OS-5.24), `tool_guard`, run-scoped identity (OS-5.11/14)
  already exceed OpenHands' security-analyzer/confirmation-mode.

## Recommended sequence (critical path)

```
Phase 0  SDD scaffolding (this report + ledger + per-concept design/spec)
Phase 1  Runtime           AU-OS.scaling.bridge-developer-workspace-mutating + ORCH-1.46 + KG-2.64     ← long pole
Phase 2  SWE agent         KG-2.65 + ORCH-1.47               (KG-2.65 parallelizable)
Phase 3  Harness+loop      AHE-3.22 + AHE-3.23               ← highest value
Phase 4  Git resolver      AU-ECO.connector.git-task-resolver
Phase 5  Browser tier      AU-ECO.toolkit.browser-agent-tier                          (optional)
Phase 6  webui stream      AU-OS.scaling.kg-provenance-panel-data
```

Critical path: `events → bridge → workspace → swe_agent → swe mode → swebench_harness →
swebench_remediation`. Phases 4–6 layer on after the core loop is proven.

## Verification anchors (per concept `success_metric` in the ledger)

Each concept ships a `*_live_path` test and must pass `check_concepts.py` + `check_wiring.py`
(≤3-hop entry→code). End-to-end acceptance is the Phase-3 self-improvement loop: a forced
SWE-bench failure mints a gap-topic, produces a remediation branch, and is regression-gated on
the exact instance re-resolving.
