# Spec: Agent-Step Policy Optimization (ARPO-derived)

> **Status: PROPOSED.** Concept: new **AHE-3.x** (Agentic Harness Engineering).
> Source: ARPO — *Agentic Reinforced Policy Optimization* (arXiv:2507.19849).
> Derived from `.specify/specs/reasoning-rl-2026/COMPARATIVE_ANALYSIS.md` (rank 1).

## Pre-Flight Checklist (Mandatory — DSTDD)

- [ ] **KG search completed** — `.specify/design/arpo-agent-step-po/design.md` exists
- [ ] **Extension point identified** — extends AHE-3.1 reward + ORCH routing (New Concept Proposal: AHE-3.x)
- [ ] **C4 diagram created** — agent-step credit + entropy-branching into ORCH routing topology
- [ ] **No new CONCEPT: tag** without pillar reference — AHE-3.x under Agentic Harness Engineering
- [ ] **`code-enhancer` audit** run against proposed changes
- [ ] **Design validation passes** — `SDDManager.validate_design("arpo-agent-step-po")`

## Problem

Our reward/credit today is largely trajectory-terminal: the capability router records an
outcome reward (EMA) after a task, and `reward_decomposition` blends a trajectory reward with
step rewards at a fixed α. For **multi-turn tool agents**, the decisive uncertainty is at
intermediate *tool-call steps*, where a single bad branch dooms the trajectory. ARPO shows
that branching/rollout at **high-entropy agent steps** and assigning advantage **per agent
step** materially improves multi-turn tool-use agents over answer-only credit.

## Existing anchors (reuse, don't reinvent)

- `agent_utilities/graph/reward_decomposition.py` — step vs trajectory outcome + decomposed reward.
- `agent_utilities/knowledge_graph/retrieval/capability_index.py` — `designate()` (reward-EMA
  blended selection) and `record_outcome()` (per-capability reward EMA, α=0.3).
- `agent_utilities/graph/routing/strategies/policy.py` — `SubagentLifecyclePolicy` (complexity→
  inline/fan-out/team) is where branching decisions already live.
- `agent_utilities/knowledge_graph/adaptation/trace_distiller.py` — `EpisodeToPreferenceRule`.

## User Stories

### US-1: Entropy-gated branching at tool steps
**As an** agentic orchestrator, **I want** to branch additional rollouts at high-uncertainty
tool/decision steps, **so that** a single bad intermediate choice doesn't doom the trajectory.
**Acceptance Criteria:**
- [ ] A step-entropy/uncertainty signal is computed at tool-call boundaries and exposed to `SubagentLifecyclePolicy`.
- [ ] Branching triggers only above a configurable entropy threshold (default OFF-safe, wired ON with a sane default per Wire-First).
- [ ] Branch budget is bounded (no unbounded fan-out) and observable in telemetry.

### US-2: Agent-step advantage write-back
**As the** evolution loop, **I want** advantage assigned at the agent-step granularity and
written back into the capability reward-EMA, **so that** routing learns which *intermediate
actions* (not just final answers) improve outcomes.
**Acceptance Criteria:**
- [ ] `reward_decomposition` emits per-step advantage; `capability_index.record_outcome` accepts a step-scoped outcome.
- [ ] A `*_live_path` test asserts that running the router on a multi-step task records step-level outcomes as a side effect (not just an isolated helper test).

## Non-Functional Requirements
- [ ] All existing tests pass (zero regression); tests complete < 60s (`pytest-timeout`).
- [ ] Pre-commit clean; no stubs (`# ABSTRACT-OK` only for genuine abstracts).
- [ ] Branching is bounded and cannot wedge the worker pool.

## Post-Modification Artifact Mandate (all 7 required before merge)
1. [ ] **/docs** — `docs/pillars/3_agentic_harness_engineering/AHE-3.x-Agent-Step-PO.md`
2. [ ] **AGENTS.md** — note the new agent-step credit/branching capability
3. [ ] **CHANGELOG.md** — Unreleased entry citing arXiv:2507.19849
4. [ ] **README.md** — AHE pillar concept list/count update
5. [ ] **.specify/** — this spec + `tasks.md` + `design.md`; dual-write to KG via `graph_ingest`
6. [ ] **.specify/reports/** — C4 diagram of step-credit + branching into ORCH routing
7. [ ] **Pytests** — unit (step-advantage math) + `*_live_path` (router records step outcomes)
