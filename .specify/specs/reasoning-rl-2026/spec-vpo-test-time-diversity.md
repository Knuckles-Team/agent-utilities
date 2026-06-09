# Spec: Test-Time Diversity (VPO-derived)

> **Status: PROPOSED.** Concept: new **AHE-3.x**. Source: VPO — *Vector Policy Optimization:
> Training for Diversity Improves Test-Time Search* (arXiv:2605.22817). Rank 2 in
> `COMPARATIVE_ANALYSIS.md`.

## Pre-Flight Checklist (Mandatory — DSTDD)

- [ ] **KG search completed** — `.specify/design/vpo-test-time-diversity/design.md` exists
- [ ] **Extension point identified** — extends `reasoning_effort` + subagent fan-out (New Concept: AHE-3.x)
- [ ] **C4 diagram created** — diversity-scored fan-out into the test-time-compute topology
- [ ] **No new CONCEPT: tag** without pillar reference
- [ ] **`code-enhancer` audit** run against proposed changes
- [ ] **Design validation passes** — `SDDManager.validate_design("vpo-test-time-diversity")`

## Problem

We scale test-time compute (`harness/reasoning_effort.py`) and fan out subagents
(`SubagentLifecyclePolicy`, `rlm/` parallel sub-calls), but we sample toward a *single best*
answer. VPO shows that explicitly optimizing for a **diverse solution set** (under reward
vectors rather than a scalar reward) raises **best@k / pass@k** at test time — exactly the
regime our best-of-k fan-out operates in.

## Existing anchors (reuse, don't reinvent)

- `agent_utilities/harness/reasoning_effort.py` — continuous effort → discrete search budget.
- `agent_utilities/graph/routing/strategies/policy.py::SubagentLifecyclePolicy` — fan-out/team.
- `agent_utilities/rlm/` — parallel sub-calls (the fan-out we'd diversify).
- `epistemic-graph` `personalized_pagerank()` — seed-diverse propagation, a natural diversity kernel.

## User Stories

### US-1: Diversity-scored best-of-k fan-out
**As an** orchestrator under a test-time-compute budget, **I want** the k parallel rollouts to
be *diverse* (scored by a reward vector / embedding spread), **so that** best@k/pass@k improves
without simply increasing k.
**Acceptance Criteria:**
- [ ] A diversity score (embedding spread / reward-vector divergence) is computed over the candidate set.
- [ ] Selection trades off quality vs. diversity (configurable weight; sensible default ON per Wire-First).
- [ ] `*_live_path` test: a fan-out task yields a measurably more diverse candidate set than scalar-best sampling, asserted on the existing fan-out entry point.

### US-2: Budget-aware diversity
**As the** harness, **I want** diversity effort tied to `reasoning_effort`, **so that** harder
queries get wider, more diverse search and easy ones don't waste compute.
**Acceptance Criteria:**
- [ ] Diversity width derives from the existing effort estimate; no new agent-facing knob.

## Non-Functional Requirements
- [ ] Zero regression; tests < 60s; pre-commit clean; no stubs.
- [ ] Diversity computation is O(k) over candidates, off the latency-critical single-answer path.

## Post-Modification Artifact Mandate (all 7 required before merge)
1. [ ] **/docs** — `docs/pillars/3_agentic_harness_engineering/AHE-3.x-Test-Time-Diversity.md`
2. [ ] **AGENTS.md** — diversity-aware fan-out capability
3. [ ] **CHANGELOG.md** — Unreleased entry citing arXiv:2605.22817
4. [ ] **README.md** — AHE pillar concept list/count update
5. [ ] **.specify/** — spec + tasks + design; dual-write to KG
6. [ ] **.specify/reports/** — C4 diagram of diversity-scored fan-out
7. [ ] **Pytests** — unit (diversity score) + `*_live_path` (fan-out diversity side effect)
