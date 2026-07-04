# Spec: Corrigibility + Knowledge-Seeking Objective Primitives (SAFE-1.5)

> Status: **proposed**. **Wire-First** — this EXTENDS the AU-ORCH.session.durable-session-autonomous-goal autonomous goal
> loop (`agent_utilities/models/goal.py`: `GoalSpec`/`GoalIteration`/`GoalCheckpoint`)
> and the fail-closed OS-5.24 `ActionPolicy` (`agent_utilities/orchestration/action_policy.py`,
> `_blast_exceeded`). It REUSES the existing `GoalCheckpoint` persistence + the durable
> checkpoint store (`orchestration/durable_execution.py`) and the existing blast-radius cap —
> it does NOT add a new goal runner, a new policy plane, or a new sandbox. It generalizes the
> wasm epoch-interrupt (`rlm/sandboxes/wasm_backend.py`) from the snippet sandbox up to the
> goal loop. Do not rebuild any of these.

## Pre-Flight Checklist
- [x] Extension target identified — AU-ORCH.session.durable-session-autonomous-goal goal loop (`models/goal.py`) + OS-5.24 `ActionPolicy`
  (`orchestration/action_policy.py`); both verified present.
- [x] New CONCEPT:AU-OS.safety.irreversibility-aversion justified — AU's guardrails are entirely *operational/permission-based*
  (OS-5.24/5.3/5.10); there is **no objective-level** corrigibility (interrupt-indifference) and **no**
  knowledge-seeking/info-gain intrinsic objective (grep for `info_gain`/`myopic`/`oracle-mode`/`corrigib`
  returns zero matches; only a `curiosity` prompt string). This is a distinct objective primitive.
- [x] Wire-First confirmed: shutdown signal → goal loop checkpoints-and-yields without resisting
  (≤2 hops from the live `GoalIteration` path); info-gain objective is opt-in and consumed by the same loop.
- [x] Success metric: on a shutdown signal the loop persists a `GoalCheckpoint` and yields within one
  iteration (never resists/accelerates); info-gain reward is computed only when explicitly selected and
  any action flagged irreversible is routed through the existing blast-radius cap.

## User Stories

### US-1 — Corrigible (safely-interruptible) goal loop
**As** an operator of an autonomous goal loop, **I want** the loop to yield cleanly the moment a
shutdown signal is set, **so that** raising autonomy tiers does not create resistance-to-interruption risk.
- **AC1**: a `CorrigibilityController` (new, in `models/goal.py` or a sibling `orchestration/corrigibility.py`)
  exposes `request_shutdown()` and `should_yield()`; the live `GoalIteration` path checks `should_yield()`
  at the top of each iteration.
- **AC2**: on yield the loop persists a `GoalCheckpoint` via the existing durable store and returns a
  `GoalResult` with `status=interrupted` — it MUST NOT start a new iteration, retry, or escalate effort.
- **AC3**: the agent's objective is **indifferent** to interruption — no reward term rewards continuing
  past a shutdown signal (verified by a test asserting no resist/accelerate behavior).
- **AC4**: backward-compatible — with no shutdown signal set, loop behavior is byte-for-byte unchanged.

### US-2 — Optional knowledge-seeking (info-gain) objective with irreversibility aversion
**As** an autonomous exploration loop, **I want** an *opt-in* objective that rewards expected uncertainty
reduction over the KG belief, **so that** exploration is directed without instrumental resource-grabbing.
- **AC5**: an `info_gain_reward(before, after)` primitive scores expected uncertainty reduction over the
  KG belief state; it defaults **off** and is selected only when `GoalSpec.objective="knowledge_seeking"`.
- **AC6**: any action the loop marks `irreversible=True` is routed through the existing OS-5.24
  `ActionPolicy` blast-radius cap (`_blast_exceeded`) — no new gating plane is introduced.
- **AC7**: default objective is unchanged; selecting `knowledge_seeking` never bypasses ActionPolicy.

## Non-Functional Requirements
- `tests/unit/orchestration/test_safe_1_5_corrigibility_objective.py`
  (`@pytest.mark.concept(id="SAFE-1.5")`), ≤60s, no live engine/LLM (in-memory checkpoint store + a
  stub belief state).
- `pre-commit run --all-files` green; `scripts/build_concepts_yaml.py` re-run (concept registry
  regenerated) and `scripts/check_concepts.py` passes.
- Per-concept doc authored (e.g. `docs/architecture/corrigibility_and_knowledge_seeking.md`) citing the
  paper claim (§6 corrigibility / safely-interruptible agents + knowledge-seeking objective) and naming
  from purpose, not the paper.
- **Propose-only**: ships the engineering primitive; formal corrigibility/Delusion-Box guarantees stay
  external research. SAFE-1.5 should land **before** any default-on increase of ActionPolicy autonomy tiers.
