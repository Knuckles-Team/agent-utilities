# Spec: AHE-3.1 Reward-Primitive Hardening (Dr.GRPO + DAPO + EP-GRPO + TR-GRPO)

> **Status: PROPOSED (banked).** Extends existing **AHE-3.1** (Continuous Evaluation & Reward
> Signals) — no new concept. Sources: Dr.GRPO (arXiv:2503.20783), DAPO (arXiv:2503.14476),
> EP-GRPO (arXiv:2605.04960), TR-GRPO (arXiv:2511.00066). Rank 4 in `COMPARATIVE_ANALYSIS.md`.

> **Why "banked":** these are token-level on-policy *trainer* mechanics. They only pay off
> when we run an actual policy-gradient trainer over the AHE-3.1 reward primitives. Per the
> repo's Wire-First mandate we do NOT implement speculatively; this spec defines the primitives
> so they're ready, and each ships only when a live trainer path consumes it.

## Pre-Flight Checklist (Mandatory — DSTDD)

- [ ] **KG search completed** — `.specify/design/ahe31-reward-primitive-hardening/design.md` exists
- [ ] **Extension point identified** — extends `graph/training_signals.py` (existing AHE-3.1; no new concept tag)
- [ ] **C4 diagram created** — only if topology changes (likely N/A — internal primitive upgrades)
- [ ] **`code-enhancer` audit** run against proposed changes
- [ ] **Design validation passes** — `SDDManager.validate_design("ahe31-reward-primitive-hardening")`

## Existing anchors (reuse, don't reinvent)

- `agent_utilities/graph/training_signals.py` — `batch_normalized_advantage()`, `composite_reward()`,
  `difficulty_floor_filter()`.
- `agent_utilities/graph/reward_decomposition.py` — step/trajectory decomposition.
- `epistemic-graph` `cross_sectional_rank()` / `rolling_zscore()` — the normalization kernels.

## User Stories (each independently shippable when a consumer exists)

### US-1: Length-unbiased normalization (Dr.GRPO)
**Acceptance:** `batch_normalized_advantage(..., length_unbiased=True)` normalizes against a fixed
max/completion length so short answers aren't over-rewarded nor long traces penalized. Unit test
vs. a known length-bias example. **Single most surgical, clearly-correct upgrade.**

### US-2: Dynamic sampling — drop zero-variance groups (DAPO)
**Acceptance:** `dynamic_sample(groups)` removes reward groups with ~zero variance (no gradient)
before advantage; logged drop count. Composes with EP-GRPO's zero-variance-collapse target.

### US-3: Entropy-progress step reweighting (EP-GRPO)
**Acceptance:** an entropy-progress signal (entropy delta across reasoning steps) reweights
`reward_decomposition` step weights toward steps that advance the solution; off by default until a
consumer wires it.

### US-4: Per-token regulation (TR-GRPO) + decoupled clip (DAPO)
**Acceptance:** token-contribution weights extend `composite_reward`; decoupled clip-higher/
clip-lower bounds available as parameters.

## Non-Functional Requirements
- [ ] Zero regression; new params default to current behavior (opt-in), so nothing changes until consumed.
- [ ] Tests < 60s; pre-commit clean; no stubs; no dead code (each primitive ships with a consumer or stays in this spec).

## Post-Modification Artifact Mandate (all 7 required before merge)
1. [ ] **/docs** — update `docs/pillars/3_agentic_harness_engineering/` AHE-3.1 page with the new primitives
2. [ ] **AGENTS.md** — note expanded reward primitives (provenance in docstrings, not identifiers — see NAMING)
3. [ ] **CHANGELOG.md** — Unreleased entry citing arXiv:2503.20783, 2503.14476, 2605.04960, 2511.00066
4. [ ] **README.md** — AHE-3.1 description refresh
5. [ ] **.specify/** — spec + tasks + design; dual-write to KG
6. [ ] **.specify/reports/** — C4 only if topology changes (else mark N/A with rationale)
7. [ ] **Pytests** — unit per primitive (length-unbias, dynamic-sample, entropy-progress, token-regulation)
