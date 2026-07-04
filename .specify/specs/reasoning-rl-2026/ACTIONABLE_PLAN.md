# Actionable Plan — Closing the 2026 Reasoning-RL Gaps

Derived from [`COMPARATIVE_ANALYSIS.md`](./COMPARATIVE_ANALYSIS.md). Turns the verdict table
into a **sequenced, execution-ready roadmap** across `agent-utilities` and `epistemic-graph`.

## Framing (what we are and aren't building)

We are an **agentic harness + KG-driven evolution** system, not a base-model GRPO trainer. So we
close gaps that land on our **live mechanisms** — the capability router's reward-EMA, the
evaluation harness, test-time fan-out, and the distilled preference corpus — and we **defer**
the token-level trainer micro-mechanics until/unless we run an on-policy trainer.

- **Already covered (no work):** GRPO, REINFORCE++ (`training_signals.batch_normalized_advantage`),
  DHPO (`reward_decomposition`). Action = cite as prior art in docstrings only.
- **Explicitly deferred (banked, do NOT build now):** GSPO sequence-ratio, DAPO clip mechanics,
  DPPO rollout pruning *as trainer features*. They presume an on-policy token trainer we don't run;
  building them now = speculative dead code (violates the repo Wire-First mandate). Captured as
  AHE-3.1 primitive notes, shipped only when a trainer consumes them.

## Dependency graph (drives the wave order)

```
W1 ──► PreferencePair export (AU-AHE.harness.preference-corpus-reliability core) ──┐
       Dr.GRPO length_unbiased primitive       │ (foundation: corpus + a correct primitive +
       epistemic-graph group-relative kernel    │  the native math kernel — low-risk, enabling)
                                                 │
W2 ──► ARPO Agent-Step PO (AU-AHE.reward.this-is-read-back) ───────────┼─ independent, HIGHEST leverage, parallelizable
       VPO Test-Time Diversity (AU-AHE.harness.width-diverse-best-k) ───────┘
                                                 │
W3 ──► RAPPO reliability filter ─────────────────┤
       TI-DPO token weights        ── all LAYER ON the W1 PreferencePair export
       InSPO reflective conditioning ────────────┘
       AHE-3.1 hardening (dynamic_sample / entropy-progress / token-regulation) — banked
```

`W2` does not depend on `W1` and is the highest-value work — start it in parallel with `W1`.
`W3` (DPO-family refinements) **requires `W1`'s PreferencePair export** as substrate.

---

## Wave 1 — Foundation & surgical wins (low risk, broadly enabling)

### W1.1 — `PreferencePair` export  ·  **AU-AHE.harness.preference-corpus-reliability (core)**  ·  agent-utilities  ·  effort: M
Spec: [`spec-preference-corpus-reliability.md`](./spec-preference-corpus-reliability.md) (US-1).
- **Build:** a `PreferencePair{prompt, chosen, rejected, metadata}` model + an exporter that
  consolidates pairs from `harness/eval_corpus.py::EvalCorpus`, `knowledge_graph/adaptation/
  trace_distiller.py::EpisodeToPreferenceRule` (successful-vs-failed episode), and
  `knowledge_graph/adaptation/feedback.py::FeedbackService._apply_outcome` (human corrections).
  Incremental / content-addressed (no full-corpus recompute).
- **Files:** new `harness/preference_pairs.py`; wire into `eval_corpus.py` + `trace_distiller.py`.
- **Acceptance (Wire-First live-path):** record a human correction + one failed and one succeeded
  episode → assert a retrievable, deduped `PreferencePair` is produced *as a side effect of the
  existing distiller/feedback path* (not just a helper unit test).
- **Why first:** it's the substrate the entire DPO family (W3) layers onto.

### W1.2 — Dr.GRPO length-unbiased normalization  ·  **AHE-3.1 (extend)**  ·  agent-utilities  ·  effort: S
Spec: [`spec-ahe31-reward-primitive-hardening.md`](./spec-ahe31-reward-primitive-hardening.md) (US-1).
- **Build:** `batch_normalized_advantage(..., length_unbiased=True)` in `graph/training_signals.py`
  — normalize against a fixed max/completion length so short answers aren't over-rewarded nor long
  traces penalized. Also expose `mode={"group","global"}` (REINFORCE++ distinction).
- **Acceptance:** unit test vs a known length-bias example; default `False` ⇒ zero behavior change.
- **Why now:** the one trainer-primitive that is **correct regardless of a trainer** and surgical.

### W1.3 — epistemic-graph group-relative advantage kernel  ·  epistemic-graph  ·  effort: S–M
Analysis §Synthesis item 5.
- **Build:** promote the existing normalization (`FinanceClient.cross_sectional_rank()` /
  `rolling_zscore()`) to a named `group_relative_advantage` kernel + a `length_unbiased` variant,
  so the reward math runs natively in the Rust engine, off the Python hot path (batch over the
  wire — see [[epistemic-graph-transport]]). `agent-utilities` calls it from `training_signals`.
- **Acceptance:** parity test — Python `batch_normalized_advantage` vs the kernel on the same batch.

**Wave-1 exit:** PreferencePair export live + tested; `length_unbiased` available; kernel parity green.

---

## Wave 2 — Highest-leverage live-path wins (start in parallel with W1)

### W2.1 — Agent-Step Policy Optimization (ARPO)  ·  **AU-AHE.reward.this-is-read-back**  ·  agent-utilities  ·  effort: L  ·  **TOP**
Spec: [`spec-arpo-agent-step-po.md`](./spec-arpo-agent-step-po.md).
- **Build:** (a) a step-entropy/uncertainty signal at tool/decision boundaries exposed to
  `graph/routing/strategies/policy.py::SubagentLifecyclePolicy`; (b) entropy-gated branching above
  a threshold (bounded fan-out, telemetry); (c) per-agent-step advantage from
  `graph/reward_decomposition.py` written back into `retrieval/capability_index.py::record_outcome`.
- **Acceptance (Wire-First live-path):** run the router on a multi-step task → assert step-level
  outcomes are recorded as a side effect (not only a helper test); branching is bounded and cannot
  wedge the worker pool.
- **Why top:** best fit for our agentic architecture — it upgrades *when* and *at what granularity*
  the router (which we already run) branches and credits. Pure live-path.

### W2.2 — Test-Time Diversity (VPO)  ·  **AU-AHE.harness.width-diverse-best-k**  ·  agent-utilities + epistemic-graph  ·  effort: M
Spec: [`spec-vpo-test-time-diversity.md`](./spec-vpo-test-time-diversity.md).
- **Build:** a diversity score (embedding spread / reward-vector divergence) over the candidate set
  in the subagent fan-out (`SubagentLifecyclePolicy`, `rlm/` parallel sub-calls); selection trades
  quality vs diversity; width derives from `harness/reasoning_effort.py` (no new agent knob).
  `epistemic-graph::personalized_pagerank` as the seed-diverse diversity kernel.
- **Acceptance (Wire-First live-path):** a fan-out task yields a measurably more diverse candidate
  set than scalar-best sampling, asserted on the existing fan-out entry point; improves best@k/pass@k.

**Wave-2 exit:** ARPO step-credit + branching live on the router; VPO diversity live in fan-out;
both with live-path tests; no regression in the unit suite.

---

## Wave 3 — Preference-corpus refinements (layer on W1.1) + banked primitives

### W3.1 — RAPPO reliability filter  ·  **AU-AHE.harness.preference-corpus-reliability**  ·  agent-utilities  ·  effort: S
Spec: `spec-preference-corpus-reliability.md` (US-2). Depends on **W1.1**.
- **Build:** a `reliability_filter` over the `PreferencePair` export — drop ambiguous / low-margin
  pairs ("keep the best, forget the rest"); log dropped counts (no silent truncation).
- **Acceptance:** ambiguous pairs are excluded; drop count surfaced. **Cheap, broadly enabling.**

### W3.2 — TI-DPO token-importance weights  ·  **AU-AHE.harness.preference-corpus-reliability**  ·  agent-utilities  ·  effort: S–M
Spec: `spec-preference-corpus-reliability.md` (US-3). Depends on **W1.1**.
- **Build:** attach token-importance weights to a `PreferencePair` (pairs with TR-GRPO's
  per-token regulation idea on the reward side).

### W3.3 — InSPO reflective conditioning  ·  **AU-AHE.harness.preference-corpus-reliability**  ·  agent-utilities  ·  effort: S
Spec: `spec-preference-corpus-reliability.md` (US-3). Depends on **W1.1**.
- **Build:** an optional, off-by-default reflective-conditioning hook (condition the policy on an
  alternative response) on the `PreferencePair` export; reuse `capabilities/adversarial_verifier.py`
  as the critic and `rlm/` reflection.

### W3.4 — AHE-3.1 reward-primitive hardening (banked)  ·  **AHE-3.1**  ·  agent-utilities
Spec: `spec-ahe31-reward-primitive-hardening.md` (US-2/3/4). **Ship each only when a consumer
exists** (no speculative dead code): `dynamic_sample()` zero-variance drop (DAPO), entropy-progress
step reweighting (EP-GRPO), per-token regulation (TR-GRPO). Until a policy-gradient trainer consumes
them, these stay specified, not implemented.

**Wave-3 exit:** the distilled trace corpus is a clean, reliability-filtered, optionally
token-weighted DPO-ready `PreferencePair` set; banked primitives specified with consumers identified.

---

## Per-task constitution checklist (applies to every W item on merge)

Each task carries the 7 mandated artifacts (already enumerated in the linked specs): **/docs** pillar
page · **AGENTS.md** capability note · **CHANGELOG.md** Unreleased entry citing the paper · **README.md**
concept-count refresh · **.specify/** spec+tasks+design (dual-write to KG) · **.specify/reports/** C4
diagram · **Pytests** (unit + `*_live_path`). New concept IDs (`AU-AHE.reward.this-is-read-back/3.16/3.17`) must be added to
`docs/concepts.yaml` (single source of truth) and pass `scripts/check_concepts.py`.

## Suggested execution order (one PR per task)

1. **W1.1 PreferencePair export** + **W1.2 length_unbiased** (parallel; both low-risk) — unblocks W3.
2. **W2.1 ARPO** (start immediately, parallel with W1 — highest leverage, independent).
3. **W2.2 VPO** + **W1.3 epistemic-graph kernel**.
4. **W3.1 → W3.3** preference refinements (after W1.1 lands).
5. **W3.4** banked primitives — only alongside a real consumer.

## How this reconciles with the KG assimilation

Once `enrichment/` distils the ingested paper chunks into concept-features, the graph-native
`assimilate task=synthesize` pass should (a) **auto-satisfy** GRPO/REINFORCE++/DHPO against
AHE-3.1/`reward_decomposition` concepts, and (b) surface **ARPO / VPO / preference-reliability** as
the top-ranked open gaps — i.e. the same ranking this plan encodes by hand. Divergence = a signal to
refine the concept embeddings or this plan. The hand-authored plan is authoritative until then.
