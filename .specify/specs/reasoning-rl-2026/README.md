# SDD Proposals — Reasoning RL 2026

**Execution roadmap:** [`ACTIONABLE_PLAN.md`](./ACTIONABLE_PLAN.md) — the sequenced, dependency-ordered
plan (Waves 1–3) that turns the analysis verdicts into per-task work with files, acceptance criteria,
and effort. Start here to execute.

Proposal-stage specs derived from [`COMPARATIVE_ANALYSIS.md`](./COMPARATIVE_ANALYSIS.md)
(15 papers + survey on the 2026 reasoning-RL landscape vs. `agent-utilities` /
`epistemic-graph`). **Status: PROPOSED — not implemented.** These are grounded in
code merge points; the KG assimilation pass (`graph_orchestrate action=assimilate
task=synthesize`) will reconcile/rank them once paper ingestion + enrichment finish.

Ranked by leverage for our **agentic / KG-driven** architecture (we are not running an
on-policy base-model trainer — see the analysis "Framing" section).

| Rank | Spec | Concept | Primary papers | Verdict closed |
|---|---|---|---|---|
| 1 | [agent-step-policy-optimization](./spec-arpo-agent-step-po.md) | AU-AHE.optimization.telemetry-optimization | ARPO (2507.19849) | partial→gap, HIGH |
| 2 | [test-time-diversity](./spec-vpo-test-time-diversity.md) | AU-AHE.optimization.telemetry-optimization | VPO (2605.22817) | partial→gap, HIGH |
| 3 | [preference-corpus-reliability](./spec-preference-corpus-reliability.md) | AU-AHE.optimization.telemetry-optimization | RAPPO (OR LrHfYPFTtg), TI-DPO (2505.19653), InSPO (2512.23126), DPO (2305.18290) | partial, broadly enabling |
| 4 | [reward-primitive-hardening](./spec-ahe31-reward-primitive-hardening.md) | AHE-3.1 | Dr.GRPO (2503.20783), DAPO (2503.14476), EP-GRPO (2605.04960), TR-GRPO (2511.00066) | partial/net-new, banked |

**Already covered (no spec needed):** GRPO (2402.03300), REINFORCE++ (2501.03262),
DHPO (2601.05607) — implemented in `training_signals.batch_normalized_advantage()` and
`reward_decomposition.py`. Cited as prior art in the relevant docstrings.

**Constitution note.** Each spec carries the 7-artifact Post-Modification Mandate checklist
(docs / AGENTS.md / CHANGELOG / README / .specify sync / C4 / pytests). Before any spec is
promoted to implementation, its `.specify/design/<feature>/design.md` + C4 diagram must exist
and `SDDManager.validate_design()` must pass (DSTDD pre-flight).
