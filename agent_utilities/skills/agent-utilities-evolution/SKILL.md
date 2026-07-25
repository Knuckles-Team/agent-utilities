---
name: agent-utilities-evolution
skill_type: skill
description: >-
  Evolve agent-utilities from evidence to a reviewed proposal and measured
  improvement. Use for research assimilation, gap analysis, proposal generation,
  skill or prompt optimization, regression-driven hardening, concept
  traceability, or controlled improvement loops before implementation approval.
  For a concrete approved repository change or diff review, use
  agent-utilities-development.
---

# Agent Utilities evolution

Turn new evidence or runtime failures into a bounded proposal, validate it
against the existing system, and promote only improvements that pass review and
regression gates.

Keep evidence triage, proposal, and evaluation ownership here; hand concrete
approved implementation and diff review to `agent-utilities-development`.

## Workflow

### 1. Select the signal

Start from one concrete signal: a research result, evaluation gap, repeated
failure, operator correction, or stale capability. Query the existing knowledge
graph before acquiring more material so the loop does not rediscover completed
work.

Use the skill directly for bounded evidence triage or a proposal. Delegate when
retrieval, adversarial review, implementation, and regression validation can run
as independent or dependency-ordered work.

### 2. Assemble and assess evidence

- Ingest new sources with `graph-ingestion-and-integration` when necessary.
- Use `graph-research-and-analysis` to compare the evidence with current code,
  concepts, tests, and measured behavior.
- Separate demonstrated gaps from interesting but unsupported ideas.
- Record provenance, uncertainty, and expected leverage.

### 3. Propose before implementing

Create a reviewable proposal with:

- problem and evidence;
- affected live path and consumers;
- intended behavior and non-goals;
- safety, migration, and rollback considerations;
- tests, documentation, and acceptance gates.

Do not turn mined patterns or generated text directly into production changes.

### 4. Route the work

Use `graph-orchestration-and-automation` for a multi-stage loop. Run deterministic
extraction, classification, and formatting with an economical model class. Use
stronger reasoning for comparative judgment, architecture, adversarial review,
and final synthesis.

Use `graph_evolution` for assimilation, a code-correctness/security `audit_scan`
(each Macroscope-class finding files one canonical `:Gap`), skill distillation,
standardization, failure ingestion, component optimization, and proposal
publication. Use `graph_rlm` for confined long-context execution, its benchmark
scoreboard, and `evolve_prompt` (the GEPA reflective, DW-GRPO-weighted evolutionary
optimizer for an RLM/PredictRLM signature's prompt).

For prompt, skill, routing, or extraction optimization, use the engine-owned
`ProgramOptimize` job. It is the sole optimization path for that generic
system-prompt/tool-description/skill-content surface; record its opaque job and
candidate references in proposal evidence without provider endpoints or
credentials. `graph_rlm`'s `evolve_prompt` is the separate, non-overlapping
RLM-specific evolutionary optimizer above — the two are not the same mechanism.

Every discovery track (failure/research/skill/audit) files its findings as the
SAME canonical `:Gap` (the unified Gap->SDD->Implement->Promote->Close spine).
Use `graph_loops` for the whole lifecycle: `gaps` (the open backlog, highest
priority first), `submit_gap` (file one by hand), `gap` (one gap plus its
provenance — the SpecProposal it was specified by, the develop-Loop that resolves
it), `run`/`drive` to advance a gap's derived Loop, and `state`/`specs`/`review`
for transparency and steering. Use `graph_governance`'s `ownership_report` /
`ownership_apply` for the read-only per-graph RBAC-ownership disposition pass
(W2.8) — the live mutating grant apply itself stays a separate, human-gated CLI
(HG-4), never this or any other network surface.

Implement approved changes with `agent-utilities-development`. Keep skill,
prompt, routing, and extraction optimizations tied to a held-out evaluation or
clear regression test.

### 5. Validate improvement

- Compare against the pre-change baseline.
- Include negative and edge cases.
- Verify the live invocation path, not only the optimized artifact.
- Reject a change that moves one score while violating safety, grounding, cost,
  or latency constraints.
- Record failures and corrections as inputs to a later loop.

### 6. Publish the outcome

Report what changed, evidence supporting it, tests and evaluations run,
uncertainty, and any blocked follow-up. Promotion, merge, deployment, or external
publication remains subject to its normal review boundary.

## Guardrails

- Keep autonomous discovery and proposal generation separate from approval and
  production mutation.
- Do not fabricate research coverage, evaluation scores, or implementation
  evidence.
- Bound loop depth, fan-out, cost, and recurrence.
- Do not use live secrets, private endpoints, personal data, or machine-specific
  paths in evaluation material.
