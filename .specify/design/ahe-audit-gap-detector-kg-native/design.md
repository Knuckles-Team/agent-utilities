# Design Document: Security-audit review runs over the already-ingested code KG, not a fresh scan

CONCEPT:AU-AHE.harness.audit-gap-detector

> `agent_utilities/harness/audit_gap_detector.py`, feeding
> `agent_utilities/knowledge_graph/research/gaps.py` (`submit_gap`,
> `SOURCE_AUDIT`).

## Decision — the 4th discovery track reads `:CodeUnit`/`:Symbol` nodes from the graph; it never re-scans the filesystem

The module docstring (`audit_gap_detector.py:4-9`) states the decision
directly: this is Wave-6's D1-ext, "the Macroscope-level review, reframed as
engine-native." An AI review runs over the **already-ingested code KG** — all
codebases, specs, and git history are already ingested, so there is no fresh
scan; the detector "naturally evolves within the epistemic-graph." It looks
for seven Macroscope finding-classes (resource-lifecycle,
transaction-durability, id-consistency, serialization-validity,
secret-redaction, error-handling, audit-integrity — `audit_gap_detector.py:11-18`).

**The rejected alternative is a standalone security scanner that walks the
filesystem fresh on each run** (the conventional Macroscope-style tool shape).
That model has two costs this decision avoids: (1) duplicated ingestion work —
a fresh scan re-parses what the KG ingestion pipeline already parsed and
indexed; (2) the "block-and-forget" failure mode named explicitly in the
docstring — a standalone scanner that flags an issue and stops has no
lifecycle for that finding. Instead, every finding calls `submit_gap` to
create ONE canonical `:Gap` (source `audit`, severity mapped to priority) that
flows the same `Gap → SDD → code-synth → W2.7 → resolved` lifecycle as every
other discovery track, so a flagged issue becomes governed, spec'd, tracked
work rather than a report nobody re-visits.

A second, smaller decision lives in the same module: detection is **opt-in**
via `KG_LOOP_AUDIT`, not default-on. The flywheel proposes; a human vetoes.
The rejected alternative here is default-on autonomous detection — rejected
because the detector uses an LLM reviewer (local vLLM via the model factory,
`role="reviewer"`, escalating large/high-risk units to `role="critic"`), and
an LLM-driven finding stream running unattended by default is exactly the kind
of autonomy this codebase's governance model (human veto on flywheel
proposals) is built to avoid.

## Risk Assessment

- **Blast Radius**: `agent_utilities/harness/audit_gap_detector.py`,
  `agent_utilities/knowledge_graph/research/gaps.py`,
  `agent_utilities/mcp/tools/evolution_tools.py`,
  `agent_utilities/knowledge_graph/research/loop_controller.py`.
- **Backward Compatible**: Yes — opt-in via `KG_LOOP_AUDIT`; no effect on any
  pipeline until an operator enables it.
- **Known weak point**: detection quality is bounded by how completely the
  code KG has actually ingested the target codebase — a unit whose
  `:CodeUnit`/`:Symbol` nodes are stale or missing (source changed since the
  last ingestion pass) is invisible to this detector even though a
  filesystem-walking scanner would have caught it.
