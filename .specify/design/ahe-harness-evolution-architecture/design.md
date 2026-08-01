# Design Document: AHE reasons over a distilled EvidenceCorpus and a three-layer state split, never raw logs in one undifferentiated store

CONCEPT:AU-AHE.harness.harness-evolution

> `agent_utilities/harness/evolve_agent.py:1-22` (primary),
> `docs/guides/AHE_ARCHITECTURE.md:1-30`.

## Decision — Epistemic/Normative/Causal are three separate state layers with three separate storage mechanisms, and the Evolve Agent reads distilled evidence, never raw traces

`AHE_ARCHITECTURE.md` names the hybrid state model directly: **Epistemic**
(what the agent knows) lives in `IntelligenceGraphEngine` + MAGMA views
(`knowledge_graph.db`); **Normative** (what the agent is allowed to do) lives
in component files — prompts, middleware, tools — on the filesystem plus
git; **Causal** (what caused an improvement) lives in Change Manifests
(`.specify/manifests/` + KG). `EvolveAgent`'s own docstring
(`evolve_agent.py:18-21`) makes the read-side of this explicit: "Reads:
EvidenceCorpus (distilled traces, NOT raw logs). Writes: ComponentEdits +
ChangeManifest. Uses: KG for epistemic state, files for normative state, git
for causal boundary."

**The rejected alternative, named explicitly by the "NOT raw logs"
qualifier, is reasoning directly over raw production traces/logs.** Raw logs
are noisy and unstructured; the Evolve Agent instead consumes a
**distilled** `EvidenceCorpus` — clustered, summarized evidence — so
component-level failure attribution operates on signal, not on
re-deriving structure from scratch on every round. A second rejected
alternative is collapsing all three state kinds into one store (e.g.
treating "what changed" and "why it's allowed" and "what we know" as all
just KG facts, or all just files): the docstring's explicit three-way split
means a causal claim (a Change Manifest's falsifiable prediction) is
distinguishable from normative state (the actual prompt/tool files) and from
epistemic state (accumulated knowledge), so each can be reasoned over,
audited, or rolled back independently. Both modes of the agent (lightweight
in-graph specialist vs. full background evolution round) spawn from the
single agent server — "no external deployment dependencies" — rather than
the background mode requiring its own separate deployment.

## Risk Assessment

- **Blast Radius**: This is the umbrella architecture for the whole AHE
  pillar — 18 source files across `agent_utilities/harness/`,
  `agent_utilities/knowledge_graph/`, and `agent_utilities/models/` build on
  this three-layer split, notably `evolve_agent.py`, `manifest.py`,
  `evidence_corpus.py`, `trace_backend.py`, `component_registry.py`,
  `verifier.py`, `constraint_engine.py`, `replay_buffer.py`.
- **Backward Compatible**: N/A — this is the foundational architecture
  document for the pillar, not a point change.
- **Known weak point**: the three-layer split relies on every new AHE
  subsystem correctly classifying its own state as epistemic, normative, or
  causal; a subsystem that writes causal-shaped data (a claim about what
  caused an improvement) into the epistemic store instead of a Change
  Manifest would blur the audit boundary the split exists to preserve, with
  nothing structural to catch the misclassification.
