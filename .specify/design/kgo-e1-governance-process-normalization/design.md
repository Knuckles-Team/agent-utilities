# Design Document: Every governing process — and every connector's mapped work — is translated into ONE ontology-driven representation, never handled per-connector

CONCEPT:AU-KG.ontology.connector-agnostic-proposal ·
CONCEPT:AU-KG.ontology.connector-skill-proposal-shapes

> `agent_utilities/knowledge_graph/governance_import.py`,
> `agent_utilities/knowledge_graph/distillation/skill_synthesizer.py`,
> `agent_utilities/knowledge_graph/ontology/interfaces.py:894-955`.

## Decision — the keystone reframe: any governing process becomes a `:WorkflowDefinition`/`:WorkflowStep` DAG; any connector's mapped work becomes a propose-only skill candidate — both driven by the shared ontology classes, never a per-connector adapter

`governance_import.py:3-11` names this "the keystone reframe" (design doc
`reports/autonomous-sdlc-loop-design.md` §6/§7.1, decision #4): "every
governing process — Camunda BPMN, ARIS EPC, ArchiMate business process,
OneTrust/ERPNext approval flow — is translated INTO the fleet's one
executable process language, the `:WorkflowDefinition`/`:WorkflowStep` DAG
... so an agent can execute any of them via `WorkflowRunner`." Approval/
sign-off constructs become `kind="gate"` steps uniformly, and a `REALIZES`
back-edge records the descriptive→executable bridge for lineage. Export is
symmetric: `export_workflow` serializes a stored `:WorkflowDefinition` back to
BPMN 2.0 XML, JSON, or a SKILL.md — "the round-trip the fleet was missing."

The MACHINE TRIAGE TOOL flagged this id "review" because the marker text at
one site is truncated by the parsing grammar — but reading both cited files
confirms the underlying decision is real, deliberate, and load-bearing (the
same pattern seen with `CONCEPT:AU-KG.ontology.do-not-auto-merge` and
`CONCEPT:AU-KG.ontology.kyle-insider-stealth-surveillance` in this domain: a
marker-shape heuristic is not a substitute for reading the site).

**The rejected alternative is a bespoke importer/executor per governance
tool** — a Camunda-specific runner, an ARIS-specific runner, an ArchiMate-
specific runner, each understanding its own process semantics natively. That
would mean adding a new governance source requires teaching the *executor*
about a new process model. Instead, each importer's ONLY job is translating
its source's native shape into the ONE shared DAG (`ProcessPlanCompiler` reuse
for Camunda BPMN; walking `:EPCFunction`/`:flowsTo` for ARIS; walking
`Triggering`/`Flow` for ArchiMate; a single `kind="gate"` step for
OneTrust/ERPNext's simpler approval shape) — `WorkflowRunner` itself never
changes. Every importer is explicitly best-effort/guarded: "a missing source
subgraph, an absent engine, or a malformed model degrades to an error dict
rather than raising," and gate detection is one shared heuristic
(`looks_like_gate`) over node type/name, not reimplemented per source.

`skill_synthesizer.py:1-34` applies the identical principle to a different
artifact: connector→skill synthesis. The distiller "is generic over the
ONTOLOGY (`BusinessProcess`/`BusinessTask`/`flowsTo`/`Capability`), never
per-connector: every connector already normalizes into those same
ArchiMate/capability classes, so one ontology-driven pass covers them all" —
egeria, leanix, aris, and camunda all feed the SAME discover→classify→dedup→
propose→draft_artifact pipeline with no connector-specific branch. The
pipeline is propose-only and non-destructive: `propose()` writes proposal
nodes and provenance edges and "NOTHING lands in any repo" until a human/Claude
approves; materialization only ever writes to a staging dir, never a source
repo.

### Pointer — `CONCEPT:AU-KG.ontology.connector-skill-proposal-shapes`

`interfaces.py:894-955`. The proposal artifacts this distiller produces
(`SkillProposal`, `SkillWorkflowProposal`) are themselves modeled as ontology
Interfaces, not opaque JSON blobs — the comment states why: "making these
ontology interfaces lets the reasoner extrapolate over
automates/derived_from/composes so a proposal is related to the process it
automates and the source it came from, not treated as opaque text."
`SkillProposal` extends `HasProvenance` "so every proposal records what
source it was distilled from," with `AUTOMATES`/`DERIVED_FROM` link
constraints (`min_count=0` — declared but not mandatory, since a fresh
proposal may not yet have both edges written). **The rejected alternative is
a plain data record** (a dict/row) for each proposal — reachable only by
direct id lookup, with no way for OWL reasoning to answer "what proposals
came from this source" or "what processes does this candidate automate"
without a bespoke query; modeling the shape as an Interface makes those
questions ordinary reasoning.

## Risk Assessment

- **Blast Radius**: `knowledge_graph/governance_import.py`,
  `knowledge_graph/distillation/skill_synthesizer.py`,
  `knowledge_graph/ontology/interfaces.py` (`SkillProposal`/
  `SkillWorkflowProposal`), `WorkflowRunner`.
- **Backward Compatible**: Yes — both are additive translation/proposal
  layers; nothing auto-applies without human/Claude review.
- **Known weak point**: `looks_like_gate`'s heuristic (approval/review/
  sign-off/authorize/DPIA keyword matching over node type/name) is shared
  across every importer — a governance tool whose approval nodes don't match
  those keywords (a differently-worded custom approval step) would silently
  fail to be detected as a gate, degrading to an ordinary step with no
  suspend/resume semantics.
