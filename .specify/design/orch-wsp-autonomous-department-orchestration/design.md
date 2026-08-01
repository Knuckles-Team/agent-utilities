# Design Document: Swarm topology is derived from the company's OWL org chart, not hand-authored per department

CONCEPT:AU-ORCH.execution.autonomous-department-orchestration

> `agent_utilities/graph/manifest_generators.py` (`manifest_from_department()`,
> `manifest_for_enterprise()`); `agent_utilities/models/company.py`
> (`AgentDepartment`, the OWL-mapped node this is materialized from). Also
> referenced (non-authoritative) in `docs/journey.md:85` and
> `docs/pillars/1_graph_orchestration/ORCH-1.8-Parallel_Engine.md`.

## Decision — department manifests are generated generically from the KG's OWL company ontology, by department name, not hand-authored per department

`manifest_generators.py`'s own docstring lists seven ways an
`ExecutionManifest` can be produced (lines 11-18): from an HTN planner plan,
from a KG `TeamComposition`, from a skill workflow, for heavy-thinking
fan-out, from a **named preset** (`manifest_from_preset`), from an
**OWL-materialized company department** (`manifest_from_department`), and a
**full-enterprise** manifest across all departments
(`manifest_for_enterprise`). The last two are what this concept covers.

`manifest_from_department()` (248-322) is generic by department name: it
runs one Cypher pattern —
`(d:Department {name})-[:HAS_AGENT_ROLE]->(r:AgentRole)` with
`OPTIONAL MATCH` legs for `USES_TOOL`/`REPORTS_TO` (274-282) — against
whatever the KG's OWL company ontology currently holds for that department
(`AgentDepartment`, `company.py:118-125`, "Maps to OWL class :AgentDepartment
in ontology_company.ttl"), and turns each agent role's `REPORTS_TO` edge
directly into a `depends_on` dependency edge (285-297), so the generated
manifest's execution order mirrors the real reporting hierarchy rather than
an author's guess at it. `manifest_for_enterprise()` (328-415) is explicitly
"the 300-agent case" (docstring line 336): it enumerates every `Department`
node the KG has (350-359), falls back to a hardcoded department name list
*only* when the KG has none yet (366-376), builds one `manifest_from_department()`
call per department, and stitches the results together through a **static**
inter-department DAG (`dept_deps`, 378-385) reflecting real corporate
reporting order — e.g. `QA` depends on `Engineering`; `Compliance` depends on
`QA`, `Finance`, and `Operations`.

**The rejected alternative** is visible by contrast with the neighboring
generator in the same file: keep department composition as another
hand-authored `manifest_from_preset()`-style named team, or a manually
maintained per-department Python literal. That does not scale to "all
departments, all agents" — the enterprise case the docstring calls out by
name — and it goes stale the moment the org chart in the KG changes: a new
hire (a new `AgentRole` node) or a re-org (a changed `REPORTS_TO` edge)
would need a matching code change instead of simply being picked up by the
next `manifest_from_department()` call. The design accepts one deliberate
exception to "everything sourced from the KG": the *cross*-department
dependency table (`dept_deps`) is still a static Python dict, not
KG-derived — topology *within* a department is fully dynamic; topology
*across* departments is not (yet).

Graceful degradation is coded explicitly rather than left to fail: when a
department has no `AgentRole` nodes in the KG yet, `manifest_from_department()`
falls back to a single generic `"{department}_executor"` agent (304-313)
instead of raising or returning an empty manifest — the department is still
runnable, just as a single node rather than a hierarchy, until it is staffed
in the KG.

## Risk Assessment

- **Blast Radius**: `agent_utilities/graph/manifest_generators.py`,
  `agent_utilities/models/company.py`; documentation references in
  `docs/journey.md` and `docs/pillars/1_graph_orchestration/ORCH-1.8-Parallel_Engine.md`
  are descriptive only, not code.
- **Backward Compatible**: Yes — a department absent from the KG still
  produces a runnable (single-executor) manifest.
- **Known weak point**: `dept_deps` (the inter-department DAG,
  `manifest_generators.py:378-385`) is a hardcoded Python dict, unlike every
  other piece of this decision, which is KG-sourced. A department newly
  added to the KG needs a matching manual entry in `dept_deps` or it enters
  `manifest_for_enterprise()` with no inter-department dependency edges at
  all — a silent topology gap, not an error.
