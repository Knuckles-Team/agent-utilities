# Design Document: A package install auto-extends the KG by re-driving existing ingestion primitives off a manifest change-signal, never by reimplementing ingestion

CONCEPT:AU-KG.ingest.package-install-autoingest

> `agent_utilities/knowledge_graph/ingestion/package_install_ingest.py:1-50`
> (primary), `agent_utilities/knowledge_graph/core/source_sync.py:4339-4350`
> (thin `source_sync` wiring), `agent_utilities/knowledge_graph/core/engine_tasks.py:386-391`
> (the poll-cadence schedule entry).

## Decision — the graph-os-side consumer reads the universal-installer's manifest purely as a change signal and re-drives the THREE existing ingestion primitives (prompts/ontologies/skills), rather than parsing the manifest into new KG nodes itself

`package_install_ingest.py:1-22` frames this as closing a deliberately
split loop: the universal-installer already materializes a newly-installed
package's skills/prompts/ontology into the unified XDG tree and, on change,
drops a summary manifest (`install-manifest.json`). The installer's own
`SKILL.md` "explicitly documents this as a deliberate half-measure — the
installer makes the artifacts *discoverable*... but never itself parses the
manifest, mints KG nodes, or calls `source_sync` ('that's the graph-os/
epistemic-graph side's job'). **This module is that graph-os-side
consumer.**"

**The rejected alternative, named explicitly in the module's own "Design —
reuse, never reimplement" section** (`package_install_ingest.py:24-42`), is
for this module to parse the manifest and mint KG updates itself — diffing
file paths, building its own ingestion logic per leg. It is rejected on two
counts:

1. **The manifest is read purely as a change signal** (its content hash is
   the dedup watermark via `DeltaManifest`); "the manifest does NOT
   enumerate individual file paths, so this module does not try to diff
   files itself." A reimplementation would have to invent that
   file-enumeration capability the manifest deliberately doesn't provide.
2. **Each leg re-drives an existing, independently idempotent primitive**
   instead of a new one: prompts → `registry_builder.ingest_prompts_to_graph`
   (the same reload the CLI/boot path uses); ontologies →
   `ontology_tools._sync_package_ontologies` (the same reload
   `graph_ontology action='sync_packages'` and graph-os boot already call);
   skills → `skill_workflow_ingest.ingest_skill_workflows` (the one existing
   "*.md skill corpus → KG" path). Because each is upsert-keyed by stable
   id/content hash, re-running them on an unrelated manifest change costs a
   no-op write, not duplicate/drifted state — a property a bespoke
   reimplementation would have to earn independently rather than inherit.

The module is explicit about the gap it does NOT paper over: "the manifest
itself does not yet itemize atomic (non-workflow) skills per provider — a
documented upstream gap in the installer, not something this module papers
over with new ingestion logic" — a second instance of the same "reuse, never
reimplement" discipline: rather than build a workaround for a gap in an
upstream contract, the gap stays visible and unaddressed here.

**Integration decisions, thin by design**: registered as source
`"package_install"` in `source_sync._DELTA_HANDLERS`
(`source_sync.py:4339-4350`) so it "rides the ONE existing `source_sync` MCP
tool / REST twin... no new tool or route," and polled via a cheap
manifest-hash check on a 300s interval (`engine_tasks.py:387-391`) that
"dedup no-ops when nothing installed since last tick, so it can poll far
more often than a heavy sweep without wasted work."

## Risk Assessment

- **Blast Radius**: `knowledge_graph/ingestion/package_install_ingest.py`,
  `core/source_sync.py` (the `_sync_package_install` handler),
  `core/engine_tasks.py` (the poll schedule entry).
- **Backward Compatible**: Yes — additive; re-running on an unchanged
  manifest is a no-op by construction.
- **Breaking Changes**: None.
- **Known weak point**: the documented upstream gap (atomic skills not
  itemized per provider in the manifest) means this consumer's coverage is
  bounded by what the installer's manifest actually enumerates — a category
  of installed content the installer doesn't manifest is invisible to this
  auto-extend loop entirely, by design, not by oversight here.
