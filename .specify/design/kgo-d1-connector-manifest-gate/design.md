# Design Document: A connector's manifest is compiled, signature-checked, and hash-reverified BEFORE `source_sync` pulls its data — not just at manifest-apply time

CONCEPT:AU-KG.ontology.connector-manifest-gate

> `agent_utilities/knowledge_graph/ontology/connector_manifest_gate.py`,
> wired into `agent_utilities/knowledge_graph/core/source_sync.py:1342, 4560`.

## Decision — wire the manifest's compile+signature+hash checks into the LIVE `source_sync` entrypoint, not just a standalone CLI/CI check

`connector_manifest_gate.py:1-13` states the wiring directly (labeled D17):
this module "Wires the C5 Connector Ontology Manifest ... into the live
`source_sync.sync_source` entrypoint: before a source's data is pulled, its
owned `connector_manifest.yml` is required, compiled, release-signature
checked, and its `provenance.integrity.hash` re-verified. The boundary fails
closed on a missing or hand-edited manifest, a missing/changed installed
preset provider, or a changed server/tool contract." The module explicitly
shares its check function with the offline surface — "this module is the
`source_sync` wiring leg (D17); the CLI sweep gate lives in
`scripts/check_connector_manifests.py` and shares the same
`check_manifest_bytes`/compile path so both surfaces agree" — a single
verification implementation consumed from two call sites, not two
independently-maintained checks that could drift apart.

**The rejected alternative is a CI-only gate** — `check_connector_manifests.py`
alone, run in the pipeline but never consulted at actual sync time. That would
leave a real gap: a connector could pass CI with a valid signed manifest, and
then have its manifest hand-edited (or its installed preset provider silently
changed/downgraded) between CI and the actual production `source_sync` call,
with nothing at RUNTIME catching the drift. Wiring the identical check into
`sync_source` itself closes that window — every sync, not just every CI run,
re-verifies the manifest is what it claims to be.

## Risk Assessment

- **Blast Radius**: `knowledge_graph/core/source_sync.py` (every connector
  sync call), `knowledge_graph/ontology/connector_manifest_gate.py`,
  `scripts/check_connector_manifests.py`.
- **Backward Compatible**: Yes for compliant connectors (a valid signed
  manifest passes silently); fails closed for a connector with no manifest,
  a hand-edited one, or a changed preset/tool contract — which is the
  intended behavior change from "unchecked" to "gated".
- **Known weak point**: `MANDATORY_NAMED_CONNECTOR_SOURCES` /
  `INTERNAL_MANIFEST_EXEMPT_SOURCES` are curated allowlists
  (`connector_manifest_gate.py:34-49`) — a new connector source that should be
  gated but isn't added to the mandatory set would sync ungated, silently,
  until someone notices it's missing from the list.
