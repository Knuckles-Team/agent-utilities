# Design Document: A connector's a2a.json is DERIVED from what it already ships, never hand-typed

CONCEPT:AU-KG.ontology.a2a-card-generation ·
CONCEPT:AU-KG.ontology.connector-manifest-generator

> `scripts/generate_connector_manifests.py`.

## Decision — the deterministic manifest generator projects a connector's a2a.json / ontology / mcp presets into ONE `connector_manifest.yml`, zero LLM calls, zero network

`CONCEPT:AU-KG.ontology.connector-manifest-generator`

`generate_connector_manifests.py:2-25` states the contract: "**ZERO LLM
calls, no network.** For a given connector package ... projects the artifacts
the connector ALREADY ships into a single declarative `connector_manifest.yml`":
`<module>/ontology/*.ttl` → resources/schema_mappings/relations,
`mcp_source_presets.json` → sync/identity/events, `a2a.json` → actions. Every
field that cannot be derived losslessly is filled with a "documented heuristic
default and flagged in `review_todos` — never silently guessed, never invented
by an LLM." Same input + timestamp + release key produces byte-identical
output — the generator is a pure projection function, not a generative one.

**The rejected alternative is an LLM-assisted or hand-authored manifest** —
either would make the manifest an independent, driftable source of truth
alongside the code it's supposed to describe. A deterministic projection
instead means the manifest can be regenerated and byte-compared against the
signed one at any time (the basis for `CONCEPT:AU-KG.ontology.supply-chain-integrity`'s
fail-closed re-verification, see `.specify/design/kgo-c1-connector-manifest-compiler/design.md`)
— an LLM-authored manifest could never support that invariant, since two runs
of an LLM are not guaranteed byte-identical.

### Pointer — `CONCEPT:AU-KG.ontology.a2a-card-generation`

`generate_connector_manifests.py:132-145, 167-180`,
`scripts/generate_connector_capability_bundles.py:404`. Two concrete
derivation decisions specific to the `a2a.json` capability card: (1)
`DEFAULT_A2A_CAPABILITIES` states every agent-utilities-built connector ships
the SAME two capabilities — `run_graph_flow` (universal to the framework) and
the shared `EPISTEMIC_CAPABILITY` ("every live AgentCard already advertises")
— rather than letting each connector hand-declare its own capability list and
risk declaring something it doesn't actually support; and (2)
`_resolve_a2a_version` resolves the connector's version from
`[project.version]` first, else `[tool.setuptools.dynamic] version.attr` —
"never a hand-typed a2a.json value, which is exactly what drifts"
(`generate_connector_manifests.py:170`). Both decisions eliminate a class of
value that could be typed once and never updated again.

## Risk Assessment

- **Blast Radius**: `scripts/generate_connector_manifests.py`, every
  `agents/<pkg>` connector's `connector_manifest.yml` and `a2a.json`.
- **Backward Compatible**: Yes — a generation tool over existing artifacts;
  no runtime behavior changes until a manifest is regenerated and applied.
- **Known weak point**: fields that genuinely cannot be derived losslessly
  (ontology-class crosswalk, PII/RLS policy) still rely on a human reviewing
  `review_todos` before the manifest is signed — the generator flags the gap
  honestly but does not close it.
