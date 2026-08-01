# Design Document: A registered external graph is profiled and label-mapped on demand, not re-introspected by every caller that touches it

CONCEPT:AU-KG.ontology.external-graph-profiling

> `agent_utilities/knowledge_graph/core/connection_profiler.py`.

## Decision — `profile` (read-only schema introspection) and `map` (deterministic label→ontology-class mapping) let an agent discover an external graph's shape without re-querying it each time

`connection_profiler.py:1-19` states the extension point directly: once an
external Neo4j/FalkorDB/Postgres-AGE/Ladybug graph is registered via
`graph_configure add_connection`, this module "lets an agent *discover what's
in it and how to use it* without re-introspecting every time." `profile` is
"backend-portable: prefers the `db.*` procedures (Neo4j/FalkorDB), degrades to
a bounded sampled scan where they're unavailable." `map` "deterministically
maps each external label onto the closest class in our ontology vocabulary
... by name; the unmatched ones are flagged `novel` (candidates for a new
ontology class)."

**The rejected alternative is ad hoc, per-caller introspection** — every
consumer of an external graph connection running its own schema-discovery
queries against the live external system, with no shared cache and no
consistent mapping onto this platform's own ontology vocabulary. Centralizing
profile+map here means the discovery cost is paid once (governed by
`knowledge_graph.ingestion.external_graph_schema`), and the label→ontology
mapping is deterministic and name-based rather than ad hoc guessing per
caller — a `novel` flag surfaces genuinely-unmapped labels as ontology-
extension candidates instead of silently dropping them.

A privacy decision rides alongside: "raw discovery and mappings are stored
only by the governed workflow ...; public/catalog state contains only
digests, counts, and pseudonyms" (`connection_profiler.py:17-19`) — an
external graph's actual label/property names (which may encode
customer-specific or sensitive vocabulary) are not exposed in the general
catalog surface, only in the governed workflow's own storage.

## Risk Assessment

- **Blast Radius**: `knowledge_graph/core/connection_profiler.py`,
  `knowledge_graph/ingestion/external_graph_schema.py` (governed storage),
  the multi-connection registry (`graph_configure add_connection`).
- **Backward Compatible**: Yes — an additive discovery layer over registered
  connections; no connection behavior changes.
- **Known weak point**: the `db.*`-procedures-unavailable fallback is "a
  bounded sampled scan" — on a backend without introspection procedures, the
  profile is necessarily incomplete (labels/properties absent from the sample
  are simply not seen), so `map`'s `novel` flagging can under-report genuinely
  unmapped types that the sample happened to miss.
