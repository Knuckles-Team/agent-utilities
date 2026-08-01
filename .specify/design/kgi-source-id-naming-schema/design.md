# Design Document: One hierarchical, colon-delimited source-id grammar for every connector — never per-connector free-form naming

> `agent_utilities/knowledge_graph/backends/sparql/source_partition.py:35-45`,
> `:125-132` (`make_source_id`).

CONCEPT:AU-KG.ingest.source-id-naming-schema

## Decision — `system:instance:kind`, one formatter, predictable named graphs

`source_partition.py:40-45`, `128-132`.

**The rejected alternative**: letting each connector author choose its own
`source_system` string format — the natural outcome of NOT having a
canonical schema, and the alternative this module explicitly closes.
Free-form per-connector naming means an operator who wants to "query/clear a
whole system, a single instance, or a sub-kind" has to know (or guess) each
connector's own spelling convention — `gitlab-prod-code` vs
`gitlab:prod:code` vs `gitlab_prod_code` are all equally plausible outputs
from independently-authored connectors, and none is programmatically
distinguishable from the others without per-connector special-casing.

**The design chosen**: EVERY connector's source id follows ONE hierarchical,
colon-delimited grammar via `make_source_id` — the ONE formatter for a
connector's `source_system` value. Every part is slugged and joined by `:`
(e.g. `make_source_id("gitlab", "gitlab.example.test", "code")` →
`"gitlab:gitlab.example.test:code"`); empty optional parts are dropped. This
makes the whole fleet partition into PREDICTABLE named graphs, so an
operator can query or clear at any level of the hierarchy (a whole system,
one instance, one sub-kind) without guessing spelling per connector — a
mechanical consequence of the grammar being fixed rather than
per-connector-chosen.

A related guard, in the same file: `_GENERIC_SOURCES` (`""`, `"system"`,
`"internal"`, `"kg"`) are explicitly excluded from being treated as a real
external source — the engine stamps `source="system"` on internal edges
(`link_nodes`), and that must NOT create a spurious `urn:source:system` named
graph alongside genuine external sources; see
`.specify/design/kg-mirror-target-graph/design.md` (`AU-KG.ingest.default-graph-leak-guard`)
for the companion decision this guard shares its "don't let something land
in the wrong graph silently" instinct with.

## Risk Assessment

- **Blast Radius**: `agent_utilities/knowledge_graph/backends/sparql/source_partition.py`
  (`make_source_id`) and every connector that calls it to derive its
  `source_system` value.
- **Backward Compatible**: A connector migrated to `make_source_id` changes
  its named-graph identity — this is a ONE-TIME migration concern for any
  connector whose prior source id predates the schema, not an ongoing
  compatibility risk.
- **Breaking Changes**: None for connectors already using `make_source_id`.
- **Known weak point**: the grammar is enforced by CONVENTION (every
  connector author calling `make_source_id` rather than hand-building a
  source-id string) — nothing mechanically prevents a new connector from
  bypassing the formatter and hand-rolling a differently-shaped source id,
  which would silently defeat the "one predictable grammar" guarantee for
  that one source.
