# Design Document: An asset routes to its enterprise standard by capability tag first, node type only as fallback

CONCEPT:AU-KG.ingest.then-by-its-node

> `agent_utilities/knowledge_graph/standardization/standards.py:40-53`.

**Triage note**: the id-shape heuristic flagged this as a slugified prose
fragment ("...then by its node...") and suggested retire. Reading the site
shows it names a real routing-precedence decision — "[route by capability]
THEN BY ITS NODE [type]" — not filler. Documented, not retired.

## Decision — domain routing keys on the vendor-neutral `capability` tag FIRST, and falls back to the node's `type` field only when no capability tag is present

`standards.py:48-53` states the rule directly in a comment above
`STANDARD_DOMAINS`: **"Which assets each standard governs. An asset is
routed to a standard by its vendor-neutral `capability` tag first (the
egeria harvest cross-vendor join key), then by its node `type` as a
fallback. Lower-cased matching throughout."**

This sits inside a larger, explicitly-named architectural choice
(`standards.py:1-9`): rather than flatten every organization's standard into
one superset, or let orgs freely override, enterprise standards reuse the
Foundry-parity **interface type** layer verbatim — an enterprise standard
*is an* `Interface`, whose required `InterfaceProperty`/`InterfaceLinkConstraint`
encode the mandatory enterprise contract every governed asset must carry
(owner, lifecycle_state, data_classification, the vendor-neutral
`capability` tag, an organization link). The capability-then-type routing
rule is how an arbitrary ingested asset gets matched to one of those
standards.

**The rejected alternative is routing by node `type` alone** (e.g. a
ServiceNow CI's `sys_class_name`, a GitLab project's native type field).
It loses because node `type` is vendor-specific — the same conceptual asset
(an application) arrives with a different type string depending on which
source system ingested it (ServiceNow CMDB vs. GitLab vs. LeanIX). The
`capability` tag is named explicitly as "the egeria harvest cross-vendor
join key" — a normalized, vendor-neutral label every source is expected to
carry, making it the more reliable routing key. Type-based routing is kept
only as a **fallback**, not removed, for assets that haven't (yet) been
tagged with a capability — an explicit acknowledgment that capability
tagging isn't universal across every ingested source.

## Risk Assessment

- **Blast Radius**: `knowledge_graph/standardization/standards.py`'s
  `STANDARD_DOMAINS` routing table and every consumer of enterprise-standard
  assignment.
- **Backward Compatible**: Yes — the type-based fallback preserves routing
  for untagged assets.
- **Breaking Changes**: None.
- **Known weak point**: an asset with an incorrect or stale `capability` tag
  routes confidently to the wrong standard (capability wins over type with
  no cross-check), whereas an untagged asset at least falls back to a
  type-based guess — a mistagged asset is arguably worse off than an
  untagged one under this precedence order.
