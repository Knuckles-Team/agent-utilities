# Design Document: A spec's external tracker ids are a typed dict on the model, not a side table

> `agent_utilities/models/sdd.py:31-33`.

CONCEPT:AU-KG.ingest.external-tracker-ids

## Decision — `Spec.external_links: dict[str, str]`, mirrored to `:trackedBy`, not a separate join table

`sdd.py:20-34`.

**The rejected alternative**: a separate cross-reference/join table (or a
dedicated `SpecTrackerLink` model) mapping spec ids to external tracker ids
(Jira, Plane, ...) — the conventional relational-modeling choice for a
many-tracker-systems-per-spec relationship. That alternative is not present
anywhere in the codebase; instead the field is a plain
`dict[str, str]` directly on `Spec` (e.g.
`{"jira": "PROJ-123", "plane": "<project>/<work_item>"}`), keeping the spec's
own external identity co-located with the spec model rather than requiring a
join across models/tables to answer "what ticket tracks this spec". The
tradeoff accepted deliberately: this only models a spec tracked by AT MOST
ONE ticket per tracker system (the dict key is the tracker name) — a spec
split across two Jira tickets cannot be represented, which the simpler
join-table alternative would have allowed.

This field is not merely local bookkeeping — it is mirrored to the KG spec
node's `:trackedBy` property by the spec↔ticket link flow, so the same
external-tracker-id data that lives on the Pydantic `Spec` model is also
queryable in the graph without a second, independently-maintained mapping.

## Risk Assessment

- **Blast Radius**: `agent_utilities/models/sdd.py` (the `Spec` model) and
  whatever spec↔ticket link flow reads `external_links` to write `:trackedBy`.
- **Backward Compatible**: Yes — `external_links` defaults to `{}`; an
  existing `Spec` with no external links is unaffected.
- **Breaking Changes**: None.
- **Known weak point**: the one-ticket-per-tracker-system constraint (dict
  key = tracker name) is not validated anywhere — nothing prevents a caller
  from needing a second Jira ticket for the same spec and silently
  overwriting the first key's value.
