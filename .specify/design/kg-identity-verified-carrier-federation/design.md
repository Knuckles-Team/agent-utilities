# Design Document: Federation reader requires a verified identity carrier before touching any backend

CONCEPT:AU-KG.identity.verified-carrier-required-federation

> `agent_utilities/knowledge_graph/orchestration/engine_federation.py:154` (BUG-036/GOC-15)

## Decision — `resolve_session(required_scope="kg:read")` at the federation entrypoint, closing all three internal branches at once

`plans/graph-os-completion-program/decisions/GOC-15-carrier-authority.md`
(DEC-015, GOC-15 verified-identity-carrier-authority record) inventoried every
AU/EG surface that mints (or fails to mint) a caller identity before touching a
backend. It found the AU federation reader
(`FederationMixin.execute_federated_query`/`_execute_federated_connection`) was
the WORST finding in that record's fail-open class: unlike every other
`graph_query` dialect — `cypher` explicitly requires `kg:read`
(`engine_query.py:185`); `sql`/`sparql` are gated by the session-routed wire
client (`graph_compute.py`'s `_SessionRoutedAsyncClient._send()`) — federation
had NO carrier check at all. It does not deny by default; it executes
unconditionally against the target backend.

The fix adds one `resolve_session(required_scope="kg:read")` call at the
PUBLIC entrypoint of `execute_federated_query`, before any of its three
internal branches run: the REST-virtual-source extractor invocation, the
local-graph connection-alias lookup, and the external-backend read. Placing it
at the single public entrypoint — rather than duplicating the check inside
each branch — closes the gap for all three in one place, matching this
codebase's own "enforce at the chokepoint, not one entrypoint" lesson. A
failing-first test (`tests/unit/knowledge_graph/test_federation_carrier_authority.py`)
encodes the GOC-15 carrier contract ("No data-plane surface may infer identity
from an untrusted header, session string, browser credential, or process
default... Missing/expired/wrong-audience context fails closed") and is
expected to go green only once this check is in place.

**The rejected alternative** was gating each of the three internal branches
independently. That would triple the surface for a future regression (a
fourth branch added later inherits nothing) and does not match the pattern
every other working dialect in this file already uses — a single scope check
at the entrypoint the whole dispatch funnels through.

## Risk Assessment

- **Blast Radius**: `agent_utilities/knowledge_graph/orchestration/engine_federation.py`
  only; every `graph_query(scope='federated')` caller now requires a bound
  session.
- **Backward Compatible**: No for unauthenticated callers — a caller with no
  bound session that previously received data now receives
  `SessionRequiredError`. This is the intended fix (closing a fail-open, not a
  behavior-preserving refactor).
- **Known weak point**: this closes the ONE surface DEC-015 found worst; the
  broader GOC-15 carrier-authority program still lists other fail-open/degrade
  findings (EG observability ingest, the AU row-visibility ACL silent-degrade,
  the standalone SPARQL router) that are separate lanes, not fixed by this
  change.
