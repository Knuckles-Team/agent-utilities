#!/usr/bin/python
from __future__ import annotations

"""GOC-62 conformance-suite skeleton (D2/D3(b)).

CONCEPT:AU-OS.identity.stack-wide-auth-conformance — see
``plans/graph-os-completion-program/decisions/GOC-62-keycloak-auth-standard.md``
§D2 for the full design.

This package is the first, working instance of the enumeration methodology
that record specifies: a **live introspection pass** over one surface family
(the ``graph_query`` dialect dispatcher — ``mcp/tools/query_tools.py``),
cross-checked against a reviewed disposition manifest (:mod:`surface_manifest`)
via a drift test that FAILS when introspection finds a surface the manifest
does not know about — never a hardcoded pass list.

Deliberately scoped to one surface family rather than the full 28-surface
GOC-15 inventory: building live introspection for FastAPI routes requires
constructing a full agent app (LLM provider + MCP config — too heavy for a
fast unit-test harness), and the EG wire-``Method`` family already has its
own complete, generated enumeration on the Rust side
(``eg_capabilities::ALL_METHODS`` / ``policy()``) that does not need a Python
mirror. The query-dialect family is real, live, mechanically enumerable with
none of those blockers, and is exactly where BUG-036 (the AU federation
fail-open) lives — so it doubles as the working proof-of-concept AND the
harness for this pass's headline fail-open.

**What this package does NOT attempt**: auto-classifying a surface's
authorization posture (safe / fail-open) via static analysis. An early design
considered scanning each dialect branch's callee for a literal
``resolve_session(``/``current_session(`` call, but this codebase's own
surfaces defeat a shallow heuristic — ``sql``/``sparql`` are session-required
only at the WIRE layer (``graph_compute.py``'s ``_SessionRoutedAsyncClient``),
never as a literal call inside ``engine_query.py``'s ``sql()``/``sparql()``
methods, so the same shallow check that correctly flags ``federated`` would
ALSO incorrectly flag ``sql``/``sparql`` as fail-open. A gate that "looks like
it enforces something" but gives an unreliable signal is worse than no gate
(this repo's own "Gates report more coverage than they have" lesson) — so
**disposition stays a reviewed, human-authored manifest field**, exactly the
same shape as ``tenant_sharing.py``'s ``COMMONS_PRIVATE_NODE_TYPES``/
``COMMONS_SHAREABLE_NODE_TYPES`` (a mechanically-enumerable domain —
node types actually written/read — cross-checked against a reviewed,
deny-by-default classification list). What this package DOES automate is the
**enumeration** (so a new dialect can never be silently uncovered) — the
narrower, honest, and load-bearing half of "no hardcoded list."
"""
