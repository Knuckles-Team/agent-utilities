#!/usr/bin/python
from __future__ import annotations

"""The reviewed disposition manifest — seed data, drift-checked, never
authoritative on its own (GOC-62 D2).

CONCEPT:AU-OS.identity.stack-wide-auth-conformance — see this package's
``__init__.py``. Every entry here is a HUMAN-REVIEWED classification of one
surface :mod:`surface_inventory` (or, for surfaces this skeleton does not yet
live-introspect, the GOC-15 28-surface inventory in
``decisions/GOC-15-carrier-authority.md`` §1) already found. This module never
invents a surface; :mod:`surface_inventory`'s live introspection is the only
source of truth for WHAT exists, this module is only the record of WHAT WAS
DECIDED about each one.
"""

from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "Disposition",
    "SurfaceEntry",
    "QUERY_DIALECT_MANIFEST",
    "GOC15_SURFACE_MANIFEST",
    "lookup_query_dialect",
]


class Disposition(StrEnum):
    """What a surface is supposed to do. Mirrors DEC-015 §1's own three-way
    split (fail-open / always-deny / working) plus the state every surface is
    SUPPOSED to reach once GOC-15's carrier lands."""

    #: Authenticated callers succeed; everyone else is denied per GOC-62 §4.
    AUTHENTICATED_REQUIRED = "authenticated_required"
    #: No credential mechanism exists yet for this protocol; it denies
    #: unconditionally in every configuration (DEC-015 §1's "always-deny" set).
    ALWAYS_DENY_NO_CREDENTIAL_MECHANISM = "always_deny_no_credential_mechanism"
    #: A CONFIRMED, OPEN violation being tracked — never an excuse. A
    #: conformance test for this disposition is EXPECTED to be red today; a
    #: green run on one of these is itself a signal something is wrong (either
    #: silently fixed with a stale manifest, or a vacuous test).
    KNOWN_FAIL_OPEN = "known_fail_open"


@dataclass(frozen=True, slots=True)
class SurfaceEntry:
    """One reviewed disposition. ``proof`` names the exact test (file, and
    function where useful) that demonstrates this disposition is real —
    required for ``KNOWN_FAIL_OPEN`` entries, so a claim of "this is broken"
    is never asserted without a citation to code that proves it."""

    surface_id: str
    disposition: Disposition
    citation: str
    owning_bug: str | None = None
    proof: str | None = None


# ---------------------------------------------------------------------------
# The live-introspected family: graph_query dialects (surface_inventory.py)
# ---------------------------------------------------------------------------

QUERY_DIALECT_MANIFEST: tuple[SurfaceEntry, ...] = (
    SurfaceEntry(
        surface_id="query_dialect:local",
        disposition=Disposition.AUTHENTICATED_REQUIRED,
        citation=(
            "mcp/tools/query_tools.py's implicit default -> "
            "kg_server._resolve_read_engines -> engine.query_cypher "
            "(knowledge_graph/orchestration/engine_query.py:137-185, explicit "
            "resolve_session(required_scope='kg:read') at :185)"
        ),
    ),
    SurfaceEntry(
        surface_id="query_dialect:sql",
        disposition=Disposition.AUTHENTICATED_REQUIRED,
        citation=(
            "engine_query.py:385 sql() -- session required at the WIRE layer "
            "(graph_compute.py's _SessionRoutedAsyncClient._send()), not a "
            "literal resolve_session() call in this method itself; DEC-015 "
            "finding #16 grades this weaker than cypher (no explicit scope "
            "check) but session-required on missing session"
        ),
    ),
    SurfaceEntry(
        surface_id="query_dialect:sparql",
        disposition=Disposition.AUTHENTICATED_REQUIRED,
        citation="engine_query.py:439 sparql() -- same wire-layer tier as sql, above",
    ),
    SurfaceEntry(
        surface_id="query_dialect:federated",
        disposition=Disposition.AUTHENTICATED_REQUIRED,
        citation=(
            "FIXED (BUG-036): mcp/tools/query_tools.py:417-429 -> "
            "engine_federation.py FederationMixin.execute_federated_query/"
            "_execute_federated_connection now call "
            "resolve_session(required_scope='kg:read') before touching any "
            "backend -- matching the cypher/sql/sparql dialects above. "
            "ActorContextMiddleware (au:actor-context-middleware, below) is "
            "also now fail-closed for graph-os, so this holds even for a "
            "caller that reaches the MCP-native dispatch path directly."
        ),
        owning_bug="BUG-036",
        proof="tests/unit/knowledge_graph/test_federation_carrier_authority.py::test_federated_query_denies_without_a_verified_session",
    ),
)


def lookup_query_dialect(surface_id: str) -> SurfaceEntry | None:
    for entry in QUERY_DIALECT_MANIFEST:
        if entry.surface_id == surface_id:
            return entry
    return None


# ---------------------------------------------------------------------------
# GOC-15's full 28-surface inventory, transcribed as manifest seed data.
# NOT yet live-introspected (no drift test covers this set — see the
# package docstring for why: FastAPI/MCP-registry introspection needs a full
# agent app, and EG's wire-Method family already has its own generated
# enumeration on the Rust side). Recorded here so this manifest is the single
# place GOC-62's consumers look, per DEC-015 §3.2 item 4's "capability
# inventory is a CI-checked artifact, not tribal knowledge" — the CI check
# itself (asserting eg_capabilities::policy() matches this list) is DEC-015's
# W02 to build, not duplicated here.
# ---------------------------------------------------------------------------

GOC15_SURFACE_MANIFEST: tuple[SurfaceEntry, ...] = (
    SurfaceEntry(
        surface_id="au:generic-node-write-chokepoints",
        disposition=Disposition.KNOWN_FAIL_OPEN,
        citation=(
            "knowledge_graph/core/engine.py:723,1178; core/graph_compute.py:2281; "
            "core/materialization.py:270; enrichment/pipeline.py:236 -- five "
            "`except PermissionError: pass` ownership-stamp swallows"
        ),
        owning_bug="BUG-033 / BUG-039",
        proof="GOC-61-W00 (not yet landed as of this manifest's authoring)",
    ),
    SurfaceEntry(
        surface_id="au:federation-reader",
        disposition=Disposition.AUTHENTICATED_REQUIRED,
        citation="FIXED -- see query_dialect:federated above -- same surface, different name",
        owning_bug="BUG-036",
        proof="tests/unit/knowledge_graph/test_federation_carrier_authority.py",
    ),
    SurfaceEntry(
        surface_id="eg:observability-ingest",
        disposition=Disposition.KNOWN_FAIL_OPEN,
        citation="epistemic-graph/src/server/obs/mod.rs:854-1038",
        owning_bug="BUG-037",
        proof="epistemic-graph src/server/obs/mod.rs::tests::bug_037_obs_ingest_post_bypasses_the_deny_gate",
    ),
    SurfaceEntry(
        surface_id="au:actor-context-middleware",
        disposition=Disposition.AUTHENTICATED_REQUIRED,
        citation=(
            "FIXED (BUG-036): mcp/middlewares.py:110-171 -- was a documented "
            "unconditional no-op with no token (the structural root cause of "
            "au:federation-reader's fail-open); now takes a "
            "require_verified_session flag, wired True ONLY for the graph-os "
            "server (server_factory.py's _configure_middleware(args, "
            "server_name=name)) since it is the one MCP server whose tools "
            "reach privileged KG reads/writes. In that mode a claim-less call "
            "is refused (SessionRequiredError) unless an ambient GraphSession "
            "or the tiny-profile local process authority "
            "(kg_server._PROCESS_SESSION) is already legitimately bound. The "
            "other ~60 fleet MCP servers built via the same factory keep the "
            "prior no-op (require_verified_session=False default) -- each "
            "retains its own authorization contract; making this fail-closed "
            "fleet-wide is a separate, larger change needing its own "
            "per-package audit."
        ),
    ),
    SurfaceEntry(
        surface_id="au:sparql-server-standalone",
        disposition=Disposition.ALWAYS_DENY_NO_CREDENTIAL_MECHANISM,
        citation="api/sparql_server.py:24-59 -- unwired/dormant, no importer",
    ),
    SurfaceEntry(
        surface_id="au:sql-sparql-row-visibility-acl",
        disposition=Disposition.AUTHENTICATED_REQUIRED,
        citation=(
            "knowledge_graph/orchestration/engine_query.py:387-396,428-436 -- "
            "silent-degrade risk (except Exception: return unfiltered rows), not a "
            "full bypass; DEC-015 finding #5"
        ),
    ),
    SurfaceEntry(
        surface_id="eg:ros2-bridge-outbound",
        disposition=Disposition.AUTHENTICATED_REQUIRED,
        citation="epistemic-graph/src/server/ros2_bridge.rs:239-255 -- conditional on RLS being configured",
    ),
    SurfaceEntry(
        surface_id="eg:iceberg-rest-catalog",
        disposition=Disposition.ALWAYS_DENY_NO_CREDENTIAL_MECHANISM,
        citation="epistemic-graph/src/server/lake/rest.rs:63-73",
    ),
    SurfaceEntry(
        surface_id="eg:observability-reads",
        disposition=Disposition.ALWAYS_DENY_NO_CREDENTIAL_MECHANISM,
        citation="epistemic-graph/src/server/obs/mod.rs:854-908",
    ),
    SurfaceEntry(
        surface_id="eg:federation-sparql-service",
        disposition=Disposition.ALWAYS_DENY_NO_CREDENTIAL_MECHANISM,
        citation="epistemic-graph/src/server/federation/mod.rs:974-998",
    ),
    SurfaceEntry(
        surface_id="eg:sparql-read",
        disposition=Disposition.ALWAYS_DENY_NO_CREDENTIAL_MECHANISM,
        citation="epistemic-graph/src/server/sparql_http.rs:415-475",
    ),
    SurfaceEntry(
        surface_id="eg:sparql-graph-store-read",
        disposition=Disposition.ALWAYS_DENY_NO_CREDENTIAL_MECHANISM,
        citation="epistemic-graph/src/server/sparql_http.rs:900-921",
    ),
    SurfaceEntry(
        surface_id="eg:nl-facade",
        disposition=Disposition.ALWAYS_DENY_NO_CREDENTIAL_MECHANISM,
        citation="epistemic-graph/src/server/sparql_http.rs:356-379",
    ),
    SurfaceEntry(
        surface_id="au:fastapi-app-all-routers",
        disposition=Disposition.AUTHENTICATED_REQUIRED,
        citation="server/app.py:665-882; security/request_identity.py:494-647 ActorIdentityMiddleware",
    ),
    SurfaceEntry(
        surface_id="au:canonical-sparql-endpoint",
        disposition=Disposition.AUTHENTICATED_REQUIRED,
        citation="gateway/graph_api.py:60-111 -- explicit resolve_session(required_scope='kg:read')",
    ),
    SurfaceEntry(
        surface_id="au:usage-observability-dashboard",
        disposition=Disposition.AUTHENTICATED_REQUIRED,
        citation="gateway/usage_api.py, usage/authorization.py:38-76",
    ),
    SurfaceEntry(
        surface_id="au:nl-query-ask",
        disposition=Disposition.AUTHENTICATED_REQUIRED,
        citation=(
            "mcp/tools/query_tools.py:660-724 -> core/nl_query.py:187-280 -- deny "
            "swallowed into a 200-status payload (audit-trail gap, not a bypass), "
            "DEC-015 finding #19"
        ),
    ),
    SurfaceEntry(
        surface_id="au:websocket-dashboard",
        disposition=Disposition.AUTHENTICATED_REQUIRED,
        citation=(
            "gateway/ws.py:78-154 -- correct today but a SECOND, independent auth "
            "implementation (ActorIdentityMiddleware structurally skips WS ASGI "
            "scopes); GOC-62 standard requires this collapse onto the one verifier"
        ),
    ),
    SurfaceEntry(
        surface_id="au:agent-dispatch-worker",
        disposition=Disposition.KNOWN_FAIL_OPEN,
        citation="orchestration/agent_dispatch_worker.py main() -- no resolve_session/GraphSession/process-identity bootstrap found",
        owning_bug="BUG-002 (GOC-18's open question, DEC-015 finding #21)",
    ),
    SurfaceEntry(
        surface_id="au:cli-stdio-local-process-bootstrap",
        disposition=Disposition.AUTHENTICATED_REQUIRED,
        citation=(
            "security/request_identity.py:339-392 mint_local_process_session -- "
            "the ONE intentionally credential-free path, correctly scoped"
        ),
    ),
    SurfaceEntry(
        surface_id="eg:sparql-write",
        disposition=Disposition.AUTHENTICATED_REQUIRED,
        citation="epistemic-graph/src/server/sparql_http.rs:211-232 -- genuine per-caller eg2. envelope tunnel",
    ),
    SurfaceEntry(
        surface_id="eg:kvcache-http",
        disposition=Disposition.AUTHENTICATED_REQUIRED,
        citation="epistemic-graph/src/server/kvcache_http/mod.rs:521-545 -- fixed shared service identity",
    ),
    SurfaceEntry(
        surface_id="eg:s3-rest",
        disposition=Disposition.AUTHENTICATED_REQUIRED,
        citation="epistemic-graph/src/server/s3/mod.rs:910-925 -- SigV4",
    ),
    SurfaceEntry(
        surface_id="eg:graphql-subscriptions",
        disposition=Disposition.AUTHENTICATED_REQUIRED,
        citation="epistemic-graph/src/server/graphql_sub.rs:190",
    ),
    SurfaceEntry(
        surface_id="eg:ros2-bridge-inbound",
        disposition=Disposition.AUTHENTICATED_REQUIRED,
        citation="epistemic-graph/src/server/ros2_bridge.rs:220-231 -- routes through dispatch()'s real chokepoint",
    ),
)
