"""Failing-first carrier-authority test for the federation reader (BUG-005 / GOC-15).

CONCEPT:AU-KG.identity.verified-carrier-required-federation

GOC-15 (verified identity carrier on every protocol) names the federation reader as
one of the surfaces that "cannot mint a caller and always deny" (`PHASE-0-GREEN-
FOUNDATION.md` Group E / BUG-005). The GOC-15-W01 inventory pass
(`plans/graph-os-completion-program/decisions/GOC-15-carrier-authority.md`) found a
worse defect for this specific surface: it does not deny -- it FAILS OPEN.
`FederationMixin.execute_federated_query` / `_execute_federated_connection`
(`agent_utilities/knowledge_graph/orchestration/engine_federation.py`) never calls
`resolve_session`/`current_session` before reaching a backend's `execute_read`,
unlike every other `graph_query` dialect: `cypher` explicitly requires `kg:read`
(`engine_query.py:185`) and `sql`/`sparql` are gated by the session-routed wire
client (`graph_compute.py`'s `_SessionRoutedAsyncClient._send()`). The federation
dialect has neither.

This test encodes the carrier contract every surface must uphold (GOC-15 lane
contract: "No data-plane surface may infer identity from an untrusted header,
session string, browser credential, or process default... Missing/expired/wrong-
audience context fails closed"). It is RED against the current implementation and
is expected to turn green only once the federation reader is wired to
`resolve_session(required_scope=...)` (or an equivalent verified-carrier check)
before touching any backend, per BUG-005's remediation design ("one
`CarrierAuthority` adapter... Every listener either implements the TCK or is
absent in secure builds").
"""

from __future__ import annotations

import contextvars

import pytest

from agent_utilities.knowledge_graph.core.session import (
    SessionRequiredError,
    current_session,
)
from agent_utilities.knowledge_graph.orchestration.engine_federation import (
    FederationMixin,
)


class _StubFederatedEngine:
    """Stands in for `get_connection_registry().get_engine(connection)`.

    This is the exact shape `_execute_federated_connection` calls into
    (`engine_federation.py:309-314`): ``getattr(engine, "backend", None) or
    engine`` then ``execute_read(query, parameters)``.
    """

    def __init__(self) -> None:
        self.calls = 0

    def execute_read(self, query: str, parameters: dict) -> list[dict]:
        self.calls += 1
        return [{"leaked": "unauthenticated federated row"}]


class _FederationHarness(FederationMixin):
    def __init__(self) -> None:
        self.backend = None


def test_federated_query_denies_without_a_verified_session(monkeypatch):
    """The federation reader must fail closed like every other graph_query dialect.

    With NO ambient GraphSession bound (the same precondition
    `test_no_ambient_session_is_never_synthesized` in `test_graph_session.py` uses
    for every other authority boundary in this repo), a federated query must raise
    `SessionRequiredError` before invoking the backend. Today it does not -- it
    calls straight through to the stub engine's `execute_read` and returns real
    rows, proving zero identity verification exists on this path.
    """
    harness = _FederationHarness()
    stub_engine = _StubFederatedEngine()

    class _Registry:
        @staticmethod
        def get_engine(_connection: str) -> _StubFederatedEngine:
            return stub_engine

    monkeypatch.setattr(
        "agent_utilities.mcp.kg_server.get_connection_registry",
        lambda: _Registry(),
    )

    def isolated() -> None:
        assert current_session() is None, (
            "test precondition: no ambient GraphSession bound in this context"
        )
        with pytest.raises(SessionRequiredError):
            harness._execute_federated_connection(
                "external-graph", "SELECT * WHERE { ?s ?p ?o }"
            )

    contextvars.Context().run(isolated)

    assert stub_engine.calls == 0, (
        "BUG-005/GOC-15 FAIL-OPEN: the federation reader executed a backend read "
        "with no verified caller identity. Every surface behind graph_query must "
        "resolve a verified GraphSession (resolve_session(required_scope=...)) "
        "before reaching a backend, exactly like the cypher/sql/sparql dialects "
        "already do -- see the GOC-15 carrier-authority decision record."
    )
