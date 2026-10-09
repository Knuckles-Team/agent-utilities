"""CONCEPT:X1 -- a tiny-profile process cannot perform admin-scoped EG operations.

Drives the REAL ``epistemic-graph-server`` (the session ``tiny_engine``). The
scopes and roles each local-process grant projects onto the engine carrier are
taken verbatim from ``GraphSession.engine_verified_context()``. Only audience,
tenant and policy revision are swapped for the test engine's, because the
ephemeral engine is started with the suite's own policy. The engine's
``allows_method`` therefore judges exactly the scope set a tiny process sends.

* The ambient session (every stdio tool call and background write) is refused
  ``CreateGraph`` (``graph:admin``) and ``BlobGc`` (``blob:admin``).
* The one-shot provisioning authority may create a graph, and nothing wider:
  ``BlobGc`` is still refused, which proves it carries no ``kg:admin``.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest

pytestmark = pytest.mark.integration


def _carrier_for(session: Any) -> dict[str, object]:
    """The session's own roles/scopes on the test engine's policy fields."""
    from _test_engine import request_context

    carrier = session.engine_verified_context()
    return request_context(roles=carrier["roles"], scopes=carrier["scopes"])


@contextmanager
def _client(socket_path: str, session: Any) -> Iterator[Any]:
    from _test_engine import TEST_AUTH_SECRET
    from epistemic_graph.client import SyncEpistemicGraphClient

    client = SyncEpistemicGraphClient.connect(
        socket_path=socket_path,
        auth_secret=TEST_AUTH_SECRET,
        verified_context=_carrier_for(session),
    )
    try:
        yield client
    finally:
        client.close()


def _mint(minter_name: str) -> Any:
    from agent_utilities.security import request_identity

    return getattr(request_identity, minter_name)()


def _assert_scope_denied(call: Any, scope: str) -> None:
    with pytest.raises(Exception, match="ACCESS_DENIED") as denied:
        call()
    assert scope in str(denied.value)


@pytest.mark.spec("AU-SEC-R001")
def test_ambient_tiny_session_cannot_perform_admin_engine_operations(
    tiny_engine,
) -> None:
    session = _mint("mint_local_process_session")
    graph_name = f"x1_ambient_{uuid.uuid4().hex[:12]}"
    with _client(tiny_engine, session) as client:
        _assert_scope_denied(
            lambda: client.tenants.create(graph_name, "Agent"), "graph:admin"
        )
        _assert_scope_denied(client.blob.gc, "blob:admin")
        # Least privilege, not an outage: the read it needs still works.
        client.tenants.list()


@pytest.mark.spec("AU-SEC-R001")
def test_provisioning_authority_creates_a_graph_and_nothing_wider(
    tiny_engine,
) -> None:
    session = _mint("mint_local_process_bootstrap_authority")
    graph_name = f"x1_provision_{uuid.uuid4().hex[:12]}"
    with _client(tiny_engine, session) as client:
        client.tenants.create(graph_name, "Agent")
        try:
            listed = {
                str(entry.get("name") if isinstance(entry, dict) else entry)
                for entry in client.tenants.list() or []
            }
            assert graph_name in listed
            _assert_scope_denied(client.blob.gc, "blob:admin")
        finally:
            client.tenants.delete(graph_name)
