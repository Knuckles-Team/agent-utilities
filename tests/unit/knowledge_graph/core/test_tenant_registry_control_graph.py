"""BUG-295-class retarget bug in ``tenant_registry`` — the tenant hierarchy
registry never worked, for ANY caller, including ``kg:admin``.

Production symptom: every ``set_parent``/``clear_parent`` call, and every
``_load_snapshot`` read, raised (write side) or silently degraded to flat
tenancy at DEBUG (read side, caught in ``_hierarchy_snapshot``) with::

    PermissionError: "A graph-scoped view cannot retarget the verified GraphSession"

Root cause: ``_control_backend()`` returns a graph-scoped view pinned to
``__control__`` (``EpistemicGraphBackend.for_graph``), but the caller's
ambient ``GraphSession`` is bound to whatever tenant graph it actually runs
under (e.g. a ``kg:admin`` acting from an ``acme``-scoped session) —
never ``__control__``. ``graph_compute._send_routed`` fails closed on that
mismatch. This is the exact same shape of bug as
``core.schedule_engine``'s ``_control_session_scope`` (BUG-295), and the
fix here reuses that SAME shared helper
(``knowledge_graph.core.session.control_session_scope``) rather than
reimplementing the retarget.

This is why zero ``:TenantHierarchy`` nodes ever existed: nobody, at any
privilege level, could ever successfully call ``set_parent``.

These tests reproduce the production ``PermissionError`` in a fake backend
(mirroring ``graph_compute._send_routed``'s own check) rather than mocking
the retargeting seam under test — ``_EnforcingControlBackend`` raises
exactly like the real engine when driven by a session whose ``graph``
disagrees with the backend's own ``graph_name``.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.core import tenant_registry as tr
from agent_utilities.knowledge_graph.core.session import (
    GraphSession,
    current_session,
    use_session,
)
from agent_utilities.models.company_brain import ActorType
from agent_utilities.security.brain_context import ActorContext


def _actor(actor_id: str = "root", tenant: str = "acme", roles=("kg:admin",)):
    return ActorContext(
        actor_id=actor_id,
        actor_type=ActorType.HUMAN,
        roles=tuple(roles),
        tenant_id=tenant,
        authenticated=True,
    )


def _session(graph: str, *, tenant: str = "acme", roles=("kg:admin",)) -> GraphSession:
    return GraphSession(
        actor=_actor(tenant=tenant, roles=roles),
        tenant=tenant,
        scopes=frozenset({"kg:read", "kg:write"}),
        graph=graph,
        policy_version="test-policy",
        audience="test-audience",
    )


class _EnforcingControlBackend:
    """Fake control-plane backend reproducing the EXACT production check
    ``graph_compute._send_routed`` performs for a graph-scoped view pinned
    to a fixed graph::

        if self._fixed_graph and session.graph != self._fixed_graph:
            raise PermissionError(
                "A graph-scoped view cannot retarget the verified GraphSession"
            )

    ``graph_name`` mirrors the attribute the real fix reads
    (``EpistemicGraphBackend.graph_name``) to learn what graph to retarget
    onto — so a genuine mismatch between the ambient session and this graph
    raises, and a correctly-retargeted session does not.
    """

    def __init__(self, graph_name: str = "__control__") -> None:
        self.graph_name = graph_name
        self.nodes: dict[str, dict] = {}

    def _enforce(self) -> None:
        session = current_session()
        if session is None or session.graph != self.graph_name:
            raise PermissionError(
                "A graph-scoped view cannot retarget the verified GraphSession"
            )

    def add_node(self, node_id: str, *, node_type: str, **props) -> None:
        self._enforce()
        self.nodes[node_id] = {"id": node_id, "node_type": node_type, **props}

    def nodes_by_label(self, label: str, limit: int = 0):
        self._enforce()
        return [
            (nid, dict(row))
            for nid, row in self.nodes.items()
            if row.get("node_type") == label
        ]


@pytest.fixture
def backend(monkeypatch):
    fake = _EnforcingControlBackend()
    monkeypatch.setattr(tr, "_control_backend", lambda: fake)
    tr.invalidate_cache()
    yield fake
    tr.invalidate_cache()


def test_fake_control_backend_rejects_a_mismatched_session_unretargeted() -> None:
    """Negative control: proves the fake faithfully reproduces the real
    ``PermissionError`` a graph-scoped control-plane view raises when driven
    by an ambient session bound to a different graph, so the positive tests
    below are proving something real — calls the backend directly,
    bypassing any retargeting.
    """
    backend = _EnforcingControlBackend()
    with use_session(_session(graph="acme")):
        with pytest.raises(PermissionError, match="cannot retarget"):
            backend.nodes_by_label(tr.TENANT_HIERARCHY_LABEL)


def test_set_parent_works_under_a_kg_admin_session_scoped_to_a_tenant_graph(
    backend,
) -> None:
    """THE production condition: a real ``kg:admin`` identity's ambient
    session is bound to a tenant graph (``acme``), never ``__control__``.
    Before the fix this raised on EVERY call, for every caller regardless of
    privilege — the bug this test pins.
    """
    with use_session(_session(graph="acme")):
        result = tr.set_parent("eng", "acme", actor=_actor(roles=("kg:admin",)))

    assert result.tenant_id == "eng"
    assert result.parent_tenant_id == "acme"
    # The record is durably persisted in __control__, not lost/no-op'd.
    assert backend.nodes[tr.registry_node_id("eng")]["parent_tenant_id"] == "acme"


def test_clear_parent_works_under_a_kg_admin_session_scoped_to_a_tenant_graph(
    backend,
) -> None:
    with use_session(_session(graph="acme")):
        tr.set_parent("eng", "acme", actor=_actor(roles=("kg:admin",)))
        tr.invalidate_cache()
        result = tr.clear_parent("eng", actor=_actor(roles=("kg:admin",)))

    assert result.parent_tenant_id == ""
    assert backend.nodes[tr.registry_node_id("eng")]["parent_tenant_id"] == ""


def test_load_snapshot_and_ancestor_chain_work_under_a_mismatched_session(
    backend,
) -> None:
    """Read side (``_load_snapshot`` via ``ancestor_chain``): before the fix
    this never raised to the caller (it is caught in ``_hierarchy_snapshot``)
    but silently degraded to flat tenancy — the registry looked empty even
    with records durably written, because the read itself always failed.
    """
    with use_session(_session(graph="acme")):
        tr.set_parent("eng", "acme", actor=_actor(roles=("kg:admin",)))
        tr.invalidate_cache()
        chain = tr.ancestor_chain("eng")

    assert chain == ["acme"]


def test_a_correctly_scoped_control_session_is_a_no_op_retarget(backend) -> None:
    """When the ambient session is ALREADY scoped to ``__control__`` (e.g. a
    control-plane-native caller), retargeting must be a no-op, not an
    additional narrowing — the shared helper must not needlessly touch a
    session that is already correctly scoped.
    """
    with use_session(_session(graph="__control__")):
        result = tr.set_parent("eng", "acme", actor=_actor(roles=("kg:admin",)))
    assert result.parent_tenant_id == "acme"
