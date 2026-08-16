"""U-17 — ontology lifecycle activation must not retarget an already
graph-scoped verified session.

Root cause: ``OntologyLifecycle._resolve_graph_compute`` builds a
graph-scoped, zero-transport view fixed to the tenant's dedicated ontology
graph (``for_graph(self._ontology_graph)``) -- the SAME ``_client_for(graph)``
shape R-23 fixed in ``mcp/tools/engine_tools.py``. Issuing an RPC through
that fixed view while the ambient verified session still names a DIFFERENT
graph (whatever graph the calling boundary bound the session to -- commonly
the tenant's default content graph) raised the wire layer's own guard ("A
graph-scoped view cannot retarget the verified GraphSession"), which
``_make_store``/every CRUD method caught broadly and silently downgraded to
the process-local, non-durable in-memory registry -- exactly the observed
live symptom: "Durable per-tenant ontology registry unavailable ... falling
back to the process-local, non-durable registry" on every boot.

The fix (mirrors R-02/R-23): narrow the SESSION onto the ontology graph
FIRST (``GraphSession.with_graph()`` + ``use_session()``), before any RPC
through the fixed view, then restore the caller's original session
afterward -- see ``OntologyLifecycle._graph_scope``/
``_with_ontology_graph_scope``.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.core.session import (
    GraphSession,
    current_session,
    use_session,
)
from agent_utilities.knowledge_graph.ontology.lifecycle import (
    OntologyLifecycle,
    _EngineRegistryStore,
    reset_registry,
)
from agent_utilities.security.brain_context import ActorContext, ActorType

PETS_TTL = """@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix ex: <http://example.org/pets#> .
<http://example.org/pets> a owl:Ontology .
ex:Animal a owl:Class .
ex:Dog a owl:Class ; rdfs:subClassOf ex:Animal .
"""


@pytest.fixture(autouse=True)
def _clean_registry():
    reset_registry()
    yield
    reset_registry()


def _session(*, graph: str, tenant: str = "acme") -> GraphSession:
    actor = ActorContext(
        actor_id="boot-service",
        actor_type=ActorType.AUTOMATED_SERVICE,
        roles=("test",),
        tenant_id=tenant,
        authenticated=True,
    )
    return GraphSession(
        actor=actor,
        tenant=tenant,
        scopes=frozenset({"kg:read", "kg:write"}),
        graph=graph,
        policy_version="test-policy",
        audience="test-audience",
    )


class _SharedEngineState:
    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}
        self.tenants_created: set[str] = set()


class _FixedGraphView:
    """Mirrors ``graph_compute.py``'s real ``_send()`` retarget guard: every
    call requires the AMBIENT session's ``.graph`` to already equal this
    view's fixed graph, or it raises ``PermissionError`` -- exactly the "A
    graph-scoped view cannot retarget the verified GraphSession" behavior.
    """

    def __init__(self, fixed_graph: str, state: _SharedEngineState) -> None:
        self._fixed_graph = fixed_graph
        self._state = state

    def _check(self) -> None:
        session = current_session()
        if session is None or session.graph != self._fixed_graph:
            raise PermissionError(
                "A graph-scoped view cannot retarget the verified GraphSession"
            )

    # ── typed-node registry surface (_EngineRegistryStore) ──────────────
    def has_node(self, node_id: str) -> bool:
        self._check()
        return node_id in self._state.nodes

    def add_node(self, node_id: str, node_type: str | None = None, **props) -> None:
        self._check()
        self._state.nodes[node_id] = dict(props)

    def remove_node(self, node_id: str) -> None:
        self._check()
        self._state.nodes.pop(node_id, None)

    def get_nodes_by_label(self, label: str):
        self._check()
        return list(self._state.nodes.items())

    @property
    def client(self):
        outer = self

        class _Nodes:
            def properties(self, node_id):
                outer._check()
                return outer._state.nodes.get(node_id, {})

        class _Tenants:
            def list(self):
                outer._check()
                return (
                    [{"name": outer._fixed_graph}]
                    if outer._fixed_graph in outer._state.tenants_created
                    else []
                )

            def create(self, name, kind):
                outer._check()
                outer._state.tenants_created.add(name)

        return type("Client", (), {"nodes": _Nodes(), "tenants": _Tenants()})()

    # ── RDF axiom surface (_load_axioms/_retract_axioms) ────────────────
    def add_triples(self, turtle=None, ntriples=None):
        self._check()
        return {"triples": 1}

    def remove_triples(self, turtle=None, ntriples=None):
        self._check()
        return {"removed": 1}

    def for_graph(self, graph_name: str) -> _FixedGraphView:
        if graph_name == self._fixed_graph:
            return self
        return _FixedGraphView(graph_name, self._state)


class _UnscopedGraphCompute:
    """The engine's own default (unscoped) compute client -- exists only to
    hand out `for_graph()` views, same shape as the real GraphComputeEngine.
    """

    def __init__(self, state: _SharedEngineState) -> None:
        self._state = state

    def for_graph(self, graph_name: str) -> _FixedGraphView:
        return _FixedGraphView(graph_name, self._state)


class _FakeEngine:
    def __init__(self, state: _SharedEngineState) -> None:
        self.graph_compute = _UnscopedGraphCompute(state)


def test_load_succeeds_and_uses_the_durable_store_when_ambient_session_differs():
    """The ambient session is deliberately bound to a DIFFERENT graph than
    the tenant's dedicated ontology graph (exactly the boot-time situation:
    the process/request session names the default content graph, not the
    ontology graph) -- `load()` must still narrow, activate against the real
    engine, and use the DURABLE per-tenant store, never the process-local
    in-memory fallback."""
    state = _SharedEngineState()
    engine = _FakeEngine(state)
    session = _session(graph="tenant-default-content-graph", tenant="acme")

    with use_session(session):
        lc = OntologyLifecycle(engine=engine, tenant="acme")
        assert lc._ontology_graph != "tenant-default-content-graph"
        # The durable store was actually selected -- NOT the non-durable
        # in-memory fallback (the exact live symptom this closes).
        assert isinstance(lc._store, _EngineRegistryStore)

        result = lc.load(PETS_TTL, source_type="text")
        assert result["status"] == "ok"
        assert result["ontology"]["active"] is True

        # The caller's original session is restored after the call.
        assert current_session().graph == "tenant-default-content-graph"

    # A second call (list_ontologies) also narrows/restores correctly.
    with use_session(session):
        listed = lc.list_ontologies()
        assert listed["count"] == 1
        assert current_session().graph == "tenant-default-content-graph"


def test_KNOWN_BAD_the_fixed_view_really_does_reject_a_mismatched_session():
    """Negative proof the guard is real, not vacuous: bypass
    `OntologyLifecycle`'s narrowing entirely and call the fixed view
    directly (the same call `_ensure_ontology_graph` makes) while the
    ambient session still names a different graph -- it must raise, proving
    the positive test above is exercising a real fix and not a scenario
    that would have succeeded regardless.
    """
    state = _SharedEngineState()
    engine = _FakeEngine(state)
    session = _session(graph="tenant-default-content-graph", tenant="acme")

    with use_session(session):
        lc = OntologyLifecycle(engine=engine, tenant="acme")
        with pytest.raises(PermissionError, match="cannot retarget"):
            lc._gc.client.tenants.list()
