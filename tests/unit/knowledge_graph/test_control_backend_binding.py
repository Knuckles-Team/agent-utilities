"""U-16/BUG-113: the WorkItem control-plane authority must always resolve to
the fixed ``__control__`` system graph, independent of whatever content graph
THIS particular engine instance's own ``self.backend`` happens to be scoped
to.

Before this fix, ``IntelligenceGraphEngine._build_control_backend()`` returned
``self.backend`` verbatim. That is correct only when the engine instance
itself was constructed already scoped to ``__control__`` -- it is wrong for
an engine constructed already scoped to a content graph (an explicit
``db_path``/``graph=`` construction argument, or a session-bound backend
picked up via ``get_active_backend()``), which is exactly the r15 live defect
(U-16): a codebase-ingest submission ran through a content-graph-scoped
engine, so its WorkItem was created IN that content graph
(``kf-pilot:code-ingest``); the public ``job_status``/background-claim path
ran through a different engine instance whose ``control_backend`` was the
true ``__control__`` graph, so it could never find the WorkItem -- split
WorkItem authority plus content-graph leakage of control state.
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
from agent_utilities.knowledge_graph.core.shard_topology import CONTROL_GRAPH_NAME


class _FakeGraphScopedBackend:
    """Minimal stand-in for ``EpistemicGraphBackend`` -- just enough surface
    for ``for_graph`` to behave like the real cheap graph-scoped view factory
    (no new transport; same name returns ``self``; a different name returns a
    new view carrying that name)."""

    def __init__(self, graph_name: str) -> None:
        self.graph_name = graph_name

    def for_graph(self, graph_name: str) -> _FakeGraphScopedBackend:
        target = str(graph_name)
        if target == self.graph_name:
            return self
        return _FakeGraphScopedBackend(target)


class _FakeNoScopeBackend:
    """A backend with no ``for_graph`` at all (a single-scope store)."""


def test_build_control_backend_resolves_control_graph_even_when_self_backend_is_content_scoped():
    """The r15 defect exactly: this engine instance's OWN backend is already
    scoped to a content graph (as happens for a session-bound engine
    constructed via an explicit content-graph db_path/graph=). The resolved
    control backend must still be ``__control__``, not that content graph."""
    inst = object.__new__(IntelligenceGraphEngine)
    inst.backend = _FakeGraphScopedBackend("kf-pilot:code-ingest")

    control = inst._build_control_backend()

    assert control.graph_name == CONTROL_GRAPH_NAME
    assert control.graph_name != "kf-pilot:code-ingest"


def test_build_control_backend_is_stable_across_different_content_scoped_instances():
    """Two engine instances scoped to two DIFFERENT content graphs must both
    resolve to the identical control-graph NAME -- this is what makes
    submit (via instance A) and status/claim (via instance B) agree on one
    WorkItem authority instead of splitting across whichever content graph
    each instance happened to be constructed with."""
    inst_a = object.__new__(IntelligenceGraphEngine)
    inst_a.backend = _FakeGraphScopedBackend("kf-pilot:code-ingest")
    inst_b = object.__new__(IntelligenceGraphEngine)
    inst_b.backend = _FakeGraphScopedBackend("research:papers")

    control_a = inst_a._build_control_backend()
    control_b = inst_b._build_control_backend()

    assert control_a.graph_name == control_b.graph_name == CONTROL_GRAPH_NAME


def test_build_control_backend_falls_back_to_self_backend_when_no_view_factory():
    """A backend with no ``for_graph`` (a single-scope store, or a minimal
    test double) must fall back to ``self.backend`` unchanged -- today's
    behavior for a backend that genuinely has only one scope."""
    inst = object.__new__(IntelligenceGraphEngine)
    backend = _FakeNoScopeBackend()
    inst.backend = backend

    assert inst._build_control_backend() is backend


def test_build_control_backend_falls_back_when_view_factory_raises():
    """A view-factory failure must not block engine construction -- fall back
    to this instance's own backend scope rather than raising."""

    class _RaisingBackend:
        graph_name = "kf-pilot:code-ingest"

        def for_graph(self, _name: str) -> None:
            raise RuntimeError("transport unavailable")

    inst = object.__new__(IntelligenceGraphEngine)
    backend = _RaisingBackend()
    inst.backend = backend

    assert inst._build_control_backend() is backend
