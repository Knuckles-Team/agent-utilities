"""NE-037 acceptance — explicit graph selection across async work (`92bb8578`),
the "graph-view cache order" half of the gate.

``agent_utilities.knowledge_graph.core.ingest_routing.engine_for_graph`` caches
one lightweight ``for_graph()`` view PER GRAPH NAME in a process-level dict
(``_engine_cache``) so repeated routing lookups for the same content graph
reuse the same view instead of re-deriving it. The WorkItem control-plane
authority (``IntelligenceGraphEngine.control_backend`` /
``_build_control_backend``, U-05/U-16/BUG-113 -- see
``tests/unit/knowledge_graph/core/test_workitem_control_graph_isolation.py``)
must stay fixed to the process root's control graph regardless of:

* which content graph the caller narrowed onto most recently, AND
* the ORDER in which this cache was warmed -- a content graph view cached
  BEFORE the control graph view is ever resolved, or the control graph
  resolved and cached first and a content view built afterward.

Both orders must produce the SAME two invariants: (1) every cached view's
``control_backend`` is identical object identity (the one true control
authority, never re-derived per view), and (2) a WorkItem written through a
content-scoped host still lands in the control graph only, never leaking
into whichever content graph happened to be resolved/cached around it.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.core import ingest_routing
from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
from agent_utilities.knowledge_graph.core.engine_tasks import (
    _ControlPlaneWorkItemEngine,
)
from agent_utilities.knowledge_graph.core.shard_topology import CONTROL_GRAPH_NAME
from tests.unit.knowledge_graph.core.test_workitem_control_graph_isolation import (
    _FakeGraphBackend,
    _FakeHost,
    _make_root_engine,
)


@pytest.fixture(autouse=True)
def _clean_cache():
    ingest_routing._reset_for_tests()
    yield
    ingest_routing._reset_for_tests()


@pytest.fixture
def root_engine(monkeypatch: pytest.MonkeyPatch) -> IntelligenceGraphEngine:
    root_backend = _FakeGraphBackend("default")
    engine = _make_root_engine(root_backend)
    monkeypatch.setattr(
        IntelligenceGraphEngine,
        "get_or_create",
        classmethod(lambda cls, *a, **k: engine),
    )
    return engine


def _assert_no_workitem_leak(
    root_engine: IntelligenceGraphEngine, content_view, workitem_id: str
) -> None:
    host = _FakeHost(content_view.control_backend)
    control_engine = _ControlPlaneWorkItemEngine(host)
    control_engine.add_node(workitem_id, "WorkItem", {"status": "ready"})

    rows = control_engine.query_cypher(
        "MATCH (w:WorkItem {id: $id}) RETURN w.id AS id", {"id": workitem_id}
    )
    assert rows and rows[0]["id"] == workitem_id
    assert workitem_id not in content_view.backend.nodes
    assert workitem_id not in root_engine.backend.nodes
    assert workitem_id in root_engine.control_backend.nodes


def test_content_graph_warmed_first_then_control_graph(root_engine) -> None:
    """Cache order A: ``code:repo-a`` resolved/cached BEFORE ``__control__``."""
    content_view = ingest_routing.engine_for_graph("code:repo-a")
    control_view = ingest_routing.engine_for_graph(CONTROL_GRAPH_NAME)

    assert content_view.control_backend is root_engine.control_backend
    assert control_view.control_backend is root_engine.control_backend
    # Warming order must not alias the content and control views onto the
    # same backend -- they stay genuinely distinct physical graphs.
    assert content_view.backend is not control_view.backend

    _assert_no_workitem_leak(root_engine, content_view, "workitem:order-a:job-1")

    # A second lookup for the SAME graph name reuses the cached view object.
    assert ingest_routing.engine_for_graph("code:repo-a") is content_view


def test_control_graph_warmed_first_then_content_graph(root_engine) -> None:
    """Cache order B: ``__control__`` resolved/cached BEFORE ``code:repo-a``
    -- the mirror image of the order above. Must produce identical invariants;
    a defect that only manifests in one warm order would otherwise hide
    behind whichever order the live process happens to hit first.
    """
    control_view = ingest_routing.engine_for_graph(CONTROL_GRAPH_NAME)
    content_view = ingest_routing.engine_for_graph("code:repo-a")

    assert content_view.control_backend is root_engine.control_backend
    assert control_view.control_backend is root_engine.control_backend
    assert content_view.backend is not control_view.backend

    _assert_no_workitem_leak(root_engine, content_view, "workitem:order-b:job-1")

    assert ingest_routing.engine_for_graph(CONTROL_GRAPH_NAME) is control_view


def test_cache_order_does_not_change_which_object_is_cached_for_a_name(
    root_engine,
) -> None:
    """The SAME graph name must resolve to the SAME cached view object no
    matter what else was warmed around it, in either order."""
    a1 = ingest_routing.engine_for_graph("code:repo-a")
    ingest_routing.engine_for_graph(CONTROL_GRAPH_NAME)
    a2 = ingest_routing.engine_for_graph("code:repo-a")
    assert a1 is a2

    ingest_routing._reset_for_tests()

    c1 = ingest_routing.engine_for_graph(CONTROL_GRAPH_NAME)
    ingest_routing.engine_for_graph("code:repo-a")
    c2 = ingest_routing.engine_for_graph(CONTROL_GRAPH_NAME)
    assert c1 is c2
