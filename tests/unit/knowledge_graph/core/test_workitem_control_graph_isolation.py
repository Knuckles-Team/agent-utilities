"""U-05/U-16 — WorkItem control authority must stay fixed to the process
root's control backend across a `for_graph()` content-scope view, so
submitting/claiming/committing a WorkItem while scoped to an explicit
content graph never creates or reads that WorkItem in the content graph.

Mechanism under test: ``IntelligenceGraphEngine.for_graph()`` builds its view
via ``view.__dict__ = self.__dict__.copy()`` (agent_utilities/knowledge_graph/
core/engine.py) WITHOUT re-deriving ``control_backend`` from the new graph's
backend -- ``_build_control_backend()`` only ever runs once, at root
construction. ``_ControlPlaneWorkItemEngine`` (engine_tasks.py) always reads
``self._host.control_backend`` (via the ``_control`` property), never
``self._host.backend``. Together these keep every WorkItem
create/read/claim/renew/commit pinned to the control graph regardless of
which content graph the caller most recently selected.

This is exactly the fix U-05/U-16 in
plans/graph-os-completion-program/bug-analysis.md describe: "derive the
control graph solely from the verified tenant and configured default;
rebind the trusted session around every create/read/claim/renew/commit and
control query". The live U-16 defect was a WorkItem visible only in the
explicit content graph while `job_status` (control-routed) reported "not
found" -- proving control authority had been retargeted to the content
graph instead of staying fixed.
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
from agent_utilities.knowledge_graph.core.engine_tasks import (
    _ControlPlaneWorkItemEngine,
)


class _FakeGraphBackend:
    """Minimal named-graph-view backend. ``for_graph()`` returns a distinct
    instance with its own isolated node store, mirroring the real
    EpistemicGraphBackend/GraphComputeEngine per-graph-view contract (a
    genuinely different physical graph, not a filtered view of the same
    store)."""

    def __init__(self, graph_name: str) -> None:
        self.graph_name = graph_name
        self.nodes: dict[str, dict] = {}
        # Real backends expose `.graph` as themselves for the single-process
        # compute-is-authority case; mirrored here so `for_graph()` doesn't
        # need a second graph_compute client to resolve identity.
        self.graph = self

    def for_graph(self, name: str) -> _FakeGraphBackend:
        if name == self.graph_name:
            return self
        return _FakeGraphBackend(name)

    def add_node(self, node_id, label="", **props):
        self.nodes[node_id] = {"label": label, **props}

    def execute_read(self, cypher, params=None):
        params = params or {}
        ids = params.get("ids")
        if ids is None and "id" in params:
            ids = [params["id"]]
        return [
            {"id": nid, **{k: v for k, v in row.items() if k != "label"}}
            for nid, row in self.nodes.items()
            if not ids or nid in ids
        ]


class _FakeHost:
    """Stand-in for ``TaskManagerMixin``'s control-plane surface: routes
    exclusively through ``control_backend``, the same shape
    ``IntelligenceGraphEngine._control``/``_control_cypher`` use."""

    def __init__(self, control_backend: _FakeGraphBackend) -> None:
        self.control_backend = control_backend

    @property
    def _control(self):
        return self.control_backend

    def _control_cypher(self, cypher, params=None):
        return self._control.execute_read(cypher, params)


def _make_root_engine(backend: _FakeGraphBackend) -> IntelligenceGraphEngine:
    """Build a root ``IntelligenceGraphEngine`` without its full ``__init__``
    side effects (no live transport/backend factory) -- the same
    ``object.__new__`` bypass ``for_graph()`` itself uses for its view."""
    engine = object.__new__(IntelligenceGraphEngine)
    engine.backend = backend
    engine.graph_compute = backend
    engine.graph = backend
    engine._compute_is_authority = True
    engine._process_owned = True
    engine._process_root = engine
    engine.active_schema_pack = None
    engine.control_backend = engine._build_control_backend()
    return engine


def test_for_graph_view_narrows_content_but_keeps_the_root_control_backend():
    root_backend = _FakeGraphBackend("default")
    engine = _make_root_engine(root_backend)

    view = engine.for_graph("kf-pilot:code-ingest")

    assert view.backend is not engine.backend  # content graph really narrowed
    assert view.backend.graph_name == "kf-pilot:code-ingest"
    # The control authority must NOT follow the content narrowing.
    assert view.control_backend is engine.control_backend


def test_workitem_submitted_under_a_content_view_lands_in_control_not_content():
    root_backend = _FakeGraphBackend("default")
    engine = _make_root_engine(root_backend)
    content_view = engine.for_graph("kf-pilot:code-ingest")

    # Real call shape: `TaskManagerMixin._work_item_engine` builds
    # `_ControlPlaneWorkItemEngine(self)` where `self.control_backend` is
    # whatever the (possibly content-scoped) host engine carries.
    host = _FakeHost(content_view.control_backend)
    control_engine = _ControlPlaneWorkItemEngine(host)
    control_engine.add_node(
        "workitem:ingest_task:job-1", "WorkItem", {"status": "ready"}
    )

    # Queryable via the control-routed read (the same path `job_status` uses)...
    rows = control_engine.query_cypher(
        "MATCH (w:WorkItem {id: $id}) RETURN w.id AS id",
        {"id": "workitem:ingest_task:job-1"},
    )
    assert rows and rows[0]["id"] == "workitem:ingest_task:job-1"

    # ...and it is correctly ABSENT from the selected content graph's own
    # backend -- the control write never touched it. It lands in the
    # dedicated __control__-graph view (U-16/BUG-113:
    # `_build_control_backend` resolves a genuinely distinct backend via
    # `for_graph(CONTROL_GRAPH_NAME)`, not `root_backend` itself verbatim --
    # see its docstring for why that split is the fix, not a regression).
    assert "workitem:ingest_task:job-1" not in content_view.backend.nodes
    assert "workitem:ingest_task:job-1" not in root_backend.nodes
    assert "workitem:ingest_task:job-1" in engine.control_backend.nodes


def test_KNOWN_BAD_control_authority_rederived_from_content_view_leaks_the_workitem():
    """Negative proof: if control authority were (incorrectly) re-derived
    from the graph-scoped view's OWN backend -- the U-16 live defect's root
    cause -- the WorkItem leaks into the content graph and a control-routed
    `job_status` read never finds it. This shows the unsafe wiring really
    does misbehave, so the two assertions above are pinning a real
    invariant rather than a vacuous one.
    """
    root_backend = _FakeGraphBackend("default")
    engine = _make_root_engine(root_backend)
    content_view = engine.for_graph("kf-pilot:code-ingest")

    # Simulate the BUGGY wiring: a host whose control authority was
    # retargeted to the content view's backend instead of staying fixed to
    # the root's `control_backend`.
    buggy_host = _FakeHost(content_view.backend)  # WRONG: content, not control
    buggy_control_engine = _ControlPlaneWorkItemEngine(buggy_host)
    buggy_control_engine.add_node(
        "workitem:ingest_task:job-2", "WorkItem", {"status": "ready"}
    )

    # The WorkItem is now split into the content graph (leaked)...
    assert "workitem:ingest_task:job-2" in content_view.backend.nodes

    # ...and invisible to a job_status-style read against the REAL control
    # graph -- exactly the observed "Job ... not found" symptom.
    real_control_engine = _ControlPlaneWorkItemEngine(_FakeHost(engine.control_backend))
    rows = real_control_engine.query_cypher(
        "MATCH (w:WorkItem {id: $id}) RETURN w.id AS id",
        {"id": "workitem:ingest_task:job-2"},
    )
    assert rows == []
    assert "workitem:ingest_task:job-2" not in root_backend.nodes
