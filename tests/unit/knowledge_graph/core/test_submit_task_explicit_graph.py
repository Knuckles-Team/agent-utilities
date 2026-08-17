"""U-06 — an explicit `graph` given to `submit_task` (the codebase/document
ingest submission chokepoint every `graph_ingest` async job funnels through,
see `agent_utilities.mcp.tools.write_ingest_tools.graph_ingest`) must be
persisted on the durable WorkItem's own metadata. The background worker that
LATER claims and executes the job runs under its OWN ambient session (a
different thread/process, with no memory of the original caller's
request-scoped session) -- without this stamp, the content write silently
lands wherever that worker's own default/ambient graph happens to be,
exactly the defect this closes ("codebase ingest could fan out or fall
back despite an explicit graph").

See ``agent_utilities.knowledge_graph.core.engine_tasks._bound_to_explicit_ingest_graph``
for the companion worker-side re-narrowing this metadata stamp feeds.
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.core.engine_tasks import TaskManagerMixin
from agent_utilities.knowledge_graph.core.session import GraphSession, use_session
from agent_utilities.security.brain_context import ActorContext, ActorType


class _FakeControlEngine:
    """Minimal WorkItem control authority: enough for `submit_work_item`'s
    create path (no pre-existing item, no dependencies), AND for
    `ensure_ingest_task_work_item`'s post-submit durable admission readback
    (U-24, `agent_utilities.orchestration.work_item.get_work_item`), which
    re-queries by id and raises `WorkItemBackendUnavailable` if the row isn't
    found. Before the node exists this still answers "not found" (so the
    pre-create idempotency probe in `_submit_work_item` behaves correctly);
    once `add_node` has stored it, the same id resolves to its properties so
    the readback that immediately follows `add_node` in the real code path
    sees what was actually written, instead of the always-empty stub making
    every submission look like a stranded admission.
    """

    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}

    def query_cypher(self, cypher, params=None):
        node_id = (params or {}).get("id")
        if node_id is not None and node_id in self.nodes:
            return [{"id": node_id, **self.nodes[node_id]}]
        return []

    def add_node(self, node_id, node_type, properties=None):
        self.nodes[node_id] = dict(properties or {})

    def link_nodes(self, *a, **k):
        pass


class _FakeHost:
    def __init__(self) -> None:
        self._work_item_engine = _FakeControlEngine()
        self._task_queue_backend_name = "sqlite"
        self.backend = None

    def start_task_workers(self):
        pass


def _session() -> GraphSession:
    actor = ActorContext(
        actor_id="test-service",
        actor_type=ActorType.AUTOMATED_SERVICE,
        roles=("test",),
        tenant_id="test-tenant",
        authenticated=True,
    )
    return GraphSession(
        actor=actor,
        tenant="test-tenant",
        scopes=frozenset({"kg:read", "kg:write"}),
        policy_version="test-policy",
        audience="test-audience",
    )


def test_submit_task_stamps_explicit_graph_on_workitem_metadata():
    host = _FakeHost()
    with use_session(_session()):
        job_id = TaskManagerMixin.submit_task(
            host,
            target_path="checkout/repo-a",
            is_codebase=True,
            provenance={},
            task_type="codebase",
            skip_dedupe=True,
            graph="kf-pilot:code-ingest",
        )

    work_item_id = f"workitem:ingest_task:{job_id}"
    stored = host._work_item_engine.nodes[work_item_id]
    assert stored["metadata"]["graph"] == "kf-pilot:code-ingest"


def test_submit_task_omits_graph_key_when_not_explicit():
    """Backward compatibility: no `graph` kwarg -> no `graph` key at all, not
    an empty string -- the worker-side check (`cb_meta.get("graph") or ""`)
    treats both as "no narrowing", but the metadata shape itself must not
    silently change for every pre-existing caller that never passes `graph`.
    """
    host = _FakeHost()
    with use_session(_session()):
        job_id = TaskManagerMixin.submit_task(
            host,
            target_path="checkout/repo-b",
            is_codebase=True,
            provenance={},
            task_type="codebase",
            skip_dedupe=True,
        )

    work_item_id = f"workitem:ingest_task:{job_id}"
    stored = host._work_item_engine.nodes[work_item_id]
    assert "graph" not in stored["metadata"]


def test_submit_task_extra_meta_cannot_spoof_the_graph_key():
    """The explicit `graph` kwarg -- the caller-validated selector checked
    against the engine's own catalog by `resolve_explicit_graph` at the
    MCP/REST boundary -- must always win over anything an `extra_meta`
    caller happens to set under the same key. `extra_meta` is a free-form
    dict threaded from less-trusted call sites; it must never be able to
    retarget a WorkItem's control-validated graph.
    """
    host = _FakeHost()
    with use_session(_session()):
        job_id = TaskManagerMixin.submit_task(
            host,
            target_path="checkout/repo-c",
            is_codebase=True,
            provenance={},
            task_type="codebase",
            skip_dedupe=True,
            extra_meta={"graph": "attacker-supplied-graph"},
            graph="kf-pilot:code-ingest",
        )

    work_item_id = f"workitem:ingest_task:{job_id}"
    stored = host._work_item_engine.nodes[work_item_id]
    assert stored["metadata"]["graph"] == "kf-pilot:code-ingest"
