"""``AGENT_CLAIM_BACKEND=workitem`` — the MIGRATED ``:AgentTask``/dispatch path (AU-P1-1).

Exercises ``execute_agent_task_turn`` end to end with the WorkItem state
machine as the authoritative backend (rather than the KG ``:AgentLease``-only
path or the raw engine-native probe): a legacy ``:AgentTask`` node is shadowed
1:1 by a :class:`~agent_utilities.orchestration.work_item.WorkItemStatus`
node, claimed/committed through it, with the legacy ``:AgentTask``/
``:AgentLease`` nodes mirrored (not read) for unmigrated consumers
(``fleet_reconciler``), and a real cross-task dependency released atomically
through WorkItem's own dep_count mechanics.

Follows the same minimal-engine-double pattern as
``tests/unit/knowledge_graph/test_agentos_gap6_objects.py``'s ``_Gap6Engine``,
extended with the WorkItem-by-id read work_item.py needs.
"""

from __future__ import annotations

import pytest

from agent_utilities.orchestration import agent_dispatch_worker as worker
from agent_utilities.orchestration import work_item as wi


class _BridgeEngine:
    """Engine double covering AgentTask/AgentLease/AgentCapabilityGrant/
    ActionDecision (what ``execute_agent_task_turn`` touches) PLUS a
    WorkItem-by-id read (what the ``workitem`` claim backend touches)."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}
        self.edges: list[tuple[str, str, str]] = []

    def add_node(self, node_id, node_type, properties=None, ephemeral=False):
        node = self.nodes.setdefault(node_id, {})
        node["type"] = node_type
        node.update(properties or {})
        return node

    def add_edge(self, source, target, rel_type, **properties):
        self.edges.append((source, target, str(rel_type)))

    # work_item._link tries link_nodes first — same shape as add_edge here.
    def link_nodes(
        self, source_id, target_id, rel_type, properties=None, ephemeral=False
    ):
        self.add_edge(source_id, target_id, rel_type)

    def compare_and_set_node_fields(self, node_id, conditions, updates):
        node = self.nodes.get(node_id)
        if node is None:
            return False
        for k, v in conditions.items():
            if node.get(k) != v:
                return False
        node.update(updates)
        return True

    def by_type(self, node_type: str) -> list[dict]:
        return [n for n in self.nodes.values() if n.get("type") == node_type]

    def query_cypher(self, q, params=None):
        params = params or {}

        if "WorkItem {id: $id}" in q:
            node = self.nodes.get(params.get("id"))
            if node is None or node.get("type") != "WorkItem":
                return []
            row = {"id": params["id"]}
            for f in wi._FIELDS:
                row[f] = node.get(f)
            return [row]

        if "AgentTask {id: $id}" in q and "depends_on_task_ids" in q:
            node = self.nodes.get(params.get("id"))
            if node is None:
                return []
            return [
                {
                    "status": node.get("status"),
                    "depends_on_task_ids": node.get("depends_on_task_ids") or [],
                    "dag_id": node.get("dag_id"),
                    "checkpoint_id": node.get("checkpoint_id"),
                }
            ]

        if "AgentTask {id: $id}" in q:  # the narrower dag_id/checkpoint_id-only read
            node = self.nodes.get(params.get("id"))
            if node is None:
                return []
            return [
                {
                    "dag_id": node.get("dag_id"),
                    "checkpoint_id": node.get("checkpoint_id"),
                }
            ]

        if "AgentLease {resource_id: $rid}" in q:
            leases = [
                n
                for n in self.nodes.values()
                if n.get("type") == "AgentLease"
                and n.get("resource_id") == params.get("rid")
            ]
            leases.sort(key=lambda n: n.get("acquired_at", 0.0), reverse=True)
            if not leases:
                return []
            top = leases[0]
            return [
                {
                    "owner_token": top.get("owner_token"),
                    "lease_expires_at": top.get("lease_expires_at"),
                    "lease_epoch": top.get("lease_epoch"),
                }
            ]

        if "governance_rule" in q:
            return []
        if "ActionDecision {kind:" in q:
            return []
        if "AgentCapabilityGrant {capability:" in q:
            agent_id = params.get("agent_id")
            capability = params.get("capability")
            grants = [
                (nid, n)
                for nid, n in self.nodes.items()
                if n.get("type") == "AgentCapabilityGrant"
                and n.get("agent_id") == agent_id
                and n.get("capability") == capability
            ]
            grants.sort(key=lambda pair: pair[1].get("granted_at", 0.0), reverse=True)
            if not grants:
                return []
            gid, top = grants[0]
            return [
                {
                    "id": gid,
                    "issuer": top.get("issuer"),
                    "granted_at": top.get("granted_at"),
                    "expires_at": top.get("expires_at"),
                    "revoked": top.get("revoked"),
                }
            ]
        return []


@pytest.fixture(autouse=True)
def _workitem_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_CLAIM_BACKEND", "workitem")


def test_agent_task_completes_via_work_item_backend_and_mirrors_legacy_nodes() -> None:
    engine = _BridgeEngine()
    engine.add_node("task-1", "AgentTask", properties={"status": "pending"})

    outcome = worker.execute_agent_task_turn(
        engine, "task-1", agent_id="agent-1", executor=lambda claim: "did the work"
    )

    assert outcome == "completed"
    # Legacy mirrors, for unmigrated readers (fleet_reconciler / dashboards).
    assert engine.nodes["task-1"]["status"] == "completed"
    assert engine.by_type("AgentLease")

    # WorkItem is authoritative: the shadow is succeeded, with a result_ref.
    item_id = wi.agent_task_work_item_id("task-1")
    item = wi.get_work_item(engine, item_id)
    assert item is not None
    assert item["status"] == wi.WorkItemStatus.SUCCEEDED.value
    assert item["result_ref"] == "outcome:agent_task:task-1"


def test_agent_task_unroutable_is_terminal_failed_not_retried_via_work_item() -> None:
    engine = _BridgeEngine()
    engine.add_node("task-2", "AgentTask", properties={"status": "pending"})

    outcome = worker.execute_agent_task_turn(engine, "task-2", agent_id="")
    assert outcome == "unroutable"

    item_id = wi.agent_task_work_item_id("task-2")
    item = wi.get_work_item(engine, item_id)
    assert item["status"] == wi.WorkItemStatus.FAILED.value
    assert item["attempt"] == 1  # no retry — unroutable is non-retryable


def test_agent_task_cross_dependency_released_atomically_via_work_item_backend() -> (
    None
):
    """task-B depends on task-A (a real ``TASK_DEPENDS_ON`` DAG edge, as
    ``TeamComposition.to_durable_task_dag`` would write). B's WorkItem shadow
    is materialized by an early (failed) claim attempt — realistic once a
    reconciler/worker is polling both tasks — then A's completion atomically
    releases it."""
    engine = _BridgeEngine()
    engine.add_node(
        "task-a",
        "AgentTask",
        properties={"status": "pending", "depends_on_task_ids": []},
    )
    engine.add_node(
        "task-b",
        "AgentTask",
        properties={"status": "blocked", "depends_on_task_ids": ["task-a"]},
    )

    # Something tries task-B first (e.g. an eager dispatcher) — not ready yet;
    # this materializes B's WorkItem shadow (and A's) via ensure_agent_task_work_item.
    early = worker.execute_agent_task_turn(engine, "task-b", agent_id="agent-1")
    assert early == "skipped"  # claim_specific declines: still 'submitted'

    b_item_id = wi.agent_task_work_item_id("task-b")
    a_item_id = wi.agent_task_work_item_id("task-a")
    assert (
        wi.get_work_item(engine, b_item_id)["status"]
        == wi.WorkItemStatus.SUBMITTED.value
    )
    assert b_item_id in wi.get_work_item(engine, a_item_id)["downstream_ids"]

    # A completes — B's dependency count hits zero atomically, in the same
    # CAS that flips it to ready.
    outcome_a = worker.execute_agent_task_turn(
        engine, "task-a", agent_id="agent-1", executor=lambda claim: "a done"
    )
    assert outcome_a == "completed"
    assert (
        wi.get_work_item(engine, b_item_id)["status"] == wi.WorkItemStatus.READY.value
    )

    # Now B is genuinely claimable and completes too.
    outcome_b = worker.execute_agent_task_turn(
        engine, "task-b", agent_id="agent-1", executor=lambda claim: "b done"
    )
    assert outcome_b == "completed"
    assert (
        wi.get_work_item(engine, b_item_id)["status"]
        == wi.WorkItemStatus.SUCCEEDED.value
    )


def test_engine_claim_workitem_backend_is_selectable_and_exclusive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The three-way backend switch stays EXCLUSIVE: selecting ``workitem``
    never touches the kg or raw-engine-probe primitives."""
    from agent_utilities.orchestration import engine_claim

    kg_calls: list[str] = []
    monkeypatch.setattr(
        engine_claim,
        "_claim_agent_task_kg",
        lambda engine, task_id, **kw: kg_calls.append(task_id) or None,
    )

    def _fail_if_engine_probed(*a, **k):  # pragma: no cover - must never run
        raise AssertionError(
            "raw engine-native probe must never run for the workitem backend"
        )

    monkeypatch.setattr(engine_claim, "_try_engine_claim", _fail_if_engine_probed)

    engine = _BridgeEngine()
    engine.add_node("task-9", "AgentTask", properties={"status": "pending"})

    claim = engine_claim.claim_agent_task(
        engine, "task-9", backend=engine_claim.AGENT_CLAIM_BACKEND_WORKITEM
    )
    assert claim is not None
    assert claim["task_id"] == "task-9"
    assert claim["fence_token"] == 1
    assert kg_calls == []
