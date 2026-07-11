"""The unified engine-native ``WorkItem`` state machine (AU-P1-1).

Exercises the full lifecycle::

    submitted -> ready -> leased(fencing_token) -> running(heartbeat,attempt)
        -> succeeded(result_ref) | failed(error_ref) | cancelled | dead_letter

against a minimal in-memory fake engine (dict-backed nodes + a REAL
compare_and_set_node_fields, mirroring the ``_ClaimHarness``/fake-CAS pattern
``tests/unit/knowledge_graph/test_task_claim_cas.py`` already uses for the
``:Task`` claim) — no live epistemic-graph engine required.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from agent_utilities.orchestration import work_item as wi

# ---------------------------------------------------------------------------
# Fake engine: dict-backed nodes with a real atomic compare_and_set_node_fields
# ---------------------------------------------------------------------------


class FakeEngine:
    """Minimal engine double: add_node/link_nodes/query_cypher/CAS over an
    in-memory node store, with just enough Cypher pattern recognition to
    answer the exact queries ``work_item.py`` issues."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: list[tuple[str, str, str]] = []
        self._lock = threading.Lock()

    # -- write surface (GraphEngineProtocol-shaped) --------------------

    def add_node(
        self,
        node_id: str,
        node_type: str,
        properties: dict[str, Any] | None = None,
        ephemeral: bool = False,
    ) -> dict[str, Any]:
        props = dict(properties or {})
        with self._lock:
            existing = self.nodes.get(node_id, {})
            merged = {**existing, **props, "label": node_type}
            self.nodes[node_id] = merged
            return dict(merged)

    def link_nodes(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: dict | None = None,
        ephemeral: bool = False,
    ) -> None:
        self.edges.append((source_id, target_id, str(rel_type)))

    def compare_and_set_node_fields(
        self, node_id: str, conditions: dict[str, Any], updates: dict[str, Any]
    ) -> bool:
        with self._lock:
            node = self.nodes.get(node_id)
            if node is None:
                return False
            for k, v in conditions.items():
                if node.get(k) != v:
                    return False
            node.update(updates)
            return True

    # -- read surface ----------------------------------------------------

    def query_cypher(
        self, cypher: str, params: dict | None = None
    ) -> list[dict[str, Any]]:
        params = params or {}
        q = " ".join(cypher.split())

        if q.startswith("MATCH (w:WorkItem {id: $id}) RETURN w.id"):
            node = self.nodes.get(params["id"])
            if node is None or node.get("label") != "WorkItem":
                return []
            row = {"id": params["id"]}
            for f in wi._FIELDS:
                row[f] = node.get(f)
            return [row]

        if q.startswith("MATCH (w:WorkItem {status: $status, prio_bucket: $bucket})"):
            rows = []
            for nid, node in self.nodes.items():
                if node.get("label") != "WorkItem":
                    continue
                if (
                    node.get("status") != params["status"]
                    or node.get("prio_bucket") != params["bucket"]
                ):
                    continue
                rows.append(
                    {
                        "id": nid,
                        "created_at": node.get("created_at"),
                        "next_retry_at": node.get("next_retry_at"),
                        "resource_class": node.get("resource_class"),
                        "tenant": node.get("tenant"),
                        "fairness_group": node.get("fairness_group"),
                    }
                )
            return rows

        if q.startswith("MATCH (w:WorkItem) WHERE w.status IN $statuses AND"):
            rows = []
            for nid, node in self.nodes.items():
                if node.get("label") != "WorkItem":
                    continue
                if node.get("status") not in params["statuses"]:
                    continue
                expires = node.get("lease_expires_at")
                if expires is None or not (expires < params["now"]):
                    continue
                rows.append({"id": nid})
            return rows

        if q.startswith(
            "MATCH (w:WorkItem {tenant: $tenant}) WHERE NOT w.status IN $terminal"
        ):
            c = 0
            for node in self.nodes.values():
                if node.get("label") != "WorkItem":
                    continue
                if node.get("tenant") != params["tenant"]:
                    continue
                if node.get("status") in params["terminal"]:
                    continue
                c += 1
            return [{"c": c}]

        if q.startswith("MATCH (t:AgentTask {id: $id}) RETURN t.status"):
            node = self.nodes.get(params["id"])
            if node is None or node.get("label") != "AgentTask":
                return []
            return [
                {
                    "status": node.get("status"),
                    "depends_on_task_ids": node.get("depends_on_task_ids") or [],
                    "dag_id": node.get("dag_id"),
                    "checkpoint_id": node.get("checkpoint_id"),
                }
            ]

        if q.startswith("MATCH (t:AgentTask {id: $id}) RETURN t.dag_id"):
            node = self.nodes.get(params["id"])
            if node is None:
                return []
            return [
                {
                    "dag_id": node.get("dag_id"),
                    "checkpoint_id": node.get("checkpoint_id"),
                }
            ]

        if q.startswith("MATCH (c:Concept) WHERE c.id = $id"):
            node = self.nodes.get(params["id"])
            if node is None or node.get("label") != "Concept":
                return []
            return [
                {
                    "id": params["id"],
                    "status": node.get("status"),
                    "updated_at": node.get("updated_at"),
                }
            ]

        if q.startswith("MATCH (t:Task {id: $id}) RETURN t.id"):
            node = self.nodes.get(params["id"])
            if node is None or node.get("label") != "Task":
                return []
            return [{"id": params["id"], "status": node.get("status")}]

        raise AssertionError(f"FakeEngine: unrecognized query: {q[:160]!r}")


class NoCasEngine(FakeEngine):
    """Otherwise-identical to FakeEngine, but with no atomic CAS — must fail
    loud when a WorkItem transition needs one, never silently no-op."""

    compare_and_set_node_fields = None  # type: ignore[assignment]


@pytest.fixture
def engine() -> FakeEngine:
    return FakeEngine()


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------


def test_submit_with_no_deps_is_immediately_ready(engine: FakeEngine) -> None:
    item_id = wi.submit_work_item(engine, kind="generic", payload_ref="p1")
    item = wi.get_work_item(engine, item_id)
    assert item is not None
    assert item["status"] == wi.WorkItemStatus.READY.value
    assert item["dep_count"] == 0


def test_submit_with_unmet_dep_is_submitted_not_ready(engine: FakeEngine) -> None:
    parent_id = wi.submit_work_item(engine, kind="generic", payload_ref="parent")
    child_id = wi.submit_work_item(
        engine, kind="generic", payload_ref="child", depends_on=[parent_id]
    )

    child = wi.get_work_item(engine, child_id)
    assert child is not None
    assert child["status"] == wi.WorkItemStatus.SUBMITTED.value
    assert child["dep_count"] == 1

    parent = wi.get_work_item(engine, parent_id)
    assert parent is not None
    assert child_id in parent["downstream_ids"]


def test_submit_is_idempotent_upsert_on_explicit_id(engine: FakeEngine) -> None:
    fixed_id = "workitem:fixed-1"
    first = wi.submit_work_item(
        engine, kind="generic", payload_ref="a", work_item_id=fixed_id
    )
    second = wi.submit_work_item(
        engine, kind="generic", payload_ref="a-changed", work_item_id=fixed_id
    )
    assert first == second == fixed_id
    item = wi.get_work_item(engine, fixed_id)
    assert item["payload_ref"] == "a"  # second submit was a no-op, not an overwrite


def test_tenant_quota_exceeded_raises(engine: FakeEngine) -> None:
    wi.submit_work_item(
        engine, kind="generic", payload_ref="1", tenant="acme", max_tenant_in_flight=1
    )
    with pytest.raises(wi.TenantQuotaExceeded):
        wi.submit_work_item(
            engine,
            kind="generic",
            payload_ref="2",
            tenant="acme",
            max_tenant_in_flight=1,
        )


# ---------------------------------------------------------------------------
# claim / lease / fencing
# ---------------------------------------------------------------------------


def test_claim_specific_transitions_ready_to_leased_with_fencing_token(
    engine: FakeEngine,
) -> None:
    item_id = wi.submit_work_item(engine, kind="generic", payload_ref="p")
    claim = wi.claim_specific(engine, item_id, token="host:1", now=1000.0)
    assert claim is not None
    assert claim["fence_token"] == 1
    assert claim["attempt"] == 1

    item = wi.get_work_item(engine, item_id)
    assert item["status"] == wi.WorkItemStatus.LEASED.value
    assert item["lease_owner"] == "host:1"
    assert item["lease_epoch"] == 1


def test_claim_specific_skips_live_lease_held_elsewhere(engine: FakeEngine) -> None:
    item_id = wi.submit_work_item(engine, kind="generic", payload_ref="p")
    first = wi.claim_specific(
        engine, item_id, token="host:1", now=1000.0, lease_ttl_s=3600.0
    )
    assert first is not None

    second = wi.claim_specific(engine, item_id, token="host:2", now=1000.0 + 10.0)
    assert second is None  # lease is still fresh


def test_claim_specific_reclaims_after_stale_lease_bumps_fencing(
    engine: FakeEngine,
) -> None:
    item_id = wi.submit_work_item(engine, kind="generic", payload_ref="p")
    first = wi.claim_specific(
        engine, item_id, token="host:1", now=1000.0, lease_ttl_s=10.0
    )
    assert first["fence_token"] == 1

    # Lease has now expired (11s later, ttl was 10s) — a new claimer reclaims it.
    second = wi.claim_specific(
        engine, item_id, token="host:2", now=1011.0, lease_ttl_s=3600.0
    )
    assert second is not None
    assert second["fence_token"] == 2  # strictly greater than the stale holder's epoch


def test_mark_running_and_heartbeat_extend_the_lease(engine: FakeEngine) -> None:
    item_id = wi.submit_work_item(engine, kind="generic", payload_ref="p")
    claim = wi.claim_specific(
        engine, item_id, token="host:1", now=1000.0, lease_ttl_s=60.0
    )
    assert wi.mark_running(engine, item_id, claim, now=1001.0)
    item = wi.get_work_item(engine, item_id)
    assert item["status"] == wi.WorkItemStatus.RUNNING.value

    assert wi.heartbeat(engine, item_id, claim, now=1030.0, lease_ttl_s=60.0)
    item = wi.get_work_item(engine, item_id)
    assert item["lease_expires_at"] == 1030.0 + 60.0


def test_claim_next_respects_priority_bucket_ordering(engine: FakeEngine) -> None:
    low_id = wi.submit_work_item(
        engine, kind="generic", payload_ref="low", priority="background", now=1.0
    )
    high_id = wi.submit_work_item(
        engine, kind="generic", payload_ref="high", priority="critical", now=2.0
    )

    claim = wi.claim_next(engine, now=1000.0)
    assert claim["work_item_id"] == high_id

    claim2 = wi.claim_next(engine, now=1001.0)
    assert claim2["work_item_id"] == low_id


def test_claim_next_filters_by_resource_class(engine: FakeEngine) -> None:
    wi.submit_work_item(
        engine, kind="generic", payload_ref="cpu-1", resource_class="cpu"
    )
    gpu_id = wi.submit_work_item(
        engine, kind="generic", payload_ref="gpu-1", resource_class="gpu"
    )

    claim = wi.claim_next(engine, resource_class="gpu", now=1000.0)
    assert claim["work_item_id"] == gpu_id


def test_cas_backend_unavailable_fails_loud_not_silent() -> None:
    no_cas = NoCasEngine()
    no_cas.add_node(
        "workitem:x", "WorkItem", properties={"status": wi.WorkItemStatus.READY.value}
    )
    with pytest.raises(wi.WorkItemBackendUnavailable):
        wi.claim_specific(no_cas, "workitem:x", now=1.0)


# ---------------------------------------------------------------------------
# lease-expiry reaping — re-ready with bumped fencing, or dead_letter
# ---------------------------------------------------------------------------


def test_reap_expired_lease_requeues_to_ready_and_stale_commit_is_fenced(
    engine: FakeEngine,
) -> None:
    item_id = wi.submit_work_item(
        engine, kind="generic", payload_ref="p", max_attempts=5
    )
    claim = wi.claim_specific(
        engine, item_id, token="host:1", now=1000.0, lease_ttl_s=10.0
    )
    wi.mark_running(engine, item_id, claim, now=1000.0)

    # Worker "dies" — the lease is now expired at t=1500.
    result = wi.reap_expired_leases(engine, now=1500.0)
    assert result["reaped_ready"] == [item_id]
    assert result["reaped_dead_letter"] == []

    item = wi.get_work_item(engine, item_id)
    assert item["status"] == wi.WorkItemStatus.READY.value
    assert item["lease_epoch"] == 2  # bumped past the dead holder's epoch (1)

    # The dead holder eventually "finishes" and tries to commit with its
    # stale claim — must be rejected, never overwrite the reclaimed item.
    outcome = wi.commit_result(
        engine, item_id, claim, outcome="succeeded", result_ref="ref:1"
    )
    assert outcome == "fenced"
    assert wi.get_work_item(engine, item_id)["status"] == wi.WorkItemStatus.READY.value


def test_reap_expired_lease_exhausted_retries_goes_to_dead_letter(
    engine: FakeEngine,
) -> None:
    item_id = wi.submit_work_item(
        engine, kind="generic", payload_ref="p", max_attempts=1
    )
    claim = wi.claim_specific(
        engine, item_id, token="host:1", now=1000.0, lease_ttl_s=10.0
    )
    wi.mark_running(engine, item_id, claim, now=1000.0)
    assert wi.get_work_item(engine, item_id)["attempt"] == 1  # == max_attempts already

    result = wi.reap_expired_leases(engine, now=1500.0)
    assert result["reaped_dead_letter"] == [item_id]
    assert result["reaped_ready"] == []
    item = wi.get_work_item(engine, item_id)
    assert item["status"] == wi.WorkItemStatus.DEAD_LETTER.value
    assert "lease_expired" in item["error_ref"]


# ---------------------------------------------------------------------------
# commit_result — idempotent double-commit, retry-then-DLQ, cancellation
# ---------------------------------------------------------------------------


def test_commit_result_success_is_idempotent_noop_on_redelivery(
    engine: FakeEngine,
) -> None:
    item_id = wi.submit_work_item(engine, kind="generic", payload_ref="p")
    claim = wi.claim_and_start(engine, item_id, token="host:1", now=1000.0)

    first = wi.commit_result(
        engine, item_id, claim, outcome="succeeded", result_ref="ref:1", now=1010.0
    )
    assert first == "committed"
    item = wi.get_work_item(engine, item_id)
    assert item["status"] == wi.WorkItemStatus.SUCCEEDED.value
    assert item["result_ref"] == "ref:1"

    # Redelivery of the identical turn (at-least-once queue semantics):
    # must be a no-op, never re-running downstream release or overwriting result_ref.
    second = wi.commit_result(
        engine,
        item_id,
        claim,
        outcome="succeeded",
        result_ref="ref:DIFFERENT",
        now=1020.0,
    )
    assert second == "noop"
    item_after = wi.get_work_item(engine, item_id)
    assert item_after["result_ref"] == "ref:1"  # untouched by the redelivered commit


def test_commit_result_retryable_failure_then_exhausts_to_dead_letter(
    engine: FakeEngine,
) -> None:
    item_id = wi.submit_work_item(
        engine, kind="generic", payload_ref="p", max_attempts=2, backoff_base_s=5.0
    )

    claim1 = wi.claim_and_start(engine, item_id, token="host:1", now=1000.0)
    assert claim1["attempt"] == 1
    outcome1 = wi.commit_result(
        engine, item_id, claim1, outcome="failed", error_ref="boom-1", now=1001.0
    )
    assert outcome1 == "retry_scheduled"
    item = wi.get_work_item(engine, item_id)
    assert item["status"] == wi.WorkItemStatus.READY.value
    assert item["next_retry_at"] == 1001.0 + 5.0
    assert item["lease_epoch"] == 2  # fenced past the failed attempt

    # Backoff hasn't elapsed yet — not claimable.
    assert wi.claim_specific(engine, item_id, token="host:2", now=1002.0) is None

    # Backoff elapsed — second (and last) attempt.
    claim2 = wi.claim_and_start(engine, item_id, token="host:2", now=1010.0)
    assert claim2 is not None
    assert claim2["attempt"] == 2

    outcome2 = wi.commit_result(
        engine, item_id, claim2, outcome="failed", error_ref="boom-2", now=1011.0
    )
    assert outcome2 == "dead_letter"
    item = wi.get_work_item(engine, item_id)
    assert item["status"] == wi.WorkItemStatus.DEAD_LETTER.value
    assert item["error_ref"] == "boom-2"


def test_commit_result_non_retryable_failure_is_terminal_immediately(
    engine: FakeEngine,
) -> None:
    item_id = wi.submit_work_item(
        engine, kind="generic", payload_ref="p", max_attempts=5
    )
    claim = wi.claim_and_start(engine, item_id, token="host:1", now=1000.0)
    outcome = wi.commit_result(
        engine,
        item_id,
        claim,
        outcome="failed",
        error_ref="no executor bound",
        retryable=False,
        now=1001.0,
    )
    assert outcome == "committed"
    item = wi.get_work_item(engine, item_id)
    assert item["status"] == wi.WorkItemStatus.FAILED.value
    assert item["attempt"] == 1  # never retried despite max_attempts=5


def test_cancel_work_item_from_ready_and_is_idempotent(engine: FakeEngine) -> None:
    item_id = wi.submit_work_item(engine, kind="generic", payload_ref="p")
    assert wi.cancel_work_item(engine, item_id, reason="user requested") is True
    item = wi.get_work_item(engine, item_id)
    assert item["status"] == wi.WorkItemStatus.CANCELLED.value

    # Idempotent: cancelling an already-cancelled item is a truthy no-op.
    assert wi.cancel_work_item(engine, item_id) is True


def test_cancel_work_item_cannot_override_a_real_terminal_outcome(
    engine: FakeEngine,
) -> None:
    item_id = wi.submit_work_item(engine, kind="generic", payload_ref="p")
    claim = wi.claim_and_start(engine, item_id, token="host:1", now=1000.0)
    wi.commit_result(
        engine, item_id, claim, outcome="succeeded", result_ref="ref:1", now=1001.0
    )

    assert wi.cancel_work_item(engine, item_id) is False
    assert (
        wi.get_work_item(engine, item_id)["status"] == wi.WorkItemStatus.SUCCEEDED.value
    )


# ---------------------------------------------------------------------------
# atomic dependency release
# ---------------------------------------------------------------------------


def test_child_becomes_ready_exactly_when_all_parents_succeed(
    engine: FakeEngine,
) -> None:
    parent1 = wi.submit_work_item(engine, kind="generic", payload_ref="parent1")
    parent2 = wi.submit_work_item(engine, kind="generic", payload_ref="parent2")
    child = wi.submit_work_item(
        engine, kind="generic", payload_ref="child", depends_on=[parent1, parent2]
    )

    assert (
        wi.get_work_item(engine, child)["status"] == wi.WorkItemStatus.SUBMITTED.value
    )
    assert wi.get_work_item(engine, child)["dep_count"] == 2

    claim1 = wi.claim_and_start(engine, parent1, token="host:1", now=1000.0)
    wi.commit_result(
        engine, parent1, claim1, outcome="succeeded", result_ref="r1", now=1001.0
    )

    # Only one of two parents done — child must still be blocked.
    child_state = wi.get_work_item(engine, child)
    assert child_state["status"] == wi.WorkItemStatus.SUBMITTED.value
    assert child_state["dep_count"] == 1

    claim2 = wi.claim_and_start(engine, parent2, token="host:2", now=1002.0)
    wi.commit_result(
        engine, parent2, claim2, outcome="succeeded", result_ref="r2", now=1003.0
    )

    # Second (and last) parent done — released atomically, in the same CAS
    # that decremented the counter to zero.
    child_state = wi.get_work_item(engine, child)
    assert child_state["status"] == wi.WorkItemStatus.READY.value
    assert child_state["dep_count"] == 0


def test_downstream_release_is_idempotent_no_double_release(engine: FakeEngine) -> None:
    parent = wi.submit_work_item(engine, kind="generic", payload_ref="parent")
    child = wi.submit_work_item(
        engine, kind="generic", payload_ref="child", depends_on=[parent]
    )

    claim = wi.claim_and_start(engine, parent, token="host:1", now=1000.0)
    wi.commit_result(
        engine, parent, claim, outcome="succeeded", result_ref="r1", now=1001.0
    )
    assert wi.get_work_item(engine, child)["status"] == wi.WorkItemStatus.READY.value

    # Redelivered commit of the same parent (idempotent no-op) must not
    # touch the child a second time.
    wi.commit_result(
        engine, parent, claim, outcome="succeeded", result_ref="r1-again", now=1002.0
    )
    assert wi.get_work_item(engine, child)["dep_count"] == 0
    assert wi.get_work_item(engine, child)["status"] == wi.WorkItemStatus.READY.value


# ---------------------------------------------------------------------------
# Loop / ingestion-Task read shims
# ---------------------------------------------------------------------------


def test_work_item_view_of_loop_maps_statuses(engine: FakeEngine) -> None:
    engine.add_node(
        "loop:develop:x",
        "Concept",
        properties={"status": "running", "updated_at": 123.0},
    )
    view = wi.work_item_view_of_loop(engine, "loop:develop:x")
    assert view["status"] == wi.WorkItemStatus.RUNNING.value
    assert view["native_status"] == "running"
    assert view["shim"] is True


def test_work_item_view_of_loop_unknown_returns_none(engine: FakeEngine) -> None:
    assert wi.work_item_view_of_loop(engine, "loop:nope") is None


def test_work_item_view_of_task_maps_statuses(engine: FakeEngine) -> None:
    engine.add_node("job-1", "Task", properties={"status": "dead_letter"})
    view = wi.work_item_view_of_task(engine, "job-1")
    assert view["status"] == wi.WorkItemStatus.DEAD_LETTER.value
    assert view["shim"] is True
