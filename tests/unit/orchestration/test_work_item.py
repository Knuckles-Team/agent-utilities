"""Strict native WorkItem authority tests."""

from __future__ import annotations

import threading
from typing import Any

import pytest

from agent_utilities.orchestration import work_item as wi


def _negative_claim(reason: str) -> dict[str, Any]:
    return {
        "schema_version": "1",
        "claimed": False,
        "reason": reason,
        "work_item_id": None,
        "kind": None,
        "payload_ref": None,
        "lease_holder_ref": None,
        "lease_epoch": None,
        "fencing_token": None,
        "lease_expires_at_ms": None,
        "attempt": None,
        "max_attempts": None,
        "tenant_in_flight": 0,
        "changed_work_item_ids": [],
    }


class NativeEngine:
    """In-memory double for the generated native WorkItem verbs."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: list[tuple[str, str, str]] = []
        self.native_calls: list[str] = []
        self._lock = threading.Lock()

    def add_node(
        self,
        node_id: str,
        node_type: str,
        properties: dict[str, Any] | None = None,
        ephemeral: bool = False,
    ) -> None:
        self.nodes[node_id] = {
            **self.nodes.get(node_id, {}),
            **dict(properties or {}),
            "label": node_type,
        }

    def link_nodes(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: dict[str, Any] | None = None,
        ephemeral: bool = False,
    ) -> None:
        self.edges.append((source_id, target_id, str(rel_type)))

    def compare_and_set_node_fields(
        self, node_id: str, conditions: dict[str, Any], updates: dict[str, Any]
    ) -> bool:
        """Submission uses CAS only to build immutable dependency indexes."""
        with self._lock:
            node = self.nodes.get(node_id)
            if node is None or any(node.get(k) != v for k, v in conditions.items()):
                return False
            node.update(updates)
            return True

    def query_cypher(
        self, cypher: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        params = params or {}
        query = " ".join(cypher.split())
        if query.startswith("MATCH (w:WorkItem {id: $id}) RETURN w.id"):
            node = self.nodes.get(str(params["id"]))
            if node is None or node.get("label") != "WorkItem":
                return []
            return [
                {"id": params["id"], **{field: node.get(field) for field in wi._FIELDS}}
            ]
        if query.startswith(
            "MATCH (w:WorkItem {tenant: $tenant}) WHERE NOT w.status IN $terminal"
        ):
            return [
                {
                    "c": sum(
                        node.get("label") == "WorkItem"
                        and node.get("tenant") == params["tenant"]
                        and node.get("status") not in params["terminal"]
                        for node in self.nodes.values()
                    )
                }
            ]
        raise AssertionError(f"unrecognized query: {query}")

    def _candidate(self, request: Any) -> tuple[str, dict[str, Any]] | None:
        item_id = request.work_item_id
        candidates = (
            [(str(item_id), self.nodes.get(str(item_id)))]
            if item_id
            else sorted(
                self.nodes.items(),
                key=lambda pair: (
                    int(pair[1].get("prio_bucket") or 0),
                    float(pair[1].get("created_at") or 0),
                ),
            )
        )
        now = float(request.now_ms) / 1000.0
        for candidate_id, node in candidates:
            if not node or node.get("label") != "WorkItem":
                continue
            if request.queue_ref and node.get("queue") != request.queue_ref:
                continue
            if (
                request.resource_class
                and node.get("resource_class") != request.resource_class
            ):
                continue
            status = node.get("status")
            if status in {"leased", "running"}:
                if float(node.get("lease_expires_at") or 0) >= now:
                    continue
            elif status != "ready":
                continue
            if float(node.get("next_retry_at") or 0) > now:
                continue
            return candidate_id, node
        return None

    def claim_work_item(self, request: Any) -> dict[str, Any]:
        self.native_calls.append("claim")
        with self._lock:
            selected = self._candidate(request)
            if selected is None:
                return _negative_claim("empty")
            item_id, node = selected
            attempt = int(node.get("attempt") or 0) + 1
            if attempt > int(node.get("max_attempts") or 1):
                node.update(status="dead_letter", error_ref="lease_exhausted")
                return _negative_claim("empty")
            epoch = int(node.get("lease_epoch") or 0) + 1
            node.update(
                status="leased",
                lease_owner=request.worker_ref,
                lease_epoch=epoch,
                fencing_token=epoch,
                attempt=attempt,
                lease_expires_at=float(request.now_ms + request.lease_ms) / 1000.0,
            )
            return {
                "schema_version": "1",
                "claimed": True,
                "reason": "claimed",
                "work_item_id": item_id,
                "kind": node.get("kind"),
                "payload_ref": node.get("payload_ref"),
                "lease_holder_ref": request.worker_ref,
                "lease_epoch": epoch,
                "fencing_token": epoch,
                "lease_expires_at_ms": request.now_ms + request.lease_ms,
                "attempt": attempt,
                "max_attempts": node["max_attempts"],
                "tenant_in_flight": 1,
                "changed_work_item_ids": [item_id],
            }

    def renew_work_item_lease(self, request: dict[str, Any]) -> dict[str, Any]:
        self.native_calls.append("renew")
        node = self.nodes.get(request["work_item_id"])
        if not self._owns(node, request):
            return {"renewed": False}
        node["lease_expires_at"] = float(request["now_unix"]) + float(
            request["lease_ttl"]
        )
        return {"renewed": True}

    @staticmethod
    def _owns(node: dict[str, Any] | None, request: dict[str, Any]) -> bool:
        return bool(
            node
            and node.get("lease_owner") == request.get("worker_ref")
            and node.get("lease_epoch") == request.get("expected_epoch")
            and node.get("fencing_token") == request.get("fencing_token")
        )

    def commit_work_item_result(self, request: dict[str, Any]) -> dict[str, Any]:
        self.native_calls.append("commit")
        node = self.nodes.get(request["work_item_id"])
        if node is None:
            return {"status": "missing"}
        if node.get("status") in wi.TERMINAL_WORK_ITEM_STATUSES:
            return {"status": "noop"}
        if not self._owns(node, request):
            return {"status": "fenced"}
        outcome = request["outcome"]
        if outcome == "failed" and request["retryable"]:
            if int(node["attempt"]) >= int(node["max_attempts"]):
                node.update(status="dead_letter", error_ref=request.get("error_ref"))
                return {"status": "dead_letter"}
            node.update(
                status="ready",
                next_retry_at=float(request["now_unix"])
                + float(node["backoff_base_s"]) * (2 ** (int(node["attempt"]) - 1)),
                lease_epoch=int(node["lease_epoch"]) + 1,
                lease_owner=None,
                lease_expires_at=None,
            )
            return {"status": "retry_scheduled"}
        node.update(
            status=outcome,
            result_ref=request.get("result_ref"),
            error_ref=request.get("error_ref"),
            completed_at=request["now_unix"],
            lease_owner=None,
            lease_expires_at=None,
        )
        if outcome == "succeeded":
            for child_id in node.get("downstream_ids") or []:
                child = self.nodes[child_id]
                child["dep_count"] = max(0, int(child["dep_count"]) - 1)
                if child["dep_count"] == 0:
                    child["status"] = "ready"
        return {"status": "committed"}

    def cancel_work_item(self, request: dict[str, Any]) -> dict[str, Any]:
        self.native_calls.append("cancel")
        node = self.nodes.get(request["work_item_id"])
        if node is None:
            return {"status": "missing"}
        if node.get("status") == "cancelled":
            return {"status": "cancelled"}
        if node.get("status") in wi.TERMINAL_WORK_ITEM_STATUSES:
            return {"status": "conflict"}
        node["status"] = "cancelled"
        return {"status": "cancelled"}

    def defer_work_item(self, request: dict[str, Any]) -> dict[str, Any]:
        self.native_calls.append("defer")
        node = self.nodes.get(request["work_item_id"])
        if not self._owns(node, request):
            return {"status": "fenced"}
        node.update(
            status="ready",
            next_retry_at=request["next_retry_at"],
            lease_owner=None,
            lease_expires_at=None,
        )
        return {"status": "deferred"}


class NoNativeEngine(NativeEngine):
    claim_work_item = None  # type: ignore[assignment]
    renew_work_item_lease = None  # type: ignore[assignment]
    commit_work_item_result = None  # type: ignore[assignment]
    cancel_work_item = None  # type: ignore[assignment]
    defer_work_item = None  # type: ignore[assignment]


class HostEngine:
    """Content host whose only WorkItem surface is its control view."""

    def __init__(self, authority: NativeEngine) -> None:
        self._work_item_engine = authority


class BackendOnlyFacade:
    """A backend attribute is not an alternate WorkItem authority."""

    def __init__(self, backend: NativeEngine) -> None:
        self.backend = backend

    def query_cypher(self, query: str, params: dict | None = None) -> list[dict]:
        return self.backend.query_cypher(query, params)

    def add_node(self, *args, **kwargs) -> None:
        self.backend.add_node(*args, **kwargs)


@pytest.fixture
def engine() -> NativeEngine:
    return NativeEngine()


def test_submit_is_idempotent_and_stores_privacy_normalized_metadata(
    engine: NativeEngine,
) -> None:
    item_id = wi.submit_work_item(
        engine,
        kind="ingest_task",
        payload_ref="job-1",
        tenant="tenant-a",
        metadata={"contact": "person@example.com", "target": "workspace:repo"},
        work_item_id="workitem:ingest_task:job-1",
    )
    assert (
        wi.submit_work_item(
            engine,
            kind="ingest_task",
            tenant="tenant-a",
            work_item_id=item_id,
        )
        == item_id
    )
    item = wi.get_work_item(engine, item_id)
    assert item is not None
    assert item["status"] == "ready"
    assert "person@example.com" not in str(item["metadata"])
    assert item["metadata"]["target"] == "workspace:repo"


def test_native_api_is_required_and_no_cas_claim_fallback_exists() -> None:
    engine = NoNativeEngine()
    item_id = wi.submit_work_item(
        engine, kind="generic", payload_ref="p", tenant="tenant-a"
    )
    with pytest.raises(wi.NativeWorkItemRequired):
        wi.claim_specific(engine, item_id, token="worker", now=1.0)


def test_host_engine_routes_all_work_item_operations_to_its_single_control_view() -> (
    None
):
    authority = NativeEngine()
    host = HostEngine(authority)
    item_id = wi.submit_work_item(
        host, kind="generic", payload_ref="payload:opaque", tenant="tenant-a"
    )

    claim = wi.claim_specific(host, item_id, token="worker", now=1.0)

    assert claim is not None
    assert item_id in authority.nodes
    assert authority.native_calls == ["claim"]


def test_native_verbs_are_never_discovered_through_backend_fallbacks() -> None:
    authority = NativeEngine()
    item_id = wi.submit_work_item(authority, kind="generic", tenant="tenant-a")
    facade = BackendOnlyFacade(authority)

    with pytest.raises(wi.NativeWorkItemRequired):
        wi.claim_specific(facade, item_id, token="worker", now=1.0)


def test_native_claim_renew_commit_and_idempotent_redelivery(
    engine: NativeEngine,
) -> None:
    item_id = wi.submit_work_item(
        engine, kind="generic", payload_ref="p", tenant="tenant-a"
    )
    claim = wi.claim_and_start(engine, item_id, token="worker", now=10.0)
    assert claim and claim["_native"] is True
    assert wi.heartbeat(engine, item_id, claim, now=11.0, lease_ttl_s=30.0)
    assert (
        wi.commit_result(
            engine, item_id, claim, outcome="succeeded", result_ref="result:1", now=12.0
        )
        == "committed"
    )
    assert (
        wi.commit_result(
            engine, item_id, claim, outcome="succeeded", result_ref="result:2", now=13.0
        )
        == "noop"
    )
    assert wi.get_work_item(engine, item_id)["result_ref"] == "result:1"
    assert engine.native_calls == ["claim", "renew", "commit", "commit"]


def test_checkpoint_is_fenced_on_the_native_lease(engine: NativeEngine) -> None:
    item_id = wi.submit_work_item(
        engine, kind="goal_loop", payload_ref="loop:opaque", tenant="tenant-a"
    )
    claim = wi.claim_and_start(engine, item_id, token="worker", now=10.0)
    assert claim is not None
    assert wi.checkpoint_work_item(
        engine,
        item_id,
        claim,
        "checkpoint:iteration:1",
        now=11.0,
    )
    assert wi.get_work_item(engine, item_id)["checkpoint_id"] == (
        "checkpoint:iteration:1"
    )

    stale = {**claim, "fencing_token": int(claim["fencing_token"]) + 1}
    assert not wi.checkpoint_work_item(
        engine,
        item_id,
        stale,
        "checkpoint:iteration:2",
        now=12.0,
    )
    assert wi.get_work_item(engine, item_id)["checkpoint_id"] == (
        "checkpoint:iteration:1"
    )


def test_checkpoint_rejects_nonopaque_content(engine: NativeEngine) -> None:
    item_id = wi.submit_work_item(
        engine,
        kind="goal_loop",
        payload_ref="loop:opaque",
        tenant="tenant-a",
    )
    claim = wi.claim_and_start(engine, item_id, token="worker", now=10.0)
    with pytest.raises(ValueError, match="opaque checkpoint reference"):
        wi.checkpoint_work_item(
            engine,
            item_id,
            claim,
            "local path or human content",
            now=11.0,
        )


def test_native_retry_then_dead_letter(engine: NativeEngine) -> None:
    item_id = wi.submit_work_item(
        engine,
        kind="generic",
        payload_ref="p",
        tenant="tenant-a",
        max_attempts=2,
        backoff_base_s=5.0,
    )
    first = wi.claim_and_start(engine, item_id, token="one", now=100.0)
    assert (
        wi.commit_result(
            engine, item_id, first, outcome="failed", error_ref="e1", now=101.0
        )
        == "retry_scheduled"
    )
    assert wi.claim_specific(engine, item_id, token="two", now=102.0) is None
    second = wi.claim_and_start(engine, item_id, token="two", now=106.0)
    assert (
        wi.commit_result(
            engine, item_id, second, outcome="failed", error_ref="e2", now=107.0
        )
        == "dead_letter"
    )
    assert wi.get_work_item(engine, item_id)["status"] == "dead_letter"


def test_native_dependency_release_is_atomic(engine: NativeEngine) -> None:
    parent = wi.submit_work_item(
        engine, kind="generic", payload_ref="parent", tenant="tenant-a"
    )
    child = wi.submit_work_item(
        engine,
        kind="generic",
        payload_ref="child",
        tenant="tenant-a",
        depends_on=[parent],
    )
    assert wi.get_work_item(engine, child)["status"] == "submitted"
    claim = wi.claim_and_start(engine, parent, token="worker", now=10.0)
    assert (
        wi.commit_result(engine, parent, claim, outcome="succeeded", now=11.0)
        == "committed"
    )
    assert wi.get_work_item(engine, child)["status"] == "ready"


def test_native_cancel_and_defer(engine: NativeEngine) -> None:
    deferred = wi.submit_work_item(
        engine, kind="generic", payload_ref="d", tenant="tenant-a"
    )
    claim = wi.claim_and_start(engine, deferred, token="worker", now=10.0)
    assert wi.defer_work_item(engine, deferred, claim, next_retry_at=70.0, now=11.0)
    assert wi.get_work_item(engine, deferred)["next_retry_at"] == 70.0
    cancelled = wi.submit_work_item(
        engine, kind="generic", payload_ref="c", tenant="tenant-a"
    )
    assert wi.cancel_work_item(engine, cancelled, reason="operator")
    assert wi.get_work_item(engine, cancelled)["status"] == "cancelled"


def test_ingest_claim_never_creates_a_missing_work_item(engine: NativeEngine) -> None:
    assert wi.claim_ingest_task_work_item(engine, "missing", token="worker") is None
    item_id = wi.ensure_ingest_task_work_item(
        engine,
        "job-1",
        tenant="tenant-a",
        metadata={"target": "workspace:repo", "type": "codebase"},
    )
    claim = wi.claim_ingest_task_work_item(engine, "job-1", token="worker")
    assert claim and claim["work_item_id"] == item_id
    assert wi.claim_ingest_task_work_item(engine, "job-1", token="other") is None


def test_reaper_has_no_python_transition_writer(engine: NativeEngine) -> None:
    assert wi.reap_expired_leases(engine, now=100.0) == {
        "reaped_ready": [],
        "reaped_dead_letter": [],
    }
