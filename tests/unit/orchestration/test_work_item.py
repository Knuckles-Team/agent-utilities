"""The unified engine-native ``WorkItem`` state machine (AU-P1-1).

Exercises the full lifecycle::

    submitted -> ready -> leased(fencing_token) -> running(heartbeat,attempt)
        -> succeeded(result_ref) | failed(error_ref) | cancelled | dead_letter

against two fake engine doubles:

* :class:`NativeEngine` — the strict, engine-native surface (``claim_work_item``/
  ``renew_work_item_lease``/``commit_work_item_result``/``cancel_work_item``/
  ``defer_work_item``); an engine build lacking these verbs raises
  :class:`~agent_utilities.orchestration.work_item.NativeWorkItemRequired` rather
  than silently falling back.
* :class:`CasEngine` — a dict-backed double with a real atomic
  ``compare_and_set_node_fields`` (mirroring the ``_ClaimHarness``/fake-CAS
  pattern ``tests/unit/knowledge_graph/test_task_claim_cas.py`` uses for the
  ``:Task`` claim), used for the lease/CAS-transition mechanics and the
  Loop/ingestion-Task/team-collaboration bridges layered on top of the core
  ``submit``/``claim``/``commit`` primitives.

No live epistemic-graph engine is required either way.
"""

from __future__ import annotations

import hashlib
import json
import threading
from typing import Any

import pytest

from agent_utilities.orchestration import work_item as wi


class FakeStatechartClient:
    """Generic in-memory reference interpreter for the ``eg-statechart`` wire
    surface (``define``/``instantiate``/``send_event``/``get_state``/``list``)
    used by the W2.5 control-plane migration tests.

    Faithful to ``eg-statechart``'s semantics for the guard vocabulary this
    codebase actually emits: the FIRST transition (in declaration order)
    whose ``from`` matches the instance's current active state, whose
    ``event`` matches, and whose guard (``always``/``event_eq``/``all``/
    ``any``) holds against the event payload fires; no match is a
    well-defined no-op (``fired: False``). Generic over whatever
    ``StatechartDef``-shaped dict it is given — NOT hardcoded to any one
    chart — so a test loading it with the real ``LOOP_STATECHART_DEF``
    genuinely exercises that data through real guard evaluation, not a
    scripted response.
    """

    def __init__(self) -> None:
        self._defs: dict[str, dict[str, Any]] = {}
        self._instances: dict[str, dict[str, Any]] = {}
        self._seq = 0

    def define(self, definition: dict[str, Any]) -> str:
        payload = json.dumps(definition, sort_keys=True, default=str).encode()
        def_id = "eg:statechart:" + hashlib.sha256(payload).hexdigest()[:24]
        self._defs[def_id] = definition
        return def_id

    def instantiate(
        self, def_id: str, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        definition = self._defs[def_id]
        self._seq += 1
        instance_id = f"eg:statechart-instance:{self._seq}"
        self._instances[instance_id] = {
            "def_id": def_id,
            "active": definition["initial"],
            "context": dict(context or {}),
            "version": 0,
        }
        return self._describe(instance_id)

    def send_event(
        self,
        instance_id: str,
        event: str,
        payload: dict[str, Any] | None = None,
        *,
        expected_version: int | None = None,
    ) -> dict[str, Any]:
        inst = self._instances[instance_id]
        payload = dict(payload or {})
        if expected_version is not None and expected_version != inst["version"]:
            return {
                "instance": self._describe(instance_id),
                "fired": False,
                "no_op_reason": "version_conflict",
                "fired_label": None,
                "actions": [],
                "effects": [],
            }
        definition = self._defs[inst["def_id"]]
        fired = False
        fired_label = None
        for t in definition["transitions"]:
            if t["from"] != inst["active"] or t["event"] != event:
                continue
            if self._guard_holds(t.get("guard") or {"op": "always"}, payload, inst["context"]):
                inst["active"] = t["to"]
                inst["version"] += 1
                fired = True
                fired_label = t.get("label")
                break
        return {
            "instance": self._describe(instance_id),
            "fired": fired,
            "no_op_reason": None if fired else "no_matching_transition",
            "fired_label": fired_label,
            "actions": [],
            "effects": [],
        }

    def get_state(self, instance_id: str) -> dict[str, Any]:
        return self._describe(instance_id)

    def list(self, def_id: str | None = None) -> dict[str, Any]:
        ids = [
            iid
            for iid, inst in self._instances.items()
            if def_id is None or inst["def_id"] == def_id
        ]
        return {"instances": [self._describe(iid) for iid in ids]}

    def _describe(self, instance_id: str) -> dict[str, Any]:
        inst = self._instances[instance_id]
        return {
            "instance_id": instance_id,
            "configuration": {"active": [inst["active"]]},
            "version": inst["version"],
        }

    @staticmethod
    def _guard_holds(
        guard: dict[str, Any], payload: dict[str, Any], context: dict[str, Any]
    ) -> bool:
        op = guard.get("op")
        if op == "always":
            return True
        if op == "event_eq":
            return payload.get(guard["key"]) == guard["value"]
        if op == "all":
            return all(
                FakeStatechartClient._guard_holds(g, payload, context)
                for g in guard.get("guards", [])
            )
        if op == "any":
            return any(
                FakeStatechartClient._guard_holds(g, payload, context)
                for g in guard.get("guards", [])
            )
        raise ValueError(f"FakeStatechartClient: unsupported guard op {op!r}")


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


# ---------------------------------------------------------------------------
# NativeEngine: in-memory double for the generated native WorkItem verbs
# ---------------------------------------------------------------------------


class NativeEngine:
    """In-memory double for the generated native WorkItem verbs."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: list[tuple[str, str, str]] = []
        self.native_calls: list[str] = []
        self._lock = threading.Lock()
        # W2.5: generic eg-statechart reference interpreter double, so the
        # Loop lifecycle wiring (research.loops / loop_controller.run_loop)
        # has a real ``.statechart`` sub-client to drive under test.
        self.statechart = FakeStatechartClient()

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


# ---------------------------------------------------------------------------
# CasEngine: dict-backed nodes with a real atomic compare_and_set_node_fields
#
# Covers the lease/CAS-transition mechanics (priority buckets, resource-class
# filtering, stale-lease reclaim/fencing, retry-then-dead-letter, atomic
# dependency release) plus the Loop/ingestion-Task/team-collaboration read
# and bridge helpers layered on top of submit/claim/commit — exercised
# against the same fake-CAS pattern as ``test_task_claim_cas.py``.
# ---------------------------------------------------------------------------


class CasEngine:
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

        raise AssertionError(f"CasEngine: unrecognized query: {q[:160]!r}")


class NoCasEngine(CasEngine):
    """Otherwise-identical to CasEngine, but with no atomic CAS — must fail
    loud when a WorkItem transition needs one, never silently no-op."""

    compare_and_set_node_fields = None  # type: ignore[assignment]


@pytest.fixture
def cas_engine() -> CasEngine:
    return CasEngine()


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------


def test_submit_with_no_deps_is_immediately_ready(cas_engine: CasEngine) -> None:
    item_id = wi.submit_work_item(cas_engine, kind="generic", payload_ref="p1")
    item = wi.get_work_item(cas_engine, item_id)
    assert item is not None
    assert item["status"] == wi.WorkItemStatus.READY.value
    assert item["dep_count"] == 0


def test_submit_with_unmet_dep_is_submitted_not_ready(cas_engine: CasEngine) -> None:
    parent_id = wi.submit_work_item(cas_engine, kind="generic", payload_ref="parent")
    child_id = wi.submit_work_item(
        cas_engine, kind="generic", payload_ref="child", depends_on=[parent_id]
    )

    child = wi.get_work_item(cas_engine, child_id)
    assert child is not None
    assert child["status"] == wi.WorkItemStatus.SUBMITTED.value
    assert child["dep_count"] == 1

    parent = wi.get_work_item(cas_engine, parent_id)
    assert parent is not None
    assert child_id in parent["downstream_ids"]


def test_submit_is_idempotent_upsert_on_explicit_id(cas_engine: CasEngine) -> None:
    fixed_id = "workitem:fixed-1"
    first = wi.submit_work_item(
        cas_engine, kind="generic", payload_ref="a", work_item_id=fixed_id
    )
    second = wi.submit_work_item(
        cas_engine, kind="generic", payload_ref="a-changed", work_item_id=fixed_id
    )
    assert first == second == fixed_id
    item = wi.get_work_item(cas_engine, fixed_id)
    assert item["payload_ref"] == "a"  # second submit was a no-op, not an overwrite


def test_tenant_quota_exceeded_raises(cas_engine: CasEngine) -> None:
    wi.submit_work_item(
        cas_engine,
        kind="generic",
        payload_ref="1",
        tenant="acme",
        max_tenant_in_flight=1,
    )
    with pytest.raises(wi.TenantQuotaExceeded):
        wi.submit_work_item(
            cas_engine,
            kind="generic",
            payload_ref="2",
            tenant="acme",
            max_tenant_in_flight=1,
        )


# ---------------------------------------------------------------------------
# claim / lease / fencing
# ---------------------------------------------------------------------------


def test_claim_specific_transitions_ready_to_leased_with_fencing_token(
    cas_engine: CasEngine,
) -> None:
    item_id = wi.submit_work_item(cas_engine, kind="generic", payload_ref="p")
    claim = wi.claim_specific(cas_engine, item_id, token="host:1", now=1000.0)
    assert claim is not None
    assert claim["fence_token"] == 1
    assert claim["attempt"] == 1

    item = wi.get_work_item(cas_engine, item_id)
    assert item["status"] == wi.WorkItemStatus.LEASED.value
    assert item["lease_owner"] == "host:1"
    assert item["lease_epoch"] == 1


def test_claim_specific_skips_live_lease_held_elsewhere(cas_engine: CasEngine) -> None:
    item_id = wi.submit_work_item(cas_engine, kind="generic", payload_ref="p")
    first = wi.claim_specific(
        cas_engine, item_id, token="host:1", now=1000.0, lease_ttl_s=3600.0
    )
    assert first is not None

    second = wi.claim_specific(cas_engine, item_id, token="host:2", now=1000.0 + 10.0)
    assert second is None  # lease is still fresh


def test_claim_specific_reclaims_after_stale_lease_bumps_fencing(
    cas_engine: CasEngine,
) -> None:
    item_id = wi.submit_work_item(cas_engine, kind="generic", payload_ref="p")
    first = wi.claim_specific(
        cas_engine, item_id, token="host:1", now=1000.0, lease_ttl_s=10.0
    )
    assert first["fence_token"] == 1

    # Lease has now expired (11s later, ttl was 10s) — a new claimer reclaims it.
    second = wi.claim_specific(
        cas_engine, item_id, token="host:2", now=1011.0, lease_ttl_s=3600.0
    )
    assert second is not None
    assert second["fence_token"] == 2  # strictly greater than the stale holder's epoch


def test_mark_running_and_heartbeat_extend_the_lease(cas_engine: CasEngine) -> None:
    item_id = wi.submit_work_item(cas_engine, kind="generic", payload_ref="p")
    claim = wi.claim_specific(
        cas_engine, item_id, token="host:1", now=1000.0, lease_ttl_s=60.0
    )
    assert wi.mark_running(cas_engine, item_id, claim, now=1001.0)
    item = wi.get_work_item(cas_engine, item_id)
    assert item["status"] == wi.WorkItemStatus.RUNNING.value

    assert wi.heartbeat(cas_engine, item_id, claim, now=1030.0, lease_ttl_s=60.0)
    item = wi.get_work_item(cas_engine, item_id)
    assert item["lease_expires_at"] == 1030.0 + 60.0


def test_claim_next_respects_priority_bucket_ordering(cas_engine: CasEngine) -> None:
    low_id = wi.submit_work_item(
        cas_engine, kind="generic", payload_ref="low", priority="background", now=1.0
    )
    high_id = wi.submit_work_item(
        cas_engine, kind="generic", payload_ref="high", priority="critical", now=2.0
    )

    claim = wi.claim_next(cas_engine, now=1000.0)
    assert claim["work_item_id"] == high_id

    claim2 = wi.claim_next(cas_engine, now=1001.0)
    assert claim2["work_item_id"] == low_id


def test_claim_next_filters_by_resource_class(cas_engine: CasEngine) -> None:
    wi.submit_work_item(
        cas_engine, kind="generic", payload_ref="cpu-1", resource_class="cpu"
    )
    gpu_id = wi.submit_work_item(
        cas_engine, kind="generic", payload_ref="gpu-1", resource_class="gpu"
    )

    claim = wi.claim_next(cas_engine, resource_class="gpu", now=1000.0)
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
    cas_engine: CasEngine,
) -> None:
    item_id = wi.submit_work_item(
        cas_engine, kind="generic", payload_ref="p", max_attempts=5
    )
    claim = wi.claim_specific(
        cas_engine, item_id, token="host:1", now=1000.0, lease_ttl_s=10.0
    )
    wi.mark_running(cas_engine, item_id, claim, now=1000.0)

    # Worker "dies" — the lease is now expired at t=1500.
    result = wi.reap_expired_leases(cas_engine, now=1500.0)
    assert result["reaped_ready"] == [item_id]
    assert result["reaped_dead_letter"] == []

    item = wi.get_work_item(cas_engine, item_id)
    assert item["status"] == wi.WorkItemStatus.READY.value
    assert item["lease_epoch"] == 2  # bumped past the dead holder's epoch (1)

    # The dead holder eventually "finishes" and tries to commit with its
    # stale claim — must be rejected, never overwrite the reclaimed item.
    outcome = wi.commit_result(
        cas_engine, item_id, claim, outcome="succeeded", result_ref="ref:1"
    )
    assert outcome == "fenced"
    assert (
        wi.get_work_item(cas_engine, item_id)["status"] == wi.WorkItemStatus.READY.value
    )


def test_reap_expired_lease_exhausted_retries_goes_to_dead_letter(
    cas_engine: CasEngine,
) -> None:
    item_id = wi.submit_work_item(
        cas_engine, kind="generic", payload_ref="p", max_attempts=1
    )
    claim = wi.claim_specific(
        cas_engine, item_id, token="host:1", now=1000.0, lease_ttl_s=10.0
    )
    wi.mark_running(cas_engine, item_id, claim, now=1000.0)
    assert wi.get_work_item(cas_engine, item_id)["attempt"] == 1  # == max_attempts

    result = wi.reap_expired_leases(cas_engine, now=1500.0)
    assert result["reaped_dead_letter"] == [item_id]
    assert result["reaped_ready"] == []
    item = wi.get_work_item(cas_engine, item_id)
    assert item["status"] == wi.WorkItemStatus.DEAD_LETTER.value
    assert "lease_expired" in item["error_ref"]


# ---------------------------------------------------------------------------
# commit_result — idempotent double-commit, retry-then-DLQ, cancellation
# ---------------------------------------------------------------------------


def test_commit_result_success_is_idempotent_noop_on_redelivery(
    cas_engine: CasEngine,
) -> None:
    item_id = wi.submit_work_item(cas_engine, kind="generic", payload_ref="p")
    claim = wi.claim_and_start(cas_engine, item_id, token="host:1", now=1000.0)

    first = wi.commit_result(
        cas_engine, item_id, claim, outcome="succeeded", result_ref="ref:1", now=1010.0
    )
    assert first == "committed"
    item = wi.get_work_item(cas_engine, item_id)
    assert item["status"] == wi.WorkItemStatus.SUCCEEDED.value
    assert item["result_ref"] == "ref:1"

    # Redelivery of the identical turn (at-least-once queue semantics):
    # must be a no-op, never re-running downstream release or overwriting result_ref.
    second = wi.commit_result(
        cas_engine,
        item_id,
        claim,
        outcome="succeeded",
        result_ref="ref:DIFFERENT",
        now=1020.0,
    )
    assert second == "noop"
    item_after = wi.get_work_item(cas_engine, item_id)
    assert item_after["result_ref"] == "ref:1"  # untouched by the redelivered commit


def test_commit_result_retryable_failure_then_exhausts_to_dead_letter(
    cas_engine: CasEngine,
) -> None:
    item_id = wi.submit_work_item(
        cas_engine, kind="generic", payload_ref="p", max_attempts=2, backoff_base_s=5.0
    )

    claim1 = wi.claim_and_start(cas_engine, item_id, token="host:1", now=1000.0)
    assert claim1["attempt"] == 1
    outcome1 = wi.commit_result(
        cas_engine, item_id, claim1, outcome="failed", error_ref="boom-1", now=1001.0
    )
    assert outcome1 == "retry_scheduled"
    item = wi.get_work_item(cas_engine, item_id)
    assert item["status"] == wi.WorkItemStatus.READY.value
    assert item["next_retry_at"] == 1001.0 + 5.0
    assert item["lease_epoch"] == 2  # fenced past the failed attempt

    # Backoff hasn't elapsed yet — not claimable.
    assert wi.claim_specific(cas_engine, item_id, token="host:2", now=1002.0) is None

    # Backoff elapsed — second (and last) attempt.
    claim2 = wi.claim_and_start(cas_engine, item_id, token="host:2", now=1010.0)
    assert claim2 is not None
    assert claim2["attempt"] == 2

    outcome2 = wi.commit_result(
        cas_engine, item_id, claim2, outcome="failed", error_ref="boom-2", now=1011.0
    )
    assert outcome2 == "dead_letter"
    item = wi.get_work_item(cas_engine, item_id)
    assert item["status"] == wi.WorkItemStatus.DEAD_LETTER.value
    assert item["error_ref"] == "boom-2"


def test_commit_result_non_retryable_failure_is_terminal_immediately(
    cas_engine: CasEngine,
) -> None:
    item_id = wi.submit_work_item(
        cas_engine, kind="generic", payload_ref="p", max_attempts=5
    )
    claim = wi.claim_and_start(cas_engine, item_id, token="host:1", now=1000.0)
    outcome = wi.commit_result(
        cas_engine,
        item_id,
        claim,
        outcome="failed",
        error_ref="no executor bound",
        retryable=False,
        now=1001.0,
    )
    assert outcome == "committed"
    item = wi.get_work_item(cas_engine, item_id)
    assert item["status"] == wi.WorkItemStatus.FAILED.value
    assert item["attempt"] == 1  # never retried despite max_attempts=5


def test_cancel_work_item_from_ready_and_is_idempotent(cas_engine: CasEngine) -> None:
    item_id = wi.submit_work_item(cas_engine, kind="generic", payload_ref="p")
    assert wi.cancel_work_item(cas_engine, item_id, reason="user requested") is True
    item = wi.get_work_item(cas_engine, item_id)
    assert item["status"] == wi.WorkItemStatus.CANCELLED.value

    # Idempotent: cancelling an already-cancelled item is a truthy no-op.
    assert wi.cancel_work_item(cas_engine, item_id) is True


def test_cancel_work_item_cannot_override_a_real_terminal_outcome(
    cas_engine: CasEngine,
) -> None:
    item_id = wi.submit_work_item(cas_engine, kind="generic", payload_ref="p")
    claim = wi.claim_and_start(cas_engine, item_id, token="host:1", now=1000.0)
    wi.commit_result(
        cas_engine, item_id, claim, outcome="succeeded", result_ref="ref:1", now=1001.0
    )

    assert wi.cancel_work_item(cas_engine, item_id) is False
    assert (
        wi.get_work_item(cas_engine, item_id)["status"]
        == wi.WorkItemStatus.SUCCEEDED.value
    )


# ---------------------------------------------------------------------------
# atomic dependency release
# ---------------------------------------------------------------------------


def test_child_becomes_ready_exactly_when_all_parents_succeed(
    cas_engine: CasEngine,
) -> None:
    parent1 = wi.submit_work_item(cas_engine, kind="generic", payload_ref="parent1")
    parent2 = wi.submit_work_item(cas_engine, kind="generic", payload_ref="parent2")
    child = wi.submit_work_item(
        cas_engine, kind="generic", payload_ref="child", depends_on=[parent1, parent2]
    )

    assert (
        wi.get_work_item(cas_engine, child)["status"]
        == wi.WorkItemStatus.SUBMITTED.value
    )
    assert wi.get_work_item(cas_engine, child)["dep_count"] == 2

    claim1 = wi.claim_and_start(cas_engine, parent1, token="host:1", now=1000.0)
    wi.commit_result(
        cas_engine, parent1, claim1, outcome="succeeded", result_ref="r1", now=1001.0
    )

    # Only one of two parents done — child must still be blocked.
    child_state = wi.get_work_item(cas_engine, child)
    assert child_state["status"] == wi.WorkItemStatus.SUBMITTED.value
    assert child_state["dep_count"] == 1

    claim2 = wi.claim_and_start(cas_engine, parent2, token="host:2", now=1002.0)
    wi.commit_result(
        cas_engine, parent2, claim2, outcome="succeeded", result_ref="r2", now=1003.0
    )

    # Second (and last) parent done — released atomically, in the same CAS
    # that decremented the counter to zero.
    child_state = wi.get_work_item(cas_engine, child)
    assert child_state["status"] == wi.WorkItemStatus.READY.value
    assert child_state["dep_count"] == 0


def test_downstream_release_is_idempotent_no_double_release(
    cas_engine: CasEngine,
) -> None:
    parent = wi.submit_work_item(cas_engine, kind="generic", payload_ref="parent")
    child = wi.submit_work_item(
        cas_engine, kind="generic", payload_ref="child", depends_on=[parent]
    )

    claim = wi.claim_and_start(cas_engine, parent, token="host:1", now=1000.0)
    wi.commit_result(
        cas_engine, parent, claim, outcome="succeeded", result_ref="r1", now=1001.0
    )
    assert (
        wi.get_work_item(cas_engine, child)["status"] == wi.WorkItemStatus.READY.value
    )

    # Redelivered commit of the same parent (idempotent no-op) must not
    # touch the child a second time.
    wi.commit_result(
        cas_engine, parent, claim, outcome="succeeded", result_ref="r1-again", now=1002.0
    )
    assert wi.get_work_item(cas_engine, child)["dep_count"] == 0
    assert (
        wi.get_work_item(cas_engine, child)["status"] == wi.WorkItemStatus.READY.value
    )


# ---------------------------------------------------------------------------
# Loop / ingestion-Task read shims
# ---------------------------------------------------------------------------


def test_work_item_view_of_loop_maps_statuses(cas_engine: CasEngine) -> None:
    wi.submit_work_item(
        cas_engine,
        kind="goal_loop",
        payload_ref="loop:develop:x",
        work_item_id=wi.loop_work_item_id("loop:develop:x"),
    )
    view = wi.work_item_view_of_loop(cas_engine, "loop:develop:x")
    assert view["status"] == wi.WorkItemStatus.READY.value
    assert view["kind"] == "goal_loop"


def test_work_item_view_of_loop_unknown_returns_none(cas_engine: CasEngine) -> None:
    assert wi.work_item_view_of_loop(cas_engine, "loop:nope") is None


def test_work_item_view_of_task_maps_statuses(cas_engine: CasEngine) -> None:
    cas_engine.add_node("job-1", "Task", properties={"status": "dead_letter"})
    view = wi.work_item_view_of_task(cas_engine, "job-1")
    assert view["status"] == wi.WorkItemStatus.DEAD_LETTER.value
    assert view["shim"] is True


# ---------------------------------------------------------------------------
# AU-P1-CL: ingestion-:Task bridge (engine_tasks.py claim/reap authority)
# ---------------------------------------------------------------------------


def test_ingest_task_work_item_id_round_trips(cas_engine: CasEngine) -> None:
    item_id = wi.ingest_task_work_item_id("job-42")
    assert item_id == "workitem:ingest_task:job-42"
    assert wi.ingest_task_job_id_from_work_item_id(item_id) == "job-42"
    assert wi.ingest_task_job_id_from_work_item_id("workitem:agent_task:x") is None


def test_ensure_ingest_task_work_item_is_idempotent(cas_engine: CasEngine) -> None:
    first = wi.ensure_ingest_task_work_item(
        cas_engine,
        "job-1",
        prio_bucket=1,
        resource_class="ingestion",
        fairness_group="codebase",
    )
    second = wi.ensure_ingest_task_work_item(cas_engine, "job-1", prio_bucket=3)
    assert first == second == wi.ingest_task_work_item_id("job-1")
    item = wi.get_work_item(cas_engine, first)
    assert item["status"] == wi.WorkItemStatus.READY.value
    assert item["prio_bucket"] == 1  # first call's stamp wins (upsert no-op)
    assert item["resource_class"] == "ingestion"
    assert item["fairness_group"] == "codebase"


def test_claim_ingest_task_work_item_wins_then_a_second_claim_loses(
    cas_engine: CasEngine,
) -> None:
    claim1 = wi.claim_ingest_task_work_item(cas_engine, "job-1", token="host-a")
    assert claim1 is not None
    assert claim1["work_item_id"] == wi.ingest_task_work_item_id("job-1")

    claim2 = wi.claim_ingest_task_work_item(cas_engine, "job-1", token="host-b")
    assert claim2 is None  # already leased/running by host-a

    item = wi.get_work_item(cas_engine, claim1["work_item_id"])
    assert item["status"] == wi.WorkItemStatus.RUNNING.value
    assert item["lease_owner"] == "host-a"


# ---------------------------------------------------------------------------
# AU-P1-CL: team-collaboration :TaskNode bridge (teams.py TeamCapability)
# ---------------------------------------------------------------------------


def test_team_task_work_item_id_round_trips(cas_engine: CasEngine) -> None:
    assert wi.team_task_work_item_id("task_abc") == "workitem:team_task:task_abc"


def test_ensure_team_task_work_item_is_ready_no_dependencies(
    cas_engine: CasEngine,
) -> None:
    item_id = wi.ensure_team_task_work_item(cas_engine, "task_1", tenant="team_x")
    item = wi.get_work_item(cas_engine, item_id)
    assert item["status"] == wi.WorkItemStatus.READY.value
    assert item["kind"] == "team_task"
    assert item["tenant"] == "team_x"
    assert item["dep_count"] == 0


def test_start_team_task_work_item_claims_and_runs(cas_engine: CasEngine) -> None:
    claim = wi.start_team_task_work_item(cas_engine, "task_1", tenant="team_x")
    assert claim is not None
    item = wi.get_work_item(cas_engine, claim["work_item_id"])
    assert item["status"] == wi.WorkItemStatus.RUNNING.value

    # A second start (already running) is a no-op (None) — matches
    # TeamCapability.update_task_status's "nothing to transition" handling.
    assert wi.start_team_task_work_item(cas_engine, "task_1", tenant="team_x") is None


def test_team_task_status_view_maps_canonical_vocabulary(cas_engine: CasEngine) -> None:
    assert wi.team_task_status_view(cas_engine, "task_nope") is None

    wi.ensure_team_task_work_item(cas_engine, "task_1")
    assert wi.team_task_status_view(cas_engine, "task_1") == "pending"

    claim = wi.start_team_task_work_item(cas_engine, "task_1")
    assert wi.team_task_status_view(cas_engine, "task_1") == "in_progress"

    wi.commit_result(cas_engine, claim["work_item_id"], claim, outcome="succeeded")
    assert wi.team_task_status_view(cas_engine, "task_1") == "completed"
