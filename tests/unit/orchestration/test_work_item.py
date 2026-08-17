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
            if self._guard_holds(
                t.get("guard") or {"op": "always"}, payload, inst["context"]
            ):
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
        if not self._owns(node, request) or float(
            (node or {}).get("lease_expires_at") or 0
        ) < float(request["now_unix"]):
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
        if not self._owns(node, request) or float(
            (node or {}).get("lease_expires_at") or 0
        ) < float(request["now_unix"]):
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
            return {"status": "not_cancellable"}
        node["status"] = "cancelled"
        return {"status": "cancelled"}

    def defer_work_item(self, request: dict[str, Any]) -> dict[str, Any]:
        self.native_calls.append("defer")
        node = self.nodes.get(request["work_item_id"])
        if not self._owns(node, request) or float(
            (node or {}).get("lease_expires_at") or 0
        ) < float(request["now_unix"]):
            return {"status": "fenced"}
        next_epoch = int(node.get("lease_epoch") or 0) + 1
        node.update(
            status="ready",
            next_retry_at=request["next_retry_at"],
            attempt=max(0, int(node.get("attempt") or 0) - 1),
            defer_count=int(node.get("defer_count") or 0) + 1,
            lease_owner=None,
            lease_expires_at=None,
            lease_epoch=next_epoch,
            fencing_token=next_epoch,
        )
        return {"status": "deferred"}

    def cas_work_item_metadata(self, request: dict[str, Any]) -> dict[str, Any]:
        """In-memory double for the native ``CasWorkItemMetadata`` RPC
        (BUG-111): the same three outcomes (``applied``/``conflict``/
        ``not_found``) the real engine returns, atomically compare-and-set
        against ``self.nodes`` under ``self._lock`` -- never a silent
        overwrite of a losing CAS.
        """
        self.native_calls.append("cas_metadata")
        with self._lock:
            node = self.nodes.get(request["work_item_id"])
            if node is None:
                return {"outcome": "not_found"}
            if node.get("tenant") != request["tenant"]:
                return {"outcome": "conflict"}
            if node.get("status") not in set(request["expected_status"]):
                return {"outcome": "conflict"}
            lease = request.get("expected_lease")
            if lease is not None and not self._owns(
                node,
                {
                    "worker_ref": lease["worker_ref"],
                    "expected_epoch": lease["lease_epoch"],
                    "fencing_token": lease["fencing_token"],
                },
            ):
                return {"outcome": "conflict"}
            if request.get("set_checkpoint_id") is not None:
                if node.get("checkpoint_id") != request.get("expected_checkpoint_id"):
                    return {"outcome": "conflict"}
                node["checkpoint_id"] = request["set_checkpoint_id"]
            elif request.get("set_metadata") is not None:
                if (node.get("metadata") or {}) != (
                    request.get("expected_metadata") or {}
                ):
                    return {"outcome": "conflict"}
                node["metadata"] = request["set_metadata"]
            elif request.get("set_prio_bucket") is not None:
                if int(node.get("prio_bucket") or 0) != int(
                    request.get("expected_prio_bucket") or 0
                ):
                    return {"outcome": "conflict"}
                node["prio_bucket"] = request["set_prio_bucket"]
            else:
                raise AssertionError("cas_work_item_metadata: no set_* field given")
            node["updated_at"] = float(request["now_ms"]) / 1000.0
            return {"outcome": "applied"}


class NoNativeEngine(NativeEngine):
    claim_work_item = None  # type: ignore[assignment]
    renew_work_item_lease = None  # type: ignore[assignment]
    commit_work_item_result = None  # type: ignore[assignment]
    cancel_work_item = None  # type: ignore[assignment]
    defer_work_item = None  # type: ignore[assignment]
    cas_work_item_metadata = None  # type: ignore[assignment]


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


def test_request_and_submit_work_item_input_round_trip(engine: NativeEngine) -> None:
    item_id = wi.submit_work_item(
        engine, kind="generic", payload_ref="p", tenant="tenant-a"
    )
    claim = wi.claim_and_start(engine, item_id, token="worker", now=10.0)
    assert claim is not None

    assert wi.request_work_item_input(
        engine,
        item_id,
        claim,
        request={"prompt": "confirm deletion?"},
        now=11.0,
    )
    item = wi.get_work_item(engine, item_id)
    # Unchanged: input_required is projected from metadata, not a new status.
    assert item["status"] in {"leased", "running"}
    assert item["metadata"]["pending_input_request"] == {"prompt": "confirm deletion?"}
    assert "pending_input_response" not in item["metadata"]

    assert wi.submit_work_item_input(
        engine,
        item_id,
        tenant="tenant-a",
        response={"confirmed": True},
        now=12.0,
    )
    item = wi.get_work_item(engine, item_id)
    assert item["metadata"]["pending_input_response"] == {"confirmed": True}
    assert "pending_input_request" not in item["metadata"]

    # Worker can keep checkpointing/heartbeating normally afterward -- the
    # lease was never touched by either call.
    assert wi.heartbeat(engine, item_id, claim, now=13.0)


def test_request_work_item_input_is_fenced_on_the_native_lease(
    engine: NativeEngine,
) -> None:
    item_id = wi.submit_work_item(
        engine, kind="generic", payload_ref="p", tenant="tenant-a"
    )
    claim = wi.claim_and_start(engine, item_id, token="worker", now=10.0)
    assert claim is not None
    stale = {**claim, "fencing_token": int(claim["fencing_token"]) + 1}
    assert not wi.request_work_item_input(
        engine, item_id, stale, request={"prompt": "x"}, now=11.0
    )
    assert wi.get_work_item(engine, item_id)["metadata"] == {}


def test_submit_work_item_input_requires_a_live_pending_request(
    engine: NativeEngine,
) -> None:
    item_id = wi.submit_work_item(
        engine, kind="generic", payload_ref="p", tenant="tenant-a"
    )
    wi.claim_and_start(engine, item_id, token="worker", now=10.0)
    assert not wi.submit_work_item_input(
        engine, item_id, tenant="tenant-a", response={"confirmed": True}, now=11.0
    )


def test_submit_work_item_input_rejects_wrong_tenant(engine: NativeEngine) -> None:
    item_id = wi.submit_work_item(
        engine, kind="generic", payload_ref="p", tenant="tenant-a"
    )
    claim = wi.claim_and_start(engine, item_id, token="worker", now=10.0)
    assert claim is not None
    assert wi.request_work_item_input(
        engine, item_id, claim, request={"prompt": "x"}, now=11.0
    )
    assert not wi.submit_work_item_input(
        engine, item_id, tenant="tenant-b", response={"confirmed": True}, now=12.0
    )
    assert wi.get_work_item(engine, item_id)["metadata"]["pending_input_request"] == {
        "prompt": "x"
    }


def test_submit_work_item_input_double_submit_only_wins_once(
    engine: NativeEngine,
) -> None:
    item_id = wi.submit_work_item(
        engine, kind="generic", payload_ref="p", tenant="tenant-a"
    )
    claim = wi.claim_and_start(engine, item_id, token="worker", now=10.0)
    assert claim is not None
    assert wi.request_work_item_input(
        engine, item_id, claim, request={"prompt": "x"}, now=11.0
    )
    assert wi.submit_work_item_input(
        engine, item_id, tenant="tenant-a", response={"confirmed": True}, now=12.0
    )
    # A second submission finds no live pending request anymore.
    assert not wi.submit_work_item_input(
        engine, item_id, tenant="tenant-a", response={"confirmed": False}, now=13.0
    )
    assert wi.get_work_item(engine, item_id)["metadata"]["pending_input_response"] == {
        "confirmed": True
    }


def test_set_work_item_priority_applies_via_native_cas(engine: NativeEngine) -> None:
    item_id = wi.submit_work_item(
        engine, kind="generic", payload_ref="p", tenant="tenant-a"
    )
    assert wi.set_work_item_priority(engine, item_id, 3, now=10.0)
    assert wi.get_work_item(engine, item_id)["prio_bucket"] == 3
    assert "cas_metadata" in engine.native_calls


def test_set_work_item_priority_is_terminal_status_closed(
    engine: NativeEngine,
) -> None:
    item_id = wi.submit_work_item(
        engine, kind="generic", payload_ref="p", tenant="tenant-a"
    )
    claim = wi.claim_and_start(engine, item_id, token="worker", now=10.0)
    assert claim is not None
    assert (
        wi.commit_result(
            engine, item_id, claim, outcome="succeeded", result_ref="r", now=11.0
        )
        == "committed"
    )
    # A terminal item is closed to priority changes -- the pre-check short-
    # circuits before the CAS RPC is ever reached.
    assert not wi.set_work_item_priority(engine, item_id, 9, now=12.0)
    assert wi.get_work_item(engine, item_id)["prio_bucket"] != 9


def test_cas_work_item_metadata_deterministic_conflict_never_silently_overwrites(
    engine: NativeEngine,
) -> None:
    """BUG-111 at the Python integration layer: two contenders derive their
    checkpoint CAS from the SAME pre-claim read; the loser gets a distinct
    ``False`` (conflict), never a silent overwrite of the winner's write --
    proven deterministically (two sequential calls against one engine
    double), never by spawning threads and hoping they race (GOC-70)."""
    item_id = wi.submit_work_item(
        engine, kind="goal_loop", payload_ref="loop:opaque", tenant="tenant-a"
    )
    claim = wi.claim_and_start(engine, item_id, token="worker", now=10.0)
    assert claim is not None

    # Contender A wins.
    assert wi.checkpoint_work_item(
        engine, item_id, claim, "checkpoint:1", now=11.0
    )
    assert wi.get_work_item(engine, item_id)["checkpoint_id"] == "checkpoint:1"

    # Contender B holds the SAME claim/lease (still fenced correctly -- this
    # is not a stale-lease rejection) but its CAS condition
    # (``expected_checkpoint_id=None``, baked into the request the moment it
    # was built) is now stale because A already committed. It must be told
    # CONFLICT (surfaced here as a plain ``False``, per this wrapper's
    # documented bool-collapsing contract), not silently overwrite A's write.
    with pytest.MonkeyPatch.context() as mp:
        # Force the read `checkpoint_work_item` performs internally to still
        # observe the PRE-A state, so its request is built exactly like a
        # genuine second contender's would be -- the deterministic
        # equivalent of "both readers observed the same value before either
        # wrote."
        original_get = wi.get_work_item

        def stale_read(engine: Any, item_id_inner: str) -> dict[str, Any] | None:
            item = original_get(engine, item_id_inner)
            if item is not None:
                item = {**item, "checkpoint_id": None}
            return item

        mp.setattr(wi, "get_work_item", stale_read)
        assert not wi.checkpoint_work_item(
            engine, item_id, claim, "checkpoint:2", now=12.0
        )

    # The loser's write never landed.
    assert wi.get_work_item(engine, item_id)["checkpoint_id"] == "checkpoint:1"


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
    assert wi.get_work_item(engine, deferred)["attempt"] == 1
    assert wi.defer_work_item(engine, deferred, claim, next_retry_at=70.0, now=11.0)
    deferred_item = wi.get_work_item(engine, deferred)
    assert deferred_item["next_retry_at"] == 70.0
    assert deferred_item["attempt"] == 0
    cancelled = wi.submit_work_item(
        engine, kind="generic", payload_ref="c", tenant="tenant-a"
    )
    assert wi.cancel_work_item(engine, cancelled, reason="operator")
    assert wi.get_work_item(engine, cancelled)["status"] == "cancelled"


def test_native_defer_rejects_expired_lease_without_changing_attempt(
    engine: NativeEngine,
) -> None:
    item_id = wi.submit_work_item(
        engine, kind="generic", payload_ref="d", tenant="tenant-a"
    )
    claim = wi.claim_and_start(
        engine,
        item_id,
        token="worker",
        now=10.0,
        lease_ttl_s=1.0,
    )

    assert not wi.defer_work_item(
        engine,
        item_id,
        claim,
        next_retry_at=80.0,
        now=12.0,
    )
    item = wi.get_work_item(engine, item_id)
    assert item["status"] == "leased"
    assert item["attempt"] == 1


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

    # -- engine-native WorkItem verbs (AU-P1-1) --------------------------
    #
    # ``work_item.py``'s claim/renew/commit/cancel/defer primitives now
    # dispatch exclusively through these five generated verbs (mirroring
    # ``graph_compute.py``'s real ``self._client.work_items.*`` bridge) —
    # there is no Python-side CAS-scan fallback left in production. This
    # double implements them directly over its own CAS-backed node store so
    # it keeps exercising the same lease/fencing/idempotency mechanics the
    # "cas_engine" fixture name promises, just through the current
    # engine-native surface instead of the retired raw-CAS claim path.

    @staticmethod
    def _owns(node: dict[str, Any] | None, request: dict[str, Any]) -> bool:
        return bool(
            node
            and node.get("lease_owner") == request.get("worker_ref")
            and node.get("lease_epoch") == request.get("expected_epoch")
            and node.get("fencing_token") == request.get("fencing_token")
        )

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
        node = self.nodes.get(request["work_item_id"])
        if not self._owns(node, request):
            return {"renewed": False}
        node["lease_expires_at"] = float(request["now_unix"]) + float(
            request["lease_ttl"]
        )
        return {"renewed": True}

    def commit_work_item_result(self, request: dict[str, Any]) -> dict[str, Any]:
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
        node = self.nodes.get(request["work_item_id"])
        if node is None:
            return {"status": "missing"}
        if node.get("status") == "cancelled":
            return {"status": "cancelled"}
        if node.get("status") in wi.TERMINAL_WORK_ITEM_STATUSES:
            return {"status": "not_cancellable"}
        node["status"] = "cancelled"
        return {"status": "cancelled"}

    def defer_work_item(self, request: dict[str, Any]) -> dict[str, Any]:
        node = self.nodes.get(request["work_item_id"])
        if not self._owns(node, request) or float(
            (node or {}).get("lease_expires_at") or 0
        ) < float(request["now_unix"]):
            return {"status": "fenced"}
        next_epoch = int(node.get("lease_epoch") or 0) + 1
        node.update(
            status="ready",
            next_retry_at=request["next_retry_at"],
            attempt=max(0, int(node.get("attempt") or 0) - 1),
            defer_count=int(node.get("defer_count") or 0) + 1,
            lease_owner=None,
            lease_expires_at=None,
            lease_epoch=next_epoch,
            fencing_token=next_epoch,
        )
        return {"status": "deferred"}

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
    """Otherwise-identical to CasEngine, but with no atomic backend at all —
    must fail loud when a WorkItem transition needs one, never silently
    no-op. Nils out both the raw CAS primitive and the engine-native verbs
    built on top of it, mirroring ``NativeEngine``'s ``NoNativeEngine``."""

    compare_and_set_node_fields = None  # type: ignore[assignment]
    claim_work_item = None  # type: ignore[assignment]
    renew_work_item_lease = None  # type: ignore[assignment]
    commit_work_item_result = None  # type: ignore[assignment]
    cancel_work_item = None  # type: ignore[assignment]
    defer_work_item = None  # type: ignore[assignment]


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
    assert item["status"] == "ready"
    assert item["dep_count"] == 0


def test_submit_with_unmet_dep_is_submitted_not_ready(cas_engine: CasEngine) -> None:
    parent_id = wi.submit_work_item(cas_engine, kind="generic", payload_ref="parent")
    child_id = wi.submit_work_item(
        cas_engine, kind="generic", payload_ref="child", depends_on=[parent_id]
    )

    child = wi.get_work_item(cas_engine, child_id)
    assert child is not None
    assert child["status"] == "submitted"
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
    assert item["status"] == "leased"
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
    # ``mark_running`` only VALIDATES that the claim came from the native
    # ClaimWorkItem transaction (AU-P1-1) — it never writes a separate
    # "running" status; the engine-native lease already IS the running
    # ownership decision (see ``work_item.mark_running``'s docstring).
    assert item["status"] == "leased"

    assert wi.heartbeat(cas_engine, item_id, claim, now=1030.0, lease_ttl_s=60.0)
    item = wi.get_work_item(cas_engine, item_id)
    assert item["lease_expires_at"] == 1030.0 + 60.0


def test_claim_next_respects_priority_bucket_ordering(cas_engine: CasEngine) -> None:
    # ``prio_bucket`` is a plain integer 0..3 (0 = highest priority, claimed
    # first); ``_coerce_prio_bucket`` no longer accepts string labels.
    low_id = wi.submit_work_item(
        cas_engine, kind="generic", payload_ref="low", priority=3, now=1.0
    )
    high_id = wi.submit_work_item(
        cas_engine, kind="generic", payload_ref="high", priority=0, now=2.0
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


def test_claim_next_without_resource_class_searches_all_lanes(
    cas_engine: CasEngine,
) -> None:
    maintenance_id = wi.submit_work_item(
        cas_engine,
        kind="ingest_task",
        payload_ref="maintenance-1",
        resource_class="maintenance",
    )

    claim = wi.claim_next(cas_engine, now=1000.0)

    assert claim["work_item_id"] == maintenance_id


# ---------------------------------------------------------------------------
# consent + expiry gate (D-25-3, CONCEPT:AU-ORCH.dispatch.workitem-consent-gate)
# ---------------------------------------------------------------------------


def test_ordinary_work_item_is_unaffected_by_the_consent_gate(
    cas_engine: CasEngine,
) -> None:
    """The D-25-3 migration decision: consent_required defaults False, so an
    ordinary/legacy WorkItem (no consent fields at all) claims exactly as before."""
    item_id = wi.submit_work_item(cas_engine, kind="generic", payload_ref="p")
    claim = wi.claim_specific(cas_engine, item_id, token="host:1", now=1000.0)
    assert claim is not None


def test_claim_specific_denies_when_consent_is_absent(cas_engine: CasEngine) -> None:
    """consent_required=True with no consent_granted_at is the ABSENT state —
    denied, and (unlike a lapsed grant) never had a lease to release."""
    item_id = wi.submit_work_item(
        cas_engine,
        kind="generic",
        payload_ref="p",
        consent_required=True,
    )
    item = wi.get_work_item(cas_engine, item_id)
    assert item["status"] == "ready"  # not yet claimed

    claim = wi.claim_specific(cas_engine, item_id, token="host:1", now=1000.0)

    assert claim is None
    item = wi.get_work_item(cas_engine, item_id)
    assert item["status"] == "ready"  # denied before the engine was ever touched
    assert item["lease_owner"] is None


def test_claim_specific_denies_when_consent_has_lapsed(cas_engine: CasEngine) -> None:
    """A consent that WAS granted and has since expired is LAPSED — a state
    distinct from absent, but the claim path denies both."""
    item_id = wi.submit_work_item(
        cas_engine,
        kind="generic",
        payload_ref="p",
        consent_required=True,
        consent_scope="data_processing:analytics",
        consent_subject="subject:opaque-1",
        consent_basis="explicit",
        consent_granted_at=100.0,
        consent_expires_at=500.0,
    )

    claim = wi.claim_specific(cas_engine, item_id, token="host:1", now=1000.0)

    assert claim is None


def test_claim_specific_allows_active_consent(cas_engine: CasEngine) -> None:
    """A live, unexpired grant claims normally — the gate isn't overzealous."""
    item_id = wi.submit_work_item(
        cas_engine,
        kind="generic",
        payload_ref="p",
        consent_required=True,
        consent_granted_at=100.0,
        consent_expires_at=5000.0,
    )

    claim = wi.claim_specific(cas_engine, item_id, token="host:1", now=1000.0)

    assert claim is not None
    item = wi.get_work_item(cas_engine, item_id)
    assert item["status"] == "leased"


def test_claim_specific_denies_on_malformed_consent_record(
    cas_engine: CasEngine,
) -> None:
    """Fail CLOSED: an unreadable consent_granted_at must deny, never default
    to allow (constraint 4)."""
    item_id = wi.submit_work_item(
        cas_engine,
        kind="generic",
        payload_ref="p",
        consent_required=True,
        consent_granted_at=100.0,
    )
    # Corrupt the persisted grant timestamp directly on the fake node store —
    # simulating a malformed/unreadable record already in the graph.
    cas_engine.nodes[item_id]["consent_granted_at"] = "not-a-timestamp"

    claim = wi.claim_specific(cas_engine, item_id, token="host:1", now=1000.0)

    assert claim is None


def test_claim_next_releases_a_consent_denied_item_instead_of_returning_it(
    cas_engine: CasEngine,
) -> None:
    """claim_next selects blind; a consent-denied item it picks up must be
    released (deferred, not handed to the caller) and the caller sees None."""
    item_id = wi.submit_work_item(
        cas_engine,
        kind="generic",
        payload_ref="p",
        consent_required=True,
        consent_granted_at=100.0,
        consent_expires_at=500.0,  # already lapsed by now=1000.0
    )

    claim = wi.claim_next(cas_engine, now=1000.0)

    assert claim is None
    item = wi.get_work_item(cas_engine, item_id)
    # Deferred back to "ready" (a bounded cooldown, not a terminal tombstone —
    # consent could still be restored by an operator) rather than left
    # dangling under a lease the caller never received.
    assert item["status"] == "ready"
    assert item["next_retry_at"] == 1000.0 + wi.CONSENT_RECHECK_BACKOFF_S
    assert item["lease_owner"] is None


def test_heartbeat_denies_once_a_running_item_s_consent_lapses(
    cas_engine: CasEngine,
) -> None:
    """Consent lapsing mid-flight must stop renewal at the next heartbeat
    (bounded-time enforcement), not just block new claims."""
    item_id = wi.submit_work_item(
        cas_engine,
        kind="generic",
        payload_ref="p",
        consent_required=True,
        consent_granted_at=100.0,
        consent_expires_at=1500.0,
    )
    claim = wi.claim_specific(cas_engine, item_id, token="host:1", now=1000.0)
    assert claim is not None

    # Still within the grant's window: renewal succeeds normally.
    assert wi.heartbeat(cas_engine, item_id, claim, now=1010.0, lease_ttl_s=60.0)

    # Past consent_expires_at: renewal is now denied even though the lease
    # itself is still fresh.
    assert (
        wi.heartbeat(cas_engine, item_id, claim, now=1600.0, lease_ttl_s=60.0) is False
    )


def test_consent_absent_and_lapsed_are_distinct_states() -> None:
    """The model must not collapse 'never consented' into 'consent expired' —
    they are different facts for an auditor and are reported differently."""
    from agent_utilities.models.knowledge_graph import WorkItemNode

    now = 1000.0
    never_consented = WorkItemNode(
        id="w1", name="n", tenant="t", kind="generic", consent_required=True
    )
    lapsed = WorkItemNode(
        id="w2",
        name="n",
        tenant="t",
        kind="generic",
        consent_required=True,
        consent_granted_at=100.0,
        consent_expires_at=500.0,
    )
    not_required = WorkItemNode(id="w3", name="n", tenant="t", kind="generic")
    active = WorkItemNode(
        id="w4",
        name="n",
        tenant="t",
        kind="generic",
        consent_required=True,
        consent_granted_at=100.0,
        consent_expires_at=5000.0,
    )

    assert never_consented.consent_state(now=now) == "absent"
    assert lapsed.consent_state(now=now) == "lapsed"
    assert not_required.consent_state(now=now) == "not_required"
    assert active.consent_state(now=now) == "active"
    assert len({"absent", "lapsed", "not_required", "active"}) == 4


def test_cas_backend_unavailable_fails_loud_not_silent() -> None:
    no_cas = NoCasEngine()
    no_cas.add_node("workitem:x", "WorkItem", properties={"status": "ready"})
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

    # Worker "dies" — the lease is now expired at t=1500. There is no
    # Python-side reaper transition writer (AU-P1-1): ClaimWorkItem reclaims
    # expired leases atomically as part of selection, so the confirmation
    # call itself is always a no-op (matches
    # ``test_reaper_has_no_python_transition_writer``).
    result = wi.reap_expired_leases(cas_engine, now=1500.0)
    assert result == {"reaped_ready": [], "reaped_dead_letter": []}

    # The actual reclaim happens on the next claim attempt.
    reclaimed = wi.claim_specific(cas_engine, item_id, token="host:2", now=1500.0)
    assert reclaimed is not None
    assert reclaimed["lease_epoch"] == 2  # bumped past the dead holder's epoch (1)

    item = wi.get_work_item(cas_engine, item_id)
    assert item["status"] == "leased"
    assert item["lease_owner"] == "host:2"

    # The dead holder eventually "finishes" and tries to commit with its
    # stale claim — must be rejected, never overwrite the reclaimed item.
    outcome = wi.commit_result(
        cas_engine, item_id, claim, outcome="succeeded", result_ref="ref:1"
    )
    assert outcome == "fenced"
    assert wi.get_work_item(cas_engine, item_id)["status"] == "leased"


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

    # No Python-side reaper transition writer (AU-P1-1) — confirmation only.
    result = wi.reap_expired_leases(cas_engine, now=1500.0)
    assert result == {"reaped_ready": [], "reaped_dead_letter": []}

    # Exhausted-retry dead-lettering happens inside the next native claim
    # attempt's selection, same as a live engine's ClaimWorkItem.
    reclaim = wi.claim_specific(cas_engine, item_id, token="host:2", now=1500.0)
    assert reclaim is None  # attempts exhausted -> dead_letter, not reclaimable

    item = wi.get_work_item(cas_engine, item_id)
    assert item["status"] == "dead_letter"
    assert item["error_ref"] == "lease_exhausted"


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
    assert item["status"] == "succeeded"
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
    assert item["status"] == "ready"
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
    assert item["status"] == "dead_letter"
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
    assert item["status"] == "failed"
    assert item["attempt"] == 1  # never retried despite max_attempts=5


def test_cancel_work_item_from_ready_and_is_idempotent(cas_engine: CasEngine) -> None:
    item_id = wi.submit_work_item(cas_engine, kind="generic", payload_ref="p")
    assert wi.cancel_work_item(cas_engine, item_id, reason="user requested") is True
    item = wi.get_work_item(cas_engine, item_id)
    assert item["status"] == "cancelled"

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
    assert wi.get_work_item(cas_engine, item_id)["status"] == "succeeded"


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

    assert wi.get_work_item(cas_engine, child)["status"] == "submitted"
    assert wi.get_work_item(cas_engine, child)["dep_count"] == 2

    claim1 = wi.claim_and_start(cas_engine, parent1, token="host:1", now=1000.0)
    wi.commit_result(
        cas_engine, parent1, claim1, outcome="succeeded", result_ref="r1", now=1001.0
    )

    # Only one of two parents done — child must still be blocked.
    child_state = wi.get_work_item(cas_engine, child)
    assert child_state["status"] == "submitted"
    assert child_state["dep_count"] == 1

    claim2 = wi.claim_and_start(cas_engine, parent2, token="host:2", now=1002.0)
    wi.commit_result(
        cas_engine, parent2, claim2, outcome="succeeded", result_ref="r2", now=1003.0
    )

    # Second (and last) parent done — released atomically, in the same CAS
    # that decremented the counter to zero.
    child_state = wi.get_work_item(cas_engine, child)
    assert child_state["status"] == "ready"
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
    assert wi.get_work_item(cas_engine, child)["status"] == "ready"

    # Redelivered commit of the same parent (idempotent no-op) must not
    # touch the child a second time.
    wi.commit_result(
        cas_engine,
        parent,
        claim,
        outcome="succeeded",
        result_ref="r1-again",
        now=1002.0,
    )
    assert wi.get_work_item(cas_engine, child)["dep_count"] == 0
    assert wi.get_work_item(cas_engine, child)["status"] == "ready"


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
    assert view["status"] == "ready"
    assert view["kind"] == "goal_loop"


def test_work_item_view_of_loop_unknown_returns_none(cas_engine: CasEngine) -> None:
    assert wi.work_item_view_of_loop(cas_engine, "loop:nope") is None


def test_work_item_view_of_task_maps_statuses(cas_engine: CasEngine) -> None:
    cas_engine.add_node("job-1", "Task", properties={"status": "dead_letter"})
    view = wi.work_item_view_of_task(cas_engine, "job-1")
    assert view["status"] == "dead_letter"
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
    assert item["status"] == "ready"
    assert item["prio_bucket"] == 1  # first call's stamp wins (upsert no-op)
    assert item["resource_class"] == "ingestion"
    assert item["fairness_group"] == "codebase"


def test_claim_ingest_task_work_item_wins_then_a_second_claim_loses(
    cas_engine: CasEngine,
) -> None:
    # claim_ingest_task_work_item never creates/adopts a missing WorkItem
    # (see ``test_ingest_claim_never_creates_a_missing_work_item``) — the
    # ingestion queue must have already indexed it first.
    wi.ensure_ingest_task_work_item(cas_engine, "job-1")

    claim1 = wi.claim_ingest_task_work_item(cas_engine, "job-1", token="host-a")
    assert claim1 is not None
    assert claim1["work_item_id"] == wi.ingest_task_work_item_id("job-1")

    claim2 = wi.claim_ingest_task_work_item(cas_engine, "job-1", token="host-b")
    assert claim2 is None  # already leased/running by host-a

    item = wi.get_work_item(cas_engine, claim1["work_item_id"])
    assert item["status"] == "leased"
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
    assert item["status"] == "ready"
    assert item["kind"] == "team_task"
    assert item["tenant"] == "team_x"
    assert item["dep_count"] == 0


def test_start_team_task_work_item_claims_and_runs(cas_engine: CasEngine) -> None:
    claim = wi.start_team_task_work_item(cas_engine, "task_1", tenant="team_x")
    assert claim is not None
    item = wi.get_work_item(cas_engine, claim["work_item_id"])
    assert item["status"] == "leased"

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


# ---------------------------------------------------------------------------
# U-24 — immutable WorkItem admission and terminal-result authority
# ---------------------------------------------------------------------------
#
# Two confirmed defects on this tree (verified against the audit's design
# intent, not applied as a literal patch -- the upstream diff targets a
# different result-metadata shape this tree never had):
#
# 1. ``_normalize_native_claim`` hardcoded ``"tenant": None`` -- the native
#    ClaimWorkItem response deliberately omits tenant (it was verified at
#    admission), but the claim dict callers pass to heartbeat/checkpoint/
#    commit_result never carried it forward from anywhere. Two existing call
#    sites already papered over this with an ``claim.get("tenant") or
#    item.get("tenant")`` fallback, but ``_normalize_native_claim`` itself --
#    the single place that SHOULD have produced the right value -- always
#    produced ``None``, which is fragile: any future direct reader of
#    ``claim["tenant"]`` (with no fallback) would silently get an untenanted
#    claim.
# 2. ``ensure_ingest_task_work_item`` returned a job id straight from
#    ``submit_work_item`` with no readback proving the WorkItem it just
#    admitted is durably observable through the same control authority.
#
# The "hostile metadata.result key collision" half of the audit's design
# intent does NOT apply here: this tree has no function that merges
# worker/caller-controlled data into TOP-LEVEL WorkItem metadata at all --
# ``request_work_item_input``/``submit_work_item_input`` confine caller data
# to the single namespaced ``metadata["pending_input_request"]``/
# ``["pending_input_response"]`` keys, and ``checkpoint_work_item``/
# ``set_work_item_priority`` write dedicated typed fields
# (``checkpoint_id``/``prio_bucket``), never a generic metadata merge. That
# vulnerability class is refuted below with a hostile-key test proving the
# structural containment holds, rather than "fixed" by adding a namespace
# nothing writes outside of.


def test_claim_specific_preserves_tenant_for_later_fenced_calls(
    cas_engine: CasEngine,
) -> None:
    """KNOWN-BAD (pre-fix): ``_normalize_native_claim`` always returned
    ``tenant=None`` regardless of the item's real, already-verified tenant.
    A later fenced call (heartbeat/checkpoint/commit_result) reading
    ``claim["tenant"]`` directly -- with no ad-hoc fallback -- would lose
    tenant authority. This proves the claim itself now carries it."""
    item_id = wi.submit_work_item(
        cas_engine, kind="generic", payload_ref="p", tenant="acme-corp"
    )
    claim = wi.claim_specific(cas_engine, item_id, token="host:1", now=1000.0)
    assert claim is not None
    assert claim["tenant"] == "acme-corp"


def test_claim_next_preserves_tenant_for_later_fenced_calls(
    cas_engine: CasEngine,
) -> None:
    """Same invariant as above for the BLIND claim path (claim_next), which
    cannot know the tenant before the engine selects an item -- the fix
    backfills it from the same readback the consent gate already performs,
    at no extra round trip."""
    item_id = wi.submit_work_item(
        cas_engine, kind="generic", payload_ref="p", tenant="beta-inc", now=1.0
    )
    claim = wi.claim_next(cas_engine, now=1000.0)
    assert claim is not None
    assert claim["work_item_id"] == item_id
    assert claim["tenant"] == "beta-inc"


def test_reclaim_after_stale_lease_also_preserves_tenant(
    cas_engine: CasEngine,
) -> None:
    """The reclaim-under-a-different-worker path (existing fencing coverage:
    test_claim_specific_reclaims_after_stale_lease_bumps_fencing) must ALSO
    carry the correct tenant through to the new claimer, not just a fresh
    fencing token."""
    item_id = wi.submit_work_item(
        cas_engine, kind="generic", payload_ref="p", tenant="gamma-llc"
    )
    first = wi.claim_specific(
        cas_engine, item_id, token="host:1", now=1000.0, lease_ttl_s=10.0
    )
    assert first["tenant"] == "gamma-llc"

    second = wi.claim_specific(
        cas_engine, item_id, token="host:2", now=1011.0, lease_ttl_s=3600.0
    )
    assert second is not None
    assert second["fence_token"] == 2
    assert second["tenant"] == "gamma-llc"


def test_hostile_request_input_keys_never_escape_the_namespaced_result(
    engine: NativeEngine,
) -> None:
    """Refutes the "hostile metadata.result key collision" defect class
    against this tree's ACTUAL surface: the only worker-facing metadata
    writer (request_work_item_input) confines caller data to
    metadata["pending_input_request"]. A hostile dict naming reserved
    admission fields (tenant/kind/payload_ref/physical_graph/status) must
    stay trapped inside that one namespaced key -- never overwrite the
    WorkItem's own admission-authority fields, and never leak as sibling
    top-level metadata keys."""
    item_id = wi.submit_work_item(
        engine, kind="generic", payload_ref="original-payload", tenant="tenant-a"
    )
    claim = wi.claim_and_start(engine, item_id, token="worker", now=10.0)
    assert claim is not None

    hostile = {
        "tenant": "attacker-tenant",
        "kind": "hijacked_kind",
        "payload_ref": "attacker-payload",
        "physical_graph": "attacker-graph",
        "status": "succeeded",
    }
    assert wi.request_work_item_input(
        engine, item_id, claim, request=hostile, now=11.0
    )
    item = wi.get_work_item(engine, item_id)
    assert item["tenant"] == "tenant-a"
    assert item["kind"] == "generic"
    assert item["payload_ref"] == "original-payload"
    assert item["status"] in {"leased", "running"}
    assert item["metadata"]["pending_input_request"] == hostile
    assert "physical_graph" not in item["metadata"]
    assert "tenant" not in item["metadata"]
    assert "kind" not in item["metadata"]


def test_ensure_ingest_task_work_item_readback_confirms_admission(
    cas_engine: CasEngine,
) -> None:
    """PASS case: the durable admission readback confirms kind/payload_ref
    and does not reject a normal, correctly-admitted ingest task."""
    item_id = wi.ensure_ingest_task_work_item(
        cas_engine, "job-readback-ok", tenant="tenant-x"
    )
    assert item_id == wi.ingest_task_work_item_id("job-readback-ok")
    item = wi.get_work_item(cas_engine, item_id)
    assert item["kind"] == "ingest_task"
    assert item["payload_ref"] == "job-readback-ok"
    assert item["tenant"] == "tenant-x"


class SilentAdmissionEngine(CasEngine):
    """``add_node`` reports success (no exception) but never actually
    persists the row -- simulates an admission write that crashed/was lost
    between commit and durable observability, so an immediate readback
    finds nothing."""

    def add_node(self, node_id, node_type, properties=None, ephemeral=False):
        return {}  # accepted, nothing stored


def test_ensure_ingest_task_work_item_fails_when_admission_is_unobservable() -> None:
    """KNOWN-BAD: submit_work_item did not raise, but the WorkItem it claims
    to have admitted is not durably observable through the same control
    authority. Must raise rather than hand back a job id for a WorkItem that
    does not durably exist -- an "accepted but unobservable" job."""
    engine = SilentAdmissionEngine()
    with pytest.raises(wi.WorkItemBackendUnavailable):
        wi.ensure_ingest_task_work_item(engine, "job-silent")


def test_ensure_ingest_task_work_item_rejects_a_mismatched_tenant_on_reuse(
    cas_engine: CasEngine,
) -> None:
    """KNOWN-BAD: idempotent reuse of the deterministic ingest-task WorkItem
    id must still readback-verify the ADMITTED tenant matches what THIS call
    asked for -- a caller must not silently believe it admitted a job under
    its own tenant when the durable row actually belongs to another."""
    wi.ensure_ingest_task_work_item(cas_engine, "job-tenant-mismatch", tenant="real-owner")
    with pytest.raises(wi.WorkItemBackendUnavailable):
        wi.ensure_ingest_task_work_item(
            cas_engine, "job-tenant-mismatch", tenant="different-tenant"
        )


class ResultCommitCrashEngine(CasEngine):
    """``commit_work_item_result`` raises BEFORE writing anything to the
    node -- simulates a worker/transport crash during the single native RPC
    that atomically persists ``result_ref``/``error_ref`` together with the
    terminal status transition.

    This tree has no Python-side two-phase "write result metadata, then
    commit terminal status" -- ``commit_result`` makes exactly one
    ``commit_work_item_result`` native call carrying both the result
    reference and the outcome, so "crash between result persistence and
    terminal commit" cannot produce a partially-committed WorkItem: either
    that one call lands (both together) or it doesn't (neither). This proves
    the "doesn't" side: a raise leaves status/result_ref exactly as they
    were pre-call, never a result recorded against a non-terminal item or a
    terminal item with no result.
    """

    def commit_work_item_result(self, request: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("transport dropped mid-call")


def test_crash_during_commit_leaves_no_partial_result_or_terminal_state(
) -> None:
    """KNOWN-BAD class: crash-between-result-and-terminal-commit. On this
    tree the two are not separable RPCs, so the only reachable failure mode
    is "the single atomic call never landed" -- proven here by asserting the
    WorkItem is untouched (still running, no result_ref, no terminal
    status) after the native call raises. A caller retrying commit_result
    afterward reuses the same idempotency key, so no duplicate/partial
    commit is possible on retry either."""
    engine = ResultCommitCrashEngine()
    item_id = wi.submit_work_item(engine, kind="generic", payload_ref="p")
    claim = wi.claim_and_start(engine, item_id, token="host:1", now=1000.0)
    assert claim is not None

    before = wi.get_work_item(engine, item_id)
    assert before["status"] in {"leased", "running"}
    assert before.get("result_ref") is None

    with pytest.raises(RuntimeError):
        wi.commit_result(
            engine, item_id, claim, outcome="succeeded", result_ref="ref:1", now=1010.0
        )

    after = wi.get_work_item(engine, item_id)
    assert after["status"] == before["status"]
    assert after.get("result_ref") is None
    assert after.get("status") not in wi.TERMINAL_WORK_ITEM_STATUSES
