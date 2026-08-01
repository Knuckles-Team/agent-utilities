#!/usr/bin/python
"""In-memory mock engine for the SCALE-P2-1 load generator + soak/chaos harness.

Same shape as the per-file ``FakeEngine`` doubles already used across the unit suite
(``tests/unit/orchestration/test_work_item.py``, ``tests/unit/knowledge_graph/
test_task_claim_cas.py``): a dict-backed node store with a REAL atomic
``compare_and_set_node_fields``, so :mod:`agent_utilities.orchestration.work_item`'s
CAS-based claim/lease/fencing/commit runs against genuine optimistic-concurrency
semantics, not a stub. This module generalizes that pattern (recognizing the closed
set of Cypher shapes ``work_item.py`` AND ``messaging.bus.AgentBus``'s graph-fallback
path issue) so the load generator can drive BOTH WorkItem turns and AgentBus messages
against ONE mockable engine, with no live epistemic-graph engine required — the CI-safe
path this harness needs (a real, running fleet is the OTHER supported mode; see
``scripts/scale/loadgen.py``).

Latency is injected deliberately (:class:`LatencyModel`) rather than left at
whatever a Python dict access costs, so the load generator's write/query-latency
percentiles measure something meaningful in mock mode instead of ~0ms every time —
calibrated near the measured ``AddNode`` anchor (``docs/scaling/capacity_model.py``
``MEASURED_ADDNODE_P50_MS``), with a configurable multiplier so chaos scenarios can
inflate it (simulated broker backpressure, degraded shard, etc.).
"""

from __future__ import annotations

import asyncio
import random
import threading
import time
from dataclasses import dataclass
from typing import Any

from agent_utilities.orchestration import work_item as _wi


class WallClock:
    """Real wall-clock timing — used for ``--engine live`` (a real deployment).

    Mock-engine runs do NOT use a "logical clock with async sleeps" — an earlier
    version of this harness tried that and found it fundamentally broken for
    concurrent asyncio tasks: N tasks each independently advancing ONE shared
    mutable clock on every ``sleep()`` call compounds (the clock races ahead
    roughly N-times faster than intended, since every concurrently-sleeping task
    contributes its own advance each round), which corrupts submit/claim/commit
    ORDERING and inflates queue-latency measurements with a simulation artifact,
    not a real system property. The fix (see
    :func:`scripts.scale.loadgen._run_mock_workload`) is a proper single-threaded
    discrete-event simulation (a time-ordered heap, one event processed at a
    time) — inherently free of that race because nothing runs concurrently with
    anything else. ``WallClock`` remains for ``--engine live``, where genuine OS
    wall-clock time correctly synchronizes truly-concurrent asyncio tasks (real
    time does not have the shared-mutable-state race a synthetic clock does).
    """

    def now(self) -> float:
        return time.monotonic()

    async def sleep(self, dt: float) -> None:
        if dt > 0:
            await asyncio.sleep(dt)


@dataclass
class LatencyModel:
    """Synthetic per-operation latency, seconds. Calibrated near the measured anchor.

    ``write_mean_s``/``query_mean_s`` default near ``MEASURED_ADDNODE_P50_MS`` (0.187ms)
    with a small log-normal-ish jitter (via ``random.gauss`` clamped >= 0) so repeated
    calls produce a realistic percentile spread rather than a single fixed number.
    """

    write_mean_s: float = 0.0002
    write_jitter_s: float = 0.0001
    query_mean_s: float = 0.0004
    query_jitter_s: float = 0.0002
    #: Multiplier applied to both means — chaos scenarios crank this up to simulate
    #: a degraded/backpressured shard without needing a second latency model.
    degradation_multiplier: float = 1.0

    def write_delay(self) -> float:
        return max(
            0.0,
            random.gauss(
                self.write_mean_s * self.degradation_multiplier, self.write_jitter_s
            ),
        )

    def query_delay(self) -> float:
        return max(
            0.0,
            random.gauss(
                self.query_mean_s * self.degradation_multiplier, self.query_jitter_s
            ),
        )


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


class FakeScaleEngine:
    """Dict-backed node store implementing the engine surface WorkItem + AgentBus need.

    Thread- and asyncio-task-safe (a real lock guards every mutation) so concurrent
    simulated workers racing a claim, or a duplicate-delivery replay, exercise REAL
    optimistic-concurrency arbitration rather than a serialized illusion of one.
    """

    def __init__(
        self,
        latency: LatencyModel | None = None,
        pace_mode: str = "sleep",
    ) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: list[tuple[str, str, str]] = []
        self._lock = threading.Lock()
        self.latency = latency or LatencyModel()
        #: ``"sleep"`` (default, standalone/interactive use): a genuine
        #: ``time.sleep`` per op, so this class behaves sensibly used on its own.
        #: ``"none"``: skip pacing entirely — used by the discrete-event soak/chaos
        #: driver (:mod:`scripts.scale.loadgen`), which accounts for synthetic
        #: op latency itself via :attr:`latency` sampled directly into its own
        #: time-ordered event heap, so the engine must not ALSO consume real or
        #: simulated time internally (that would double-count/desync the model).
        if pace_mode not in ("sleep", "none"):
            raise ValueError(f"pace_mode must be 'sleep' or 'none', got {pace_mode!r}")
        self.pace_mode = pace_mode
        # Observability counters the soak/chaos tests read directly (never inferred).
        self.write_count = 0
        self.query_count = 0
        self.cas_attempts = 0
        self.cas_wins = 0

    def _pace(self, delay: float) -> None:
        if self.pace_mode == "sleep":
            time.sleep(delay)

    # -- write surface (GraphEngineProtocol-shaped) --------------------------

    def add_node(
        self,
        node_id: str,
        node_type: str,
        properties: dict[str, Any] | None = None,
        ephemeral: bool = False,
    ) -> dict[str, Any]:
        self._pace(self.latency.write_delay())
        props = dict(properties or {})
        with self._lock:
            self.write_count += 1
            existing = self.nodes.get(node_id, {})
            merged = {**existing, **props, "label": node_type}
            self.nodes[node_id] = merged
            return dict(merged)

    def delete_node(self, node_id: str) -> bool:
        with self._lock:
            return self.nodes.pop(node_id, None) is not None

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: dict[str, Any] | None = None,
        ephemeral: bool = False,
    ) -> None:
        with self._lock:
            self.edges.append((source_id, target_id, str(rel_type)))

    # link_nodes is `_link`'s preferred name (falls back to add_edge) — same op.
    link_nodes = add_edge

    def compare_and_set_node_fields(
        self, node_id: str, conditions: dict[str, Any], updates: dict[str, Any]
    ) -> bool:
        self._pace(self.latency.write_delay())
        with self._lock:
            self.cas_attempts += 1
            node = self.nodes.get(node_id)
            if node is None:
                return False
            for k, v in conditions.items():
                if node.get(k) != v:
                    return False
            node.update(updates)
            self.cas_wins += 1
            self.write_count += 1
            return True

    # -- engine-native WorkItem verbs (D-OTD-2) ------------------------------
    #
    # ``work_item.py``'s claim/renew/commit/cancel/defer primitives dispatch
    # exclusively through these five generated verbs (mirroring
    # ``graph_compute.py``'s real ``self._client.work_items.*`` bridge) — there
    # is no Python-side CAS-scan fallback in production. This mirrors
    # ``tests/unit/orchestration/test_work_item.py``'s ``CasEngine`` double
    # (the load generator's own module docstring claimed this pattern but never
    # actually implemented it — these five methods were missing until D-OTD-2,
    # which is why every soak/chaos scenario driving a real claim/commit cycle
    # failed with ``NativeWorkItemRequired`` the first time this directory was
    # ever collected by pytest).

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
        self._pace(self.latency.write_delay())
        with self._lock:
            self.cas_attempts += 1
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
            self.cas_wins += 1
            self.write_count += 1
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
        self._pace(self.latency.write_delay())
        with self._lock:
            node = self.nodes.get(request["work_item_id"])
            if not self._owns(node, request):
                return {"renewed": False}
            assert node is not None
            node["lease_expires_at"] = float(request["now_unix"]) + float(
                request["lease_ttl"]
            )
            self.write_count += 1
            return {"renewed": True}

    def commit_work_item_result(self, request: dict[str, Any]) -> dict[str, Any]:
        self._pace(self.latency.write_delay())
        with self._lock:
            node = self.nodes.get(request["work_item_id"])
            if node is None:
                return {"status": "missing"}
            if node.get("status") in _wi.TERMINAL_WORK_ITEM_STATUSES:
                return {"status": "noop"}
            if not self._owns(node, request):
                return {"status": "fenced"}
            self.write_count += 1
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
        self._pace(self.latency.write_delay())
        with self._lock:
            node = self.nodes.get(request["work_item_id"])
            if node is None:
                return {"status": "missing"}
            if node.get("status") == "cancelled":
                return {"status": "cancelled"}
            if node.get("status") in _wi.TERMINAL_WORK_ITEM_STATUSES:
                return {"status": "not_cancellable"}
            node["status"] = "cancelled"
            self.write_count += 1
            return {"status": "cancelled"}

    def defer_work_item(self, request: dict[str, Any]) -> dict[str, Any]:
        self._pace(self.latency.write_delay())
        with self._lock:
            node = self.nodes.get(request["work_item_id"])
            if not self._owns(node, request) or float(
                (node or {}).get("lease_expires_at") or 0
            ) < float(request["now_unix"]):
                return {"status": "fenced"}
            assert node is not None
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
            self.write_count += 1
            return {"status": "deferred"}

    # -- read surface ----------------------------------------------------------

    def query_cypher(
        self, cypher: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        self._pace(self.latency.query_delay())
        params = params or {}
        with self._lock:
            self.query_count += 1
            return self._dispatch_query(" ".join(cypher.split()), params)

    # -- query dispatch: the closed set work_item.py + messaging.bus.AgentBus issue --

    def _dispatch_query(self, q: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        # ---- orchestration.work_item ----
        if q.startswith("MATCH (w:WorkItem {id: $id}) RETURN w.id"):
            node = self.nodes.get(params["id"])
            if node is None or node.get("label") != "WorkItem":
                return []
            row: dict[str, Any] = {"id": params["id"]}
            for f in _wi._FIELDS:
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

        # ---- messaging.bus.AgentBus graph fallback (log backend unconfigured) ----

        if q.startswith("MATCH (a:BusAgent {agent_id: $aid}) RETURN a"):
            node = self.nodes.get(f"busagent:{params['aid']}")
            if node is None or node.get("label") != "BusAgent":
                return []
            return [{"a": {"properties": node}}]

        if q.startswith("MATCH (a:BusAgent) RETURN a"):
            return [
                {"a": {"properties": n}}
                for n in self.nodes.values()
                if n.get("label") == "BusAgent"
            ]

        if q.startswith("MATCH (s:BusSubscription {topic: $t}) RETURN s"):
            return [
                {"s": {"properties": n}}
                for n in self.nodes.values()
                if n.get("label") == "BusSubscription" and n.get("topic") == params["t"]
            ]

        if q.startswith("MATCH (s:BusSubscription {agent_id: $aid}) RETURN s"):
            return [
                {"s": {"properties": n}}
                for n in self.nodes.values()
                if n.get("label") == "BusSubscription"
                and n.get("agent_id") == params["aid"]
            ]

        if q.startswith(
            "MATCH (c:BusTopicCursor {agent_id: $aid, topic: $t}) RETURN c"
        ):
            for n in self.nodes.values():
                if (
                    n.get("label") == "BusTopicCursor"
                    and n.get("agent_id") == params["aid"]
                    and n.get("topic") == params["t"]
                ):
                    return [{"c": {"properties": n}}]
            return []

        if q.startswith("MATCH (m:BusMessage {recipient: $aid}) RETURN m"):
            return [
                {"m": {"properties": n}}
                for n in self.nodes.values()
                if n.get("label") == "BusMessage"
                and n.get("recipient") == params["aid"]
            ]

        if q.startswith("MATCH (m:BusMessage {topic: $t, kind: 'topic'}) RETURN m"):
            return [
                {"m": {"properties": n}}
                for n in self.nodes.values()
                if n.get("label") == "BusMessage"
                and n.get("topic") == params["t"]
                and n.get("kind") == "topic"
            ]

        if q.startswith("MATCH (m:BusMessage {kind: 'topic'}) RETURN m"):
            return [
                {"m": {"properties": n}}
                for n in self.nodes.values()
                if n.get("label") == "BusMessage" and n.get("kind") == "topic"
            ]

        if q.startswith("MATCH (m:BusMessage {msg_group: $g}) RETURN m"):
            return [
                {"m": {"properties": n}}
                for n in self.nodes.values()
                if n.get("label") == "BusMessage" and n.get("msg_group") == params["g"]
            ]

        raise AssertionError(f"FakeScaleEngine: unrecognized query: {q[:200]!r}")

    # -- introspection for soak/chaos invariant assertions --------------------

    def work_items(self) -> list[dict[str, Any]]:
        return [
            dict(n, id=nid)
            for nid, n in self.nodes.items()
            if n.get("label") == "WorkItem"
        ]

    def bus_messages(self) -> list[dict[str, Any]]:
        return [dict(n) for n in self.nodes.values() if n.get("label") == "BusMessage"]

    def snapshot(self) -> dict[str, dict[str, Any]]:
        """A durable-store-equivalent snapshot (for simulating a process restart)."""
        with self._lock:
            return {nid: dict(n) for nid, n in self.nodes.items()}

    @classmethod
    def from_snapshot(
        cls,
        snapshot: dict[str, dict[str, Any]],
        *,
        latency: LatencyModel | None = None,
        pace_mode: str = "sleep",
    ) -> FakeScaleEngine:
        """Rehydrate a fresh engine instance from a durable snapshot.

        Simulates a full process restart / cold activation: process-local state
        (locks, in-flight asyncio tasks, worker registries) is gone, but everything
        the durable store persisted (every WorkItem/BusMessage/BusAgent node) comes
        back exactly as it was — the same guarantee the real tiered engine gives via
        its durable authority (``tenant_engine_pool.py``'s eviction-is-never-lossy
        contract).
        """
        engine = cls(latency=latency, pace_mode=pace_mode)
        engine.nodes = {nid: dict(n) for nid, n in snapshot.items()}
        return engine
