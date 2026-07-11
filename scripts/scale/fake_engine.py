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
        its L3 durable mirror (``tenant_engine_pool.py``'s eviction-is-never-lossy
        contract).
        """
        engine = cls(latency=latency, pace_mode=pace_mode)
        engine.nodes = {nid: dict(n) for nid, n in snapshot.items()}
        return engine
