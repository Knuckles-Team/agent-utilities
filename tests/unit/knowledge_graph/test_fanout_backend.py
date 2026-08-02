"""Unit tests for the fan-out (N-way mirror) backend (CONCEPT:AU-KG.backend.mirror-health-repair).

Verifies the lossless-mirroring contract with recording fakes (no real
Postgres/Neo4j/FalkorDB server required):

* every write reaches every mirror (eventual convergence via the drainer);
* a mirror that is DOWN for a while loses nothing — it replays its outbox tail
  from the persisted cursor once it recovers;
* the durable outbox survives a "process restart" (close + reopen) and resumes
  from the cursor;
* reads are served from the authority and never block on a mirror.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from inspect import signature
from pathlib import Path
from typing import Any

import pytest

import agent_utilities.knowledge_graph.backends.fanout_backend as fanout_module
from agent_utilities.knowledge_graph.backends.base import GraphBackend
from agent_utilities.knowledge_graph.backends.fanout_backend import FanOutBackend
from agent_utilities.knowledge_graph.backends.outbox import GraphOutbox


class RecordingBackend(GraphBackend):
    """Fake GraphBackend recording applied writes; can be toggled down."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.down = False
        self._lock = threading.Lock()
        self.writes: list[tuple[str, Any]] = []
        self.execute_params: list[dict[str, Any]] = []

    def _check(self) -> None:
        if self.down:
            raise RuntimeError(f"{self.name} unreachable")

    def execute(self, query, params=None, *, include_epistemic=False):
        self._check()
        with self._lock:
            self.writes.append(("execute", query))
            self.execute_params.append(dict(params or {}))
        return [{"backend": self.name}]

    def execute_batch(self, query, batch):
        self._check()
        with self._lock:
            self.writes.append(("execute_batch", len(batch)))
        return [{"backend": self.name}]

    def create_schema(self):
        with self._lock:
            self.writes.append(("create_schema", None))

    def add_embedding(self, node_id, embedding):
        self._check()
        with self._lock:
            self.writes.append(("add_embedding", node_id))

    def verify_node_embedding(self, node_id, embedding):
        self._check()
        del embedding
        return ("add_embedding", node_id) in self.writes

    def semantic_search(self, query_embedding, n_results=5):
        return [{"backend": self.name}]

    def prune(self, criteria):
        self._check()

    def close(self):
        pass

    def n_execute(self) -> int:
        with self._lock:
            return sum(1 for op, _ in self.writes if op == "execute")


class StatefulBackend(RecordingBackend):
    """Typed recording backend with real node CAS semantics."""

    typed_mutation_support = "native"

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.nodes = {
            "node-1": {
                "id": "node-1",
                "node_type": "Service",
                "name": "billing",
                "classification": "INTERNAL",
                "external_access": {"read": ["team-a"]},
            }
        }

    def add_node(self, node_id, **properties):
        self._check()
        self.nodes.setdefault(node_id, {}).update(properties)
        with self._lock:
            self.writes.append(("add_node", node_id))

    def get_node_properties(self, node_id):
        node = self.nodes.get(node_id)
        return dict(node) if node is not None else None

    def compare_and_set_node_fields(self, node_id, conditions, updates):
        node = self.nodes.get(node_id)
        if node is None or any(
            node.get(field) != expected for field, expected in conditions.items()
        ):
            return False
        node.update(updates)
        return True

    def compare_and_set_node_embedding(self, node_id, conditions, updates, embedding):
        if not self.compare_and_set_node_fields(node_id, conditions, updates):
            return False
        self.add_embedding(node_id, embedding)
        return True


@pytest.fixture(autouse=True)
def _inert_epistemic_authority(monkeypatch):
    """Replace the fixed native authority with an inert contract test double."""
    monkeypatch.setattr(
        fanout_module,
        "_new_epistemic_authority",
        lambda: RecordingBackend("authority"),
    )


def _make(tmp_path: Path, mirrors: dict[str, RecordingBackend]) -> FanOutBackend:
    return FanOutBackend(
        mirrors,
        outbox_path=str(tmp_path / "outbox.db"),
    )


def test_public_constructor_has_no_authority_selector():
    assert "authority" not in signature(FanOutBackend).parameters


def test_compare_and_set_updates_authority_and_mirror_only_for_winner(
    tmp_path, monkeypatch
):
    authority = StatefulBackend("authority")
    mirror = StatefulBackend("mirror")
    monkeypatch.setattr(
        fanout_module,
        "_new_epistemic_authority",
        lambda: authority,
    )
    fan = FanOutBackend({"mirror": mirror}, outbox_path=str(tmp_path / "ob.db"))
    try:
        assert fan.compare_and_set_node_fields(
            "node-1",
            {"embedding": None, "name": "billing"},
            {"embedding": [1.0, 2.0]},
        )
        assert fan.flush_mirrors(timeout=10.0)

        assert authority.nodes["node-1"]["embedding"] == [1.0, 2.0]
        assert mirror.nodes["node-1"] == authority.nodes["node-1"]
        assert mirror.nodes["node-1"]["classification"] == "INTERNAL"
        assert mirror.nodes["node-1"]["external_access"] == {"read": ["team-a"]}

        writes_before = len(mirror.writes)
        assert not fan.compare_and_set_node_fields(
            "node-1",
            {"embedding": None},
            {"embedding": [3.0, 4.0]},
        )
        assert fan.flush_mirrors(timeout=10.0)
        assert len(mirror.writes) == writes_before
        assert authority.nodes["node-1"]["embedding"] == [1.0, 2.0]
    finally:
        fan.close()


def test_atomic_embedding_update_replays_one_idempotent_mirror_operation(
    tmp_path, monkeypatch
):
    authority = StatefulBackend("authority")
    mirror = StatefulBackend("mirror")
    monkeypatch.setattr(
        fanout_module,
        "_new_epistemic_authority",
        lambda: authority,
    )
    fan = FanOutBackend({"mirror": mirror}, outbox_path=str(tmp_path / "ob.db"))
    vector = [1.0, 2.0]
    try:
        assert fan.compare_and_set_node_embedding(
            "node-1",
            {"embedding": None, "name": "billing"},
            {"embedding": vector},
            vector,
        )
        assert fan.flush_mirrors(timeout=10.0)

        assert authority.nodes["node-1"]["embedding"] == vector
        assert mirror.nodes["node-1"]["embedding"] == vector
        assert ("add_embedding", "node-1") in authority.writes
        assert ("add_embedding", "node-1") in mirror.writes

        mirror_writes = len(mirror.writes)
        assert not fan.compare_and_set_node_embedding(
            "node-1",
            {"embedding": None},
            {"embedding": [3.0, 4.0]},
            [3.0, 4.0],
        )
        assert fan.flush_mirrors(timeout=10.0)
        assert len(mirror.writes) == mirror_writes
    finally:
        fan.close()


def test_atomic_embedding_replay_converges_mirror_without_cas_and_unblocks_tail(
    tmp_path, monkeypatch
):
    """Standard mirrors without optional CAS still converge and advance cursor."""
    authority = StatefulBackend("authority")
    mirror = RecordingBackend("mirror-without-cas")
    monkeypatch.setattr(
        fanout_module,
        "_new_epistemic_authority",
        lambda: authority,
    )
    fan = FanOutBackend({"mirror": mirror}, outbox_path=str(tmp_path / "ob.db"))
    vector = [1.0, 2.0]
    try:
        assert fan.compare_and_set_node_embedding(
            "node-1",
            {"embedding": None, "name": "billing"},
            {"embedding": vector},
            vector,
        )
        fan.execute("CREATE (tail)", is_write=True)
        assert fan.flush_mirrors(timeout=2.0)

        assert ("add_embedding", "node-1") in mirror.writes
        assert mirror.writes.count(("add_embedding", "node-1")) == 1
        assert all("embedding" not in params for params in mirror.execute_params)
        assert ("execute", "CREATE (tail)") in mirror.writes
        assert fan._outbox.lag("mirror") == 0
    finally:
        fan.close()


def test_atomic_embedding_replay_does_not_ack_swallowed_mirror_writes(
    tmp_path, monkeypatch
):
    """A silent operational failure keeps the durable cursor behind the entry."""

    class _SwallowingMirror(RecordingBackend):
        def execute(self, query, params=None, *, include_epistemic=False):
            del query, params, include_epistemic
            return []

        def add_embedding(self, node_id, embedding):
            del node_id, embedding

        def verify_node_embedding(self, node_id, embedding):
            del node_id, embedding
            return False

    authority = StatefulBackend("authority")
    mirror = _SwallowingMirror("swallowing-mirror")
    monkeypatch.setattr(
        fanout_module,
        "_new_epistemic_authority",
        lambda: authority,
    )
    fan = FanOutBackend({"mirror": mirror}, outbox_path=str(tmp_path / "ob.db"))
    try:
        assert fan.compare_and_set_node_embedding(
            "node-1",
            {"embedding": None, "name": "billing"},
            {"embedding": [1.0, 2.0]},
            [1.0, 2.0],
        )

        assert not fan.flush_mirrors(timeout=1.0)
        assert fan._outbox.lag("mirror") == 1
        assert mirror.writes == []
    finally:
        fan.close()


def test_atomic_embedding_node_property_mirror_writes_vector_once(
    tmp_path, monkeypatch
):
    """Node-property mirrors verify the upsert without a duplicate add call."""

    class _NodePropertyMirror(RecordingBackend):
        embedding_is_node_property = True

        def verify_node_embedding(self, node_id, embedding):
            del node_id
            return sum(
                1 for params in self.execute_params if params.get("embedding") == embedding
            ) == 1

    authority = StatefulBackend("authority")
    mirror = _NodePropertyMirror("node-property-mirror")
    monkeypatch.setattr(
        fanout_module,
        "_new_epistemic_authority",
        lambda: authority,
    )
    fan = FanOutBackend({"mirror": mirror}, outbox_path=str(tmp_path / "ob.db"))
    try:
        assert fan.compare_and_set_node_embedding(
            "node-1",
            {"embedding": None, "name": "billing"},
            {"embedding": [1.0, 2.0]},
            [1.0, 2.0],
        )

        assert fan.flush_mirrors(timeout=2.0)
        assert ("add_embedding", "node-1") not in mirror.writes
        assert sum(
            1
            for params in mirror.execute_params
            if params.get("embedding") == [1.0, 2.0]
        ) == 1
    finally:
        fan.close()


def test_independent_typed_writes_overlap_authority_rpcs(tmp_path, monkeypatch):
    """D-BFR-7: unrelated entities must not share one authority-RPC lock."""

    class _ConcurrentAuthority(StatefulBackend):
        def __init__(self):
            super().__init__("authority")
            self.active = 0
            self.max_active = 0
            self.activity_lock = threading.Lock()
            self.both_started = threading.Event()
            self.release = threading.Event()

        def add_node(self, node_id, **properties):
            with self.activity_lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
                if self.active == 2:
                    self.both_started.set()
            try:
                assert self.release.wait(timeout=5.0)
                super().add_node(node_id, **properties)
            finally:
                with self.activity_lock:
                    self.active -= 1

    authority = _ConcurrentAuthority()
    mirror = StatefulBackend("mirror")
    monkeypatch.setattr(
        fanout_module,
        "_new_epistemic_authority",
        lambda: authority,
    )
    fan = FanOutBackend({"mirror": mirror}, outbox_path=str(tmp_path / "ob.db"))
    node_a = "independent-a"
    suffix = 0
    node_b = "independent-b"
    while fan._mutation_lock(node_a) is fan._mutation_lock(node_b):
        suffix += 1
        node_b = f"independent-b-{suffix}"

    errors: list[BaseException] = []

    def _write(node_id):
        try:
            fan.add_node(node_id, node_type="Service", name=node_id)
        except BaseException as exc:  # noqa: BLE001 - thread assertion relay
            errors.append(exc)

    workers = [
        threading.Thread(target=_write, args=(node_a,)),
        threading.Thread(target=_write, args=(node_b,)),
    ]
    try:
        for worker in workers:
            worker.start()
        assert authority.both_started.wait(timeout=5.0)
        authority.release.set()
        for worker in workers:
            worker.join(timeout=5.0)
        assert not errors
        assert authority.max_active == 2
        assert fan.flush_mirrors(timeout=10.0)
        assert mirror.nodes[node_a] == authority.nodes[node_a]
        assert mirror.nodes[node_b] == authority.nodes[node_b]
    finally:
        authority.release.set()
        for worker in workers:
            worker.join(timeout=5.0)
        fan.close()


def test_same_entity_conflicting_writes_finish_with_mirror_convergence(
    tmp_path, monkeypatch
):
    """CAS and a typed upsert may race, but mirror final state stays authoritative."""
    authority = StatefulBackend("authority")
    mirror = StatefulBackend("mirror")
    monkeypatch.setattr(
        fanout_module,
        "_new_epistemic_authority",
        lambda: authority,
    )
    fan = FanOutBackend({"mirror": mirror}, outbox_path=str(tmp_path / "ob.db"))
    start = threading.Barrier(3)
    cas_results: list[bool] = []

    def _upsert():
        start.wait(timeout=5.0)
        fan.add_node(
            "node-1",
            node_type="Service",
            name="billing-renamed",
            classification="RESTRICTED",
        )

    def _cas():
        start.wait(timeout=5.0)
        cas_results.append(
            fan.compare_and_set_node_fields(
                "node-1",
                {"embedding": None, "name": "billing"},
                {"embedding": [1.0, 2.0]},
            )
        )

    workers = [threading.Thread(target=_upsert), threading.Thread(target=_cas)]
    try:
        for worker in workers:
            worker.start()
        start.wait(timeout=5.0)
        for worker in workers:
            worker.join(timeout=5.0)
        assert cas_results in ([True], [False])
        assert fan.flush_mirrors(timeout=10.0)
        assert mirror.nodes["node-1"] == authority.nodes["node-1"]
        assert authority.nodes["node-1"]["name"] == "billing-renamed"
        assert authority.nodes["node-1"]["classification"] == "RESTRICTED"
    finally:
        fan.close()


def _stop_drainers(fan: FanOutBackend) -> None:
    """Stop + join ALL background threads (the persister + per-mirror drainers), so
    the ring→outbox→mirror path is driven purely synchronously by the test (no thread
    race / CPU-starvation flake). After this, the async hand-off ring may still hold
    un-persisted writes — :func:`_drain_outbox_sync` flushes them first."""
    fan._stop.set()
    if fan._persister is not None:
        fan._persister.join(timeout=10.0)
    for st in fan._state.values():
        t = getattr(st, "thread", None)
        if t is not None:
            t.join(timeout=10.0)


def _drain_outbox_sync(fan: FanOutBackend) -> None:
    """Drive the full mirror hand-off synchronously: first flush the in-memory ring
    into the durable outbox (the persister's job, CONCEPT:AU-KG.backend.authority-has-already-acked), then apply every
    mirror's outbox tail (the drainer's apply→ack path). The background threads must
    already be stopped (see :func:`_stop_drainers`) so nothing races this drain.
    Deterministic regardless of box load."""
    outbox = fan._outbox
    assert outbox is not None
    fan._drain_handoff_remaining()  # ring -> durable outbox (persister stopped)
    for mirror, backend in fan._mirrors.items():
        while outbox.lag(mirror) > 0:
            pending = outbox.pending(mirror)
            if not pending:
                break
            for entry in pending:
                fan._apply(backend, entry)
                outbox.ack(mirror, entry.seq)


def test_write_fans_out_to_every_mirror(tmp_path):
    a, b = RecordingBackend("a"), RecordingBackend("b")
    fan = _make(tmp_path, {"a": a, "b": b})
    try:
        # Stop the background drainers up front so writes only append to the
        # outbox; the apply is then driven synchronously below. This makes the
        # convergence assertion deterministic regardless of how starved the
        # drainer thread would be on a saturated box (the contract is that every
        # write reaches every mirror — not how fast a background thread schedules).
        _stop_drainers(fan)
        for i in range(20):
            fan.execute(f"CREATE (n:Doc {{id:'{i}'}})", is_write=True)
        _drain_outbox_sync(fan)
        assert a.n_execute() == 20
        assert b.n_execute() == 20
    finally:
        fan.close()


def test_ack_does_not_wait_on_mirror_enqueue(tmp_path):
    """THE KEY PRINCIPLE (CONCEPT:AU-KG.backend.authority-has-already-acked): the authority ack must NOT wait on the
    mirror enqueue. A slow/blocked durable outbox ``append`` is absorbed by the async
    hand-off — the write returns immediately — and the mirror still receives the write
    asynchronously once the persister catches up. Operator's law: blocked time on the
    ack path is wasted compute."""
    m = RecordingBackend("m")
    fan = _make(tmp_path, {"m": m})
    try:
        slow_s = 0.6
        real_append = fan._outbox.append

        def slow_append(op, payload):
            # The persister thread blocks here; the producer (execute) must not.
            time.sleep(slow_s)
            return real_append(op, payload)

        fan._outbox.append = slow_append  # type: ignore[method-assign]

        start = time.monotonic()
        fan.execute("CREATE (n:Doc {id:'1'})", is_write=True)
        ack_latency = time.monotonic() - start
        # The ack returned essentially instantly — it did NOT block on the 0.6s outbox
        # append happening on the persister thread.
        assert ack_latency < slow_s / 2, (
            f"ack waited {ack_latency:.3f}s on the mirror enqueue (should be ~0)"
        )

        # ...and the mirror still receives the write asynchronously (eventual).
        assert fan.flush_mirrors(timeout=10.0)
        assert m.n_execute() == 1
    finally:
        fan.close()


def test_overflow_falls_back_to_durable_outbox(tmp_path, monkeypatch):
    """Bounded + backpressure (CONCEPT:AU-KG.backend.authority-has-already-acked): when the in-memory ring is full
    (the persister can't keep up), a further write is NOT dropped and cannot overtake
    older ring items: the producer helps persist the oldest item first. Memory stays
    bounded."""
    monkeypatch.setattr(fanout_module, "_auto_handoff_capacity", lambda: 4)
    m = RecordingBackend("m")
    fan = fanout_module.FanOutBackend({"m": m}, outbox_path=str(tmp_path / "ob.db"))
    try:
        # Stop the persister + drainers so the ring cannot drain — forces overflow.
        _stop_drainers(fan)
        cap = fan._handoff.maxsize  # 4
        for i in range(cap):
            fan.execute(f"CREATE (n{i})", is_write=True)  # fills the ring exactly
        assert fan._handoff.full()
        depth_before = fan._outbox.depth()
        # This write overflows the ring -> the OLDEST ring item lands durably,
        # then the newer overflow mutation is admitted behind the remaining tail.
        fan.execute("CREATE (overflow)", is_write=True)
        assert fan._outbox.depth() == depth_before + 1  # landed durably, not dropped
        _drain_outbox_sync(fan)
        assert [value for op, value in m.writes if op == "execute"] == [
            *(f"CREATE (n{i})" for i in range(cap)),
            "CREATE (overflow)",
        ]
    finally:
        fan.close()


def test_concurrent_overflow_producers_keep_fifo_admission(tmp_path, monkeypatch):
    """D-BFR-7: a helper's freed ring slot cannot be stolen by a later writer."""
    monkeypatch.setattr(fanout_module, "_auto_handoff_capacity", lambda: 2)
    mirror = RecordingBackend("mirror")
    fan = fanout_module.FanOutBackend(
        {"mirror": mirror}, outbox_path=str(tmp_path / "ob.db")
    )
    _stop_drainers(fan)
    real_append = fan._outbox.append
    first_append_started = threading.Event()
    release_first_append = threading.Event()
    blocked_once = False

    def _blocking_append(op, payload):
        nonlocal blocked_once
        if not blocked_once:
            blocked_once = True
            first_append_started.set()
            assert release_first_append.wait(timeout=5.0)
        return real_append(op, payload)

    fan._outbox.append = _blocking_append  # type: ignore[method-assign]
    errors: list[BaseException] = []

    def _write(name):
        try:
            fan.execute(f"CREATE ({name})", is_write=True)
        except BaseException as exc:  # noqa: BLE001 - thread assertion relay
            errors.append(exc)

    try:
        fan.execute("CREATE (oldest-0)", is_write=True)
        fan.execute("CREATE (oldest-1)", is_write=True)
        first = threading.Thread(target=_write, args=("overflow-first",))
        second = threading.Thread(target=_write, args=("overflow-second",))
        first.start()
        assert first_append_started.wait(timeout=5.0)
        second.start()
        # The second producer must queue behind the first overflow waiter even
        # though the helper has popped one old ring item and exposed a slot.
        time.sleep(0.05)
        assert second.is_alive()
        release_first_append.set()
        first.join(timeout=5.0)
        second.join(timeout=5.0)
        assert not first.is_alive() and not second.is_alive()
        assert not errors

        fan._outbox.append = real_append  # type: ignore[method-assign]
        _drain_outbox_sync(fan)
        assert [value for op, value in mirror.writes if op == "execute"] == [
            "CREATE (oldest-0)",
            "CREATE (oldest-1)",
            "CREATE (overflow-first)",
            "CREATE (overflow-second)",
        ]
    finally:
        release_first_append.set()
        fan._outbox.append = real_append  # type: ignore[method-assign]
        fan.close()


def test_close_fences_producers_and_durably_hands_off_active_writer(
    tmp_path, monkeypatch
):
    """D-LRR-1: close cannot strand a write that already entered authority."""

    class _BlockingAuthority(StatefulBackend):
        def __init__(self):
            super().__init__("authority")
            self.write_started = threading.Event()
            self.release_write = threading.Event()

        def add_node(self, node_id, **properties):
            if node_id == "active-node":
                self.write_started.set()
                assert self.release_write.wait(timeout=5.0)
            super().add_node(node_id, **properties)

    authority = _BlockingAuthority()
    mirror = StatefulBackend("mirror")
    monkeypatch.setattr(
        fanout_module,
        "_new_epistemic_authority",
        lambda: authority,
    )
    outbox_path = tmp_path / "ob.db"
    fan = FanOutBackend({"mirror": mirror}, outbox_path=str(outbox_path))
    real_append = fan._outbox.append
    real_close = fan._outbox.close
    outbox_closed = threading.Event()
    append_after_close: list[str] = []

    def _guarded_append(op, payload):
        if outbox_closed.is_set():
            append_after_close.append(op)
        return real_append(op, payload)

    def _tracked_close():
        outbox_closed.set()
        return real_close()

    fan._outbox.append = _guarded_append  # type: ignore[method-assign]
    fan._outbox.close = _tracked_close  # type: ignore[method-assign]
    errors: list[BaseException] = []

    def _active_write():
        try:
            fan.add_node("active-node", node_type="Service", name="active")
        except BaseException as exc:  # noqa: BLE001 - thread assertion relay
            errors.append(exc)

    def _close():
        try:
            fan.close()
        except BaseException as exc:  # noqa: BLE001 - thread assertion relay
            errors.append(exc)

    writer = threading.Thread(target=_active_write)
    closer = threading.Thread(target=_close)
    writer.start()
    assert authority.write_started.wait(timeout=5.0)
    closer.start()
    deadline = time.monotonic() + 5.0
    while (
        getattr(fan, "_lifecycle_state", None) != "closing"
        and time.monotonic() < deadline
    ):
        time.sleep(0.01)
    try:
        assert getattr(fan, "_lifecycle_state", None) == "closing"
        assert closer.is_alive(), "close returned while an authority writer was active"
        with pytest.raises(RuntimeError, match="closing|closed"):
            fan.add_node("late-node", node_type="Service", name="late")
        assert "late-node" not in authority.nodes

        authority.release_write.set()
        writer.join(timeout=5.0)
        closer.join(timeout=5.0)
        assert not writer.is_alive() and not closer.is_alive()
        assert not errors
        assert append_after_close == []
        assert authority.nodes["active-node"]["name"] == "active"

        reopened = GraphOutbox(outbox_path, ["mirror"])
        try:
            assert reopened.depth() >= 1
        finally:
            reopened.close()
    finally:
        authority.release_write.set()
        writer.join(timeout=5.0)
        closer.join(timeout=5.0)


def test_close_waits_for_active_sqlite_append_then_retries_before_close(
    tmp_path, monkeypatch
):
    """D-LRR-1: an in-flight failed append is retried before SQLite closes."""
    mirror = RecordingBackend("mirror")
    fan = FanOutBackend({"mirror": mirror}, outbox_path=str(tmp_path / "ob.db"))
    real_append = fan._outbox.append
    real_close = fan._outbox.close
    append_started = threading.Event()
    release_first_append = threading.Event()
    retry_completed = threading.Event()
    outbox_closed = threading.Event()
    append_after_close: list[str] = []
    append_calls = 0

    def _retrying_append(op, payload):
        nonlocal append_calls
        append_calls += 1
        if outbox_closed.is_set():
            append_after_close.append(op)
        if append_calls == 1:
            append_started.set()
            assert release_first_append.wait(timeout=5.0)
            raise sqlite3.OperationalError("injected transient append failure")
        value = real_append(op, payload)
        retry_completed.set()
        return value

    def _tracked_close():
        outbox_closed.set()
        return real_close()

    fan._outbox.append = _retrying_append  # type: ignore[method-assign]
    fan._outbox.close = _tracked_close  # type: ignore[method-assign]
    fan.execute("CREATE (active-append)", is_write=True)
    assert append_started.wait(timeout=5.0)
    errors: list[BaseException] = []

    def _close():
        try:
            fan.close()
        except BaseException as exc:  # noqa: BLE001 - thread assertion relay
            errors.append(exc)

    closer = threading.Thread(target=_close)
    closer.start()
    try:
        time.sleep(0.05)
        assert closer.is_alive()
        assert not outbox_closed.is_set()
        assert not retry_completed.is_set()

        release_first_append.set()
        closer.join(timeout=5.0)
        assert not closer.is_alive()
        assert not errors
        assert retry_completed.is_set()
        assert append_calls == 2
        assert append_after_close == []
        assert outbox_closed.is_set()

        reopened = GraphOutbox(tmp_path / "ob.db", ["mirror"])
        try:
            assert reopened.depth() >= 1
        finally:
            reopened.close()
    finally:
        release_first_append.set()
        closer.join(timeout=5.0)


def test_handoff_write_lands_durably_before_mirror_ships(tmp_path):
    """Crash-safety preserved (CONCEPT:AU-KG.backend.authority-has-already-acked): an async-handed-off write reaches the
    DURABLE outbox (so it survives a crash and replays from the cursor on restart)
    before any mirror applies it — proven here while the mirror is DOWN."""
    down = RecordingBackend("down")
    down.down = True  # offline before any write
    fan = _make(tmp_path, {"down": down})
    try:
        fan.execute("CREATE (n:Doc {id:'1'})", is_write=True)
        # Wait for the persister to land the hand-off durably.
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not fan._durable_caught_up():
            time.sleep(0.02)
        assert fan._durable_caught_up()
        # It is durably queued (lag==1) for the down mirror — a crash here REPLAYS it.
        assert fan._outbox.lag("down") == 1
        # Reopen the same outbox file fresh: the entry persisted (survives restart).
        reopened = GraphOutbox(str(tmp_path / "outbox.db"), ["down"])
        try:
            assert [e.op for e in reopened.pending("down")] == ["execute"]
        finally:
            reopened.close()
        # Recover the mirror — it ships from the durable cursor, no loss.
        down.down = False
        assert fan.flush_mirrors(timeout=10.0)
        assert down.n_execute() == 1
    finally:
        fan.close()


def test_reads_go_to_authority_not_mirrors(tmp_path):
    a = RecordingBackend("a")
    fan = _make(tmp_path, {"a": a})
    try:
        rows = fan.execute("MATCH (n) RETURN n", is_write=False)
        assert rows == [{"backend": "authority"}]
        # A pure read must never be enqueued to a mirror.
        assert fan.flush_mirrors(timeout=5.0)
        assert a.n_execute() == 0
    finally:
        fan.close()


def test_down_mirror_loses_nothing_and_replays(tmp_path):
    """A mirror that is offline during writes catches up fully on recovery."""
    up, down = RecordingBackend("up"), RecordingBackend("down")
    down.down = True  # offline before any write
    fan = _make(tmp_path, {"up": up, "down": down})
    try:
        for i in range(15):
            fan.execute(f"CREATE (n:Doc {{id:'{i}'}})", is_write=True)
        # The healthy mirror converges; the down one is fully behind.
        assert fan.flush_mirrors(timeout=2.0) is False
        assert up.n_execute() == 15
        stats = fan.durability_stats()
        assert stats["mirrors"]["down"]["lag"] == 15
        # Recover the mirror — the drainer replays the persisted tail.
        down.down = False
        assert fan.flush_mirrors(timeout=10.0)
        assert down.n_execute() == 15  # NO LOSS
        assert fan.durability_stats()["mirrors"]["down"]["lag"] == 0
    finally:
        fan.close()


def test_outbox_survives_restart(tmp_path):
    """The durable log resumes from its cursor after a process restart."""
    path = tmp_path / "ob.db"
    ob = GraphOutbox(path, ["m"])
    ob.append("execute", {"query": "CREATE (n)", "params": None})
    ob.append("execute", {"query": "CREATE (m)", "params": None})
    assert ob.lag("m") == 2
    ob.ack("m", 1)  # applied the first only
    ob.close()

    # Reopen — the cursor persisted; only the unapplied tail remains pending.
    ob2 = GraphOutbox(path, ["m"])
    try:
        pending = ob2.pending("m")
        assert [e.seq for e in pending] == [2]
        assert ob2.applied_seq("m") == 1
    finally:
        ob2.close()


def test_concurrent_writer_waits_not_locked(tmp_path):
    """A second writer on the SAME outbox file (the client + host both open the
    shared ``graph_mirror_outbox.db``) must WAIT for the write lock, not fail with
    "database is locked". Reproduces the split-storage throttle and proves the
    busy_timeout fix: while a raw connection holds the write lock, ``append`` blocks
    briefly and then succeeds once the lock is released (CONCEPT:AU-KG.backend.mirror-health-repair)."""
    import sqlite3
    import time

    path = tmp_path / "shared_outbox.db"
    ob = GraphOutbox(path, ["m"])
    try:
        hold_s = 0.4
        locked = threading.Event()
        released = threading.Event()

        # A separate connection (a stand-in for the OTHER graph-os process) grabs
        # the write lock and holds it for ``hold_s`` seconds. Created INSIDE the
        # thread (sqlite connections are thread-affine).
        def _hold():
            holder = sqlite3.connect(str(path), isolation_level=None, timeout=5.0)
            holder.execute("PRAGMA busy_timeout=5000")
            holder.execute("BEGIN IMMEDIATE")
            holder.execute(
                "INSERT INTO outbox (mirror, seq, op, payload, created_at) "
                "VALUES ('m', 9999, 'x', '{}', 0.0)"
            )
            locked.set()  # the write lock is now held
            time.sleep(hold_s)
            holder.execute("COMMIT")
            released.set()
            holder.close()

        t = threading.Thread(target=_hold)
        t.start()
        assert locked.wait(2.0)  # ensure the holder owns the write lock first

        # This append would raise "database is locked" instantly WITHOUT a busy
        # timeout; with it, the call waits out the holder and returns a real seq.
        start = time.time()
        seq = ob.append("execute", {"query": "CREATE (n)", "params": None})
        waited = time.time() - start

        assert seq >= 1  # the write durably landed
        assert released.is_set()  # we waited until the holder released the lock
        assert waited >= hold_s - 0.1  # i.e. we blocked rather than failing fast
        t.join()
    finally:
        ob.close()


class _GraphStub:
    def _get_all_nodes(self):
        return ["a", "b"]

    def _get_node_properties(self, nid):
        return {"type": "Concept", "name": nid}

    def _get_all_edges(self):
        return [("a", "b", {"type": "RELATED_TO"})]


class _AuthorityWithGraph(RecordingBackend):
    @property
    def graph(self):
        return _GraphStub()


def test_reconcile_repairs_each_mirror(tmp_path, monkeypatch):
    """reconcile() repairs every mirror from the authority, fast and
    without spinning — proves the authority→mirror drift-repair wrapper."""
    mirror = RecordingBackend("m")
    monkeypatch.setattr(
        fanout_module,
        "_new_epistemic_authority",
        lambda: _AuthorityWithGraph("auth"),
    )
    fan = FanOutBackend(
        {"m": mirror},
        outbox_path=str(tmp_path / "ob.db"),
    )
    try:
        report = fan.reconcile()  # all mirrors
        assert "m" in report
        assert report["m"]["nodes"] == 2  # both authority nodes re-synced
        assert report["m"]["edges"] == 1
        assert report["m"]["errors"] == 0
        # The mirror received node CREATE + edge MERGE writes.
        writes = [c for c in mirror.writes if c[0] == "execute"]
        assert writes  # drift repair actually wrote to the mirror
    finally:
        fan.close()


def test_reconcile_stops_on_owned_task_cancellation(tmp_path, monkeypatch):
    """Mirror repair observes its queue owner's timeout before another write."""
    from agent_utilities.core.task_cancellation import (
        TaskCancellationRequested,
        use_task_cancellation,
    )

    mirror = RecordingBackend("m")
    monkeypatch.setattr(
        fanout_module,
        "_new_epistemic_authority",
        lambda: _AuthorityWithGraph("auth"),
    )
    fan = FanOutBackend({"m": mirror}, outbox_path=str(tmp_path / "ob.db"))
    cancelled = threading.Event()
    cancelled.set()
    try:
        with (
            use_task_cancellation(cancelled),
            pytest.raises(TaskCancellationRequested),
        ):
            fan.reconcile()
        assert not [
            call
            for call in mirror.writes
            if call[0] == "execute" and "MERGE" in call[1]
        ]
    finally:
        fan.close()


class LadybugBackend(RecordingBackend):
    """Recording fake whose CLASS NAME drives the strict-schema edge dialect."""


class Neo4jBackend(RecordingBackend):
    """Recording fake whose CLASS NAME drives the native-cypher edge dialect."""


def _edge_merge_writes(mirror: RecordingBackend) -> list[str]:
    """The MERGE edge cypher applied to a mirror (not the label lookups)."""
    return [q for op, q in mirror.writes if op == "execute" and "MERGE (s)-[r:" in q]


def test_edge_write_replays_structurally_per_dialect(tmp_path):
    """An edge MERGE fans out STRUCTURALLY: each mirror gets a dialect-correct
    write. Ladybug folds props into its `properties` JSON column; native-cypher
    mirrors keep per-prop SET. This is what lets Ladybug edges stream live rather
    than only converge on a reconcile sweep (the raw forwarded cypher would drop
    Ladybug edge props)."""
    lady, neo = LadybugBackend("lady"), Neo4jBackend("neo")
    fan = FanOutBackend(
        {"lady": lady, "neo": neo},
        outbox_path=str(tmp_path / "ob.db"),
    )
    try:
        # The engine's edge-write shape (IntelligenceGraphEngine._upsert_edge).
        # cypher-write-subset-allow: constructs
        # `lady, neo = LadybugBackend("lady"), Neo4jBackend("neo")` (both
        # RecordingBackend test fakes) and fans a write through them via
        # FanOutBackend — never through the native eg-query parser.
        fan.execute(
            "MATCH (s {id: $sid}) MATCH (t {id: $tid}) "
            "MERGE (s)-[r:DEPENDS_ON]->(t) SET r.`confidence` = $confidence",
            {"sid": "a", "tid": "b", "confidence": 0.9},
            is_write=True,
        )
        assert fan.flush_mirrors(timeout=10.0)

        lady_edges = _edge_merge_writes(lady)
        neo_edges = _edge_merge_writes(neo)
        assert len(lady_edges) == 1 and len(neo_edges) == 1
        # Same relationship type reaches both, derived structurally from the MERGE.
        assert "[r:DEPENDS_ON]" in lady_edges[0]
        assert "[r:DEPENDS_ON]" in neo_edges[0]
        # Ladybug folds the edge prop into its JSON `properties` column...
        assert "r.`properties`" in lady_edges[0]
        assert "r.`confidence`" not in lady_edges[0]
        # ...while the native-cypher mirror keeps the per-prop SET.
        assert "r.`confidence`" in neo_edges[0]
        assert "r.`properties`" not in neo_edges[0]
    finally:
        fan.close()


def test_non_edge_write_still_forwards_raw_cypher(tmp_path):
    """Node MERGE / ad-hoc writes are NOT edge upserts — they forward verbatim
    (portable for every backend), so the structural path is edge-only."""
    m = RecordingBackend("m")
    fan = _make(tmp_path, {"m": m})
    try:
        fan.execute("MERGE (n:Agent {id: $id})", {"id": "x"}, is_write=True)
        assert fan.flush_mirrors(timeout=10.0)
        forwarded = [q for op, q in m.writes if op == "execute"]
        assert forwarded == ["MERGE (n:Agent {id: $id})"]
    finally:
        fan.close()


def test_durability_stats_shape(tmp_path):
    fan = _make(tmp_path, {"a": RecordingBackend("a")})
    try:
        fan.execute("CREATE (n:Doc {id:'1'})", is_write=True)
        assert fan.flush_mirrors(timeout=5.0)
        stats = fan.durability_stats()
        assert stats["authority_writes"] == 1
        assert "a" in stats["mirrors"]
        assert stats["mirrors"]["a"]["writes"] == 1
        assert stats["mirrors"]["a"]["lag"] == 0
    finally:
        fan.close()
