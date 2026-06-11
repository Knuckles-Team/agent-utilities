"""Tranche-3 ingest scale-out tests (CONCEPT:KG-2.55 / KG-2.56 / KG-2.57).

Covers fail-loud queue selection, keyed partitions + idempotent topic
provisioning, the decoupled ``kg-ingest`` consumer loop (idempotent claims,
at-least-once commits), uniform queue-depth backpressure, and the new gateway
metrics. No live Kafka and no ``confluent_kafka`` install required: the
library is stubbed into ``sys.modules`` and clients are injected fakes.
An optional live-broker test is gated on ``AGENT_UTILITIES_KAFKA_LIVE``.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_utilities.knowledge_graph.core.engine_tasks import (
    TaskManagerMixin,
    _decode_metadata,
    _encode_metadata,
    compute_ingest_worker_count,
)
from agent_utilities.knowledge_graph.core.queue_backend import (
    TaskQueueUnavailable,
    create_task_queue,
    resolve_task_queue_backend,
)


def _cfg(**overrides):
    base = {
        "task_queue_backend": None,
        "queue_backend": "sqlite",
        "state_db_uri": None,
        "kafka_bootstrap_servers": None,
        "nats_url": None,
        "kg_tasks_partitions": 6,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


# ── KG-2.55: selection resolution ───────────────────────────────────────


def test_resolve_auto_defaults_to_sqlite():
    assert resolve_task_queue_backend(_cfg()) == ("sqlite", False)


def test_resolve_auto_prefers_postgres_with_state_db_uri():
    cfg = _cfg(state_db_uri="postgresql://x/y")
    assert resolve_task_queue_backend(cfg) == ("postgres", False)


def test_resolve_explicit_wins_over_state_db_uri():
    cfg = _cfg(task_queue_backend="sqlite", state_db_uri="postgresql://x/y")
    assert resolve_task_queue_backend(cfg) == ("sqlite", True)


@pytest.mark.parametrize("value", ["kafka", "postgres", "sqlite", " Kafka "])
def test_resolve_explicit_values(value):
    choice, explicit = resolve_task_queue_backend(_cfg(task_queue_backend=value))
    assert choice == value.strip().lower()
    assert explicit is True


def test_resolve_rejects_unknown_value():
    with pytest.raises(ValueError, match="TASK_QUEUE_BACKEND"):
        resolve_task_queue_backend(_cfg(task_queue_backend="rabbitmq"))


def test_resolve_legacy_queue_backend_shim_warns():
    cfg = _cfg(queue_backend="kafka")
    with pytest.warns(DeprecationWarning, match="QUEUE_BACKEND is deprecated"):
        choice, explicit = resolve_task_queue_backend(cfg)
    assert choice == "kafka"
    assert explicit is False  # legacy alias keeps graceful-fallback semantics


def test_create_task_queue_auto_sqlite(tmp_path):
    queue, name = create_task_queue(_cfg(), str(tmp_path / "q.db"))
    assert name == "sqlite"
    queue.put({"job_id": "j1", "props": {}})
    assert queue.get_queue_size() == 1


def test_create_task_queue_auto_postgres_degrades_gracefully(tmp_path, monkeypatch):
    """Auto mode (STATE_DB_URI set, unreachable) keeps the SQLite fallback."""
    from agent_utilities.knowledge_graph.core import postgres_queue_backend

    class _Boom:
        def __init__(self, *a, **kw):
            raise ConnectionError("state store down")

    monkeypatch.setattr(postgres_queue_backend, "PostgresTaskQueue", _Boom)
    cfg = _cfg(state_db_uri="postgresql://down/now")
    queue, name = create_task_queue(cfg, str(tmp_path / "q.db"))
    assert name == "sqlite"


def test_create_task_queue_explicit_postgres_fails_loud(tmp_path, monkeypatch):
    from agent_utilities.knowledge_graph.core import postgres_queue_backend

    class _Boom:
        def __init__(self, *a, **kw):
            raise ConnectionError("state store down")

    monkeypatch.setattr(postgres_queue_backend, "PostgresTaskQueue", _Boom)
    cfg = _cfg(task_queue_backend="postgres", state_db_uri="postgresql://down/now")
    with pytest.raises(TaskQueueUnavailable, match="postgresql://down/now"):
        create_task_queue(cfg, str(tmp_path / "q.db"))


# ── confluent_kafka stub (no install / no broker needed) ────────────────


class _FakeFuture:
    def __init__(self, exc=None):
        self._exc = exc

    def result(self, timeout=None):
        if self._exc:
            raise self._exc
        return None


class _FakeTopicMeta:
    def __init__(self, partitions: int):
        self.partitions = {i: object() for i in range(partitions)}
        self.error = None


class _FakeAdmin:
    """confluent AdminClient fake: in-memory topic → partition-count map."""

    def __init__(self, topics: dict[str, int] | None = None, down: bool = False):
        self.topics = dict(topics or {})
        self.down = down
        self.created: list[tuple[str, int]] = []
        self.grown: list[tuple[str, int]] = []

    def list_topics(self, topic=None, timeout=None):
        if self.down:
            raise ConnectionError("broker unreachable")
        topics = {n: _FakeTopicMeta(p) for n, p in self.topics.items()}
        if topic is not None:
            topics = {k: v for k, v in topics.items() if k == topic}
        return SimpleNamespace(topics=topics)

    def create_topics(self, new_topics):
        out = {}
        for nt in new_topics:
            self.topics[nt.topic] = nt.num_partitions
            self.created.append((nt.topic, nt.num_partitions))
            out[nt.topic] = _FakeFuture()
        return out

    def create_partitions(self, new_partitions):
        out = {}
        for np_ in new_partitions:
            self.topics[np_.topic] = np_.new_total_count
            self.grown.append((np_.topic, np_.new_total_count))
            out[np_.topic] = _FakeFuture()
        return out


class _FakeProducer:
    def __init__(self):
        self.messages: list[tuple[str, bytes, bytes]] = []

    def produce(self, topic, value=None, key=None):
        self.messages.append((topic, value, key))

    def flush(self, timeout=None):
        return 0


@pytest.fixture
def confluent_stub(monkeypatch):
    """Install importable ``confluent_kafka`` stub modules."""
    ck = types.ModuleType("confluent_kafka")

    class TopicPartition:
        def __init__(self, topic, partition, offset=-1001):
            self.topic, self.partition, self.offset = topic, partition, offset

    class KafkaException(Exception):
        pass

    ck.TopicPartition = TopicPartition
    ck.KafkaException = KafkaException
    ck.Producer = lambda conf: _FakeProducer()
    ck.Consumer = lambda conf: pytest.fail("tests must inject consumers")

    admin = types.ModuleType("confluent_kafka.admin")

    class NewTopic:
        def __init__(self, topic, num_partitions=1, replication_factor=1):
            self.topic, self.num_partitions = topic, num_partitions

    class NewPartitions:
        def __init__(self, topic, new_total_count):
            self.topic, self.new_total_count = topic, new_total_count

    admin.NewTopic = NewTopic
    admin.NewPartitions = NewPartitions
    admin.AdminClient = lambda conf: pytest.fail("tests must inject admin clients")
    ck.admin = admin

    monkeypatch.setitem(sys.modules, "confluent_kafka", ck)
    monkeypatch.setitem(sys.modules, "confluent_kafka.admin", admin)
    return ck


def _kafka_backend(admin, producer=None, **kw):
    from agent_utilities.knowledge_graph.core.kafka_queue_backend import (
        KafkaQueueBackend,
    )

    return KafkaQueueBackend(
        bootstrap_servers="broker.test:9092",
        producer=producer or _FakeProducer(),
        admin_client=admin,
        **kw,
    )


# ── KG-2.55: fail-loud vs graceful Kafka startup ─────────────────────────


def test_explicit_kafka_unreachable_raises_actionable(confluent_stub):
    with pytest.raises(TaskQueueUnavailable) as exc:
        _kafka_backend(_FakeAdmin(down=True), fail_loud=True)
    msg = str(exc.value)
    assert "broker.test:9092" in msg  # names the broker
    assert "TASK_QUEUE_BACKEND" in msg  # says how to fall back
    assert "sqlite" in msg


def test_legacy_kafka_unreachable_falls_back_to_sqlite(confluent_stub, tmp_path):
    backend = _kafka_backend(
        _FakeAdmin(down=True),
        fail_loud=False,
        fallback_db_path=str(tmp_path / "fb.db"),
    )
    backend.put({"job_id": "j1", "props": {}})
    assert backend.get_queue_size() == 1  # served by the SQLite fallback


def test_create_task_queue_explicit_kafka_fail_loud(confluent_stub, monkeypatch, tmp_path):
    """End-to-end through the factory: explicit kafka + dead broker raises."""
    from agent_utilities.knowledge_graph.core import kafka_queue_backend as kqb

    class _DeadAdminBackend(kqb.KafkaQueueBackend):
        def _admin_client(self):
            return _FakeAdmin(down=True)

    monkeypatch.setattr(kqb, "KafkaQueueBackend", _DeadAdminBackend)
    cfg = _cfg(task_queue_backend="kafka", kafka_bootstrap_servers="broker.test:9092")
    with pytest.raises(TaskQueueUnavailable, match="broker.test:9092"):
        create_task_queue(cfg, str(tmp_path / "q.db"))


# ── KG-2.56: idempotent topic provisioning (grow-only) ──────────────────


def test_ensure_topics_creates_missing(confluent_stub):
    admin = _FakeAdmin()
    _kafka_backend(admin, partitions=4)
    assert ("kg_tasks", 4) in admin.created
    assert ("kg_staging", 1) in admin.created


def test_ensure_topics_grows_but_never_shrinks(confluent_stub):
    admin = _FakeAdmin({"kg_tasks": 3, "kg_staging": 1})
    _kafka_backend(admin, partitions=8)
    assert ("kg_tasks", 8) in admin.grown  # 3 → 8 grows

    admin2 = _FakeAdmin({"kg_tasks": 12, "kg_staging": 1})
    _kafka_backend(admin2, partitions=8)
    assert admin2.grown == [] and admin2.created == []  # 12 stays 12


# ── KG-2.56: partition-key hierarchy ─────────────────────────────────────


def _envelope(job_id="job-1", *, full_path=None, target=None, task_type="document"):
    props: dict = {}
    if full_path:
        props["full_path"] = full_path
    meta = {}
    if target:
        meta["target"] = target
    meta["type"] = task_type
    props["metadata"] = _encode_metadata(meta)
    return {"job_id": job_id, "props": props}


def test_partition_key_tenant_wins():
    from agent_utilities.models.company_brain import ActorType
    from agent_utilities.security.brain_context import ActorContext, use_actor
    from agent_utilities.knowledge_graph.core.kafka_queue_backend import (
        partition_key_for,
    )

    env = _envelope(full_path="org/repo", target="/x/y.py")
    actor = ActorContext("u1", ActorType.HUMAN, tenant_id="acme")
    with use_actor(actor):
        assert partition_key_for(env) == "tenant:acme"


def test_partition_key_repo_provenance_then_corpus_then_type():
    from agent_utilities.knowledge_graph.core.kafka_queue_backend import (
        partition_key_for,
    )

    assert (
        partition_key_for(_envelope(full_path="group/repo", target="/a/b/c.py"))
        == "corpus:group/repo"
    )
    key = partition_key_for(
        _envelope(target="/home/apps/workspace/agent-packages/egeria-mcp/src/a.py")
    )
    assert key == "corpus:home/apps/workspace/agent-packages/egeria-mcp"
    assert (
        partition_key_for(_envelope(task_type="relevance_sweep"))
        == "type:relevance_sweep"
    )


def test_partition_key_same_repo_files_share_key():
    from agent_utilities.knowledge_graph.core.kafka_queue_backend import (
        partition_key_for,
    )

    k1 = partition_key_for(
        _envelope(target="/work/agent-packages/repo-a/src/one.py")
    )
    k2 = partition_key_for(
        _envelope(target="/work/agent-packages/repo-a/docs/two.md")
    )
    k3 = partition_key_for(
        _envelope(target="/work/agent-packages/repo-b/src/three.py")
    )
    assert k1 == k2 != k3


def test_put_produces_keyed_message(confluent_stub):
    producer = _FakeProducer()
    backend = _kafka_backend(
        _FakeAdmin({"kg_tasks": 6, "kg_staging": 1}), producer=producer
    )
    backend.put(_envelope(full_path="org/repo"))
    (topic, value, key) = producer.messages[0]
    assert topic == "kg_tasks"
    assert key == b"corpus:org/repo"
    assert json.loads(value)["job_id"] == "job-1"


# ── consumer lag / queue depth ───────────────────────────────────────────


class _FakeLagProbe:
    """committed() echoes stored offsets; watermarks fixed per partition."""

    def __init__(self, committed: dict[int, int], high: dict[int, int]):
        self._committed, self._high = committed, high

    def committed(self, tps, timeout=None):
        for tp in tps:
            tp.offset = self._committed.get(tp.partition, -1001)
        return tps

    def get_watermark_offsets(self, tp, timeout=None):
        return 0, self._high.get(tp.partition, 0)


def test_consumer_lag_sums_partitions(confluent_stub):
    backend = _kafka_backend(
        _FakeAdmin({"kg_tasks": 2, "kg_staging": 1}),
        consumer_factory=lambda **kw: _FakeLagProbe({0: 5, 1: -1001}, {0: 9, 1: 7}),
    )
    # partition 0: 9-5=4; partition 1: never committed → 7-0=7
    assert backend.consumer_lag() == 11
    assert backend.get_queue_size() == 11


# ── KG-2.57: idempotent claim + consumer loop ────────────────────────────


class _FakeEngine:
    """Just enough TaskManagerMixin surface for the worker path."""

    def __init__(self):
        self.nodes: dict[str, dict] = {}
        self.executed: list[tuple] = []
        self.failed: dict[str, dict] = {}

    def query_cypher(self, q, params=None):
        node = self.nodes.get((params or {}).get("id"))
        return [{"s": node["status"]}] if node else []

    def add_node(self, node_id, node_type, properties=None):
        self.nodes[node_id] = dict(properties or {})

    def _get_host_token(self):
        return "testhost:1:1"

    def _update_task_status(self, job_id, status, meta=None):
        self.nodes.setdefault(job_id, {})["status"] = status
        if status == "failed":
            self.failed[job_id] = meta or {}

    def _execute_claimed_task(self, job_id, target, is_codebase, task_type):
        self.executed.append((job_id, target, is_codebase, task_type))
        self.nodes[job_id]["status"] = "completed"


def test_claim_marks_running_with_ownership_stamp():
    from agent_utilities.knowledge_graph.ingest_worker import claim_task_envelope

    engine = _FakeEngine()
    claimed = claim_task_envelope(
        engine, _envelope(target="/repo/file.py", task_type="codebase")
    )
    assert claimed == ("job-1", Path("/repo/file.py"), True, "codebase")
    node = engine.nodes["job-1"]
    assert node["status"] == "running"
    meta = _decode_metadata(node["metadata"])
    assert meta["claimed_by"] == "testhost:1:1"
    assert meta["claim_unix"] > 0


@pytest.mark.parametrize("status", ["running", "completed", "failed", "cancelled"])
def test_claim_skips_duplicate_delivery(status):
    from agent_utilities.knowledge_graph.ingest_worker import claim_task_envelope

    engine = _FakeEngine()
    engine.nodes["job-1"] = {"status": status}
    assert claim_task_envelope(engine, _envelope(target="/x.py")) is None
    assert engine.nodes["job-1"]["status"] == status  # untouched


def test_claim_reclaims_reaper_requeued_pending():
    from agent_utilities.knowledge_graph.ingest_worker import claim_task_envelope

    engine = _FakeEngine()
    engine.nodes["job-1"] = {"status": "pending"}
    assert claim_task_envelope(engine, _envelope(target="/x.py")) is not None


def test_claim_missing_target_fails_task():
    from agent_utilities.knowledge_graph.ingest_worker import claim_task_envelope

    engine = _FakeEngine()
    assert claim_task_envelope(engine, {"job_id": "j2", "props": {}}) is None
    assert engine.nodes["j2"]["status"] == "failed"


class _FakeMsg:
    def __init__(self, payload):
        self._value = json.dumps(payload).encode("utf-8")

    def value(self):
        return self._value

    def error(self):
        return None


class _FakeConsumer:
    def __init__(self, messages, stop_event, on_empty=None):
        self._messages = list(messages)
        self._stop = stop_event
        self._on_empty = on_empty or self._stop.set
        self.commits: list = []
        self.closed = False

    def poll(self, timeout=None):
        if not self._messages:
            self._on_empty()
            return None
        return self._messages.pop(0)

    def commit(self, message=None, asynchronous=False):
        self.commits.append(message)

    def close(self):
        self.closed = True


def test_consumer_loop_processes_and_commits_after():
    from agent_utilities.knowledge_graph.ingest_worker import run_ingest_consumer_loop

    engine = _FakeEngine()
    stop = threading.Event()
    msgs = [_FakeMsg(_envelope("job-A", target="/a.py"))]
    consumer = _FakeConsumer(msgs, stop)
    run_ingest_consumer_loop(engine, consumer, stop)
    assert [e[0] for e in engine.executed] == ["job-A"]
    assert len(consumer.commits) == 1  # committed AFTER processing


def test_consumer_loop_marks_failed_and_still_commits():
    from agent_utilities.knowledge_graph.ingest_worker import run_ingest_consumer_loop

    engine = _FakeEngine()

    def _boom(job_id, *a):
        raise RuntimeError("parse exploded")

    engine._execute_claimed_task = _boom
    stop = threading.Event()
    consumer = _FakeConsumer([_FakeMsg(_envelope("job-B", target="/b.py"))], stop)
    run_ingest_consumer_loop(engine, consumer, stop)
    assert engine.nodes["job-B"]["status"] == "failed"
    assert "parse exploded" in engine.failed["job-B"]["error"]
    assert len(consumer.commits) == 1  # poison message is not redelivered forever


def test_consumer_pool_spreads_messages_and_closes():
    from agent_utilities.knowledge_graph.ingest_worker import (
        start_ingest_consumer_pool,
    )

    engine = _FakeEngine()
    stop = threading.Event()
    consumers: list[_FakeConsumer] = []
    batches = [
        [_FakeMsg(_envelope("job-1", target="/1.py"))],
        [_FakeMsg(_envelope("job-2", target="/2.py"))],
    ]

    def _maybe_stop():
        # Only stop the shared event once BOTH jobs ran (avoids one fast
        # consumer stopping the pool before its sibling polls).
        if len(engine.executed) >= 2:
            stop.set()

    def factory():
        c = _FakeConsumer(batches.pop(0), stop, on_empty=_maybe_stop)
        consumers.append(c)
        return c

    threads = start_ingest_consumer_pool(
        engine, worker_count=2, stop_event=stop, consumer_factory=factory
    )
    for t in threads:
        t.join(timeout=5.0)
    assert sorted(e[0] for e in engine.executed) == ["job-1", "job-2"]
    assert all(c.closed for c in consumers)


# ── autosizing ───────────────────────────────────────────────────────────


def test_worker_count_configured_wins():
    assert compute_ingest_worker_count(7) == 7


def test_worker_count_autosizes_with_floor():
    assert compute_ingest_worker_count(0) >= 2


# ── KG-2.57: uniform queue depth + orchestrator backpressure ────────────


class _DepthHarness:
    ingest_queue_depth = TaskManagerMixin.ingest_queue_depth

    def __init__(self, qsize, task_rows):
        self._submission_queue = SimpleNamespace(get_queue_size=lambda: qsize)
        self._rows = task_rows

    def query_cypher(self, q, params=None):
        return self._rows


def test_ingest_queue_depth_combines_queue_and_tasks():
    h = _DepthHarness(qsize=4, task_rows=[{"c": 3}])
    assert h.ingest_queue_depth() == 7


def test_batch_orchestrator_prefers_uniform_depth():
    from agent_utilities.knowledge_graph.ingestion.batch_orchestrator import (
        _inflight_count,
    )

    engine = SimpleNamespace(ingest_queue_depth=lambda: 42)
    assert _inflight_count(engine) == 42

    # Engines without the method fall back to the :Task count query.
    legacy = SimpleNamespace(query_cypher=lambda q: [{"c": 5}])
    assert _inflight_count(legacy) == 5


# ── OS-5.23 metrics surface ──────────────────────────────────────────────


def test_ingest_metrics_registered_on_gateway_registry():
    from agent_utilities.observability import gateway_metrics as gm

    assert "KG_INGEST_QUEUE_DEPTH" in gm.__all__
    assert "KG_INGEST_CONSUMER_LAG" in gm.__all__
    # Always usable, regardless of whether prometheus_client is installed.
    gm.KG_INGEST_QUEUE_DEPTH.labels(backend="kafka").set(3.0)
    gm.KG_INGEST_CONSUMER_LAG.labels(topic="kg_tasks", group="kg-ingest").set(3.0)


def test_record_queue_telemetry_sets_gauges(monkeypatch):
    from agent_utilities.observability import gateway_metrics as gm

    seen: dict[str, float] = {}

    class _Gauge:
        def __init__(self, name):
            self._name = name

        def labels(self, **labels):
            self._labels = labels
            return self

        def set(self, value):
            seen[f"{self._name}{sorted(self._labels.items())}"] = value

    monkeypatch.setattr(gm, "KG_INGEST_QUEUE_DEPTH", _Gauge("depth"))
    monkeypatch.setattr(gm, "KG_INGEST_CONSUMER_LAG", _Gauge("lag"))

    class _Harness:
        _record_queue_telemetry = TaskManagerMixin._record_queue_telemetry
        _task_queue_backend_name = "kafka"

    _Harness()._record_queue_telemetry(17)
    assert seen["depth[('backend', 'kafka')]"] == 17.0
    assert seen["lag[('group', 'kg-ingest'), ('topic', 'kg_tasks')]"] == 17.0


# ── optional live-broker test ────────────────────────────────────────────


@pytest.mark.skipif(
    not os.environ.get("AGENT_UTILITIES_KAFKA_LIVE"),
    reason="set AGENT_UTILITIES_KAFKA_LIVE=broker:9092 to run against live Kafka",
)
def test_live_kafka_roundtrip():
    pytest.importorskip("confluent_kafka")
    from agent_utilities.knowledge_graph.core.kafka_queue_backend import (
        KafkaQueueBackend,
    )

    servers = os.environ["AGENT_UTILITIES_KAFKA_LIVE"]
    backend = KafkaQueueBackend(
        bootstrap_servers=servers, fail_loud=True, partitions=2
    )
    job_id = f"live-{int(time.time())}"
    backend.put(_envelope(job_id, target="/tmp/live.txt"))
    deadline = time.time() + 30
    got = None
    while time.time() < deadline and got is None:
        item = backend.get()
        if item and item[1].get("job_id") == job_id:
            got = item
    assert got is not None
    backend.ack(got[0])
