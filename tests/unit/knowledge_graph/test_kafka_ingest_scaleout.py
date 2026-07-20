"""Tranche-3 ingest scale-out tests (CONCEPT:AU-KG.backend.selectable-queue-backend / KG-2.56 / KG-2.57).

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
    assert resolve_task_queue_backend(_cfg()) == "sqlite"


def test_resolve_auto_prefers_postgres_with_state_db_uri():
    cfg = _cfg(state_db_uri="postgresql://x/y")
    assert resolve_task_queue_backend(cfg) == "postgres"


def test_resolve_explicit_wins_over_state_db_uri():
    cfg = _cfg(task_queue_backend="sqlite", state_db_uri="postgresql://x/y")
    assert resolve_task_queue_backend(cfg) == "sqlite"


@pytest.mark.parametrize("value", ["kafka", "postgres", "sqlite", " Kafka "])
def test_resolve_explicit_values(value):
    assert (
        resolve_task_queue_backend(_cfg(task_queue_backend=value))
        == value.strip().lower()
    )


def test_resolve_rejects_unknown_value():
    with pytest.raises(ValueError, match="TASK_QUEUE_BACKEND"):
        resolve_task_queue_backend(_cfg(task_queue_backend="rabbitmq"))


def test_create_task_queue_auto_sqlite(tmp_path):
    queue, name = create_task_queue(_cfg(), str(tmp_path / "q.db"))
    assert name == "sqlite"
    queue.put({"job_id": "j1", "props": {}})
    assert queue.get_queue_size() == 1


def test_create_task_queue_auto_postgres_fails_closed(tmp_path, monkeypatch):
    """A configured external authority never switches to a local queue."""
    from agent_utilities.knowledge_graph.core import postgres_queue_backend

    class _Boom:
        def __init__(self, *a, **kw):
            raise ConnectionError("state store down")

    monkeypatch.setattr(postgres_queue_backend, "PostgresTaskQueue", _Boom)
    cfg = _cfg(state_db_uri="postgresql://down/now")
    with pytest.raises(TaskQueueUnavailable) as exc_info:
        create_task_queue(cfg, str(tmp_path / "q.db"))
    assert "postgresql://down/now" not in str(exc_info.value)


def test_create_task_queue_explicit_postgres_fails_loud(tmp_path, monkeypatch):
    from agent_utilities.knowledge_graph.core import postgres_queue_backend

    class _Boom:
        def __init__(self, *a, **kw):
            raise ConnectionError("state store down")

    monkeypatch.setattr(postgres_queue_backend, "PostgresTaskQueue", _Boom)
    cfg = _cfg(task_queue_backend="postgres", state_db_uri="postgresql://down/now")
    with pytest.raises(TaskQueueUnavailable) as exc_info:
        create_task_queue(cfg, str(tmp_path / "q.db"))
    assert "postgresql://down/now" not in str(exc_info.value)


# ── confluent_kafka test double (no install / no broker needed) ────────────────


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
        self.callbacks: list = []
        self.flush_calls = 0

    def produce(self, topic, value=None, key=None, on_delivery=None):
        self.messages.append((topic, value, key))
        if on_delivery is not None:
            self.callbacks.append(on_delivery)

    def flush(self, timeout=None):
        self.flush_calls += 1
        callbacks, self.callbacks = self.callbacks, []
        for callback in callbacks:
            callback(None, SimpleNamespace())
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


def test_kafka_unreachable_fails_without_exposing_endpoint(confluent_stub):
    with pytest.raises(TaskQueueUnavailable) as exc:
        _kafka_backend(_FakeAdmin(down=True))
    msg = str(exc.value)
    assert "broker.test:9092" not in msg
    assert "sqlite" not in msg.casefold()


def test_create_task_queue_explicit_kafka_fail_loud(
    confluent_stub, monkeypatch, tmp_path
):
    """End-to-end through the factory: explicit kafka + dead broker raises."""
    from agent_utilities.knowledge_graph.core import kafka_queue_backend as kqb

    class _DeadAdminBackend(kqb.KafkaQueueBackend):
        def _admin_client(self):
            return _FakeAdmin(down=True)

    monkeypatch.setattr(kqb, "KafkaQueueBackend", _DeadAdminBackend)
    cfg = _cfg(task_queue_backend="kafka", kafka_bootstrap_servers="broker.test:9092")
    with pytest.raises(TaskQueueUnavailable) as exc_info:
        create_task_queue(cfg, str(tmp_path / "q.db"))
    assert "broker.test:9092" not in str(exc_info.value)


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
    from agent_utilities.knowledge_graph.core.kafka_queue_backend import (
        partition_key_for,
    )
    from agent_utilities.models.company_brain import ActorType
    from agent_utilities.security.brain_context import ActorContext, use_actor

    env = _envelope(full_path="org/repo", target="/x/y.py")
    actor = ActorContext("u1", ActorType.HUMAN, tenant_id="acme")
    with use_actor(actor):
        key = partition_key_for(env)
    assert key.startswith("tenant:pref_tenant_")
    assert "acme" not in key


def test_partition_key_repo_provenance_then_corpus_then_type(monkeypatch):
    from agent_utilities.knowledge_graph.core.kafka_queue_backend import (
        partition_key_for,
    )
    from agent_utilities.security import brain_context

    monkeypatch.setattr(
        brain_context, "current_actor", lambda: SimpleNamespace(tenant_id="")
    )

    provenance_key = partition_key_for(
        _envelope(full_path="group/repo", target="/a/b/c.py")
    )
    assert provenance_key.startswith("corpus:pref_corpus_")
    assert "group/repo" not in provenance_key
    key = partition_key_for(
        _envelope(
            target="/home/agent-user/workspace/agent-packages/egeria-mcp/src/a.py"
        )
    )
    assert key.startswith("corpus:pref_corpus_")
    assert "egeria-mcp" not in key
    type_key = partition_key_for(_envelope(task_type="relevance_sweep"))
    assert type_key.startswith("type:pref_task_type_")
    assert "relevance_sweep" not in type_key


def test_partition_key_same_repo_files_share_key(monkeypatch):
    from agent_utilities.knowledge_graph.core.kafka_queue_backend import (
        partition_key_for,
    )
    from agent_utilities.security import brain_context

    monkeypatch.setattr(
        brain_context, "current_actor", lambda: SimpleNamespace(tenant_id="")
    )

    k1 = partition_key_for(_envelope(target="/work/agent-packages/repo-a/src/one.py"))
    k2 = partition_key_for(_envelope(target="/work/agent-packages/repo-a/docs/two.md"))
    k3 = partition_key_for(_envelope(target="/work/agent-packages/repo-b/src/three.py"))
    assert k1 == k2 != k3


def test_put_produces_keyed_message(confluent_stub):
    producer = _FakeProducer()
    backend = _kafka_backend(
        _FakeAdmin({"kg_tasks": 6, "kg_staging": 1}), producer=producer
    )
    backend.put(_envelope(full_path="org/repo"))
    (topic, value, key) = producer.messages[0]
    assert topic == "kg_tasks"
    assert key.startswith(b"corpus:pref_corpus_")
    assert b"org/repo" not in key
    assert json.loads(value)["job_id"] == "job-1"


def test_put_many_uses_one_delivery_confirmed_flush(confluent_stub):
    producer = _FakeProducer()
    backend = _kafka_backend(
        _FakeAdmin({"kg_tasks": 6, "kg_staging": 1}), producer=producer
    )
    backend.put_many([_envelope(f"job-{index}") for index in range(3)])
    assert len(producer.messages) == 3
    assert producer.flush_calls == 1


def test_concurrent_puts_coalesce_behind_one_flush(confluent_stub, monkeypatch):
    from agent_utilities.knowledge_graph.core import kafka_queue_backend as kqb

    monkeypatch.setattr(kqb, "_PRODUCER_BATCH_LINGER_S", 0.1)
    producer = _FakeProducer()
    backend = _kafka_backend(
        _FakeAdmin({"kg_tasks": 6, "kg_staging": 1}), producer=producer
    )
    barrier = threading.Barrier(2)

    def publish(index: int) -> None:
        barrier.wait()
        backend.put(_envelope(f"job-{index}"))

    threads = [threading.Thread(target=publish, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2.0)

    assert all(not thread.is_alive() for thread in threads)
    assert len(producer.messages) == 2
    assert producer.flush_calls == 1


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


def test_kafka_admission_rejects_at_lag_bound(confluent_stub):
    producer = _FakeProducer()
    backend = _kafka_backend(
        _FakeAdmin({"kg_tasks": 1, "kg_staging": 1}),
        producer=producer,
        consumer_factory=lambda **kw: _FakeLagProbe({0: 0}, {0: 5}),
    )
    assert not backend.put_if_below(_envelope("rejected"), 5)
    assert producer.messages == []
    assert backend.put_if_below(_envelope("accepted"), 6)
    assert len(producer.messages) == 1


# ── KG-2.57: idempotent claim + consumer loop ────────────────────────────


class _FakeEngine:
    """Native WorkItem surface for the decoupled worker path."""

    def __init__(self):
        self.nodes: dict[str, dict] = {}
        self.executed: list[tuple] = []
        self.failed: dict[str, dict] = {}
        self.claims: dict[str, dict] = {}
        self._work_item_engine = self

    def query_cypher(self, q, params=None):
        node = self.nodes.get((params or {}).get("id"))
        if not node:
            return []
        from agent_utilities.orchestration import work_item as wi

        return [
            {
                "id": (params or {})["id"],
                **{field: node.get(field) for field in wi._FIELDS},
            }
        ]

    def add_node(self, node_id, node_type, properties=None):
        self.nodes[node_id] = dict(properties or {})

    def _get_host_token(self):
        return "worker-opaque"

    def claim_work_item(self, request):
        item_id = request.work_item_id
        node = self.nodes.get(item_id)
        if not node or node.get("status") != "ready":
            return {
                "schema_version": "1",
                "claimed": False,
                "reason": "empty",
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
        node.update(
            status="leased",
            lease_owner=request.worker_ref,
            lease_epoch=1,
            fencing_token=1,
        )
        return {
            "schema_version": "1",
            "claimed": True,
            "reason": "claimed",
            "work_item_id": item_id,
            "kind": node.get("kind"),
            "payload_ref": node["payload_ref"],
            "lease_holder_ref": request.worker_ref,
            "lease_epoch": 1,
            "fencing_token": 1,
            "lease_expires_at_ms": request.now_ms + request.lease_ms,
            "attempt": 1,
            "max_attempts": 3,
            "tenant_in_flight": 1,
            "changed_work_item_ids": [item_id],
        }

    def _ingest_task_metadata(self, job_id):
        return dict(self.nodes[f"workitem:ingest_task:{job_id}"]["metadata"])

    def _remember_work_item_claim(self, job_id, claim):
        self.claims[job_id] = dict(claim)

    def _update_task_status(self, job_id, status, meta=None):
        self.nodes[f"workitem:ingest_task:{job_id}"]["status"] = status
        if status == "failed":
            self.failed[job_id] = meta or {}

    def _execute_claimed_task(self, job_id, target, is_codebase, task_type):
        self.executed.append((job_id, target, is_codebase, task_type))
        self.nodes[f"workitem:ingest_task:{job_id}"]["status"] = "completed"


def _seed_work_item(
    engine, job_id="job-1", *, target=None, task_type="document", status="ready"
):
    engine.nodes[f"workitem:ingest_task:{job_id}"] = {
        "status": status,
        "kind": "ingest_task",
        "queue": "ingest_task",
        "tenant": "tenant-a",
        "payload_ref": job_id,
        "resource_class": "ingestion",
        "prio_bucket": 2,
        "max_attempts": 3,
        "metadata": {"target": target, "type": task_type},
    }


def test_claim_marks_running_with_ownership_stamp():
    from agent_utilities.knowledge_graph.ingest_worker import claim_task_envelope

    engine = _FakeEngine()
    _seed_work_item(engine, target="repo/file.py", task_type="codebase")
    claimed = claim_task_envelope(engine, {"job_id": "job-1"})
    assert claimed == ("job-1", Path("repo/file.py"), True, "codebase")
    assert engine.nodes["workitem:ingest_task:job-1"]["status"] == "leased"
    assert engine.claims["job-1"]["fencing_token"] == 1


@pytest.mark.parametrize(
    "status", ["leased", "running", "succeeded", "failed", "cancelled"]
)
def test_claim_skips_duplicate_delivery(status):
    from agent_utilities.knowledge_graph.ingest_worker import claim_task_envelope

    engine = _FakeEngine()
    _seed_work_item(engine, target="x.py", status=status)
    assert claim_task_envelope(engine, {"job_id": "job-1"}) is None
    assert engine.nodes["workitem:ingest_task:job-1"]["status"] == status


def test_claim_reclaims_reaper_requeued_pending():
    from agent_utilities.knowledge_graph.ingest_worker import claim_task_envelope

    engine = _FakeEngine()
    _seed_work_item(engine, target="x.py")
    assert claim_task_envelope(engine, {"job_id": "job-1"}) is not None


def test_claim_missing_target_fails_task():
    from agent_utilities.knowledge_graph.ingest_worker import claim_task_envelope

    engine = _FakeEngine()
    _seed_work_item(engine, "j2")
    assert claim_task_envelope(engine, {"job_id": "j2"}) is None
    assert engine.nodes["workitem:ingest_task:j2"]["status"] == "failed"


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


def test_backend_consumer_is_owned_per_worker_thread(confluent_stub):
    stop = threading.Event()
    consumers: list[_FakeConsumer] = []
    factory_lock = threading.Lock()

    def factory(**_kwargs):
        with factory_lock:
            index = len(consumers)
            consumer = _FakeConsumer([_FakeMsg({"job_id": f"thread-{index}"})], stop)
            consumers.append(consumer)
            return consumer

    backend = _kafka_backend(
        _FakeAdmin({"kg_tasks": 2, "kg_staging": 1}),
        consumer_factory=factory,
    )
    barrier = threading.Barrier(2)
    payloads: list[str] = []

    def consume() -> None:
        barrier.wait()
        item = backend.get()
        assert item is not None
        receipt, payload = item
        payloads.append(payload["job_id"])
        backend.ack(receipt)

    threads = [threading.Thread(target=consume) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2.0)

    assert sorted(payloads) == ["thread-0", "thread-1"]
    assert len(consumers) == 2
    assert all(len(consumer.commits) == 1 for consumer in consumers)


def test_consumer_loop_processes_and_commits_after():
    from agent_utilities.knowledge_graph.ingest_worker import run_ingest_consumer_loop

    engine = _FakeEngine()
    _seed_work_item(engine, "job-A", target="a.py")
    stop = threading.Event()
    msgs = [_FakeMsg({"job_id": "job-A"})]
    consumer = _FakeConsumer(msgs, stop)
    run_ingest_consumer_loop(engine, consumer, stop)
    assert [e[0] for e in engine.executed] == ["job-A"]
    assert len(consumer.commits) == 1  # committed AFTER processing


def test_consumer_loop_marks_failed_and_still_commits():
    from agent_utilities.knowledge_graph.ingest_worker import run_ingest_consumer_loop

    engine = _FakeEngine()
    _seed_work_item(engine, "job-B", target="b.py")

    def _boom(job_id, *a):
        raise RuntimeError("parse exploded")

    engine._execute_claimed_task = _boom
    stop = threading.Event()
    consumer = _FakeConsumer([_FakeMsg({"job_id": "job-B"})], stop)
    run_ingest_consumer_loop(engine, consumer, stop)
    assert engine.nodes["workitem:ingest_task:job-B"]["status"] == "failed"
    assert "parse exploded" in engine.failed["job-B"]["error"]
    assert len(consumer.commits) == 1  # poison message is not redelivered forever


def test_consumer_pool_spreads_messages_and_closes():
    from agent_utilities.knowledge_graph.core.session import GraphSession
    from agent_utilities.knowledge_graph.ingest_worker import (
        start_ingest_consumer_pool,
    )
    from agent_utilities.models.company_brain import ActorType
    from agent_utilities.security.brain_context import ActorContext

    engine = _FakeEngine()
    _seed_work_item(engine, "job-1", target="1.py")
    _seed_work_item(engine, "job-2", target="2.py")
    stop = threading.Event()
    consumers: list[_FakeConsumer] = []
    batches = [
        [_FakeMsg({"job_id": "job-1"})],
        [_FakeMsg({"job_id": "job-2"})],
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

    actor = ActorContext(
        actor_id="test-worker",
        actor_type=ActorType.SYSTEM,
        roles=("system",),
        tenant_id="test-tenant",
        authenticated=True,
    )
    session = GraphSession(
        actor=actor,
        tenant=actor.tenant_id,
        scopes=frozenset({"kg:admin"}),
        policy_version="current",
        audience="test-runtime",
    )
    threads = start_ingest_consumer_pool(
        engine,
        worker_count=2,
        stop_event=stop,
        consumer_factory=factory,
        background_session=session,
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

    def _ingest_work_item_index(self):
        return {
            f"job-{index}": {"status": "ready"} for index, _row in enumerate(self._rows)
        }


def test_ingest_queue_depth_counts_authoritative_workitems_once():
    h = _DepthHarness(qsize=4, task_rows=[{}, {}, {}])
    assert h.ingest_queue_depth() == 3


def test_batch_orchestrator_prefers_uniform_depth():
    from agent_utilities.knowledge_graph.ingestion.batch_orchestrator import (
        _inflight_count,
    )

    engine = SimpleNamespace(ingest_queue_depth=lambda: 42)
    assert _inflight_count(engine) == 42

    assert _inflight_count(SimpleNamespace()) is None


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
    backend = KafkaQueueBackend(bootstrap_servers=servers, partitions=2)
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
