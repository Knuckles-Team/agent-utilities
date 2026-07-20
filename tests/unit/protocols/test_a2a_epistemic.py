"""Focused adversarial contracts for native durable FastA2A persistence."""

from __future__ import annotations

import asyncio
import inspect
import json
from types import SimpleNamespace
from typing import Any

import pytest
from fasta2a.schema import Artifact, Message, Task
from pydantic_ai.messages import ModelRequest, UserPromptPart

from agent_utilities.knowledge_graph.core.session import GraphSession
from agent_utilities.models.company_brain import ActorType
from agent_utilities.protocols.a2a_epistemic import (
    _DELIVERY_CONTROL,
    A2AStorageConflict,
    EpistemicGraphA2ABroker,
    EpistemicGraphA2ARuntime,
    EpistemicGraphA2AStorage,
    EpistemicGraphAgentWorker,
    _payload_ref,
)
from agent_utilities.security import persistence_privacy
from agent_utilities.security.brain_context import ActorContext


@pytest.fixture(autouse=True)
def _stable_test_reference_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        persistence_privacy, "_persistence_identity_key", lambda: b"unit-test-key"
    )


class FakeNodes:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _copy(value: Any) -> Any:
        return json.loads(json.dumps(value))

    def create_if_absent(self, node_id: str, properties: dict[str, Any]) -> bool:
        if node_id in self.rows:
            return False
        self.rows[node_id] = self._copy(properties)
        return True

    def properties(self, node_id: str) -> dict[str, Any] | None:
        value = self.rows.get(node_id)
        return None if value is None else self._copy(value)

    def compare_and_set(
        self, node_id: str, conditions: dict[str, Any], updates: dict[str, Any]
    ) -> bool:
        row = self.rows.get(node_id)
        if row is None or any(
            row.get(key) != value for key, value in conditions.items()
        ):
            return False
        row.update(self._copy(updates))
        return True

    def list_by_label(
        self, label: str, limit: int = 0, *, after: str | None = None
    ) -> list[tuple[str, dict[str, Any]]]:
        rows = [
            (node_id, self._copy(properties))
            for node_id, properties in sorted(self.rows.items())
            if properties.get("node_type") == label
            and (after is None or node_id > after)
        ]
        return rows[:limit] if limit else rows


class FakeTxn:
    def __init__(self, nodes: FakeNodes) -> None:
        self.nodes = nodes
        self.sequence = 0
        self.pending: dict[str, list[tuple[str, dict[str, Any], dict[str, Any]]]] = {}

    def begin(self) -> str:
        self.sequence += 1
        txn_id = f"txn-{self.sequence}"
        self.pending[txn_id] = []
        return txn_id

    def cas(
        self,
        txn_id: str,
        node_id: str,
        conditions: dict[str, Any],
        updates: dict[str, Any],
        graph: str | None = None,
    ) -> bool:
        assert graph is None
        self.pending[txn_id].append((node_id, conditions, updates))
        return True

    def commit(self, txn_id: str) -> bool:
        staged = self.pending.pop(txn_id)
        scratch = FakeNodes._copy(self.nodes.rows)
        for node_id, conditions, updates in staged:
            row = scratch.get(node_id)
            if row is None or any(
                row.get(key) != value for key, value in conditions.items()
            ):
                return False
            row.update(FakeNodes._copy(updates))
        self.nodes.rows = scratch
        return True

    def rollback(self, txn_id: str) -> bool:
        self.pending.pop(txn_id, None)
        return True


class FakeBroker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.messages: list[dict[str, Any]] = []
        self.high_water: dict[str, int] = {}
        self.acks: list[int] = []
        self.nacks: list[tuple[int, bool]] = []
        self.renewals: list[tuple[int, str]] = []
        self.renew_allowed = True
        self.sequence = 0
        self.delivery_tag = 0
        self.max_delivery_count = 5

    def _record(self, name: str, *args: Any, **kwargs: Any) -> None:
        self.calls.append((name, args, kwargs))

    def declare_exchange(self, exchange: str, kind: str = "direct") -> str:
        self._record("declare_exchange", exchange, kind=kind)
        return "ok"

    def declare_queue(self, queue: str, **policy: Any) -> str:
        self._record("declare_queue", queue, **policy)
        if policy.get("max_delivery_count") is not None:
            self.max_delivery_count = int(policy["max_delivery_count"])
        return "ok"

    def bind_queue(self, exchange: str, queue: str, routing_key: str) -> str:
        self._record("bind_queue", exchange, queue, routing_key)
        return "ok"

    def publish_idempotent(
        self,
        exchange: str,
        routing_key: str,
        payload: bytes,
        *,
        producer_id: str | None = None,
        seq: int = 0,
        priority: int = 0,
        delay_ms: int | None = None,
        ttl_ms: int | None = None,
        now_ms: int | None = None,
    ) -> dict[str, Any]:
        del priority, delay_ms, ttl_ms, now_ms
        self._record(
            "publish_idempotent",
            exchange,
            routing_key,
            payload,
            producer_id=producer_id,
            seq=seq,
        )
        assert producer_id is not None
        if seq <= self.high_water.get(producer_id, -1):
            return {"confirmed": True, "duplicate": True, "delivered": 0}
        self.high_water[producer_id] = seq
        self.sequence += 1
        self.messages.append(
            {
                "node_id": f"delivery-{self.sequence}",
                "payload": payload.hex(),
                "status": "pending",
                "delivery_tag": None,
                "owner_consumer": None,
                "lease_until": None,
                "delivery_count": 0,
            }
        )
        return {"confirmed": True, "duplicate": False, "delivered": 1}

    def consume(
        self,
        queue: str,
        *,
        group: str,
        consumer: str,
        now_ms: int,
        lease_ms: int = 0,
        prefetch: int = 0,
    ) -> tuple[str, dict[str, Any]] | None:
        self._record(
            "consume",
            queue,
            group=group,
            consumer=consumer,
            now_ms=now_ms,
            lease_ms=lease_ms,
            prefetch=prefetch,
        )
        for message in self.messages:
            if (
                message["status"] == "claimed"
                and int(message["lease_until"] or 0) > now_ms
            ):
                continue
            self.delivery_tag += 1
            message.update(
                {
                    "status": "claimed",
                    "delivery_tag": self.delivery_tag,
                    "owner_consumer": consumer,
                    "lease_until": now_ms + lease_ms,
                    "delivery_count": int(message["delivery_count"]) + 1,
                }
            )
            return message["node_id"], FakeNodes._copy(message)
        return None

    def renew_tag(
        self,
        delivery_tag: int,
        *,
        consumer: str,
        now_ms: int,
        lease_ms: int,
    ) -> bool:
        self.renewals.append((delivery_tag, consumer))
        if not self.renew_allowed:
            return False
        for message in self.messages:
            if (
                message["status"] == "claimed"
                and message["delivery_tag"] == delivery_tag
                and message["owner_consumer"] == consumer
                and int(message["lease_until"] or 0) > now_ms
            ):
                message["lease_until"] = now_ms + lease_ms
                return True
        return False

    def ack_tag(self, delivery_tag: int, *, consumer: str) -> bool:
        for index, message in enumerate(self.messages):
            if (
                message["delivery_tag"] == delivery_tag
                and message["status"] == "claimed"
                and message["owner_consumer"] == consumer
            ):
                self.acks.append(delivery_tag)
                self.messages.pop(index)
                return True
        return False

    def nack_tag(
        self,
        delivery_tag: int,
        *,
        consumer: str,
        requeue: bool,
        now_ms: int,
    ) -> str:
        del now_ms
        for index, message in enumerate(self.messages):
            if (
                message["delivery_tag"] == delivery_tag
                and message["status"] == "claimed"
                and message["owner_consumer"] == consumer
            ):
                self.nacks.append((delivery_tag, requeue))
                if requeue and int(message["delivery_count"]) < self.max_delivery_count:
                    message.update(
                        {
                            "status": "pending",
                            "delivery_tag": None,
                            "owner_consumer": None,
                            "lease_until": None,
                        }
                    )
                    return "requeued"
                self.messages.pop(index)
                return "dropped"
        return "absent"

    def inject(self, payload: bytes) -> None:
        self.sequence += 1
        self.messages.insert(
            0,
            {
                "node_id": f"injected-{self.sequence}",
                "payload": payload.hex(),
                "status": "pending",
                "delivery_tag": None,
                "owner_consumer": None,
                "lease_until": None,
                "delivery_count": 0,
            },
        )


def _runtime() -> tuple[EpistemicGraphA2ARuntime, FakeBroker, FakeNodes, FakeTxn]:
    actor = ActorContext(
        actor_id="service-subject",
        actor_type=ActorType.AUTOMATED_SERVICE,
        roles=frozenset({"kg:read", "kg:write"}),
        tenant_id="tenant-fixture",
        authenticated=True,
    )
    session = GraphSession(
        actor=actor,
        tenant="tenant-fixture",
        scopes=frozenset({"kg:read", "kg:write"}),
        graph="tenant-graph",
        policy_version="policy-v1",
        audience="agent-services",
    )
    broker = FakeBroker()
    nodes = FakeNodes()
    txn = FakeTxn(nodes)
    client = SimpleNamespace(broker=broker, nodes=nodes, txn=txn)
    return EpistemicGraphA2ARuntime(client=client, session=session), broker, nodes, txn


def _message(text: str = "perform the task") -> Message:
    return {
        "role": "user",
        "parts": [{"kind": "text", "text": text}],
        "kind": "message",
        "message_id": "runtime-message-id",
    }


def _governed_file_message() -> Message:
    return {
        "role": "user",
        "parts": [
            {
                "kind": "file",
                "file": {
                    "uri": "urn:agent-utilities:content:sha256:" + "a" * 64,
                    "mime_type": "application/octet-stream",
                },
            }
        ],
        "kind": "message",
        "message_id": "governed-message-id",
    }


def _run_params(task: Task) -> dict[str, Any]:
    return {
        "id": task["id"],
        "context_id": task["context_id"],
        "message": _message(),
    }


def test_adapter_is_pinned_to_fenced_current_engine_contract() -> None:
    from epistemic_graph.client import BrokerClient, NodeClient

    assert tuple(inspect.signature(NodeClient.create_if_absent).parameters) == (
        "self",
        "node_id",
        "properties",
    )
    expected = {
        "publish_idempotent": (
            "self",
            "exchange",
            "routing_key",
            "payload",
            "producer_id",
            "seq",
            "priority",
            "delay_ms",
            "ttl_ms",
            "now_ms",
        ),
        "consume": (
            "self",
            "queue",
            "group",
            "consumer",
            "now_ms",
            "lease_ms",
            "prefetch",
        ),
        "renew_tag": (
            "self",
            "delivery_tag",
            "consumer",
            "now_ms",
            "lease_ms",
        ),
        "ack_tag": ("self", "delivery_tag", "consumer"),
        "nack_tag": (
            "self",
            "delivery_tag",
            "consumer",
            "requeue",
            "now_ms",
        ),
    }
    observed = {
        method: tuple(inspect.signature(getattr(BrokerClient, method)).parameters)
        for method in expected
    }
    assert observed == expected


def test_dispatch_reconciler_configuration_is_bounded_at_adapter_boundary() -> None:
    runtime, _native, _nodes, _txn = _runtime()
    storage = EpistemicGraphA2AStorage(runtime)

    with pytest.raises(ValueError, match="interval"):
        EpistemicGraphA2ABroker(runtime, storage, reconcile_interval_ms=0)
    with pytest.raises(ValueError, match="page limit"):
        EpistemicGraphA2ABroker(runtime, storage, reconcile_limit=0)
    with pytest.raises(ValueError, match="cancellation poll"):
        EpistemicGraphA2ABroker(runtime, storage, cancellation_poll_interval_ms=0)


@pytest.mark.asyncio
async def test_dead_dispatch_reconciler_fails_loud_to_active_worker() -> None:
    runtime, _native, _nodes, _txn = _runtime()
    storage = EpistemicGraphA2AStorage(runtime)
    broker = EpistemicGraphA2ABroker(runtime, storage)

    async def fail() -> None:
        raise ValueError("source-specific detail must not escape")

    broker._active = True
    broker._reconcile_task = asyncio.create_task(fail())
    await asyncio.sleep(0)
    with pytest.raises(
        RuntimeError, match="reconciliation failed.*ValueError"
    ) as captured:
        broker._raise_reconciler_failure()
    assert captured.value.__cause__ is None
    broker._active = False


@pytest.mark.asyncio
async def test_submission_fails_closed_on_privacy_change_and_inline_bytes() -> None:
    runtime, _broker, nodes, _txn = _runtime()
    storage = EpistemicGraphA2AStorage(runtime)

    with pytest.raises(ValueError, match="prohibited"):
        await storage.submit_task("context", _message("contact person@example.test"))
    assert nodes.rows == {}

    inline: Message = {
        "role": "user",
        "parts": [
            {
                "kind": "file",
                "file": {"bytes": "cGVyc29uQGV4YW1wbGUudGVzdA=="},
            }
        ],
        "kind": "message",
        "message_id": "inline-message-id",
    }
    with pytest.raises(ValueError, match="inline file bytes"):
        await storage.submit_task("context", inline)
    assert nodes.rows == {}


@pytest.mark.asyncio
async def test_governed_reference_is_the_only_persisted_file_material() -> None:
    runtime, _broker, nodes, _txn = _runtime()
    storage = EpistemicGraphA2AStorage(runtime)
    task = await storage.submit_task("context", _governed_file_message())
    persisted = json.dumps(nodes.rows, sort_keys=True)
    assert "urn:agent-utilities:content:sha256:" in persisted
    assert '"bytes"' not in persisted
    assert "tenant-fixture" not in persisted
    assert task["history"][0]["parts"][0]["kind"] == "file"


@pytest.mark.asyncio
async def test_context_create_if_absent_never_overwrites_existing_revision() -> None:
    runtime, _broker, nodes, _txn = _runtime()
    await runtime.start()
    context_id = runtime.context_id("shared-context")
    nodes.rows[context_id] = {
        "record_kind": "a2a_context_v1",
        "node_type": "A2AContext",
        "tenant_ref": runtime.tenant_ref,
        "revision": 7,
        "payload": [],
        "payload_ref": _payload_ref([], tenant_key=runtime.tenant_key),
    }
    storage = EpistemicGraphA2AStorage(runtime)
    task = await storage.submit_task("shared-context", _message())
    assert nodes.rows[context_id]["revision"] == 7
    assert nodes.rows[task["id"]]["context_revision"] == 7


@pytest.mark.asyncio
async def test_payload_digest_tamper_is_rejected_before_load_or_cas() -> None:
    runtime, _broker, nodes, _txn = _runtime()
    storage = EpistemicGraphA2AStorage(runtime)
    task = await storage.submit_task("context", _message())
    nodes.rows[task["id"]]["payload"]["history"][0]["parts"][0]["text"] = "tampered"
    with pytest.raises(RuntimeError, match="record is invalid"):
        await storage.load_task(task["id"])


@pytest.mark.asyncio
async def test_dispatch_reconciler_closes_create_publish_window_idempotently() -> None:
    runtime, native, nodes, _txn = _runtime()
    storage = EpistemicGraphA2AStorage(runtime)
    task = await storage.submit_task("context", _message())
    broker = EpistemicGraphA2ABroker(runtime, storage, reconcile_interval_ms=60_000)

    async with broker:
        assert len(native.messages) == 1
        assert nodes.rows[task["id"]]["run_dispatch_state"] == "published"
        await broker.run_task(_run_params(task))
        assert len(native.messages) == 1
        publish_results = [
            call for call in native.calls if call[0] == "publish_idempotent"
        ]
        assert len(publish_results) == 2


@pytest.mark.asyncio
async def test_http_cancel_is_durable_before_cancel_dispatch() -> None:
    runtime, native, nodes, _txn = _runtime()
    storage = EpistemicGraphA2AStorage(runtime)
    task = await storage.submit_task("context", _message())
    broker = EpistemicGraphA2ABroker(runtime, storage, reconcile_interval_ms=60_000)

    async with broker:
        await broker.cancel_task({"id": task["id"]})
        loaded = await storage.load_task(task["id"])
        assert loaded is not None
        assert loaded["status"]["state"] == "canceled"
        assert nodes.rows[task["id"]]["cancel_dispatch_state"] == "published"
        assert len(native.messages) == 2


@pytest.mark.asyncio
async def test_heartbeat_renews_the_current_delivery_tag() -> None:
    runtime, native, _nodes, _txn = _runtime()
    storage = EpistemicGraphA2AStorage(runtime)
    await storage.submit_task("context", _message())
    broker = EpistemicGraphA2ABroker(
        runtime,
        storage,
        lease_ms=300,
        reconcile_interval_ms=60_000,
    )

    async with broker:
        iterator = broker.receive_task_operations()
        await anext(iterator)
        await asyncio.sleep(0.13)
        assert native.renewals
        await iterator.aclose()


@pytest.mark.asyncio
async def test_cross_process_cancel_aborts_active_delivery_and_acks_run() -> None:
    runtime, native, _nodes, _txn = _runtime()
    storage = EpistemicGraphA2AStorage(runtime)
    task = await storage.submit_task("context", _message())
    broker = EpistemicGraphA2ABroker(
        runtime,
        storage,
        lease_ms=300_000,
        reconcile_interval_ms=60_000,
        cancellation_poll_interval_ms=20,
    )

    async with broker:
        iterator = broker.receive_task_operations()
        operation = await anext(iterator)
        assert operation["operation"] == "run"
        await storage.update_task(task["id"], "working")
        await broker.cancel_task({"id": task["id"]})
        control = _DELIVERY_CONTROL.get()
        assert control is not None
        await asyncio.wait_for(control.abort_event.wait(), timeout=0.5)
        assert control.abort_reason == "task_canceled"
        assert native.renewals == []
        second = await anext(iterator)
        assert second["operation"] == "cancel"
        assert native.acks
        await iterator.aclose()


@pytest.mark.asyncio
async def test_cancel_wins_atomically_over_late_context_completion() -> None:
    runtime, _native, nodes, _txn = _runtime()
    storage = EpistemicGraphA2AStorage(runtime)
    task = await storage.submit_task("context", _message())
    broker = EpistemicGraphA2ABroker(runtime, storage, reconcile_interval_ms=60_000)

    async with broker:
        iterator = broker.receive_task_operations()
        await anext(iterator)
        await storage.update_task(task["id"], "working")
        await storage.cancel_task(task["id"])
        context = [ModelRequest(parts=[UserPromptPart(content="safe result")])]
        with pytest.raises(A2AStorageConflict):
            await storage.complete_task(
                task["id"], context, new_artifacts=[], new_messages=[]
            )
        context_id = task["context_id"]
        assert nodes.rows[context_id]["revision"] == 0
        assert nodes.rows[context_id]["payload"] == []
        await iterator.aclose()


@pytest.mark.asyncio
async def test_context_and_terminal_task_commit_in_one_transaction() -> None:
    runtime, _native, nodes, _txn = _runtime()
    storage = EpistemicGraphA2AStorage(runtime)
    task = await storage.submit_task("context", _message())
    broker = EpistemicGraphA2ABroker(runtime, storage, reconcile_interval_ms=60_000)
    artifact: Artifact = {
        "artifact_id": "result-id",
        "name": "result",
        "parts": [{"kind": "text", "text": "safe answer"}],
    }
    response: Message = {
        "role": "agent",
        "parts": [{"kind": "text", "text": "safe answer"}],
        "kind": "message",
        "message_id": "response-id",
    }

    async with broker:
        iterator = broker.receive_task_operations()
        await anext(iterator)
        await storage.update_task(task["id"], "working")
        context = [ModelRequest(parts=[UserPromptPart(content="safe answer")])]
        completed = await storage.complete_task(
            task["id"],
            context,
            new_artifacts=[artifact],
            new_messages=[response],
        )
        assert completed["status"]["state"] == "completed"
        assert nodes.rows[task["context_id"]]["revision"] == 1
        assert nodes.rows[task["id"]]["state"] == "completed"
        await iterator.aclose()


@pytest.mark.asyncio
async def test_deep_poison_payload_is_nacked_without_stopping_consumer() -> None:
    runtime, native, _nodes, _txn = _runtime()
    storage = EpistemicGraphA2AStorage(runtime)
    task = await storage.submit_task("context", _message())
    broker = EpistemicGraphA2ABroker(runtime, storage, reconcile_interval_ms=60_000)

    async with broker:
        nested: Any = "leaf"
        for _ in range(40):
            nested = [nested]
        native.inject(
            json.dumps(
                {"schema_version": 1, "operation": "run", "params": nested}
            ).encode()
        )
        iterator = broker.receive_task_operations()
        operation = await anext(iterator)
        assert operation["operation"] == "run"
        assert native.nacks and native.nacks[0][1] is False
        assert operation["params"]["id"] == task["id"]
        await iterator.aclose()


@pytest.mark.asyncio
async def test_noncanonical_hex_payload_is_nacked_without_stopping_consumer() -> None:
    runtime, native, _nodes, _txn = _runtime()
    storage = EpistemicGraphA2AStorage(runtime)
    task = await storage.submit_task("context", _message())
    broker = EpistemicGraphA2ABroker(runtime, storage, reconcile_interval_ms=60_000)

    async with broker:
        native.inject(b"{}")
        native.messages[0]["payload"] = "7b 7d"
        iterator = broker.receive_task_operations()
        operation = await anext(iterator)
        assert operation["operation"] == "run"
        assert native.nacks and native.nacks[0][1] is False
        assert operation["params"]["id"] == task["id"]
        await iterator.aclose()


@pytest.mark.asyncio
async def test_strict_context_id_and_preprojection_collection_bounds() -> None:
    runtime, _broker, _nodes, _txn = _runtime()
    await runtime.start()
    malformed = f"a2a.context.{runtime.tenant_key}." + "z" * 64
    with pytest.raises(ValueError, match="canonical opaque"):
        runtime.context_id(malformed)

    storage = EpistemicGraphA2AStorage(runtime, max_history=2)
    oversized = _message()
    oversized["reference_task_ids"] = ["one", "two", "three"]
    with pytest.raises(ValueError, match="collection bound"):
        await storage.submit_task("context", oversized)


class _SlowAgent:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.canceled = asyncio.Event()

    async def run(self, **_kwargs: Any) -> Any:
        self.started.set()
        try:
            await asyncio.Event().wait()
        finally:
            self.canceled.set()


class _ExplodingWorker(EpistemicGraphAgentWorker):
    async def _handle_task_operation(self, task_operation: Any) -> None:
        del task_operation
        raise RuntimeError("simulated handler failure")


@pytest.mark.asyncio
async def test_exhausted_delivery_fails_task_before_broker_drops_operation() -> None:
    runtime, native, _nodes, _txn = _runtime()
    storage = EpistemicGraphA2AStorage(runtime)
    task = await storage.submit_task("context", _message())
    broker = EpistemicGraphA2ABroker(
        runtime,
        storage,
        max_delivery_count=1,
        reconcile_interval_ms=60_000,
    )
    worker = _ExplodingWorker(agent=_SlowAgent(), broker=broker, storage=storage)

    async with broker:
        worker_task = asyncio.create_task(worker._loop())
        for _attempt in range(50):
            loaded = await storage.load_task(task["id"])
            if loaded is not None and loaded["status"]["state"] == "failed":
                break
            await asyncio.sleep(0.01)
        assert loaded is not None and loaded["status"]["state"] == "failed"
        assert native.nacks and native.nacks[-1][1] is True
        assert native.messages == []
        worker_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker_task


@pytest.mark.asyncio
async def test_current_worker_stops_active_execution_after_durable_cancel() -> None:
    runtime, native, _nodes, _txn = _runtime()
    storage = EpistemicGraphA2AStorage(runtime)
    task = await storage.submit_task("context", _message())
    broker = EpistemicGraphA2ABroker(
        runtime,
        storage,
        lease_ms=150,
        reconcile_interval_ms=60_000,
    )
    agent = _SlowAgent()
    worker = EpistemicGraphAgentWorker(agent=agent, broker=broker, storage=storage)

    async with broker:
        worker_task = asyncio.create_task(worker._loop())
        await asyncio.wait_for(agent.started.wait(), timeout=1)
        await broker.cancel_task({"id": task["id"]})
        await asyncio.wait_for(agent.canceled.wait(), timeout=1)
        loaded = await storage.load_task(task["id"])
        assert loaded is not None and loaded["status"]["state"] == "canceled"
        await asyncio.sleep(0.05)
        assert native.acks
        worker_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker_task
