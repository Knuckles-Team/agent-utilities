"""Live native durability contracts for the Epistemic Graph A2A runtime.

These scenarios intentionally use the current ``SyncEpistemicGraphClient``
against an isolated full engine binary.  Fakes cannot certify the broker lease,
transaction, WAL, or restart behavior exercised here.  The harness retains only
generic test authority and throwaway opaque identifiers, and it removes all
state when the module finishes.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import multiprocessing
import os
import shutil
import signal
import socket
import subprocess
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from _test_engine import (
    IDLE_SHUTDOWN_SECS,
    TEST_AGENT_ID,
    TEST_AUDIENCE,
    TEST_AUTH_SECRET,
    TEST_POLICY_VERSION,
    TEST_SIGNER_KEY,
    TEST_TENANT,
    EngineUnavailable,
    bootstrap_context,
    request_context,
    resolve_engine_binary,
    strict_server_env,
)
from fasta2a.schema import Message
from pydantic_ai.messages import ModelRequest, UserPromptPart

from agent_utilities.knowledge_graph.core.session import GraphSession
from agent_utilities.models.company_brain import ActorType
from agent_utilities.protocols.a2a_epistemic import (
    _DELIVERY_CONTROL,
    _EXECUTION_BINDING,
    A2AStorageConflict,
    EpistemicGraphA2ABroker,
    EpistemicGraphA2ARuntime,
    EpistemicGraphA2AStorage,
)
from agent_utilities.security.brain_context import ActorContext

pytestmark = pytest.mark.integration

_STARTUP_TIMEOUT_SECONDS = 30.0
_SHUTDOWN_TIMEOUT_SECONDS = 15.0
_OPERATION_TIMEOUT_SECONDS = 12.0
_ABORT_TIMEOUT_SECONDS = 8.0
_EXECUTOR_TIMEOUT_SECONDS = 20.0


class _DeliveryRetryProbe(RuntimeError):
    """Test-only signal that drives the adapter's real nack/requeue path."""


class _RestartableNativeEngine:
    """One exact binary over one throwaway durable store, restartable in place."""

    def __init__(self, binary: Path, root: Path) -> None:
        self.binary = binary
        self.root = root
        self.persist_dir = root / "persist"
        self.security_dir = root / "security"
        self.log_path = root / "engine.log"
        self.socket_path = ""
        self.process: subprocess.Popen[bytes] | None = None
        self._log: Any | None = None

    def start(self, *, bootstrap: bool) -> None:
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.security_dir.mkdir(parents=True, exist_ok=True)
        self.socket_path = str(self.root / f"eg-{uuid.uuid4().hex[:8]}.sock")
        self._log = open(self.log_path, "ab")  # noqa: SIM115 - closed by stop/crash
        env = {
            **os.environ,
            **strict_server_env(
                str(self.security_dir),
                auth_secret=TEST_AUTH_SECRET,
            ),
            "GRAPH_SERVICE_PERSIST_DIR": str(self.persist_dir),
        }
        self.process = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
            [
                str(self.binary),
                "--socket-path",
                self.socket_path,
                "--persist-dir",
                str(self.persist_dir),
                "--auth-secret",
                TEST_AUTH_SECRET,
                "--idle-shutdown-secs",
                str(IDLE_SHUTDOWN_SECS),
            ],
            stdout=self._log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        try:
            self._wait_until_ready()
            if bootstrap:
                self._bootstrap_identity()
        except BaseException:
            self.stop()
            raise

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + _STARTUP_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            process = self.process
            if process is not None and process.poll() is not None:
                raise EngineUnavailable(
                    "native A2A certification engine exited during startup"
                )
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
                    probe.settimeout(0.25)
                    probe.connect(self.socket_path)
                return
            except OSError:
                time.sleep(0.05)
        raise EngineUnavailable(
            "native A2A certification engine did not become ready in time"
        )

    def _bootstrap_identity(self) -> None:
        from epistemic_graph.client import SyncEpistemicGraphClient

        client = SyncEpistemicGraphClient.connect(
            socket_path=self.socket_path,
            auth_secret=TEST_AUTH_SECRET,
            verified_context=bootstrap_context(),
        )
        try:
            client.consensus.bootstrap_system_identity(
                agent_id=TEST_AGENT_ID,
                signer_id=TEST_AGENT_ID,
                signer_key=TEST_SIGNER_KEY,
            )
        finally:
            client.close()

    def connect(self, graph_name: str) -> Any:
        from epistemic_graph.client import SyncEpistemicGraphClient

        return SyncEpistemicGraphClient.connect(
            socket_path=self.socket_path,
            auth_secret=TEST_AUTH_SECRET,
            graph_name=graph_name,
            verified_context=request_context(),
        )

    def crash(self) -> None:
        process = self.process
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=_SHUTDOWN_TIMEOUT_SECONDS)
        self.process = None
        self._close_log()
        self._unlink_socket()

    def restart(self) -> None:
        if self.process is not None:
            raise RuntimeError("native A2A certification engine is already running")
        self.start(bootstrap=False)

    def stop(self) -> None:
        process = self.process
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=_SHUTDOWN_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=_SHUTDOWN_TIMEOUT_SECONDS)
        if process is not None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
        self.process = None
        self._close_log()
        self._unlink_socket()

    def _close_log(self) -> None:
        if self._log is not None:
            self._log.close()
            self._log = None

    def _unlink_socket(self) -> None:
        if self.socket_path:
            Path(self.socket_path).unlink(missing_ok=True)


@pytest.fixture(scope="module")
def native_a2a_engine(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[_RestartableNativeEngine]:
    """Run one isolated exact engine for all native A2A scenarios in this module."""

    exact_binary_required = bool(os.environ.get("EPISTEMIC_GRAPH_TEST_BINARY"))
    try:
        binary = resolve_engine_binary()
    except EngineUnavailable as exc:
        if exact_binary_required:
            pytest.fail(f"configured native A2A engine is unavailable: {exc}")
        pytest.skip(str(exc))
    engine = _RestartableNativeEngine(
        binary,
        tmp_path_factory.mktemp("native-a2a"),
    )
    try:
        engine.start(bootstrap=True)
    except EngineUnavailable as exc:
        if exact_binary_required:
            pytest.fail(f"configured native A2A engine failed startup: {exc}")
        pytest.skip(str(exc))
    try:
        yield engine
    finally:
        engine.stop()
        shutil.rmtree(engine.root, ignore_errors=False)


def _session(graph_name: str) -> GraphSession:
    actor = ActorContext(
        actor_id=TEST_AGENT_ID,
        actor_type=ActorType.AUTOMATED_SERVICE,
        roles=frozenset({"test"}),
        tenant_id=TEST_TENANT,
        authenticated=True,
    )
    return GraphSession(
        actor=actor,
        tenant=TEST_TENANT,
        scopes=frozenset({"kg:read", "kg:write", "kg:admin"}),
        graph=graph_name,
        policy_version=TEST_POLICY_VERSION,
        audience=TEST_AUDIENCE,
    )


@dataclass
class _NativeCase:
    engine: _RestartableNativeEngine
    graph_name: str
    client: Any
    session: GraphSession
    runtime: EpistemicGraphA2ARuntime
    storage: EpistemicGraphA2AStorage

    @classmethod
    def open(cls, engine: _RestartableNativeEngine) -> _NativeCase:
        graph_name = f"a2a-cert-{uuid.uuid4().hex}"
        client = engine.connect(graph_name)
        client.tenants.create(graph_name)
        session = _session(graph_name)
        runtime = EpistemicGraphA2ARuntime(client=client, session=session)
        return cls(
            engine=engine,
            graph_name=graph_name,
            client=client,
            session=session,
            runtime=runtime,
            storage=EpistemicGraphA2AStorage(runtime),
        )

    def reconnect(self) -> None:
        self.client = self.engine.connect(self.graph_name)
        self.runtime = EpistemicGraphA2ARuntime(
            client=self.client,
            session=self.session,
        )
        self.storage = EpistemicGraphA2AStorage(self.runtime)

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.client.tenants.delete(self.graph_name)
        with contextlib.suppress(Exception):
            self.client.close()


@contextlib.contextmanager
def _native_case(engine: _RestartableNativeEngine) -> Iterator[_NativeCase]:
    case = _NativeCase.open(engine)
    try:
        yield case
    finally:
        case.close()


def _message(text: str = "perform bounded work") -> Message:
    return {
        "role": "user",
        "parts": [{"kind": "text", "text": text}],
        "kind": "message",
        "message_id": f"message-{uuid.uuid4().hex}",
    }


def _model_context(text: str = "bounded result") -> list[ModelRequest]:
    return [ModelRequest(parts=[UserPromptPart(content=text)])]


async def _ack_and_stop(
    broker: EpistemicGraphA2ABroker,
    iterator: AsyncIterator[Any],
) -> None:
    """Resume one yielded delivery so its real ack lands, then stop polling."""

    broker._active = False
    with pytest.raises(StopAsyncIteration):
        await anext(iterator)


async def _retry_delivery(iterator: AsyncIterator[Any]) -> None:
    """Drive the real nack/requeue branch and close the delivery generator."""

    with pytest.raises(_DeliveryRetryProbe):
        await iterator.athrow(_DeliveryRetryProbe("retry delivery"))


async def _publish_raw(
    broker: EpistemicGraphA2ABroker,
    payload: bytes,
) -> None:
    result = await broker.runtime.call(
        "broker",
        "publish_idempotent",
        broker._exchange,
        "task",
        payload,
        producer_id=f"a2a.test-producer.{uuid.uuid4().hex}",
        seq=1,
    )
    assert result == {"confirmed": True, "duplicate": False, "delivered": 1}


async def _executor_process_main(
    socket_path: str,
    graph_name: str,
    task_id: str,
    connection: Any,
    *,
    complete: bool,
) -> None:
    """Hold one real broker delivery in a separately killable executor process."""

    from epistemic_graph.client import SyncEpistemicGraphClient

    client = SyncEpistemicGraphClient.connect(
        socket_path=socket_path,
        auth_secret=TEST_AUTH_SECRET,
        graph_name=graph_name,
        verified_context=request_context(),
    )
    runtime = EpistemicGraphA2ARuntime(client=client, session=_session(graph_name))
    storage = EpistemicGraphA2AStorage(runtime)
    broker = EpistemicGraphA2ABroker(
        runtime,
        storage,
        poll_interval_ms=10,
        lease_ms=600,
        reconcile_interval_ms=60_000,
        cancellation_poll_interval_ms=20,
    )
    try:
        async with broker:
            iterator = broker.receive_task_operations()
            operation = await asyncio.wait_for(
                anext(iterator), timeout=_OPERATION_TIMEOUT_SECONDS
            )
            assert operation["params"]["id"] == task_id
            await storage.update_task(task_id, "working")
            binding = _EXECUTION_BINDING.get()
            assert binding is not None
            connection.send(("claimed", binding))
            if not complete:
                await asyncio.Event().wait()
            command = await asyncio.to_thread(connection.recv)
            assert command == "complete"
            completed = await storage.complete_task(
                task_id,
                _model_context("current result"),
                new_artifacts=[],
                new_messages=[],
            )
            await _ack_and_stop(broker, iterator)
            connection.send(("completed", completed["status"]["state"]))
    finally:
        client.close()
        connection.close()


def _executor_process_entry(
    socket_path: str,
    graph_name: str,
    task_id: str,
    connection: Any,
    complete: bool,
) -> None:
    try:
        asyncio.run(
            _executor_process_main(
                socket_path,
                graph_name,
                task_id,
                connection,
                complete=complete,
            )
        )
    except BaseException as error:
        with contextlib.suppress(BaseException):
            connection.send(("failed", type(error).__name__))
        raise


def _receive_process_message(connection: Any) -> tuple[str, Any]:
    if not connection.poll(_EXECUTOR_TIMEOUT_SECONDS):
        raise TimeoutError("separate A2A executor did not report in time")
    value = connection.recv()
    if not isinstance(value, tuple) or len(value) != 2:
        raise RuntimeError("separate A2A executor returned an invalid result")
    return value


def _terminate_executor(process: multiprocessing.Process | None) -> None:
    if process is None:
        return
    if process.is_alive():
        process.kill()
    process.join(timeout=_SHUTDOWN_TIMEOUT_SECONDS)
    if process.is_alive():
        raise RuntimeError("separate A2A executor resisted SIGKILL")


@pytest.mark.asyncio
async def test_native_create_reconcile_and_terminal_commit_are_atomic(
    native_a2a_engine: _RestartableNativeEngine,
) -> None:
    """Real create-if-absent, reconciliation, and two-record txn stay atomic."""

    with _native_case(native_a2a_engine) as case:
        tasks = await asyncio.gather(
            *(case.storage.submit_task("shared-context", _message()) for _ in range(3))
        )
        assert len({task["id"] for task in tasks}) == 3
        assert len({task["context_id"] for task in tasks}) == 1
        context_id = tasks[0]["context_id"]
        context_before = await case.runtime.call("nodes", "properties", context_id)
        assert context_before["revision"] == 0

    # Use a fresh graph for the dispatch/transaction leg so the three legitimate
    # concurrent submissions above cannot be mistaken for duplicate dispatches.
    with _native_case(native_a2a_engine) as case:
        task = await case.storage.submit_task("terminal-context", _message())
        broker = EpistemicGraphA2ABroker(
            case.runtime,
            case.storage,
            poll_interval_ms=10,
            reconcile_interval_ms=60_000,
        )
        async with broker:
            iterator = broker.receive_task_operations()
            operation = await asyncio.wait_for(
                anext(iterator), timeout=_OPERATION_TIMEOUT_SECONDS
            )
            task_id = operation["params"]["id"]
            assert task_id == task["id"]
            await case.storage.update_task(task_id, "working")
            completed = await case.storage.complete_task(
                task_id,
                _model_context(),
                new_artifacts=[],
                new_messages=[],
            )
            assert completed["status"]["state"] == "completed"

            context_after = await case.runtime.call(
                "nodes", "properties", task["context_id"]
            )
            task_after = await case.runtime.call("nodes", "properties", task_id)
            assert context_after["revision"] == 1
            assert task_after["state"] == "completed"
            assert task_after["context_revision"] == context_after["revision"]
            assert task_after["context_payload_ref"] == context_after["payload_ref"]

            await broker.run_task(operation["params"])
            await _ack_and_stop(broker, iterator)
            assert (
                await case.runtime.call(
                    "broker",
                    "consume",
                    broker._queue,
                    group="a2a-workers",
                    consumer="duplicate-check",
                    now_ms=int(time.time() * 1_000),
                    lease_ms=1_000,
                    prefetch=1,
                )
            ) is None


@pytest.mark.asyncio
async def test_native_transaction_conflict_never_partially_commits(
    native_a2a_engine: _RestartableNativeEngine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A competing native CAS makes the context/task transaction apply neither leg."""

    with _native_case(native_a2a_engine) as case:
        task = await case.storage.submit_task("atomic-context", _message())
        broker = EpistemicGraphA2ABroker(
            case.runtime,
            case.storage,
            poll_interval_ms=10,
            reconcile_interval_ms=60_000,
        )
        async with broker:
            iterator = broker.receive_task_operations()
            await asyncio.wait_for(
                anext(iterator), timeout=_OPERATION_TIMEOUT_SECONDS
            )
            await case.storage.update_task(task["id"], "working")
            context_before = await case.runtime.call(
                "nodes", "properties", task["context_id"]
            )
            task_before = await case.runtime.call(
                "nodes", "properties", task["id"]
            )
            original_call = case.runtime.call
            injected = False

            async def inject_conflict(
                namespace: str,
                method: str,
                *args: Any,
                **kwargs: Any,
            ) -> Any:
                nonlocal injected
                result = await original_call(namespace, method, *args, **kwargs)
                if (
                    not injected
                    and namespace == "txn"
                    and method == "cas"
                    and len(args) > 1
                    and args[1] == task["context_id"]
                ):
                    injected = True
                    assert await original_call(
                        "nodes",
                        "compare_and_set",
                        task["id"],
                        {
                            "revision": task_before["revision"],
                            "payload_ref": task_before["payload_ref"],
                        },
                        {"revision": task_before["revision"] + 1},
                    )
                return result

            with monkeypatch.context() as scoped:
                scoped.setattr(case.runtime, "call", inject_conflict)
                with pytest.raises(A2AStorageConflict, match="transaction"):
                    await case.storage.complete_task(
                        task["id"],
                        _model_context(),
                        new_artifacts=[],
                        new_messages=[],
                    )

            context_after = await original_call(
                "nodes", "properties", task["context_id"]
            )
            task_after = await original_call("nodes", "properties", task["id"])
            assert injected
            assert context_after == context_before
            assert task_after["revision"] == task_before["revision"] + 1
            assert task_after["state"] == "working"
            assert task_after["payload_ref"] == task_before["payload_ref"]
            await _retry_delivery(iterator)


@pytest.mark.asyncio
async def test_native_delivery_lease_renews_and_stale_generation_is_fenced(
    native_a2a_engine: _RestartableNativeEngine,
) -> None:
    """The live heartbeat renews; a requeued delivery invalidates its old tag."""

    with _native_case(native_a2a_engine) as case:
        await case.storage.submit_task("lease-context", _message())
        first = EpistemicGraphA2ABroker(
            case.runtime,
            case.storage,
            poll_interval_ms=10,
            lease_ms=450,
            reconcile_interval_ms=60_000,
            cancellation_poll_interval_ms=20,
        )
        async with first:
            iterator = first.receive_task_operations()
            await asyncio.wait_for(
                anext(iterator), timeout=_OPERATION_TIMEOUT_SECONDS
            )
            first_control = _DELIVERY_CONTROL.get()
            assert first_control is not None
            await asyncio.sleep(0.55)
            assert await case.runtime.call(
                "broker",
                "renew_tag",
                first_control.delivery_tag,
                consumer=first_control.consumer,
                now_ms=int(time.time() * 1_000),
                lease_ms=450,
            )
            await _retry_delivery(iterator)

        second = EpistemicGraphA2ABroker(
            case.runtime,
            case.storage,
            poll_interval_ms=10,
            lease_ms=450,
            reconcile_interval_ms=60_000,
        )
        async with second:
            iterator = second.receive_task_operations()
            await asyncio.wait_for(
                anext(iterator), timeout=_OPERATION_TIMEOUT_SECONDS
            )
            second_control = _DELIVERY_CONTROL.get()
            assert second_control is not None
            assert second_control.delivery_tag != first_control.delivery_tag
            assert second_control.consumer != first_control.consumer
            assert not await case.runtime.call(
                "broker",
                "renew_tag",
                first_control.delivery_tag,
                consumer=first_control.consumer,
                now_ms=int(time.time() * 1_000),
                lease_ms=450,
            )
            assert not await case.runtime.call(
                "broker",
                "ack_tag",
                first_control.delivery_tag,
                consumer=first_control.consumer,
            )
            await _retry_delivery(iterator)


@pytest.mark.asyncio
async def test_native_poison_bounds_and_record_tamper_fail_closed(
    native_a2a_engine: _RestartableNativeEngine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Poison deliveries are dropped, while durable-record tamper is rejected."""

    with _native_case(native_a2a_engine) as case:
        broker = EpistemicGraphA2ABroker(
            case.runtime,
            case.storage,
            poll_interval_ms=10,
            max_payload_bytes=1_024,
            reconcile_interval_ms=60_000,
        )
        async with broker:
            nested: Any = "leaf"
            for _ in range(40):
                nested = [nested]
            await _publish_raw(
                broker,
                json.dumps(
                    {"schema_version": 1, "operation": "run", "params": nested}
                ).encode(),
            )
            await _publish_raw(broker, b"x" * 1_025)
            await _publish_raw(
                broker,
                json.dumps(
                    {
                        "schema_version": 1,
                        "operation": "cancel",
                        "params": {"id": "not-an-opaque-task"},
                    }
                ).encode(),
            )
            task = await case.storage.submit_task("poison-context", _message())
            await broker._reconcile_once()

            original_call = case.runtime.call
            nacks: list[tuple[bool, str]] = []

            async def record_nacks(
                namespace: str,
                method: str,
                *args: Any,
                **kwargs: Any,
            ) -> Any:
                result = await original_call(namespace, method, *args, **kwargs)
                if namespace == "broker" and method == "nack_tag":
                    nacks.append((bool(kwargs["requeue"]), str(result)))
                return result

            with monkeypatch.context() as scoped:
                scoped.setattr(case.runtime, "call", record_nacks)
                iterator = broker.receive_task_operations()
                operation = await asyncio.wait_for(
                    anext(iterator), timeout=_OPERATION_TIMEOUT_SECONDS
                )
                assert operation["params"]["id"] == task["id"]
                assert nacks == [(False, "dropped")] * 3

                record = await original_call("nodes", "properties", task["id"])
                tampered_payload = json.loads(json.dumps(record["payload"]))
                tampered_payload["history"][0]["parts"][0]["text"] = "tampered"
                assert await original_call(
                    "nodes",
                    "compare_and_set",
                    task["id"],
                    {"payload_ref": record["payload_ref"]},
                    {"payload": tampered_payload},
                )
                with pytest.raises(RuntimeError, match="record is invalid"):
                    await case.storage.load_task(task["id"])
                await _retry_delivery(iterator)


@pytest.mark.asyncio
async def test_native_cancellation_wins_and_late_completion_is_rejected(
    native_a2a_engine: _RestartableNativeEngine,
) -> None:
    """Durable cancellation aborts work and fences both late terminal writes."""

    with _native_case(native_a2a_engine) as case:
        task = await case.storage.submit_task("cancel-context", _message())
        broker = EpistemicGraphA2ABroker(
            case.runtime,
            case.storage,
            poll_interval_ms=10,
            lease_ms=1_000,
            reconcile_interval_ms=60_000,
            cancellation_poll_interval_ms=10,
        )
        async with broker:
            iterator = broker.receive_task_operations()
            operation = await asyncio.wait_for(
                anext(iterator), timeout=_OPERATION_TIMEOUT_SECONDS
            )
            assert operation["operation"] == "run"
            await case.storage.update_task(task["id"], "working")
            context_before = await case.runtime.call(
                "nodes", "properties", task["context_id"]
            )
            await broker.cancel_task({"id": task["id"]})
            control = _DELIVERY_CONTROL.get()
            assert control is not None
            await asyncio.wait_for(
                control.abort_event.wait(), timeout=_ABORT_TIMEOUT_SECONDS
            )
            assert control.abort_reason == "task_canceled"

            with pytest.raises(A2AStorageConflict, match="completion"):
                await case.storage.complete_task(
                    task["id"],
                    _model_context(),
                    new_artifacts=[],
                    new_messages=[],
                )
            assert (
                await case.runtime.call(
                    "nodes", "properties", task["context_id"]
                )
                == context_before
            )
            loaded = await case.storage.load_task(task["id"])
            assert loaded is not None and loaded["status"]["state"] == "canceled"

            cancel_operation = await asyncio.wait_for(
                anext(iterator), timeout=_OPERATION_TIMEOUT_SECONDS
            )
            assert cancel_operation["operation"] == "cancel"
            assert cancel_operation["params"] == {"id": task["id"]}
            await _ack_and_stop(broker, iterator)


@pytest.mark.asyncio
async def test_native_crash_restart_recovers_and_fences_precrash_completion(
    native_a2a_engine: _RestartableNativeEngine,
) -> None:
    """Abrupt executor/engine loss recovers; the precrash executor stays fenced."""

    case = _NativeCase.open(native_a2a_engine)
    context = multiprocessing.get_context("spawn")
    first_process: multiprocessing.Process | None = None
    second_process: multiprocessing.Process | None = None
    first_parent: Any | None = None
    second_parent: Any | None = None
    try:
        task = await case.storage.submit_task("restart-context", _message())
        first_parent, first_child = context.Pipe(duplex=True)
        first_process = context.Process(
            target=_executor_process_entry,
            args=(
                native_a2a_engine.socket_path,
                case.graph_name,
                task["id"],
                first_child,
                False,
            ),
        )
        first_process.start()
        first_child.close()
        status, old_binding = await asyncio.to_thread(
            _receive_process_message, first_parent
        )
        assert status == "claimed"

        # This is an actual abrupt process loss: no broker __aexit__, nack, or
        # client close runs in the first executor before SIGKILL.
        first_process.kill()
        await asyncio.to_thread(first_process.join, _SHUTDOWN_TIMEOUT_SECONDS)
        assert first_process.exitcode is not None and first_process.exitcode < 0

        await asyncio.to_thread(case.client.close)
        await asyncio.to_thread(native_a2a_engine.crash)
        await asyncio.to_thread(native_a2a_engine.restart)
        case.reconnect()

        recovered = await case.storage.load_task(task["id"])
        assert recovered is not None
        assert recovered["status"]["state"] == "working"

        second_parent, second_child = context.Pipe(duplex=True)
        second_process = context.Process(
            target=_executor_process_entry,
            args=(
                native_a2a_engine.socket_path,
                case.graph_name,
                task["id"],
                second_child,
                True,
            ),
        )
        second_process.start()
        second_child.close()
        status, new_binding = await asyncio.to_thread(
            _receive_process_message, second_parent
        )
        assert status == "claimed"
        assert new_binding.delivery_tag != old_binding.delivery_tag
        assert new_binding.consumer != old_binding.consumer

        token = _EXECUTION_BINDING.set(old_binding)
        try:
            with pytest.raises(A2AStorageConflict, match="fence"):
                await case.storage.complete_task(
                    task["id"],
                    _model_context("stale result"),
                    new_artifacts=[],
                    new_messages=[],
                )
        finally:
            _EXECUTION_BINDING.reset(token)

        second_parent.send("complete")
        status, state = await asyncio.to_thread(
            _receive_process_message, second_parent
        )
        assert (status, state) == ("completed", "completed")
        await asyncio.to_thread(second_process.join, _SHUTDOWN_TIMEOUT_SECONDS)
        assert second_process.exitcode == 0
        persisted = await case.storage.load_task(task["id"])
        assert persisted is not None
        assert persisted["status"]["state"] == "completed"
    finally:
        for connection in (first_parent, second_parent):
            if connection is not None:
                with contextlib.suppress(BaseException):
                    connection.close()
        for process in (first_process, second_process):
            with contextlib.suppress(BaseException):
                _terminate_executor(process)
            if process is not None:
                with contextlib.suppress(BaseException):
                    process.close()
        case.close()
