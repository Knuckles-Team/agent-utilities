"""Regression coverage for test-only graph transport cleanup."""

from __future__ import annotations

import asyncio
import gc
import socket
import sys
import threading
import warnings
from pathlib import Path

import pytest


def _conftest_module():
    conftest_path = str(Path(__file__).resolve().parents[1] / "conftest.py")
    for module in list(sys.modules.values()):
        if getattr(module, "__file__", None) == conftest_path:
            return module
    pytest.skip("root conftest not loaded")


class _TrackedTransport:
    open_transports = 0
    live_threads = 0

    def __init__(self) -> None:
        self.closed = False
        self.close_calls = 0
        type(self).open_transports += 1
        type(self).live_threads += 1

    def close(self) -> None:
        self.close_calls += 1
        if not self.closed:
            self.closed = True
            type(self).open_transports -= 1
            type(self).live_threads -= 1


class _NonOwningView:
    def __init__(self) -> None:
        self.close_calls = 0
        self.clear_calls = 0
        self.tenants = _TenantNamespace()

    def close(self) -> None:
        self.close_calls += 1

    def clear(self) -> None:
        self.clear_calls += 1
        raise AssertionError("tenant deletion already owns isolated-graph cleanup")


class _TenantNamespace:
    def __init__(self) -> None:
        self.delete_calls: list[str] = []

    def delete(self, graph_name: str) -> None:
        self.delete_calls.append(graph_name)


class _AsyncTenantNamespace:
    def __init__(self, *, failure: BaseException | None = None) -> None:
        self.delete_calls: list[str] = []
        self.delete_loops: list[asyncio.AbstractEventLoop] = []
        self.delete_threads: list[int] = []
        self.failure = failure

    async def delete(self, graph_name: str) -> None:
        self.delete_calls.append(graph_name)
        self.delete_loops.append(asyncio.get_running_loop())
        self.delete_threads.append(threading.get_ident())
        if self.failure is not None:
            raise self.failure


class _AsyncClient:
    def __init__(
        self,
        tenants: _AsyncTenantNamespace,
        *,
        owning_loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self.tenants = tenants
        if owning_loop is not None:
            self._loop = owning_loop


class _AsyncRoot:
    def __init__(
        self,
        tenants: _AsyncTenantNamespace,
        *,
        owning_loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self._client = _AsyncClient(tenants, owning_loop=owning_loop)


class _SyncTenantNamespace:
    """Small stand-in for SyncEpistemicGraphClient's loop-bound namespace."""

    def __init__(
        self,
        tenants: _AsyncTenantNamespace,
        owning_loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._tenants = tenants
        self._loop = owning_loop

    def delete(self, graph_name: str) -> None:
        future = asyncio.run_coroutine_threadsafe(
            self._tenants.delete(graph_name), self._loop
        )
        future.result()


class _SyncClient:
    def __init__(
        self,
        tenants: _AsyncTenantNamespace,
        owning_loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._loop = owning_loop
        self.tenants = _SyncTenantNamespace(tenants, owning_loop)


class _SyncRoot:
    def __init__(
        self,
        tenants: _AsyncTenantNamespace,
        owning_loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._client = _SyncClient(tenants, owning_loop)


class _RootEngine:
    def __init__(self, transport: _TrackedTransport) -> None:
        self._process_root = self
        self._transport_client = transport
        self._client = _NonOwningView()
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        self._transport_client.close()


class _GraphView:
    def __init__(self, root: _RootEngine) -> None:
        self._process_root = root
        self._client = root._client


def test_created_root_transports_are_closed_once_without_resource_growth() -> None:
    """Repeated isolated-test roots release their socket/loop ownership."""
    conftest = _conftest_module()
    clients = [_TrackedTransport() for _ in range(32)]
    roots = [_RootEngine(client) for client in clients]
    views = [_GraphView(root) for root in roots]

    assert _TrackedTransport.open_transports == 32
    assert _TrackedTransport.live_threads == 32

    # Views and duplicate roots model graph-scoped consumers that share a root
    # transport.  Only the owning roots may close it, exactly once.
    conftest._close_created_graph_transports([*roots, *views, *roots])

    assert all(client.closed for client in clients)
    assert all(client.close_calls == 1 for client in clients)
    assert all(root.close_calls == 1 for root in roots)
    assert all(root._client.close_calls == 0 for root in roots)
    assert _TrackedTransport.open_transports == 0
    assert _TrackedTransport.live_threads == 0


def test_isolated_graph_cleanup_uses_one_lifecycle_delete_without_clear() -> None:
    """Tenant purge subsumes a redundant, potentially timeout-bound clear."""
    conftest = _conftest_module()
    transport = _TrackedTransport()
    root = _RootEngine(transport)
    view = _GraphView(root)

    conftest._delete_created_test_graph([root, view, root], "test_graph")

    assert root._client.tenants.delete_calls == ["test_graph"]
    assert root._client.clear_calls == 0


def test_async_graph_cleanup_awaits_native_delete_without_unawaited_warning() -> None:
    """A coroutine-returning client is awaited exactly once on a fresh loop."""
    conftest = _conftest_module()
    tenants = _AsyncTenantNamespace()
    root = _AsyncRoot(tenants)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        conftest._delete_created_test_graph([root], "async_graph")
        gc.collect()

    assert tenants.delete_calls == ["async_graph"]
    assert len(tenants.delete_loops) == 1
    assert not [
        warning for warning in caught if issubclass(warning.category, RuntimeWarning)
    ]


def test_sync_client_cleanup_does_not_deadlock_its_owning_loop() -> None:
    """Sync namespaces must be invoked outside their loop before awaiting it."""
    conftest = _conftest_module()
    owning_loop = asyncio.new_event_loop()
    loop_started = threading.Event()

    def run_owner_loop() -> None:
        asyncio.set_event_loop(owning_loop)
        loop_started.set()
        owning_loop.run_forever()

    owner_thread = threading.Thread(target=run_owner_loop)
    owner_thread.start()
    assert loop_started.wait(timeout=2)
    tenants = _AsyncTenantNamespace()
    root = _SyncRoot(tenants, owning_loop)
    try:
        conftest._delete_created_test_graph([root], "sync_client_graph")
    finally:
        owning_loop.call_soon_threadsafe(owning_loop.stop)
        owner_thread.join(timeout=2)
        owning_loop.close()

    assert tenants.delete_calls == ["sync_client_graph"]
    assert tenants.delete_loops == [owning_loop]
    assert tenants.delete_threads == [owner_thread.ident]


@pytest.mark.asyncio
async def test_async_graph_cleanup_runs_on_caller_loop_when_already_async() -> None:
    """Async callers use the owning async boundary instead of nested asyncio.run."""
    conftest = _conftest_module()
    tenants = _AsyncTenantNamespace()
    root = _AsyncRoot(tenants)

    await conftest._delete_created_test_graph_async([root], "caller_loop_graph")

    assert tenants.delete_calls == ["caller_loop_graph"]
    assert tenants.delete_loops == [asyncio.get_running_loop()]


@pytest.mark.asyncio
async def test_sync_graph_cleanup_schedules_on_a_distinct_owning_loop() -> None:
    """A running caller loop does not deadlock while a client-owned loop deletes."""
    conftest = _conftest_module()
    owning_loop = asyncio.new_event_loop()
    loop_started = threading.Event()

    def run_owner_loop() -> None:
        asyncio.set_event_loop(owning_loop)
        loop_started.set()
        owning_loop.run_forever()

    owner_thread = threading.Thread(target=run_owner_loop)
    owner_thread.start()
    assert loop_started.wait(timeout=2)
    tenants = _AsyncTenantNamespace()
    root = _AsyncRoot(tenants, owning_loop=owning_loop)
    try:
        conftest._delete_created_test_graph([root], "owned_loop_graph")
    finally:
        owning_loop.call_soon_threadsafe(owning_loop.stop)
        owner_thread.join(timeout=2)
        owning_loop.close()

    assert tenants.delete_calls == ["owned_loop_graph"]
    assert tenants.delete_loops == [owning_loop]
    assert tenants.delete_threads == [owner_thread.ident]


@pytest.mark.asyncio
async def test_sync_graph_cleanup_rejects_same_loop_blocking_without_leaking() -> None:
    """The sync boundary fails clearly; the async boundary remains usable."""
    conftest = _conftest_module()
    tenants = _AsyncTenantNamespace()
    root = _AsyncRoot(tenants)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        with pytest.raises(RuntimeError, match="cannot block a running event loop"):
            conftest._delete_created_test_graph([root], "same_loop_graph")
        gc.collect()

    assert tenants.delete_calls == []
    assert not [
        warning for warning in caught if issubclass(warning.category, RuntimeWarning)
    ]
    await conftest._delete_created_test_graph_async([root], "same_loop_graph")
    assert tenants.delete_calls == ["same_loop_graph"]


def test_graph_cleanup_failure_is_not_suppressed() -> None:
    """A failed lifecycle delete reaches teardown callers with its cause."""
    conftest = _conftest_module()
    failure = RuntimeError("engine rejected graph deletion")
    tenants = _AsyncTenantNamespace(failure=failure)
    root = _AsyncRoot(tenants)

    with pytest.raises(RuntimeError, match="engine rejected graph deletion") as raised:
        conftest._delete_created_test_graph([root], "failed_graph")

    assert raised.value is failure
    assert tenants.delete_calls == ["failed_graph"]


def test_closed_owning_loop_is_rejected_before_async_delete_is_called() -> None:
    """A closed client loop cannot manufacture an abandoned delete coroutine."""
    conftest = _conftest_module()
    owning_loop = asyncio.new_event_loop()
    owning_loop.close()
    tenants = _AsyncTenantNamespace()
    root = _AsyncRoot(tenants, owning_loop=owning_loop)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        with pytest.raises(RuntimeError, match="event loop that is not running"):
            conftest._delete_created_test_graph([root], "closed_loop_graph")
        gc.collect()

    assert tenants.delete_calls == []
    assert not [
        warning for warning in caught if issubclass(warning.category, RuntimeWarning)
    ]


def test_auxiliary_socket_cleanup_rejects_non_socket_without_deleting_it(
    tmp_path,
) -> None:
    """Socket teardown is exact and fail-closed for regular files/symlinks."""
    conftest = _conftest_module()
    regular = tmp_path / "not-a-socket"
    regular.write_text("caller data")

    with pytest.raises(RuntimeError, match="non-Unix-socket"):
        conftest._remove_owned_unix_socket(regular)
    assert regular.read_text() == "caller data"

    target = tmp_path / "real.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(target))
    link = tmp_path / "socket-link"
    link.symlink_to(target.name)
    try:
        with pytest.raises(RuntimeError, match="non-Unix-socket"):
            conftest._remove_owned_unix_socket(link)
        assert link.is_symlink()
        assert target.exists()
    finally:
        link.unlink()
        listener.close()
        target.unlink(missing_ok=True)


def test_auxiliary_socket_cleanup_is_idempotent(tmp_path) -> None:
    """A server that removes its own socket still has a clean second stop."""
    conftest = _conftest_module()
    socket_path = tmp_path / "owned.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(socket_path))
    listener.close()

    conftest._remove_owned_unix_socket(socket_path)
    conftest._remove_owned_unix_socket(socket_path)
    assert not socket_path.exists()


def test_auxiliary_socket_cleanup_rejects_foreign_owner(tmp_path, monkeypatch) -> None:
    """A valid socket inode is still protected when ownership does not match."""
    conftest = _conftest_module()
    socket_path = tmp_path / "foreign.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(socket_path))
    try:
        owner = conftest.os.getuid()
        monkeypatch.setattr(conftest.os, "getuid", lambda: owner + 1)
        with pytest.raises(RuntimeError, match="not owned"):
            conftest._remove_owned_unix_socket(socket_path)
        assert socket_path.exists()
    finally:
        listener.close()
        socket_path.unlink(missing_ok=True)


def test_auxiliary_server_stops_after_graph_delete_and_transport_close(
    tmp_path,
) -> None:
    """Explicit auxiliary registration owns the complete teardown order once."""
    conftest = _conftest_module()
    events: list[str] = []
    socket_path = tmp_path / "ordered.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(socket_path))
    listener.close()

    class _OrderedTenants(_AsyncTenantNamespace):
        async def delete(self, graph_name: str) -> None:
            events.append("graph-delete")
            await super().delete(graph_name)

    class _OrderedRoot(_AsyncRoot):
        def close(self) -> None:
            events.append("transport-close")

    class _Server:
        def stop(self) -> None:
            events.append("server-stop")

    lifecycle = conftest._GraphTestLifecycle()
    lifecycle.track_engine(_OrderedRoot(_OrderedTenants()), "ordered_graph")
    registration = lifecycle.register_auxiliary_engine(
        _Server(), socket_path=socket_path
    )

    try:
        registration.stop()
        registration.stop()
    finally:
        socket_path.unlink(missing_ok=True)

    assert events == ["graph-delete", "transport-close", "server-stop"]
    assert not socket_path.exists()


def test_auxiliary_server_stops_even_when_graph_cleanup_fails(tmp_path) -> None:
    """A teardown failure is surfaced only after the auxiliary process is stopped."""
    conftest = _conftest_module()
    events: list[str] = []
    failure = RuntimeError("delete rejected")
    socket_path = tmp_path / "failed.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(socket_path))
    listener.close()

    class _FailingRoot(_AsyncRoot):
        def close(self) -> None:
            events.append("transport-close")

    class _Server:
        def stop(self) -> None:
            events.append("server-stop")

    lifecycle = conftest._GraphTestLifecycle()
    lifecycle.track_engine(
        _FailingRoot(_AsyncTenantNamespace(failure=failure)), "failed_ordered_graph"
    )
    registration = lifecycle.register_auxiliary_engine(
        _Server(), socket_path=socket_path
    )

    try:
        with pytest.raises(RuntimeError, match="auxiliary engine teardown failed"):
            registration.stop()
    finally:
        socket_path.unlink(missing_ok=True)

    assert events == ["transport-close", "server-stop"]
    assert not socket_path.exists()


def test_auxiliary_socket_failure_is_reported_after_server_stop(tmp_path) -> None:
    """A bad socket entry cannot hide that the auxiliary process was reaped."""
    conftest = _conftest_module()
    events: list[str] = []
    socket_path = tmp_path / "unexpected-file"
    socket_path.write_text("must not be deleted")

    class _Server:
        def stop(self) -> None:
            events.append("server-stop")

    lifecycle = conftest._GraphTestLifecycle()
    registration = lifecycle.register_auxiliary_engine(
        _Server(), socket_path=socket_path
    )

    with pytest.raises(RuntimeError, match="auxiliary engine teardown failed"):
        registration.stop()

    assert events == ["server-stop"]
    assert socket_path.read_text() == "must not be deleted"
