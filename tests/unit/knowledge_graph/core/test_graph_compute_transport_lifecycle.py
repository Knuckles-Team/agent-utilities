"""Root graph transports own shutdown; routed views never do."""

from __future__ import annotations

import os
import select
import socket
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine


class _OwningTransport:
    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class _NonOwningClientView:
    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class _BridgeBackend:
    def __init__(self) -> None:
        self.subscribed = threading.Event()
        self.unsubscribed = threading.Event()
        self.handler = None

    async def subscribe(self, _topic, _group, handler) -> None:
        self.handler = handler
        self.subscribed.set()

    async def unsubscribe(self, _topic, _group, handler) -> None:
        assert handler is self.handler
        self.unsubscribed.set()


class _QueuedLoop:
    def __init__(self) -> None:
        self.scheduled = 0

    def is_closed(self) -> bool:
        return False

    def call_soon_threadsafe(self, _callback) -> None:
        self.scheduled += 1


class _AsyncStop:
    def set(self) -> None:
        return None


class _NativeLikeTransport:
    """Real OS resources matching one native client's ownership shape."""

    def __init__(self) -> None:
        self.close_calls = 0
        self.closed = False
        self._reader, self._writer = socket.socketpair()
        self._epoll = select.epoll()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._stop.wait, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self.close_calls += 1
        if self.closed:
            return
        self.closed = True
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._reader.close()
        self._writer.close()
        self._epoll.close()


def _fd_count() -> int:
    return len(os.listdir("/proc/self/fd"))


def _patch_duplicate_constructor(
    monkeypatch, transports: list[_NativeLikeTransport]
) -> None:
    """Make the final singleton race deterministic without a live engine."""
    from epistemic_graph.client import SyncEpistemicGraphClient

    from agent_utilities.core import config as core_config
    from agent_utilities.knowledge_graph.core import (
        engine_breaker,
        engine_resolver,
        graph_compute,
        session,
        shard_topology,
    )

    class _Breaker:
        def before_call(self) -> None:
            return None

        def record_success(self) -> None:
            return None

    def connect(**_kwargs):
        transport = _NativeLikeTransport()
        transports.append(transport)
        return transport

    def start_bridge(self) -> None:
        self._event_bridge_stop = threading.Event()
        self._event_bridge_thread = threading.Thread(
            target=self._event_bridge_stop.wait,
            daemon=True,
        )
        self._event_bridge_thread.start()
        # Simulate a peer winning the singleton race after this constructor
        # acquired its transport and started its bridge.
        GraphComputeEngine._PROCESS_ENGINE = object()

    def stop_bridge(self) -> None:
        stop = self._event_bridge_stop
        worker = self._event_bridge_thread
        if stop is not None:
            stop.set()
        if worker is not None:
            worker.join(timeout=1.0)
        self._event_bridge_stop = None
        self._event_bridge_thread = None

    monkeypatch.setattr(
        core_config,
        "AgentConfig",
        lambda: SimpleNamespace(kg_default_graph="__commons__"),
    )
    monkeypatch.setattr(
        engine_resolver,
        "resolve_engine",
        lambda *_args: engine_resolver.ResolvedEngine(
            endpoint="unix:///tmp/graph-compute-duplicate-test.sock",
            auth_secret="test-secret",
            mode="shared",
            autostart_allowed=False,
            idle_shutdown_secs=0,
        ),
    )
    monkeypatch.setattr(
        shard_topology, "resolve_endpoints", lambda _config: ["unix://x"]
    )
    monkeypatch.setattr(shard_topology, "record_shard_connect", lambda *_args: None)
    monkeypatch.setattr(session, "graph_session_required", lambda: True)
    monkeypatch.setattr(engine_breaker, "get_breaker", lambda _endpoint: _Breaker())
    monkeypatch.setattr(
        engine_breaker, "wrap_client_with_breaker", lambda view, _breaker: view
    )
    monkeypatch.setattr(SyncEpistemicGraphClient, "connect", staticmethod(connect))
    monkeypatch.setattr(graph_compute, "_sync_client_view", lambda _transport: object())
    monkeypatch.setattr(graph_compute, "setting", lambda _name: "")
    monkeypatch.setattr(GraphComputeEngine, "_start_event_bridge", start_bridge)
    monkeypatch.setattr(GraphComputeEngine, "_stop_event_bridge", stop_bridge)
    monkeypatch.setattr(GraphComputeEngine, "_PROCESS_ENGINE", None)


def test_root_close_uses_owning_transport_not_routed_view(monkeypatch) -> None:
    """The view's no-op close cannot leak the root's socket and loop."""
    transport = _OwningTransport()
    routed_view = _NonOwningClientView()
    root = object.__new__(GraphComputeEngine)
    root._process_root = root
    root._transport_client = transport
    root._transport_closed = False
    root._client = routed_view
    root._event_bridge_stop = None
    root._event_bridge_thread = None
    root._event_bridge_loop = None
    root._event_bridge_async_stop = None
    monkeypatch.setattr(GraphComputeEngine, "_PROCESS_ENGINE", root)

    root.close()
    root.close()

    assert transport.close_calls == 1
    assert routed_view.close_calls == 0
    assert root._transport_client is None
    assert GraphComputeEngine.get_active() is None


def test_scoped_view_close_never_closes_root_transport() -> None:
    transport = _OwningTransport()
    root = object.__new__(GraphComputeEngine)
    root._process_root = root
    root._transport_client = transport
    root._transport_closed = False
    view = object.__new__(GraphComputeEngine)
    view._process_root = root

    view.close()

    assert transport.close_calls == 0


def test_root_close_stops_and_joins_its_event_bridge(monkeypatch) -> None:
    """Each isolated root reclaims its event loop and worker thread."""
    from agent_utilities.knowledge_graph.core import event_backend

    backend = _BridgeBackend()
    monkeypatch.setattr(event_backend, "get_event_backend", lambda: backend)
    transport = _OwningTransport()
    root = object.__new__(GraphComputeEngine)
    root._process_root = root
    root._transport_client = transport
    root._transport_closed = False
    root._event_bridge_stop = None
    root._event_bridge_thread = None
    root._event_bridge_loop = None
    root._event_bridge_async_stop = None
    root._client = _NonOwningClientView()

    root._start_event_bridge()
    assert backend.subscribed.wait(timeout=1.0)
    worker = root._event_bridge_thread
    assert worker is not None and worker.is_alive()

    root.close()

    assert backend.unsubscribed.wait(timeout=1.0)
    assert not worker.is_alive()
    assert transport.close_calls == 1
    assert root._event_bridge_thread is None
    assert root._event_bridge_loop is None
    assert root._event_bridge_async_stop is None


def test_stop_queues_async_signal_during_bridge_startup() -> None:
    """A close before the worker loop runs still releases its awaiter."""
    root = object.__new__(GraphComputeEngine)
    root._process_root = root
    root._event_bridge_stop = threading.Event()
    root._event_bridge_thread = None
    root._event_bridge_loop = loop = _QueuedLoop()
    root._event_bridge_async_stop = _AsyncStop()

    root._stop_event_bridge()

    assert root._event_bridge_stop is None
    assert loop.scheduled == 1


def test_repeated_root_close_retries_bridge_shutdown_without_reclosing_transport() -> (
    None
):
    root = object.__new__(GraphComputeEngine)
    root._process_root = root
    root._transport_closed = True
    root._transport_client = None
    root._stop_event_bridge = MagicMock()

    root.close()

    root._stop_event_bridge.assert_called_once_with()


@pytest.mark.skipif(
    not os.path.isdir("/proc/self/fd") or not hasattr(select, "epoll"),
    reason="requires Linux /proc FD accounting and epoll",
)
def test_failed_duplicate_construction_releases_native_resources(monkeypatch) -> None:
    """A post-connect singleton rejection leaves no socket, epoll, or thread."""
    transports: list[_NativeLikeTransport] = []
    _patch_duplicate_constructor(monkeypatch, transports)
    baseline_fds = _fd_count()
    baseline_threads = threading.active_count()

    for _ in range(32):
        with pytest.raises(
            RuntimeError, match="Concurrent duplicate graph transport rejected"
        ):
            GraphComputeEngine()
        assert transports[-1].closed
        assert transports[-1].close_calls == 1
        assert _fd_count() == baseline_fds
        assert threading.active_count() == baseline_threads
        GraphComputeEngine._PROCESS_ENGINE = None

    assert len(transports) == 32
    assert all(transport.closed for transport in transports)
