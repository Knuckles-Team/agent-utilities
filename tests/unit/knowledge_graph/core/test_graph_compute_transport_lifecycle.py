"""Root graph transports own shutdown; routed views never do."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

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
