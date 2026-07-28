"""Regression coverage for test-only graph transport cleanup."""

from __future__ import annotations

import sys
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
        type(self).open_transports += 1
        type(self).live_threads += 1

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            type(self).open_transports -= 1
            type(self).live_threads -= 1


class _RootEngine:
    def __init__(self, client: _TrackedTransport) -> None:
        self._process_root = self
        self._client = client


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
    assert _TrackedTransport.open_transports == 0
    assert _TrackedTransport.live_threads == 0
