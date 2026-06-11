"""Tests for gateway multi-worker readiness (CONCEPT:OS-5.23).

The pre-fork model itself (os.fork + shared socket) is exercised in
deployment; here we pin the safety contract around it: the default is 1
(single process — historical behaviour), pytest and the terminal UI force 1,
and the shared listen socket is bound inheritable.
"""

from __future__ import annotations

import pytest

from agent_utilities.core.config import config
from agent_utilities.server import _bind_gateway_socket, _resolve_gateway_workers


class TestResolveGatewayWorkers:
    def test_default_is_single_process(self):
        assert config.gateway_workers == 1
        assert _resolve_gateway_workers(is_pytest=False, enable_terminal_ui=False) == 1

    def test_config_value_honoured(self, monkeypatch):
        monkeypatch.setattr(config, "gateway_workers", 4)
        assert _resolve_gateway_workers(is_pytest=False, enable_terminal_ui=False) == 4

    def test_pytest_forces_single_worker(self, monkeypatch):
        monkeypatch.setattr(config, "gateway_workers", 4)
        assert _resolve_gateway_workers(is_pytest=True, enable_terminal_ui=False) == 1

    def test_terminal_ui_forces_single_worker(self, monkeypatch):
        monkeypatch.setattr(config, "gateway_workers", 4)
        assert _resolve_gateway_workers(is_pytest=False, enable_terminal_ui=True) == 1

    def test_garbage_value_falls_back_to_one(self, monkeypatch):
        monkeypatch.setattr(config, "gateway_workers", 0)
        assert _resolve_gateway_workers(is_pytest=False, enable_terminal_ui=False) == 1
        monkeypatch.setattr(config, "gateway_workers", -3)
        assert _resolve_gateway_workers(is_pytest=False, enable_terminal_ui=False) == 1


class TestBindGatewaySocket:
    def test_socket_is_bound_and_inheritable(self):
        sock = _bind_gateway_socket("127.0.0.1", 0)  # ephemeral port
        try:
            host, port = sock.getsockname()[:2]
            assert host == "127.0.0.1"
            assert port > 0
            # forked workers must inherit the listen socket
            assert sock.get_inheritable() is True
        finally:
            sock.close()

    def test_rebinding_same_port_fails_fast(self):
        sock = _bind_gateway_socket("127.0.0.1", 0)
        try:
            port = sock.getsockname()[1]
            sock.listen(8)
            with pytest.raises(OSError):
                other = _bind_gateway_socket("127.0.0.1", port)
                other.listen(8)  # pragma: no cover - bind already raised
        finally:
            sock.close()
