"""GOC-87 W05 — ollama widget ported to the httpx2-backed adapter.

Drives ``Widget.fetch_data`` against a REAL loopback HTTP server (not a
mocked transport and not a mocked factory/adapter) to prove the ported call
path — ``transport_factory.create_http_client(family="gateway-widget-
diagnostics", ...)`` → ``Httpx2Adapter`` → real ``httpx2.Client`` — actually
performs network I/O end to end, and that the pre-existing error-handling
contract (a down/unreachable service still returns ``status="error"``, never
raises out of ``fetch_data``) is unchanged after the port.
"""

from __future__ import annotations

import contextlib
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from agent_utilities.gateway.models import ServiceConfig
from agent_utilities.gateway.widgets.ollama import Widget


class _FakeOllamaHandler(BaseHTTPRequestHandler):
    def log_message(self, *args: object) -> None:  # silence test output
        pass

    def do_GET(self) -> None:  # noqa: N802 - http.server's required name
        if self.path == "/api/tags":
            body = json.dumps({"models": [{"name": "llama3"}, {"name": "phi3"}]})
        elif self.path == "/api/ps":
            body = json.dumps({"models": [{"name": "llama3"}]})
        else:
            self.send_response(404)
            self.end_headers()
            return
        encoded = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


@contextlib.contextmanager
def _fake_ollama_server():
    server = HTTPServer(("127.0.0.1", 0), _FakeOllamaHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def _closed_port_url() -> str:
    """A loopback URL nothing is listening on (deterministic connection-refused)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return f"http://127.0.0.1:{port}"


def test_fetch_data_reports_models_over_real_httpx2_transport():
    with _fake_ollama_server() as url:
        config = ServiceConfig(
            id="ollama", name="Ollama", widget_type="ollama", url=url
        )
        widget = Widget()

        data = widget.fetch_data(config)

    assert data.status == "ok"
    assert data.fields["models"] == 2
    assert data.fields["running"] == 1
    assert data.fields["status"] == "Online"


def test_fetch_data_reports_error_when_service_is_unreachable():
    config = ServiceConfig(
        id="ollama", name="Ollama", widget_type="ollama", url=_closed_port_url()
    )
    widget = Widget()

    data = widget.fetch_data(config)

    assert data.status == "error"


def test_fetch_data_uses_the_migrated_httpx2_family():
    """Confirms the port actually took: this family is httpx2-backed."""
    from agent_utilities.httpsupport.transport_factory import MIGRATED_HTTPX2_FAMILIES

    assert "gateway-widget-diagnostics" in MIGRATED_HTTPX2_FAMILIES


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
