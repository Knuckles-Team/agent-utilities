"""LANE H-http: unit tests for the browser-path + API-gateway harness stage
module (``scripts/_harness_browser_api.py``).

Everything here runs against a local fake HTTP server or a monkeypatched
identity/admission chain — no live cluster required, per the lane brief.
"""

from __future__ import annotations

import contextlib
import http.server
import importlib.util
import socket
import sys
import threading
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def _load_module() -> ModuleType:
    source = Path(__file__).parents[3] / "scripts" / "_harness_browser_api.py"
    spec = importlib.util.spec_from_file_location("_harness_browser_api", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register in sys.modules BEFORE exec: this module's dataclasses use
    # `from __future__ import annotations`, so the dataclass machinery
    # resolves field type strings via `sys.modules[cls.__module__]` — without
    # this line that lookup returns None and every dataclass decoration
    # raises AttributeError.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeHandler(http.server.BaseHTTPRequestHandler):
    """Returns a fixed status per exact path, from the class-level ROUTES map.
    A path not present in ROUTES answers 404 (never accidentally "passes")."""

    ROUTES: dict[str, int] = {}

    def do_GET(self) -> None:
        status = self.ROUTES.get(self.path, 404)
        self.send_response(status)
        if status in (301, 302, 303, 307, 308):
            self.send_header("Location", "/elsewhere")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:  # quiet tests
        pass


@contextlib.contextmanager
def _fake_server(routes: dict[str, int]) -> Iterator[str]:
    handler_cls = type("_RoutedHandler", (_FakeHandler,), {"ROUTES": dict(routes)})
    server = http.server.HTTPServer(("127.0.0.1", 0), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def _unreachable_url() -> str:
    """A loopback URL nothing is listening on, to force a connection failure."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()  # freed immediately; nothing rebinds it during the test
    return f"http://127.0.0.1:{port}"


# ---------------------------------------------------------------------------
# (a) a 5xx on a browser path FAILS
# ---------------------------------------------------------------------------
def test_browser_5xx_fails() -> None:
    mod = _load_module()
    routes = {path: 200 for path in mod._BROWSER_PATHS}
    routes["/graph"] = 503

    with _fake_server(routes) as base_url:
        with pytest.raises(mod.BrowserStageError) as excinfo:
            mod.stage_browser(base_url, timeout=5.0)

    report = excinfo.value.report
    assert report.stage == "browser"
    assert report.ok is False
    failing = [c for c in report.checks if c.path == "/graph"]
    assert failing and failing[0].status == 503 and failing[0].ok is False
    # every path is reported, not just the failing one
    assert {c.path for c in report.checks} == set(mod._BROWSER_PATHS)


# ---------------------------------------------------------------------------
# (b) a 401 on a browser path PASSES
# ---------------------------------------------------------------------------
def test_browser_401_passes() -> None:
    mod = _load_module()
    routes = {path: 401 for path in mod._BROWSER_PATHS}

    with _fake_server(routes) as base_url:
        report = mod.stage_browser(base_url, timeout=5.0)

    assert report.ok is True
    assert report.checks
    assert all(check.status == 401 and check.ok for check in report.checks)


def test_browser_redirect_status_is_graded_not_followed() -> None:
    """302/303 must be graded as themselves, not silently followed."""
    mod = _load_module()
    routes = {path: 200 for path in mod._BROWSER_PATHS}
    routes["/auth/login"] = 302

    with _fake_server(routes) as base_url:
        report = mod.stage_browser(base_url, timeout=5.0)

    assert report.ok is True
    login = [c for c in report.checks if c.path == "/auth/login"][0]
    assert login.status == 302
    assert login.ok is True


# ---------------------------------------------------------------------------
# (c) a 401 on a service-API path FAILS
# ---------------------------------------------------------------------------
def test_api_gateway_401_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_module()
    monkeypatch.setattr(
        "agent_utilities.security.request_identity.acquire_process_identity_token",
        lambda config: "fake-bearer-token",
    )
    routes = {path: 200 for path in mod._api_gateway_paths()}
    routes["/api/registry/tools?limit=5"] = 401

    with _fake_server(routes) as base_url:
        with pytest.raises(mod.ApiGatewayStageError) as excinfo:
            mod.stage_api_gateway(base_url, timeout=5.0, config="unused-in-test")

    report = excinfo.value.report
    assert report.stage == "api_gateway"
    assert report.ok is False
    failing = [c for c in report.checks if c.path == "/api/registry/tools?limit=5"]
    assert failing and failing[0].status == 401 and failing[0].ok is False


def test_api_gateway_200_passes_and_reports_latency_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module()
    monkeypatch.setattr(
        "agent_utilities.security.request_identity.acquire_process_identity_token",
        lambda config: "fake-bearer-token",
    )
    routes = {path: 200 for path in mod._api_gateway_paths()}

    with _fake_server(routes) as base_url:
        report = mod.stage_api_gateway(base_url, timeout=5.0, config="unused-in-test")

    assert report.ok is True
    assert all(check.ok for check in report.checks)
    # the per-endpoint latency comparison the lane brief calls out explicitly
    assert "latency_signal" in report.detail
    assert "registry/tools=" in report.detail
    assert "enhanced/tools=" in report.detail


def test_api_gateway_bearer_acquisition_failure_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(config: object) -> str:
        raise RuntimeError("Graph process identity acquisition failed")

    mod = _load_module()
    monkeypatch.setattr(
        "agent_utilities.security.request_identity.acquire_process_identity_token", _boom
    )

    with pytest.raises(mod.ApiGatewayStageError) as excinfo:
        mod.stage_api_gateway("http://127.0.0.1:1", timeout=2.0, config="unused-in-test")

    report = excinfo.value.report
    assert report.ok is False
    assert "bearer" in report.detail


# ---------------------------------------------------------------------------
# (d) a connection refusal FAILS
# ---------------------------------------------------------------------------
def test_connection_refusal_fails() -> None:
    mod = _load_module()
    base_url = _unreachable_url()

    with pytest.raises(mod.BrowserStageError) as excinfo:
        mod.stage_browser(base_url, timeout=2.0)

    report = excinfo.value.report
    assert report.ok is False
    assert report.checks
    assert all(check.status == 0 and not check.ok for check in report.checks)


# ---------------------------------------------------------------------------
# admission stage: surrogate labeling (no live engine required — the
# identity/admission chain is monkeypatched end to end)
# ---------------------------------------------------------------------------
def test_admission_stage_succeeds_and_is_labeled_as_surrogate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module()
    fake_actor = SimpleNamespace(actor_id="probe-actor")
    fake_session = SimpleNamespace(actor=fake_actor, engine_verified_context=lambda: {})
    fake_result = SimpleNamespace(tenant_slug="homelab", role="Agent", all_admitted=True)

    monkeypatch.setattr(
        "agent_utilities.security.request_identity.acquire_process_identity_token",
        lambda config: "fake-bearer-token",
    )
    monkeypatch.setattr(
        "agent_utilities.security.request_identity.mint_actor_from_token_sync",
        lambda token: fake_actor,
    )
    monkeypatch.setattr(
        "agent_utilities.security.request_identity.mint_graph_session",
        lambda actor: fake_session,
    )
    monkeypatch.setattr(
        "agent_utilities.security.brain_context.use_actor",
        lambda actor: contextlib.nullcontext(actor),
    )
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.core.session.use_session",
        lambda session: contextlib.nullcontext(session),
    )
    monkeypatch.setattr(
        "agent_utilities.security.tenant_admission_cli.run_tenant_admission",
        lambda tenant_slug, principals, apply: fake_result,
    )

    report = mod.stage_admission(tenant_slug="homelab", config="unused-in-test")

    assert report.ok is True
    assert report.surrogate is True
    assert "SURROGATE" in report.surrogate_note
    assert "sign in" in report.surrogate_note
    assert "SURROGATE" in report.detail


def test_admission_stage_failure_is_still_labeled_as_surrogate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module()

    def _boom(config: object) -> str:
        raise RuntimeError("no verified identity source configured")

    monkeypatch.setattr(
        "agent_utilities.security.request_identity.acquire_process_identity_token", _boom
    )

    with pytest.raises(mod.AdmissionStageError) as excinfo:
        mod.stage_admission(tenant_slug="homelab", config="unused-in-test")

    report = excinfo.value.report
    assert report.ok is False
    assert report.surrogate is True
    assert "SURROGATE" in report.surrogate_note
