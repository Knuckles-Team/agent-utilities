"""Unit tests for scripts/dual_principal_validation.py.

Hermetic -- no live cluster, no Keycloak, no kubectl. Exercises the pure
fingerprint/classify logic and the HTTP call/sweep plumbing against a local
fake HTTP server, following the ``tests/unit/scripts/test_harness_browser_api.py``
convention (module loaded via ``importlib`` from its file path, since this
module's dataclasses use ``from __future__ import annotations`` and need to
be registered in ``sys.modules`` before ``exec_module`` for field-type
resolution to work).

Provisioning (Keycloak admin, browser OIDC dance, k8s Secret reads) is
exercised by the live run documented in this lane's report, not here --
mirroring how ``scripts/delegation_probe.py``/``full_validation_harness.py``
keep their live-cluster legs out of the hermetic unit-test tier.
"""

from __future__ import annotations

import base64
import contextlib
import http.server
import importlib.util
import json
import socket
import sys
import threading
from collections.abc import Iterator
from types import ModuleType

import pytest


def _load_module() -> ModuleType:
    from pathlib import Path

    source = Path(__file__).parents[3] / "scripts" / "dual_principal_validation.py"
    spec = importlib.util.spec_from_file_location("dual_principal_validation", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeHandler(http.server.BaseHTTPRequestHandler):
    """Serves a fixed (status, json-body) per exact path from a class-level
    ROUTES map. A path not present in ROUTES answers 404 with an empty body."""

    ROUTES: dict[str, tuple[int, object]] = {}

    def _respond(self) -> None:
        status, body = self.ROUTES.get(self.path, (404, None))
        payload = b"" if body is None else json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        if payload:
            self.wfile.write(payload)

    def do_GET(self) -> None:
        self._respond()

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)
        self._respond()

    def log_message(self, format: str, *args: object) -> None:  # quiet tests
        pass


@contextlib.contextmanager
def _fake_server(routes: dict[str, tuple[int, object]]) -> Iterator[str]:
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
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return f"http://127.0.0.1:{port}"


# ---------------------------------------------------------------------------
# response_fingerprint()
# ---------------------------------------------------------------------------
def test_fingerprint_list_counts_items() -> None:
    mod = _load_module()
    raw = json.dumps([{"a": 1}, {"a": 2}, {"a": 3}]).encode()
    fp = mod.response_fingerprint("application/json", raw, truncated=False)
    assert fp["kind"] == "list"
    assert fp["count"] == 3


def test_fingerprint_dict_prefers_explicit_count_field() -> None:
    mod = _load_module()
    raw = json.dumps({"count": 7, "items": [1, 2, 3]}).encode()
    fp = mod.response_fingerprint("application/json", raw, truncated=False)
    assert fp["kind"] == "dict"
    assert fp["count"] == 7  # explicit "count" wins over len(items)


def test_fingerprint_dict_falls_back_to_known_list_field() -> None:
    mod = _load_module()
    raw = json.dumps({"tools": [1, 2, 3, 4]}).encode()
    fp = mod.response_fingerprint("application/json", raw, truncated=False)
    assert fp["kind"] == "dict"
    assert fp["count"] == 4


def test_fingerprint_truncated_response_has_no_count() -> None:
    """A truncated read must never fabricate a count from a partial body."""
    mod = _load_module()
    raw = json.dumps([{"a": i} for i in range(50)]).encode()
    fp = mod.response_fingerprint("application/json", raw, truncated=True)
    assert fp["kind"] == "list"
    assert fp["count"] is None


def test_fingerprint_non_json_body() -> None:
    mod = _load_module()
    fp = mod.response_fingerprint("text/plain", b"not json at all {{{", truncated=False)
    assert fp["kind"] == "non-json"


def test_fingerprint_empty_body() -> None:
    mod = _load_module()
    assert mod.response_fingerprint("application/json", b"", truncated=False) == {
        "kind": "empty"
    }


# ---------------------------------------------------------------------------
# classify() -- the MATCH / DISCREPANCY / UNTESTED verdict logic. This is the
# single most important piece of logic in the harness per the explicit user
# directive: two principals failing identically must NEVER read as a match.
# ---------------------------------------------------------------------------
def _result(mod: ModuleType, status: int, fp: dict | None = None) -> object:
    return mod.RouteResult(status, 0.01, fp or {"kind": "empty"}, False)


def test_classify_both_2xx_same_shape_is_match() -> None:
    mod = _load_module()
    service = _result(mod, 200, {"kind": "list", "count": 5})
    human = _result(mod, 200, {"kind": "list", "count": 5})
    verdict, _ = mod.classify(service, human, tolerance=0)
    assert verdict == mod.VERDICT_MATCH


def test_classify_status_mismatch_is_discrepancy() -> None:
    mod = _load_module()
    service = _result(mod, 200, {"kind": "dict", "count": 10})
    human = _result(mod, 503, {"kind": "dict", "count": None})
    verdict, reason = mod.classify(service, human, tolerance=0)
    assert verdict == mod.VERDICT_DISCREPANCY
    assert "status mismatch" in reason


def test_classify_identical_non_2xx_is_untested_not_match() -> None:
    """The explicit user directive: both sides 401ing identically is NOT
    parity -- it means the route was never actually exercised."""
    mod = _load_module()
    service = _result(mod, 401)
    human = _result(mod, 401)
    verdict, reason = mod.classify(service, human, tolerance=0)
    assert verdict == mod.VERDICT_UNTESTED
    assert verdict != mod.VERDICT_MATCH
    assert "not proof of parity" in reason


def test_classify_identical_404_is_untested() -> None:
    mod = _load_module()
    service = _result(mod, 404)
    human = _result(mod, 404)
    verdict, _ = mod.classify(service, human, tolerance=0)
    assert verdict == mod.VERDICT_UNTESTED


def test_classify_both_transport_failure_is_untested() -> None:
    mod = _load_module()
    service = _result(mod, 0, {"kind": "transport-error"})
    human = _result(mod, 0, {"kind": "transport-error"})
    verdict, reason = mod.classify(service, human, tolerance=0)
    assert verdict == mod.VERDICT_UNTESTED
    assert "transport failure" in reason


def test_classify_one_transport_failure_is_discrepancy() -> None:
    mod = _load_module()
    service = _result(mod, 200, {"kind": "dict", "count": 1})
    human = _result(mod, 0, {"kind": "transport-error"})
    verdict, _ = mod.classify(service, human, tolerance=0)
    assert verdict == mod.VERDICT_DISCREPANCY


def test_classify_shape_mismatch_is_discrepancy() -> None:
    mod = _load_module()
    service = _result(mod, 200, {"kind": "list", "count": 3})
    human = _result(mod, 200, {"kind": "dict", "count": 3})
    verdict, reason = mod.classify(service, human, tolerance=0)
    assert verdict == mod.VERDICT_DISCREPANCY
    assert "shape mismatch" in reason


def test_classify_count_mismatch_beyond_tolerance_is_discrepancy() -> None:
    mod = _load_module()
    service = _result(mod, 200, {"kind": "list", "count": 53})
    human = _result(mod, 200, {"kind": "list", "count": 43})
    verdict, reason = mod.classify(service, human, tolerance=0)
    assert verdict == mod.VERDICT_DISCREPANCY
    assert "count mismatch" in reason


def test_classify_count_mismatch_within_tolerance_is_match() -> None:
    mod = _load_module()
    service = _result(mod, 200, {"kind": "list", "count": 100})
    human = _result(mod, 200, {"kind": "list", "count": 98})
    verdict, _ = mod.classify(service, human, tolerance=2)
    assert verdict == mod.VERDICT_MATCH


def test_classify_missing_count_signal_does_not_fabricate_discrepancy() -> None:
    """When neither side has a count signal (e.g. a plain object with no
    list/count field), agreement on status+shape alone is a MATCH."""
    mod = _load_module()
    service = _result(mod, 200, {"kind": "dict", "count": None})
    human = _result(mod, 200, {"kind": "dict", "count": None})
    verdict, _ = mod.classify(service, human, tolerance=0)
    assert verdict == mod.VERDICT_MATCH


# ---------------------------------------------------------------------------
# call_route() against a real (fake) HTTP server
# ---------------------------------------------------------------------------
def test_call_route_records_status_and_fingerprint() -> None:
    mod = _load_module()
    with _fake_server({"/x": (200, {"count": 4, "items": [1, 2, 3, 4]})}) as base_url:
        ctx = mod.tls_context(None, insecure=True)
        opener = mod.build_opener(ctx)
        result = mod.call_route(
            opener, base_url, "GET", "/x", None, headers={}, timeout=5.0
        )
    assert result.status == 200
    assert result.fingerprint["kind"] == "dict"
    assert result.fingerprint["count"] == 4


def test_call_route_reports_non_2xx_status() -> None:
    mod = _load_module()
    with _fake_server({"/x": (503, {"status": "unavailable"})}) as base_url:
        ctx = mod.tls_context(None, insecure=True)
        opener = mod.build_opener(ctx)
        result = mod.call_route(
            opener, base_url, "GET", "/x", None, headers={}, timeout=5.0
        )
    assert result.status == 503


def test_call_route_transport_failure_is_status_zero() -> None:
    mod = _load_module()
    ctx = mod.tls_context(None, insecure=True)
    opener = mod.build_opener(ctx)
    result = mod.call_route(
        opener, _unreachable_url(), "GET", "/x", None, headers={}, timeout=2.0
    )
    assert result.status == 0
    assert result.fingerprint["kind"] == "transport-error"


def test_call_route_does_not_follow_redirects() -> None:
    """The route-sweep opener must grade a 30x as itself (matching
    scripts/_harness_browser_api.py's convention), never silently follow it
    -- an API redirecting a bad credential elsewhere must not be masked."""
    mod = _load_module()
    with _fake_server({"/x": (302, None)}) as base_url:
        ctx = mod.tls_context(None, insecure=True)
        opener = mod.build_opener(ctx)
        result = mod.call_route(
            opener, base_url, "GET", "/x", None, headers={}, timeout=5.0
        )
    assert result.status == 302


def test_call_route_sends_authorization_header() -> None:
    mod = _load_module()
    seen_auth = {}

    class _AuthCapturingHandler(_FakeHandler):
        ROUTES = {"/x": (200, {"ok": True})}

        def do_GET(self) -> None:  # noqa: N802
            seen_auth["value"] = self.headers.get("Authorization")
            self._respond()

    server = http.server.HTTPServer(("127.0.0.1", 0), _AuthCapturingHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_port}"
        ctx = mod.tls_context(None, insecure=True)
        opener = mod.build_opener(ctx)
        mod.call_route(
            opener,
            base_url,
            "GET",
            "/x",
            None,
            headers={"Authorization": "Bearer fake-token-value"},
            timeout=5.0,
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
    assert seen_auth["value"] == "Bearer fake-token-value"


# ---------------------------------------------------------------------------
# run_route_sweep() -- both principals hit the same fake server; verifies
# the sweep correctly pairs results per route and classifies them.
# ---------------------------------------------------------------------------
def test_run_route_sweep_end_to_end_against_fake_server() -> None:
    mod = _load_module()
    routes: dict[str, tuple[int, object]] = {
        "/match": (200, {"count": 2, "items": [1, 2]}),
        "/discrepancy": (200, {"count": 9, "items": list(range(9))}),
        "/guarded": (401, {"detail": "no"}),
    }
    with _fake_server(routes) as base_url:
        ctx = mod.tls_context(None, insecure=True)
        service_ctx = mod.PrincipalCtx("service", mod.build_opener(ctx), {})
        human_ctx = mod.PrincipalCtx("human", mod.build_opener(ctx), {})
        route_table: list[tuple[str, str, dict | None]] = [
            ("GET", "/match", None),
            ("GET", "/guarded", None),
        ]
        verdicts = mod.run_route_sweep(
            route_table,
            [service_ctx, human_ctx],
            base_url,
            concurrency=2,
            timeout=5.0,
            tolerance=0,
        )
    by_path = {v.path: v for v in verdicts}
    assert by_path["/match"].verdict == mod.VERDICT_MATCH
    # both principals get an identical 401 against this fake server -- must
    # be UNTESTED, never MATCH, per the core directive.
    assert by_path["/guarded"].verdict == mod.VERDICT_UNTESTED


def test_run_route_sweep_detects_discrepancy_between_distinct_servers() -> None:
    """Simulates a real human-vs-service split by pointing the two
    principals' openers at two different fake servers with different
    response bodies for the same nominal path -- exactly what a real
    tenant/principal-scoped divergence looks like from the caller's side."""
    mod = _load_module()
    service_routes = {"/x": (200, {"count": 53, "items": list(range(53))})}
    human_routes = {"/x": (200, {"count": 43, "items": list(range(43))})}
    with (
        _fake_server(service_routes) as service_url,
        _fake_server(human_routes) as human_url,
    ):
        ctx = mod.tls_context(None, insecure=True)
        service_result = mod.call_route(
            mod.build_opener(ctx),
            service_url,
            "GET",
            "/x",
            None,
            headers={},
            timeout=5.0,
        )
        human_result = mod.call_route(
            mod.build_opener(ctx), human_url, "GET", "/x", None, headers={}, timeout=5.0
        )
    verdict, reason = mod.classify(service_result, human_result, tolerance=0)
    assert verdict == mod.VERDICT_DISCREPANCY
    assert "count mismatch" in reason


# ---------------------------------------------------------------------------
# decode_jwt_claims()
# ---------------------------------------------------------------------------
def _fake_jwt(payload: dict) -> str:
    def b64(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")

    header = b64(json.dumps({"alg": "none"}).encode())
    body = b64(json.dumps(payload).encode())
    return f"{header}.{body}.signature"


def test_decode_jwt_claims_extracts_email_presence() -> None:
    mod = _load_module()
    token = _fake_jwt({"sub": "abc", "email": "user@example.com"})
    claims = mod.decode_jwt_claims(token)
    assert claims["email"] == "user@example.com"
    assert "email" in claims


def test_decode_jwt_claims_service_token_has_no_email() -> None:
    mod = _load_module()
    token = _fake_jwt({"sub": "svc", "client_id": "graph-os"})
    claims = mod.decode_jwt_claims(token)
    assert "email" not in claims


def test_decode_jwt_claims_rejects_malformed_token() -> None:
    mod = _load_module()
    with pytest.raises(ValueError):
        mod.decode_jwt_claims("not-a-jwt")


# ---------------------------------------------------------------------------
# kube_secret() -- dot-escaping regression test. A literal "." in a Secret
# key (e.g. "ca.crt") is a jsonpath field separator unless escaped; this bit
# this lane during development (kubectl exits 0 with EMPTY stdout instead of
# an error, which looks exactly like "the key doesn't exist").
# ---------------------------------------------------------------------------
def test_kube_secret_escapes_dots_in_jsonpath(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_module()
    captured_args: list[list[str]] = []

    class _FakeCompletedProcess:
        returncode = 0
        stdout = base64.b64encode(b"hello-world").decode()
        stderr = ""

    def _fake_run(args, **kwargs):  # noqa: ANN001, ANN003
        captured_args.append(args)
        return _FakeCompletedProcess()

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)
    value = mod.kube_secret("some-secret", "ca.crt", namespace="cert-manager")
    assert value == "hello-world"
    jsonpath_arg = captured_args[0][-1]
    assert jsonpath_arg == r"jsonpath={.data.ca\.crt}"


def test_kube_secret_raises_on_empty_output(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_module()

    class _EmptyCompletedProcess:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: _EmptyCompletedProcess())
    with pytest.raises(RuntimeError):
        mod.kube_secret("missing-secret", "key")
