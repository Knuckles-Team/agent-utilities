"""Live proof that graph-os serves real MCP over the streamable-HTTP wire, and
that the auth boundary genuinely rejects unauthenticated callers.

This is deliberately NOT a mocked round trip: it builds the exact production
FastMCP application (:func:`agent_utilities.mcp.kg_server._build_server`,
the same call the served CLI entrypoint and the API gateway use), applies the
real middleware/auth stack from ``create_mcp_server``/``_configure_auth``, and
serves it with a real ``uvicorn`` server bound to a real loopback TCP port. A
real JWT (RS256, verified against a real local JWKS HTTP endpoint — the same
verification path FastMCP uses against Keycloak in production) is used as the
Bearer credential.

CONCEPT:AU-OS.identity.authenticated-identity-enforcement / OS-5.14

Scope note: this intentionally calls ``_build_server()`` directly rather than
the ``mcp_server()`` CLI entrypoint, so it exercises the transport/auth
surface without also depending on ``_mint_process_session``'s *outbound*
engine bootstrap (a separate, unrelated concern gated by the compiled
``epistemic-graph[full]`` kernel, unavailable in this sandbox — see
``tests/_test_engine.py`` / the ``tiny_engine`` fixture, which every other
engine-backed test in this suite already gates the same way). Everything this
test asserts — the wire handshake, the tool catalog, and the auth
accept/reject boundary — is real production code running unmodified.
"""

from __future__ import annotations

import contextlib
import json
import socket
import threading
import time
from typing import Any

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm

pytestmark = pytest.mark.timeout(120)


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class _JWKSServer:
    """A real, tiny local JWKS endpoint backed by a fresh RSA keypair.

    Stands in for Keycloak's ``/protocol/openid-connect/certs`` for this test
    only — FastMCP's JWT verifier fetches it over real HTTP exactly as it
    would fetch a real issuer's JWKS.
    """

    def __init__(self) -> None:
        from http.server import BaseHTTPRequestHandler, HTTPServer

        self.key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self.kid = "e2e-test-key-1"
        jwk = json.loads(RSAAlgorithm.to_jwk(self.key.public_key()))
        jwk.update(kid=self.kid, use="sig", alg="RS256")
        body = json.dumps({"keys": [jwk]}).encode()

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_a: Any) -> None:  # silence stdout
                pass

        self.port = _free_port()
        self.httpd = HTTPServer(("127.0.0.1", self.port), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/jwks.json"

    def mint(
        self, *, sub: str, tenant_id: str, issuer: str, audience: str, ttl: int = 3600
    ) -> str:
        now = int(time.time())
        claims = {
            "sub": sub,
            "iss": issuer,
            "aud": audience,
            "iat": now,
            "exp": now + ttl,
            "tenant_id": tenant_id,
            "realm_access": {"roles": ["kg:read", "kg:write"]},
        }
        return jwt.encode(
            claims, self.key, algorithm="RS256", headers={"kid": self.kid}
        )

    def stop(self) -> None:
        self.httpd.shutdown()
        self.thread.join(timeout=5)


@contextlib.contextmanager
def _live_streamable_http_server(
    monkeypatch, jwks: _JWKSServer, *, issuer: str, audience: str
):
    """Build and serve the REAL graph-os FastMCP app over a real TCP port."""
    import uvicorn

    from agent_utilities.core.config import config as live_config
    from agent_utilities.mcp import kg_server

    # Configure inbound JWT verification (FastMCP's own provider) exactly as
    # --auth-type jwt --token-jwks-uri/--token-issuer/--token-audience would.
    argv = [
        "--transport",
        "streamable-http",
        "--auth-type",
        "jwt",
        "--token-jwks-uri",
        jwks.url,
        "--token-issuer",
        issuer,
        "--token-audience",
        audience,
    ]
    # The served-security profile (apply_served_security_profile, exercised
    # separately/unit-tested in test_request_identity.py) additionally
    # requires an audience + policy revision on the shared config object.
    monkeypatch.setattr(live_config, "auth_jwt_audience", audience, raising=False)
    monkeypatch.setattr(
        live_config, "kg_policy_version", "e2e-test-policy-v1", raising=False
    )

    args, mcp, middlewares = kg_server._build_server(bootstrap=False)
    # Re-parse our transport/auth argv through the real parser+auth configurer
    # (the same objects the CLI entrypoint builds), since _build_server(bootstrap=False)
    # deliberately ignores argv (see its docstring) and uses defaults.
    from agent_utilities.mcp.server_factory import _configure_auth, create_mcp_parser

    parser = create_mcp_parser(transport_choices=("stdio", "streamable-http"))
    real_args, _ = parser.parse_known_args(argv)
    auth = _configure_auth(real_args)
    assert auth is not None, "jwt auth-type must produce a real FastMCP auth provider"
    mcp.auth = auth

    for middleware in middlewares:
        mcp.add_middleware(middleware)

    # A purpose-built, read-only introspection tool: proves the REAL identity
    # bridge (ActorContextMiddleware.on_call_tool, CONCEPT:AU-OS.identity.idp-agnostic-role-inheritance)
    # actually ran for this call and minted a caller-scoped ActorContext +
    # GraphSession — the precise, engine-independent contract a duplicate/
    # shadowed middleware definition previously broke (it minted the actor but
    # never the session, so every downstream tenant/ACL check silently saw no
    # verified session on the network transport). Reads only contextvars, so
    # it needs no compiled engine and is a hard, unconditional regression
    # guard regardless of what engine backend is available in CI.
    @mcp.tool(name="_e2e_identity_probe")
    def _e2e_identity_probe() -> str:
        from agent_utilities.knowledge_graph.core.session import current_session
        from agent_utilities.security.brain_context import current_actor

        actor = current_actor()
        session = current_session()
        return json.dumps(
            {
                "actor_id": getattr(actor, "actor_id", None),
                "authenticated": getattr(actor, "authenticated", None),
                "actor_tenant_id": getattr(actor, "tenant_id", None),
                "has_session": session is not None,
                "session_tenant": getattr(session, "tenant", None),
            }
        )

    app = mcp.http_app(path="/mcp")
    port = _free_port()
    config = uvicorn.Config(
        app, host="127.0.0.1", port=port, lifespan="on", log_level="error"
    )
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 30
    base = f"http://127.0.0.1:{port}"
    while time.time() < deadline and not getattr(server, "started", False):
        time.sleep(0.05)
    if not getattr(server, "started", False):
        raise RuntimeError("uvicorn server did not report started in time")
    try:
        yield base
    finally:
        server.should_exit = True
        thread.join(timeout=15)


@pytest.fixture()
def jwks_server():
    server = _JWKSServer()
    try:
        yield server
    finally:
        server.stop()


def test_streamable_http_serves_real_mcp_and_rejects_unauthenticated(
    monkeypatch, jwks_server
):
    """initialize handshake + real tool catalog, and a hard 401 with no token."""
    issuer = "https://e2e-test-issuer.invalid/realms/homelab"
    audience = "graph-os-e2e-test"

    with _live_streamable_http_server(
        monkeypatch, jwks_server, issuer=issuer, audience=audience
    ) as base:
        mcp_headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        }
        init_body = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "e2e-probe", "version": "0"},
            },
        }

        # ---- 1. UNAUTHENTICATED request is rejected -----------------------
        r = httpx.post(f"{base}/mcp", json=init_body, headers=mcp_headers, timeout=10.0)
        assert r.status_code == 401, (
            f"expected 401 for a request with no Authorization header, got "
            f"{r.status_code}: {r.text[:300]}"
        )

        # A garbage/invalid bearer token must also be rejected, not merely a
        # missing one (guards against a verifier that only checks presence).
        bad_headers = {**mcp_headers, "Authorization": "Bearer not-a-real-jwt"}
        r_bad = httpx.post(
            f"{base}/mcp", json=init_body, headers=bad_headers, timeout=10.0
        )
        assert r_bad.status_code == 401, (
            f"expected 401 for an invalid bearer token, got {r_bad.status_code}: "
            f"{r_bad.text[:300]}"
        )

        # A validly-SIGNED token for the WRONG audience must also be rejected.
        wrong_aud_token = jwks_server.mint(
            sub="user:e2e",
            tenant_id="tenant:e2e",
            issuer=issuer,
            audience="some-other-service",
        )
        wrong_aud_headers = {
            **mcp_headers,
            "Authorization": f"Bearer {wrong_aud_token}",
        }
        r_wrong_aud = httpx.post(
            f"{base}/mcp", json=init_body, headers=wrong_aud_headers, timeout=10.0
        )
        assert r_wrong_aud.status_code == 401, (
            f"expected 401 for a token minted for a different audience, got "
            f"{r_wrong_aud.status_code}: {r_wrong_aud.text[:300]}"
        )

        # ---- 2. AUTHENTICATED initialize + tools/list succeed -------------
        token = jwks_server.mint(
            sub="user:e2e-tester",
            tenant_id="tenant:e2e",
            issuer=issuer,
            audience=audience,
        )
        auth_headers = {**mcp_headers, "Authorization": f"Bearer {token}"}
        r_init = httpx.post(
            f"{base}/mcp", json=init_body, headers=auth_headers, timeout=15.0
        )
        assert r_init.status_code == 200, (
            f"authenticated initialize should succeed, got {r_init.status_code}: "
            f"{r_init.text[:500]}"
        )
        session_id = r_init.headers.get("mcp-session-id")
        assert session_id, "streamable-http initialize must mint a session id"

        session_headers = {**auth_headers, "mcp-session-id": session_id}
        # Required by the streamable-http lifecycle before further calls.
        httpx.post(
            f"{base}/mcp",
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
            headers=session_headers,
            timeout=10.0,
        )

        r_list = httpx.post(
            f"{base}/mcp",
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            headers=session_headers,
            timeout=15.0,
        )
        assert r_list.status_code == 200, r_list.text[:500]
        list_payload = _parse_mcp_response(r_list)
        tool_names = {t["name"] for t in list_payload["result"]["tools"]}
        assert len(tool_names) >= 50, (
            f"expected the real graph-os tool catalog (dozens of tools), got "
            f"{len(tool_names)}: {sorted(tool_names)[:20]}"
        )
        assert "graph_query" in tool_names or "kg_query" in tool_names

        # ---- 3. A REAL, authenticated tool call reaches real dispatch -----
        # Pick a REAL, production-registered tool (not the diagnostic probe)
        # with no required args so the call is well-formed.
        candidate = next(
            (
                t
                for t in list_payload["result"]["tools"]
                if t["name"] != "_e2e_identity_probe"
                and not t.get("inputSchema", {}).get("required")
            ),
            None,
        )
        assert candidate is not None, "expected at least one zero-required-arg tool"
        r_call = httpx.post(
            f"{base}/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": candidate["name"], "arguments": {}},
            },
            headers=session_headers,
            timeout=60.0,
        )
        assert r_call.status_code == 200, r_call.text[:500]
        call_payload = _parse_mcp_response(r_call)
        # The call must have reached REAL tool dispatch under the caller's
        # verified identity — i.e. it must NOT be an auth/permission rejection
        # (that would mean the identity bridge never ran). It is allowed to
        # fail with an engine-backend error in this sandbox, which has no
        # compiled epistemic-graph[full] kernel (see module docstring) — the
        # same, pre-existing environment gap every engine-backed test in this
        # repository skips around via the `tiny_engine` fixture.
        rendered = json.dumps(call_payload).lower()
        for forbidden in (
            "verified bearer identity required",
            "verified graphsession required",
            "token validation failed",
            "unauthorized",
            "401",
        ):
            assert forbidden not in rendered, (
                f"authenticated tool call ({candidate['name']!r}) was rejected as "
                f"if unauthenticated ({forbidden!r} present) — the identity bridge "
                f"did not run: {rendered[:500]}"
            )

        # ---- 4. Precise proof the identity bridge attempted (or completed)
        # a real session mint for this exact call --------------------------
        # This is the regression guard for the duplicate/shadowed
        # ActorContextMiddleware bug this branch fixes (agent_utilities/mcp/
        # middlewares.py): the shadowed definition minted an ActorContext but
        # never a GraphSession, so a bare, session-less tool body ran and
        # returned success immediately — silently defeating tenant/ACL
        # scoping for every network tool call. The FIXED middleware mints the
        # session BEFORE the tool body runs, so the ``_e2e_identity_probe``
        # tool body below either (a) observes a real, caller-scoped session —
        # in an environment with the compiled epistemic-graph[full] kernel —
        # or (b) never runs at all because minting failed first, surfacing
        # the kernel's own error instead of a silent success. Case (b) is
        # this sandbox (see module docstring); either outcome proves the
        # middleware ran and tried. A bare 200 with ``has_session: false``
        # would prove the OLD, broken behavior and must never happen.
        r_probe = httpx.post(
            f"{base}/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {"name": "_e2e_identity_probe", "arguments": {}},
            },
            headers=session_headers,
            timeout=15.0,
        )
        assert r_probe.status_code == 200, r_probe.text[:500]
        probe_payload = _parse_mcp_response(r_probe)
        if "result" in probe_payload and not probe_payload["result"].get("isError"):
            # A real engine/kernel is available in this environment: the probe
            # tool body ran and must show a real, caller-scoped session.
            probe = json.loads(probe_payload["result"]["content"][0]["text"])
            assert probe["authenticated"] is True, probe
            assert probe["actor_id"] == "user:e2e-tester", probe
            assert probe["actor_tenant_id"] == "tenant:e2e", probe
            assert probe["has_session"] is True, (
                f"no GraphSession was minted for this authenticated tool call — "
                f"ActorContextMiddleware did not bridge identity into a "
                f"verified session: {probe}"
            )
            assert probe["session_tenant"] == "tenant:e2e", probe
        else:
            # No compiled kernel here: the probe tool body must NOT have run
            # (that would mean the middleware skipped session-minting) — the
            # failure must be the middleware's own session-mint attempt
            # hitting the environment's known, pre-existing kernel gap.
            rendered_probe = json.dumps(probe_payload).lower()
            assert (
                "epistemic-graph" in rendered_probe
                or "epistemic_graph" in rendered_probe
            ), (
                "the identity probe failed WITHOUT going through the "
                "session-minting path — this means ActorContextMiddleware "
                "skipped minting a GraphSession for the call (the exact "
                f"regression this fix prevents): {probe_payload}"
            )


def _parse_mcp_response(response: httpx.Response) -> dict[str, Any]:
    """Streamable-HTTP responses may be a bare JSON body or an SSE stream of
    ``data: <json>`` frames; return the parsed JSON-RPC payload either way."""
    content_type = response.headers.get("content-type", "")
    if "text/event-stream" in content_type:
        for line in response.text.splitlines():
            if line.startswith("data:"):
                return json.loads(line[len("data:") :].strip())
        raise AssertionError(f"no data frame in SSE response: {response.text[:500]}")
    return response.json()
