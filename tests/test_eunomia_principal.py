"""Native Eunomia authorization and transport hardening tests."""

import json
from types import SimpleNamespace

import httpx
import pytest
from eunomia_core import schemas
from fastmcp.server.middleware import MiddlewareContext


def _policy(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text(
        json.dumps(
            {
                "version": "1.0",
                "name": "test-zero-trust",
                "default_effect": "deny",
                "rules": [
                    {
                        "name": "verified-agent-allow",
                        "effect": "allow",
                        "principal_conditions": [
                            {
                                "path": "uri",
                                "operator": "equals",
                                "value": "agent:allowed-agent",
                            }
                        ],
                        "resource_conditions": [],
                        "actions": ["list", "execute"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return str(path)


def _request(uri="agent:allowed-agent", action="list"):
    return schemas.CheckRequest(
        principal=schemas.PrincipalCheck(uri=uri, attributes={}),
        resource=schemas.ResourceCheck(
            uri="mcp:tool:sample",
            attributes={"component_type": "tool", "name": "sample"},
        ),
        action=action,
    )


def test_verified_principal_overrides_spoofed_headers(tmp_path, monkeypatch):
    from agent_utilities.mcp.eunomia_principal import create_eunomia_middleware

    middleware = create_eunomia_middleware(
        _policy(tmp_path), require_verified_principal=True
    )
    monkeypatch.setattr(
        "agent_utilities.mcp.eunomia_principal.get_http_headers",
        lambda: {
            "x-agent-id": "spoofed-agent",
            "x-user-id": "spoofed-user",
            "authorization": "Bearer must-not-escape",
            "user-agent": "test-client",
        },
    )

    token = SimpleNamespace(
        client_id="allowed-agent", claims={"sub": "verified-user"}
    )
    monkeypatch.setattr("fastmcp.server.dependencies.get_access_token", lambda: token)

    principal = middleware._extract_principal()
    assert principal.uri == "agent:allowed-agent"
    assert principal.attributes["user_id"] == "verified-user"
    assert principal.attributes["jwt_verified"] is True
    assert "authorization" not in principal.attributes
    assert "api_key" not in principal.attributes
    assert "must-not-escape" not in repr(principal)


def test_authenticated_mode_fails_closed_without_token(tmp_path, monkeypatch):
    from agent_utilities.mcp.eunomia_principal import create_eunomia_middleware

    middleware = create_eunomia_middleware(
        _policy(tmp_path), require_verified_principal=True
    )
    monkeypatch.setattr(
        "agent_utilities.mcp.eunomia_principal.get_http_headers",
        lambda: {"x-agent-id": "allowed-agent"},
    )

    def no_context():
        raise RuntimeError("no auth context")

    monkeypatch.setattr("fastmcp.server.dependencies.get_access_token", no_context)
    assert middleware._extract_principal().uri == "agent:unknown"


def test_local_stdio_mode_can_use_bounded_header_identity(tmp_path, monkeypatch):
    from agent_utilities.mcp.eunomia_principal import create_eunomia_middleware

    middleware = create_eunomia_middleware(_policy(tmp_path))
    monkeypatch.setattr(
        "agent_utilities.mcp.eunomia_principal.get_http_headers",
        lambda: {"x-agent-id": "allowed-agent", "user-agent": "local-client"},
    )
    monkeypatch.setattr(
        "fastmcp.server.dependencies.get_access_token",
        lambda: (_ for _ in ()).throw(RuntimeError("stdio")),
    )
    assert middleware._extract_principal().uri == "agent:allowed-agent"


@pytest.mark.asyncio
async def test_embedded_policy_default_deny_and_allow(tmp_path):
    from agent_utilities.mcp.eunomia_principal import create_eunomia_middleware

    middleware = create_eunomia_middleware(_policy(tmp_path))
    assert (await middleware._eunomia.check(_request())).allowed is True
    assert (
        await middleware._eunomia.check(_request("agent:not-allowed"))
    ).allowed is False


def test_resource_projection_never_contains_argument_values(tmp_path):
    from agent_utilities.mcp.eunomia_principal import create_eunomia_middleware

    middleware = create_eunomia_middleware(_policy(tmp_path))
    context = MiddlewareContext(
        message=SimpleNamespace(
            arguments={"api_key": "secret-value", "query": "personal-content"}
        ),
        method="tools/call",
    )
    resource = middleware._extract_resource(
        context, SimpleNamespace(name="sample", enabled=True)
    )
    assert resource.attributes["argument_names"] == ["api_key", "query"]
    assert "secret-value" not in repr(resource)
    assert "personal-content" not in repr(resource)


def test_policy_file_is_bounded(tmp_path):
    from agent_utilities.mcp.eunomia_principal import create_eunomia_middleware

    path = tmp_path / "oversize.json"
    path.write_bytes(b" " * (1024 * 1024 + 1))
    with pytest.raises(ValueError, match="limit|bounded"):
        create_eunomia_middleware(str(path))


def test_remote_plaintext_requires_explicit_exception():
    from agent_utilities.mcp.eunomia_principal import create_eunomia_middleware

    with pytest.raises(ValueError, match="HTTPS"):
        create_eunomia_middleware(
            use_remote_eunomia=True,
            eunomia_endpoint="http://policy.example",
        )


@pytest.mark.asyncio
async def test_remote_bridge_uses_bounded_native_transport(monkeypatch):
    from agent_utilities.mcp.eunomia_principal import create_eunomia_middleware

    monkeypatch.setenv("TEST_EUNOMIA_KEY", "a" * 32)
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        assert request.headers["way-api-key"] == "a" * 32
        payload = json.loads(request.content)
        decisions = [
            {"allowed": item["principal"]["uri"] == "agent:allowed-agent"}
            for item in payload
        ]
        return httpx.Response(200, json=decisions)

    middleware = create_eunomia_middleware(
        use_remote_eunomia=True,
        eunomia_endpoint="https://policy.example/base",
        api_key_ref="env://TEST_EUNOMIA_KEY",
        transport=httpx.MockTransport(handler),
    )
    remote = middleware._eunomia._bridge
    raw = await remote._post(
        remote._bulk_url,
        [_request().model_dump(mode="json")],
    )
    assert raw == [{"allowed": True}]
    responses = await middleware._eunomia.bulk_check(
        [_request(), _request("agent:not-allowed")]
    )
    assert [item.allowed for item in responses] == [True, False]
    assert seen[0].url.path == "/base/check/bulk"


@pytest.mark.asyncio
async def test_remote_response_misalignment_fails_closed():
    from agent_utilities.mcp.eunomia_principal import create_eunomia_middleware

    middleware = create_eunomia_middleware(
        use_remote_eunomia=True,
        eunomia_endpoint="https://policy.example",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=[])
        ),
    )
    responses = await middleware._eunomia.bulk_check([_request(), _request()])
    assert len(responses) == 2
    assert not any(item.allowed for item in responses)


class _FakeBridge:
    def __init__(self):
        self.batch_sizes = []

    async def bulk_check(self, requests):
        selected = list(requests)
        self.batch_sizes.append(len(selected))
        return [schemas.CheckResponse(allowed=True) for _ in selected]


@pytest.mark.asyncio
async def test_bulk_check_chunks_at_server_limit():
    from agent_utilities.mcp.eunomia_principal import _ChunkingBulkCheckBridge

    bridge = _FakeBridge()
    wrapped = _ChunkingBulkCheckBridge(bridge)
    responses = await wrapped.bulk_check([_request() for _ in range(250)])
    assert bridge.batch_sizes == [100, 100, 50]
    assert len(responses) == 250


def test_bulk_check_limit_cannot_exceed_server_contract():
    from agent_utilities.mcp.eunomia_principal import _ChunkingBulkCheckBridge

    with pytest.raises(ValueError, match="between"):
        _ChunkingBulkCheckBridge(_FakeBridge(), max_batch=101)
