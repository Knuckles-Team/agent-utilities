"""JWT-principal Eunomia authorization (plan Phase 3, embedded — no live server)."""

import json

import pytest
from eunomia_core import schemas


def _policy(tmp_path):
    """A default-deny policy that allows only agent:claude-code."""
    p = tmp_path / "policy.json"
    p.write_text(
        json.dumps(
            {
                "version": "1.0",
                "name": "test-zero-trust",
                "default_effect": "deny",
                "rules": [
                    {
                        "name": "claude-code-allow-all",
                        "effect": "allow",
                        "principal_conditions": [
                            {
                                "path": "uri",
                                "operator": "equals",
                                "value": "agent:claude-code",
                            }
                        ],
                        "resource_conditions": [],
                        "actions": ["list", "execute"],
                    }
                ],
            }
        )
    )
    return str(p)


def test_jwt_principal_overrides_spoofed_header(tmp_path, monkeypatch):
    """A spoofed x-agent-id must NOT win over the verified JWT client_id."""
    from agent_utilities.mcp.eunomia_principal import create_jwt_eunomia_middleware

    mw = create_jwt_eunomia_middleware(_policy(tmp_path))
    monkeypatch.setattr(
        "eunomia_mcp.middleware.get_http_headers",
        lambda: {"x-agent-id": "evil-spoof", "user-agent": "t"},
    )

    class _Tok:
        client_id = "claude-code"

    monkeypatch.setattr("fastmcp.server.dependencies.get_access_token", lambda: _Tok())

    principal = mw._extract_principal()
    assert principal.uri == "agent:claude-code"  # JWT wins, not "evil-spoof"
    assert principal.attributes["jwt_verified"] is True


def test_header_fallback_when_no_token(tmp_path, monkeypatch):
    """Without an auth context the principal falls back to the header value."""
    from agent_utilities.mcp.eunomia_principal import create_jwt_eunomia_middleware

    mw = create_jwt_eunomia_middleware(_policy(tmp_path))
    monkeypatch.setattr(
        "eunomia_mcp.middleware.get_http_headers",
        lambda: {"x-agent-id": "from-header", "user-agent": "t"},
    )

    def _no_ctx():
        raise RuntimeError("no auth context")

    monkeypatch.setattr("fastmcp.server.dependencies.get_access_token", _no_ctx)

    principal = mw._extract_principal()
    assert principal.uri == "agent:from-header"
    assert "jwt_verified" not in principal.attributes


@pytest.mark.asyncio
async def test_zero_trust_default_deny(tmp_path):
    """claude-code is allowed; every other principal is denied by default."""
    from agent_utilities.mcp.eunomia_principal import create_jwt_eunomia_middleware

    mw = create_jwt_eunomia_middleware(_policy(tmp_path))

    async def allowed(uri, action):
        result = await mw._eunomia.check(
            schemas.CheckRequest(
                principal=schemas.PrincipalCheck(uri=uri, attributes={}),
                resource=schemas.ResourceCheck(
                    uri="mcp:tool:cadd__config",
                    attributes={"component_type": "tool", "name": "cadd__config"},
                ),
                action=action,
            )
        )
        return result.allowed

    assert await allowed("agent:claude-code", "list") is True
    assert await allowed("agent:claude-code", "execute") is True
    assert await allowed("agent:unknown", "list") is False
    assert await allowed("agent:unknown", "execute") is False


def test_fastmcp3_component_exposes_enabled():
    """A fastmcp 3.x tool component reads ``.enabled`` (eunomia 2.x gate).

    Reproduces the live failure: eunomia-mcp's ``_authorize_execution`` does
    ``if not component.enabled`` on every call; fastmcp 3.x components dropped that
    attribute, so the access raised ``'FunctionTool' object has no attribute
    'enabled'`` and every tool call on a eunomia-enforced server became an internal
    error. After the compat the component reports enabled (3.x semantics).
    """
    pytest.importorskip("fastmcp")
    from fastmcp.tools.function_tool import FunctionTool

    from agent_utilities.mcp.eunomia_principal import apply_fastmcp_enabled_compat

    def sample(x: int) -> int:
        """doc"""
        return x

    apply_fastmcp_enabled_compat()  # idempotent; also runs at import
    tool = FunctionTool.from_function(sample)
    # the exact access eunomia-mcp's _authorize_execution performs (``if not
    # component.enabled``) must resolve without raising and report enabled.
    assert tool.enabled is True
    disabled = not tool.enabled
    assert disabled is False


# --- /check/bulk chunking (CONCEPT:ECO-4.88) -------------------------------


def _req(name):
    """A minimal CheckRequest whose resource uri encodes its index/name."""
    return schemas.CheckRequest(
        principal=schemas.PrincipalCheck(uri="agent:claude-code", attributes={}),
        resource=schemas.ResourceCheck(uri=f"mcp:tool:{name}", attributes={}),
        action="list",
    )


class _FakeBridge:
    """Stand-in EunomiaBridge with a mocked transport.

    Records the size of every ``bulk_check`` call it receives and rejects any
    request larger than ``cap`` (mirroring the remote server's HTTP 400). Returns
    one CheckResponse per request, positionally aligned, marking a resource allowed
    iff its uri is in ``allowed_uris``.
    """

    def __init__(self, cap=100, allowed_uris=None):
        self.cap = cap
        self.allowed_uris = set(allowed_uris or [])
        self.batch_sizes = []
        self.mode = "client"  # arbitrary passthrough attribute

    async def bulk_check(self, requests):
        requests = list(requests)
        self.batch_sizes.append(len(requests))
        if len(requests) > self.cap:
            raise RuntimeError(f"Too many requests. Maximum allowed: {self.cap}")
        return [
            schemas.CheckResponse(allowed=(r.resource.uri in self.allowed_uris))
            for r in requests
        ]


@pytest.mark.asyncio
async def test_bulk_check_chunks_250_into_100_100_50():
    """250 items must be split into 100/100/50 batches, none exceeding the cap."""
    from agent_utilities.mcp.eunomia_principal import _ChunkingBulkCheckBridge

    fake = _FakeBridge(cap=100)
    wrapped = _ChunkingBulkCheckBridge(fake, max_batch=100)

    requests = [_req(i) for i in range(250)]
    results = await wrapped.bulk_check(requests)

    assert fake.batch_sizes == [100, 100, 50]
    assert all(b <= 100 for b in fake.batch_sizes)
    assert len(results) == 250


@pytest.mark.asyncio
async def test_bulk_check_merges_results_in_order():
    """Merged responses stay positionally aligned with the request list."""
    from agent_utilities.mcp.eunomia_principal import _ChunkingBulkCheckBridge

    # Allow only every 7th tool, spread across all three chunks.
    requests = [_req(i) for i in range(250)]
    allowed = {f"mcp:tool:{i}" for i in range(250) if i % 7 == 0}
    fake = _FakeBridge(cap=100, allowed_uris=allowed)
    wrapped = _ChunkingBulkCheckBridge(fake, max_batch=100)

    results = await wrapped.bulk_check(requests)

    assert len(results) == 250
    for i, res in enumerate(results):
        assert res.allowed == (i % 7 == 0), f"index {i} misaligned after merge"


@pytest.mark.asyncio
async def test_bulk_check_single_batch_when_under_cap():
    """<=cap items go in one request (no needless chunking)."""
    from agent_utilities.mcp.eunomia_principal import _ChunkingBulkCheckBridge

    fake = _FakeBridge(cap=100)
    wrapped = _ChunkingBulkCheckBridge(fake, max_batch=100)

    results = await wrapped.bulk_check([_req(i) for i in range(100)])
    assert fake.batch_sizes == [100]
    assert len(results) == 100


@pytest.mark.asyncio
async def test_unwrapped_bridge_would_fail_on_oversize():
    """Sanity: without chunking the fake transport rejects >cap (the live 400)."""
    fake = _FakeBridge(cap=100)
    with pytest.raises(RuntimeError, match="Maximum allowed: 100"):
        await fake.bulk_check([_req(i) for i in range(250)])


def test_chunking_bridge_delegates_other_attrs():
    """Everything except bulk_check passes through to the real bridge."""
    from agent_utilities.mcp.eunomia_principal import _ChunkingBulkCheckBridge

    fake = _FakeBridge()
    wrapped = _ChunkingBulkCheckBridge(fake)
    assert wrapped.mode == "client"  # delegated attribute


def test_apply_bulk_check_chunking_is_idempotent():
    """Wrapping a middleware twice must not double-wrap the bridge."""
    from agent_utilities.mcp.eunomia_principal import (
        _ChunkingBulkCheckBridge,
        apply_bulk_check_chunking,
    )

    class _MW:
        def __init__(self):
            self._eunomia = _FakeBridge()

    mw = _MW()
    apply_bulk_check_chunking(mw)
    assert isinstance(mw._eunomia, _ChunkingBulkCheckBridge)
    inner = mw._eunomia._bridge
    apply_bulk_check_chunking(mw)  # second call
    assert mw._eunomia._bridge is inner  # not re-wrapped


def test_apply_bulk_check_chunking_noop_without_bridge():
    """No ``_eunomia`` attribute → return middleware unchanged (defensive)."""
    from agent_utilities.mcp.eunomia_principal import apply_bulk_check_chunking

    class _Bare:
        pass

    mw = _Bare()
    assert apply_bulk_check_chunking(mw) is mw
