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
