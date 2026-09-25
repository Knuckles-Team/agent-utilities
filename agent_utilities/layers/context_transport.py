"""Bind the verified GraphOS context MCP endpoint for one harness run."""

from __future__ import annotations

from urllib.parse import urlsplit

import httpx

from agent_utilities.layers.contracts import McpEndpoint
from agent_utilities.security.cli_secrets import (
    resolve_runtime_secret_reference,
    validate_runtime_secret_reference,
)


class ContextEndpointUnavailable(RuntimeError):
    """The mandatory EG MCP connection could not be bound securely."""


def _resolve_bearer(reference: str) -> str:
    if reference.startswith("openbao://"):
        from agent_connector_sdk.credentials.resolution import resolve_secret_reference

        return resolve_secret_reference(reference)
    return resolve_runtime_secret_reference(reference)


class _ReferenceBearerAuth(httpx.Auth):
    """Resolve the secret reference for each request, including long runs."""

    def __init__(self, reference: str) -> None:
        if reference.startswith("openbao://"):
            from agent_connector_sdk.credentials.references import (
                parse_secret_reference,
            )

            parse_secret_reference(reference)
            self.reference = reference
        else:
            self.reference = validate_runtime_secret_reference(reference)
        _resolve_bearer(self.reference)

    def auth_flow(self, request: httpx.Request):  # type: ignore[no-untyped-def]
        request.headers["Authorization"] = f"Bearer {_resolve_bearer(self.reference)}"
        yield request


def bind_context_toolset(endpoint: McpEndpoint) -> object:
    """Build exactly the authorized endpoint; never resolve it via fleet names."""

    parsed = urlsplit(endpoint.url)
    if (
        not parsed.hostname
        or parsed.scheme not in {"http", "https"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or not parsed.path.rstrip("/").endswith("/mcp")
    ):
        raise ContextEndpointUnavailable("EG MCP endpoint is invalid")
    if parsed.scheme == "http" and parsed.hostname not in {
        "localhost",
        "127.0.0.1",
        "::1",
    }:
        raise ContextEndpointUnavailable("EG MCP endpoint requires HTTPS")
    if endpoint.transport != "http" or not endpoint.bearer_ref:
        raise ContextEndpointUnavailable(
            "EG MCP endpoint has no supported bearer reference"
        )
    try:
        auth = _ReferenceBearerAuth(endpoint.bearer_ref)
        from agent_utilities.mcp.toolset_factory import build_http_toolset

        return build_http_toolset(
            endpoint.url,
            auth=auth,
            timeout=60,
            toolset_id=endpoint.name,
            tls_service="graph-os",
        )
    except Exception as exc:
        raise ContextEndpointUnavailable("EG MCP endpoint could not be bound") from exc
