"""Standalone AU client for GraphOS's authorized EG MCP endpoint export."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from agent_utilities.layers.contracts import McpEndpoint

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_TOOLS = {"ask", "find"}
_REQUIRED_OPS = {"context.view", "query.uql"}


class ContextExportUnavailable(RuntimeError):
    """GraphOS did not prove an authorized EG context MCP connection."""


def parse_context_export(payload: Any) -> McpEndpoint:
    """Accept only the exact live probe result for this caller."""

    if not isinstance(payload, Mapping) or payload.get("ok") is not True:
        raise ContextExportUnavailable("GraphOS context export was refused")
    result = payload.get("result")
    if not isinstance(result, Mapping):
        raise ContextExportUnavailable("GraphOS context export has no result")
    proof = result.get("proof")
    meta = payload.get("meta")
    if not isinstance(proof, Mapping) or not isinstance(meta, Mapping):
        raise ContextExportUnavailable("GraphOS context export has no proof")
    try:
        endpoint = McpEndpoint.model_validate(result["endpoint"])
        tools = set(proof["tools"])
        operations = set(proof["operations"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ContextExportUnavailable("GraphOS context export is malformed") from exc
    if (
        endpoint.url != proof.get("endpoint_url")
        or _DIGEST.fullmatch(str(proof.get("registry_digest") or "")) is None
        or proof.get("registry_digest") != meta.get("registry_digest")
        or not _REQUIRED_TOOLS.issubset(tools)
        or not _REQUIRED_OPS.issubset(operations)
        or not endpoint.bearer_ref
    ):
        raise ContextExportUnavailable("GraphOS context capability proof is incomplete")
    return endpoint


@dataclass(frozen=True, slots=True)
class GraphOSContextClient:
    """Fetch a fresh caller-authorized endpoint immediately before each run."""

    base_url: str
    token_provider: Callable[[], str]
    post: Callable[[str, str], Any] | None = None

    def _url(self) -> str:
        parsed = urlsplit(self.base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path.rstrip("/") not in {"", "/"}
        ):
            raise ContextExportUnavailable("GraphOS API base URL is invalid")
        if parsed.scheme == "http" and parsed.hostname not in {
            "localhost",
            "127.0.0.1",
            "::1",
        }:
            raise ContextExportUnavailable("GraphOS API requires HTTPS")
        return self.base_url.rstrip("/") + "/api/v1/ops/harness.context_endpoint"

    def fetch(self) -> McpEndpoint:
        url = self._url()
        token = self.token_provider()
        if not isinstance(token, str) or not token:
            raise ContextExportUnavailable("GraphOS API bearer is unavailable")
        if self.post is not None:
            return parse_context_export(self.post(url, token))

        from agent_utilities.core.http_client import create_http_client
        from agent_utilities.core.transport_security import (
            resolve_configured_tls_profile,
        )

        trust = resolve_configured_tls_profile("graph-os")
        try:
            with create_http_client(
                headers={"Authorization": f"Bearer {token}"},
                timeout=20.0,
                **trust.httpx_kwargs(),
            ) as client:
                response = client.post(url, json={})
                response.raise_for_status()
                return parse_context_export(response.json())
        except ContextExportUnavailable:
            raise
        except Exception as exc:
            raise ContextExportUnavailable(
                "GraphOS context export request failed"
            ) from exc
        finally:
            trust.cleanup()
