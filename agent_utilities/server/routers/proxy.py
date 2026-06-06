"""CONCEPT:ORCH-1.34 — BYOK provider-normalizing proxy router (Wire-First entry point).

Exposes ``POST /api/proxy/{provider}/stream``: forwards a chat request to an upstream LLM provider and
streams back a **canonical SSE** event stream, after a DNS-resolved SSRF check on any custom
``base_url`` (:mod:`agent_utilities.security.egress`) and three-tier credential resolution
(:mod:`agent_utilities.core.credentials`). Mirrors open-design's ``/api/proxy/<provider>/stream``.

Mounted in ``server/app.py`` (``app.include_router(proxy.router)``).
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from agent_utilities.core.credentials import CredentialResolver
from agent_utilities.core.execution.provider_proxy import (
    SUPPORTED_PROVIDERS,
    check_egress,
    event_to_sse,
    stream_proxy,
)
from agent_utilities.core.execution.adapters.base import ExecEvent, ExecEventType

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Provider Proxy"])

# Default upstream endpoints (overridable per-request via base_url, then env/file credentials).
_DEFAULT_URLS = {
    "openai": "https://api.openai.com/v1/chat/completions",
    "anthropic": "https://api.anthropic.com/v1/messages",
    "google": "https://generativelanguage.googleapis.com/v1beta/models",
    "ollama": "http://127.0.0.1:11434/v1/chat/completions",
}


async def _upstream_lines(url: str, headers: dict, body: dict) -> AsyncIterator[str]:
    """Yield raw lines from the upstream provider's streaming response."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        async with client.stream("POST", url, headers=headers, json=body) as resp:
            async for line in resp.aiter_lines():
                if line:
                    yield line


@router.post(
    "/api/proxy/{provider}/stream",
    summary="BYOK provider proxy → canonical SSE stream",
    response_model=None,
)
async def proxy_stream(provider: str, request: Request) -> StreamingResponse | JSONResponse:
    """Proxy a chat completion to ``provider`` and stream canonical SSE events.

    Body: ``{base_url?, api_key?, model, messages, system?, max_tokens?, allow_loopback?}``.
    """
    provider = provider.lower()
    if provider not in SUPPORTED_PROVIDERS:
        return JSONResponse({"error": f"unsupported provider: {provider}"}, status_code=400)
    try:
        data = await request.json()
    except (json.JSONDecodeError, ValueError):
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)

    allow_loopback = bool(data.get("allow_loopback", True))
    base_url = data.get("base_url")
    decision = check_egress(base_url, allow_loopback=allow_loopback)
    if not decision.allowed:
        # SSRF gate: reject BEFORE any upstream fetch.
        return JSONResponse({"error": "blocked base_url", "reason": decision.reason}, status_code=400)

    creds = CredentialResolver().resolve(provider)
    api_key = data.get("api_key") or creds.api_key
    url = base_url or creds.base_url or _DEFAULT_URLS.get(provider)
    if not url:
        return JSONResponse({"error": f"no endpoint for provider {provider}"}, status_code=400)

    headers = {"content-type": "application/json"}
    if provider == "anthropic":
        headers["x-api-key"] = api_key or ""
        headers["anthropic-version"] = "2023-06-01"
    elif api_key:
        headers["authorization"] = f"Bearer {api_key}"

    body = {
        "model": data.get("model"),
        "messages": data.get("messages", []),
        "stream": True,
    }
    if data.get("max_tokens"):
        body["max_tokens"] = data["max_tokens"]
    if data.get("system"):
        body["system"] = data["system"]

    async def gen() -> AsyncIterator[str]:
        try:
            async for sse in stream_proxy(provider, _upstream_lines(url, headers, body)):
                yield sse
        except httpx.HTTPError as exc:  # upstream failure → canonical error event, not a 500 mid-stream
            logger.warning("proxy upstream error for %s: %s", provider, exc)
            yield event_to_sse(ExecEvent(ExecEventType.ERROR, text=str(exc)))
            yield event_to_sse(ExecEvent(ExecEventType.END))

    return StreamingResponse(gen(), media_type="text/event-stream")
