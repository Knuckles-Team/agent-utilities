"""CONCEPT:AU-ORCH.adapter.byok-provider-proxy — BYOK provider-normalizing proxy router (Wire-First entry point).

Exposes ``POST /api/proxy/{provider}/stream``: forwards a chat request to an upstream LLM provider and
streams back a **canonical SSE** event stream, after a DNS-resolved SSRF check on any custom
``base_url`` (:mod:`agent_utilities.security.egress`) and three-tier credential resolution
(:mod:`agent_utilities.core.credentials`). Mirrors open-design's ``/api/proxy/<provider>/stream``.

Mounted in ``server/app.py`` (``app.include_router(proxy.router)``).
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterator

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from agent_utilities.core.config import config
from agent_utilities.core.contextual_model import (
    ContextCompilationError,
    compile_model_context,
)
from agent_utilities.core.credentials import CredentialResolver
from agent_utilities.core.execution.adapters.base import ExecEvent, ExecEventType
from agent_utilities.core.execution.provider_proxy import (
    SUPPORTED_PROVIDERS,
    check_egress,
    event_to_sse,
    stream_proxy,
)
from agent_utilities.core.http_client import create_async_http_client

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Provider Proxy"])

# Default upstream endpoints (overridable per-request via base_url, then env/file credentials).
_DEFAULT_URLS = {
    "openai": "https://api.openai.com/v1/chat/completions",
    "anthropic": "https://api.anthropic.com/v1/messages",
    "google": "https://generativelanguage.googleapis.com/v1beta/models",
}

_CONTEXT_MARKER = "[agent-utilities:compiled-evidence:v1]"
_MAX_PROXY_REQUEST_BYTES = 4 * 1024 * 1024
_MAX_PROXY_RESPONSE_BYTES = 64 * 1024 * 1024
_MAX_PROXY_LINE_BYTES = 1024 * 1024
_MAX_PROXY_MESSAGES = 256
_MAX_API_KEY_BYTES = 16 * 1024
_MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$")


def _require_model_invoke(request: Request) -> None:
    """Prevent an authenticated but unprivileged caller from spending BYOK quota."""

    claims = getattr(request.state, "user_claims", None)
    if not claims or claims.get("auth_type") == "api_key":
        return
    try:
        from agent_utilities.security.identity import (
            base_capabilities,
            normalize_identity,
        )

        capabilities = set(
            base_capabilities(
                normalize_identity(claims), config.identity_group_capability_map
            )
        )
    except Exception:
        raise HTTPException(
            status_code=403, detail="model invocation capability required"
        ) from None
    if not capabilities.intersection(
        {"model:invoke", "agent:invoke", "model:admin", "admin"}
    ):
        raise HTTPException(
            status_code=403, detail="model invocation capability required"
        )


def _message_query(messages: list[dict], system: object = None) -> str:
    """Derive retrieval text without persisting or logging caller content."""

    candidates: list[str] = []
    if isinstance(system, str) and system.strip():
        candidates.append(system.strip())
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        text = ""
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = " ".join(
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            )
        if text.strip():
            candidates.append(text.strip())
    return "\n".join(candidates[-4:])


def _govern_proxy_messages(
    provider: str, messages: list[dict], system: object = None
) -> tuple[list[dict], str | None]:
    """Compile policy-filtered evidence for the raw HTTP provider boundary."""

    bundle = compile_model_context(_message_query(messages, system))
    governed = (
        f"{_CONTEXT_MARKER}\n"
        "This governed evidence bundle is the only factual context for the "
        "request. Cite it or state that evidence is absent.\n\n"
        f"{bundle.as_text()}"
    )
    if isinstance(system, str) and system.strip():
        governed = f"{governed}\n\nCaller instructions:\n{system.strip()}"
    if provider == "anthropic":
        return list(messages), governed
    return [{"role": "system", "content": governed}, *messages], None


async def _upstream_lines(
    url: str,
    headers: dict,
    body: dict,
    *,
    allowed_private_hosts: list[str],
) -> AsyncIterator[str]:
    """Yield raw lines from the upstream provider's streaming response."""
    async with create_async_http_client(
        timeout=httpx.Timeout(120.0),
        pin_egress=True,
        allowed_private_hosts=allowed_private_hosts,
        allow_loopback=False,
    ) as client:
        async with client.stream("POST", url, headers=headers, json=body) as resp:
            resp.raise_for_status()
            total = 0
            buffer = bytearray()
            async for chunk in resp.aiter_bytes(chunk_size=64 * 1024):
                total += len(chunk)
                if total > _MAX_PROXY_RESPONSE_BYTES:
                    raise RuntimeError("provider_response_too_large")
                buffer.extend(chunk)
                if len(buffer) > _MAX_PROXY_LINE_BYTES and b"\n" not in buffer:
                    raise RuntimeError("provider_stream_line_too_large")
                while b"\n" in buffer:
                    raw_line, _, remainder = buffer.partition(b"\n")
                    buffer = bytearray(remainder)
                    if len(raw_line) > _MAX_PROXY_LINE_BYTES:
                        raise RuntimeError("provider_stream_line_too_large")
                    if raw_line:
                        yield raw_line.decode("utf-8", errors="replace")
            if buffer:
                if len(buffer) > _MAX_PROXY_LINE_BYTES:
                    raise RuntimeError("provider_stream_line_too_large")
                yield buffer.decode("utf-8", errors="replace")


async def _bounded_request_json(request: Request) -> dict:
    """Decode one bounded JSON object without reflecting its contents."""

    content_length = request.headers.get("content-length")
    if content_length:
        try:
            parsed_length = int(content_length)
            if parsed_length < 0 or parsed_length > _MAX_PROXY_REQUEST_BYTES:
                raise ValueError("request_too_large")
        except ValueError as exc:
            raise ValueError("invalid_content_length") from exc
    payload = bytearray()
    async for chunk in request.stream():
        payload.extend(chunk)
        if len(payload) > _MAX_PROXY_REQUEST_BYTES:
            raise ValueError("request_too_large")
    decoded = json.loads(payload)
    if not isinstance(decoded, dict):
        raise ValueError("request_not_object")
    return decoded


@router.post(
    "/api/proxy/{provider}/stream",
    summary="BYOK provider proxy → canonical SSE stream",
    response_model=None,
)
async def proxy_stream(
    provider: str, request: Request
) -> StreamingResponse | JSONResponse:
    """Proxy a chat completion to ``provider`` and stream canonical SSE events.

    Body: ``{base_url?, model, messages, system?, max_tokens?}``.

    Provider credentials are resolved from AgentConfig/runtime secret sources;
    request-body credentials are rejected so access tokens cannot leak through
    request capture, exception tooling, or observability middleware.
    """
    _require_model_invoke(request)
    provider = provider.lower()
    if provider not in SUPPORTED_PROVIDERS:
        return JSONResponse({"error": "unsupported provider"}, status_code=400)
    try:
        data = await _bounded_request_json(request)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)

    base_url = data.get("base_url")
    if base_url is not None and not isinstance(base_url, str):
        return JSONResponse({"error": "invalid base_url"}, status_code=400)
    decision = check_egress(
        base_url,
        allow_loopback=False,
        allowed_private_hosts=config.model_http_allowed_private_hosts,
    )
    if not decision.allowed:
        # SSRF gate: reject BEFORE any upstream fetch.
        return JSONResponse({"error": "blocked base_url"}, status_code=400)

    if "api_key" in data:
        return JSONResponse(
            {"error": "request credentials are not permitted"}, status_code=400
        )
    try:
        creds = CredentialResolver().resolve(provider)
    except Exception:
        return JSONResponse(
            {"error": "provider credentials are unavailable"}, status_code=503
        )
    api_key = creds.api_key
    url = base_url or creds.base_url or _DEFAULT_URLS.get(provider)
    if not url:
        return JSONResponse(
            {"error": "provider endpoint is not configured"}, status_code=400
        )
    final_decision = check_egress(
        url,
        allow_loopback=False,
        allowed_private_hosts=config.model_http_allowed_private_hosts,
    )
    if not final_decision.allowed:
        return JSONResponse({"error": "blocked endpoint"}, status_code=400)
    if api_key is not None:
        if (
            not isinstance(api_key, str)
            or not api_key
            or len(api_key.encode("utf-8")) > _MAX_API_KEY_BYTES
            or any(character in api_key for character in "\r\n\x00")
        ):
            return JSONResponse({"error": "invalid credential"}, status_code=400)

    messages = data.get("messages", [])
    if (
        not isinstance(messages, list)
        or len(messages) > _MAX_PROXY_MESSAGES
        or not all(isinstance(message, dict) for message in messages)
    ):
        return JSONResponse(
            {"error": "messages must be a list of objects"}, status_code=400
        )
    try:
        messages, governed_system = _govern_proxy_messages(
            provider, messages, data.get("system")
        )
    except (ContextCompilationError, PermissionError):
        return JSONResponse(
            {"error": "governed model context is unavailable"}, status_code=403
        )

    headers = {"content-type": "application/json"}
    if provider == "anthropic":
        headers["x-api-key"] = api_key or ""
        headers["anthropic-version"] = "2023-06-01"
    elif api_key:
        headers["authorization"] = f"Bearer {api_key}"

    model = str(data.get("model") or "")
    if not _MODEL_RE.fullmatch(model):
        return JSONResponse({"error": "invalid model"}, status_code=400)
    body = {
        "model": model,
        "messages": messages,
        "stream": True,
    }
    max_tokens = data.get("max_tokens")
    if max_tokens is not None:
        if (
            not isinstance(max_tokens, int)
            or isinstance(max_tokens, bool)
            or not 1 <= max_tokens <= 131_072
        ):
            return JSONResponse({"error": "invalid max_tokens"}, status_code=400)
        body["max_tokens"] = max_tokens
    if governed_system:
        body["system"] = governed_system

    async def gen() -> AsyncIterator[str]:
        try:
            async for sse in stream_proxy(
                provider,
                _upstream_lines(
                    url,
                    headers,
                    body,
                    allowed_private_hosts=config.model_http_allowed_private_hosts,
                ),
            ):
                yield sse
        except (httpx.HTTPError, RuntimeError):  # bounded upstream failure
            logger.warning("provider proxy upstream request failed")
            yield event_to_sse(
                ExecEvent(ExecEventType.ERROR, text="provider request failed")
            )
            yield event_to_sse(ExecEvent(ExecEventType.END))

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )
