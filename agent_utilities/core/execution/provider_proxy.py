"""CONCEPT:AU-ORCH.adapter.byok-provider-proxy — Provider-Normalizing Stream Proxy.

Assimilated from open-design's ``/api/proxy/<provider>/stream`` (chat-routes.ts): normalize each LLM
provider's native streaming format into one canonical event union ``{start|text_delta|error|end}``,
behind a DNS-resolved SSRF guard (:mod:`agent_utilities.security.egress`) and the three-tier
credential resolver (:mod:`agent_utilities.core.credentials`).

This module owns the **normalization** (pure, fully testable) and a thin async streamer; the FastAPI
route in ``server/routers/proxy.py`` is the Wire-First entry point.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Iterable, Iterator
from typing import Any

from agent_utilities.core.execution.adapters.base import ExecEvent, ExecEventType
from agent_utilities.security.egress import EgressDecision, validate_base_url_resolved

logger = logging.getLogger(__name__)

SUPPORTED_PROVIDERS = ("anthropic", "openai", "azure", "google", "ollama")


def _sse_data(line: str) -> str | None:
    """Extract the JSON payload from an ``data: {...}`` SSE line (OpenAI/Anthropic style)."""
    s = line.strip()
    if not s or s.startswith(":"):
        return None
    if s.startswith("data:"):
        return s[len("data:") :].strip()
    return s


def normalize_chunk(provider: str, raw: str) -> list[ExecEvent]:
    """Normalize one raw stream line from ``provider`` into canonical events.

    Recognizes the common shapes across providers; unknown shapes degrade to an empty list (skipped)
    rather than raising, so a single malformed chunk never breaks the stream.
    """
    payload = _sse_data(raw)
    if payload is None:
        return []
    if payload == "[DONE]":  # OpenAI terminal sentinel
        return [ExecEvent(ExecEventType.END)]
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        return []

    p = provider.lower()
    # OpenAI / Azure / Ollama (OpenAI-compatible): choices[].delta.content
    if p in {"openai", "azure", "ollama"}:
        choices = obj.get("choices") or []
        out: list[ExecEvent] = []
        for ch in choices:
            delta = (ch.get("delta") or {}).get("content")
            if delta:
                out.append(ExecEvent(ExecEventType.TEXT_DELTA, text=str(delta)))
            if ch.get("finish_reason"):
                out.append(
                    ExecEvent(
                        ExecEventType.END, data={"finish_reason": ch["finish_reason"]}
                    )
                )
        return out
    # Anthropic Messages API: {type: content_block_delta, delta:{text}} / {type: message_stop}
    if p == "anthropic":
        t = obj.get("type", "")
        if t == "content_block_delta":
            txt = (obj.get("delta") or {}).get("text", "")
            return [ExecEvent(ExecEventType.TEXT_DELTA, text=str(txt))] if txt else []
        if t == "message_stop":
            return [ExecEvent(ExecEventType.END)]
        if t == "error":
            return [
                ExecEvent(
                    ExecEventType.ERROR,
                    text=str(obj.get("error", {}).get("message", "error")),
                )
            ]
        return []
    # Google Gemini: {candidates:[{content:{parts:[{text}]}}]}
    if p == "google":
        out_g: list[ExecEvent] = []
        for cand in obj.get("candidates", []) or []:
            for part in (cand.get("content") or {}).get("parts", []) or []:
                if part.get("text"):
                    out_g.append(
                        ExecEvent(ExecEventType.TEXT_DELTA, text=str(part["text"]))
                    )
        return out_g
    return []


def normalize_stream(provider: str, lines: Iterable[str]) -> Iterator[ExecEvent]:
    """Normalize an iterable of raw provider lines into a canonical event stream."""
    yield ExecEvent(ExecEventType.START)
    saw_end = False
    for line in lines:
        for ev in normalize_chunk(provider, line):
            if ev.type is ExecEventType.END:
                saw_end = True
            yield ev
    if not saw_end:
        yield ExecEvent(ExecEventType.END)


def event_to_sse(ev: ExecEvent) -> str:
    """Serialize a canonical event as an SSE ``data:`` line (the proxy's wire format)."""
    body: dict[str, Any] = {"type": ev.type.value}
    if ev.text:
        body["text"] = ev.text
    if ev.data:
        body["data"] = ev.data
    return f"data: {json.dumps(body)}\n\n"


def check_egress(
    base_url: str | None, *, allow_loopback: bool = True
) -> EgressDecision:
    """Validate a custom ``base_url`` (DNS-resolved SSRF guard) before any upstream fetch."""
    if not base_url:
        return EgressDecision(True, "no custom base_url")
    return validate_base_url_resolved(base_url, allow_loopback=allow_loopback)


async def stream_proxy(
    provider: str,
    upstream_lines: AsyncIterator[str],
) -> AsyncIterator[str]:
    """Async generator: normalize upstream lines → canonical SSE wire lines.

    The caller is responsible for the upstream HTTP request (after :func:`check_egress` passes); this
    keeps normalization independent of the HTTP client and unit-testable.
    """
    yield event_to_sse(ExecEvent(ExecEventType.START))
    saw_end = False
    async for line in upstream_lines:
        for ev in normalize_chunk(provider, line):
            if ev.type is ExecEventType.END:
                saw_end = True
            yield event_to_sse(ev)
    if not saw_end:
        yield event_to_sse(ExecEvent(ExecEventType.END))
