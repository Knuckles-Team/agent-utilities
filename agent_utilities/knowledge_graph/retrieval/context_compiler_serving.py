#!/usr/bin/python
from __future__ import annotations

"""The sole provider wire for governed ContextCompiler model calls (CONCEPT:AU-KG.retrieval.context-compiler-kv-seam, Seam 6 — deep half).

Every direct OpenAI-compatible completion in agent-utilities is centralized
here. The endpoint and model come only from typed runtime configuration (or an
explicit dependency-injected test client); there is no embedded host fallback.
Both sync and async calls receive a compiled, policy-filtered evidence prefix.

**The gap this closes.** :meth:`ContextBundle.as_prompt_messages` (in
``context_compiler.py``) makes the bundle's rendered text a byte-stable prefix
of the ``messages`` list — but a stable prefix only pays off if the caller
actually sends it to vLLM the same way every time, through the SAME governed
client construction. Historically each caller (``knowledge_graph/enrichment/
cards.py``'s ``make_llm_fn``, ``knowledge_graph/extraction/fact_extractor.py``'s
``make_streaming_extract_fn``, ``harness/g_eval.py``'s ``_live_endpoint``)
re-derived its own ``openai.OpenAI`` client from typed runtime configuration.
This module gives the context-compiler bundle the same real wire, so a
repeated bundle reaches the SAME endpoint with the SAME leading tokens and
vLLM's automatic prefix cache (on by default — no server change required)
reuses the KV blocks instead of recomputing them.

:meth:`ContextBundle.as_prompt_messages` has no network dependency at all
(pure string assembly); :func:`bundle_chat_completion` /
:func:`bundle_async_chat_completion` are the thin real-call wrappers around it
for callers that want the end-to-end wire in one call. Nothing in
``ContextCompiler.compile`` changes — this module is downstream of an
already-assembled :class:`ContextBundle`.
"""

import logging
import time
from typing import Any

from .context_compiler import ContextBundle

logger = logging.getLogger(__name__)

__all__ = [
    "resolve_bundle_chat_client",
    "resolve_bundle_async_chat_client",
    "bundle_chat_completion",
    "bundle_async_chat_completion",
    "compiled_chat_completion",
    "compiled_async_chat_completion",
]

# Bounded so a stalled vLLM call degrades instead of wedging a caller
# indefinitely — mirrors the timeout/retry discipline
# ``enrichment/cards.py``/``extraction/fact_extractor.py`` already apply to the
# same endpoint (CONCEPT:EG-KG.storage.nonblocking-checkpoint).
_DEFAULT_TIMEOUT_S = 60.0
_DEFAULT_MAX_RETRIES = 2


def resolve_bundle_chat_client(
    *,
    base_url: str | None = None,
    model: str | None = None,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    max_retries: int = _DEFAULT_MAX_RETRIES,
) -> tuple[Any, str]:
    """Resolve a sync ``openai.OpenAI`` client + model id for the live chat endpoint.

    Resolution order is an explicit dependency-injected override followed by a
    configured model id/role (including ``lite``), then
    ``config.default_chat_model``. Lazy import means importing this module never
    requires the provider package or a reachable endpoint.

    Returns:
        ``(client, model_id)``.
    """
    from openai import OpenAI

    from agent_utilities.core.config import config
    from agent_utilities.core.http_client import create_http_client
    from agent_utilities.core.model_runtime_auth import (
        resolve_model_api_key,
        resolve_model_headers,
    )
    from agent_utilities.core.transport_security import (
        resolve_configured_tls_profile,
    )

    selected_cfg = config.resolve_chat_model_config(model)
    cfg = selected_cfg or config.default_chat_model
    resolved_base_url = base_url or (cfg.base_url if cfg else None)
    if not resolved_base_url:
        raise RuntimeError("a configured chat-model base URL is required")
    resolved_model = (
        (selected_cfg.id if selected_cfg is not None else model)
        or (cfg.id if cfg else None)
        or "default"
    )
    tls_profile = resolve_configured_tls_profile(
        "model",
        profile_name=config.model_tls_profile,
        profile_ref=config.model_tls_profile_ref,
        config=config,
    )
    oauth2_auth = None
    if cfg and cfg.oauth2:
        from agent_utilities.security.oauth_client_credentials import (
            httpx_auth_from_config,
        )

        oauth2_auth = httpx_auth_from_config(cfg.oauth2)
    http_client = create_http_client(
        timeout=timeout_s,
        verify=tls_profile.ssl_context,
        headers=resolve_model_headers(
            reference=cfg.headers_ref if cfg else None
        ),
        auth=oauth2_auth,
    )
    client = OpenAI(
        base_url=resolved_base_url,
        api_key=(
            resolve_model_api_key(reference=cfg.api_key_ref) if cfg else None
        )
        or "oauth2-managed",
        http_client=http_client,
        timeout=timeout_s,
        max_retries=max_retries,
    )
    return client, resolved_model


def resolve_bundle_async_chat_client(
    *,
    base_url: str | None = None,
    model: str | None = None,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    max_retries: int = _DEFAULT_MAX_RETRIES,
) -> tuple[Any, str]:
    """Resolve the async equivalent of :func:`resolve_bundle_chat_client`."""

    from openai import AsyncOpenAI

    from agent_utilities.core.config import config
    from agent_utilities.core.http_client import create_async_http_client
    from agent_utilities.core.model_runtime_auth import (
        resolve_model_api_key,
        resolve_model_headers,
    )
    from agent_utilities.core.transport_security import (
        resolve_configured_tls_profile,
    )

    selected_cfg = config.resolve_chat_model_config(model)
    cfg = selected_cfg or config.default_chat_model
    resolved_base_url = base_url or (cfg.base_url if cfg else None)
    if not resolved_base_url:
        raise RuntimeError("a configured chat-model base URL is required")
    resolved_model = (
        (selected_cfg.id if selected_cfg is not None else model)
        or (cfg.id if cfg else None)
        or "default"
    )
    tls_profile = resolve_configured_tls_profile(
        "model",
        profile_name=config.model_tls_profile,
        profile_ref=config.model_tls_profile_ref,
        config=config,
    )
    oauth2_auth = None
    if cfg and cfg.oauth2:
        from agent_utilities.security.oauth_client_credentials import (
            httpx_auth_from_config,
        )

        oauth2_auth = httpx_auth_from_config(cfg.oauth2)
    http_client = create_async_http_client(
        timeout=timeout_s,
        verify=tls_profile.ssl_context,
        headers=resolve_model_headers(
            reference=cfg.headers_ref if cfg else None
        ),
        auth=oauth2_auth,
    )
    return (
        AsyncOpenAI(
            base_url=resolved_base_url,
            api_key=(
                resolve_model_api_key(reference=cfg.api_key_ref) if cfg else None
            )
            or "oauth2-managed",
            http_client=http_client,
            timeout=timeout_s,
            max_retries=max_retries,
        ),
        resolved_model,
    )


def bundle_chat_completion(
    bundle: ContextBundle,
    turn_text: str,
    *,
    client: Any | None = None,
    model: str | None = None,
    base_url: str | None = None,
    system_preamble: str | None = None,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    **create_kwargs: Any,
) -> Any:
    """Send ``bundle`` + ``turn_text`` to the live chat endpoint as one prefix-stable call.

    CONCEPT:AU-KG.retrieval.context-compiler-kv-seam — the serving-layer half of Seam 6. Builds
    ``messages`` via :meth:`ContextBundle.as_prompt_messages` (the bundle's
    ``as_text()`` as the stable system prefix, ``turn_text`` as the varying
    user suffix) and calls ``client.chat.completions.create`` with them. A
    ``client`` may be passed explicitly (e.g. a test double, or a client already
    resolved by a caller); otherwise one is built via
    :func:`resolve_bundle_chat_client` against the SAME live endpoint every
    other AU→vLLM call path uses.

    Prefix reuse remains an inference-runtime concern; this function guarantees
    a deterministic leading evidence block without assuming a particular host.

    Args:
        bundle: The compiled :class:`ContextBundle` (from
            ``ContextCompiler.compile``) whose ``as_text()`` becomes the stable
            prefix.
        turn_text: The turn-specific suffix — the caller's actual question for
            this call.
        client: Optional pre-built OpenAI-compatible client (sync
            ``chat.completions.create`` shape). When omitted, one is resolved
            via :func:`resolve_bundle_chat_client`.
        model: Optional model id override, forwarded to
            :func:`resolve_bundle_chat_client` when ``client`` is omitted, or
            used directly as the ``model=`` request field when ``client`` is
            supplied without a paired model id being resolvable.
        base_url: Optional endpoint override, forwarded to
            :func:`resolve_bundle_chat_client` (ignored if ``client`` is
            supplied).
        system_preamble: Forwarded to
            :meth:`ContextBundle.as_prompt_messages` (keep it a CONSTANT across
            calls — see that method's docstring).
        timeout_s: Governed provider-client timeout used when ``client`` is omitted.
        max_retries: Provider retry bound used when ``client`` is omitted.
        **create_kwargs: Forwarded verbatim to ``chat.completions.create``
            (e.g. ``max_tokens``, ``temperature``, ``logprobs``).

    Returns:
        The raw ``chat.completions.create`` response (usage/timing/content all
        available on it — this wrapper does not unwrap it, so callers can read
        provider-specific fields like cached-token usage when exposed).
    """
    resolved_model = model
    if client is None:
        client, resolved_model = resolve_bundle_chat_client(
            base_url=base_url,
            model=model,
            timeout_s=timeout_s,
            max_retries=max_retries,
        )
    kwargs = {}
    if system_preamble is not None:
        kwargs["system_preamble"] = system_preamble
    messages = bundle.as_prompt_messages(turn_text, **kwargs)
    logger.debug(
        "[CONCEPT:AU-KG.retrieval.context-compiler-kv-seam] bundle_chat_completion "
        "model=%s cache_key=%s items=%d",
        resolved_model,
        bundle.cache_key,
        len(bundle.items),
    )
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=resolved_model or "default", messages=messages, **create_kwargs
    )
    _record_ttft(time.perf_counter() - start, bundle)
    return response


async def bundle_async_chat_completion(
    bundle: ContextBundle,
    turn_text: str,
    *,
    client: Any | None = None,
    model: str | None = None,
    base_url: str | None = None,
    system_preamble: str | None = None,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    **create_kwargs: Any,
) -> Any:
    """Async/stream-capable governed completion over an existing bundle."""

    resolved_model = model
    if client is None:
        client, resolved_model = resolve_bundle_async_chat_client(
            base_url=base_url,
            model=model,
            timeout_s=timeout_s,
            max_retries=max_retries,
        )
    kwargs = {}
    if system_preamble is not None:
        kwargs["system_preamble"] = system_preamble
    return await client.chat.completions.create(
        model=resolved_model or "default",
        messages=bundle.as_prompt_messages(turn_text, **kwargs),
        **create_kwargs,
    )


def compiled_chat_completion(
    turn_text: str,
    *,
    engine: Any | None = None,
    session: Any | None = None,
    client: Any | None = None,
    model: str | None = None,
    base_url: str | None = None,
    **create_kwargs: Any,
) -> Any:
    """Compile governed evidence and execute one synchronous model call."""

    from agent_utilities.core.contextual_model import compile_model_context

    bundle = compile_model_context(
        turn_text, session=session, engine=engine, model_version=model or ""
    )
    return bundle_chat_completion(
        bundle,
        turn_text,
        client=client,
        model=model,
        base_url=base_url,
        **create_kwargs,
    )


async def compiled_async_chat_completion(
    turn_text: str,
    *,
    engine: Any | None = None,
    session: Any | None = None,
    client: Any | None = None,
    model: str | None = None,
    base_url: str | None = None,
    **create_kwargs: Any,
) -> Any:
    """Compile governed evidence and execute one asynchronous model call."""

    from agent_utilities.core.contextual_model import compile_model_context

    bundle = compile_model_context(
        turn_text, session=session, engine=engine, model_version=model or ""
    )
    return await bundle_async_chat_completion(
        bundle,
        turn_text,
        client=client,
        model=model,
        base_url=base_url,
        **create_kwargs,
    )


def _record_ttft(duration_s: float, bundle: ContextBundle) -> None:
    """Observe the WS-4 TTFT-proxy histogram (additive, best-effort, never raises).

    CONCEPT:AU-KG.retrieval.context-compiler-kv-seam — ``duration_s`` is the
    wall-clock latency of the (non-streaming) ``chat.completions.create`` call,
    the same latency-based signal ``scripts/measure_bundle_kv_reuse.py`` already
    treats as the fallback proof of prefix-cache reuse when vLLM's own
    ``/metrics`` isn't reachable; this just makes that signal a standing
    Prometheus histogram instead of a one-off script run, split by whether
    ``bundle`` itself was served from the Seam-6 KV cache.
    """
    try:
        from agent_utilities.observability.gateway_metrics import (
            CONTEXT_COMPILER_TTFT,
        )

        CONTEXT_COMPILER_TTFT.labels(
            kv_cache_hit=str(bool(bundle.kv_cache_hit)).lower()
        ).observe(duration_s)
    except Exception as exc:  # noqa: BLE001 — metrics must never break the call
        logger.debug("context-compiler ttft metric recording failed: %s", exc)
