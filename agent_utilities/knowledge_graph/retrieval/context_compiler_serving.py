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

import hashlib
import logging
import time
from dataclasses import dataclass
from types import SimpleNamespace
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


@dataclass
class _BundleModelConfig:
    """Everything shared by the sync/async client-resolution paths — the
    dependency-injected override / configured model-id / default-chat-model
    resolution order and TLS/auth material, with only the concrete client
    class (``OpenAI`` vs ``AsyncOpenAI``) and http-client factory left to the
    caller."""

    base_url: str
    model: str
    tls_profile: Any
    headers: dict[str, str] | None
    oauth2_auth: Any
    api_key: str


def _resolve_bundle_endpoint(
    *, base_url: str | None, model: str | None
) -> tuple[Any, str, str]:
    """Resolve the configured model entry plus the base URL / model id to use.

    Resolution order is an explicit dependency-injected override followed by a
    configured model id/role (including ``lite``), then
    ``config.default_chat_model``.

    Returns:
        ``(cfg, resolved_base_url, resolved_model)`` — ``cfg`` is the selected
        (or default) chat-model config entry, or ``None`` if neither resolved.
    """
    from agent_utilities.core.config import config

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
    return cfg, resolved_base_url, resolved_model


def _resolve_bundle_auth_material(
    cfg: Any,
) -> tuple[Any, dict[str, str] | None, Any, str]:
    """Resolve the TLS profile, headers, oauth2 auth, and API key for ``cfg``
    (the entry returned by :func:`_resolve_bundle_endpoint`). Lazy imports mean
    importing this module never requires the provider package or a reachable
    endpoint.

    Returns:
        ``(tls_profile, headers, oauth2_auth, api_key)``.
    """
    from agent_utilities.core.config import config
    from agent_utilities.core.model_runtime_auth import (
        resolve_model_api_key,
        resolve_model_headers,
    )
    from agent_utilities.core.transport_security import (
        resolve_configured_tls_profile,
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
    api_key = (
        resolve_model_api_key(reference=cfg.api_key_ref) if cfg else None
    ) or "oauth2-managed"
    headers = resolve_model_headers(reference=cfg.headers_ref if cfg else None)
    return tls_profile, headers, oauth2_auth, api_key


def _resolve_bundle_model_config(
    *, base_url: str | None, model: str | None
) -> _BundleModelConfig:
    """Resolve the model/endpoint/auth material shared by both
    :func:`resolve_bundle_chat_client` and :func:`resolve_bundle_async_chat_client`.
    """
    cfg, resolved_base_url, resolved_model = _resolve_bundle_endpoint(
        base_url=base_url, model=model
    )
    tls_profile, headers, oauth2_auth, api_key = _resolve_bundle_auth_material(cfg)
    return _BundleModelConfig(
        base_url=resolved_base_url,
        model=resolved_model,
        tls_profile=tls_profile,
        headers=headers,
        oauth2_auth=oauth2_auth,
        api_key=api_key,
    )


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

    from agent_utilities.core.http_client import create_http_client

    resolved = _resolve_bundle_model_config(base_url=base_url, model=model)
    http_client = create_http_client(
        timeout=timeout_s,
        verify=resolved.tls_profile.ssl_context,
        headers=resolved.headers,
        auth=resolved.oauth2_auth,
    )
    client = OpenAI(
        base_url=resolved.base_url,
        api_key=resolved.api_key,
        http_client=http_client,
        timeout=timeout_s,
        max_retries=max_retries,
    )
    return client, resolved.model


def resolve_bundle_async_chat_client(
    *,
    base_url: str | None = None,
    model: str | None = None,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    max_retries: int = _DEFAULT_MAX_RETRIES,
) -> tuple[Any, str]:
    """Resolve the async equivalent of :func:`resolve_bundle_chat_client`."""

    from openai import AsyncOpenAI

    from agent_utilities.core.http_client import create_async_http_client

    resolved = _resolve_bundle_model_config(base_url=base_url, model=model)
    http_client = create_async_http_client(
        timeout=timeout_s,
        verify=resolved.tls_profile.ssl_context,
        headers=resolved.headers,
        auth=resolved.oauth2_auth,
    )
    return (
        AsyncOpenAI(
            base_url=resolved.base_url,
            api_key=resolved.api_key,
            http_client=http_client,
            timeout=timeout_s,
            max_retries=max_retries,
        ),
        resolved.model,
    )


def _semantic_cache_key_for_bundle(
    bundle: ContextBundle, *, model: str | None, system_preamble: str | None
) -> Any:
    """Build the :class:`~agent_utilities.caching.semantic_cache.SemanticCacheKey` for a
    Seam-6 bundle call (CONCEPT:AU-KG.memory.semantic-response-cache).

    Reuses the bundle's OWN identity fields rather than re-deriving them:
    ``bundle.session_tenant``/``bundle.policy_version`` (already resolved by
    ``ContextCompiler.compile`` from the governing ``GraphSession``) and
    ``bundle.cache_key`` (the Seam-6 KV-layer retrieval/evidence-set fingerprint — see
    ``compute_bundle_cache_key``) AS the ``retrieval_snapshot`` component, so the two caches
    share one evidence-identity notion instead of computing it twice. ``prompt_version`` is a
    content hash of ``system_preamble`` (the stable rendering directive), since this seam never
    binds tools (plain ``chat.completions.create``), ``tool_schema_version`` stays constant.
    """
    from agent_utilities.caching.semantic_cache import resolve_semantic_cache_key

    return resolve_semantic_cache_key(
        tenant=bundle.session_tenant or None,
        policy_version=bundle.policy_version if bundle.policy_version else None,
        model_identity=model or "",
        prompt_version=hashlib.sha256(
            (system_preamble or "").encode("utf-8")
        ).hexdigest()[:16],
        retrieval_snapshot=bundle.cache_key,
    )


def _synthetic_cache_response(lookup: Any, *, model: str | None) -> Any:
    """A duck-typed ``chat.completions.create``-shaped stand-in for a semantic-cache HIT.

    Never indistinguishable from a fresh call (CONCEPT:AU-KG.memory.semantic-response-cache):
    ``.au_cache_hit``/``.au_cache_similarity``/``.au_cache_age_seconds``/``.au_cache_fingerprint``
    are always present on the returned object IN ADDITION to the normal ``.choices[0].message.
    content``/``.usage`` shape callers already read off a live response.
    """
    message = SimpleNamespace(content=lookup.response_text, role="assistant")
    choice = SimpleNamespace(message=message, finish_reason="stop", index=0)
    usage = SimpleNamespace(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    return SimpleNamespace(
        choices=[choice],
        usage=usage,
        model=model or "",
        au_cache_hit=True,
        au_cache_similarity=lookup.similarity,
        au_cache_age_seconds=lookup.age_seconds,
        au_cache_fingerprint=lookup.key.fingerprint,
    )


def _semantic_cache_lookup_for_call(
    bundle: ContextBundle,
    turn_text: str,
    *,
    model: str | None,
    system_preamble: str | None,
    semantic_cache_policy: Any | None,
    log_prefix: str,
) -> Any | None:
    """Opt-in semantic-cache lookup shared by the sync/async completion calls.

    Returns ``None`` when the caller did not opt in, OR the lookup best-effort
    failed (mirrors the original inline ``cache_lookup = None`` reset on
    exception). The caller is responsible for checking ``.hit`` and returning
    the synthetic response — this only resolves the lookup object.
    """
    if semantic_cache_policy is None:
        return None
    try:
        from agent_utilities.caching.semantic_cache import get_semantic_cache

        cache_key = _semantic_cache_key_for_bundle(
            bundle, model=model, system_preamble=system_preamble
        )
        return get_semantic_cache().lookup(
            cache_key, turn_text, policy=semantic_cache_policy
        )
    except Exception as exc:  # noqa: BLE001 — semantic cache is best-effort
        logger.debug("%s semantic-cache lookup failed: %s", log_prefix, exc)
        return None


def _apply_prompt_cache_hint(
    create_kwargs: dict[str, Any],
    *,
    system_preamble: str | None,
    resolved_model: str | None,
    bundle: ContextBundle,
    log_prefix: str,
) -> dict[str, Any]:
    """Default-on OpenAI ``prompt_cache_key`` routing hint, shared by the
    sync/async completion calls. Never overrides an explicit caller value;
    best-effort — a failure here must never break the call, so it falls back
    to the unmodified ``create_kwargs`` (CONCEPT:AU-ORCH.optimization.provider-
    prompt-cache)."""
    try:
        from agent_utilities.caching.prompt_cache import prompt_cache_create_kwargs

        return prompt_cache_create_kwargs(
            create_kwargs,
            system_prompt=system_preamble,
            model_identity=resolved_model or "",
            tenant=bundle.session_tenant or None,
            policy_version=bundle.policy_version if bundle.policy_version else None,
        )
    except Exception as exc:  # noqa: BLE001 — prompt-cache hint is best-effort
        logger.debug("%s prompt-cache hint failed: %s", log_prefix, exc)
        return create_kwargs


def _store_semantic_cache_result(
    cache_lookup: Any | None,
    *,
    turn_text: str,
    response: Any,
    semantic_cache_policy: Any | None,
    log_prefix: str,
) -> None:
    """Store a live response for next time on a semantic-cache MISS, shared by
    the sync/async completion calls. A no-op when caching was not opted into,
    there was no lookup to pair with, or the lookup was already a HIT (the
    caller returns early on a hit, so this only runs on the miss path)."""
    if semantic_cache_policy is None or cache_lookup is None or cache_lookup.hit:
        return
    try:
        from agent_utilities.caching.semantic_cache import get_semantic_cache

        text = response.choices[0].message.content
        get_semantic_cache().store(
            cache_lookup.key, turn_text, text or "", policy=semantic_cache_policy
        )
    except Exception as exc:  # noqa: BLE001 — semantic cache is best-effort
        logger.debug("%s semantic-cache store failed: %s", log_prefix, exc)


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
    semantic_cache_policy: Any | None = None,
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
        semantic_cache_policy: Optional ``SemanticCachePolicy`` (CONCEPT:AU-KG.memory.
            semantic-response-cache). ``None`` (the default) means "do not consult the
            semantic cache" — this seam's LLM-response caching is opt-in per call, never
            silently on. When supplied AND the policy is enabled, a similarity hit
            SKIPS the live client call entirely and returns a synthetic, clearly-marked
            response (see :func:`_synthetic_cache_response` — ``.au_cache_hit`` is
            always present so a cached answer can never be mistaken for a fresh one); a
            miss falls through to the live call and, on success, stores the response for
            next time. Best-effort end to end — any failure in the cache path (including
            an unresolvable tenant) silently falls back to a live call.
        **create_kwargs: Forwarded verbatim to ``chat.completions.create``
            (e.g. ``max_tokens``, ``temperature``, ``logprobs``).

    Returns:
        The raw ``chat.completions.create`` response (usage/timing/content all
        available on it — this wrapper does not unwrap it, so callers can read
        provider-specific fields like cached-token usage when exposed), or the
        synthetic cache-hit stand-in described above.
    """
    resolved_model = model
    cache_lookup = _semantic_cache_lookup_for_call(
        bundle,
        turn_text,
        model=model,
        system_preamble=system_preamble,
        semantic_cache_policy=semantic_cache_policy,
        log_prefix="bundle_chat_completion",
    )
    if cache_lookup is not None and cache_lookup.hit:
        return _synthetic_cache_response(cache_lookup, model=model)

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
    create_kwargs = _apply_prompt_cache_hint(
        create_kwargs,
        system_preamble=system_preamble,
        resolved_model=resolved_model,
        bundle=bundle,
        log_prefix="bundle_chat_completion",
    )
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=resolved_model or "default", messages=messages, **create_kwargs
    )
    _record_ttft(time.perf_counter() - start, bundle)
    _store_semantic_cache_result(
        cache_lookup,
        turn_text=turn_text,
        response=response,
        semantic_cache_policy=semantic_cache_policy,
        log_prefix="bundle_chat_completion",
    )
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
    semantic_cache_policy: Any | None = None,
    **create_kwargs: Any,
) -> Any:
    """Async/stream-capable governed completion over an existing bundle.

    Same opt-in semantic-cache (``semantic_cache_policy``) and default-on prompt-cache-key
    behavior as :func:`bundle_chat_completion` — see its docstring for the full contract.
    """

    resolved_model = model
    cache_lookup = _semantic_cache_lookup_for_call(
        bundle,
        turn_text,
        model=model,
        system_preamble=system_preamble,
        semantic_cache_policy=semantic_cache_policy,
        log_prefix="bundle_async_chat_completion",
    )
    if cache_lookup is not None and cache_lookup.hit:
        return _synthetic_cache_response(cache_lookup, model=model)

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
    create_kwargs = _apply_prompt_cache_hint(
        create_kwargs,
        system_preamble=system_preamble,
        resolved_model=resolved_model,
        bundle=bundle,
        log_prefix="bundle_async_chat_completion",
    )
    response = await client.chat.completions.create(
        model=resolved_model or "default",
        messages=bundle.as_prompt_messages(turn_text, **kwargs),
        **create_kwargs,
    )
    _store_semantic_cache_result(
        cache_lookup,
        turn_text=turn_text,
        response=response,
        semantic_cache_policy=semantic_cache_policy,
        log_prefix="bundle_async_chat_completion",
    )
    return response


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
    ``bundle`` itself was served from the Seam-6 KV cache. Labeled
    ``path="bundle_chat_completion"`` (the background enrichment/extraction call
    population) to keep it distinct from the interactive delegated-run population
    ``contextual_model._record_delegated_run_ttft`` records into the SAME series
    (CONCEPT:AU-KG.retrieval.context-compiler, W3.7).
    """
    try:
        from agent_utilities.observability.gateway_metrics import (
            CONTEXT_COMPILER_TTFT,
        )

        CONTEXT_COMPILER_TTFT.labels(
            kv_cache_hit=str(bool(bundle.kv_cache_hit)).lower(),
            path="bundle_chat_completion",
        ).observe(duration_s)
    except Exception as exc:  # noqa: BLE001 — metrics must never break the call
        logger.debug("context-compiler ttft metric recording failed: %s", exc)
