#!/usr/bin/python
"""Embedding Utilities Module.

CONCEPT:AU-KG.memory.auto-similarity-memory-graph

This module provides factory functions for initializing LlamaIndex-compatible
embedding models. It supports various providers including OpenAI, Ollama,
HuggingFace, and local models, with robust environment-based configuration.
"""

import asyncio
import json
import math
import threading
from typing import TYPE_CHECKING, Any

import httpx

from agent_utilities._version import __version__ as __version__
from agent_utilities.core.config import setting

if TYPE_CHECKING:
    from llama_index.core.embeddings import BaseEmbedding


from agent_utilities.core.config import config
from agent_utilities.core.http_client import (
    create_async_http_client,
    create_http_client,
)
from agent_utilities.core.model_runtime_auth import (
    resolve_model_api_key,
    resolve_model_headers,
)

try:
    from llama_index.embeddings.ollama import OllamaEmbedding
except ImportError:
    OllamaEmbedding = None


# CONCEPT:AU-KG.compute.config-keyed-embedder-client — process-scoped embedder-client cache.
#
# ``create_embedding_model`` was rebuilding a fresh LlamaIndex embedding client on
# EVERY call. On the ingest hot path that is per-window / per-document / per-fact
# (e.g. ``FactDeduper`` builds one per ``extract_facts`` call, document processing
# and derived-property enrichers per item), so the live host log showed a
# ``Creating OpenAIEmbedding`` line on every embedding call — a new httpx client,
# TLS context, and tokenizer constructed each time on top of the actual POST.
#
# The client is stateless w.r.t. content (only the resolved provider/model/endpoint/
# key/TLS/timeout matter) and its underlying httpx client is already used
# concurrently by the batched embedder (``make_embed_fn`` fans ``get_text_embedding_batch``
# across threads on ONE model), so a shared instance keyed by those resolved inputs
# is safe to reuse for the whole run. Thread-safe (double-checked under a lock). The
# fail-loud KG-2.3 contract is unchanged — a missing provider/dep still raises; we
# only cache successful constructions.
_EMBED_MODEL_CACHE: dict[tuple[Any, ...], "BaseEmbedding"] = {}
_EMBED_MODEL_LOCK = threading.Lock()


def clear_embedding_model_cache() -> None:
    """Drop every cached embedder client (CONCEPT:AU-KG.compute.config-keyed-embedder-client).

    Mainly for tests / config hot-reload — the next ``create_embedding_model`` for a
    given key rebuilds the client.
    """
    with _EMBED_MODEL_LOCK:
        _EMBED_MODEL_CACHE.clear()


def create_embedding_model(
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    oauth2: dict[str, Any] | None = None,
    timeout: float = 300.0,
) -> "BaseEmbedding":
    """Initialize an embedding model based on provider and environment.

    Args:
        provider: Name of the embedding provider ('openai', 'ollama',
            'huggingface', 'local').
        model: Specific model identifier.
        base_url: Base URL for provider API requests.
        api_key: Optional API key for authentication.
        oauth2: OAuth2 client_credentials block (CONCEPT:AU-OS.identity.oauth2-client-credentials-lifecycle)
            — mutually exclusive with ``api_key``. See
            ``agent_utilities.security.oauth_client_credentials.OAuth2ClientCredentialsConfig``.
        timeout: Request timeout in seconds.

    Returns:
        An initialized LlamaIndex BaseEmbedding instance.

    Raises:
        ImportError: If a requested provider's dependency is missing.
        ValueError: If an unsupported provider is specified, or if both ``api_key`` and
            ``oauth2`` are supplied.

    """
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(float(timeout))
        or not 0 < float(timeout) <= 3_600
    ):
        raise ValueError(
            "embedding timeout must be finite and between 0 and 3600 seconds"
        )
    if oauth2 and api_key:
        raise ValueError(
            "create_embedding_model: 'api_key' and 'oauth2' are mutually exclusive — "
            "pass exactly one."
        )
    # Resolve defaults from the model registry. When the caller pins nothing
    # explicit (the "give me the default embedder" call — make_embed_fn and every
    # enrichment/query embed), resolve the ACTIVE failover endpoint (CONCEPT:AU-KG.enrichment.each-call-resolves-active)
    # instead of the static primary: while the primary embedder is down its breaker
    # is OPEN and this returns the FALLBACK endpoint's base_url/provider, so every
    # embed caller transparently follows the failover. The cache below keys on the
    # resolved base_url, so the cached client SWAPS to the fallback's and back on
    # recovery (no stale primary client).
    _embed_cfg = config.default_embedding_model
    _chat_cfg = config.default_chat_model
    _active_provider = _embed_cfg.provider if _embed_cfg else None
    _active_model = _embed_cfg.id if _embed_cfg else None
    _active_base_url = _embed_cfg.base_url if _embed_cfg else None
    _active_api_key_ref = _embed_cfg.api_key_ref if _embed_cfg else None
    _active_oauth2 = _embed_cfg.oauth2 if _embed_cfg else None
    # Per-model static headers are honored natively. TLS trust is selected only
    # through the runtime embedding TLS profile.
    _active_headers_ref = _embed_cfg.headers_ref if _embed_cfg else None
    if provider is None and model is None and base_url is None:
        try:
            from agent_utilities.core.embedding_failover import (
                active_embedding_endpoint,
            )

            _ep = active_embedding_endpoint()
            _active_provider = _ep.provider or _active_provider
            _active_model = _ep.model_id or _active_model
            _active_base_url = _ep.base_url or _active_base_url
            _active_api_key_ref = _ep.api_key_ref
            _active_oauth2 = _ep.oauth2
            # A failed-over embedder carries its own headers while active.
            _active_headers_ref = _ep.headers_ref
        except Exception:  # noqa: BLE001 — failover is best-effort; keep static defaults
            pass
    provider_str = (
        provider
        or _active_provider
        or (_chat_cfg.provider if _chat_cfg else None)
        or "openai"
    )
    provider_str = provider_str.lower()
    model_str = model or _active_model
    if not model_str:
        raise ValueError(
            "No embedding model is configured; set an embedding model in "
            "AgentConfig or pass model explicitly."
        )
    base_url_str = (
        base_url or _active_base_url or (_chat_cfg.base_url if _chat_cfg else None)
    )
    selected_api_key_ref = _active_api_key_ref
    oauth2_val: dict[str, Any] | None = oauth2 or _active_oauth2
    if api_key is None and selected_api_key_ref is None and oauth2_val is None:
        selected_api_key_ref = _chat_cfg.api_key_ref if _chat_cfg else None
        oauth2_val = _chat_cfg.oauth2 if _chat_cfg else None
    api_key_str = resolve_model_api_key(
        value=api_key,
        reference=selected_api_key_ref if api_key is None else None,
    )
    selected_headers_ref = _active_headers_ref or (
        _chat_cfg.headers_ref if _chat_cfg else None
    )
    _active_headers = resolve_model_headers(reference=selected_headers_ref)
    # The selected endpoint owns exactly one auth source. A chat-model fallback
    # is consulted only when the embedder declares neither API-key nor OAuth2.
    if oauth2_val and api_key_str:
        raise ValueError(
            "embedding authentication source is ambiguous"
        )

    if provider_str == "mock":
        raise ValueError(
            "Mock embeddings are strictly forbidden by Zero-Stub Compliance. Please configure a real embedding provider in AgentConfig."
        )

    # OpenAI's LM-Studio/local fallback key is resolved here so it participates in
    # the cache key (otherwise an empty-key call and a "Test-1234"-key call would
    # build two clients). The openai SDK requires a non-empty ``api_key`` string at
    # construction time even when oauth2 is configured — it is a harmless placeholder in
    # that case, immediately overwritten on every request by the oauth2 httpx.Auth below.
    if provider_str == "openai" and not api_key_str:
        api_key_str = "oauth2-managed" if oauth2_val else config.openai_api_key
        if not api_key_str:
            raise ValueError(
                "The OpenAI-compatible embedding provider requires explicit "
                "credentials; configure an API-key secret reference or OAuth2."
            )
    if provider_str == "ollama" and not base_url_str:
        raise ValueError(
            "The Ollama embedding endpoint is not configured; set its base_url "
            "in AgentConfig or pass base_url explicitly."
        )

    # CONCEPT:AU-KG.compute.config-keyed-embedder-client — return the cached client for this exact resolved config
    # instead of constructing a new one on every call. Key on every input that
    # changes the client's identity/behaviour.
    oauth2_key = json.dumps(oauth2_val, sort_keys=True) if oauth2_val else None
    headers_key = (
        json.dumps(_active_headers, sort_keys=True) if _active_headers else None
    )
    cache_key: tuple[Any, ...] = (
        provider_str,
        model_str,
        base_url_str,
        api_key_str,
        oauth2_key,
        headers_key,
        float(timeout),
    )
    cached = _EMBED_MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    with _EMBED_MODEL_LOCK:
        cached = _EMBED_MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached
        model_obj = _build_embedding_model(
            provider_str=provider_str,
            model_str=model_str,
            base_url_str=base_url_str,
            api_key_str=api_key_str,
            oauth2_cfg=oauth2_val,
            timeout=timeout,
            provider=provider,
            headers=_active_headers or None,
        )
        _EMBED_MODEL_CACHE[cache_key] = model_obj
        return model_obj


def _build_embedding_model(
    *,
    provider_str: str,
    model_str: str,
    base_url_str: str | None,
    api_key_str: str | None,
    timeout: float,
    provider: str | None,
    oauth2_cfg: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> "BaseEmbedding":
    """Construct a fresh embedding client (the un-cached path, CONCEPT:AU-KG.compute.config-keyed-embedder-client).

    Split out of :func:`create_embedding_model` so the cache wraps exactly one
    construction site. Logs once per distinct config because the caller only
    invokes it on a cache miss.
    """
    # CONCEPT:AU-OS.identity.oauth2-client-credentials-lifecycle — mint/attach an OAuth2
    # client-credentials bearer instead of relying on the static api_key baked into the client.
    # ``None`` when oauth2 is not configured (zero behaviour change).
    oauth2_auth = None
    if oauth2_cfg:
        from agent_utilities.security.oauth_client_credentials import (
            httpx_auth_from_config,
        )

        oauth2_auth = httpx_auth_from_config(oauth2_cfg)

    from agent_utilities.core.transport_security import (
        TransportSecurityError,
        resolve_configured_tls_profile,
    )

    tls_profile = resolve_configured_tls_profile(
        "embedding",
        profile_name=config.embedding_tls_profile,
        profile_ref=config.embedding_tls_profile_ref,
        config=config,
    )
    if tls_profile.proxy_url:
        raise TransportSecurityError("embedding_proxy_incompatible_with_dns_pinning")

    # OpenAI-compatible embedding requests always use the same DNS-pinned,
    # finite, redirect-free transport as chat models.  CA/mTLS policy comes
    # from the runtime TLS profile; no endpoint or certificate material is
    # copied into durable configuration or traces.
    http_client: httpx.Client | None = None
    async_http_client: httpx.AsyncClient | None = None
    if provider_str == "openai":
        http_client = create_http_client(
            verify=tls_profile.ssl_context,
            timeout=timeout,
            auth=oauth2_auth,
            headers=headers,
            pin_egress=True,
            allowed_private_hosts=config.model_http_allowed_private_hosts,
            allow_loopback=False,
            trust_env=False,
            follow_redirects=False,
        )
        async_http_client = create_async_http_client(
            verify=tls_profile.ssl_context,
            timeout=timeout,
            auth=oauth2_auth,
            headers=headers,
            pin_egress=True,
            allowed_private_hosts=config.model_http_allowed_private_hosts,
            allow_loopback=False,
            trust_env=False,
            follow_redirects=False,
        )

    if provider_str == "openai":
        import sys

        from llama_index.embeddings.openai import OpenAIEmbedding

        # One non-sensitive line per distinct embedder config (cache-miss only).
        # Credentials, endpoints, and filesystem-backed trust material are never logged.
        print(f"Creating OpenAIEmbedding model={model_str}", file=sys.stderr)

        return OpenAIEmbedding(
            model_name=model_str,
            api_key=api_key_str,
            api_base=base_url_str,
            timeout=timeout,
            http_client=http_client,
            async_http_client=async_http_client,
        )

    elif provider_str == "huggingface":
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        cache_folder = setting("HF_HOME")

        return HuggingFaceEmbedding(
            model_name=model_str,
            cache_folder=cache_folder,
            request_timeout=timeout,
        )

    elif provider_str == "ollama":
        if OllamaEmbedding is None:
            raise ImportError("llama-index-embeddings-ollama is not installed.")
        if not base_url_str:
            raise ValueError("Ollama embedding endpoint is not configured")
        ollama_headers = dict(headers or {})
        if api_key_str and not any(
            key.casefold() == "authorization" for key in ollama_headers
        ):
            ollama_headers["Authorization"] = f"Bearer {api_key_str}"
        model_obj = OllamaEmbedding(
            model_name=model_str,
            base_url=base_url_str,
            client_kwargs={
                "verify": tls_profile.ssl_context,
                "timeout": timeout,
                "follow_redirects": False,
                "trust_env": False,
                "headers": ollama_headers,
            },
        )
        # Ollama constructs its own httpx clients. Replace their transports
        # before first use so local/private endpoints obey the exact AgentConfig
        # allow-list and every request is DNS-pinned and peer-verified.
        safe_sync = create_http_client(
            base_url=base_url_str,
            verify=tls_profile.ssl_context,
            timeout=timeout,
            headers=ollama_headers,
            pin_egress=True,
            allowed_private_hosts=config.model_http_allowed_private_hosts,
            allow_loopback=False,
            trust_env=False,
            follow_redirects=False,
        )
        safe_async = create_async_http_client(
            base_url=base_url_str,
            verify=tls_profile.ssl_context,
            timeout=timeout,
            headers=ollama_headers,
            pin_egress=True,
            allowed_private_hosts=config.model_http_allowed_private_hosts,
            allow_loopback=False,
            trust_env=False,
            follow_redirects=False,
        )
        old_sync = model_obj._client._client  # noqa: SLF001
        old_async = model_obj._async_client._client  # noqa: SLF001
        model_obj._client._client = safe_sync  # noqa: SLF001
        model_obj._async_client._client = safe_async  # noqa: SLF001
        old_sync.close()
        try:
            asyncio.get_running_loop().create_task(old_async.aclose())
        except RuntimeError:
            asyncio.run(old_async.aclose())
        return model_obj

    elif provider_str == "local":
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        return HuggingFaceEmbedding(model_name=model_str)

    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")
