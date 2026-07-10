#!/usr/bin/python
"""Embedding Utilities Module.

CONCEPT:AU-KG.memory.auto-similarity-memory-graph

This module provides factory functions for initializing LlamaIndex-compatible
embedding models. It supports various providers including OpenAI, Ollama,
HuggingFace, and local models, with robust environment-based configuration.
"""

import json
import threading
from typing import TYPE_CHECKING, Any

import httpx

from agent_utilities.core.config import setting

if TYPE_CHECKING:
    from llama_index.core.embeddings import BaseEmbedding


from agent_utilities.base_utilities import to_boolean
from agent_utilities.core.config import config
from agent_utilities.core.http_client import create_http_client

try:
    from llama_index.embeddings.ollama import OllamaEmbedding
except ImportError:
    OllamaEmbedding = None


__version__ = "0.2.40"


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
    ssl_verify: bool = to_boolean(string=setting("SSL_VERIFY", "true")),
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
        ssl_verify: Whether to verify SSL certificates.
        timeout: Request timeout in seconds.

    Returns:
        An initialized LlamaIndex BaseEmbedding instance.

    Raises:
        ImportError: If a requested provider's dependency is missing.
        ValueError: If an unsupported provider is specified, or if both ``api_key`` and
            ``oauth2`` are supplied.

    """
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
    _active_api_key = _embed_cfg.api_key if _embed_cfg else None
    _active_oauth2 = _embed_cfg.oauth2 if _embed_cfg else None
    if provider is None and model is None and base_url is None:
        try:
            from agent_utilities.core.embedding_failover import (
                active_embedding_endpoint,
            )

            _ep = active_embedding_endpoint()
            _active_provider = _ep.provider or _active_provider
            _active_model = _ep.model_id or _active_model
            _active_base_url = _ep.base_url or _active_base_url
            _active_api_key = _ep.api_key or _active_api_key
            _active_oauth2 = _ep.oauth2 or _active_oauth2
        except Exception:  # noqa: BLE001 — failover is best-effort; keep static defaults
            pass
    provider_str = (
        provider
        or _active_provider
        or (_chat_cfg.provider if _chat_cfg else None)
        or "openai"
    )
    provider_str = provider_str.lower()
    model_str = model or _active_model or "text-embedding-nomic-embed-text-v2-moe"
    base_url_str = (
        base_url
        or _active_base_url
        or (_chat_cfg.base_url if _chat_cfg else None)
        or "http://vllm-embed.arpa/v1"
    )
    api_key_str = (
        api_key or _active_api_key or (_chat_cfg.api_key if _chat_cfg else None)
    )
    # CONCEPT:AU-OS.identity.oauth2-client-credentials-lifecycle — an explicit call-site oauth2
    # wins; else the resolved (possibly failed-over) embedding endpoint's oauth2 block; else the
    # default chat model's, mirroring the api_key_str fallback chain above.
    oauth2_val: dict[str, Any] | None = (
        oauth2 or _active_oauth2 or (_chat_cfg.oauth2 if _chat_cfg else None)
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
        api_key_str = "oauth2-managed" if oauth2_val else (config.openai_api_key or "Test-1234")

    # CONCEPT:AU-KG.compute.config-keyed-embedder-client — return the cached client for this exact resolved config
    # instead of constructing a new one on every call. Key on every input that
    # changes the client's identity/behaviour.
    oauth2_key = json.dumps(oauth2_val, sort_keys=True) if oauth2_val else None
    cache_key: tuple[Any, ...] = (
        provider_str,
        model_str,
        base_url_str,
        api_key_str,
        oauth2_key,
        bool(ssl_verify),
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
            ssl_verify=ssl_verify,
            timeout=timeout,
            provider=provider,
        )
        _EMBED_MODEL_CACHE[cache_key] = model_obj
        return model_obj


def _build_embedding_model(
    *,
    provider_str: str,
    model_str: str,
    base_url_str: str,
    api_key_str: str | None,
    ssl_verify: bool,
    timeout: float,
    provider: str | None,
    oauth2_cfg: dict[str, Any] | None = None,
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

    # TLS verification is ON unless the deployment's SSL_VERIFY flag (or an
    # explicit ssl_verify=False argument) opts out — the canonical factory
    # keeps verify=True the default; insecure stays a per-call decision. An oauth2-configured
    # endpoint always gets a real client (sync + async) so the bearer is injected on every
    # request regardless of the ssl_verify setting.
    http_client: httpx.Client | None = None
    async_http_client: httpx.AsyncClient | None = None
    if not ssl_verify or oauth2_auth is not None:
        http_client = create_http_client(
            verify=ssl_verify, timeout=timeout, auth=oauth2_auth
        )  # nosec B501
        if oauth2_auth is not None:
            from agent_utilities.core.http_client import create_async_http_client

            async_http_client = create_async_http_client(
                verify=ssl_verify, timeout=timeout, auth=oauth2_auth
            )

    if provider_str == "openai":
        import sys

        from llama_index.embeddings.openai import OpenAIEmbedding

        # One line per distinct embedder config now (cache-miss only), not per call. Never logs
        # the oauth2 bearer/secret — only the static api_key placeholder, which is a harmless
        # sentinel ("Test-1234"/"EMPTY"-style) when oauth2 is configured (CONCEPT:AU-OS.identity.oauth2-client-credentials-lifecycle).
        print(f"Creating OpenAIEmbedding with key={api_key_str}", file=sys.stderr)

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

        return OllamaEmbedding(
            model_name=model_str,
            base_url=base_url_str,
            timeout=timeout,
        )

    elif provider_str == "local":
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        return HuggingFaceEmbedding(model_name=model_str)

    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")
