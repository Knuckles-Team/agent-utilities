#!/usr/bin/python
"""Embedding Utilities Module.

CONCEPT:AU-KG.memory.auto-similarity-memory-graph

This module provides factory functions for initializing embedding model
clients over plain OpenAI-compatible / Ollama HTTP (D2: no llama-index — see
:class:`_HttpEmbeddingModel`). Local/HuggingFace inference is not served from
core; see :func:`_build_huggingface_embedding`.
"""

import json
import math
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from agent_utilities._version import __version__ as __version__

if TYPE_CHECKING:
    import httpx


from agent_utilities.core.config import config
from agent_utilities.core.http_client import create_http_client
from agent_utilities.core.model_runtime_auth import (
    resolve_model_api_key,
    resolve_model_headers,
)

EmbedBatchFn = Callable[["httpx.Client", list[str], str], list[list[float]]]


class _HttpEmbeddingModel:
    """Minimal embedding client (D2 — no llama-index import anywhere in core).

    Implements exactly the subset of the BaseEmbedding-shaped interface every
    AU call site uses: ``model_name``, ``embed_batch_size`` (mutable — a
    caller may raise it before a bulk embed), ``get_text_embedding``, and
    ``get_text_embedding_batch``. Every provider (openai/ollama) supplies its
    own ``embed_fn`` closure; this class owns only sub-batching by
    ``embed_batch_size`` and the single-text convenience wrapper.
    """

    def __init__(
        self,
        model_name: str,
        client: "httpx.Client",
        embed_fn: EmbedBatchFn,
        embed_batch_size: int = 10,
    ) -> None:
        self.model_name = model_name
        self.embed_batch_size = embed_batch_size
        self._client = client
        self._embed_fn = embed_fn

    def get_text_embedding(self, text: str) -> list[float]:
        return self.get_text_embedding_batch([text])[0]

    def get_text_embedding_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        step = max(1, self.embed_batch_size)
        for i in range(0, len(texts), step):
            out.extend(
                self._embed_fn(self._client, texts[i : i + step], self.model_name)
            )
        return out


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
_EMBED_MODEL_CACHE: dict[tuple[Any, ...], "_HttpEmbeddingModel"] = {}
_EMBED_MODEL_LOCK = threading.Lock()

# CONCEPT:AU-KG.retrieval.embedding-fast-fail — bound the OpenAI SDK's OWN internal
# retry loop instead of inheriting llama-index's default (``max_retries=10``,
# exponential backoff up to 8s per retry — worst case ~55s of pure backoff sleep,
# or up to 10x the per-attempt ``timeout`` on a genuine hang, BEFORE the caller ever
# sees a failure). agent-utilities already owns a SEPARATE, endpoint-aware
# circuit-breaker/backoff layer (``core.model_circuit_breaker`` fed by
# ``core.model_concurrency.map_concurrent_sync`` for bulk embeds and directly by
# the retrieval query-time embed) — a second, SDK-internal retry loop on top of
# that only adds latency without adding resilience. One retry tolerates a single
# transient blip; repeated failures are the breaker's job, not the SDK's.
_EMBED_SDK_MAX_RETRIES = 1


def clear_embedding_model_cache() -> None:
    """Drop every cached embedder client (CONCEPT:AU-KG.compute.config-keyed-embedder-client).

    Mainly for tests / config hot-reload — the next ``create_embedding_model`` for a
    given key rebuilds the client.
    """
    with _EMBED_MODEL_LOCK:
        _EMBED_MODEL_CACHE.clear()


def _validate_embedding_timeout(timeout: float) -> None:
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, int | float)
        or not math.isfinite(float(timeout))
        or not 0 < float(timeout) <= 3_600
    ):
        raise ValueError(
            "embedding timeout must be finite and between 0 and 3600 seconds"
        )


_ActiveEndpointT = tuple[
    str | None, str | None, str | None, str | None, dict[str, Any] | None, str | None
]


def _static_embedding_endpoint_defaults() -> _ActiveEndpointT:
    # Resolve defaults from the model registry.
    _embed_cfg = config.default_embedding_model
    _active_provider = _embed_cfg.provider if _embed_cfg else None
    _active_model = _embed_cfg.id if _embed_cfg else None
    _active_base_url = _embed_cfg.base_url if _embed_cfg else None
    _active_api_key_ref = _embed_cfg.api_key_ref if _embed_cfg else None
    _active_oauth2 = _embed_cfg.oauth2 if _embed_cfg else None
    # Per-model static headers are honored natively. TLS trust is selected only
    # through the runtime embedding TLS profile.
    _active_headers_ref = _embed_cfg.headers_ref if _embed_cfg else None
    return (
        _active_provider,
        _active_model,
        _active_base_url,
        _active_api_key_ref,
        _active_oauth2,
        _active_headers_ref,
    )


def _apply_active_failover_override(
    static_defaults: _ActiveEndpointT,
) -> _ActiveEndpointT:
    # When the caller pins nothing explicit (the "give me the default embedder"
    # call — make_embed_fn and every enrichment/query embed), resolve the
    # ACTIVE failover endpoint (CONCEPT:AU-KG.enrichment.each-call-resolves-active)
    # instead of the static primary: while the primary embedder is down its
    # breaker is OPEN and this returns the FALLBACK endpoint's
    # base_url/provider, so every embed caller transparently follows the
    # failover.
    (
        _active_provider,
        _active_model,
        _active_base_url,
        _active_api_key_ref,
        _active_oauth2,
        _active_headers_ref,
    ) = static_defaults
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
    return (
        _active_provider,
        _active_model,
        _active_base_url,
        _active_api_key_ref,
        _active_oauth2,
        _active_headers_ref,
    )


def _resolve_active_embedding_endpoint(
    provider: str | None, model: str | None, base_url: str | None
) -> _ActiveEndpointT:
    # The cache below keys on the resolved base_url, so the cached client
    # SWAPS to the fallback's and back on recovery (no stale primary client).
    static_defaults = _static_embedding_endpoint_defaults()
    if provider is None and model is None and base_url is None:
        return _apply_active_failover_override(static_defaults)
    return static_defaults


def _resolve_embedding_identity(
    provider: str | None,
    model: str | None,
    base_url: str | None,
    _active_provider: str | None,
    _active_model: str | None,
    _active_base_url: str | None,
    _chat_cfg: Any,
) -> tuple[str, str, str | None]:
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
    return provider_str, model_str, base_url_str


def _resolve_embedding_auth_refs(
    api_key: str | None,
    oauth2: dict[str, Any] | None,
    _active_api_key_ref: str | None,
    _active_oauth2: dict[str, Any] | None,
    _chat_cfg: Any,
) -> tuple[str | None, dict[str, Any] | None]:
    selected_api_key_ref = _active_api_key_ref
    oauth2_val: dict[str, Any] | None = oauth2 or _active_oauth2
    if api_key is None and selected_api_key_ref is None and oauth2_val is None:
        selected_api_key_ref = _chat_cfg.api_key_ref if _chat_cfg else None
        oauth2_val = _chat_cfg.oauth2 if _chat_cfg else None
    return selected_api_key_ref, oauth2_val


def _resolve_embedding_credentials(
    api_key: str | None,
    oauth2: dict[str, Any] | None,
    _active_api_key_ref: str | None,
    _active_oauth2: dict[str, Any] | None,
    _active_headers_ref: str | None,
    _chat_cfg: Any,
) -> tuple[str | None, dict[str, Any] | None, dict[str, str] | None]:
    selected_api_key_ref, oauth2_val = _resolve_embedding_auth_refs(
        api_key, oauth2, _active_api_key_ref, _active_oauth2, _chat_cfg
    )
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
        raise ValueError("embedding authentication source is ambiguous")
    return api_key_str, oauth2_val, _active_headers


def _validate_provider_credentials(
    provider_str: str,
    api_key_str: str | None,
    oauth2_val: dict[str, Any] | None,
    base_url_str: str | None,
) -> str | None:
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
    return api_key_str


def _embedding_cache_key(
    provider_str: str,
    model_str: str,
    base_url_str: str | None,
    api_key_str: str | None,
    oauth2_val: dict[str, Any] | None,
    _active_headers: dict[str, str] | None,
    timeout: float,
) -> tuple[Any, ...]:
    # CONCEPT:AU-KG.compute.config-keyed-embedder-client — key on every input that
    # changes the client's identity/behaviour.
    oauth2_key = json.dumps(oauth2_val, sort_keys=True) if oauth2_val else None
    headers_key = (
        json.dumps(_active_headers, sort_keys=True) if _active_headers else None
    )
    return (
        provider_str,
        model_str,
        base_url_str,
        api_key_str,
        oauth2_key,
        headers_key,
        float(timeout),
    )


def create_embedding_model(
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    oauth2: dict[str, Any] | None = None,
    timeout: float = 300.0,
) -> "_HttpEmbeddingModel":
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
        An initialized embedding client (see `_HttpEmbeddingModel`).

    Raises:
        ImportError: If a requested provider's dependency is missing.
        ValueError: If an unsupported provider is specified, or if both ``api_key`` and
            ``oauth2`` are supplied.

    """
    _validate_embedding_timeout(timeout)
    if oauth2 and api_key:
        raise ValueError(
            "create_embedding_model: 'api_key' and 'oauth2' are mutually exclusive — "
            "pass exactly one."
        )
    _chat_cfg = config.default_chat_model
    (
        _active_provider,
        _active_model,
        _active_base_url,
        _active_api_key_ref,
        _active_oauth2,
        _active_headers_ref,
    ) = _resolve_active_embedding_endpoint(provider, model, base_url)

    provider_str, model_str, base_url_str = _resolve_embedding_identity(
        provider,
        model,
        base_url,
        _active_provider,
        _active_model,
        _active_base_url,
        _chat_cfg,
    )

    api_key_str, oauth2_val, _active_headers = _resolve_embedding_credentials(
        api_key,
        oauth2,
        _active_api_key_ref,
        _active_oauth2,
        _active_headers_ref,
        _chat_cfg,
    )

    api_key_str = _validate_provider_credentials(
        provider_str, api_key_str, oauth2_val, base_url_str
    )

    cache_key = _embedding_cache_key(
        provider_str,
        model_str,
        base_url_str,
        api_key_str,
        oauth2_val,
        _active_headers,
        timeout,
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


def _resolve_embedding_oauth2_auth(oauth2_cfg: dict[str, Any] | None) -> Any | None:
    # CONCEPT:AU-OS.identity.oauth2-client-credentials-lifecycle — mint/attach an OAuth2
    # client-credentials bearer instead of relying on the static api_key baked into the client.
    # ``None`` when oauth2 is not configured (zero behaviour change).
    oauth2_auth = None
    if oauth2_cfg:
        from agent_utilities.security.oauth_client_credentials import (
            httpx_auth_from_config,
        )

        oauth2_auth = httpx_auth_from_config(oauth2_cfg)
    return oauth2_auth


def _resolve_embedding_tls_profile() -> Any:
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
    return tls_profile


def _raise_for_embedding_status(response: "httpx.Response") -> None:
    if response.status_code >= 400:
        # Never echo response body: it may carry request text or provider detail.
        raise ValueError(f"embedding request failed with HTTP {response.status_code}")


def _post_with_one_retry(
    client: "httpx.Client", url: str, payload: dict[str, Any]
) -> "httpx.Response":
    import httpx

    attempts = _EMBED_SDK_MAX_RETRIES + 1
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            response = client.post(url, json=payload)
            if response.status_code >= 500 and attempt < attempts - 1:
                continue
            _raise_for_embedding_status(response)
            return response
        except httpx.TransportError as exc:
            last_exc = exc
            if attempt >= attempts - 1:
                raise
    raise last_exc or RuntimeError("embedding request failed")


def _openai_embed_fn(
    client: "httpx.Client", texts: list[str], model_name: str
) -> list[list[float]]:
    response = _post_with_one_retry(
        client, "/embeddings", {"model": model_name, "input": texts}
    )
    rows = response.json()["data"]
    return [row["embedding"] for row in sorted(rows, key=lambda r: r["index"])]


def _ollama_embed_fn(
    client: "httpx.Client", texts: list[str], model_name: str
) -> list[list[float]]:
    response = _post_with_one_retry(
        client, "/api/embed", {"model": model_name, "input": texts}
    )
    embeddings = response.json().get("embeddings")
    if embeddings is None:
        raise ValueError("Ollama embeddings response is missing 'embeddings'")
    return embeddings


def _openai_base_url(base_url_str: str | None) -> str:
    return (base_url_str or "https://api.openai.com/v1").rstrip("/")


def _openai_auth_headers(
    headers: dict[str, str] | None, api_key_str: str | None, oauth2_auth: Any | None
) -> dict[str, str]:
    # oauth2, when configured, is the http_client's own `auth=` -- a static
    # Authorization header would fight it, so only set one from api_key_str
    # when oauth2 is absent.
    request_headers = dict(headers or {})
    if api_key_str and oauth2_auth is None:
        request_headers.setdefault("Authorization", f"Bearer {api_key_str}")
    return request_headers


def _build_openai_embedding(
    model_str: str,
    api_key_str: str | None,
    base_url_str: str | None,
    timeout: float,
    tls_profile: Any,
    oauth2_auth: Any | None,
    headers: dict[str, str] | None,
) -> "_HttpEmbeddingModel":
    import sys

    # One non-sensitive line per distinct embedder config (cache-miss only).
    # Credentials, endpoints, and filesystem-backed trust material are never logged.
    print(
        f"Creating OpenAI-compatible embedding client model={model_str}",
        file=sys.stderr,
    )

    request_headers = _openai_auth_headers(headers, api_key_str, oauth2_auth)
    client = create_http_client(
        base_url=_openai_base_url(base_url_str),
        verify=tls_profile.ssl_context,
        timeout=timeout,
        auth=oauth2_auth,
        headers=request_headers,
        pin_egress=True,
        allowed_private_hosts=config.model_http_allowed_private_hosts,
        allow_loopback=False,
        trust_env=False,
        follow_redirects=False,
    )
    return _HttpEmbeddingModel(model_str, client, _openai_embed_fn)


def _build_huggingface_embedding(
    model_str: str, timeout: float
) -> "_HttpEmbeddingModel":
    raise ValueError(
        "Local/HuggingFace embedding inference is not served from agent-utilities "
        "core (heavy ML deps stay out of the serving plane — see AGENTS.md "
        "'Dependency discipline'). Use agents/data-science-mcp for local embedding "
        "inference, or configure a remote OpenAI-compatible/Ollama endpoint instead."
    )


def _ollama_auth_headers(
    headers: dict[str, str] | None, api_key_str: str | None
) -> dict[str, str]:
    ollama_headers = dict(headers or {})
    if api_key_str and not any(
        key.casefold() == "authorization" for key in ollama_headers
    ):
        ollama_headers["Authorization"] = f"Bearer {api_key_str}"
    return ollama_headers


def _build_ollama_embedding(
    model_str: str,
    base_url_str: str | None,
    api_key_str: str | None,
    timeout: float,
    tls_profile: Any,
    headers: dict[str, str] | None,
) -> "_HttpEmbeddingModel":
    if not base_url_str:
        raise ValueError("Ollama embedding endpoint is not configured")
    ollama_headers = _ollama_auth_headers(headers, api_key_str)
    client = create_http_client(
        base_url=base_url_str.rstrip("/"),
        verify=tls_profile.ssl_context,
        timeout=timeout,
        headers=ollama_headers,
        pin_egress=True,
        allowed_private_hosts=config.model_http_allowed_private_hosts,
        allow_loopback=False,
        trust_env=False,
        follow_redirects=False,
    )
    return _HttpEmbeddingModel(model_str, client, _ollama_embed_fn)


def _build_local_embedding(model_str: str) -> "_HttpEmbeddingModel":
    raise ValueError(
        "Local embedding inference is not served from agent-utilities core "
        "(heavy ML deps stay out of the serving plane — see AGENTS.md "
        "'Dependency discipline'). Use agents/data-science-mcp for local "
        "embedding inference, or configure a remote OpenAI-compatible/Ollama "
        "endpoint instead."
    )


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
) -> "_HttpEmbeddingModel":
    """Construct a fresh embedding client (the un-cached path, CONCEPT:AU-KG.compute.config-keyed-embedder-client).

    Split out of :func:`create_embedding_model` so the cache wraps exactly one
    construction site. Logs once per distinct config because the caller only
    invokes it on a cache miss.
    """
    oauth2_auth = _resolve_embedding_oauth2_auth(oauth2_cfg)
    tls_profile = _resolve_embedding_tls_profile()

    if provider_str == "openai":
        return _build_openai_embedding(
            model_str,
            api_key_str,
            base_url_str,
            timeout,
            tls_profile,
            oauth2_auth,
            headers,
        )
    elif provider_str == "huggingface":
        return _build_huggingface_embedding(model_str, timeout)
    elif provider_str == "ollama":
        return _build_ollama_embedding(
            model_str, base_url_str, api_key_str, timeout, tls_profile, headers
        )
    elif provider_str == "local":
        return _build_local_embedding(model_str)
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")
