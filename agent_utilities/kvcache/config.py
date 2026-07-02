"""Configuration for the epistemic-graph remote KV-cache connector.

CONCEPT:KG-2.306 — the Python LMCache/vLLM connector reads its endpoint and
bearer token from the SAME environment the engine's HTTP KV-cache listener
(CONCEPT:EG-187) is configured with, so a co-located deploy shares one source of
truth. All reads route through :func:`agent_utilities.core.config.setting`
(configuration discipline — never ``os.environ`` directly).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from agent_utilities.core.config import setting

#: Engine default bind for the KV-cache HTTP listener (EG-187). A connector on the
#: same host points here unless overridden.
DEFAULT_KVCACHE_ADDR = "127.0.0.1:9130"


def _addr_to_base_url(addr: str) -> str:
    """Coerce an engine ``EPISTEMIC_GRAPH_KVCACHE_ADDR`` value into a client URL.

    The engine accepts a bare enable token (bind localhost default), a bare port,
    or a full ``host:port``. On the client side we only ever need a reachable
    ``http://host:port`` base URL, so:

    * an already-qualified ``http://`` / ``https://`` value passes through;
    * a bare integer port ``9130`` becomes ``http://127.0.0.1:9130``;
    * a ``host:port`` becomes ``http://host:port``;
    * anything else (a bare enable token like ``"1"``/``"on"``) falls back to the
      localhost default.
    """
    value = (addr or "").strip()
    if not value:
        return f"http://{DEFAULT_KVCACHE_ADDR}"
    if value.startswith(("http://", "https://")):
        return value.rstrip("/")
    # A bare truthy enable token (e.g. "1", "true", "on") ⇒ localhost default.
    # Checked before the numeric-port case so "1" is read as "enabled", matching
    # the engine's "bare enable token binds the localhost default" semantics.
    if value.lower() in {"1", "true", "yes", "on", "enable", "enabled"}:
        return f"http://{DEFAULT_KVCACHE_ADDR}"
    if value.isdigit():
        return f"http://127.0.0.1:{value}"
    if ":" in value:
        return f"http://{value}"
    # Any other bare token ⇒ localhost default.
    return f"http://{DEFAULT_KVCACHE_ADDR}"


class KvCacheConfig(BaseModel):
    """Endpoint + auth + timeout settings for :class:`EpistemicGraphKVBackend`.

    CONCEPT:KG-2.306. Prefer :meth:`from_env`, which mirrors the engine's
    EG-187 environment variables so client and server stay in lockstep.
    """

    base_url: str = Field(
        default=f"http://{DEFAULT_KVCACHE_ADDR}",
        description="Base URL of the engine KV-cache HTTP surface, e.g. "
        "'http://127.0.0.1:9130'. Endpoints hang off '/kv/...'.",
    )
    token: str | None = Field(
        default=None,
        description="Bearer token sent as 'Authorization: Bearer <token>'. "
        "None ⇒ anonymous (engine loopback default).",
    )
    timeout_s: float = Field(
        default=2.0,
        gt=0,
        description="Per-request timeout (seconds). Deliberately short: this sits "
        "on the inference hot path, so a slow engine must degrade to a local miss "
        "quickly rather than stall token generation.",
    )
    max_connections: int = Field(
        default=32,
        gt=0,
        description="Upper bound on pooled keep-alive connections for reuse across "
        "the many small get/put calls an inference worker issues.",
    )
    verify_tls: bool = Field(
        default=True,
        description="TLS verification. Only disable for an explicit, justified "
        "insecure (e.g. plain-http loopback is unaffected).",
    )

    @classmethod
    def from_env(cls) -> KvCacheConfig:
        """Build config from the engine's EG-187 environment variables.

        CONCEPT:KG-2.306. Recognized variables:

        * ``EPISTEMIC_GRAPH_KVCACHE_URL`` — explicit client base URL (wins).
        * ``EPISTEMIC_GRAPH_KVCACHE_ADDR`` — the engine bind value, coerced to a
          base URL via :func:`_addr_to_base_url`.
        * ``EPISTEMIC_GRAPH_KVCACHE_TOKEN`` — bearer token.
        * ``EPISTEMIC_GRAPH_KVCACHE_TIMEOUT_S`` — per-request timeout override.
        * ``EPISTEMIC_GRAPH_KVCACHE_TLS_VERIFY`` — TLS verification toggle.
        """
        explicit_url = setting("EPISTEMIC_GRAPH_KVCACHE_URL", "")
        if explicit_url:
            base_url = explicit_url.rstrip("/")
        else:
            addr = setting("EPISTEMIC_GRAPH_KVCACHE_ADDR", DEFAULT_KVCACHE_ADDR)
            base_url = _addr_to_base_url(addr)

        return cls(
            base_url=base_url,
            token=setting("EPISTEMIC_GRAPH_KVCACHE_TOKEN", None),
            timeout_s=setting("EPISTEMIC_GRAPH_KVCACHE_TIMEOUT_S", 2.0),
            max_connections=setting("EPISTEMIC_GRAPH_KVCACHE_MAX_CONNECTIONS", 32),
            verify_tls=setting("EPISTEMIC_GRAPH_KVCACHE_TLS_VERIFY", True),
        )
