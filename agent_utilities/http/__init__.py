"""``agent_utilities.http`` — the shared fleet HTTP client library.

CONCEPT:ECO-4.35 — Fleet HTTP Client Library

The consolidation audit's highest-leverage item: 10+ connector repos each
carried a ~95%-identical ``api_client_base.py`` with no retry, no rate-limit
capture, and no log redaction. This package is the single shared base —
built on the canonical :mod:`agent_utilities.core.http_client` factory —
that fleet connectors strangle their local copies onto.

Components:

* :class:`BaseApiClient` / :class:`AsyncBaseApiClient` — envelope-returning
  httpx clients with auth injection, 429 backoff, error mapping, redaction,
  destructive gating and pagination (:mod:`agent_utilities.http.client`).
* Auth strategies — :class:`TokenAuth` (``Bearer``/``SSWS``/bare; static or
  ``token_provider`` callable), :class:`BasicAuth`,
  :class:`QueryApiKeyAuth` (:mod:`agent_utilities.http.auth`).
* Rate-limit telemetry — :class:`RateLimitSnapshot` /
  :class:`RateLimitCapture` / :func:`backoff_seconds`
  (:mod:`agent_utilities.http.rate_limit`).
* Pagination — :class:`PaginationIterator` / :class:`AsyncPaginationIterator`
  over cursor / page / offset / Link-header / since-id dialects
  (:mod:`agent_utilities.http.pagination`).
* Log redaction — :class:`LogRedactor` / :func:`redact_text`
  (:mod:`agent_utilities.http.redaction`).

Retry: which abstraction to use
-------------------------------

Two retry mechanisms exist in agent-utilities; they are **not** redundant
and serve different layers — pick by what is being retried:

* :class:`~agent_utilities.orchestration.resilience.ResiliencePolicy`
  (CONCEPT:ORCH-1.36) retries an **in-process callable** — for HTTP that
  means transport failures (connect errors, resets, timeouts) via the
  ``retry=`` parameter on these clients / the core factory's
  :func:`~agent_utilities.core.http_client.http_retry_policy`. HTTP 429
  backoff is handled separately by the client itself (rate-limit aware).
  **This is the retry abstraction for HTTP clients.**
* :class:`RetryManager` / :class:`RetryConfig` (CONCEPT:ORCH-1.3, from
  :mod:`agent_utilities.security.execution_stability_engine`) retry a whole
  **agent execution** verified by shell ``SuccessCheck`` commands, with
  ``on_failure`` remediation hooks — orchestration-level run-until-green
  loops, not network calls. They are re-exported here because the
  consolidation audit found them unexported; do **not** reach for them to
  retry HTTP requests.
"""

from __future__ import annotations

from typing import Any

from agent_utilities.core.http_client import (
    DEFAULT_HTTP_TIMEOUT_S,
    create_async_http_client,
    create_http_client,
    http_retry_policy,
)
from agent_utilities.http.auth import (
    AuthHeaderInjector,
    BasicAuth,
    QueryApiKeyAuth,
    TokenAuth,
)
from agent_utilities.http.client import (
    DEFAULT_ERROR_MAP,
    DEFAULT_MAX_RETRIES_429,
    AsyncBaseApiClient,
    BaseApiClient,
    DestructiveOperationError,
)
from agent_utilities.http.pagination import (
    AsyncPaginationIterator,
    PaginationIterator,
)
from agent_utilities.http.rate_limit import (
    DEFAULT_RETRY_AFTER_CAP_S,
    RateLimitCapture,
    RateLimitSnapshot,
    backoff_seconds,
    parse_rate_limit,
)
from agent_utilities.http.redaction import REDACTED, LogRedactor, redact_text

__all__ = [
    "DEFAULT_ERROR_MAP",
    "DEFAULT_HTTP_TIMEOUT_S",
    "DEFAULT_MAX_RETRIES_429",
    "DEFAULT_RETRY_AFTER_CAP_S",
    "REDACTED",
    "AsyncBaseApiClient",
    "AsyncPaginationIterator",
    "AuthHeaderInjector",
    "BaseApiClient",
    "BasicAuth",
    "DestructiveOperationError",
    "LogRedactor",
    "PaginationIterator",
    "QueryApiKeyAuth",
    "RateLimitCapture",
    "RateLimitSnapshot",
    "RetryConfig",
    "RetryManager",
    "TokenAuth",
    "backoff_seconds",
    "create_async_http_client",
    "create_http_client",
    "http_retry_policy",
    "parse_rate_limit",
    "redact_text",
]


def __getattr__(name: str) -> Any:
    # RetryManager/RetryConfig are loaded lazily: they live in the (heavier)
    # security pillar and are exported here for discoverability only — see
    # the module docstring for when to use them vs ResiliencePolicy.
    if name in ("RetryConfig", "RetryManager"):
        from agent_utilities.security.execution_stability_engine import (
            RetryConfig,
            RetryManager,
        )

        return {"RetryConfig": RetryConfig, "RetryManager": RetryManager}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
