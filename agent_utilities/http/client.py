"""Shared base API client for the connector fleet.

CONCEPT:AU-ECO.ui.fleet-http-client-library — Fleet HTTP Client Library

The consolidation audit found a ~95%-identical ``api_client_base.py``
duplicated across 10+ connector repos, none with retry, rate-limit capture,
or log redaction. :class:`BaseApiClient` (sync) and :class:`AsyncBaseApiClient`
are the single shared base those repos strangle onto, built on the canonical
:mod:`agent_utilities.core.http_client` factory so every fleet client inherits
the unified safety defaults (finite timeout, TLS verification on, standard
headers, optional :class:`~agent_utilities.orchestration.resilience.ResiliencePolicy`
transport retry).

What every fleet client now gets for free:

* base-URL joining (httpx semantics: relative endpoints join under
  ``base_url``; absolute URLs pass through — pagination next-links work);
* a uniform response envelope ``{"status_code", "data", "rate_limit"}``
  (dockerhub-api's shape, the newest fleet convention; add ``"headers"``
  via ``include_response_headers=True``);
* typed rate-limit capture on every response plus bounded auto-backoff on
  HTTP 429 (``Retry-After`` / ``X-Rate-Limit-Reset``, capped — okta/dockerhub
  semantics);
* a typed error-mapping hook (status → exception class, override via
  ``error_map`` or :meth:`_map_error`) raising the canonical
  :mod:`agent_utilities.core.exceptions` types;
* pluggable auth strategies (:mod:`agent_utilities.http.auth`) with one
  transparent retry after 401 when the strategy exposes ``invalidate()``;
* log redaction by default — the module logger carries a
  :class:`~agent_utilities.http.LogRedactor`, and error messages are
  redacted before they are raised;
* destructive-action gating (:meth:`guard_destructive`);
* pagination (:meth:`paginate`) over five dialects.

Migrating from ``requests.Session`` (the older fleet convention): this base
is httpx-only for one-stack coherence with the core factory. The constructor
covers the same surface — ``verify`` and default headers move to constructor
arguments, ``session.auth=(user, pass)`` becomes
``auth=BasicAuth(user, pass)``, header tokens become ``auth=TokenAuth(...)``,
and tests swap ``requests_mock`` for ``transport=httpx.MockTransport(...)``.
Per-request ``timeout=`` keeps its meaning.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

import httpx

from agent_utilities.base_utilities import get_logger
from agent_utilities.core.exceptions import (
    ApiError,
    AuthError,
    ParameterError,
    UnauthorizedError,
)
from agent_utilities.core.http_client import (
    DEFAULT_HTTP_TIMEOUT_S,
    create_async_http_client,
    create_http_client,
)
from agent_utilities.http.auth import AuthHeaderInjector
from agent_utilities.http.pagination import AsyncPaginationIterator, PaginationIterator
from agent_utilities.http.rate_limit import (
    DEFAULT_RETRY_AFTER_CAP_S,
    RateLimitCapture,
    RateLimitSnapshot,
    backoff_seconds,
)
from agent_utilities.http.redaction import LogRedactor, redact_text

if TYPE_CHECKING:
    from agent_utilities.orchestration.resilience import ResiliencePolicy

__all__ = [
    "DEFAULT_ERROR_MAP",
    "DEFAULT_MAX_RETRIES_429",
    "AsyncBaseApiClient",
    "BaseApiClient",
    "DestructiveOperationError",
]

logger = get_logger(__name__)
#: Default log filter for BaseApiClient — credential material never reaches
#: log sinks; clients register their literal secrets here at construction.
_log_redactor = LogRedactor()
logger.addFilter(_log_redactor)

JSON_CONTENT_TYPE = "application/json"
DEFAULT_MAX_RETRIES_429 = 3

#: Default status → exception mapping (dockerhub-api convention).
DEFAULT_ERROR_MAP: dict[int, type[Exception]] = {
    400: ParameterError,
    401: AuthError,
    403: UnauthorizedError,
    404: ParameterError,
}


class DestructiveOperationError(PermissionError):
    """Raised when a destructive operation is attempted while gated off."""


class _ApiClientCore:
    """Configuration and pure helpers shared by the sync and async clients."""

    def _init_core(
        self,
        base_url: str,
        *,
        auth: AuthHeaderInjector | None,
        headers: dict[str, str] | None,
        verify: bool,
        timeout: float,
        max_retries_429: int,
        retry_after_cap_s: float,
        error_map: dict[int, type[Exception]] | None,
        default_error: type[Exception],
        allow_destructive: bool,
        include_response_headers: bool,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.verify = verify
        self.timeout = timeout
        self.max_retries_429 = max_retries_429
        self.retry_after_cap_s = retry_after_cap_s
        self.allow_destructive = allow_destructive
        self.include_response_headers = include_response_headers
        self.error_map: dict[int, type[Exception]] = {
            **DEFAULT_ERROR_MAP,
            **(error_map or {}),
        }
        self.default_error = default_error
        self._auth = auth or AuthHeaderInjector()
        self._extra_headers = dict(headers or {})
        self._rate_limit_capture = RateLimitCapture()
        for secret in self._auth.secrets():
            _log_redactor.add_secret(secret)

    # ------------------------------------------------------------------ #
    # Headers, params, envelopes
    # ------------------------------------------------------------------ #

    @property
    def rate_limit(self) -> RateLimitSnapshot | None:
        """Most recent rate-limit snapshot observed on any response."""
        return self._rate_limit_capture.last

    def default_headers(self) -> dict[str, str]:
        """Headers every request starts from; override per-API as needed."""
        return {"Accept": JSON_CONTENT_TYPE}

    def _merged_headers(
        self,
        headers: dict[str, str] | None,
        content_type: str | None,
        accept: str | None,
    ) -> dict[str, str]:
        merged = {
            **self.default_headers(),
            **self._extra_headers,
            **self._auth.headers(),
            **(headers or {}),
        }
        if content_type:
            merged["Content-Type"] = content_type
        if accept:
            merged["Accept"] = accept
        return merged

    def _merged_params(self, params: dict[str, Any] | None) -> dict[str, Any] | None:
        merged = {
            **{k: v for k, v in (params or {}).items() if v is not None},
            **self._auth.params(),
        }
        return merged or None

    def _envelope(
        self, response: httpx.Response, data: Any, snapshot: RateLimitSnapshot | None
    ) -> dict[str, Any]:
        last = snapshot or self.rate_limit
        envelope: dict[str, Any] = {
            "status_code": response.status_code,
            "data": data,
            "rate_limit": last.to_dict() if last else None,
        }
        if self.include_response_headers:
            envelope["headers"] = dict(response.headers)
        return envelope

    @staticmethod
    def _parse_body(response: httpx.Response) -> Any:
        if response.status_code == 204 or not response.content:
            return None
        content_type = response.headers.get("Content-Type", "")
        if "json" in content_type:
            try:
                return response.json()
            except ValueError:
                return response.text
        return response.text

    # ------------------------------------------------------------------ #
    # Error mapping hook
    # ------------------------------------------------------------------ #

    def _map_error(self, response: httpx.Response, data: Any) -> Exception:
        """Map an error response to an exception (override for bespoke APIs)."""
        detail = ""
        if isinstance(data, dict):
            detail = str(
                data.get("detail")
                or data.get("message")
                or data.get("error")
                or data.get("errorSummary")
                or ""
            )
        message = (
            f"HTTP {response.status_code} for "
            f"{response.request.method} {response.request.url.path}"
        )
        if detail:
            message = f"{message}: {detail}"
        if response.status_code == 429:
            limits = self.rate_limit
            message = (
                f"{message} — rate limited after {self.max_retries_429} retries "
                f"(remaining={limits.remaining if limits else None}, "
                f"reset={limits.reset if limits else None})"
            )
        message = redact_text(message, self._auth.secrets())
        exc_class = self.error_map.get(response.status_code, self.default_error)
        return exc_class(message)

    # ------------------------------------------------------------------ #
    # Destructive gating
    # ------------------------------------------------------------------ #

    def guard_destructive(self, operation: str) -> None:
        """Raise unless the client was built with ``allow_destructive=True``."""
        if not self.allow_destructive:
            raise DestructiveOperationError(
                f"Destructive operation {operation!r} is disabled. "
                "Construct the client with allow_destructive=True to enable it."
            )

    # ------------------------------------------------------------------ #
    # 429 / 401 retry decisions (pure; sleeping is the transport's job)
    # ------------------------------------------------------------------ #

    def _rate_limit_delay(
        self, response: httpx.Response, attempts: int
    ) -> float | None:
        """Backoff delay when the response is a retryable 429, else ``None``."""
        if response.status_code != 429 or attempts >= self.max_retries_429:
            return None
        delay = backoff_seconds(response.headers, cap=self.retry_after_cap_s)
        logger.debug(
            "HTTP 429 from %s; backing off %.2fs (attempt %d/%d)",
            response.request.url.path,
            delay,
            attempts + 1,
            self.max_retries_429,
        )
        return delay

    def _should_refresh_auth(self, response: httpx.Response, refreshed: bool) -> bool:
        """One transparent retry on 401 when the auth strategy can refresh."""
        if response.status_code != 401 or refreshed:
            return False
        invalidate = getattr(self._auth, "invalidate", None)
        if not callable(invalidate):
            return False
        invalidate()
        return True


class BaseApiClient(_ApiClientCore):
    """Synchronous fleet API client base (see module docstring).

    Args:
        base_url: Root URL all relative endpoints join under.
        auth: An :class:`~agent_utilities.http.AuthHeaderInjector` strategy
            (token / basic / query API key / callable provider). ``None``
            means anonymous access.
        headers: Extra default headers merged over :meth:`default_headers`.
        verify: TLS verification (default ``True`` — keep it on).
        timeout: Default request timeout in seconds (finite; per-call
            ``timeout=`` overrides).
        retry: Optional ResiliencePolicy retrying *transport* failures
            (connect errors, resets, timeouts) under the core factory; HTTP
            status handling stays here.
        max_retries_429: Bounded automatic retries on HTTP 429.
        retry_after_cap_s: Ceiling on any single 429 backoff sleep.
        error_map: Status → exception overrides merged over
            :data:`DEFAULT_ERROR_MAP`.
        default_error: Exception class for unmapped error statuses.
        allow_destructive: Enables operations guarded by
            :meth:`guard_destructive`.
        include_response_headers: Adds ``"headers"`` to every envelope.
        transport: httpx transport override (tests use
            ``httpx.MockTransport``).
        **httpx_kwargs: Forwarded to the underlying :class:`httpx.Client`.
    """

    def __init__(
        self,
        base_url: str,
        *,
        auth: AuthHeaderInjector | None = None,
        headers: dict[str, str] | None = None,
        verify: bool = True,
        timeout: float = DEFAULT_HTTP_TIMEOUT_S,
        retry: ResiliencePolicy | None = None,
        max_retries_429: int = DEFAULT_MAX_RETRIES_429,
        retry_after_cap_s: float = DEFAULT_RETRY_AFTER_CAP_S,
        error_map: dict[int, type[Exception]] | None = None,
        default_error: type[Exception] = ApiError,
        allow_destructive: bool = False,
        include_response_headers: bool = False,
        transport: httpx.BaseTransport | None = None,
        **httpx_kwargs: Any,
    ) -> None:
        self._init_core(
            base_url,
            auth=auth,
            headers=headers,
            verify=verify,
            timeout=timeout,
            max_retries_429=max_retries_429,
            retry_after_cap_s=retry_after_cap_s,
            error_map=error_map,
            default_error=default_error,
            allow_destructive=allow_destructive,
            include_response_headers=include_response_headers,
        )
        if transport is not None:
            httpx_kwargs["transport"] = transport
        self._client = create_http_client(
            timeout=timeout,
            verify=verify,
            retry=retry,
            base_url=self.base_url,
            **httpx_kwargs,
        )

    # ------------------------------------------------------------------ #
    # Request engine
    # ------------------------------------------------------------------ #

    def _send(
        self,
        method: str,
        endpoint: str,
        *,
        params: dict[str, Any] | None = None,
        json: Any | None = None,
        data: Any | None = None,
        content: bytes | str | None = None,
        headers: dict[str, str] | None = None,
        content_type: str | None = None,
        accept: str | None = None,
        timeout: float | None = None,
    ) -> httpx.Response:
        """One request with auth injection, rate-limit capture, 429 backoff."""
        merged_params = self._merged_params(params)
        attempts = 0
        refreshed = False
        while True:
            response = self._client.request(
                method,
                endpoint,
                params=merged_params,
                json=json,
                data=data,
                content=content,
                headers=self._merged_headers(headers, content_type, accept),
                timeout=(timeout if timeout is not None else httpx.USE_CLIENT_DEFAULT),
            )
            self._rate_limit_capture.capture(response.headers)
            delay = self._rate_limit_delay(response, attempts)
            if delay is not None:
                attempts += 1
                if delay > 0:
                    time.sleep(delay)
                continue
            if self._should_refresh_auth(response, refreshed):
                refreshed = True
                continue
            return response

    def request(
        self,
        method: str,
        endpoint: str,
        *,
        params: dict[str, Any] | None = None,
        json: Any | None = None,
        data: Any | None = None,
        content: bytes | str | None = None,
        headers: dict[str, str] | None = None,
        content_type: str | None = None,
        accept: str | None = None,
        timeout: float | None = None,
        raise_for_status: bool = True,
    ) -> dict[str, Any]:
        """Send one request; return the uniform response envelope.

        Raises the mapped exception on HTTP >= 400 unless
        ``raise_for_status=False``, in which case the error envelope is
        returned for the caller to inspect.
        """
        response = self._send(
            method,
            endpoint,
            params=params,
            json=json,
            data=data,
            content=content,
            headers=headers,
            content_type=content_type,
            accept=accept,
            timeout=timeout,
        )
        body = self._parse_body(response)
        if raise_for_status and response.status_code >= 400:
            raise self._map_error(response, body)
        snapshot = self._rate_limit_capture.last
        return self._envelope(response, body, snapshot)

    # Convenience verbs ------------------------------------------------- #

    def get(self, endpoint: str, **kwargs: Any) -> dict[str, Any]:
        """GET ``endpoint`` and return the response envelope."""
        return self.request("GET", endpoint, **kwargs)

    def post(self, endpoint: str, **kwargs: Any) -> dict[str, Any]:
        """POST to ``endpoint`` and return the response envelope."""
        return self.request("POST", endpoint, **kwargs)

    def put(self, endpoint: str, **kwargs: Any) -> dict[str, Any]:
        """PUT to ``endpoint`` and return the response envelope."""
        return self.request("PUT", endpoint, **kwargs)

    def patch(self, endpoint: str, **kwargs: Any) -> dict[str, Any]:
        """PATCH ``endpoint`` and return the response envelope."""
        return self.request("PATCH", endpoint, **kwargs)

    def delete(self, endpoint: str, **kwargs: Any) -> dict[str, Any]:
        """DELETE ``endpoint`` and return the response envelope."""
        return self.request("DELETE", endpoint, **kwargs)

    def head(self, endpoint: str, **kwargs: Any) -> dict[str, Any]:
        """HEAD ``endpoint`` and return the response envelope."""
        return self.request("HEAD", endpoint, **kwargs)

    # Pagination --------------------------------------------------------- #

    def paginate(self, endpoint: str, **options: Any) -> PaginationIterator:
        """Iterate a paginated collection (see :class:`PaginationIterator`)."""

        def fetch_page(
            url: str, params: dict[str, Any] | None
        ) -> tuple[Any, httpx.Headers]:
            response = self._send("GET", url, params=params)
            body = self._parse_body(response)
            if response.status_code >= 400:
                raise self._map_error(response, body)
            return body, response.headers

        return PaginationIterator(fetch_page, endpoint, **options)

    # Lifecycle ---------------------------------------------------------- #

    def close(self) -> None:
        """Close the underlying httpx client."""
        self._client.close()

    def __enter__(self) -> BaseApiClient:
        return self

    def __exit__(self, *_exc_info: Any) -> None:
        self.close()


class AsyncBaseApiClient(_ApiClientCore):
    """Asynchronous twin of :class:`BaseApiClient` (same parameters)."""

    def __init__(
        self,
        base_url: str,
        *,
        auth: AuthHeaderInjector | None = None,
        headers: dict[str, str] | None = None,
        verify: bool = True,
        timeout: float = DEFAULT_HTTP_TIMEOUT_S,
        retry: ResiliencePolicy | None = None,
        max_retries_429: int = DEFAULT_MAX_RETRIES_429,
        retry_after_cap_s: float = DEFAULT_RETRY_AFTER_CAP_S,
        error_map: dict[int, type[Exception]] | None = None,
        default_error: type[Exception] = ApiError,
        allow_destructive: bool = False,
        include_response_headers: bool = False,
        transport: httpx.AsyncBaseTransport | None = None,
        **httpx_kwargs: Any,
    ) -> None:
        self._init_core(
            base_url,
            auth=auth,
            headers=headers,
            verify=verify,
            timeout=timeout,
            max_retries_429=max_retries_429,
            retry_after_cap_s=retry_after_cap_s,
            error_map=error_map,
            default_error=default_error,
            allow_destructive=allow_destructive,
            include_response_headers=include_response_headers,
        )
        if transport is not None:
            httpx_kwargs["transport"] = transport
        self._client = create_async_http_client(
            timeout=timeout,
            verify=verify,
            retry=retry,
            base_url=self.base_url,
            **httpx_kwargs,
        )

    async def _send(
        self,
        method: str,
        endpoint: str,
        *,
        params: dict[str, Any] | None = None,
        json: Any | None = None,
        data: Any | None = None,
        content: bytes | str | None = None,
        headers: dict[str, str] | None = None,
        content_type: str | None = None,
        accept: str | None = None,
        timeout: float | None = None,
    ) -> httpx.Response:
        """One request with auth injection, rate-limit capture, 429 backoff."""
        merged_params = self._merged_params(params)
        attempts = 0
        refreshed = False
        while True:
            response = await self._client.request(
                method,
                endpoint,
                params=merged_params,
                json=json,
                data=data,
                content=content,
                headers=self._merged_headers(headers, content_type, accept),
                timeout=(timeout if timeout is not None else httpx.USE_CLIENT_DEFAULT),
            )
            self._rate_limit_capture.capture(response.headers)
            delay = self._rate_limit_delay(response, attempts)
            if delay is not None:
                attempts += 1
                if delay > 0:
                    await asyncio.sleep(delay)
                continue
            if self._should_refresh_auth(response, refreshed):
                refreshed = True
                continue
            return response

    async def request(
        self,
        method: str,
        endpoint: str,
        *,
        params: dict[str, Any] | None = None,
        json: Any | None = None,
        data: Any | None = None,
        content: bytes | str | None = None,
        headers: dict[str, str] | None = None,
        content_type: str | None = None,
        accept: str | None = None,
        timeout: float | None = None,
        raise_for_status: bool = True,
    ) -> dict[str, Any]:
        """Send one request; return the uniform response envelope."""
        response = await self._send(
            method,
            endpoint,
            params=params,
            json=json,
            data=data,
            content=content,
            headers=headers,
            content_type=content_type,
            accept=accept,
            timeout=timeout,
        )
        body = self._parse_body(response)
        if raise_for_status and response.status_code >= 400:
            raise self._map_error(response, body)
        snapshot = self._rate_limit_capture.last
        return self._envelope(response, body, snapshot)

    # Convenience verbs ------------------------------------------------- #

    async def get(self, endpoint: str, **kwargs: Any) -> dict[str, Any]:
        """GET ``endpoint`` and return the response envelope."""
        return await self.request("GET", endpoint, **kwargs)

    async def post(self, endpoint: str, **kwargs: Any) -> dict[str, Any]:
        """POST to ``endpoint`` and return the response envelope."""
        return await self.request("POST", endpoint, **kwargs)

    async def put(self, endpoint: str, **kwargs: Any) -> dict[str, Any]:
        """PUT to ``endpoint`` and return the response envelope."""
        return await self.request("PUT", endpoint, **kwargs)

    async def patch(self, endpoint: str, **kwargs: Any) -> dict[str, Any]:
        """PATCH ``endpoint`` and return the response envelope."""
        return await self.request("PATCH", endpoint, **kwargs)

    async def delete(self, endpoint: str, **kwargs: Any) -> dict[str, Any]:
        """DELETE ``endpoint`` and return the response envelope."""
        return await self.request("DELETE", endpoint, **kwargs)

    async def head(self, endpoint: str, **kwargs: Any) -> dict[str, Any]:
        """HEAD ``endpoint`` and return the response envelope."""
        return await self.request("HEAD", endpoint, **kwargs)

    # Pagination --------------------------------------------------------- #

    def paginate(self, endpoint: str, **options: Any) -> AsyncPaginationIterator:
        """Iterate a paginated collection (``async for`` records)."""

        async def fetch_page(
            url: str, params: dict[str, Any] | None
        ) -> tuple[Any, httpx.Headers]:
            response = await self._send("GET", url, params=params)
            body = self._parse_body(response)
            if response.status_code >= 400:
                raise self._map_error(response, body)
            return body, response.headers

        return AsyncPaginationIterator(fetch_page, endpoint, **options)

    # Lifecycle ---------------------------------------------------------- #

    async def aclose(self) -> None:
        """Close the underlying httpx client."""
        await self._client.aclose()

    async def __aenter__(self) -> AsyncBaseApiClient:
        return self

    async def __aexit__(self, *_exc_info: Any) -> None:
        await self.aclose()
