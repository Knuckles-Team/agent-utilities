"""Rate-limit telemetry for fleet HTTP clients.

CONCEPT:ECO-4.35 — Fleet HTTP Client Library

Parses the two rate-limit header families used across the fleet's upstream
APIs into one typed snapshot:

* ``X-RateLimit-Limit`` / ``X-RateLimit-Remaining`` / ``X-RateLimit-Reset``
  (Docker Hub, GitHub, most REST APIs);
* ``X-Rate-Limit-Limit`` / ``X-Rate-Limit-Remaining`` / ``X-Rate-Limit-Reset``
  (Okta — reset is an epoch timestamp);
* ``Retry-After`` (seconds or HTTP-date) for 429/503 backoff.

:func:`backoff_seconds` derives a *bounded* wait for HTTP 429 responses —
``Retry-After`` wins when present, otherwise an epoch ``*-Reset`` header is
used (clamped so a skewed clock or hostile server can never stall a caller),
matching the okta-agent and dockerhub-api semantics the fleet converged on.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

__all__ = [
    "DEFAULT_RETRY_AFTER_CAP_S",
    "RateLimitCapture",
    "RateLimitSnapshot",
    "backoff_seconds",
    "parse_rate_limit",
]

#: Hard ceiling (seconds) on a single 429 backoff sleep so a hostile or
#: misconfigured server can never stall a caller indefinitely.
DEFAULT_RETRY_AFTER_CAP_S = 15.0

#: Values above this are treated as epoch timestamps rather than durations.
_EPOCH_THRESHOLD = 1e8


@dataclass(frozen=True)
class RateLimitSnapshot:
    """Typed snapshot of the rate-limit headers on one response.

    Attributes:
        limit: The request quota for the current window, if advertised.
        remaining: Requests left in the current window, if advertised.
        reset: The window reset marker — an epoch timestamp (Okta, GitHub)
            or a duration in seconds, exactly as the server sent it.
        retry_after_s: Parsed ``Retry-After`` value in seconds, if present.
    """

    limit: int | None = None
    remaining: int | None = None
    reset: float | None = None
    retry_after_s: float | None = None

    @property
    def exhausted(self) -> bool:
        """Whether the advertised quota has been fully consumed."""
        return self.remaining == 0

    def to_dict(self) -> dict[str, Any]:
        """Compact dict form for response envelopes (``None`` fields omitted)."""
        return {
            key: value
            for key, value in (
                ("limit", self.limit),
                ("remaining", self.remaining),
                ("reset", self.reset),
                ("retry_after_s", self.retry_after_s),
            )
            if value is not None
        }


def _lower_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {str(k).lower(): str(v) for k, v in headers.items()}


def _to_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _retry_after_to_seconds(raw: str) -> float | None:
    """Parse ``Retry-After`` — delta-seconds or an HTTP-date (RFC 9110)."""
    try:
        return float(raw)
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return (when - datetime.now(UTC)).total_seconds()


def parse_rate_limit(headers: Mapping[str, str]) -> RateLimitSnapshot | None:
    """Extract a :class:`RateLimitSnapshot` from response headers.

    Returns ``None`` when no rate-limit telemetry is present, so callers can
    keep their last known snapshot instead of clobbering it.
    """
    lowered = _lower_headers(headers)

    def first(*names: str) -> str | None:
        for name in names:
            value = lowered.get(name)
            if value is not None:
                return value
        return None

    limit = _to_int(first("x-ratelimit-limit", "x-rate-limit-limit"))
    remaining = _to_int(first("x-ratelimit-remaining", "x-rate-limit-remaining"))
    reset = _to_float(first("x-ratelimit-reset", "x-rate-limit-reset"))
    retry_after_raw = lowered.get("retry-after")
    retry_after = (
        _retry_after_to_seconds(retry_after_raw)
        if retry_after_raw is not None
        else None
    )

    if limit is None and remaining is None and reset is None and retry_after is None:
        return None
    return RateLimitSnapshot(
        limit=limit, remaining=remaining, reset=reset, retry_after_s=retry_after
    )


def backoff_seconds(
    headers: Mapping[str, str],
    *,
    cap: float = DEFAULT_RETRY_AFTER_CAP_S,
    now: float | None = None,
    default: float = 1.0,
) -> float:
    """Bounded wait (seconds) before retrying a rate-limited request.

    Precedence: ``Retry-After`` (clamped to ``[0, cap]``), then an
    ``X-Rate[-]Limit-Reset`` header — epoch timestamps become ``reset - now``
    while small values are taken as durations, clamped to ``[1, cap]`` so a
    skewed clock can never produce a zero or unbounded wait (Okta semantics).
    Falls back to ``default`` when nothing is parseable.
    """
    snapshot = parse_rate_limit(headers)
    if snapshot is not None and snapshot.retry_after_s is not None:
        return max(0.0, min(snapshot.retry_after_s, cap))
    if snapshot is not None and snapshot.reset is not None:
        reference = time.time() if now is None else now
        wait = (
            snapshot.reset - reference
            if snapshot.reset > _EPOCH_THRESHOLD
            else snapshot.reset
        )
        return max(1.0, min(wait, cap))
    return max(0.0, min(default, cap))


class RateLimitCapture:
    """Stateful capture of the most recent rate-limit snapshot.

    One instance lives on each :class:`~agent_utilities.http.BaseApiClient`;
    :meth:`capture` is called on every response so the latest telemetry is
    always available (``client.rate_limit``) and attached to every envelope.
    """

    def __init__(self) -> None:
        self.last: RateLimitSnapshot | None = None

    def capture(self, headers: Mapping[str, str]) -> RateLimitSnapshot | None:
        """Parse ``headers``; remember and return the snapshot if present."""
        snapshot = parse_rate_limit(headers)
        if snapshot is not None:
            self.last = snapshot
        return snapshot
