"""Tests for agent_utilities.http.rate_limit (CONCEPT:AU-ECO.ui.fleet-http-client-library).

Pins both header families (X-RateLimit-* and X-Rate-Limit-*), Retry-After
parsing (seconds + HTTP-date), bounded backoff (okta/dockerhub semantics),
and the stateful RateLimitCapture.
"""

from __future__ import annotations

from email.utils import format_datetime
from datetime import datetime, timedelta, timezone

from agent_utilities.http.rate_limit import (
    RateLimitCapture,
    RateLimitSnapshot,
    backoff_seconds,
    parse_rate_limit,
)

# --------------------------------------------------------------------------- #
# parse_rate_limit
# --------------------------------------------------------------------------- #


def test_parses_x_ratelimit_family():
    snapshot = parse_rate_limit(
        {
            "X-RateLimit-Limit": "100",
            "X-RateLimit-Remaining": "42",
            "X-RateLimit-Reset": "30",
        }
    )
    assert snapshot == RateLimitSnapshot(limit=100, remaining=42, reset=30.0)
    assert not snapshot.exhausted


def test_parses_okta_x_rate_limit_family_case_insensitively():
    snapshot = parse_rate_limit(
        {
            "x-rate-limit-limit": "600",
            "X-Rate-Limit-Remaining": "0",
            "X-Rate-Limit-Reset": "1750000000",
        }
    )
    assert snapshot is not None
    assert snapshot.limit == 600
    assert snapshot.exhausted
    assert snapshot.reset == 1750000000.0


def test_parses_retry_after_seconds_and_http_date():
    assert parse_rate_limit({"Retry-After": "7"}).retry_after_s == 7.0
    when = datetime.now(timezone.utc) + timedelta(seconds=60)
    snapshot = parse_rate_limit({"Retry-After": format_datetime(when)})
    assert snapshot is not None
    assert 55.0 <= snapshot.retry_after_s <= 61.0


def test_no_rate_limit_headers_returns_none():
    assert parse_rate_limit({"Content-Type": "application/json"}) is None


def test_unparseable_values_are_skipped():
    snapshot = parse_rate_limit(
        {"X-RateLimit-Limit": "soon", "X-RateLimit-Remaining": "3"}
    )
    assert snapshot == RateLimitSnapshot(limit=None, remaining=3)


def test_snapshot_to_dict_omits_missing_fields():
    assert RateLimitSnapshot(limit=10, remaining=2).to_dict() == {
        "limit": 10,
        "remaining": 2,
    }


# --------------------------------------------------------------------------- #
# backoff_seconds
# --------------------------------------------------------------------------- #


def test_backoff_prefers_retry_after_and_caps_it():
    assert backoff_seconds({"Retry-After": "3"}, cap=15.0) == 3.0
    assert backoff_seconds({"Retry-After": "9999"}, cap=15.0) == 15.0
    assert backoff_seconds({"Retry-After": "-5"}, cap=15.0) == 0.0


def test_backoff_uses_epoch_reset_with_floor_and_cap():
    now = 1_750_000_000.0
    headers = {"X-Rate-Limit-Reset": str(int(now) + 8)}
    assert backoff_seconds(headers, cap=60.0, now=now) == 8.0
    # Skewed clock (reset in the past) can never produce a zero/negative wait.
    past = {"X-Rate-Limit-Reset": str(int(now) - 100)}
    assert backoff_seconds(past, cap=60.0, now=now) == 1.0
    # Distant reset is capped.
    far = {"X-Rate-Limit-Reset": str(int(now) + 9000)}
    assert backoff_seconds(far, cap=60.0, now=now) == 60.0


def test_backoff_treats_small_reset_as_duration():
    assert backoff_seconds({"X-RateLimit-Reset": "5"}, cap=60.0, now=0.0) == 5.0


def test_backoff_default_when_nothing_parseable():
    assert backoff_seconds({}, cap=15.0) == 1.0
    assert backoff_seconds({"Retry-After": "not-a-date"}, cap=15.0) == 1.0


# --------------------------------------------------------------------------- #
# RateLimitCapture
# --------------------------------------------------------------------------- #


def test_capture_keeps_last_snapshot_across_empty_responses():
    capture = RateLimitCapture()
    assert capture.capture({"X-RateLimit-Remaining": "9"}) is not None
    assert capture.capture({"Content-Type": "text/html"}) is None
    assert capture.last is not None
    assert capture.last.remaining == 9
