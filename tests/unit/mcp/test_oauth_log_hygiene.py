"""Regression tests for the OAuth SDK log-hygiene filter (U-54).

Proves, on real ``logging`` machinery and the exact logger names used by the
locked ``mcp``/``fastmcp`` SDKs, that:

1. Without the filter, both leak shapes found by source inspection really do
   reach a handler (`caplog`) — the "failing-without" baseline.
2. With the filter installed, neither leaks — on the success-path (a state
   value in an ``info`` message) NOR the error-path (a token value reachable
   only via ``exc_info``/traceback rendering, which message-text redaction
   cannot touch).
3. The filter is wired at real import time: importing
   ``agent_utilities.mcp`` — the entry point every MCP consumer in this repo
   already goes through — installs it with no further action, satisfying
   this repo's Wire-First bar (a capability isn't "done" until a live
   entrypoint reaches it).
4. Installation is idempotent (no duplicate filters, no growth on repeat
   import/reload).
"""

from __future__ import annotations

import importlib
import logging

import pytest

from agent_utilities.mcp.oauth_log_hygiene import (
    _HYGIENE_LOGGER_NAMES,
    OAuthLogHygieneFilter,
    install_oauth_log_hygiene,
)

# Synthetic secret-shaped values — never real credentials, but distinctive
# enough that any substring match in captured log text proves a leak.
_SECRET_STATE = "state-9f8a7b6c5d4e3f2a1b0c-CSRF"  # sanitizer:ignore - synthetic fixture, not a real credential
_SECRET_TOKEN = "sk-live-abcdef0123456789-DO-NOT-LOG"  # noqa: S105 - synthetic test fixture, not a real credential  # sanitizer:ignore - synthetic fixture, not a real credential


@pytest.fixture(autouse=True)
def clean_oauth_loggers():
    """Snapshot + restore filters on every hygiene-target logger.

    Real ``logging.Logger`` objects are process-global and cached by name, so
    without this, whichever test (or import) runs first would leave the
    filter attached (or absent) for every test after it. Each test gets a
    guaranteed-clean slate regardless of import order across the suite.
    """
    snapshots: dict[str, list[logging.Filter]] = {}
    for name in _HYGIENE_LOGGER_NAMES:
        target = logging.getLogger(name)
        snapshots[name] = list(target.filters)
        target.filters = []
    yield
    for name, filters in snapshots.items():
        logging.getLogger(name).filters = filters


def _raise_and_log_exception(logger: logging.Logger, message: str) -> None:
    try:
        raise ValueError(f"invalid token response: access_token={_SECRET_TOKEN}")
    except ValueError:
        logger.exception(message)


class TestLeaksAreRealWithoutTheFilter:
    """Failing-without baseline for both leak shapes found by source inspection."""

    def test_authorization_url_state_leaks_via_info_message(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        logger = logging.getLogger("fastmcp.client.auth.oauth")
        with caplog.at_level(logging.INFO, logger="fastmcp.client.auth.oauth"):
            logger.info(
                "OAuth authorization URL: https://idp.example.test/authorize?state=%s",
                _SECRET_STATE,
            )
        assert _SECRET_STATE in caplog.text

    def test_token_response_leaks_via_exception_traceback(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        logger = logging.getLogger("mcp.client.auth.oauth2")
        with caplog.at_level(logging.ERROR, logger="mcp.client.auth.oauth2"):
            _raise_and_log_exception(logger, "Invalid refresh response")
        # The message text alone is secret-free ("Invalid refresh response");
        # the leak lives entirely in the rendered exception/traceback.
        assert _SECRET_TOKEN in caplog.text
        assert caplog.records[0].exc_info is not None


class TestFilterRedactsBothLeakShapes:
    def test_authorization_url_state_is_redacted(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        install_oauth_log_hygiene()
        logger = logging.getLogger("fastmcp.client.auth.oauth")
        with caplog.at_level(logging.INFO, logger="fastmcp.client.auth.oauth"):
            logger.info(
                "OAuth authorization URL: https://idp.example.test/authorize?state=%s",
                _SECRET_STATE,
            )
        assert _SECRET_STATE not in caplog.text
        assert "redacted" in caplog.text.lower()

    def test_token_response_is_redacted_including_traceback(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        install_oauth_log_hygiene()
        logger = logging.getLogger("mcp.client.auth.oauth2")
        with caplog.at_level(logging.ERROR, logger="mcp.client.auth.oauth2"):
            _raise_and_log_exception(logger, "Invalid refresh response")
        assert _SECRET_TOKEN not in caplog.text
        for record in caplog.records:
            assert record.exc_info is None
            assert record.exc_text is None
            assert record.stack_info is None

    def test_client_credentials_logger_is_also_covered(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        install_oauth_log_hygiene()
        logger = logging.getLogger("fastmcp.client.auth.client_credentials")
        with caplog.at_level(
            logging.INFO, logger="fastmcp.client.auth.client_credentials"
        ):
            logger.info("minted client-credentials token=%s", _SECRET_TOKEN)
        assert _SECRET_TOKEN not in caplog.text


class TestInstallationIsIdempotent:
    def test_repeat_install_does_not_duplicate_filters(self) -> None:
        install_oauth_log_hygiene()
        install_oauth_log_hygiene()
        install_oauth_log_hygiene()
        for name in _HYGIENE_LOGGER_NAMES:
            target = logging.getLogger(name)
            hygiene_filters = [
                f for f in target.filters if isinstance(f, OAuthLogHygieneFilter)
            ]
            assert len(hygiene_filters) == 1


class TestWiredAtPackageImport:
    """Wire-First: the capability must be reached from the real entrypoint
    (importing `agent_utilities.mcp`), not merely importable and unit-tested."""

    def test_importing_the_mcp_package_installs_hygiene(self) -> None:
        import agent_utilities.mcp as mcp_package

        importlib.reload(mcp_package)

        for name in _HYGIENE_LOGGER_NAMES:
            target = logging.getLogger(name)
            assert any(isinstance(f, OAuthLogHygieneFilter) for f in target.filters), (
                f"importing agent_utilities.mcp did not install log hygiene on {name!r}"
            )
