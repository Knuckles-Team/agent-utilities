"""Typed finance-domain errors.

These replace ``NotImplementedError`` / ``[Mock]`` placeholders so that an
unconfigured data provider surfaces an explicit, actionable failure instead of
either a fake value or a generic "not implemented" (which the no-stub CI gate
flags). Raising one of these is *correct* behaviour, not a stub.
"""

from __future__ import annotations


class FinanceProviderError(RuntimeError):
    """Base class for finance data-provider failures."""


class ProviderNotConfigured(FinanceProviderError):
    """A provider was invoked without the credentials/config it requires.

    The message must name the exact env var / config key the caller should set.
    """


class ProviderRequestError(FinanceProviderError):
    """A configured provider was reached but the request failed."""
