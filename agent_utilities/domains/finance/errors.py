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


class InvalidIntervalError(ValueError, FinanceProviderError):
    """A gap-fill/ASOF interval was empty, malformed, zero, or negative.

    Raised at the API boundary (``engine_series._normalize_step``) before either
    the engine or the pandas fallback route does any work, so a caller gets one
    documented, typed error instead of a downstream ``ZeroDivisionError``
    (zero-frequency grid) or an opaque pandas parser ``ValueError`` (D-CDX-96).
    Subclasses ``ValueError`` for drop-in compatibility with callers that
    already catch the historical malformed-string behaviour.
    """
