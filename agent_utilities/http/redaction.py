"""Importable log redaction for fleet HTTP clients.

CONCEPT:ECO-4.35 — Fleet HTTP Client Library

Promotes the secret-redaction patterns duplicated across the per-repo
``scripts/security_sanitizer.py`` copies (and okta-agent's ``redact_secrets``)
into one importable module:

* ``Authorization``-style scheme credentials (``Bearer``/``SSWS``/``Basic``/
  ``Token`` followed by token material);
* DSN / URL-embedded credentials (``scheme://user:password@host``);
* well-known token shapes (GitHub/GitLab PATs, AWS access keys, ``sk-`` keys,
  Slack ``xox*`` tokens);
* ``key=value`` / ``"key": "value"`` secret assignments;
* caller-registered literal secret values.

:class:`LogRedactor` is a :class:`logging.Filter` wired as the default filter
for :class:`~agent_utilities.http.BaseApiClient` loggers, so credential
material never reaches log sinks even when callers log raw responses.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable

__all__ = ["REDACTED", "LogRedactor", "redact_text"]

REDACTED = "***REDACTED***"

#: ``Bearer <jwt>`` / ``SSWS <token>`` / ``Basic <b64>`` / ``Token <token>``.
_AUTH_SCHEME_RE = re.compile(r"\b(Bearer|SSWS|Basic|Token)\s+[A-Za-z0-9._~+/=-]{6,}")

#: ``scheme://user:password@host`` DSN credentials (postgres/amqp/redis/...).
_DSN_CREDENTIALS_RE = re.compile(r"\b([a-zA-Z][a-zA-Z0-9+.-]*://[^/\s:@]+):([^@/\s]+)@")

#: Well-known token shapes that are secrets regardless of context.
_TOKEN_SHAPES_RE = re.compile(
    r"\b(?:ghp_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}"
    r"|glpat-[A-Za-z0-9\-]{10,}|AKIA[0-9A-Z]{16}"
    r"|sk-[A-Za-z0-9_\-]{20,}|xox[a-z]-[A-Za-z0-9\-]{10,})"
)

#: ``token=...`` / ``"api_key": "..."`` style assignments. A scheme keyword
#: as the value (``Authorization: Bearer <tok>``) is left to the scheme
#: pattern above so the redacted output keeps its readable structure.
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?P<key>\b(?:api[_-]?key|token|secret|password|passwd|authorization)\b"
    r"['\"]?\s*[:=]\s*['\"]?)"
    r"(?P<value>(?!(?:Bearer|Basic|SSWS|Token)\b)[^\s'\",;&]{6,})",
    re.IGNORECASE,
)


def redact_text(text: str, secrets: Iterable[str] | None = None) -> str:
    """Strip credential material from arbitrary text.

    Args:
        text: The text to redact.
        secrets: Optional literal secret values to replace wherever they
            appear (e.g. the client's live token), in addition to the
            pattern-based redaction.

    Returns:
        The text with every detected credential replaced by ``***REDACTED***``
        (DSN passwords keep their surrounding ``user:...@host`` structure).
    """
    for secret in secrets or ():
        if secret:
            text = text.replace(secret, REDACTED)
    text = _AUTH_SCHEME_RE.sub(rf"\1 {REDACTED}", text)
    text = _DSN_CREDENTIALS_RE.sub(rf"\1:{REDACTED}@", text)
    text = _TOKEN_SHAPES_RE.sub(REDACTED, text)
    text = _SECRET_ASSIGNMENT_RE.sub(lambda m: m.group("key") + REDACTED, text)
    return text


class LogRedactor(logging.Filter):
    """Logging filter that redacts secrets from every record it sees.

    Attach to a logger (``logger.addFilter(LogRedactor())``) to guarantee
    pattern-based redaction; register literal secrets via :meth:`add_secret`
    or a live :meth:`add_secrets_provider` callable (e.g. an auth strategy's
    ``secrets`` method) for values the patterns cannot infer.
    """

    def __init__(
        self,
        secrets: Iterable[str] = (),
        secrets_provider: Callable[[], Iterable[str]] | None = None,
    ) -> None:
        super().__init__()
        self._secrets: set[str] = {s for s in secrets if s}
        self._providers: list[Callable[[], Iterable[str]]] = []
        if secrets_provider is not None:
            self._providers.append(secrets_provider)

    def add_secret(self, secret: str) -> None:
        """Register one literal secret value for redaction."""
        if secret:
            self._secrets.add(secret)

    def add_secrets_provider(self, provider: Callable[[], Iterable[str]]) -> None:
        """Register a callable yielding the current secret values."""
        self._providers.append(provider)

    def _current_secrets(self) -> list[str]:
        secrets = list(self._secrets)
        for provider in self._providers:
            try:
                provided = [s for s in provider() if s]
            except Exception:  # noqa: BLE001 - redaction must never break logging
                provided = []
            secrets.extend(provided)
        return secrets

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        redacted = redact_text(message, self._current_secrets())
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True
