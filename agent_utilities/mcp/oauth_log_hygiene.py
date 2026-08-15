#!/usr/bin/python
from __future__ import annotations

"""OAuth log-hygiene filter for the vendored MCP SDK's OAuth modules (U-54).

CONCEPT:AU-ECO.mcp.oauth-log-hygiene

The MCP SDK's browser-OAuth client code (``mcp.client.auth.oauth2``,
``fastmcp.client.auth.oauth``) logs at INFO/DEBUG/exception level, tuned for
local, single-user CLI use. Two concrete leaks were confirmed by source
inspection at this repo's locked SDK versions (``mcp`` 2.0.0, ``fastmcp``
4.0.0b1, matching ``pyproject.toml``'s ``[mcp]`` extra):

* ``fastmcp/client/auth/oauth.py:399`` — ``logger.info(f"OAuth authorization
  URL: {authorization_url}")`` on the SUCCESS path logs the complete
  authorization-redirect URL, whose query string carries the CSRF ``state``
  parameter and PKCE ``code_challenge``.
* ``mcp/client/auth/oauth2.py:546,749,780`` — ``logger.exception(...)`` on
  error paths (an invalid token-refresh response, or any exception during the
  authorization-code exchange) sets ``exc_info=True``. The rendered traceback
  can carry the raw token-endpoint response body (``content``/``token_response``
  are still local variables in the frame that raised) and, under a
  locals-aware handler, the token/state values themselves as displayed local
  variables — the exact "token-response detail" and "state" exposure this
  module is named for.

Neither leak is a message-text *pattern* :class:`agent_utilities.httpsupport.
redaction.LogRedactor` can catch: it only regex-scrubs ``record.getMessage()``
and never touches ``record.exc_info``/``exc_text``/``stack_info`` — exactly
where the second leak lives. AGENTS.md's "Secrets & credential retrieval"
policy is unconditional ("Never echo a secret value into logs..."); a
centralized multi-user service log is a far larger blast radius than the
local CLI these SDK loggers were designed for, so this module does not try to
detect-and-scrub — it blanks every field on any record these loggers emit.

:func:`install_oauth_log_hygiene` attaches the filter to each SDK OAuth
logger **by name only** (``logging.getLogger("...")``); it does not import
``mcp`` or ``fastmcp``, so it is safe and cheap to call unconditionally from
:mod:`agent_utilities.mcp`'s package ``__init__`` — before any MCP transport,
and regardless of whether the ``[mcp]`` extra is even installed.
"""

import logging

__all__ = ["install_oauth_log_hygiene", "OAuthLogHygieneFilter"]

#: Constant replacement message. Deliberately generic — carries no request
#: identifier, endpoint, or outcome detail that could itself narrow down a
#: correlated secret elsewhere in the log stream.
_REDACTED_OAUTH_EVENT = (
    "[oauth] SDK log event redacted for credential hygiene (see U-54)"
)

#: Every ``logging.getLogger(__name__)`` name found, by source inspection, in
#: the locked MCP SDK's browser-OAuth code paths. Add to this tuple (never
#: remove without re-auditing) if the locked ``mcp``/``fastmcp`` version adds
#: another OAuth-flow logger.
_HYGIENE_LOGGER_NAMES: tuple[str, ...] = (
    "mcp.client.auth.oauth2",
    "fastmcp.client.auth.oauth",
    "fastmcp.client.auth.client_credentials",
)


class OAuthLogHygieneFilter(logging.Filter):
    """Blank every field of an OAuth SDK log record before it reaches a handler.

    Unlike a redaction filter that pattern-matches known secret shapes, this
    filter assumes *any* field on a record from one of these loggers may
    carry credential material (message, positional interpolation args, or
    the exception/stack info an ``exception()``/``exc_info=True`` call
    attaches) and unconditionally clears all of them. Runs in
    ``Logger.filter()``, before ``callHandlers`` — so it protects every
    handler on the propagation chain (including a root handler this process
    did not configure), not just one attached sink.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = _REDACTED_OAUTH_EVENT
        record.args = ()
        record.exc_info = None
        record.exc_text = None
        record.stack_info = None
        return True


def install_oauth_log_hygiene() -> None:
    """Idempotently attach :class:`OAuthLogHygieneFilter` to every known SDK OAuth logger.

    Safe to call more than once (e.g. on module reload in tests): each
    logger is checked for an existing instance of the filter before adding
    another, so records are never double-processed and repeated calls never
    grow the filter list.
    """
    for name in _HYGIENE_LOGGER_NAMES:
        target = logging.getLogger(name)
        if not any(
            isinstance(existing, OAuthLogHygieneFilter) for existing in target.filters
        ):
            target.addFilter(OAuthLogHygieneFilter())
