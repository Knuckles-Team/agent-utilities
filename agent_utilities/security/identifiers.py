#!/usr/bin/python
from __future__ import annotations

"""Centralized identifier validation for interpolated Cypher/SQL/DDL fragments.

Neither Cypher nor SQL/DDL syntax accepts a bound parameter for an
*identifier* position (a node label, relationship type, property key, table,
column, schema, or extension name) — only literal *values* bind. So every
mirror-backend adapter and pipeline module that builds a label-, table-, or
column-scoped query must interpolate that name directly into the query text.
Left unchecked, a caller/ontology/connector-supplied name crafted as
``x; DROP TABLE--`` or containing backtick/quote-smuggling characters can
break out of the intended query shape even though every *value* in the same
query is safely bound.

The correct pattern already existed once —
``agent_utilities.knowledge_graph.backends.epistemic_graph_backend``'s
``_CYPHER_IDENTIFIER_RE`` (used by its ``prune()``) — but was never shared, so
several other backends grew their own copy (or none at all). This module is
the ONE place that pattern lives; every adapter that interpolates an
identifier calls into it instead of hand-rolling its own regex.

Two call shapes:

* :func:`validate_identifier` — the general-purpose Cypher/graph identifier
  gate (labels, relationship types, property/variable names). Allows up to
  128 characters, matching the engine's own identifier ceiling.
* :func:`validate_sql_identifier` / :func:`quote_sql_identifier` — the
  SQL-specific, quoting-aware variants: validated against PostgreSQL's
  63-byte ``NAMEDATALEN`` identifier ceiling (stricter than the Cypher
  variant on purpose — a name that silently truncates past 63 bytes can
  collide with a different identifier). :func:`quote_sql_identifier` returns
  a ready-to-embed, double-quoted string so a call site never hand-rolls
  quoting.

Both raise :class:`InvalidIdentifierError` — a typed, non-echoing error (the
offending value is never included in the message, mirroring
``CypherEngineError``'s no-leak discipline) so a crafted identifier cannot
also be used to smuggle content into logs or error responses.
"""

import re

__all__ = [
    "CYPHER_IDENTIFIER_RE",
    "SQL_IDENTIFIER_RE",
    "InvalidIdentifierError",
    "validate_identifier",
    "validate_sql_identifier",
    "quote_sql_identifier",
]

# Generic identifier grammar for Cypher/graph-shaped names: an ASCII
# letter/underscore start, then up to 127 more ASCII word characters (128
# total) — the engine's own identifier ceiling. Deliberately ASCII-only (not
# Python's Unicode-aware ``\w``) so a homoglyph/lookalike identifier is
# rejected rather than silently accepted as "valid".
CYPHER_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")

# PostgreSQL identifiers are silently truncated at NAMEDATALEN-1 = 63 bytes
# (a too-long name doesn't error — it collides with another truncated name),
# so the SQL variant is intentionally stricter than the generic one.
SQL_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")


class InvalidIdentifierError(ValueError):
    """A caller/ontology/connector-supplied identifier failed validation.

    A subclass of ``ValueError`` (so existing ``except ValueError`` call
    sites keep working unmodified). The offending value is intentionally
    never included in the message or any attribute — only the ``kind`` label
    the caller supplied for its own logging — so a crafted identifier cannot
    be smuggled into a log line or error response via this exception.
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind
        super().__init__(f"invalid {kind} identifier")


def validate_identifier(name: object, kind: str = "label") -> str:
    """Validate a Cypher/graph identifier before f-string interpolation.

    Returns ``name`` (coerced to ``str``) unchanged when it matches
    :data:`CYPHER_IDENTIFIER_RE`; raises :class:`InvalidIdentifierError`
    otherwise. Use for node labels, relationship types, property keys, and
    query variable names — anywhere a Cypher-shaped backend interpolates a
    name directly into query text.

    Args:
        name: The candidate identifier (any object; coerced via ``str()``).
        kind: Free text describing what is being validated (``"label"``,
            ``"relationship type"``, ``"property"``, …), used only in the
            raised error's message — never in a way that echoes ``name``.
    """
    rendered = str(name) if name is not None else ""
    if not CYPHER_IDENTIFIER_RE.fullmatch(rendered):
        raise InvalidIdentifierError(kind)
    return rendered


def validate_sql_identifier(name: object, kind: str = "table") -> str:
    """Validate a SQL identifier (table/column/schema/extension) for direct use.

    Enforces PostgreSQL's 63-byte identifier ceiling (stricter than
    :func:`validate_identifier`'s 128-char Cypher grammar) and returns the
    bare, validated name. Callers that need a ready-to-embed quoted string
    should use :func:`quote_sql_identifier` instead.
    """
    rendered = str(name) if name is not None else ""
    if not SQL_IDENTIFIER_RE.fullmatch(rendered):
        raise InvalidIdentifierError(kind)
    return rendered


def quote_sql_identifier(name: object, kind: str = "table") -> str:
    """Validate ``name`` and return a double-quoted identifier ready to embed.

    :data:`SQL_IDENTIFIER_RE` already excludes quote/backslash characters, so
    the escape below is defense-in-depth rather than load-bearing — but this
    gives call sites one function that both validates and quotes, so no
    adapter hand-rolls its own quoting.
    """
    validated = validate_sql_identifier(name, kind=kind)
    return '"' + validated.replace('"', '""') + '"'
