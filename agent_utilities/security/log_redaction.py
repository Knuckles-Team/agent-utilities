#!/usr/bin/python
from __future__ import annotations

"""Redact runtime locations/endpoints/caller identifiers before they reach a log call.

CONCEPT:AU-OS.observability.log-location-privacy — a raw filesystem path, network
endpoint, or caller identifier in a log line leaks host/tenant topology to anyone
with log read access (a log-aggregation operator, a compromised sidecar, a support
bundle). ``tests/unit/security/test_log_location_privacy_static_gate.py`` statically
walks every ``agent_utilities`` module for a log call (``logger.info``/``warning``/
etc.) that passes a variable/attribute NAMED like one of these (``path``,
``endpoint``, ``host``, ``file_path``, ...) and fails the build if it finds one —
the AST match is a proxy for "a sensitive runtime value flows into a log record",
so the fix is to stop logging the raw value, not to rename the variable.

:func:`redact_for_log` is the one shared, deterministic redaction every log call
site uses instead: the same input always produces the same short tag within a
process, so operators can still correlate repeated log lines about the same
path/endpoint/caller across a session — without the log itself ever carrying the
literal value.
"""

import hashlib

__all__ = ["redact_for_log"]


def redact_for_log(value: object) -> str:
    """Return a short, deterministic, non-reversible tag for a sensitive value.

    ``None``/empty values redact to ``"<empty>"``. Anything else is rendered as
    ``str(value)`` and reduced to a 12-hex-character SHA-256 prefix — enough to
    tell two DIFFERENT values apart in a log stream without exposing either one.
    """
    if value is None or value == "":
        return "<empty>"
    digest = hashlib.sha256(str(value).encode("utf-8", errors="replace")).hexdigest()
    return f"<redacted:{digest[:12]}>"
