"""Outbound-text unwrap for ``run_agent``'s rich JSON envelope.

CONCEPT:AU-ORCH.execution.messaging-orchestration-transparency — ``run_agent``
returns a JSON envelope string when a caller opts into ``run_summary``,
``channel_id`` or ``mermaid``. A chat surface shows only the ``output`` field.
``run_summary`` stays available to the caller for logging and trace links.

The envelope key set lives here, next to the renderer's contract, so the
renderer and every unwrapping caller share one definition.
"""

from __future__ import annotations

import json
from typing import Any

RUN_ENVELOPE_KEYS: frozenset[str] = frozenset(
    {
        "output",
        "run_id",
        "channel_id",
        "mermaid",
        "run_summary",
        "execution_evidence",
        "provenance_recorded",
    }
)


def _as_envelope(value: Any) -> dict[str, Any] | None:
    """Return ``value`` as an envelope dict, or ``None`` when it is not one."""
    if isinstance(value, dict):
        env: Any = value
    elif isinstance(value, str):
        text = value.strip()
        if not (text.startswith("{") and '"output"' in text):
            return None
        try:
            env = json.loads(text)
        except (ValueError, TypeError):
            return None
    else:
        return None
    if isinstance(env, dict) and "output" in env and set(env) <= RUN_ENVELOPE_KEYS:
        return env
    return None


def unwrap_run_envelope(value: Any) -> tuple[str, dict[str, Any] | None]:
    """Split a run result into user-facing text and its ``run_summary``.

    ``value`` is a dict or JSON string envelope, or a bare reply. The membership
    check is exact, so a genuine JSON reply from the agent passes through
    unchanged. Returns ``(text, run_summary)``.
    """
    env = _as_envelope(value)
    if env is None:
        return ("" if value is None else str(value)), None
    summary = env.get("run_summary")
    return str(env["output"]).strip(), summary if isinstance(summary, dict) else None
