"""Dependency-free environment accessor (`setting`).

CONCEPT: config discipline. This module holds the one sanctioned way to read an
environment variable outside ``core/config.py``/``core/paths.py``. It lives in its
own module — with **no top-level ``agent_utilities`` imports** — so it is
importable even while ``core.config`` is still initializing. That breaks the
circular-import deadlock that would otherwise arise from ~100 modules doing
``from agent_utilities.core.config import setting`` while ``config`` itself (and
its import chain) is mid-load. ``config`` re-exports ``setting`` from here.

All heavier dependencies (``_ensure_env_loaded`` in ``config``; the ``to_*``
coercers in ``base_utilities``) are imported **lazily at call time**, by which
point those modules are fully loaded.
"""

from __future__ import annotations

import os
from typing import Any

# Sentinel distinguishing "no default given" from an explicit ``default=None``.
_UNSET = object()


def setting(key: str, default: Any = _UNSET, cast: Any = None) -> Any:
    """Centralized, typed, **live** environment read.

    *Configuration discipline (READ the rule in AGENTS.md).* Modules must NEVER
    call ``os.environ.get``/``os.getenv`` directly; route every read through this
    accessor (or a typed ``AgentConfig`` field for static, schema-worthy infra).
    Enforced by ``scripts/check_no_env_sprawl.py``.

    Unlike an ``AgentConfig`` field (parsed once at import), this reads
    ``os.environ`` at call time, so values injected from ``config.json`` by
    ``_load_xdg_json_config()`` apply, and runtime/daemon/test changes
    (``monkeypatch.setenv``) take effect — which is why call-time daemon tunables
    and test-varied flags use this, not a frozen field.

    Args:
        key: The environment variable name (e.g. ``"KG_DAEMON_ROLE"``).
        default: Value when unset/empty. If omitted, an unset var returns ``None``.
        cast: A coercion callable. If ``None`` it is inferred from ``type(default)``:
            ``bool`` → :func:`to_boolean`, ``int`` → ``int``, ``float`` → ``float``,
            ``list`` → :func:`to_list`, ``dict`` → :func:`to_dict`, else ``str``
            (the raw string). Pass an explicit callable to override.

    Returns:
        The coerced value, or ``default`` when the variable is unset/empty.
    """
    from agent_utilities.core.config import _ensure_env_loaded

    _ensure_env_loaded()
    dflt = None if default is _UNSET else default
    raw = os.environ.get(key)
    if raw is None or raw == "":
        return dflt
    if cast is None:
        from agent_utilities.base_utilities import to_boolean, to_dict, to_list

        if isinstance(dflt, bool):
            cast = to_boolean
        elif isinstance(dflt, int):
            cast = int
        elif isinstance(dflt, float):
            cast = float
        elif isinstance(dflt, list):
            cast = to_list
        elif isinstance(dflt, dict):
            cast = to_dict
        else:
            cast = str
    try:
        return cast(raw)
    except (TypeError, ValueError):
        return dflt
