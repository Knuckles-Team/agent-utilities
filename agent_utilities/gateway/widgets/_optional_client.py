"""Shared guarded-import helper for connector widgets.

CONCEPT:AU-OS.config.gateway-service-dashboard — Gateway Service Dashboard

Every widget that talks to a sibling ``agent-packages/agents/*`` connector
package must guard that import: the aggregator refreshes every ~10 seconds,
and an uninstalled sibling package must degrade to a ``skipped``
:class:`~agent_utilities.gateway.models.WidgetData` instead of raising
``ModuleNotFoundError`` on every cycle (mirrors the pre-existing ``ear``
widget's pattern). This one helper replaces the identical
try/import/except-ImportError block that would otherwise be copy-pasted into
every connector widget.
"""

from __future__ import annotations

import importlib
from typing import Any


def import_client(module_path: str, attr: str) -> tuple[Any, None] | tuple[None, str]:
    """Import ``attr`` from ``module_path`` without raising.

    Returns ``(client_class, None)`` on success, or ``(None, package_name)``
    when the sibling package (or the attribute) is unavailable — the caller
    should report a ``status="skipped"`` :class:`WidgetData` using
    ``package_name`` in the message.
    """
    package_name = module_path.split(".")[0].replace("_", "-")
    try:
        module = importlib.import_module(module_path)
        client_cls = getattr(module, attr)
    except (ImportError, AttributeError):
        return None, package_name
    return client_cls, None


def count_items(value: Any, key: str = "results", *, depth: int = 3) -> int:
    """Best-effort item count from a connector's list-shaped response.

    The generated fleet clients return list payloads in one of a few shapes:
    a bare list, a DRF-style ``{"results": [...]}`` envelope, a wrapping
    ``{"data": {...}}`` envelope, or a pydantic model exposing ``key`` as an
    attribute. Recurses through those shapes (bounded by ``depth``) and
    returns ``0`` for anything else rather than raising — this is a display
    count, never the source of a widget's fail-closed decision.
    """
    if depth <= 0:
        return 0
    if isinstance(value, list):
        return len(value)
    if isinstance(value, dict):
        for candidate_key in (key, "data", "results"):
            if candidate_key in value:
                return count_items(value[candidate_key], key, depth=depth - 1)
        return 0
    if hasattr(value, key):
        return count_items(getattr(value, key), key, depth=depth - 1)
    return 0
