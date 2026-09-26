"""Process-local readiness report for AU's bundled skills."""

from __future__ import annotations

from typing import Any

_REPORT: dict[str, Any] = {}


def set_bundled_skill_readiness(report: dict[str, Any]) -> None:
    """Publish the latest bootstrap result without exposing the mutable source."""
    _REPORT.clear()
    _REPORT.update(report)


def bundled_skill_readiness() -> dict[str, Any]:
    """Return the latest report, or an empty dict before bootstrap runs."""
    return dict(_REPORT)
