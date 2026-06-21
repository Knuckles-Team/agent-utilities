#!/usr/bin/python
from __future__ import annotations

"""Per-connector coverage + freshness assessment (CONCEPT:OS-5.48).

The connector analogue of the codebase coverage check (OS-5.47): the world-model
is only trustworthy if EVERY configured connector is actually ingesting and fresh —
otherwise the agent silently falls back to hitting the source system. Compares the
**expected** connectors (the delta-handler set + the ``mcp_tool`` presets) against
which ones have a recorded ``DeltaManifest`` watermark and how stale it is, so a
dark/stale connector surfaces as a doctor warning instead of a silent gap.

Pure functions (no engine) so they unit-test directly; the doctor check wires them
to the live ``DeltaManifest``.
"""

from datetime import UTC, datetime
from typing import Any

DEFAULT_SLA_DAYS = 7
#: The manifest category connectors record their incremental checkpoint under.
CONNECTOR_CATEGORY = "connector_checkpoint"


def enumerate_expected_connectors() -> list[str]:
    """The connectors the platform is configured to ingest from (best-effort)."""
    expected: set[str] = set()
    try:
        from agent_utilities.knowledge_graph.core.source_sync import _DELTA_HANDLERS

        expected |= {k for k in _DELTA_HANDLERS if k != "fleet"}
    except Exception:  # pragma: no cover - import best-effort
        pass
    try:
        from agent_utilities.protocols.source_connectors.connectors.mcp_tool import (
            list_presets,
        )

        expected |= set(list_presets())
    except Exception:  # pragma: no cover
        pass
    return sorted(expected)


def _age_days(updated_at: str, now: datetime) -> float | None:
    if not updated_at:
        return None
    try:
        ts = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        return (now - ts).total_seconds() / 86400.0
    except (ValueError, TypeError):
        return None


def assess_connector_coverage(
    expected: list[str],
    freshness: dict[str, str],
    *,
    sla_days: int = DEFAULT_SLA_DAYS,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Compare expected connectors vs their last-sync watermarks.

    ``freshness`` maps a manifest ``source_uri`` (the connector id) → ISO
    ``updated_at``. A connector is *covered* if any watermark key contains its name,
    and *stale* if its newest matching watermark is older than ``sla_days``.
    """
    now = now or datetime.now(UTC)
    covered: list[str] = []
    missing: list[str] = []
    stale: list[dict[str, Any]] = []
    for conn in expected:
        ages = [
            age
            for uri, ts in freshness.items()
            if conn in uri
            for age in [_age_days(ts, now)]
            if age is not None
        ]
        if not any(conn in uri for uri in freshness):
            missing.append(conn)
            continue
        covered.append(conn)
        if ages and min(ages) > sla_days:
            stale.append({"connector": conn, "age_days": round(min(ages), 1)})

    total = len(expected)
    return {
        "total": total,
        "covered": len(covered),
        "missing": missing,
        "stale": stale,
        "coverage_pct": round(100.0 * len(covered) / total, 1) if total else 0.0,
        "sla_days": sla_days,
    }
