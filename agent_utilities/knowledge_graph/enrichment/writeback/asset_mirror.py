"""Multi-SoR asset-mirror application (CONCEPT:AU-KG.ingest.enterprise-source-extractor).

One command that fans the KG's reconciled asset/CI inventory out to every enabled
CMDB system-of-record (ServiceNow / ERPNext / Egeria / Twenty) as a projection —
the canonical model stays in the graph. The runnable composition adapter behind
the ``asset-mirror`` CronJob is:

    python -m agent_utilities.cli.asset_mirror

Gating is layered and fail-closed (see :func:`run_asset_mirror`):
``ASSET_MIRROR_TARGETS`` selects the sinks, each sink still needs its own
``<SINK>_ENABLE_WRITE`` for a live write, and the pass is **dry-run (report-only)
by default** — pass ``--live`` to actually write (subject to the enable flags).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Protocol, cast

logger = logging.getLogger(__name__)


class AssetMirrorEngine(Protocol):
    """Minimal live graph authority required by the asset-mirror application."""

    backend: object


AssetMirrorEngineProvider = Callable[[], AssetMirrorEngine]


def _authority_unavailable(error_type: str) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "errors": 1,
        "error": "asset mirror engine authority unavailable",
        "error_type": error_type,
    }


def _resolve_authority(
    engine_provider: AssetMirrorEngineProvider,
) -> tuple[AssetMirrorEngine | None, dict[str, Any] | None]:
    try:
        engine = engine_provider()
    except Exception as exc:  # noqa: BLE001 - fail closed with a source-safe result
        logger.debug("asset-mirror: engine authority unavailable", exc_info=True)
        return None, _authority_unavailable(type(exc).__name__)
    if engine is None or getattr(engine, "backend", None) is None:
        return None, _authority_unavailable("MissingAuthority")
    return engine, None


def run(
    *,
    engine_provider: AssetMirrorEngineProvider,
    dry_run: bool = True,
    targets: list[str] | None = None,
) -> dict[str, Any]:
    """Resolve the injected live authority and run one asset-mirror pass."""
    from agent_utilities.knowledge_graph.enrichment.writeback import run_asset_mirror

    engine, unavailable = _resolve_authority(engine_provider)
    if unavailable is not None:
        return unavailable
    live_engine = cast(AssetMirrorEngine, engine)
    return run_asset_mirror(
        backend=live_engine.backend,
        engine=live_engine,
        targets=targets,
        dry_run=dry_run,
    )
