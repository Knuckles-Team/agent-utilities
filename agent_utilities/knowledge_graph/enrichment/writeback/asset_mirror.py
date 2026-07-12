"""Multi-SoR asset-mirror pass entry point (CONCEPT:AU-KG.ingest.enterprise-source-extractor).

One command that fans the KG's reconciled asset/CI inventory out to every enabled
CMDB system-of-record (ServiceNow / ERPNext / Egeria / Twenty) as a projection —
the canonical model stays in the graph. This is the thin runnable behind the
``asset-mirror`` CronJob:

    python -m agent_utilities.knowledge_graph.enrichment.writeback.asset_mirror

Gating is layered and fail-closed (see :func:`run_asset_mirror`):
``ASSET_MIRROR_TARGETS`` selects the sinks, each sink still needs its own
``<SINK>_ENABLE_WRITE`` for a live write, and the pass is **dry-run (report-only)
by default** — pass ``--live`` to actually write (subject to the enable flags).
"""

from __future__ import annotations

import argparse
import json
import logging

logger = logging.getLogger(__name__)


def run(*, dry_run: bool = True, targets: list[str] | None = None) -> dict:
    """Build the live engine/backend and run one mirror pass. Returns the manifest."""
    from agent_utilities.knowledge_graph.enrichment.writeback import run_asset_mirror

    engine = None
    try:
        from agent_utilities.mcp import kg_server

        engine = kg_server._get_engine()
    except Exception:  # noqa: BLE001 - offline → dry-run over an empty backend
        logger.debug(
            "asset-mirror: engine unavailable; running with no backend", exc_info=True
        )
    backend = getattr(engine, "backend", None) if engine is not None else None
    return run_asset_mirror(
        backend=backend, engine=engine, targets=targets, dry_run=dry_run
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="asset-mirror",
        description="Mirror the KG's asset/CI inventory to all enabled CMDB sinks.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Apply writes (subject to each sink's <SINK>_ENABLE_WRITE). "
        "Default is dry-run / report-only.",
    )
    parser.add_argument(
        "--targets",
        default="",
        help="Comma-separated sink override (else ASSET_MIRROR_TARGETS).",
    )
    args = parser.parse_args(argv)
    targets = [t.strip() for t in args.targets.split(",") if t.strip()] or None
    result = run(dry_run=not args.live, targets=targets)
    print(json.dumps(result, default=str, indent=2))
    return 0 if result.get("errors", 0) == 0 else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
