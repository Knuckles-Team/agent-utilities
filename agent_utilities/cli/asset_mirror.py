"""Composition entry point for the governed multi-SoR asset-mirror pass."""

from __future__ import annotations

import argparse
import json

from agent_utilities.knowledge_graph.enrichment.writeback.asset_mirror import run


def _process_engine_authority():
    """Resolve the graph authority owned by the Graph-OS composition root."""
    from agent_utilities.mcp.kg_server import _get_engine

    return _get_engine()


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
    targets = [target.strip() for target in args.targets.split(",") if target.strip()]
    result = run(
        engine_provider=_process_engine_authority,
        dry_run=not args.live,
        targets=targets or None,
    )
    print(json.dumps(result, default=str, indent=2))
    return 0 if result.get("errors", 0) == 0 else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
