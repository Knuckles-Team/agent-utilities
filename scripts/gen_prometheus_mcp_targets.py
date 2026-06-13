#!/usr/bin/env python3
"""Generate a Prometheus file-SD target list for the whole MCP fleet.

Reads ``deploy/mcp-fleet.registry.yml`` (the machine-generated registry of every
``agents/*`` connector exposing a streamable-http MCP server) and writes a
Prometheus ``file_sd_configs`` JSON file. Each MCP exposes an unauthenticated
``GET /metrics`` (added by ``create_mcp_server``; CONCEPT:OS-5.23), reachable on
the swarm ``caddy`` overlay at ``<stack>_<service>:<container_port>`` — the same
``stack_service`` DNS convention the static cross-stack jobs already use.

So one generated job scrapes the entire fleet; re-run on registry change (wired
into the ``service-observability-provisioner`` skill / day-0). Idempotent.

Usage:
    gen_prometheus_mcp_targets.py [--registry PATH] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

_HERE = Path(__file__).resolve().parent
_DEFAULT_REGISTRY = _HERE.parent / "deploy" / "mcp-fleet.registry.yml"


def build_targets(registry: dict) -> list[dict]:
    """One file-SD entry per MCP service: target + stack/service/job labels."""
    defaults = registry.get("defaults", {}) or {}
    default_port = defaults.get("container_port", 8000)
    entries: list[dict] = []
    for svc in registry.get("services", []) or []:
        name = svc.get("name")
        if not name:
            continue
        port = svc.get("container_port", default_port)
        # Swarm overlay DNS for a single-service stack: <stack>_<service>.
        target = f"{name}_{name}:{port}"
        entries.append(
            {
                "targets": [target],
                "labels": {"job": "mcp-fleet", "stack": name, "service": name},
            }
        )
    return entries


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registry", type=Path, default=_DEFAULT_REGISTRY)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("/home/apps/workspace/services/lgtm/targets/mcp-fleet.json"),
        help="Prometheus file_sd JSON output path.",
    )
    args = ap.parse_args()

    registry = yaml.safe_load(args.registry.read_text())
    targets = build_targets(registry)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(targets, indent=2) + "\n")
    print(f"Wrote {len(targets)} MCP scrape targets → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
