#!/usr/bin/env python3
"""Scaffold a new ontology leg through SDK pack compilation.

CONCEPT:AU-BOUNDARY-R030.9.2

Builds a typed connector declaration for the new leg and compiles it with
``agent_connector_sdk.manifest.ontology_pack.compile_manifest_ontology`` (via
``publish_declaration``). No RDF/OWL/SHACL text is rendered here.

Usage:
  python3 scripts/scaffold_ontology_leg.py CONNECTOR --resource Widget \
      [--resource Gadget] [--output PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_utilities.knowledge_graph.core.ontology_publisher import (
    publish_declaration,
)


def build_declaration(connector: str, resources: list[str]) -> dict[str, Any]:
    """Return the typed declaration for a new leg (fail closed on empty input)."""
    if not connector.strip():
        raise ValueError("connector name must be non-empty")
    if not resources:
        raise ValueError("at least one resource is required")
    return {
        "connector": connector,
        "resources": [{"name": r, "label": r} for r in resources],
        "provenance": {"integrity": {"hash": "0" * 64}},
    }


def scaffold_leg(connector: str, resources: list[str]) -> str:
    """Compile the new leg's declaration to Turtle via the SDK pack compiler."""
    return publish_declaration(build_declaration(connector, resources))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("connector")
    parser.add_argument("--resource", action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        ttl = scaffold_leg(args.connector, args.resource)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.output:
        args.output.write_text(ttl, encoding="utf-8")
    else:
        sys.stdout.write(ttl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
