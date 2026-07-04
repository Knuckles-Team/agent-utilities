#!/usr/bin/env python3
"""Breadth-ingest + assimilation pilot runner (CONCEPT:AU-KG.query.vendor-agnostic-traversal).

Thin CLI over `knowledge_graph.assimilation` for running the ecosystem-evolution
breadth ingest and the acceptance pilot against the LIVE knowledge engine. The
orchestration logic + idempotency live in the package (unit-tested); this just
wires the live engine and prints a JSON report.

Examples
--------
    # Organize + classify the OSS libraries (writes manifest.json per project)
    python scripts/run_assimilation_breadth.py organize --libraries ../../open-source-libraries

    # Breadth ingest libraries + repos + a docs dir, then run the pilot
    python scripts/run_assimilation_breadth.py ingest \
        --libraries ../../open-source-libraries --repos ../.. --pilot

    # Pilot only (assert the engine does not re-propose already-built features)
    python scripts/run_assimilation_breadth.py pilot
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys


def _engine():
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    return IntelligenceGraphEngine.get_active() or IntelligenceGraphEngine()


def _dump(obj) -> None:
    if dataclasses.is_dataclass(obj):
        obj = dataclasses.asdict(obj)
    print(json.dumps(obj, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    org = sub.add_parser("organize", help="classify projects + write manifests")
    org.add_argument("--libraries", required=True)

    ing = sub.add_parser("ingest", help="breadth ingest libraries/repos/docs")
    ing.add_argument("--libraries", action="append", default=[])
    ing.add_argument("--repos", action="append", default=[])
    ing.add_argument("--pilot", action="store_true")

    pil = sub.add_parser("pilot", help="run the assimilation acceptance pilot")
    pil.add_argument("--top-n", type=int, default=50)

    args = ap.parse_args(argv)
    from agent_utilities.knowledge_graph import assimilation as az

    if args.cmd == "organize":
        _dump([dataclasses.asdict(m) for m in az.organize_libraries(args.libraries)])
        return 0

    engine = _engine()
    if args.cmd == "ingest":
        report = az.run_breadth_ingest(
            engine, library_roots=args.libraries, repo_roots=args.repos
        )
        _dump(report)
        if args.pilot:
            print(az.summarize(az.run_pilot(engine)))
        return 0

    if args.cmd == "pilot":
        rep = az.run_pilot(engine, top_n=args.top_n)
        print(az.summarize(rep))
        return 0 if rep.passed else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
