#!/usr/bin/env python3
"""Measure the graph-os intent surface's tool-selection accuracy.

CONCEPT:AU-ECO.mcp.intent-surface-selection-accuracy — Seam 8 program-design §4 phase 4
("A/B measurement — selection accuracy... condensed vs. intent"). Runs the
REAL resolver (:func:`agent_utilities.mcp.tools.intent_tools.resolve_intent`,
CPD-backed when ``docs/capabilities-power.json`` is present) against a small,
hand-labelled corpus of natural-language phrasings
(:mod:`agent_utilities.knowledge_graph.retrieval.intent_selection_accuracy`)
and prints top-1/top-3 accuracy plus a per-case breakdown.

This is a MEASUREMENT tool, not a merge gate — pre-commit/CI does not fail the
build on a below-threshold score (the corpus is illustrative, not a
statistically rigorous benchmark); the pytest counterpart
(``tests/unit/test_intent_selection_accuracy.py``) is the regression tripwire
that DOES fail CI if accuracy craters.

Usage::

    python scripts/measure_intent_routing_accuracy.py
    python scripts/measure_intent_routing_accuracy.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent_utilities.knowledge_graph.retrieval.intent_selection_accuracy import (  # noqa: E402
    _to_jsonable,
    measure_selection_accuracy,
    render_report,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--json", action="store_true", help="Emit the machine-readable report instead."
    )
    args = ap.parse_args()

    report = measure_selection_accuracy()
    if args.json:
        print(json.dumps(_to_jsonable(report), indent=2))
    else:
        print(render_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
