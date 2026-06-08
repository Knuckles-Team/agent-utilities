#!/usr/bin/env python3
"""Reliability eval-corpus gate (CONCEPT:AHE-3.1).

Proves the reliability scorer suite has teeth: runs the seed corpus
(:data:`agent_utilities.harness.reliability_corpus.SEED_CASES`) through the real
:class:`EvalHarness` and FAILS if the fraction of cases whose outcome matches
expectation drops below a floor — catching a broken scorer before it ships.

Mirrors the other synthetic-fixture gates (``check_eval_corpus.py``,
``check_retrieval_quality.py``): no network, no live KG, deterministic.

Usage::

    python3 scripts/check_reliability_corpus.py [--degrade]

``--degrade`` corrupts every output so clean cases flip to failing and the gate
trips (used by the meta-test). Exit 0 = match-rate at/above floor. 1 =
regression. 2 = build error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

FLOOR = 0.9


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--degrade", action="store_true")
    args = ap.parse_args()
    try:
        from agent_utilities.harness.reliability_corpus import run_reliability_corpus

        report = run_reliability_corpus(degrade=args.degrade)
    except Exception as exc:  # pragma: no cover - build error
        print(f"ERROR: reliability corpus build failed: {exc}", file=sys.stderr)
        return 2

    print(
        f"Reliability corpus match-rate: {report.match_rate:.2f} "
        f"(floor {FLOOR:.2f}, {report.matched}/{report.total} cases)"
    )
    if report.match_rate < FLOOR:
        for c in report.cases:
            if not c.matched:
                print(
                    f"  MISMATCH {c.name}: expected_pass={c.expected_pass} "
                    f"actual_pass={c.actual_pass} failed={c.failed_scorers}",
                    file=sys.stderr,
                )
        print("FAIL: reliability corpus match-rate below floor.", file=sys.stderr)
        return 1
    print("OK: reliability corpus at/above floor.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
