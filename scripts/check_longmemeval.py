#!/usr/bin/env python3
"""CONCEPT:AHE-3.12 — LongMemEval-S regression gate.

CI floor check for the memory-first stack. Reads a results JSON (a list of per-question
``{"correct": bool, "question_type": str}`` rows, as emitted by ``GET /benchmark/report``'s
upstream ``/benchmark/query`` calls) and fails if accuracy on the frozen subset drops below the
floor. The full 500-question run is a nightly/on-demand target; this gate guards a small subset so
regressions are caught fast without a multi-minute CI step.

Usage:
    python scripts/check_longmemeval.py --results results.json --floor 0.95
    python scripts/check_longmemeval.py --self-test     # validate the gate logic itself

Exit code 0 = pass, 1 = below floor / error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Import the single source of truth for scoring so the gate and the live router never diverge.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent_utilities.server.routers.benchmark import (  # noqa: E402
    aggregate_report,
    judge_binary,
)


def _score_rows(rows: list[dict]) -> list[dict]:
    """Ensure each row has a 'correct' bool, judging predicted vs gold when absent."""
    scored = []
    for r in rows:
        if "correct" not in r and "answer" in r and "gold_answer" in r:
            r = {**r, "correct": judge_binary(r["answer"], r["gold_answer"])}
        scored.append(r)
    return scored


def _self_test() -> int:
    # A tiny embedded fixture proves the gate logic end-to-end without external data.
    rows = [
        {"answer": "Paris", "gold_answer": "Paris", "question_type": "single-session"},
        {"answer": "it cost $40", "gold_answer": "40", "question_type": "knowledge-update"},
        {"answer": "I don't know", "gold_answer": "Tokyo", "question_type": "multi-session"},
    ]
    report = aggregate_report(_score_rows(rows))
    assert report["total"] == 3, report
    assert report["correct"] == 2, report
    assert abs(report["accuracy"] - 2 / 3) < 1e-9, report
    assert report["by_category"]["single-session"]["accuracy"] == 1.0, report
    print("self-test OK:", json.dumps(report))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="LongMemEval-S CI floor gate (AHE-3.12)")
    ap.add_argument("--results", help="Path to a results JSON (list of per-question rows)")
    ap.add_argument("--floor", type=float, default=0.95, help="Minimum accuracy (default 0.95)")
    ap.add_argument("--self-test", action="store_true", help="Validate the gate logic only")
    args = ap.parse_args()

    if args.self_test or not args.results:
        return _self_test()

    path = Path(args.results)
    if not path.is_file():
        print(f"ERROR: results file not found: {path}", file=sys.stderr)
        return 1
    rows = json.loads(path.read_text())
    report = aggregate_report(_score_rows(rows))
    print(json.dumps(report, indent=2))
    if report["accuracy"] < args.floor:
        print(
            f"FAIL: accuracy {report['accuracy']:.3f} < floor {args.floor:.3f}",
            file=sys.stderr,
        )
        return 1
    print(f"OK: accuracy {report['accuracy']:.3f} >= floor {args.floor:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
