#!/usr/bin/env python3
"""Fast-tier forwarder: current-only (no retired-surface) contract (D-ML-1).

Delegates, unmodified, to ``scripts/check_current_only_contract.py``
— dark since 2026-07-28 because it was wired only into the push-triggered
``.github/workflows/guardrails.yml`` ("Current-only contract" step). See
``scripts/security/_fast_tier_forward.py`` for why a forwarder lives here
instead of moving or duplicating that script.

w5-ci-gates-local (D-W5CG-1): this forwarder was the ONE gap in an otherwise
complete set — every other ``guardrails.yml`` script already had a
``scripts/security/check_*.py`` forwarder wiring it into the merge queue's
``contract-checks`` fast-tier gate (``.mergequeue.yaml``, which already
carries a comment anticipating this exact check's ~403 (mergequeue.yaml's own comment estimated ~490; re-measured live this lane) pre-existing
"retired surface" violations under its ``compare: lines`` differential
rationale) except this one. Adding it closes that gap: the queue's
differential compare means main's pre-existing debt is not newly refused,
only a NEW retired-surface reference on a candidate is. NOT added to this
repo's per-commit pre-commit stage (unlike the other, already-clean
forwarders) — see the `contract-checks-current-only-debt` pre-commit hook's
own comment for why.

Usage:
  python3 scripts/security/check_current_only_contract_gate.py
  python3 scripts/security/check_current_only_contract_gate.py --repository-root DIR
  python3 scripts/security/check_current_only_contract_gate.py --self-check

Exit 0 = target passed, 1 = target failed, could not be found, or the
forwarder's own self-check failed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fast_tier_forward import ForwardError, forward, self_check  # noqa: E402

TARGET = "scripts/check_current_only_contract.py"
EXTRA_ARGS: list[str] = []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="check-current-only-contract-gate")
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)
    repo_root = args.repository_root.resolve()

    if args.self_check:
        try:
            self_check(repo_root, TARGET)
        except AssertionError as exc:
            print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
            return 1

    try:
        rc = forward(
            repository_root=repo_root, target_relative=TARGET, extra_args=EXTRA_ARGS
        )
    except ForwardError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        return 1

    if rc != 0:
        print(
            json.dumps(
                {"ok": False, "error": f"{TARGET} exited {rc}", "forwardedTo": TARGET},
                sort_keys=True,
            )
        )
        return rc
    print(
        json.dumps(
            {"ok": True, "forwardedTo": TARGET, "selfCheck": args.self_check},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
