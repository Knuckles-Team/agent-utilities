#!/usr/bin/env python3
"""Fast-tier forwarder: swallowed-error guardrail (D-ORC-5).

Delegates, unmodified, to ``scripts/check_swallowed_errors.py`` — the
ratchet gate for a handler that discards *why* an operation failed (see that
script's own module docstring: "Every major blocker in [the 2026-07-22/23
debugging session] hid behind one of these shapes"). It was wired into
``.pre-commit-config.yaml`` (``check-swallowed-errors``, ``always_run: true``)
but had **no** forwarder here in ``scripts/security/`` — the one directory
``agent_utilities/governance/merge_queue.py``'s fast tier actually discovers
(``CONTRACT_CHECK_GLOB = "scripts/security/check_*.py"``, see
``_fast_tier_forward.py``'s module docstring for why forwarders exist at
all). The merge queue never runs pre-commit, so every merge that went
through the queue — as opposed to a local ``git commit`` with hooks
installed — enforced nothing here. This is the gate with documented proof
it was actually escaped this way: it was found RED on ``main`` (three new
un-baselined cause-dropping handlers, D-SWG-3's 2026-08-01 re-verification)
the same night violations landed through the queue.

Follows the same forward-and-relay contract as every other
``scripts/security/check_*_gate.py`` in this directory (see
``_fast_tier_forward.py``): the canonical script stays exactly where
``.pre-commit-config.yaml`` and every human invocation already expect it, and
this thin file gives the fast tier a ``check_*.py`` it can discover.

Usage:
  python3 scripts/security/check_swallowed_errors_gate.py
  python3 scripts/security/check_swallowed_errors_gate.py --repository-root DIR
  python3 scripts/security/check_swallowed_errors_gate.py --self-check

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

TARGET = "scripts/check_swallowed_errors.py"
EXTRA_ARGS: list[str] = []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="check-swallowed-errors-gate")
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
