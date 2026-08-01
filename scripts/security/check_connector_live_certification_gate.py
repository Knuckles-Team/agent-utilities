#!/usr/bin/env python3
"""Fast-tier forwarder: connector live-certification source contract (D-ML-1).

Delegates, unmodified, to ``scripts/check_connector_live_certification.py`` (with ``--self-check``, matching the guardrails.yml invocation)
— dark since 2026-07-28 because it was wired only into the push-triggered
``.github/workflows/guardrails.yml`` ("Connector live-certification source contract" step). See
``scripts/security/_fast_tier_forward.py`` for why a forwarder lives here
instead of moving or duplicating that script.

Usage:
  python3 scripts/security/check_connector_live_certification_gate.py
  python3 scripts/security/check_connector_live_certification_gate.py --repository-root DIR
  python3 scripts/security/check_connector_live_certification_gate.py --self-check

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

TARGET = "scripts/check_connector_live_certification.py"
EXTRA_ARGS: list[str] = ["--self-check"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="check-connector-live-certification-gate")
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
