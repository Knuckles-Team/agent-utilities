#!/usr/bin/env python3
"""Fast-tier forwarder: exact-installed local certification source contract (D-ML-1).

Delegates, unmodified, to ``scripts/check_exact_local_gates_harness.py``
— dark since 2026-07-28 because it was wired only into the push-triggered
``.github/workflows/guardrails.yml`` ("Exact-installed local certification source contract" step). See
``scripts/security/_fast_tier_forward.py`` for why a forwarder lives here
instead of moving or duplicating that script.

Usage:
  python3 scripts/security/check_exact_local_gates_harness_gate.py
  python3 scripts/security/check_exact_local_gates_harness_gate.py --repository-root DIR
  python3 scripts/security/check_exact_local_gates_harness_gate.py --self-check

Exit 0 = target passed, 1 = target failed, could not be found, or the
forwarder's own self-check failed.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fast_tier_forward import run_gate  # noqa: E402

TARGET = "scripts/check_exact_local_gates_harness.py"
EXTRA_ARGS: list[str] = []


def main(argv: list[str] | None = None) -> int:
    return run_gate(
        argv=argv,
        prog="check-exact-local-gates-harness-gate",
        target_relative=TARGET,
        extra_args=EXTRA_ARGS,
    )


if __name__ == "__main__":
    raise SystemExit(main())
