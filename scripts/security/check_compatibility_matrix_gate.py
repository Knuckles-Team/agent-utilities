#!/usr/bin/env python3
"""Fast-tier forwarder: compatibility and release-train contract (D-ML-1).

Delegates, unmodified, to ``scripts/release/check_compatibility.py`` (with ``--matrix-only``, matching the guardrails.yml invocation)
— dark since 2026-07-28 because it was wired only into the push-triggered
``.github/workflows/guardrails.yml`` ("Compatibility and release-train contract" step). See
``scripts/security/_fast_tier_forward.py`` for why a forwarder lives here
instead of moving or duplicating that script.

Usage:
  python3 scripts/security/check_compatibility_matrix_gate.py
  python3 scripts/security/check_compatibility_matrix_gate.py --repository-root DIR
  python3 scripts/security/check_compatibility_matrix_gate.py --self-check

Exit 0 = target passed, 1 = target failed, could not be found, or the
forwarder's own self-check failed.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fast_tier_forward import run_gate  # noqa: E402

TARGET = "scripts/release/check_compatibility.py"
EXTRA_ARGS: list[str] = ["--matrix-only"]


def main(argv: list[str] | None = None) -> int:
    return run_gate(
        argv=argv,
        prog="check-compatibility-matrix-gate",
        target_relative=TARGET,
        extra_args=EXTRA_ARGS,
    )


if __name__ == "__main__":
    raise SystemExit(main())
