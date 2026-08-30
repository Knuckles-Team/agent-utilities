#!/usr/bin/env python3
"""Fast-tier forwarder: tool/skill reference-integrity gate (D-MQ2-5).

Delegates, unmodified, to ``scripts/check_tool_refs.py`` — dark under the
merge queue (D-ORC-5's audit): wired into neither
``.github/workflows/guardrails.yml`` nor ``.pre-commit-config.yaml``, and
``CONTRACT_CHECK_GLOB`` only discovers ``scripts/security/check_*.py``, one
directory up from where the canonical script lives. See
``scripts/security/_fast_tier_forward.py`` for why a forwarder lives here
instead of moving or duplicating that script.

Forwards WITHOUT ``--strict``: the canonical script's default mode reports
drift as advisory output but only exits non-zero with ``--strict`` passed
(``check_tool_refs.py:152``). Currently green in default mode (fast,
self-contained) — safe to broaden discovery immediately per D-MW-9's
caution against reddening every merge. ``--strict`` is left for a future,
deliberate tightening once the existing drift this script already reports
is burned down — not silently opted into here.

Usage:
  python3 scripts/security/check_tool_refs_gate.py
  python3 scripts/security/check_tool_refs_gate.py --repository-root DIR
  python3 scripts/security/check_tool_refs_gate.py --self-check

Exit 0 = target passed, 1 = target failed, could not be found, or the
forwarder's own self-check failed.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fast_tier_forward import run_gate  # noqa: E402

TARGET = "scripts/check_tool_refs.py"
EXTRA_ARGS: list[str] = []


def main(argv: list[str] | None = None) -> int:
    return run_gate(
        argv=argv,
        prog="check-tool-refs-gate",
        target_relative=TARGET,
        extra_args=EXTRA_ARGS,
    )


if __name__ == "__main__":
    raise SystemExit(main())
