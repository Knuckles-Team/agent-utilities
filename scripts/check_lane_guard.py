#!/usr/bin/env python3
"""Fleet entry point for the lane guard, which lives in repository-manager.

The gate itself — canonical checkout read-only, generated reservation views stay
generated, no stray ``CARGO_TARGET_DIR`` — moved with the lane arbitration it
enforces to ``repository_manager.governance.lane_guard`` (OQ-3). Every
repository's ``.pre-commit-config.yaml`` still invokes this path (they locate
agent-utilities' ``scripts/`` and run it with cwd at the repo being committed,
D-CP-3), so this file resolves repository-manager through
``scripts/governance_tool.py`` and runs that gate unmodified. Repositories can
call ``python3 -m repository_manager.governance.lane_guard`` directly instead.

Exit code 1 = refused, including when repository-manager cannot be located: a
guard that cannot run has not found nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.governance_tool import (
        GovernanceUnavailable,
        ensure_governance_importable,
    )

    try:
        ensure_governance_importable()
    except GovernanceUnavailable as exc:
        print(f"REFUSED - lane-guard cannot run: {exc}", file=sys.stderr)
        return 1
    from repository_manager.governance.lane_guard import main as guard_main

    return guard_main()


if __name__ == "__main__":
    raise SystemExit(main())
