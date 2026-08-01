#!/usr/bin/env python3
"""Fast-tier forwarder: no orphaned test file (D-RG2-1 / D-WS-3 / D-OB-13a).

Delegates, unmodified, to ``scripts/check_wiring.py --check-test-collection``
— see ``scripts/security/_fast_tier_forward.py`` for why a forwarder lives
here instead of moving or duplicating that script: the canonical sweep
(``find_orphaned_test_files``) is shared with the developer-facing
``--wire-first-report`` / ``--update-wire-first-baseline`` workflow described
in ``AGENTS.md``'s Wire-First step 4, and ``.pre-commit-config.yaml``'s
``check-wire-first`` hook already runs it locally. Neither the pre-commit
hook nor a push-triggered workflow is wired into the merge queue's fast tier
(``agent_utilities/governance/merge_queue.py`` discovers gates by globbing
``scripts/security/check_*.py`` only) — this forwarder closes that gap so
the check actually gates every merge, not just a developer's local commit.

**What this catches.** A ``test_*.py`` file under ``tests/`` that neither
``pytest.ini``'s ``testpaths`` covers nor an explicit ``pytest ...`` in
``.pre-commit-config.yaml`` / ``.github/workflows/*.yml`` points at — i.e. a
test nothing ever runs, so "the suite passes" says nothing about it (the
defect this gate exists to keep from reopening). As of the D-RG2-1/D-WS-3
audit (``reports/test-collection-audit.md``), ``pytest.ini``'s ``testpaths``
is the single recursive ``tests`` entry, so every ``test_*.py`` anywhere
under ``tests/`` is collected by construction and the ratchet baseline
(``scripts/wire_first_baseline.json``'s ``orphaned_test_files``) is empty —
this gate now fails on ANY new orphan, not just a regression against a
stale baseline.

**Fail-closed vs. an honest absence.** Mirrors every other gate in this
directory (see ``check_cypher_write_subset.py``'s module docstring for the
canonical statement): a target script that cannot even be found or launched
is a *degraded read* and this forwarder refuses (exit 1) rather than
reporting a clean pass; a repo that genuinely has zero orphaned test files
(the current state) is an honest absence and passes.

Usage:
  python3 scripts/security/check_test_collection_gate.py
  python3 scripts/security/check_test_collection_gate.py --repository-root DIR
  python3 scripts/security/check_test_collection_gate.py --self-check

Exit 0 = no orphaned test file (or none beyond the ratchet baseline), 1 = a
new orphan was found, the canonical target is missing, or the forwarder's
own self-check failed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fast_tier_forward import ForwardError, forward, self_check  # noqa: E402

TARGET = "scripts/check_wiring.py"
EXTRA_ARGS: list[str] = ["--check-test-collection"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="check-test-collection-gate")
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
                {
                    "ok": False,
                    "error": f"{TARGET} {' '.join(EXTRA_ARGS)} exited {rc}",
                    "forwardedTo": TARGET,
                },
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
