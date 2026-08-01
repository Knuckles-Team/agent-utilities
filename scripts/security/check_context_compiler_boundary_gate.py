#!/usr/bin/env python3
"""Fast-tier forwarder: mandatory ContextCompiler model boundary (D-CIM-3).

Delegates, unmodified, to ``scripts/check_context_compiler_boundary.py``
— dark since 2026-07-28 because it was wired only into the push-triggered
``.github/workflows/guardrails.yml`` ("ContextCompiler boundary gate" step).
See ``scripts/security/_fast_tier_forward.py`` for why a forwarder lives here
instead of moving or duplicating that script.

**Why this one needed work before it could be ported (D-CIM-3).** Two
problems, both fixed on this branch:

1. Four real pre-existing violations: four test files under
   ``tests/unit/capabilities/`` constructed ``pydantic_ai.Agent`` directly
   (18 call sites total across the four) instead of through
   ``create_context_agent``, the repo's sole governed constructor. Fixed by
   migrating every site to ``create_context_agent(..., default_capabilities=False)``
   (preserving each test's original capability list exactly — no default
   capability set was added, so the tests still exercise the SAME isolated
   behavior they did before), plus wrapping every live ``agent.run``/
   ``run_sync`` call in ``with use_grounding_policy("none"):`` — governed
   construction wraps the model in the mandatory ContextCompiler transport,
   which by default requires live evidence compilation before a request may
   proceed; the existing composition test in ``test_content_guardrails.py``
   already established this exact opt-out pattern for a ``FunctionModel``
   test double with no live KG behind it.
2. Runtime: profiled at ~34s over the ~3100-file scan (``agent_utilities`` +
   ``scripts`` + ``tests`` + ``examples``) — too slow to add to the merge
   queue's 180s fast tier alongside its tests. ``scripts/check_context_compiler_boundary.py``
   itself now documents the three algorithmic fixes (skip the
   canonical-file-only ``_function_ancestors`` walk for every other file;
   merge two of the three full-tree ``ast.walk`` passes into one; skip both
   walks entirely for a file that mentions none of the substrings any
   violation requires) that brought it to ~13s with an identical violation
   set (verified against the original 3-pass/4-walk implementation and a
   battery of synthetic before/after fixtures covering every violation kind
   plus the syntax-error and clean-file paths).

Usage:
  python3 scripts/security/check_context_compiler_boundary_gate.py
  python3 scripts/security/check_context_compiler_boundary_gate.py --repository-root DIR
  python3 scripts/security/check_context_compiler_boundary_gate.py --self-check

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

TARGET = "scripts/check_context_compiler_boundary.py"
EXTRA_ARGS: list[str] = []
# The target scans the whole tree (agent_utilities/scripts/tests/examples);
# ~13s measured after the D-CIM-3 speedup. ``forward``'s default timeout
# (55s, just under the merge queue's own 60s CONTRACT_CHECK_BUDGET_SECONDS
# for the WHOLE forwarder invocation — a larger inner timeout here would be
# moot, since the outer harness kills the process at 60s regardless) leaves
# ample headroom without needing an override.


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="check-context-compiler-boundary-gate")
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
