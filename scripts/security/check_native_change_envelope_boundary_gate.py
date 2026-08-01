#!/usr/bin/env python3
"""Fast-tier forwarder: native ChangeEnvelope boundary contract (D-CIM-1/2).

Delegates, unmodified, to ``scripts/check_native_change_envelope_boundary.py``
— dark since 2026-07-28 because it was wired only into the push-triggered
``.github/workflows/guardrails.yml`` ("Native ChangeEnvelope boundary gate"
step). See ``scripts/security/_fast_tier_forward.py`` for why a forwarder
lives here instead of moving or duplicating that script.

**Why this one could not simply be ported like the other 7 (D-ML-1) checks.**
This check FAILED against the merged tree: ``worldmodel_pipeline.py``'s
``_ingest_full`` reached past ``IngestionEngine``'s class boundary to call the
private ``_enrich_text`` seam directly instead of going through a public
entry point (the same anti-pattern was independently repeated in
``engine_tasks.py``'s ``feed_ingest`` task and
``knowledge_graph/memory/native_ingest.py``'s ``enrich_pending_documents``).
The underlying write path was already native (``_enrich_text`` itself commits
through ``ingest_graph_slice``, never a direct backend write) — the violation
was encapsulation, not architecture: three call sites outside the class had
no ``IngestionManifest`` to route through the normal ``ingest()`` adaptor
dispatch (they only have already-fetched article/document text), so they
reached past the underscore boundary instead. The fix adds
``IngestionEngine.enrich_text`` — a thin public wrapper around the existing
``_enrich_text`` seam — and repoints all three call sites at it. This is a
zero-tolerance, no-baseline architectural gate (see the check's own
docstring), so the ONLY correct move was fixing the code, not baselining the
violation; baselining a gate's one violation would have made it vacuous.

Usage:
  python3 scripts/security/check_native_change_envelope_boundary_gate.py
  python3 scripts/security/check_native_change_envelope_boundary_gate.py --repository-root DIR
  python3 scripts/security/check_native_change_envelope_boundary_gate.py --self-check

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

TARGET = "scripts/check_native_change_envelope_boundary.py"
EXTRA_ARGS: list[str] = []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="check-native-change-envelope-boundary-gate")
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
