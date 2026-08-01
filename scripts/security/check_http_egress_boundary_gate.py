#!/usr/bin/env python3
"""Fast-tier forwarder: governed HTTP egress factory boundary (D-CIM-4).

Delegates, unmodified, to ``scripts/check_http_egress_boundary.py``
— dark since 2026-07-28 because it was wired only into the push-triggered
``.github/workflows/guardrails.yml`` ("HTTP egress boundary" step). See
``scripts/security/_fast_tier_forward.py`` for why a forwarder lives here
instead of moving or duplicating that script.

**What changed before this could be ported (D-CIM-4).** The check passed
clean but measured ~11s — above the "well under a few seconds" fast-tier
bar. Profiling found the SAME two-full-tree-walk-per-file shape as
``check_context_compiler_boundary.py`` (D-CIM-3): every ``_BLOCKED`` pattern
is rooted at one of six module names, so a file mentioning none of them
cannot possibly match — skipping the walk for such a file (1336 of 1502,
89%, measured) cut it to ~6s.

That same investigation surfaced a REAL pre-existing bug while auditing
``_qualified``'s alias resolution: an unaliased dotted submodule import
(``import http.client`` / ``import urllib.request``, no ``as``) was stored
under its FULL dotted path instead of the top-level bound name, which made
``_qualified`` double-count the submodule segment and silently miss every
call through ``http.client.*``/``urllib.request.*`` reached that way — two
of the sixteen ``_BLOCKED`` patterns were dead code. Fixed (see
``_imports``'s updated comment). Fixing it surfaced exactly one real,
already-existing violation:
``agent_utilities/deployment/certification_oidc.py``'s
``EphemeralLoopbackOidcAuthority._verified_json`` uses
``http.client.HTTPSConnection`` directly against a loopback OIDC
authority the SAME process just bound, pinned to a freshly generated
one-run CA, and deliberately reads at most ``_MAX_BODY_BYTES + 1`` bytes off
the raw socket regardless of what the (self-controlled) peer sends — a
bounded-read guarantee the httpx-based ``core.http_client`` factory's
synchronous surface does not offer directly, and not general outbound
egress. Allowlisted (mirroring
``check_context_compiler_boundary.py``'s ``RAW_PROVIDER_ALLOWLIST`` idiom)
with the justification duplicated inline at the call site, per that same
convention ("each entry must be justified inline at its call site, not
just here").

Usage:
  python3 scripts/security/check_http_egress_boundary_gate.py
  python3 scripts/security/check_http_egress_boundary_gate.py --repository-root DIR
  python3 scripts/security/check_http_egress_boundary_gate.py --self-check

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

TARGET = "scripts/check_http_egress_boundary.py"
EXTRA_ARGS: list[str] = []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="check-http-egress-boundary-gate")
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
