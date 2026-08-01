#!/usr/bin/env python3
"""Mechanical triage of ``debug_only_swallow``-shaped handlers (D-SWG-3 / D-DST-7).

D-SWG-3 surfaced 435 pre-existing handlers whose ONLY cause-preserving log
call is at ``logger.debug`` (invisible in every production deployment this
codebase ships — see ``scripts/check_swallowed_errors.py``'s "Why DEBUG alone
doesn't count" section). D-DST-7 tracked a ~28-site residual in
``ecosystem/``, ``mcp/``, ``domains/`` after a prior sweep (D-DST-6, not
owned by this lane) fixed 407/435.

Hand-reviewing hundreds of sites one at a time does not scale across ~20
concurrent lanes. This script instead buckets every *currently live*
``debug_only_swallow`` site (re-derived from source via
``check_swallowed_errors.scan()`` — NOT the frozen baseline file, which
accumulates stale entries for sites already fixed by other lanes) into:

  (i)   BENIGN — optional/best-effort operation (telemetry, cache warm,
        health probe, dev-tooling audit, retry-with-backoff, narrow-typed
        fallback with an equivalent local computation next to it). Fix:
        keep at DEBUG, but attach a `# noqa: BLE001 — <reason>` on the
        `except` line — this is the existing, gate-recognized,
        machine-checkable "documented and accepted" convention, so the
        acceptance is enforced structurally rather than just written down
        in the register.
  (ii)  CONTROL_WRITE_PATH — the handler sits on a write/persist/record/
        governance/decision path: a silent failure here loses data or
        masks a real bug rather than merely degrading a best-effort
        extra. These are hand-reviewed (see the classification
        rationale below) and fixed by raising the log level to
        ``warning`` (already visible in production) rather than
        papering over with a noqa.
  (iii) CAUSE_DISCARDING_RERAISE — a handler that DOES re-raise but drops
        the original cause (``raise X(...)`` with no ``from e``). This
        bucket is structurally empty for `debug_only_swallow` sites: the
        gate's own ``_has_raise()`` check excludes any handler that
        re-raises from the ``debug_only_swallow`` shape entirely (see
        ``check_swallowed_errors.py::_find_violations``) — a handler that
        re-raises is a different code population from a handler that
        swallows. Reported for completeness / to make the empty result
        an explicit, checked fact rather than a silent omission.

Classification is driven by keyword matches against (function qualname,
file path, and the handler's own source text) — see ``_CONTROL_KEYWORDS``
for the list and the one-line reason each keyword encodes. Every site this
script has ever classified is hand-verified against its full function body
in the sweep that added it (see the lane's final report); the keyword list
exists so a FUTURE newly-introduced debug_only_swallow site is bucketed the
same way without requiring a human to re-derive the judgment call.

Usage:
  python3 scripts/triage_debug_swallows.py          # print bucket counts + table
  python3 scripts/triage_debug_swallows.py --json    # machine-readable dump
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from check_swallowed_errors import HandlerKey, _iter_except_handlers_with_scope, scan  # noqa: E402

# Keyword -> one-line reason it signals a write/governance/control path
# (bucket ii). Matched case-insensitively against "<file>::<symbol>".
_CONTROL_KEYWORDS: dict[str, str] = {
    "_persist": "persists governance/audit data; a silent failure here loses the record with no other trail",
    "_record_calls": "writes calibration state that later-weighted trading decisions read back",
    "record_market_outcome": "writes calibration state that later-weighted trading decisions read back",
    "governance_workflow": "governance decision/audit pipeline — feeds the evolution flywheel's review queue",
    "_audit_skills": "governance audit path (staleness auditor) whose output drives REMOVE/keep decisions",
    "run_audit_cycle": "top-level governance audit cycle orchestrator",
    "_persist_session": "persists research evidence to the KG; a silent per-item loss is a data-integrity gap",
}

# Keyword -> one-line reason it signals a genuinely optional/best-effort
# operation (bucket i), used only for the human-readable report — the
# classification itself is "not bucket ii" (default) once bucket-ii keywords
# don't match, since every live debug_only_swallow site in this codebase's
# ecosystem/mcp/domains-plus-stragglers population is one or the other.
_BENIGN_HINTS: dict[str, str] = {
    "owl": "best-effort OWL reasoning cycle/enrichment; caller already treats it as optional",
    "audit_installed_packages": "dev-tooling package audit (pip outdated/pip-audit), not a production data path",
    "list_installed_packages": "dev-tooling package listing (pip list), not a production data path",
    "size_position": "position-size helper has an equivalent local Kelly fallback formula immediately below",
    "assess_credit_quality": "one optional credit-risk sub-factor (Merton DD); caller aggregates several and tolerates a missing one",
    "gap_fill": "engine-accelerated path with an equivalent pandas-only fallback immediately below",
    "asof_align": "engine-accelerated path with an equivalent pandas-only fallback immediately below",
    "_client": "engine client construction; caller already treats None as \"use local fallback\"",
    "_check_skills_usage": "dashboard status probe; failure just leaves that panel's row unpopulated",
    "_ingest_capabilities": "per-module best-effort skip inside a loop; the outer scan already logs failures",
    "mcp_server": "best-effort teardown of a lazily-mounted fleet child at process exit",
    "_register_and_heartbeat_forever": "retry loop explicitly says \"will retry\" on the next iteration",
    "is_server_healthy": "health probe; a failed probe already returns False to the caller",
    "cleanup_rogue_instances": "best-effort process-listing lookup with another fallback method below",
    "ingest_jpeg_via_sidecar": "narrow-typed thumbnail decode fallback; degrades to no-thumbnail, not data loss",
    "schemacandidateauditor.record": "audit-log write explicitly documented \"never block the write path\"",
    "kgrulebackend.get_rules": "per-rule best-effort skip inside a loop while loading many rules",
    "fileacpsessionstore._save_sync": "fd-close-during-cleanup; the original exception is re-raised regardless",
    "evaluationengine.evaluate_disentangled": "one optional metric computation; other metrics still returned",
    "_probe_fleet": "fleet catalog probe; sibling handler two lines down already carries the same noqa+reason",
}


def _reason_for(symbol: str, file_rel: str, snippet: str) -> tuple[str, str]:
    """Returns (bucket, reason). bucket is "control_write_path" or "benign"."""
    haystack = f"{file_rel}::{symbol}".lower()
    for kw, reason in _CONTROL_KEYWORDS.items():
        if kw.lower() in haystack:
            return "control_write_path", reason
    for kw, reason in _BENIGN_HINTS.items():
        if kw.lower() in haystack:
            return "benign", reason
    # No keyword matched (new site introduced after this script was written):
    # fail safe toward hand-review rather than silently accepting risk.
    return "control_write_path", "UNCLASSIFIED — no keyword matched; hand-review before accepting"


def _count_reraise_without_from(root: Path) -> int:
    """Bucket (iii) sanity check: handlers that DO re-raise a *different*
    exception object without ``from``. Disjoint from debug_only_swallow by
    construction (see module docstring) — counted here only so the "0" in
    the report is a checked fact, not an assumption."""
    count = 0
    for py in (root / "agent_utilities").rglob("*.py"):
        if any(p in {".venv", "__pycache__", "node_modules"} for p in py.parts):
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        except (OSError, UnicodeDecodeError, SyntaxError):
            continue
        for handler, _symbol in _iter_except_handlers_with_scope(tree):
            for node in ast.walk(ast.Module(body=handler.body, type_ignores=[])):
                if not isinstance(node, ast.Raise):
                    continue
                if node.cause is not None:
                    continue  # has `from ...`
                if node.exc is None:
                    continue  # bare `raise` — re-raises the original, no cause dropped
                if isinstance(node.exc, ast.Name) and handler.name and node.exc.id == handler.name:
                    continue  # `raise exc` — same object, no new exception constructed
                count += 1
    return count


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    found = scan()
    sites = [
        (key, lineno, text)
        for key, (lineno, shape, text) in found.items()
        if shape == "debug_only_swallow"
    ]
    sites.sort(key=lambda s: (s[0][0], s[1]))

    rows = []
    for key, lineno, text in sites:
        file_rel, symbol, exc_types, _shape, _ordinal = key
        bucket, reason = _reason_for(symbol, file_rel, text)
        rows.append(
            {
                "file": file_rel,
                "line": lineno,
                "symbol": symbol,
                "exc_types": exc_types,
                "bucket": bucket,
                "reason": reason,
            }
        )

    reraise_without_from = _count_reraise_without_from(ROOT)

    if args.json:
        print(json.dumps({"sites": rows, "reraise_without_from_count": reraise_without_from}, indent=2))
        return 0

    benign = [r for r in rows if r["bucket"] == "benign"]
    control = [r for r in rows if r["bucket"] == "control_write_path"]
    unclassified = [r for r in control if r["reason"].startswith("UNCLASSIFIED")]

    print(f"debug_only_swallow live sites: {len(rows)}")
    print(f"  (i)   benign (accept w/ noqa reason):      {len(benign)}")
    print(f"  (ii)  control/write path (hand-fix):        {len(control)}  ({len(unclassified)} unclassified)")
    print(f"  (iii) cause-discarding re-raise (raise w/o from), repo-wide sanity count: {reraise_without_from}")
    print(
        "        (structurally disjoint from debug_only_swallow — a handler that re-raises\n"
        "        is excluded from this shape by check_swallowed_errors.py's _has_raise() check;\n"
        "        this count is informational, not part of this backlog.)"
    )
    print()
    for label, bucket_rows in (("BENIGN", benign), ("CONTROL/WRITE PATH", control)):
        print(f"=== {label} ({len(bucket_rows)}) ===")
        for r in bucket_rows:
            print(f"  {r['file']}:{r['line']}\t{r['symbol']}\t{r['reason']}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
