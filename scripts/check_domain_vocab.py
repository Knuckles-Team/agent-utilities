#!/usr/bin/env python3
"""Gate: every OKF-CIS marker's domain is in the closed vocabulary (CONCEPT:OS-5.77).

The anti-sprawl guarantee: breadth is governed. A marker whose ``<domain>`` is not
listed in ``governance/domain_vocab.yaml`` for its pillar fails the build — so the
flat-number sprawl cannot reconstitute itself as ad-hoc domain sprawl. Also verifies
each id parses under the OKF-CIS grammar and its SLUG is registered.

Usage: python scripts/check_domain_vocab.py [ROOT ...]  (default: cwd)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent_utilities.governance import concept_hierarchy as ch

_EXT = {".py", ".rs", ".md"}
_SKIP = {"__pycache__", ".git", ".venv", "node_modules", "target", "build", "dist"}


def scan(root: Path) -> list[str]:
    errs: list[str] = []
    known_slugs = set(ch.load_slug_registry().values())
    for p in root.rglob("*"):
        if p.suffix not in _EXT or any(s in p.parts for s in _SKIP):
            continue
        if p.name in {"check_domain_vocab.py", "domain_vocab.yaml", "slug_registry.yaml"}:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for m in ch.OKF_MARKER_RE.finditer(text):
            cid = m.group("id")
            try:
                parsed = ch.parse_okf_id(cid)
            except ValueError as ex:
                errs.append(f"{p.relative_to(root)}: {cid} — {ex}")
                continue
            if parsed.slug not in known_slugs:
                errs.append(f"{p.relative_to(root)}: {cid} — SLUG {parsed.slug!r} not registered")
            if not ch.is_valid_domain(parsed.pillar, parsed.domain):
                errs.append(
                    f"{p.relative_to(root)}: {cid} — domain {parsed.domain!r} not in "
                    f"closed vocab for pillar {parsed.pillar}"
                )
    return errs


def main(argv: list[str]) -> int:
    roots = [Path(a) for a in argv] or [Path.cwd()]
    errs: list[str] = []
    for r in roots:
        errs.extend(scan(r))
    if errs:
        print(f"FAIL: {len(errs)} OKF-CIS marker(s) violate the closed vocab / grammar:")
        for e in errs[:60]:
            print("  " + e)
        if len(errs) > 60:
            print(f"  … and {len(errs) - 60} more")
        return 1
    print("OK: all OKF-CIS markers use registered slugs + closed-vocab domains.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
