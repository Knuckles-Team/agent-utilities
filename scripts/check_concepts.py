#!/usr/bin/env python3
"""Concept Registry Validator.

Ensures that *every* ``CONCEPT:<ID>`` marker present in the codebase
(``agent_utilities/**/*.py`` and ``*.rs``) is registered in the single source
of truth ``docs/concepts.yaml``. Exits non-zero if any marker is missing from
the registry, which keeps the docs honest with the code.

Run:  python scripts/check_concepts.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "agent_utilities"
CONCEPTS_PATH = ROOT / "docs" / "concepts.yaml"

# Must stay in lock-step with scripts/build_concepts_yaml.py.
MARKER_RE = re.compile(r"CONCEPT:([A-Z]+-\d+(?:\.[0-9A-Za-z]+)?)")


def markers_in_code() -> dict[str, list[str]]:
    """Map every concept id found in code to the files it appears in."""
    found: dict[str, list[str]] = {}
    for path in sorted(SRC_DIR.rglob("*")):
        if path.suffix not in (".py", ".rs") or not path.is_file():
            continue
        if "__pycache__" in path.parts:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        rel = path.relative_to(ROOT).as_posix()
        for cid in MARKER_RE.findall(content):
            found.setdefault(cid, []).append(rel)
    return found


def registered_ids() -> set[str]:
    if not CONCEPTS_PATH.exists():
        print(
            f"ERROR: {CONCEPTS_PATH.relative_to(ROOT)} does not exist. "
            "Run `python scripts/build_concepts_yaml.py`.",
            file=sys.stderr,
        )
        sys.exit(1)
    data = yaml.safe_load(CONCEPTS_PATH.read_text(encoding="utf-8")) or {}
    return {c["id"] for c in data.get("concepts", [])}


def main() -> int:
    code = markers_in_code()
    registry = registered_ids()

    missing = sorted(set(code) - registry)

    print(
        f"Validating {len(code)} unique concept markers against "
        f"{len(registry)} registered concepts in "
        f"{CONCEPTS_PATH.relative_to(ROOT)}...\n"
    )

    if missing:
        print(f"FAIL: {len(missing)} concept marker(s) missing from registry:")
        for cid in missing:
            example = code[cid][0]
            print(f"  - {cid} (e.g. {example})")
        print(
            "\nRegenerate the registry with `python scripts/build_concepts_yaml.py`.",
            file=sys.stderr,
        )
        return 1

    print(f"OK: all {len(code)} concept markers are registered in concepts.yaml.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
