#!/usr/bin/env python3
"""Concept Registry Validator.

Ensures that *every* ``CONCEPT:<ID>`` marker present in the codebase
(``agent_utilities/**/*.py`` and ``*.rs``) is registered in the single source
of truth ``docs/concepts.yaml``. Exits non-zero if any marker is missing from
the registry, which keeps the docs honest with the code.

Run:  python scripts/check_concepts.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "agent_utilities"
CONCEPTS_PATH = ROOT / "docs" / "concepts.yaml"

# Single source of the marker grammar — shared with build_concepts_yaml.py and
# the allocator so the three scanners can never drift. ``findall`` returns the
# id (the one capturing group) for each match.
sys.path.insert(0, str(ROOT))
from agent_utilities.governance.concept_allocator import MARKER_RE  # noqa: E402
from agent_utilities.governance.concept_hierarchy import (  # noqa: E402
    canonicalize,
    observed_project_namespaces,
)


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
    """Every id the registry recognizes — raw ids PLUS their aliases + dotted forms.

    Accepting the alias set (CONCEPT:AU-OS.governance.concept-hierarchy-standardization / B5) means a marker written in
    EITHER the flat (``EG-321``) OR the canonical dotted (``EG-0.321``) form
    validates against the same registered concept, without invalidating the
    existing flat scheme.
    """
    if not CONCEPTS_PATH.exists():
        print(
            f"ERROR: {CONCEPTS_PATH.relative_to(ROOT)} does not exist. "
            "Run `python scripts/build_concepts_yaml.py`.",
            file=sys.stderr,
        )
        sys.exit(1)
    data = yaml.safe_load(CONCEPTS_PATH.read_text(encoding="utf-8")) or {}
    ids: set[str] = set()
    for c in data.get("concepts", []):
        if not isinstance(c, dict) or not c.get("id"):
            continue
        ids.add(c["id"])
        if c.get("dotted"):
            ids.add(c["dotted"])
        for alias in c.get("aliases", []) or []:
            ids.add(alias)
    return ids


def main() -> int:
    code = markers_in_code()
    registry = registered_ids()

    # A code marker validates if its raw id OR its canonical dotted form is
    # registered — so both flat and dotted markers pass (non-breaking).
    observed = observed_project_namespaces(list(code))

    def _resolved(cid: str) -> str:
        try:
            return canonicalize(cid, observed_project_ns=observed)
        except ValueError:
            return cid

    missing = sorted(
        cid for cid in code if cid not in registry and _resolved(cid) not in registry
    )

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
