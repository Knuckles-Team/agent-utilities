#!/usr/bin/env python3
"""Extend-Before-Invent concept governance gate.

Local mirror of the ``concept-governance.yml`` CI workflow so that
``pre-commit run --all-files`` catches a governance violation BEFORE a push —
no more push-then-discover.

The CI workflow scans the PR diff (``git diff origin/main...HEAD``) for newly
introduced ``CONCEPT:<ID>`` markers and FAILS unless each one:

  1. is referenced by a design document under ``.specify/design/**.md``; and
  2. uses a pillar prefix that is registered in ``docs/concepts.yaml``.

This script reproduces that intent. Two deliberate refinements over the raw CI
bash make it correct on the canonical dev box (and identical to CI on GitHub):

Base ref selection
------------------
CI diffs against ``origin/main``. On the dev box ``origin/main`` is frequently
far behind the local integration trunk ``main`` (work is merged locally and
pushed in batches), so diffing against the stale remote would flag dozens of
*already-accepted* concepts as "new" — a false positive CI on GitHub would never
produce once ``main`` is pushed. The base is therefore the *nearest available
trunk*: whichever of ``origin/main`` / ``main`` has the most-recent merge-base
with ``HEAD``. Once ``main`` is pushed, ``origin/main == main`` and this is
identical to CI. Override with ``--base <ref>``.

Genuinely-new only
------------------
A concept counts as "new" only if it does NOT already exist anywhere in the base
revision. This rejects pure churn — e.g. when a turtle/whitespace reformat moves
a line carrying an existing ``CONCEPT:`` marker, the marker appears on a ``+``
line of the diff but is not a new concept. Grandfathered concepts already in the
trunk are never re-litigated (which also matches CI once the trunk is current).

Exit codes: 0 = governance OK (or no new concepts), 1 = violation(s).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONCEPTS_PATH = ROOT / "docs" / "concepts.yaml"
DESIGN_DIR = ROOT / ".specify" / "design"

# Same id grammar as the CI workflow / canonical marker regex: a dotless id
# (SAFE-1), a dotted id (KG-2.101), or a letter-suffix id (KG-2.20g).
CONCEPT_RE = re.compile(r"(?<=CONCEPT:)([A-Z]+-\d+(?:\.[0-9A-Za-z]+)?)")
PILLAR_RE = re.compile(r"^[A-Z]+")


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


def _ref_exists(ref: str) -> bool:
    return (
        subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", ref],
            cwd=ROOT,
            capture_output=True,
            check=False,
        ).returncode
        == 0
    )


def _merge_base(ref: str) -> "str | None":
    mb = _git("merge-base", ref, "HEAD")
    return mb or None


def resolve_base(explicit: "str | None") -> "str | None":
    """Pick the diff base: explicit override, else the nearest available trunk."""
    if explicit:
        if not _ref_exists(explicit):
            print(f"ERROR: base ref '{explicit}' does not exist", file=sys.stderr)
            sys.exit(2)
        return _merge_base(explicit)

    candidates = [r for r in ("origin/main", "main") if _ref_exists(r)]
    bases = [(r, _merge_base(r)) for r in candidates]
    bases = [(r, b) for r, b in bases if b]
    if not bases:
        return None

    best_ref, best_mb = bases[0]
    for ref, mb in bases[1:]:
        is_ancestor = (
            subprocess.run(
                ["git", "merge-base", "--is-ancestor", best_mb, mb],
                cwd=ROOT,
                capture_output=True,
                check=False,
            ).returncode
            == 0
        )
        if is_ancestor:
            best_ref, best_mb = ref, mb
    return best_mb


def valid_pillars() -> "set[str]":
    """Derive valid pillar prefixes from docs/concepts.yaml (self-maintaining)."""
    try:
        import yaml

        data = yaml.safe_load(CONCEPTS_PATH.read_text(encoding="utf-8")) or {}
        pillars = {p.split("-")[0] for p in data.get("pillars", [])}
        if pillars:
            return pillars
    except Exception:  # noqa: BLE001 — fall back to documented defaults
        pass
    return {"ORCH", "KG", "AHE", "ECO", "OS", "SAFE", "EE", "ML"}


def _exists_at_base(concept: str, base: str) -> bool:
    """True if ``CONCEPT:<concept>`` already exists anywhere in the base tree."""
    return (
        subprocess.run(
            ["git", "grep", "--quiet", "-F", f"CONCEPT:{concept}", base],
            cwd=ROOT,
            capture_output=True,
            check=False,
        ).returncode
        == 0
    )


def new_concepts(base: str) -> "list[str]":
    diff = _git("diff", f"{base}...HEAD", "--unified=0")
    added: "set[str]" = set()
    for line in diff.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            added.update(CONCEPT_RE.findall(line))
    # Keep only concepts that did NOT already exist at the base (reject churn).
    return sorted(c for c in added if not _exists_at_base(c, base))


def has_design_doc(concept: str) -> bool:
    if not DESIGN_DIR.is_dir():
        return False
    for md in DESIGN_DIR.rglob("*.md"):
        try:
            if concept in md.read_text(encoding="utf-8", errors="ignore"):
                return True
        except OSError:
            continue
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", help="explicit base ref to diff against (default: nearest trunk)")
    args = ap.parse_args()

    base = resolve_base(args.base)
    if not base:
        print("No base ref available; skipping concept governance (nothing to diff).")
        return 0

    concepts = new_concepts(base)
    if not concepts:
        print("No new CONCEPT: tags found. Governance check passed.")
        return 0

    valid = valid_pillars()
    violations: "list[str]" = []
    for concept in concepts:
        m = PILLAR_RE.match(concept)
        pillar = m.group(0) if m else ""
        if not has_design_doc(concept):
            violations.append(f"  {concept} - No design document references this concept")
        if pillar not in valid:
            violations.append(
                f"  {concept} - Invalid pillar prefix: {pillar} "
                f"(must be one of: {' '.join(sorted(valid))})"
            )

    if violations:
        print(f"New CONCEPT tags introduced since {base[:12]}:")
        for c in concepts:
            print(f"  - {c}")
        print("\nGovernance violations found:")
        print("\n".join(violations))
        print(
            "\nTo fix: create a design document in .specify/design/<feature>/ that "
            "references each new CONCEPT tag (see .specify/design/_template.md)."
        )
        return 1

    print(f"All {len(concepts)} new concept(s) have design documents. Governance check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
