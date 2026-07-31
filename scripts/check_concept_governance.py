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

Merged-but-undocumented mode (``--audit-merged``,
CONCEPT:AU-OS.governance.merged-concept-visibility-audit)
-----------------------------------------------------------------------------
Every mode above is diff-based: it resolves a *base* (a merge-base against the
nearest trunk) and only ever looks at commits between that base and ``HEAD``.
That is exactly the blind spot reconciliation gate 2 found (D-RG2-2/D-RG2-3):
once a lane merges, ``main`` itself becomes the base on the next run, the diff
is empty, and the gate reports "no new concepts" forever after — even though
the concept it just merged still has no design doc. A rule that only binds
before merge and goes silent after guarantees a permanent, growing backlog.

``--audit-merged`` ignores ``--base``/diffing entirely and instead audits
**every ``CONCEPT:<id>`` marker literally present anywhere in the repo tree**
against the design-doc corpus. So it sees debt regardless of whether it
landed yesterday or two years ago, and regardless of merge status.

This deliberately does NOT reuse ``docs/concepts.yaml`` as the concept
universe, even though that is the single generated source of truth for the
*registered* concept catalog and is what the CI workflow doc / D-RG2-3's
"suggested shape" both point at. ``docs/concepts.yaml`` is built by
``build_concepts_yaml.py``, which scans only ``agent_utilities/**`` — but a
``CONCEPT:`` marker is legitimately written in ``scripts/``, ``tests/``,
``mcp_v2_gateway/`` (a sibling package outside ``agent_utilities/``), and
prose docs (``AGENTS.md``, ``docs/architecture/*.md``). Five of the 39
concepts this exact gate gap produced (D-RG2-2) live ONLY in such files —
using ``docs/concepts.yaml`` as the universe would have silently exempted
them from the audit this mode exists to provide, defeating its own purpose.
So this mode re-scans the tree directly with the same marker grammar
(``OKF_MARKER_RE``), the same way the diff-based mode's own
``_exists_at_base`` already searches the whole tree unrestricted by
directory — consistent with THIS gate's existing scope, not
``build_concepts_yaml.py``'s narrower one.

Because a bare re-scan would instantly redline every pre-existing gap the day
this mode ships, it is a **ratchet keyed by concept id** (not a bare count —
see CONCEPT:AU-AHE.evaluation.swallow-baseline-stable-key for why a count or a
line-number key rots: it can't tell "a new gap appeared" from "an old gap was
fixed and a different old gap remains", and it can be trivially satisfied by
fixing an unrelated entry). The accepted, already-known debt is frozen in
``scripts/concept_design_doc_baseline.txt`` (one concept id per line). A run:

* fails on any undocumented concept NOT in the baseline — this is what makes
  a *newly regressed* or *newly merged-without-a-doc* concept visible, forever,
  even after it reaches ``main``;
* never fails on a baselined (already-known) gap — that debt is tracked and
  cleaned up deliberately, not force-fixed by this gate;
* reports (informationally) any baseline entries that now have a doc, so the
  baseline can be shrunk with ``--update-baseline``.

Run ``--update-baseline`` to (re)freeze the current undocumented set — the
correct move immediately after fixing/retiring a batch of concepts, or when
deliberately accepting a new gap with a stated reason.

Exit codes: 0 = governance OK (or no new concepts), 1 = violation(s).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DESIGN_DIR = ROOT / ".specify" / "design"
MERGED_AUDIT_BASELINE = ROOT / "scripts" / "concept_design_doc_baseline.txt"

sys.path.insert(0, str(ROOT))
from agent_utilities.governance.concept_hierarchy import (  # noqa: E402
    OKF_MARKER_RE,
    is_valid_domain,
    load_slug_registry,
    parse_okf_id,
)

CONCEPT_RE = OKF_MARKER_RE


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


def _merge_base(ref: str) -> str | None:
    mb = _git("merge-base", ref, "HEAD")
    return mb or None


def resolve_base(explicit: str | None) -> str | None:
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

    best_mb = bases[0][1]
    for _ref, mb in bases[1:]:
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
            best_mb = mb
    return best_mb


def valid_slugs() -> set[str]:
    """Return the exact registered repository slug set."""
    return set(load_slug_registry().values())


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


def new_concepts(base: str) -> list[str]:
    diff = _git("diff", f"{base}...HEAD", "--unified=0")
    added: set[str] = set()
    for line in diff.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            added.update(CONCEPT_RE.findall(line))
    # Keep only concepts that did NOT already exist at the base (reject churn).
    return sorted(c for c in added if not _exists_at_base(c, base))


def has_design_doc(concept: str, design_dir: Path = DESIGN_DIR) -> bool:
    if not design_dir.is_dir():
        return False
    for md in design_dir.rglob("*.md"):
        try:
            if concept in md.read_text(encoding="utf-8", errors="ignore"):
                return True
        except OSError:
            continue
    return False


#: Directories never worth scanning for a marker (build/vcs/cache noise).
_SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", "build", "dist"}
#: File suffixes a ``CONCEPT:`` marker is legitimately written in — covers
#: every location the 39-concept D-RG2-2 backlog actually used: Python/Rust
#: source, prose docs (AGENTS.md, docs/architecture/*.md), YAML manifests,
#: and plain-text baseline/ledger files.
_MARKER_SUFFIXES = {".py", ".rs", ".md", ".yml", ".yaml", ".txt", ".toml", ".cfg"}


def all_registered_concepts(root: Path = ROOT) -> list[str]:
    """Every concept id with a literal ``CONCEPT:<id>`` marker anywhere in the
    repo tree — the honest, complete "what concepts exist today" universe.

    See the module docstring ("Merged-but-undocumented mode") for why this
    deliberately re-scans the tree instead of reading ``docs/concepts.yaml``:
    that registry is generated from ``agent_utilities/**`` only, and misses
    markers legitimately written in ``scripts/``, ``tests/``,
    ``mcp_v2_gateway/``, or prose docs.
    """
    found: set[str] = set()
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in _MARKER_SUFFIXES:
            continue
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        found.update(CONCEPT_RE.findall(content))
    return sorted(found)


def read_baseline(baseline: Path = MERGED_AUDIT_BASELINE) -> set[str]:
    if not baseline.exists():
        return set()
    return {
        line.strip()
        for line in baseline.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def write_baseline(ids: set[str], baseline: Path = MERGED_AUDIT_BASELINE) -> None:
    header = (
        "# Concept ids with NO design document, accepted as known/tracked debt.\n"
        "# Generated by `check_concept_governance.py --audit-merged --update-baseline`.\n"
        "# This is a RATCHET: the gate fails on any undocumented concept NOT listed\n"
        "# here (new or newly-merged debt), never on an entry already listed here.\n"
        "# Fix or retire an entry, then re-run --update-baseline to shrink this file —\n"
        "# never hand-add an entry to dodge a real violation.\n"
    )
    body = "\n".join(sorted(ids))
    baseline.write_text(f"{header}{body}\n" if body else header, encoding="utf-8")


def audit_merged(
    *,
    update: bool,
    scan_root: Path = ROOT,
    design_dir: Path = DESIGN_DIR,
    baseline_path: Path = MERGED_AUDIT_BASELINE,
) -> int:
    """Base-less mode: audit every live concept id, regardless of merge status."""
    all_ids = all_registered_concepts(scan_root)
    undocumented = {cid for cid in all_ids if not has_design_doc(cid, design_dir)}

    if update:
        write_baseline(undocumented, baseline_path)
        print(
            f"Baseline updated: {len(undocumented)} undocumented concept(s) frozen "
            f"in {baseline_path}."
        )
        return 0

    baseline = read_baseline(baseline_path)
    new_undocumented = sorted(undocumented - baseline)
    resolved = sorted(baseline - undocumented)
    stale = sorted(baseline - set(all_ids))  # baselined id no longer even exists

    print(
        f"Merged-concept audit: {len(all_ids)} live concept(s) discovered, "
        f"{len(undocumented)} without a design doc, {len(baseline)} baselined."
    )

    if resolved:
        print(
            f"\n{len(resolved)} baselined concept(s) now have a design doc "
            "(baseline can be shrunk with --update-baseline):"
        )
        for c in resolved:
            print(f"  + {c}")

    if stale:
        print(
            f"\n{len(stale)} baselined concept(s) no longer exist in code at all "
            "(retired marker — baseline can be shrunk with --update-baseline):"
        )
        for c in stale:
            print(f"  + {c}")

    if new_undocumented:
        print(
            f"\nFAIL: {len(new_undocumented)} concept(s) have NO design document "
            "and are not in the accepted baseline (new or newly-merged debt):"
        )
        for c in new_undocumented:
            print(f"  - {c}")
        print(
            "\nTo fix: create a design document in .specify/design/<feature>/ that "
            "references each concept, OR — if it is a genuinely mechanical marker "
            "with no design decision behind it — retire the marker instead. If this "
            "is deliberately-accepted debt, run --update-baseline with a stated reason."
        )
        return 1

    print("\nNo new merged-but-undocumented concepts. Governance check passed.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", help="explicit base ref to diff against (default: nearest trunk)")
    ap.add_argument(
        "--audit-merged",
        action="store_true",
        help=(
            "base-less mode: audit every live concept id in docs/concepts.yaml "
            "(not just the diff since the last trunk merge) against the design "
            "corpus, ratcheted against scripts/concept_design_doc_baseline.txt"
        ),
    )
    ap.add_argument(
        "--update-baseline",
        action="store_true",
        help="with --audit-merged, (re)freeze the current undocumented set as accepted debt",
    )
    args = ap.parse_args()

    if args.audit_merged:
        return audit_merged(update=args.update_baseline)
    if args.update_baseline:
        print("ERROR: --update-baseline only applies with --audit-merged", file=sys.stderr)
        return 2

    base = resolve_base(args.base)
    if not base:
        print("No base ref available; skipping concept governance (nothing to diff).")
        return 0

    concepts = new_concepts(base)
    if not concepts:
        print("No new CONCEPT: tags found. Governance check passed.")
        return 0

    slugs = valid_slugs()
    violations: list[str] = []
    for concept in concepts:
        parsed = parse_okf_id(concept)
        if not has_design_doc(concept):
            violations.append(f"  {concept} - No design document references this concept")
        if parsed.slug not in slugs:
            violations.append(
                f"  {concept} - Unregistered repository slug: {parsed.slug}"
            )
        if not is_valid_domain(parsed.pillar, parsed.domain):
            violations.append(
                f"  {concept} - Domain {parsed.domain!r} is not registered "
                f"for pillar {parsed.pillar}"
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
