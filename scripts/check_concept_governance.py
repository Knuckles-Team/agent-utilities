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

Parent-satisfied documentation (CONCEPT:AU-OS.governance.concept-lineage-parent-doc)
------------------------------------------------------------------------------------
The rule above — "every marker needs a design doc" — silently assumes every
marker names its own architectural decision. Auditing ``AU-KG.compute`` (79
concepts) found that assumption false by roughly a factor of four: sixteen of
those markers were per-connector declarations realising ONE decision, four were
per-surface markers realising ONE decision to drop numpy/scipy, and eleven named
no decision at all. Demanding 79 documents there demands ~58 restatements, and a
restatement is *worse* than no document: it satisfies the gate, looks like
documentation, and teaches nothing.

So a concept may instead declare a **parent** in
``agent_utilities/governance/concept_lineage.yaml`` — "the decision I realise is
documented over there". ``has_design_doc(child) or has_design_doc(parent)``
counts as documented, but only after the parent is verified to genuinely own a
document and to be a live concept. A parent link into an undocumented or dead
parent is reported as a **violation by name**, never as a silent pass — burying
a real decision behind a wrong pointer is the one way this mechanism could do
harm, so it is the one thing the gate refuses to be quiet about.

The same registry records **retirements** (markers deleted because they never
named a decision — e.g. an id auto-derived from a prose fragment). Re-introducing
a retired id is a failure: that is what makes retirement a ratchet rather than a
deletion someone silently undoes.

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
    is_valid_domain,
    iter_okf_markers,
    load_slug_registry,
    parse_okf_id,
)
from agent_utilities.governance.concept_lineage import (  # noqa: E402
    Lineage,
    load_lineage,
)


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
            added.update(marker.id for marker in iter_okf_markers(line))
    # Keep only concepts that did NOT already exist at the base (reject churn).
    return sorted(c for c in added if not _exists_at_base(c, base))


def _tracked_or_walked_files(root: Path, *patterns: str) -> list[Path]:
    """Files under ``root``, preferring the git-tracked set (BUG-043).

    A raw ``rglob`` also picks up gitignored, generated build output, which
    can carry a stale ``CONCEPT:`` marker/design-doc reference no longer in
    real source. Falls back to a filesystem walk only when ``root`` is not
    inside a git working tree (e.g. a synthetic test fixture).
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--"] + (list(patterns) or ["."]),
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        tracked = [root / line for line in out.splitlines() if line]
        if tracked:
            return [p for p in tracked if p.is_file()]
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    if patterns:
        results: list[Path] = []
        for pattern in patterns:
            results.extend(root.rglob(pattern))
        return results
    return list(root.rglob("*"))


def has_design_doc(concept: str, design_dir: Path = DESIGN_DIR) -> bool:
    if not design_dir.is_dir():
        return False
    for md in _tracked_or_walked_files(design_dir, "*.md"):
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
    for path in _tracked_or_walked_files(root):
        if not path.is_file() or path.suffix not in _MARKER_SUFFIXES:
            continue
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        found.update(marker.id for marker in iter_okf_markers(content))
    return sorted(found)


def resolve_documentation(
    concept: str,
    *,
    lineage: Lineage,
    design_dir: Path = DESIGN_DIR,
    live_ids: frozenset[str] | None = None,
) -> tuple[bool, str | None]:
    """Is *concept* documented, and is its parent link (if any) sound?

    Returns ``(documented, broken_link_reason)``. A broken link is reported
    separately from "undocumented" because it is a *harder* failure: it is not
    pre-existing debt the baseline may excuse, it is a pointer someone wrote
    that leads nowhere, and its whole risk is that it reads like coverage.
    """
    if has_design_doc(concept, design_dir):
        return True, None

    parent = lineage.parent_of(concept)
    if parent is None:
        return False, None

    if live_ids is not None and parent not in live_ids:
        return False, (
            f"{concept} declares parent {parent}, which is not a live concept "
            "(no CONCEPT: marker anywhere in the tree) — the pointer leads nowhere"
        )
    if not has_design_doc(parent, design_dir):
        return False, (
            f"{concept} declares parent {parent}, which has NO design document — "
            "a parent link may only point at a documented decision, otherwise it "
            "buries the child instead of covering it"
        )
    return True, None


def reintroduced_retirements(
    live_ids: frozenset[str], lineage: Lineage
) -> list[tuple[str, str]]:
    """Retired concept ids that have a live marker again, with their reason."""
    return sorted(
        (cid, lineage.retired[cid].reason) for cid in lineage.retired if cid in live_ids
    )


def reintroduced_renames(
    live_ids: frozenset[str], lineage: Lineage
) -> list[tuple[str, str, str]]:
    """OLD (pre-rename) ids that have a live marker again, with their new id
    and reason. Mirrors :func:`reintroduced_retirements` — a rename is a
    ratchet too, just one where the decision moved instead of vanishing."""
    return sorted(
        (cid, lineage.renamed[cid].to, lineage.renamed[cid].reason)
        for cid in lineage.renamed
        if cid in live_ids
    )


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
    lineage_path: str | None = None,
) -> int:
    """Base-less mode: audit every live concept id, regardless of merge status."""
    all_ids = all_registered_concepts(scan_root)
    live = frozenset(all_ids)
    lineage = load_lineage(lineage_path)

    undocumented: set[str] = set()
    broken_links: list[str] = []
    for cid in all_ids:
        documented, broken = resolve_documentation(
            cid, lineage=lineage, design_dir=design_dir, live_ids=live
        )
        if broken:
            broken_links.append(broken)
        if not documented:
            undocumented.add(cid)

    revived = reintroduced_retirements(live, lineage)
    respawned = reintroduced_renames(live, lineage)

    if update:
        # Refuse to freeze a baseline while a pointer is broken: --update-baseline
        # is the "accept this as known debt" button, and a broken parent link is
        # not debt, it is a claim of coverage that is false. Laundering it into
        # the baseline is exactly the failure mode the id-keyed ratchet exists to
        # prevent.
        if broken_links or revived or respawned:
            print(
                "REFUSING to update the baseline while the lineage registry is "
                "inconsistent — fix these first:",
                file=sys.stderr,
            )
            for msg in broken_links:
                print(f"  - {msg}", file=sys.stderr)
            for cid, reason in revived:
                print(
                    f"  - {cid} was retired ({reason}) but has a live marker again",
                    file=sys.stderr,
                )
            for cid, new_id, reason in respawned:
                print(
                    f"  - {cid} was renamed to {new_id} ({reason}) but has a live "
                    "marker again under the old id",
                    file=sys.stderr,
                )
            return 1
        write_baseline(undocumented, baseline_path)
        print(
            f"Baseline updated: {len(undocumented)} undocumented concept(s) frozen "
            f"in {baseline_path}."
        )
        return 0

    baseline = read_baseline(baseline_path)
    new_undocumented = sorted(undocumented - baseline)
    all_ids_set = set(all_ids)
    stale = sorted(baseline - all_ids_set)  # baselined id no longer even exists
    # "Resolved" means genuinely fixed: still exists AND now has a doc — NOT
    # merely absent from `undocumented`, which a retired (no-longer-existing)
    # marker also satisfies. Without excluding `stale` here, a removed marker
    # would double-report as both "now documented" and "no longer exists".
    resolved = sorted((baseline - undocumented) & all_ids_set)

    covered_by_parent = sorted(
        c for c in lineage.parents if c in all_ids_set and c not in undocumented
    )

    print(
        f"Merged-concept audit: {len(all_ids)} live concept(s) discovered, "
        f"{len(undocumented)} without a design doc, {len(baseline)} baselined, "
        f"{len(covered_by_parent)} covered by a declared parent, "
        f"{len(lineage.retired)} retired, {len(lineage.renamed)} renamed."
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

    failed = False

    # Broken pointers and revived retirements fail unconditionally — the baseline
    # never excuses them, because neither is pre-existing debt: both are claims
    # someone wrote in this registry that the tree contradicts.
    if broken_links:
        print(
            f"\nFAIL: {len(broken_links)} broken parent link(s) in "
            "agent_utilities/governance/concept_lineage.yaml:"
        )
        for msg in broken_links:
            print(f"  - {msg}")
        failed = True

    if revived:
        print(
            f"\nFAIL: {len(revived)} deliberately-retired concept id(s) have a live "
            "marker again:"
        )
        for cid, reason in revived:
            print(f"  - {cid} — retired because: {reason}")
        print(
            "  Either delete the re-introduced marker, or (if the decision is real "
            "now) drop the retirement entry and give the concept a design document."
        )
        failed = True

    if respawned:
        print(
            f"\nFAIL: {len(respawned)} renamed concept id(s) have a live marker "
            "again under the OLD id:"
        )
        for cid, new_id, reason in respawned:
            print(f"  - {cid} — renamed to {new_id} because: {reason}")
        print(
            "  Use the new id — the decision was deliberately moved, not deleted. "
            "If the old name genuinely needs to come back too, that is a new "
            "governance call, not a silent revert."
        )
        failed = True

    if new_undocumented:
        print(
            f"\nFAIL: {len(new_undocumented)} concept(s) have NO design document "
            "and are not in the accepted baseline (new or newly-merged debt):"
        )
        for c in new_undocumented:
            print(f"  - {c}")
        print(
            "\nTo fix, pick the one that is true:\n"
            "  * it IS a decision  -> write a design document in .specify/design/<feature>/\n"
            "  * it REALISES a decision documented elsewhere -> declare a parent in\n"
            "    agent_utilities/governance/concept_lineage.yaml (one doc, N pointers)\n"
            "  * it names NO decision -> retire the marker and record the retirement\n"
            "  `scripts/concept_domain_triage.py propose <PILLAR>.<domain>` proposes\n"
            "  which of the three each concept is, with evidence."
        )
        failed = True

    if failed:
        return 1

    print("\nNo new merged-but-undocumented concepts. Governance check passed.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--base", help="explicit base ref to diff against (default: nearest trunk)"
    )
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
        print(
            "ERROR: --update-baseline only applies with --audit-merged", file=sys.stderr
        )
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
    lineage = load_lineage()
    live = frozenset(all_registered_concepts())
    violations: list[str] = []
    for concept in concepts:
        parsed = parse_okf_id(concept)
        documented, broken = resolve_documentation(
            concept, lineage=lineage, live_ids=live
        )
        if broken:
            violations.append(f"  {broken}")
        elif not documented:
            violations.append(
                f"  {concept} - No design document references this concept, and it "
                "declares no parent concept whose document covers it"
            )
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
            "references each new CONCEPT tag (see .specify/design/_template.md), or — "
            "if the marker realises a decision that is already documented — declare "
            "that decision as its parent in "
            "agent_utilities/governance/concept_lineage.yaml."
        )
        return 1

    print(
        f"All {len(concepts)} new concept(s) have design documents. Governance check passed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
