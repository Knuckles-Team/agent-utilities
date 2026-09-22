#!/usr/bin/env python3
"""Extend-Before-Invent concept governance gate.

Local mirror of the ``concept-governance.yml`` CI workflow so that
``pre-commit run --config .config/pre-commit.yaml --all-files`` catches a governance violation BEFORE a push —
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
``CONCEPT:`` marker is legitimately written in ``scripts/``, ``tests/``, a
sibling package outside ``agent_utilities/`` (formerly ``mcp_v2_gateway/``,
retired under BUG-069 — see ``docs/architecture/mcp-2026-protocol-surface.md``),
and prose docs (``AGENTS.md``, ``docs/architecture/*.md``). Five of the 39
concepts this exact gate gap produced (D-RG2-2) live ONLY in such files —
using ``docs/concepts.yaml`` as the universe would have silently exempted
them from the audit this mode exists to provide, defeating its own purpose.
So this mode re-scans the tree directly with the same marker grammar
(``OKF_MARKER_RE``), the same way the diff-based mode's own
``_exists_at_base`` already searches the whole tree unrestricted by
directory — consistent with THIS gate's existing scope, not
``build_concepts_yaml.py``'s narrower one.

This mode originally froze the accepted-debt set into
``scripts/concept_design_doc_baseline.txt`` (one concept id per line) — a
ratchet keyed by concept id rather than a bare count (a count or line-number
key rots: it can't tell "a new gap appeared" from "an old gap was fixed and a
different old gap remains", and it can be trivially satisfied by fixing an
unrelated entry — see ``scripts/check_swallowed_errors.py``'s docstring for
the canonical example). That was still a ratchet, and this project does not
allow ratchets: a frozen debt file hides whether the real backlog is growing
or shrinking, and the only way to see the true number was to read the file
itself. Measured before retiring it (D-WD5-RAT-03): the baseline held 495
ids, essentially the whole live undocumented set — this mode had frozen
almost everything it ever found, not made a dent in it.

``--audit-merged`` is now purely an **unconditional census**: it prints the
full undocumented set on *every* run, pass or fail, nothing written to disk,
so the real number cannot go stale and cannot hide behind a file nobody
reads. It still enforces, unconditionally, the three invariants below that
are not ordinary debt but active contradictions in the governance registry
itself — a broken parent link, a revived retirement, a revived rename — the
same ``HARD_ZERO_SHAPES`` pattern the swallowed-error gate uses for a bare
``except:``. The bulk "no design doc yet" backlog is reported, never gated,
here: catching a concept BEFORE it merges without a doc is the diff-based
mode above's job (already non-ratchet, already diff-scoped against the
nearest trunk) — ``--audit-merged``'s job is to keep that backlog visible
forever, including after merge, not to force it to zero on every commit that
happens to run it.

``--update-baseline`` is retired (exits 2, writes nothing) — there is no
baseline left to freeze.

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
    raw_bases = [(r, _merge_base(r)) for r in candidates]
    # D-WD5-RAT-03 (pre-existing mypy debt, fixed in passing): an explicit
    # `is not None` guard (rather than a bare truthy filter) lets mypy narrow
    # `b: str | None` to `str` inside the comprehension -- the runtime
    # behavior (drop refs with no merge-base) is unchanged.
    bases: list[tuple[str, str]] = [(r, b) for r, b in raw_bases if b is not None]
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

    Anchored at ``ROOT`` (never at ``root`` itself) with pathspecs scoped to
    ``root``'s position under it -- see ``check_wiring._tracked_or_walked``'s
    docstring for the confirmed root cause (GOC-70 liveness investigation):
    git's ``ls-files`` output-path base silently reverts from
    "relative to ``-C``'s target" to "relative to the ambient work-tree
    root" whenever ``GIT_DIR``/``GIT_INDEX_FILE`` are already set in the
    process environment -- which git itself sets for every hook subprocess
    (``git commit``, incl. concluding a merge). When ``root`` is a
    subdirectory of ``ROOT`` (e.g. ``DESIGN_DIR``), that reverts the
    ``root / line`` reconstruction into a doubled, nonexistent path,
    silently returning an EMPTY file list -- which made ``has_design_doc``
    report "no design document references this concept" for a concept with
    a real, present design doc, every time this gate ran as an actual git
    hook (as opposed to a manual ``pre-commit run``). Anchoring at ``ROOT``
    (computed once via pure ``Path`` math, immune to git's ambient-env
    behavior) sidesteps the ambiguity entirely.
    """
    try:
        rel = root.relative_to(ROOT)
        anchor = ROOT
        prefix = "" if str(rel) == "." else f"{rel.as_posix()}/"
        pathspecs = [f"{prefix}{p}" for p in patterns] or [
            f"{prefix}." if prefix else "."
        ]
    except ValueError:
        # `root` is not under this repo's ROOT (e.g. a synthetic test
        # fixture) -- preserve the prior `-C root` behavior.
        anchor = root
        pathspecs = list(patterns) or ["."]
    try:
        out = subprocess.run(
            ["git", "-C", str(anchor), "ls-files", "--"] + pathspecs,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        tracked = [anchor / line for line in out.splitlines() if line]
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
    markers legitimately written in ``scripts/``, ``tests/``, or prose docs.
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


def undocumented_concepts(
    *,
    scan_root: Path = ROOT,
    design_dir: Path = DESIGN_DIR,
    lineage_path: str | None = None,
) -> tuple[set[str], list[str], list[str], Lineage]:
    """The LIVE undocumented-concept set, computed fresh off the tree.

    Returns ``(undocumented, all_ids, broken_links, lineage)``.

    Shared by :func:`audit_merged` (the gate) and
    ``scripts/concept_domain_triage.py`` (which used to read this off the
    frozen ``concept_design_doc_baseline.txt`` ratchet -- retired under
    D-WD5-RAT-03, see the module docstring). Computing it directly off the
    live tree, every call, is the whole point of retiring that baseline:
    there is no snapshot left to go stale, so every caller sees the same
    real number.
    """
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

    return undocumented, all_ids, broken_links, lineage


def _print_merged_census(
    *,
    all_ids: list[str],
    undocumented: set[str],
    covered_by_parent: list[str],
    lineage: Lineage,
) -> None:
    """The unconditional part: always printed, never fails on its own."""
    print(
        f"Merged-concept audit: {len(all_ids)} live concept(s) discovered "
        f"(unconditional census, no baseline), {len(undocumented)} without a "
        f"design doc, {len(covered_by_parent)} covered by a declared parent, "
        f"{len(lineage.retired)} retired, {len(lineage.renamed)} renamed."
    )
    if not undocumented:
        return
    print(
        f"\n{len(undocumented)} concept(s) with NO design document "
        "(real backlog -- informational only, not gated; fix or retire "
        "deliberately):"
    )
    for c in sorted(undocumented):
        print(f"  - {c}")


def _report_broken_links(broken_links: list[str]) -> bool:
    if not broken_links:
        return False
    print(
        f"\nFAIL: {len(broken_links)} broken parent link(s) in "
        "agent_utilities/governance/concept_lineage.yaml:"
    )
    for msg in broken_links:
        print(f"  - {msg}")
    return True


def _report_revived_retirements(revived: list[tuple[str, str]]) -> bool:
    if not revived:
        return False
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
    return True


def _report_revived_renames(respawned: list[tuple[str, str, str]]) -> bool:
    if not respawned:
        return False
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
    return True


def audit_merged(
    *,
    scan_root: Path = ROOT,
    design_dir: Path = DESIGN_DIR,
    lineage_path: str | None = None,
) -> int:
    """Base-less mode: audit every live concept id, regardless of merge status.

    No baseline any more (see the module docstring, "Merged-but-undocumented
    mode"). This is an UNCONDITIONAL CENSUS of every concept lacking a design
    doc, printed on every run whether it passes or fails -- nothing is
    written to disk, so the number cannot go stale. It fails only on the
    three ABSOLUTE invariants that were already unconditional and already at
    zero: a broken parent link, a revived retirement, a revived rename. The
    undocumented-concept backlog itself is reported, never gated, here --
    debt to burn down deliberately, not something one unrelated commit is
    forced to fix.
    """
    undocumented, all_ids, broken_links, lineage = undocumented_concepts(
        scan_root=scan_root, design_dir=design_dir, lineage_path=lineage_path
    )
    live = frozenset(all_ids)
    all_ids_set = set(all_ids)

    revived = reintroduced_retirements(live, lineage)
    respawned = reintroduced_renames(live, lineage)
    covered_by_parent = sorted(
        c for c in lineage.parents if c in all_ids_set and c not in undocumented
    )

    _print_merged_census(
        all_ids=all_ids,
        undocumented=undocumented,
        covered_by_parent=covered_by_parent,
        lineage=lineage,
    )

    # Broken pointers and revived retirements/renames fail unconditionally --
    # neither is ordinary debt, both are claims someone wrote in the lineage
    # registry that the tree now contradicts.
    failed = (
        _report_broken_links(broken_links)
        | _report_revived_retirements(revived)
        | _report_revived_renames(respawned)
    )
    if failed:
        return 1

    print(
        "\nNo governance-breaking concept found (undocumented backlog above, if "
        "any, is informational only)."
    )
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--base", help="explicit base ref to diff against (default: nearest trunk)"
    )
    ap.add_argument(
        "--audit-merged",
        action="store_true",
        help=(
            "base-less mode: audit every live concept id (not just the diff "
            "since the last trunk merge) against the design corpus -- an "
            "unconditional census, no baseline, see the module docstring"
        ),
    )
    ap.add_argument(
        "--update-baseline",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return ap


def _diff_based_violations(
    concepts: list[str], *, slugs: set[str], lineage: Lineage, live: frozenset[str]
) -> list[str]:
    """One violation line per problem found in a newly-introduced concept id."""
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
    return violations


def _report_diff_based_result(
    base: str, concepts: list[str], violations: list[str]
) -> int:
    if not violations:
        print(
            f"All {len(concepts)} new concept(s) have design documents. "
            "Governance check passed."
        )
        return 0
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


def main() -> int:
    args = _build_arg_parser().parse_args()

    if args.update_baseline:
        print(
            "--update-baseline is RETIRED. --audit-merged has no baseline: it "
            "prints the full undocumented census every run and enforces only "
            "the unconditional broken-link/revived-retirement/revived-rename "
            "invariants, so there is nothing to freeze. See the module "
            "docstring.",
            file=sys.stderr,
        )
        return 2

    if args.audit_merged:
        return audit_merged()

    base = resolve_base(args.base)
    if not base:
        print("No base ref available; skipping concept governance (nothing to diff).")
        return 0

    concepts = new_concepts(base)
    if not concepts:
        print("No new CONCEPT: tags found. Governance check passed.")
        return 0

    violations = _diff_based_violations(
        concepts,
        slugs=valid_slugs(),
        lineage=load_lineage(),
        live=frozenset(all_registered_concepts()),
    )
    return _report_diff_based_result(base, concepts, violations)


if __name__ == "__main__":
    raise SystemExit(main())
