#!/usr/bin/env python3
"""Migrate flat concept ids → the 3-level ``NS-<pillar>.<concept>.<segment>`` grammar.

CONCEPT:AU-OS.governance.concept-hierarchy-standardization — concept-hierarchy standardization (B5).

Scans ALL of ``agent-packages/**`` for ``CONCEPT:<id>`` markers (and any
``docs/concepts.yaml``), computes the canonical dotted id + permanent flat alias
for each via :mod:`agent_utilities.governance.concept_hierarchy`, and reports the
full flat→dotted mapping.

``--dry-run`` (DEFAULT)
    Read-only. Writes ``reports/w4-concept-hierarchy-dryrun.md``: per-project
    marker counts, the flat→dotted table, derived partOf edges, and every
    ambiguous / needs-curation / unmappable id flagged. Touches NO source file.

``--apply`` (NOT run as part of the proposal)
    Rewrite every ``CONCEPT:<flat>`` marker in place to its canonical dotted id,
    recording the flat id as an alias comment (``# alias:<flat>``). Idempotent
    (re-running is a no-op once markers are canonical) and reversible (the flat
    alias is preserved and the resolver accepts both forms forever).

Usage::

    python scripts/migrate_concepts_hierarchy.py                 # dry-run
    python scripts/migrate_concepts_hierarchy.py --root <path>   # custom scan root
    python scripts/migrate_concepts_hierarchy.py --apply         # cutover (gated)
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Resolve the governance helper from local source without an install.
_THIS = Path(__file__).resolve()
REPO_ROOT = _THIS.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agent_utilities.governance.concept_hierarchy import (  # noqa: E402
    HIERARCHY_MARKER_RE,
    ConceptId,
    derive_part_of_edges,
    observed_project_namespaces,
    parse_concept_id,
)


# Default fleet scan root = the canonical agent-packages workspace. This repo may
# live in a worktree OUTSIDE agent-packages, so prefer an ancestor named
# ``agent-packages``; otherwise fall back to the known canonical workspace path.
def _default_scan_root() -> Path:
    for parent in REPO_ROOT.parents:
        if parent.name == "agent-packages":
            return parent
    canonical = Path("/home/apps/workspace/agent-packages")
    if canonical.exists():
        return canonical
    return REPO_ROOT.parent


DEFAULT_SCAN_ROOT = _default_scan_root()
REPORT_PATH = REPO_ROOT / "reports" / "w4-concept-hierarchy-dryrun.md"

_SCAN_SUFFIXES = (".py", ".rs")
_SKIP_DIRS = {"__pycache__", ".git", "node_modules", ".venv", "venv", ".mypy_cache"}


def _iter_source_files(root: Path):
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in _SCAN_SUFFIXES:
            continue
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        yield path


def _project_of(path: Path, root: Path) -> str:
    """First path component under *root* = the owning project/repo."""
    try:
        rel = path.relative_to(root)
    except ValueError:
        return "<external>"
    return rel.parts[0] if rel.parts else "<root>"


def scan(root: Path) -> dict[str, dict]:
    """Return ``{raw_id: {"files": set, "projects": set}}`` across the fleet."""
    hits: dict[str, dict] = defaultdict(lambda: {"files": set(), "projects": set()})
    for path in _iter_source_files(root):
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        proj = _project_of(path, root)
        for m in HIERARCHY_MARKER_RE.finditer(content):
            cid = m.group("id")
            hits[cid]["files"].add(path.as_posix())
            hits[cid]["projects"].add(proj)
    return hits


def build_mapping(hits: dict[str, dict]) -> tuple[list[ConceptId], list[str]]:
    """Parse every raw id → (parsed list, list of unmappable raw ids)."""
    observed = observed_project_namespaces(list(hits))
    parsed: list[ConceptId] = []
    unmappable: list[str] = []
    for cid in sorted(hits):
        try:
            parsed.append(parse_concept_id(cid, observed_project_ns=observed))
        except ValueError:
            unmappable.append(cid)
    return parsed, unmappable


def _project_counts(hits: dict[str, dict]) -> Counter:
    counts: Counter = Counter()
    for meta in hits.values():
        for proj in meta["projects"]:
            counts[proj] += len(meta["files"])
    return counts


def render_report(
    root: Path,
    hits: dict[str, dict],
    parsed: list[ConceptId],
    unmappable: list[str],
) -> str:
    total_markers = sum(len(m["files"]) for m in hits.values())
    clean = [p for p in parsed if not p.flags]
    needs_curation = [p for p in parsed if p.needs_curation]
    package = [p for p in parsed if not p.is_project]
    already = [p for p in parsed if p.is_project and p.canonical == p.raw]
    rewritten = [p for p in parsed if p.is_project and p.canonical != p.raw]

    proj_counts = _project_counts(hits)
    part_of = derive_part_of_edges(parsed)

    lines: list[str] = []
    w = lines.append
    w("# W4 / B5 — Concept-Hierarchy Migration (DRY-RUN)")
    w("")
    w("> Generated by `scripts/migrate_concepts_hierarchy.py --dry-run`.")
    w("> **Read-only.** No source file was modified. This is the reviewable mapping")
    w("> that an `--apply` cutover would perform (keeping flat ids as aliases).")
    w("")
    w("## Summary")
    w("")
    w(f"- Scan root: `{root}`")
    w(f"- Total `CONCEPT:` markers scanned: **{total_markers}**")
    w(f"- Unique concept ids: **{len(parsed) + len(unmappable)}**")
    w(f"- Projects/repos with markers: **{len(proj_counts)}**")
    w(f"- Already grammar-compliant (project, no rewrite): **{len(already)}**")
    w(f"- Would be rewritten to canonical dotted (project): **{len(rewritten)}**")
    w(f"- Package-scoped (separate scheme, passthrough): **{len(package)}**")
    w(f"- Clean mappings (no flags): **{len(clean)}**")
    w(f"- Needs curation (legacy pillar 0): **{len(needs_curation)}**")
    w(f"- Unmappable (not a well-formed id): **{len(unmappable)}**")
    w("")
    w("## Per-project marker counts")
    w("")
    w("| Project | Markers |")
    w("|---------|---------|")
    for proj, n in proj_counts.most_common():
        w(f"| `{proj}` | {n} |")
    w("")

    w("## Flat → dotted mapping (project namespaces)")
    w("")
    w("Rows where **Canonical** differs from **Raw** are what `--apply` rewrites.")
    w("The **Raw** column stays a permanently-valid alias either way.")
    w("")
    w(
        "| Raw id | Canonical dotted | Namespace | Pillar | Concept | Segment | Flags | Projects |"
    )
    w(
        "|--------|------------------|-----------|--------|---------|---------|-------|----------|"
    )
    for p in parsed:
        if not p.is_project:
            continue
        seg = p.segment if p.segment is not None else "—"
        flags = ", ".join(p.flags) if p.flags else "—"
        projs = ", ".join(sorted(hits[p.raw]["projects"]))
        w(
            f"| `{p.raw}` | `{p.canonical}` | {p.namespace} | {p.pillar} | "
            f"{p.concept} | {seg} | {flags} | {projs} |"
        )
    w("")

    if needs_curation:
        w(
            "## Needs curation — legacy pillar `0` (supply `PILLAR_MAP` before `--apply`)"
        )
        w("")
        w("These flat project ids carry no pillar and default to the reserved legacy")
        w("pillar `0`. A reviewer populates `PILLAR_MAP[(NS, concept)] = pillar` in")
        w("`agent_utilities/governance/concept_hierarchy.py` to assign real pillars.")
        w("")
        by_ns: dict[str, list[str]] = defaultdict(list)
        for p in needs_curation:
            by_ns[p.namespace].append(p.raw)
        for ns in sorted(by_ns):
            ids = ", ".join(f"`{i}`" for i in sorted(by_ns[ns]))
            w(f"- **{ns}** ({len(by_ns[ns])}): {ids}")
        w("")

    w("## Package-scoped ids (separate scheme — NOT migrated)")
    w("")
    w("Letters-only local registries keep their `PKG-NNN` form untouched.")
    w("")
    pkg_ns: dict[str, int] = Counter(p.namespace for p in package)
    for ns in sorted(pkg_ns):
        w(f"- **{ns}**: {pkg_ns[ns]} concept id(s)")
    w("")

    if unmappable:
        w("## Unmappable ids (flagged — manual review)")
        w("")
        for cid in unmappable:
            w(f"- `{cid}`")
        w("")

    w("## Derived `partOf` edges (concept → pillar → namespace)")
    w("")
    w(f"Total edges: **{len(part_of)}**. First 60 shown:")
    w("")
    w("| Part | Whole |")
    w("|------|-------|")
    for child, parent in part_of[:60]:
        w(f"| `{child}` | `{parent}` |")
    if len(part_of) > 60:
        w("")
        w(f"_… and {len(part_of) - 60} more._")
    w("")

    w("## To apply (NOT done here)")
    w("")
    w("1. Populate `PILLAR_MAP` in `concept_hierarchy.py` for the needs-curation ids.")
    w(
        "2. Per repo: `python scripts/migrate_concepts_hierarchy.py --apply --root <repo>`."
    )
    w("3. Re-run the gates: `check_concepts`, `check_ontology`, `ruff`.")
    w("4. Land per-repo (each repo owns its own markers + ledger).")
    w("")
    return "\n".join(lines) + "\n"


def apply_rewrite(root: Path) -> int:
    """Rewrite markers in place to canonical dotted form. Returns files changed."""
    hits = scan(root)
    observed = observed_project_namespaces(list(hits))
    # Build raw → canonical for project ids whose canonical differs.
    remap: dict[str, str] = {}
    for cid in hits:
        try:
            p = parse_concept_id(cid, observed_project_ns=observed)
        except ValueError:
            continue
        if p.is_project and p.canonical != p.raw:
            remap[cid] = p.canonical
    if not remap:
        print("apply: nothing to rewrite (already canonical).")
        return 0
    changed = 0
    for path in _iter_source_files(root):
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        new = content
        for raw, canon in remap.items():
            # Only rewrite the marker occurrence, preserving the flat alias.
            new = new.replace(f"CONCEPT:{raw}", f"CONCEPT:{canon}  # alias:{raw}")
        if new != content:
            path.write_text(new, encoding="utf-8")
            changed += 1
    print(f"apply: rewrote markers in {changed} file(s); {len(remap)} id(s) remapped.")
    return changed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        default=str(DEFAULT_SCAN_ROOT),
        help="scan root (default: the agent-packages workspace)",
    )
    ap.add_argument(
        "--apply",
        action="store_true",
        help="REWRITE markers to canonical dotted form (default: dry-run only)",
    )
    ap.add_argument(
        "--report",
        default=str(REPORT_PATH),
        help="dry-run report output path",
    )
    args = ap.parse_args()
    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        print(f"scan root does not exist: {root}", file=sys.stderr)
        return 1

    if args.apply:
        apply_rewrite(root)
        return 0

    hits = scan(root)
    parsed, unmappable = build_mapping(hits)
    report = render_report(root, hits, parsed, unmappable)
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    total = sum(len(m["files"]) for m in hits.values())
    print(
        f"dry-run: scanned {total} markers / {len(parsed) + len(unmappable)} unique "
        f"ids across {root}. Report → {out.relative_to(REPO_ROOT) if out.is_relative_to(REPO_ROOT) else out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
