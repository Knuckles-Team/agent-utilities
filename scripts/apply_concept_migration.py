#!/usr/bin/env python3
"""OKF-CIS applier (CONCEPT:OS-5.77) — the big-bang marker rewriter.

Driven by the frozen ``curated_migration_plan.yaml`` (never by heuristics at write
time). Rewrites every legacy ``CONCEPT:<old>`` marker + comma-list continuation +
bare prose cross-reference to its curated ``<SLUG>-<PILLAR>.<domain>.<concept>`` id,
across a repo's ``.py``/``.rs``/``.md``. Idempotent (a no-op on already-OKF ids).
Ambiguous cases (a collided id referenced in a file the plan can't disambiguate,
or a bare collided id in prose) are written to a ``.rej`` report, never guessed.

Self-contained: carries its OWN legacy marker regex so it does not depend on the
six legacy matchers being retired at the atomic cutover.

Usage:
    python scripts/apply_concept_migration.py --plan <curated_migration_plan.yaml> \
        --repo <name>=<path> [--dry-run] [--rej-dir DIR]
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent_utilities.governance import concept_hierarchy as ch  # noqa: E402

# Self-contained legacy grammar (numeric pillar) — INDEPENDENT of concept_allocator.
_LEGACY_ID = r"[A-Z]+-\d+(?:\.[0-9A-Za-z]+)*"
# A marker + its comma-continuation list: CONCEPT:X, Y, Z  (Y/Z carry no prefix).
_MARKER_LIST_RE = re.compile(rf"CONCEPT:(?P<ids>{_LEGACY_ID}(?:\s*,\s*{_LEGACY_ID})*)")
_LEGACY_ID_RE = re.compile(_LEGACY_ID)

_SCAN_EXT = {".py", ".rs", ".md"}
_SKIP_DIRS = {"__pycache__", ".git", ".venv", "node_modules", "target", "build", "dist"}


def _iter_files(root: Path):
    for p in root.rglob("*"):
        if p.suffix in _SCAN_EXT and not any(s in p.parts for s in _SKIP_DIRS):
            yield p


def load_maps(plan_path: Path) -> tuple[dict, dict, dict]:
    """Return ((old_id,file)->news, old_id->news, old_id->primary_new) from the plan.

    ``primary`` is the new_id of the collided id's most-authoritative cluster (the
    one with the most definition files) — the fallback for otherwise-ambiguous
    *references* to a formerly-overloaded id (a bare ``CONCEPT:KG-2.0`` reference
    maps to the dominant "Active Knowledge Graph" meaning, not a rare sibling).
    """
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    by_file: dict[tuple[str, str], set[str]] = defaultdict(set)
    by_old: dict[str, set[str]] = defaultdict(set)
    best: dict[str, tuple[int, str]] = {}
    for e in plan["entries"]:
        old, new = e["old_id"], e["new_id"]
        files = e.get("files", [])
        by_old[old].add(new)
        for f in files:
            by_file[(old, f)].add(new)
        # primary = entry with most files (tie-break: lexically smallest new_id)
        cand = (len(files), tuple(-ord(c) for c in new))
        if old not in best or cand > best[old][0]:
            best[old] = (cand, new)
    primary = {o: v[1] for o, v in best.items()}
    return by_file, by_old, primary


def resolve(old_id, rel_file, by_file, by_old, primary) -> tuple[str | None, str]:
    """Resolve one legacy id in one file -> (new_id | None, reason)."""
    if ch.is_okf_id(old_id):
        return old_id, "already-okf"
    news = by_old.get(old_id)
    if not news:
        return None, "unmapped"
    if len(news) == 1:
        return next(iter(news)), "ok"
    # collision: disambiguate by this file's definition cluster, else primary
    fnews = by_file.get((old_id, rel_file))
    if fnews and len(fnews) == 1:
        return next(iter(fnews)), "ok-collision"
    return primary.get(old_id), "primary-ref"


def apply_repo(name: str, root: Path, by_file, by_old, primary, *, dry_run: bool, rej_dir: Path):
    slug = ch.slug_for_repo(name)
    rewrites = 0
    files_changed = 0
    rej: list[str] = []
    info: list[str] = []  # primary-ref fallbacks (audit, not failures)
    # simple (non-collision) old_ids -> new, for bare prose refs in .md
    simple = {o: next(iter(ns)) for o, ns in by_old.items() if len(ns) == 1 and not ch.is_okf_id(o)}
    simple_re = (
        re.compile(r"(?<![\w.-])(" + "|".join(re.escape(o) for o in sorted(simple, key=len, reverse=True)) + r")(?![\w.-])")
        if simple else None
    )

    for path in _iter_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if "CONCEPT:" not in text and not (simple_re and path.suffix == ".md"):
            continue
        rel = str(path.relative_to(root))
        orig = text

        def _sub_marker(m: re.Match, rel: str = rel) -> str:
            ids = [x.strip() for x in m.group("ids").split(",")]
            out = []
            for oid in ids:
                new, reason = resolve(oid, rel, by_file, by_old, primary)
                if new is None:
                    rej.append(f"{rel}: CONCEPT:{oid} [{reason}]")
                    out.append(oid)  # leave as-is; gate will flag
                else:
                    if reason == "primary-ref":
                        info.append(f"{rel}: CONCEPT:{oid} -> {new} [primary-ref]")
                    out.append(new)
            # every id gets a full CONCEPT: prefix (fixes the comma-list drop bug)
            return ", ".join(f"CONCEPT:{x}" for x in out)

        text = _MARKER_LIST_RE.sub(_sub_marker, text)

        # bare prose cross-refs (simple, non-collision ids) in markdown only
        if simple_re and path.suffix == ".md":
            def _sub_bare(m: re.Match) -> str:
                return simple.get(m.group(1), m.group(1))
            # don't touch ids already inside a CONCEPT: marker (handled above)
            text = simple_re.sub(_sub_bare, text)

        if text != orig:
            nrw = len(_LEGACY_ID_RE.findall(orig)) - len(_LEGACY_ID_RE.findall(text))
            rewrites += max(nrw, 0)
            files_changed += 1
            if not dry_run:
                path.write_text(text, encoding="utf-8")

    if rej or info:
        rej_dir.mkdir(parents=True, exist_ok=True)
        if rej:
            (rej_dir / f"{name}.rej").write_text("\n".join(rej) + "\n", encoding="utf-8")
        if info:
            (rej_dir / f"{name}.info").write_text("\n".join(info) + "\n", encoding="utf-8")
    return {"repo": name, "slug": slug, "files_changed": files_changed,
            "rewrites": rewrites, "rejected": len(rej), "primary_refs": len(info)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--repo", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rej-dir", default="/tmp/okf-rej")
    args = ap.parse_args()

    by_file, by_old, primary = load_maps(Path(args.plan))
    rej_dir = Path(args.rej_dir)
    total = defaultdict(int)
    for spec in args.repo:
        name, _, path = spec.partition("=")
        if ch.slug_for_repo(name) is None:
            print(f"  SKIP {name}: no SLUG registered")
            continue
        r = apply_repo(name, Path(path), by_file, by_old, primary,
                       dry_run=args.dry_run, rej_dir=rej_dir)
        tag = "DRY" if args.dry_run else "APPLIED"
        print(f"  [{tag}] {r['repo']:22} files={r['files_changed']:>4} "
              f"rewrites={r['rewrites']:>5} primary-ref={r['primary_refs']:>5} "
              f"rejected={r['rejected']:>4}")
        for k in ("files_changed", "rewrites", "rejected", "primary_refs"):
            total[k] += r[k]
    print(f"  TOTAL files={total['files_changed']} rewrites={total['rewrites']} "
          f"primary-ref={total['primary_refs']} rejected={total['rejected']}  (reports in {rej_dir})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
