#!/usr/bin/env python3
"""OKF-CIS curation harness (CONCEPT:OS-5.77) — the stage between plan and apply.

``emit``  : read migration_plan.yaml, pass clean entries through, and split the
            needs-curation entries into per-pillar batch JSON files (collision
            siblings kept together, the closed domain vocab inlined) for LLM
            curator agents to resolve.
``merge`` : read the curator agents' output JSONs, assemble + VALIDATE the final
            curated_migration_plan.yaml (grammar + closed vocab + global
            uniqueness), and regenerate legacy_map.yaml. Reports anything that
            failed validation for a re-run.

The unit of reference is ``eid`` — the entry's stable index in the plan's
``entries`` array (collisions share an old_id, so old_id alone is not unique).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent_utilities.governance import concept_hierarchy as ch  # noqa: E402

BATCH_SIZE = 80


def _is_collision(e: dict) -> bool:
    return any("collision" in n for n in e.get("notes", []))


def emit(plan_path: Path, out_dir: Path) -> None:
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    entries = plan["entries"]
    vocab = ch.load_domain_vocab()
    out_dir.mkdir(parents=True, exist_ok=True)

    clean, need = [], []
    for eid, e in enumerate(entries):
        (need if e["needs_curation"] else clean).append((eid, e))

    # Pass clean entries straight through (still validated at merge).
    (out_dir / "clean.json").write_text(
        json.dumps([{"eid": eid, "new_id": e["new_id"]} for eid, e in clean], indent=1),
        encoding="utf-8",
    )

    # Group needs-curation by pillar, keeping collision siblings adjacent.
    by_pillar: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for eid, e in need:
        by_pillar[e["pillar"]].append((eid, e))

    manifest = []
    for pillar, items in sorted(by_pillar.items()):
        # order by old_id so a collision's siblings land in the same batch
        items.sort(key=lambda t: (t[1]["old_id"], t[0]))
        allowed = (
            {d: sigs for d, sigs in vocab.get(pillar, {}).items()}
            if pillar in ch.PILLARS
            else {}
        )
        for bi in range(0, len(items), BATCH_SIZE):
            chunk = items[bi : bi + BATCH_SIZE]
            bname = f"batch_{pillar}_{bi // BATCH_SIZE:02d}.json"
            batch = {
                "pillar": pillar,
                "batch": bname,
                "default_domains": sorted(allowed) or "PILLAR UNKNOWN — pick a pillar below",
                "domain_signals": allowed,
                # Full closed vocab so a curator can re-home a mis-pillared concept
                # (e.g. a CTX 'Nested Subfolder Instructions' -> OS.context). The
                # SLUG (provenance) is FIXED; pillar+domain are the curator's call.
                "full_vocab": {p: sorted(ds) for p, ds in vocab.items()},
                "entries": [
                    {
                        "eid": eid,
                        "old_id": e["old_id"],
                        "slug": e["slug"],
                        "collision_group": e["old_id"] if _is_collision(e) else None,
                        "draft_new_id": e["new_id"],
                        "doc": e["doc"][:160],
                        "files": e["files"][:3],
                        "notes": e.get("notes", []),
                    }
                    for eid, e in chunk
                ],
            }
            (out_dir / bname).write_text(json.dumps(batch, indent=1), encoding="utf-8")
            manifest.append({"batch": bname, "pillar": pillar, "count": len(chunk)})

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"emitted {len(manifest)} batches for {len(need)} needs-curation entries "
          f"({len(clean)} clean passed through) -> {out_dir}")
    for m in manifest:
        print(f"  {m['batch']:26} {m['count']:>4}")


def merge(plan_path: Path, curated_dir: Path, out_dir: Path) -> int:
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    entries = plan["entries"]

    # eid -> curated new_id, from clean.json + every curated_*.json the agents wrote
    assigned: dict[int, str] = {}
    for jf in sorted(curated_dir.glob("*.json")):
        if jf.name in {"manifest.json"} or jf.name.startswith("batch_"):
            continue
        data = json.loads(jf.read_text(encoding="utf-8"))
        rows = data if isinstance(data, list) else data.get("curated", [])
        for row in rows:
            if row.get("new_id"):
                assigned[int(row["eid"])] = row["new_id"].strip()

    curated, invalid, unresolved, dupes = [], [], [], []
    seen: dict[str, tuple[int, str]] = {}  # new_id -> (eid, old_id)
    for eid, e in enumerate(entries):
        new_id = assigned.get(eid) or (e["new_id"] if not e["needs_curation"] else None)
        if not new_id:
            unresolved.append(eid)
            continue
        try:
            parsed = ch.parse_okf_id(new_id)
        except ValueError as ex:
            invalid.append((eid, new_id, str(ex)))
            continue
        if not ch.is_valid_domain(parsed.pillar, parsed.domain):
            invalid.append((eid, new_id, f"domain {parsed.domain!r} not in vocab"))
            continue
        if new_id in seen:
            # Same new_id is OK only when it's the SAME legacy concept being
            # merged back (collision siblings the curator re-joined); a clash
            # across DIFFERENT old_ids is a real uniqueness violation.
            if seen[new_id][1] != e["old_id"]:
                dupes.append((eid, new_id, seen[new_id][0]))
            continue
        seen[new_id] = (eid, e["old_id"])
        curated.append({"eid": eid, "old_id": e["old_id"], "new_id": new_id,
                        "files": e["files"], "slug": parsed.slug,
                        "pillar": parsed.pillar, "domain": parsed.domain})

    out_dir.mkdir(parents=True, exist_ok=True)
    simple: dict[str, str] = {}
    collisions: dict[str, list[dict]] = defaultdict(list)
    by_old: dict[str, list[dict]] = defaultdict(list)
    for c in curated:
        by_old[c["old_id"]].append(c)
    for old, cs in by_old.items():
        if len(cs) == 1:
            simple[old] = cs[0]["new_id"]
        else:
            for c in cs:
                for f in c["files"]:
                    collisions[old].append({"file": f, "new": c["new_id"]})

    result = {
        "generated_by": "scripts/curate_batches.py merge",
        "summary": {
            "curated_total": len(curated),
            "unresolved": len(unresolved),
            "invalid": len(invalid),
            "duplicates": len(dupes),
            "distinct_domains": len({(c["pillar"], c["domain"]) for c in curated}),
        },
        "invalid": [{"eid": e, "new_id": n, "why": w} for e, n, w in invalid],
        "duplicate_ids": [{"eid": e, "new_id": n, "clashes_with_eid": o} for e, n, o in dupes],
        "unresolved_eids": unresolved,
        "entries": curated,
    }
    (out_dir / "curated_migration_plan.yaml").write_text(
        yaml.safe_dump(result, sort_keys=False, width=100), encoding="utf-8"
    )
    (out_dir / "legacy_map.yaml").write_text(
        yaml.safe_dump({"version": 1, "generated_by": "curate_batches.py",
                        "simple": dict(sorted(simple.items())),
                        "collisions": {k: v for k, v in sorted(collisions.items())},
                        "unmapped": [entries[e]["old_id"] for e in unresolved]},
                       sort_keys=False, width=100),
        encoding="utf-8",
    )
    s = result["summary"]
    print("merged curated plan ->", out_dir)
    for k, v in s.items():
        print(f"  {k}: {v}")
    if invalid or dupes:
        print("  ⚠ re-run needed for invalid/duplicate entries (see curated_migration_plan.yaml)")
    return 0 if not (invalid or dupes) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pe = sub.add_parser("emit")
    pe.add_argument("--plan", required=True)
    pe.add_argument("--out", required=True)
    pm = sub.add_parser("merge")
    pm.add_argument("--plan", required=True)
    pm.add_argument("--curated", required=True)
    pm.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "emit":
        emit(Path(args.plan), Path(args.out))
        return 0
    return merge(Path(args.plan), Path(args.curated), Path(args.out))


if __name__ == "__main__":
    raise SystemExit(main())
