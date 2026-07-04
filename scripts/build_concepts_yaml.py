#!/usr/bin/env python3
"""Mechanically derive docs/concepts.yaml from CONCEPT:<ID> markers in code.

This is the *reproducible generator* for the single source of truth
``docs/concepts.yaml``. It walks ``agent_utilities/`` for ``*.py`` and ``*.rs``
files, extracts every ``CONCEPT:<PILLAR>-<n>[.<m>]`` marker, and emits one entry
per unique concept id:

    {id, name, pillar, status: live, code_paths: [...], doc: "<one-line>"}

The pillar is derived mechanically from the id prefix (the ``<PILLAR>-<n>``
segment, e.g. ``ORCH-1``, ``KG-2``, ``AHE-3``). ``doc`` is a best-effort
one-liner taken from the nearest descriptive text attached to a marker.

Run:  python scripts/build_concepts_yaml.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "agent_utilities"
OUT_PATH = ROOT / "docs" / "concepts.yaml"

# Import the canonical marker grammar from the governance package so the
# generator, the validator (check_concepts.py), and the allocator never drift.
# Adding ROOT lets this resolve to the local source even without an install.
sys.path.insert(0, str(ROOT))
from agent_utilities.governance.concept_allocator import MARKER_RE  # noqa: E402
from agent_utilities.governance.concept_hierarchy import (  # noqa: E402
    observed_project_namespaces,
    parse_concept_id,
)

# Pillar is the leading "<LETTERS>-<digits>" segment of the id.
# OKF-CIS (CONCEPT:AU-OS.governance.concept-2): the pillar group of an id is its SLUG-PILLAR prefix
# (e.g. AU-KG, EG-KG), so concepts.yaml groups by owning-repo + global pillar.
PILLAR_RE = re.compile(r"^([A-Z]{2}-(?:ORCH|KG|AHE|ECO|OS|GBOT))")


def _clean_doc(rest: str) -> str:
    """Extract a best-effort one-line description from text after the marker."""
    text = rest.strip()
    # Common forms: " — Confidence-Gated Router", " - Adaptive ...",
    # ": Nested Subfolder Instructions", "] Adversarial ...".
    # Strip a single leading separator.
    text = re.sub(r"^[\s\)\]\}\.:,;—\-–]+", "", text)
    # Cut at characters that usually terminate a human-readable phrase.
    # Stop at code-ish punctuation that signals the prose has ended.
    for stop in ("(", ")", "]", "}", '"', "'", "`", "{", ":", "%", "#"):
        idx = text.find(stop)
        if idx != -1:
            text = text[:idx]
    text = text.strip(" .,—-–\t")
    return text


def collect() -> dict[str, dict]:
    concepts: dict[str, dict] = {}
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
        for m in MARKER_RE.finditer(content):
            cid = m.group("id")
            # The shared marker regex captures only the id; the trailing
            # descriptive text is the remainder of the same line.
            rest = content[m.end() :].split("\n", 1)[0]
            doc = _clean_doc(rest)
            pillar_match = PILLAR_RE.match(cid)
            pillar = pillar_match.group(1) if pillar_match else cid
            entry = concepts.setdefault(
                cid,
                {
                    "id": cid,
                    "name": "",
                    "pillar": pillar,
                    "status": "live",
                    "code_paths": set(),
                    "doc": "",
                },
            )
            entry["code_paths"].add(rel)
            # Prefer the longest, most descriptive doc string seen.
            if len(doc) > len(entry["doc"]):
                entry["doc"] = doc
    # Finalize: derive name from doc (or id), sort code_paths, and project the
    # 3-level hierarchy (CONCEPT:AU-OS.governance.concept-hierarchy-standardization / B5). The dotted/alias/canonical_pillar
    # keys are ADDITIVE — existing consumers that read only ``id`` are unaffected.
    observed = observed_project_namespaces(list(concepts))
    for cid, entry in concepts.items():
        entry["code_paths"] = sorted(entry["code_paths"])
        entry["name"] = entry["doc"] if entry["doc"] else cid
        if not entry["doc"]:
            entry["doc"] = cid
        try:
            parsed = parse_concept_id(cid, observed_project_ns=observed)
            entry["dotted"] = parsed.canonical
            entry["aliases"] = list(parsed.aliases)
            entry["canonical_pillar"] = (
                f"{parsed.namespace}-{parsed.pillar}"
                if parsed.is_project
                else parsed.namespace
            )
            entry["needs_curation"] = parsed.needs_curation
        except ValueError:
            entry["dotted"] = cid
            entry["aliases"] = [cid]
            entry["canonical_pillar"] = entry["pillar"]
            entry["needs_curation"] = False
    return concepts


def _concept_sort_key(cid: str):
    pillar_match = PILLAR_RE.match(cid)
    pillar = pillar_match.group(1) if pillar_match else cid
    nums = re.findall(r"\d+", cid)
    return (pillar, [int(n) for n in nums], cid)


def build() -> dict:
    concepts = collect()
    ordered = [concepts[cid] for cid in sorted(concepts, key=_concept_sort_key)]
    pillars = sorted({c["pillar"] for c in ordered})
    return {
        "generated_by": "scripts/build_concepts_yaml.py",
        "total_concepts": len(ordered),
        "total_pillars": len(pillars),
        "pillars": pillars,
        "concepts": ordered,
    }


def main() -> None:
    data = build()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as fh:
        fh.write(
            "# AUTO-GENERATED by scripts/build_concepts_yaml.py — DO NOT EDIT BY HAND.\n"
        )
        fh.write("# Single source of truth for canonical concepts, derived from\n")
        fh.write("# CONCEPT:<ID> markers in agent_utilities/ (*.py, *.rs).\n")
        yaml.safe_dump(
            data,
            fh,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
            width=100,
        )
    print(
        f"Wrote {OUT_PATH.relative_to(ROOT)}: "
        f"{data['total_concepts']} concepts across {data['total_pillars']} pillars."
    )
    # Self-clean the reservation ledger: any reserved id whose marker now appears
    # in code becomes 'landed', and stale (TTL-expired) reservations are freed.
    # Best-effort — never let ledger reconciliation break doc generation.
    try:
        from agent_utilities.governance.concept_allocator import reconcile

        result = reconcile(repo_root=ROOT)
        if result["landed"] or result["expired"]:
            print(
                f"Reconciled reservations: {len(result['landed'])} landed, "
                f"{len(result['expired'])} expired."
            )
    except Exception as exc:  # noqa: BLE001 - reconciliation is advisory
        print(f"(skipped reservation reconcile: {exc})")


if __name__ == "__main__":
    main()
