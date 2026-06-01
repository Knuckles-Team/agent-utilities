#!/usr/bin/env python3
"""Regenerate the README.md concept block from docs/concepts.yaml.

The authoritative concept count and per-pillar table live in
``docs/concepts.yaml`` (produced by ``scripts/build_concepts_yaml.py``). This
script renders that data into README.md between the markers::

    <!-- BEGIN GENERATED: concepts -->
    ... generated count line + pillar table ...
    <!-- END GENERATED: concepts -->

Modes:
  --write   Rewrite the generated block in README.md in place.
  --check   Exit non-zero if README.md differs from a fresh generation.

Output is deterministic (pillars and rows are sorted).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONCEPTS_PATH = ROOT / "docs" / "concepts.yaml"
README_PATH = ROOT / "README.md"

BEGIN = "<!-- BEGIN GENERATED: concepts -->"
END = "<!-- END GENERATED: concepts -->"

# Human-readable focus blurbs per pillar prefix. Falls back to a derived
# default for any pillar not listed here.
PILLAR_LABELS = {
    "ORCH-1": "Graph Orchestration",
    "ORCH-2": "Orchestration Extensions",
    "ORCH-5": "Orchestration Runtime",
    "KG-1": "Knowledge Graph Core",
    "KG-2": "Epistemic Knowledge Graph",
    "AHE-3": "Agentic Harness Engineering",
    "ECO-4": "Ecosystem & Peripherals",
    "OS-5": "Agent OS Infrastructure",
    "GBOT-6": "GeniusBot Cockpit",
    "CTX-1": "Context Management",
    "LGC-1": "Logic & Governance Core",
    "SAFE-1": "Safety & Guardrails",
    "UTIL-1": "Shared Utilities",
}

NUM_RE = re.compile(r"\d+")


def load_concepts() -> dict:
    with CONCEPTS_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _id_sort_key(cid: str):
    nums = [int(n) for n in NUM_RE.findall(cid)]
    return (nums, cid)


def render_block(data: dict) -> str:
    concepts = data["concepts"]
    total = len(concepts)

    # Group concepts by pillar.
    by_pillar: dict[str, list[dict]] = {}
    for c in concepts:
        by_pillar.setdefault(c["pillar"], []).append(c)
    pillars = sorted(by_pillar)
    pillar_count = len(pillars)

    lines: list[str] = []
    lines.append(BEGIN)
    lines.append("")
    lines.append(
        f"Synthesized from concept markers in the codebase into "
        f"**{total} canonical concepts** across **{pillar_count} pillars**."
    )
    lines.append("")
    lines.append(
        "> This count and the table below are generated from "
        "`docs/concepts.yaml` by `scripts/gen_docs.py`. Do not edit by hand."
    )
    lines.append("")
    lines.append("| Pillar | ID Range | Count | Focus |")
    lines.append("|:------|:---------|:---:|:------|")
    for pillar in pillars:
        members = sorted(by_pillar[pillar], key=lambda c: _id_sort_key(c["id"]))
        ids = [m["id"] for m in members]
        if len(ids) == 1:
            id_range = ids[0]
        else:
            id_range = f"{ids[0]} – {ids[-1]}"
        label = PILLAR_LABELS.get(pillar, pillar)
        # Build a focus blurb from the distinct concept names (deduped, trimmed).
        seen: list[str] = []
        for m in members:
            name = (m.get("name") or m["id"]).strip()
            if name and name not in seen and name != m["id"]:
                seen.append(name)
        focus = ", ".join(seen[:8]) if seen else label
        lines.append(
            f"| **{pillar}** {label} | {id_range} | {len(members)} | {focus} |"
        )
    lines.append("")
    lines.append(END)
    return "\n".join(lines)


def generate_readme(current: str, data: dict) -> str:
    block = render_block(data)
    if BEGIN in current and END in current:
        pattern = re.compile(
            re.escape(BEGIN) + r".*?" + re.escape(END), re.DOTALL
        )
        return pattern.sub(lambda _m: block, current)
    raise SystemExit(
        "ERROR: README.md is missing the generated-block markers "
        f"{BEGIN!r} / {END!r}. Insert them around the concept table first."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="verify README is up to date")
    group.add_argument("--write", action="store_true", help="rewrite README in place")
    args = parser.parse_args()

    data = load_concepts()
    current = README_PATH.read_text(encoding="utf-8")
    updated = generate_readme(current, data)

    if args.write:
        if updated != current:
            README_PATH.write_text(updated, encoding="utf-8")
            print("README.md updated.")
        else:
            print("README.md already up to date.")
        return 0

    # --check
    if updated != current:
        print(
            "README.md is OUT OF DATE with docs/concepts.yaml. "
            "Run `python scripts/gen_docs.py --write`.",
            file=sys.stderr,
        )
        return 1
    print("README.md concept block is up to date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
