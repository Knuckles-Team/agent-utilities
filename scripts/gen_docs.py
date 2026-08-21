#!/usr/bin/env python3
"""Regenerate the README.md concept block from docs/concepts.yaml.

The authoritative concept count lives in ``docs/concepts.yaml`` (produced by
``scripts/build_concepts_yaml.py``). This script renders that data into
README.md between the markers::

    <!-- BEGIN GENERATED: concepts -->
    ... generated count line + compact 5-pillar table ...
    <!-- END GENERATED: concepts -->

The table is intentionally scoped to the **5 pillars agent-utilities itself
owns** (AU-ORCH/AU-KG/AU-AHE/AU-ECO/AU-OS) — one row each, matching
``docs/pillars/{1..5}_*.md``. The remaining 4 pillars (EG-AHE/EG-KG/EG-ORCH/
EG-OS) belong to the epistemic-graph engine's own pillar set and are noted,
not tabulated, here; the full per-concept breakdown across all 9 pillars
stays in ``docs/concepts.yaml`` / ``docs/status.md``.

Modes:
  --write   Rewrite the generated block in README.md in place.
  --check   Exit non-zero if README.md differs from a fresh generation.

Output is deterministic (rows are in fixed pillar order).
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

# Single source of the concept-total computation — shared with
# gen_agents_md.py so the README, AGENTS.md, and concepts.yaml's own header
# can never independently drift from each other again.
sys.path.insert(0, str(ROOT))
from agent_utilities.governance.concept_hierarchy import total_concept_count  # noqa: E402

BEGIN = "<!-- BEGIN GENERATED: concepts -->"
END = "<!-- END GENERATED: concepts -->"

# The 5 pillars agent-utilities itself owns, in canonical pillar-number
# order, with the doc each one links to and a one-line focus blurb.
# (pillar_prefix, number, name, doc_path, focus)
AU_PILLARS: list[tuple[str, int, str, str, str]] = [
    (
        "AU-ORCH",
        1,
        "Graph Orchestration",
        "docs/pillars/1_graph_orchestration.md",
        "Planning, SDD lifecycle, dynamic multi-layer execution",
    ),
    (
        "AU-KG",
        2,
        "Epistemic Knowledge Graph",
        "docs/pillars/2_epistemic_knowledge_graph.md",
        "The one engine authority — ingestion, ontology, ETL, reasoning",
    ),
    (
        "AU-AHE",
        3,
        "Agentic Harness Engineering",
        "docs/pillars/3_agentic_harness_engineering.md",
        "Self-models, evaluation, governed self-evolution",
    ),
    (
        "AU-ECO",
        4,
        "Ecosystem & Peripherals",
        "docs/pillars/4_ecosystem_peripherals.md",
        "MCP fleet, messaging, connectors, UI surfaces",
    ),
    (
        "AU-OS",
        5,
        "Agent OS Infrastructure",
        "docs/pillars/5_agent_os_infrastructure.md",
        "Auth, governance, deployment, scaling",
    ),
]


def load_concepts() -> dict:
    with CONCEPTS_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def render_block(data: dict) -> str:
    concepts = data["concepts"]
    total = total_concept_count(CONCEPTS_PATH)

    # Group concepts by pillar (all 9 — AU's 5 + epistemic-graph's 4).
    by_pillar: dict[str, list[dict]] = {}
    for c in concepts:
        by_pillar.setdefault(c["pillar"], []).append(c)
    pillar_count = len(by_pillar)

    au_total = sum(len(by_pillar.get(prefix, [])) for prefix, *_ in AU_PILLARS)
    other_pillars = pillar_count - len(AU_PILLARS)
    other_total = total - au_total

    lines: list[str] = []
    lines.append(BEGIN)
    lines.append("")
    # NOTE: the exact phrase "**N canonical concepts** across **M pillars**"
    # is asserted verbatim by tests/docs/test_docs_consistency.py — keep this
    # line's wording/markup unchanged even when editing the rest of the block.
    lines.append(
        f"Synthesized from concept markers in the codebase into "
        f"**{total} canonical concepts** across **{pillar_count} pillars**."
    )
    lines.append("")
    lines.append(
        "> This count is generated from `docs/concepts.yaml` by "
        "`scripts/gen_docs.py` — do not edit by hand. The table below covers "
        f"the 5 pillars agent-utilities itself owns; the other {other_pillars} "
        f"({other_total} concepts) belong to the epistemic-graph engine's own "
        "pillar set. Live per-pillar status: [docs/status.md](docs/status.md)."
    )
    lines.append("")
    lines.append("| # | Pillar | Focus | Concepts | Docs |")
    lines.append("|:-:|:-------|:------|:--------:|:-----|")
    for prefix, num, name, path, focus in AU_PILLARS:
        count = len(by_pillar.get(prefix, []))
        lines.append(f"| {num} | {name} | {focus} | {count} | [{path}]({path}) |")
    lines.append("")
    lines.append(END)
    return "\n".join(lines)


def generate_readme(current: str, data: dict) -> str:
    block = render_block(data)
    if BEGIN in current and END in current:
        pattern = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END), re.DOTALL)
        return pattern.sub(lambda _m: block, current)
    raise SystemExit(
        "ERROR: README.md is missing the generated-block markers "
        f"{BEGIN!r} / {END!r}. Insert them around the concept table first."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--check", action="store_true", help="verify README is up to date"
    )
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
