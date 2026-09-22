#!/usr/bin/env python3
"""Regenerate the honesty-first status page (the "Codex") at docs/status.md.

This is the fix for the concept-count drift bug: README.md, AGENTS.md, and
docs/concepts.yaml's own header used to each report a different total (1216 /
1203 / 1196) because two generator scripts independently recomputed it. This
page joins ``scripts/gen_docs.py`` / ``scripts/gen_agents_md.py`` in deriving
the total the same, deliberately simple way (``len(concepts.yaml's
"concepts" list)`` — no per-generator filtering to drift) and is the one
place that count — and the per-pillar breakdown behind it — is rendered for
a reader, so nothing downstream needs to hand-state a number again.

Sources (never hand-typed):

* ``docs/concepts.yaml`` — the canonical concept registry. Every entry in it
  already has code (its own docstring: reserved/retired concepts are removed
  from the registry outright, not carried with a status flag), so every row
  here counts as ``LIVE``.
* ``docs/concept_reservations.yaml`` — the concept-ID reservation ledger.
  Entries with ``status: reserved`` are IDs allocated ahead of code and count
  as ``RESERVED``; ``landed``/``expired`` entries are historical and are not
  counted (a landed reservation's concept is already counted via
  ``concepts.yaml``; an expired one was never built).

``BUILDING``/``ROADMAP``/``RETIRED`` are part of the shared vocabulary this
page defines (see docs/status.md's "Status vocabulary" section) but this
repo's registry model does not carry per-concept partial-build state today —
a concept is either reserved or fully live. Their counts are honestly
reported as 0 rather than invented.

Usage::

    python scripts/build_status_page.py --write
    python scripts/build_status_page.py --check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONCEPTS_PATH = ROOT / "docs" / "concepts.yaml"
RESERVATIONS_PATH = ROOT / "docs" / "concept_reservations.yaml"
STATUS_PATH = ROOT / "docs" / "status.md"

# Structural pillar -> owning doc subtree -> primary enforcing CI/pre-commit
# gate. Not named humans -- see docs/status.md's "Domain ownership" section.
PILLAR_LABEL = {
    "AU-AHE": "Agentic Harness Engineering",
    "AU-ECO": "Ecosystem & Peripherals",
    "AU-KG": "Epistemic Knowledge Graph",
    "AU-ORCH": "Graph Orchestration",
    "AU-OS": "Agent OS Infrastructure",
    "EG-AHE": "epistemic-graph: harness-facing concepts",
    "EG-KG": "epistemic-graph: knowledge-graph engine concepts",
    "EG-ORCH": "epistemic-graph: routing/orchestration concepts",
    "EG-OS": "epistemic-graph: deployment concepts",
}
PILLAR_SUBTREE = {
    "AU-AHE": "docs/pillars/3_agentic_harness_engineering.md",
    "AU-ECO": "docs/pillars/4_ecosystem_peripherals.md",
    "AU-KG": "docs/pillars/2_epistemic_knowledge_graph/",
    "AU-ORCH": "docs/pillars/1_graph_orchestration.md",
    "AU-OS": "docs/pillars/5_agent_os_infrastructure.md",
    "EG-AHE": "docs/architecture/ (epistemic-graph engine integration)",
    "EG-KG": "docs/architecture/ (epistemic-graph engine integration)",
    "EG-ORCH": "docs/architecture/ (epistemic-graph engine integration)",
    "EG-OS": "docs/architecture/ (epistemic-graph engine integration)",
}
PILLAR_GATE = {
    "AU-AHE": "`scripts/check_concepts.py` + `scripts/check_eval_corpus.py`",
    "AU-ECO": "`scripts/check_concepts.py` + `scripts/check_skill_name_collision.py`",
    "AU-KG": "`scripts/check_concepts.py` + GraphSchema authority cutover gates",
    "AU-ORCH": "`scripts/check_concepts.py` + `scripts/check_coupling.py`",
    "AU-OS": "`scripts/check_concepts.py` + `scripts/check_genesis_manifest.py`",
    "EG-AHE": "`scripts/check_concepts.py` (marker registration) + epistemic-graph's `scripts/check_documentation_contract.py`",
    "EG-KG": "`scripts/check_concepts.py` (marker registration) + epistemic-graph's `scripts/check_documentation_contract.py`",
    "EG-ORCH": "`scripts/check_concepts.py` (marker registration) + epistemic-graph's `scripts/check_documentation_contract.py`",
    "EG-OS": "`scripts/check_concepts.py` (marker registration) + epistemic-graph's `scripts/check_documentation_contract.py`",
}

HONESTY_FRAMING = (
    "**Honesty first.** This page reflects the repository's actual current "
    "state — every number below is computed from `docs/concepts.yaml` and "
    "`docs/concept_reservations.yaml` at generation time, never hand-typed. "
    "If a claim elsewhere in this repo's docs disagrees with a number here, "
    "this page is generated more recently and is the one to trust."
)

VOCAB_ROWS = [
    (
        "RESERVED",
        "",
        "A concept ID is allocated in the reservations ledger; no code implements it yet.",
    ),
    (
        "BUILDING",
        "🔶",
        "Partially implemented; unsupported paths fail honestly rather than silently.",
    ),
    ("LIVE", "✅", "Implemented and present on `main`."),
    ("ROADMAP", "🗺", "Designed, no committed date."),
    ("RETIRED", "", "Formerly live, intentionally removed."),
]


def _load_concepts() -> dict:
    return yaml.safe_load(CONCEPTS_PATH.read_text(encoding="utf-8")) or {}


def _load_reservations() -> list[dict]:
    if not RESERVATIONS_PATH.is_file():
        return []
    return yaml.safe_load(RESERVATIONS_PATH.read_text(encoding="utf-8")) or []


def _pillar_counts() -> tuple[dict[str, int], dict[str, int], int]:
    """Return (live_by_pillar, reserved_by_pillar, total_pillars_seen)."""
    data = _load_concepts()
    live: dict[str, int] = {}
    for concept in data.get("concepts", []):
        pillar = concept["pillar"]
        live[pillar] = live.get(pillar, 0) + 1

    reserved: dict[str, int] = {}
    for entry in _load_reservations():
        if entry.get("status") != "reserved":
            continue
        pillar = f"{entry['slug']}-{entry['pillar']}"
        reserved[pillar] = reserved.get(pillar, 0) + 1

    all_pillars = set(live) | set(reserved) | set(PILLAR_LABEL)
    return live, reserved, len(all_pillars)


def render() -> str:
    live, reserved, _ = _pillar_counts()
    total_live = sum(live.values())
    total_reserved = sum(reserved.values())
    pillars = sorted(set(live) | set(reserved) | set(PILLAR_LABEL))

    lines: list[str] = []
    lines.append("# Status — the Codex")
    lines.append("")
    lines.append(
        "> **Generated — do not edit by hand.** Produced by "
        "`scripts/build_status_page.py` from `docs/concepts.yaml` and "
        "`docs/concept_reservations.yaml`. See "
        '"How this page stays honest" at the bottom.'
    )
    lines.append("")
    lines.append(HONESTY_FRAMING)
    lines.append("")

    lines.append("## Concepts by pillar × status")
    lines.append("")
    lines.append(
        f"**{total_live} LIVE** concepts (every entry in `docs/concepts.yaml` — "
        "the registry only carries concepts with shipped code) and "
        f"**{total_reserved} RESERVED** concept IDs (open, unexpired entries "
        "in `docs/concept_reservations.yaml`) across "
        f"**{len(pillars)} pillars**."
    )
    lines.append("")
    lines.append("| Pillar | LIVE ✅ | RESERVED | BUILDING 🔶 | ROADMAP 🗺 | RETIRED |")
    lines.append("|:------|---:|---:|---:|---:|---:|")
    for pillar in pillars:
        lines.append(
            f"| **{pillar}** — {PILLAR_LABEL.get(pillar, pillar)} "
            f"| {live.get(pillar, 0)} | {reserved.get(pillar, 0)} | 0 | 0 | 0 |"
        )
    lines.append(f"| **Total** | {total_live} | {total_reserved} | 0 | 0 | 0 |")
    lines.append("")
    lines.append(
        "> `BUILDING`/`ROADMAP`/`RETIRED` are always 0 here: "
        "`docs/concepts.yaml`'s registry model does not carry a per-concept "
        "partial-build state (its own generator strips reserved/retired IDs "
        "out entirely rather than flagging them) — a concept is either "
        "`RESERVED` or fully `LIVE`. The three statuses are still defined "
        "below because they are part of the one vocabulary this page and "
        "epistemic-graph's status page share."
    )
    lines.append("")

    lines.append("## Status vocabulary")
    lines.append("")
    lines.append(
        "Defined once, here — every other table in this repo's docs "
        "(README capability tables, `docs/capabilities.md`) should link to "
        "this section instead of restating or omitting it. Existing emoji "
        "are the rendering of this vocabulary, not a separate scheme."
    )
    lines.append("")
    lines.append("| Status | Emoji | Meaning |")
    lines.append("|:------|:---:|:------|")
    for name, emoji, meaning in VOCAB_ROWS:
        lines.append(f"| `{name}` | {emoji} | {meaning} |")
    lines.append("")
    lines.append(
        "Lifecycle: `RESERVED` → `BUILDING` → `LIVE` → (optionally) `RETIRED`, "
        "with `ROADMAP` as the not-yet-reserved intent stage."
    )
    lines.append("")

    lines.append("## Domain ownership")
    lines.append("")
    lines.append(
        "Structural ownership — which doc subtree and which CI/pre-commit "
        "gate is authoritative for each pillar. Not named humans."
    )
    lines.append("")
    lines.append("| Pillar | Owning doc subtree | Primary gate |")
    lines.append("|:------|:------|:------|")
    for pillar in pillars:
        subtree = PILLAR_SUBTREE.get(pillar, "—")
        gate = PILLAR_GATE.get(pillar, "`scripts/check_concepts.py`")
        lines.append(f"| **{pillar}** | `{subtree}` | {gate} |")
    lines.append("")
    lines.append(
        "`EG-*` pillar concepts are markers found in *this* repo's code that "
        "tag a capability of the `epistemic-graph` engine this repo drives "
        "(cross-repo concept federation, see `docs/concept_coordination.md`); "
        "the engine's own implementation and its `EG-P0-1` generated ledger "
        "are owned by the `epistemic-graph` repo, whose "
        "`docs/status.md` is the authoritative status page for that side."
    )
    lines.append("")

    lines.append("## How this page stays honest")
    lines.append("")
    lines.append(
        "This page is produced by `scripts/build_status_page.py` from "
        "`docs/concepts.yaml` and `docs/concept_reservations.yaml` — never "
        "hand-typed. Regenerate it with:"
    )
    lines.append("")
    lines.append("```bash")
    lines.append("python scripts/build_status_page.py --write")
    lines.append("```")
    lines.append("")
    lines.append(
        "`scripts/check_status_page.py` is wired into `.github/workflows/"
        "advisory.yml` (report-only, `continue-on-error: true`, matching "
        "this repo's existing approved-vs-enforced convention) and fails "
        "loudly — without blocking a release — the moment this file drifts "
        "from `docs/concepts.yaml` / `docs/concept_reservations.yaml`. Run it "
        "locally with:"
    )
    lines.append("")
    lines.append("```bash")
    lines.append("python scripts/check_status_page.py")
    lines.append("```")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--write", action="store_true", help="write docs/status.md in place"
    )
    group.add_argument(
        "--check", action="store_true", help="exit non-zero if docs/status.md is stale"
    )
    args = parser.parse_args()

    rendered = render()

    if args.write:
        STATUS_PATH.write_text(rendered, encoding="utf-8")
        print(f"wrote {STATUS_PATH.relative_to(ROOT)}")
        return 0

    if not STATUS_PATH.is_file():
        print("docs/status.md is missing — run --write first.", file=sys.stderr)
        return 1
    current = STATUS_PATH.read_text(encoding="utf-8")
    if current != rendered:
        print(
            "docs/status.md is stale relative to docs/concepts.yaml / "
            "docs/concept_reservations.yaml. Run: "
            "python scripts/build_status_page.py --write",
            file=sys.stderr,
        )
        return 1
    print("docs/status.md is up to date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
