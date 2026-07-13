#!/usr/bin/env python3
"""Fleet-wide ``a2a.json`` epistemic capability enrichment (WS-4 item 6).

**The gap this closes.** ``reports/surpass-6mo/04-five-intersections.md`` §1
found that per-fleet-agent ``a2a.json`` manifests are thin, generic
templates — e.g. ``agents/scholarx/a2a.json`` advertises exactly one
boilerplate ``run_graph_flow`` capability, with no epistemic descriptor —
even though ``agent_utilities/server/app.py`` (CONCEPT:AU-KB-CURRENCY, commit
``389d7f19``) already advertises a real ``epistemic-answer`` skill on every
LIVE ``AgentCard`` served by ``agent_to_a2a``. The static ``a2a.json`` files
that ship in each ``agents/<pkg>`` checkout (consumed by ARD/A2A discovery
BEFORE a connection is even made — see ``ecosystem/ard_registry.py``) never
caught up: the discovery-time manifest and the live runtime card disagree.

This script closes that gap deterministically, no LLM calls, no network: it
walks a fleet root (default ``agents/``) for ``a2a.json`` files and appends
the SAME epistemic capability descriptor ``server/app.py`` already adds to
the live AgentCard — idempotent (a re-run is a no-op once a file has it),
additive-only (never removes or reorders an existing capability), and
byte-stable (canonical JSON with 2-space indent + trailing newline, so a
second run over already-enriched files diffs empty).

Usage:
    python3 scripts/enrich_fleet_a2a_epistemic.py --agents-root ../agents          # dry-run (default)
    python3 scripts/enrich_fleet_a2a_epistemic.py --agents-root ../agents --apply  # write changes
    python3 scripts/enrich_fleet_a2a_epistemic.py --file agents/scholarx/a2a.json --apply
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: The SAME capability content as the ``Skill(id="epistemic-answer", ...)``
#: appended to every live AgentCard in ``agent_utilities/server/app.py``
#: (CONCEPT:AU-KB-CURRENCY) — kept in lockstep by hand (both are short and
#: reviewed together on change) rather than importing pydantic-ai's ``Skill``
#: type here, so this script has zero runtime dependency on agent-utilities
#: itself and can enrich manifests for packages that don't even depend on it.
EPISTEMIC_CAPABILITY: dict[str, Any] = {
    "id": "epistemic-answer",
    "name": "Epistemic Answer",
    "description": (
        "Answers epistemic_status/why/what_changed queries over the shared "
        "knowledge graph: calibrated confidence, evidence/source citations, "
        "belief justification trees, bitemporal valid/tx history, and "
        "policy-redaction-aware provenance."
    ),
    "tags": ["epistemic", "provenance", "confidence", "kg"],
}

CAPABILITY_ID = str(EPISTEMIC_CAPABILITY["id"])


def already_enriched(manifest: dict[str, Any]) -> bool:
    """Is ``CAPABILITY_ID`` already present in ``manifest["capabilities"]``?"""
    for cap in manifest.get("capabilities") or []:
        if isinstance(cap, dict) and cap.get("id") == CAPABILITY_ID:
            return True
    return False


def enrich_manifest(manifest: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Return ``(possibly-updated manifest, changed)``.

    Additive only: appends the epistemic capability to the existing
    ``capabilities`` list (creating one if absent) when not already present.
    Never touches any other field.
    """
    if already_enriched(manifest):
        return manifest, False
    updated = dict(manifest)
    capabilities = list(updated.get("capabilities") or [])
    capabilities.append(dict(EPISTEMIC_CAPABILITY))
    updated["capabilities"] = capabilities
    return updated, True


def _dumps(manifest: dict[str, Any]) -> str:
    return json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"


def process_file(path: Path, *, apply: bool) -> str:
    """Process one ``a2a.json``. Returns a one-line status for reporting."""
    try:
        original_text = path.read_text(encoding="utf-8")
        manifest = json.loads(original_text)
    except (OSError, json.JSONDecodeError) as exc:
        return f"SKIP  {path} (unreadable/invalid JSON: {exc})"
    if not isinstance(manifest, dict):
        return f"SKIP  {path} (not a JSON object)"

    updated, changed = enrich_manifest(manifest)
    if not changed:
        return f"OK    {path} (already enriched)"

    if apply:
        path.write_text(_dumps(updated), encoding="utf-8")
        return f"WROTE {path}"
    return f"WOULD-WRITE {path}"


def find_a2a_manifests(agents_root: Path) -> list[Path]:
    return sorted(agents_root.glob("*/a2a.json"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agents-root",
        type=Path,
        default=None,
        help="Directory containing one subdirectory per fleet agent, each "
        "with an a2a.json (e.g. agent-packages/agents). Mutually exclusive "
        "with --file.",
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Enrich exactly one a2a.json (mutually exclusive with --agents-root).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write changes. Without this flag, the script only reports what "
        "it would change (dry-run, the default).",
    )
    args = parser.parse_args(argv)

    if bool(args.agents_root) == bool(args.file):
        parser.error("pass exactly one of --agents-root or --file")

    targets = [args.file] if args.file else find_a2a_manifests(args.agents_root)
    if not targets:
        print(f"no a2a.json files found under {args.agents_root}")
        return 0

    would_change = 0
    for target in targets:
        line = process_file(target, apply=args.apply)
        print(line)
        if line.startswith("WOULD-WRITE") or line.startswith("WROTE"):
            would_change += 1

    print(
        f"\n{len(targets)} manifest(s) scanned, {would_change} "
        f"{'updated' if args.apply else 'would be updated (pass --apply to write)'}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
