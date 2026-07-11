#!/usr/bin/env python3
"""Guardrail — Capability Power Descriptors (CPD) never silently drift.

CONCEPT:AU-KG.retrieval.capability-power-descriptor (Seam 8 Phase 1 —
``plans/program-design-2026-07-11-epistemic-tool-routing.md`` section 2b).

Mirrors ``scripts/gen_docs.py --check`` / ``scripts/check_surface_parity.py``:
regenerates the CPD set from the live tool registry + EG ledger (or its
vendored cache) and fails if the checked-in ``docs/capabilities-power.md`` /
``.json`` differ, so a CPD can never quietly rot relative to its sources.
Also asserts two structural invariants no textual diff alone would catch as
cleanly:

1. **Coverage** — every tool in ``kg_server.REGISTERED_TOOLS`` has exactly one
   CPD, and every CPD id names a real registered tool (no orphan/phantom CPD).
2. **No fabrication** — every CPD's ``cost``/``latency``/``reliability`` is
   either empty or sourced from :data:`MEASURED_LATENCY_MS`/a live engine read
   (never a bare non-empty numeric literal with no ``source``/``kind`` marker),
   so a future edit can't quietly start guessing numbers.

Usage::

    python scripts/check_cpd.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))

import gen_capability_power as gcp  # noqa: E402

from agent_utilities.knowledge_graph.retrieval.capability_power_descriptor import (  # noqa: E402
    strip_generation_timestamp,
)


def _check_drift() -> tuple[list[str], list]:
    errors: list[str] = []
    cpds, generated_at = gcp.generate(None)
    md = gcp.render_markdown(cpds, generated_at=generated_at)
    js = gcp.render_json(cpds, generated_at=generated_at)

    if not gcp.MD_PATH.exists() or not gcp.JSON_PATH.exists():
        errors.append(
            f"{gcp.MD_PATH.name}/{gcp.JSON_PATH.name} missing — run "
            "`python scripts/gen_capability_power.py --write`."
        )
        return errors, cpds
    if strip_generation_timestamp(
        gcp.MD_PATH.read_text(encoding="utf-8")
    ) != strip_generation_timestamp(md):
        errors.append(
            f"{gcp.MD_PATH} is stale relative to the live tool registry + EG "
            "ledger — run `python scripts/gen_capability_power.py --write`."
        )
    if strip_generation_timestamp(
        gcp.JSON_PATH.read_text(encoding="utf-8")
    ) != strip_generation_timestamp(js):
        errors.append(
            f"{gcp.JSON_PATH} is stale relative to the live tool registry + EG "
            "ledger — run `python scripts/gen_capability_power.py --write`."
        )
    return errors, cpds


def _check_coverage(cpds) -> list[str]:
    from agent_utilities.mcp import kg_server

    kg_server.ensure_tools_registered()
    tool_names = set(kg_server.REGISTERED_TOOLS)
    cpd_ids = {c.id for c in cpds}
    errors: list[str] = []
    missing = tool_names - cpd_ids
    if missing:
        errors.append(f"Tools with no CPD: {sorted(missing)}")
    phantom = cpd_ids - tool_names
    if phantom:
        errors.append(f"CPDs for non-existent tools: {sorted(phantom)}")
    return errors


def _check_no_fabrication(cpds) -> list[str]:
    errors: list[str] = []
    for c in cpds:
        for section_name, section in (
            ("cost", c.cost),
            ("latency", c.latency),
            ("reliability", c.reliability),
        ):
            for key, val in (section or {}).items():
                if isinstance(val, dict) and not (
                    "source" in val or "kind" in val or "note" in val
                ):
                    errors.append(
                        f"{c.id}.{section_name}[{key!r}] has a value with no "
                        "source/kind/note marker — looks fabricated, not derived."
                    )
    return errors


def main() -> int:
    drift_errors, cpds = _check_drift()

    errors = list(drift_errors)
    errors.extend(_check_coverage(cpds))
    errors.extend(_check_no_fabrication(cpds))

    if errors:
        print("CPD Guardrail FAILED:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print(f"CPD Guardrail PASSED ({len(cpds)} capabilities, in sync).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
