#!/usr/bin/env python3
"""Guardrail — Capability Power Descriptors (CPD) never silently drift.

CONCEPT:AU-KG.retrieval.capability-power-descriptor (Seam 8 Phase 1 —
``plans/program-design-2026-07-11-epistemic-tool-routing.md`` section 2b).

Mirrors ``scripts/gen_docs.py --check`` / ``scripts/check_surface_parity.py``:
regenerates the CPD set from the live tool registry + EG ledger (or its
vendored cache) and fails if the checked-in ``docs/capabilities-power.md`` /
``.json`` (and the packaged ``agent_utilities/knowledge_graph/retrieval/
capabilities-power.json`` catalog) differ, so a CPD can never quietly rot
relative to its sources. Also asserts two structural invariants no textual
diff alone would catch as cleanly:

1. **Coverage** — every tool in ``kg_server.REGISTERED_TOOLS`` has exactly one
   CPD, and every CPD id names a real registered tool (no orphan/phantom CPD).
   The packaged catalog is universal so an enabled runtime feature can never
   appear without routing authority.
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
    # prefer_cache=True: verify against the committed vendored ledger cache, the
    # SAME source CI sees (no EG clone there). Reading a live EG ledger this box
    # happens to have would make the on-box gate pass while the identical CI gate
    # fails on real drift — the reproducibility hole that let this slip past
    # pre-commit. The gate is now environment-invariant.
    cpds, generated_at = gcp.generate(None, refresh_cache=False, prefer_cache=True)
    md = gcp.render_markdown(cpds, generated_at=generated_at)
    js = gcp.render_json(cpds, generated_at=generated_at)

    if (
        not gcp.MD_PATH.exists()
        or not gcp.JSON_PATH.exists()
        or not gcp.PACKAGE_JSON_PATH.exists()
    ):
        errors.append(
            "one or more capability catalog outputs are missing — run "
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
    if strip_generation_timestamp(
        gcp.PACKAGE_JSON_PATH.read_text(encoding="utf-8")
    ) != strip_generation_timestamp(js):
        errors.append(
            f"{gcp.PACKAGE_JSON_PATH} is stale relative to the live tool "
            "registry + EG ledger — run "
            "`python scripts/gen_capability_power.py --write`."
        )
    if gcp.JSON_PATH.read_bytes() != gcp.PACKAGE_JSON_PATH.read_bytes():
        errors.append(
            f"{gcp.PACKAGE_JSON_PATH} is not byte-identical to {gcp.JSON_PATH} — "
            "run `python scripts/gen_capability_power.py --write`."
        )
    return errors, cpds


def _check_coverage(cpds) -> list[str]:
    """Coverage MUST be checked against the packaged, CHECKED-IN CPD set —
    the exact artifact ``capability_context.load_cpds()`` reads at runtime
    (the same file ``intent_tools._build_candidates`` fails closed against)
    — never against a freshly-``generate()``d ``cpds`` in this same process.

    A fresh generation is built FROM the live tool registry, so comparing it
    back to that same live registry is tautological: it can never disagree
    with itself, no matter how stale the checked-in file on disk is. That
    exact blind spot is what let ``engine_placement`` register live while
    the packaged ``capabilities-power.json`` on disk stayed 113-entries-and-
    one-short — ``_check_drift`` (the byte-diff above) happened to still
    catch that specific incident, but coverage must not silently depend on
    drift catching every case; it has to independently assert the one
    invariant it exists for against the artifact production actually loads.
    """
    from agent_utilities.knowledge_graph.retrieval.capability_context import (
        load_cpds,
    )
    from agent_utilities.mcp import kg_server
    from agent_utilities.mcp.tools.intent_tools import INTENT_VERBS

    kg_server.ensure_tools_registered()
    # Mirror intent_tools._build_candidates' own fail-closed comparison
    # exactly: live granular tools (registered tools minus the six intent
    # verbs, which have CPDs but are never resolver targets) against the
    # packaged, checked-in CPD ids actually shipped on disk.
    tool_names = set(kg_server.REGISTERED_TOOLS) - set(INTENT_VERBS)
    packaged_ids = set(load_cpds())
    errors: list[str] = []
    missing = tool_names - packaged_ids
    if missing:
        errors.append(
            "Tools with no CPD in the packaged, checked-in "
            f"capabilities-power.json: {sorted(missing)} — run "
            "`python scripts/gen_capability_power.py --write`."
        )
    # Freshly-generated ids ARE the authoritative "what should be packaged"
    # set (mirrors the coverage the drift/byte-diff check above already
    # enforces for content); a checked-in id that no longer corresponds to
    # any freshly-generated capability is a phantom entry the next --write
    # would silently drop.
    fresh_ids = {c.id for c in cpds}
    phantom = packaged_ids - fresh_ids
    if phantom:
        errors.append(
            f"Packaged CPDs for non-existent/stale tools: {sorted(phantom)} — "
            "run `python scripts/gen_capability_power.py --write`."
        )
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
