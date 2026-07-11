#!/usr/bin/env python3
"""Regenerate the Capability Power Descriptors (CPD) for the graph-os tool surface.

CONCEPT:AU-KG.retrieval.capability-power-descriptor — Seam 8 Phase 1
(``plans/program-design-2026-07-11-epistemic-tool-routing.md`` section 2b).

One CPD per registered graph-os MCP tool (the granularity the ~100-tool intent
surface (2a) actually routes at — see the CPD module docstring for why),
enriched with a ``does[]`` entry per action the tool fronts. Every field is
derived from a live or generated source — see
``agent_utilities/knowledge_graph/retrieval/capability_power_descriptor.py``'s
module docstring for the full source list. This script is pure orchestration
(mirrors ``scripts/gen_docs.py`` / ``scripts/check_surface_parity.py``):
it builds the AU tool registry, locates the EG-P0-1 ledger (live sibling
checkout or the vendored cache), assembles one CPD per tool, and writes:

* ``docs/capabilities-power.md``  — human/LLM-browsable rendering.
* ``docs/capabilities-power.json`` — the same data, machine-readable.
* ``docs/_vendor_eg_capability_ledger.json`` — a small cache of the EG ledger
  rows actually used, refreshed whenever a live EG checkout is found, so
  ``--check`` stays deterministic in an AU-only checkout that has no sibling
  epistemic-graph clone (EG's ledger is itself a GENERATED artifact of a
  different repo/build system — this cache is the honest cross-repo analogue
  of importing a generated module, not a hand-authored duplicate: it is only
  ever written by copying live ledger rows, never edited).

Usage::

    python scripts/gen_capability_power.py --write
    python scripts/gen_capability_power.py --check
    python scripts/gen_capability_power.py --write --eg-ledger /path/to/capabilities.generated.md
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent_utilities.knowledge_graph.retrieval.capability_power_descriptor import (  # noqa: E402
    MEASURED_LATENCY_MS,
    CapabilityPowerDescriptor,
    LedgerRow,
    Provenance,
    aggregate_side_effects,
    build_does_items,
    infer_intent_verbs,
    parse_eg_ledger_markdown,
    render_json,
    render_markdown,
    strip_generation_timestamp,
)

MD_PATH = ROOT / "docs" / "capabilities-power.md"
JSON_PATH = ROOT / "docs" / "capabilities-power.json"
CACHE_PATH = ROOT / "docs" / "_vendor_eg_capability_ledger.json"

# Candidate sibling-checkout locations for the EG-generated ledger, tried in
# order after an explicit --eg-ledger / $EG_CAPABILITIES_LEDGER. None of these
# being present is NOT a hard failure — the vendored cache (CACHE_PATH) is the
# fallback so this script (and its --check gate) stays runnable in an AU-only
# checkout.
_CANDIDATE_LEDGER_PATHS = (
    "../epistemic-graph/docs/capabilities.generated.md",
    "../../epistemic-graph/docs/capabilities.generated.md",
    "/home/apps/workspace/agent-packages/epistemic-graph/docs/capabilities.generated.md",
)


def locate_eg_ledger(explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit).expanduser()
        return p if p.exists() else None
    env = os.environ.get("EG_CAPABILITIES_LEDGER")
    if env:
        p = Path(env).expanduser()
        if p.exists():
            return p
    for rel in _CANDIDATE_LEDGER_PATHS:
        p = (ROOT / rel).resolve() if not rel.startswith("/") else Path(rel)
        if p.exists():
            return p
    return None


def load_ledger(explicit: str | None) -> tuple[dict[str, LedgerRow], str | None, bool]:
    """Return ``(ledger, path_used, was_live)``.

    Prefers a live EG checkout; refreshes the vendored cache from it when
    found. Falls back to the last-committed cache when no live checkout is
    reachable — never fabricates ledger rows.
    """
    live_path = locate_eg_ledger(explicit)
    if live_path is not None:
        text = live_path.read_text(encoding="utf-8")
        ledger = parse_eg_ledger_markdown(text)
        _write_cache(ledger, str(live_path))
        return ledger, str(live_path), True

    if CACHE_PATH.exists():
        cached = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        ledger = {
            method: LedgerRow(**row) for method, row in cached.get("rows", {}).items()
        }
        return ledger, cached.get("source_path"), False

    return {}, None, False


def _write_cache(ledger: dict[str, LedgerRow], source_path: str) -> None:
    from datetime import UTC, datetime

    payload = {
        "source_path": source_path,
        "cached_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "row_count": len(ledger),
        "rows": {method: row.to_dict() for method, row in ledger.items()},
    }
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Action inventory per tool
# ---------------------------------------------------------------------------

_QUOTED_TOKEN_RE = re.compile(r"'([a-z][a-z0-9_]*)'")
# Many tool descriptions lead with a bare CONCEPT id ("CONCEPT:AU-KG.foo.bar — ")
# before the actual power sentence — strip it so `one_line` starts with the
# actual power statement, not the id.
_CONCEPT_PREFIX_RE = re.compile(r"^CONCEPT:[\w.\-]+\s*[—-]\s*")


def _extract_one_line(description: str) -> str:
    """The tool's own description, minus a leading CONCEPT id, up to the first sentence."""
    text = _CONCEPT_PREFIX_RE.sub("", description.strip())
    # Split on '. ' but require what follows to start a new capitalized clause
    # (not just a mid-sentence abbreviation like "e.g. ") to avoid truncating
    # too early.
    m = re.search(r"\.\s+[A-Z]", text)
    if m:
        return text[: m.start() + 1].strip()
    return text[:200].strip()


def _parse_action_tokens_from_text(default: str, description: str) -> list[str]:
    """Best-effort fallback: pull an action enumeration out of a tool's own
    ``action`` parameter (default/description) when neither the generated
    verbose-action manifest nor a literal action-tuple constant covers it
    (currently just ``graph_ops_causal`` — everything else has a stronger
    source). Never invents an action not textually present.
    """
    quoted = _QUOTED_TOKEN_RE.findall(description)
    if len(quoted) >= 2:
        seen: list[str] = []
        for t in quoted:
            if t not in seen:
                seen.append(t)
        return seen
    pipe_source = description if "|" in description else ""
    if pipe_source:
        toks = [t.strip().strip("'\"") for t in pipe_source.split("|")]
        toks = [t for t in toks if re.fullmatch(r"[a-z][a-z0-9_]*", t)]
        if toks:
            return toks
    if default:
        return [default]
    return []


def get_actions_for_tool(
    tool_name: str,
    param_schema: dict[str, Any],
    manifest_by_tool: dict[str, list[str | None]],
    engine_domains: dict[str, list[str]],
    mining_actions: tuple[str, ...],
    graphlearn_actions: tuple[str, ...],
    deep_mining_actions: tuple[str, ...],
) -> list[str]:
    if tool_name.startswith("engine_"):
        domain = tool_name[len("engine_") :]
        if domain in engine_domains:
            return sorted(engine_domains[domain])
    if tool_name == "graph_mine":
        return list(mining_actions)
    if tool_name == "graph_learn":
        return list(graphlearn_actions)
    if tool_name == "graph_mine_deep":
        return list(deep_mining_actions)
    if tool_name in manifest_by_tool:
        actions = [a for a in manifest_by_tool[tool_name] if a is not None]
        if actions:
            return sorted(actions)
        return []  # action=None recorded: single-operation tool, no sub-actions
    action_field = (param_schema or {}).get("properties", {}).get("action") or {}
    return _parse_action_tokens_from_text(
        str(action_field.get("default", "")), str(action_field.get("description", ""))
    )


# ---------------------------------------------------------------------------
# Typed I/O + examples from the live FastMCP tool schema
# ---------------------------------------------------------------------------


def build_typed_io(tool: Any, rest_route: str) -> dict[str, Any]:
    schema = tool.parameters or {}
    props = schema.get("properties", {})
    required = set(schema.get("required", []))
    params = [
        {
            "name": name,
            "type": spec.get("type", "any"),
            "required": name in required,
            "default": spec.get("default"),
            "description": spec.get("description", ""),
        }
        for name, spec in props.items()
    ]
    return {
        "rest_route": rest_route,
        "tags": sorted(tool.tags or []),
        "input_params": params,
        "input_schema": schema,
        "output": "JSON string (json.dumps result) — see each tool's docstring/description "
        "for the returned shape; no formal output JSON Schema is declared "
        "server-side today (all graph-os tools return `str`).",
    }


def build_examples(
    tool_name: str, one_line: str, does: list[dict[str, Any]]
) -> list[str]:
    """A few intent-phrasing examples, DERIVED from the tool's own action names
    (not hand-authored copy) — the X-4/2c resolver few-shot signal.
    """
    examples = [f"{tool_name}: {one_line[:80]}"]
    for d in does[:3]:
        examples.append(f"{tool_name} action={d['action']}")
    return examples


def build_when_hints(tool_name: str, tags: set[str]) -> tuple[list[str], list[str]]:
    when_to_use: list[str] = []
    when_not: list[str] = []
    if "engine" in tags:
        when_to_use.append(
            "You need the raw, low-level engine primitive directly (1:1 over the "
            "epistemic-graph wire protocol) — precise control, or the condensed "
            "`graph_*` tool doesn't expose the operation you need."
        )
        when_not.append(
            "A condensed `graph_*` tool already covers your intent — prefer it "
            "first; the `engine_*` surface is the escape hatch, not the default."
        )
    if "mutation" in tags or "write" in tags or "write_ingest" in tags:
        when_not.append(
            "You only need to read/inspect — use the matching read-only "
            "`graph_query`/`graph_search`/`graph_analyze` capability instead."
        )
    if not when_to_use:
        when_to_use.append(
            "Your intent matches one of this capability's `does` actions."
        )
    if not when_not:
        when_not.append(
            "Another capability's `does` list is a closer match to your intent "
            "— check the index for a more specific fit."
        )
    return when_to_use, when_not


# ---------------------------------------------------------------------------
# Eligibility predicates (X-4 formula, described; live-evaluated when possible)
# ---------------------------------------------------------------------------


def build_eligibility_predicates(tool_name: str, tags: set[str]) -> dict[str, Any]:
    return {
        "formula": (
            "eligible(candidate, required) = ontology_subsumption(candidate.capability_type, "
            "required) AND tenant_match(candidate.tenant, caller.tenant) AND "
            "policy_tag_match(candidate.policy_tags, required.policy_tags), ranked by "
            "cosine(embedding) + reward_weight*(bandit_reward-0.5)"
        ),
        "source": (
            "agent_utilities.graph.routing.enrichers.capability_routing."
            "explain_routing_eligibility (X-4, CONCEPT:AU-P1-3)"
        ),
        "live_evaluation": None,
        "note": (
            "not live-evaluated in this static generation pass — "
            "explain_routing_eligibility() needs a reachable engine AND a "
            "populated capability_type/ontology assignment for this tool's node; "
            "call it directly against a live engine for a per-request eligibility "
            "proof rather than trusting a value baked in here"
        ),
        "candidate_tags": sorted(tags),
    }


# ---------------------------------------------------------------------------
# CPD assembly
# ---------------------------------------------------------------------------


def build_cpd(
    tool: Any,
    action_tool_routes: dict[str, str],
    ledger: dict[str, LedgerRow],
    ledger_path: str | None,
    ledger_live: bool,
    manifest_by_tool: dict[str, list[str | None]],
    engine_domains: dict[str, list[str]],
    mining_actions: tuple[str, ...],
    graphlearn_actions: tuple[str, ...],
    deep_mining_actions: tuple[str, ...],
) -> CapabilityPowerDescriptor:
    name = tool.name
    tags = set(tool.tags or [])
    rest_route = action_tool_routes.get(name, "")
    actions = get_actions_for_tool(
        name,
        tool.parameters or {},
        manifest_by_tool,
        engine_domains,
        mining_actions,
        graphlearn_actions,
        deep_mining_actions,
    )
    if not actions:
        actions = [name]  # single-operation tool: the tool itself is the one "action"
    does = build_does_items(name, actions, ledger)
    side_effects = aggregate_side_effects(does)

    # cost/latency: only ever set from MEASURED_LATENCY_MS, keyed by a matched
    # EG Method — never estimated for an unmatched action.
    latency: dict[str, Any] = {}
    for d in does:
        m = d.get("eg_method")
        if m and m in MEASURED_LATENCY_MS:
            latency[d["action"]] = {
                "eg_method": m,
                **MEASURED_LATENCY_MS[m],
                "kind": "measured",
            }

    one_line = _extract_one_line(tool.description or "")
    if not one_line:
        one_line = f"{name} — see `does` for its actions."
    when_to_use, when_not = build_when_hints(name, tags)

    return CapabilityPowerDescriptor(
        id=name,
        title=name.replace("_", " "),
        one_line=one_line,
        intent_verbs=infer_intent_verbs(name, tags),
        does=does,
        typed_io=build_typed_io(tool, rest_route),
        side_effects=side_effects,
        scopes=sorted({d["authz_action"] for d in does if d.get("authz_action")}),
        policy={
            "approval_class": "human_approval_required" if "admin" in tags else "auto",
            "note": "coarse default from MCP tags; see side_effects/does for the "
            "per-action ledger authz_action, which is the authoritative scope",
        },
        cost={},  # no cost-unit telemetry source exists yet; left empty, not guessed
        latency=latency,
        reliability={},  # requires a live engine's bandit reward; empty here, not guessed
        preconditions=(
            [
                "a reachable graph-os engine (IntelligenceGraphEngine) with the required backend"
            ]
        ),
        when_to_use=when_to_use,
        when_not=when_not,
        examples=build_examples(name, one_line, does),
        eligibility_predicates=build_eligibility_predicates(name, tags),
        calibrated_outcomes={},
        provenance=Provenance(
            source_method_eg=(
                does[0].get("eg_method")
                if len(does) == 1 and does[0].get("eg_method")
                else None
            ),
            eg_ledger_path=ledger_path,
            eg_ledger_available=ledger_live or bool(ledger),
        ),
    )


async def _list_tools(mcp: Any) -> list[Any]:
    return await mcp.list_tools()


def generate(eg_ledger_arg: str | None) -> tuple[list[CapabilityPowerDescriptor], str]:
    from agent_utilities.mcp import kg_server
    from agent_utilities.mcp._graphos_action_manifest import GRAPHOS_ACTIONS
    from agent_utilities.mcp.tools.engine_tools import ENGINE_DOMAINS

    args, mcp, _mw = kg_server._build_server(bootstrap=False)

    manifest_by_tool: dict[str, list[str | None]] = {}
    for entry in GRAPHOS_ACTIONS:
        manifest_by_tool.setdefault(entry["tool"], []).append(entry["action"])

    mining_actions = getattr(kg_server, "MINING_ACTIONS", ())
    graphlearn_actions = getattr(kg_server, "GRAPHLEARN_ACTIONS", ())
    deep_mining_actions = getattr(kg_server, "DEEP_MINING_ACTIONS", ())

    ledger, ledger_path, ledger_live = load_ledger(eg_ledger_arg)

    tools = asyncio.run(_list_tools(mcp))
    cpds = [
        build_cpd(
            t,
            kg_server.ACTION_TOOL_ROUTES,
            ledger,
            ledger_path,
            ledger_live,
            manifest_by_tool,
            ENGINE_DOMAINS,
            mining_actions,
            graphlearn_actions,
            deep_mining_actions,
        )
        for t in sorted(tools, key=lambda t: t.name)
    ]
    from datetime import UTC, datetime

    generated_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    return cpds, generated_at


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--write", action="store_true", help="Write docs/capabilities-power.{md,json}."
    )
    ap.add_argument(
        "--check", action="store_true", help="Exit non-zero if checked-in docs drift."
    )
    ap.add_argument(
        "--eg-ledger", default=None, help="Explicit path to the EG generated ledger."
    )
    args = ap.parse_args()

    if not args.write and not args.check:
        args.check = True  # default: safe, side-effect-free

    cpds, generated_at = generate(args.eg_ledger)
    md = render_markdown(cpds, generated_at=generated_at)
    js = render_json(cpds, generated_at=generated_at)

    if args.write:
        MD_PATH.parent.mkdir(parents=True, exist_ok=True)
        MD_PATH.write_text(md, encoding="utf-8")
        JSON_PATH.write_text(js, encoding="utf-8")
        print(f"Wrote {len(cpds)} CPDs to {MD_PATH} and {JSON_PATH}")
        return 0

    # --check: regenerate (content only, ignoring the generation timestamp,
    # which appears both as a JSON `generated_at` field and inline markdown
    # prose) and diff against what's committed.
    ok = True
    if not MD_PATH.exists() or not JSON_PATH.exists():
        print("capabilities-power.md/.json missing — run --write first.")
        return 1
    if strip_generation_timestamp(
        MD_PATH.read_text(encoding="utf-8")
    ) != strip_generation_timestamp(md):
        print(f"DRIFT: {MD_PATH} does not match the live tool registry + ledger.")
        ok = False
    if strip_generation_timestamp(
        JSON_PATH.read_text(encoding="utf-8")
    ) != strip_generation_timestamp(js):
        print(f"DRIFT: {JSON_PATH} does not match the live tool registry + ledger.")
        ok = False
    if ok:
        print(
            f"CPD set is in sync with the live tool registry ({len(cpds)} capabilities)."
        )
        return 0
    print("Run: python scripts/gen_capability_power.py --write")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
