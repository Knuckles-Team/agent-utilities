#!/usr/bin/python
from __future__ import annotations

"""Capability Power Descriptor (CPD) — CONCEPT:AU-KG.retrieval.capability-power-descriptor.

Seam 8 Phase 1 (``plans/program-design-2026-07-11-epistemic-tool-routing.md``
section 2b) — "the power behind each tool", assembled so the future resolver
(2c) — and any human or LLM browsing ``docs/capabilities-power.md`` today —
can answer *what can this capability do, when should I use it, how reliable is
it* WITHOUT loading a schema per tool.

**Derived, not hand-maintained.** Every field below is either read straight off
an existing, already-generated source, or computed from one by a pure function
in this module — never typed in by hand for a specific capability id. That is
the whole point: a CPD can never silently rot because it is regenerated from
the same ground truth every time (:mod:`scripts.gen_capability_power` is the
generator; ``scripts/check_cpd.py`` is the drift gate). Sources, one per field
group:

* **Tool surface** (``id``, ``title``, ``one_line``, part of ``does``,
  ``typed_io``) — the live FastMCP tool registry built by
  ``agent_utilities.mcp.kg_server._build_server`` (``mcp.list_tools()``): the
  REAL registered ``name``/``description``/``tags``/``parameters`` JSON Schema,
  not a hand copy of it.
* **Action inventory** (rest of ``does``) — the exhaustive, ALREADY GENERATED
  ``agent_utilities.mcp._graphos_action_manifest.GRAPHOS_ACTIONS`` (one row per
  condensed-tool action, produced by ``scripts/gen_graphos_manifest.py``) when a
  tool has one, else a best-effort parse of the tool's own ``action`` parameter
  schema (some tools — e.g. ``graph_mine``/``graph_learn`` — predate the
  manifest and declare their action set as a literal tuple in ``kg_server.py``;
  those are read directly, never re-typed here).
* **REST route** (``id`` → REST twin) — ``kg_server.ACTION_TOOL_ROUTES`` (the
  canonical tool⇄REST parity map ``check_surface_parity.py`` already gates).
  A CPD with no REST route entry is a signal the parity gate should already be
  catching, not a CPD concern.
* **Side effects / durability / authz / CDC / txn** — the EG-P0-1 generated
  capability ledger (``epistemic-graph/docs/capabilities.generated.md``),
  matched to an AU action by a dependency-free, best-effort NAME-DERIVED
  fuzzy match (:func:`match_action_to_method`) — because AU's action-routed
  surface and EG's raw ``Method`` enum are maintained in two different repos
  with no existing 1:1 crosswalk. Every match records its own confidence and
  the exact tokens that matched; an action with no confident match is left
  unmatched with an honest note, NEVER guessed.
* **Cost / latency / reliability** — a small table of numbers TRANSCRIBED
  (not estimated) from EG's measured benchmark docs
  (``docs/benchmarks.md`` / ``docs/benchmarks-soak.md``), keyed by the exact
  ``Method`` they measured (``AddNode``, ``GetNodeProperties``, ``ClaimNext``,
  ``CompareAndSetNodeFields``). Everything else is left ``None`` — explicitly
  labeled ``modeled`` only where the AU capacity model
  (``docs/scaling/capacity_model.py``) provides a derivation, else
  unmeasured/unmodeled, never fabricated. ``reliability`` mirrors the durable
  contextual-bandit reward EMA (:mod:`.durable_outcome_store`) when a live
  engine is reachable; otherwise it is left ``None``.
* **Eligibility predicates** — the X-4 routing logic itself
  (:func:`agent_utilities.graph.routing.enrichers.capability_routing.explain_routing_eligibility`)
  is DESCRIBED here (the formula: ontology subsumption + tenant match + policy
  tag match + bandit reward blend); a live per-capability evaluation additionally
  runs when an engine is reachable, else the field states plainly that it
  requires a live engine + a populated ``capability_type`` node.
* **Calibrated outcomes** — read from the durable bandit store when an engine
  is reachable; otherwise left empty, never invented.

This module owns only the SCHEMA + the pure assembly/matching functions. The
orchestration (build the tool registry, load the EG ledger, write the docs) is
:mod:`scripts.gen_capability_power`, mirroring the existing
``scripts/gen_docs.py`` / ``scripts/check_surface_parity.py`` split between
"logic lives in a script" and "reusable pieces live in the package".
"""

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

__all__ = [
    "CapabilityPowerDescriptor",
    "Provenance",
    "LedgerRow",
    "parse_eg_ledger_markdown",
    "match_action_to_method",
    "camel_tokens",
    "snake_tokens",
    "build_does_items",
    "aggregate_side_effects",
    "infer_intent_verbs",
    "MEASURED_LATENCY_MS",
    "render_markdown",
    "render_json",
    "strip_generation_timestamp",
]

_ISO_TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")


def strip_generation_timestamp(text: str) -> str:
    """Normalize away the one field that legitimately changes every run.

    Used by both the generator's ``--check`` mode and ``scripts/check_cpd.py``
    so a rendering is compared for CONTENT drift, not for having been
    generated at a different second than what's checked in.
    """
    return _ISO_TIMESTAMP_RE.sub("<generated_at>", text)


# ---------------------------------------------------------------------------
# Measured telemetry — TRANSCRIBED from EG's measured benchmark docs, never
# estimated. Keyed by the exact EG ``Method`` name the benchmark exercised.
# Update these only by copying a new number out of a fresh benchmark run;
# never interpolate/guess for a Method not directly measured.
# ---------------------------------------------------------------------------
MEASURED_LATENCY_MS: dict[str, dict[str, Any]] = {
    "AddNode": {
        "p50_ms": 0.187,
        "p99_ms": 0.223,
        "source": "epistemic-graph/docs/benchmarks.md#results (2026-06-01, UDS, in-memory graph)",
    },
    "GetNodeProperties": {
        "p50_ms": 0.179,
        "p99_ms": 0.210,
        "source": "epistemic-graph/docs/benchmarks.md#results (2026-06-01, UDS, in-memory graph)",
    },
    # Soak/chaos run (2026-07-11) — a SHARED, contended box; latencies are an
    # upper bound per that doc's own caveat, carried through here verbatim.
    "CompareAndSetNodeFields": {
        "p50_ms": 14.17,
        "p95_ms": 43.72,
        "p99_ms": 57.24,
        "source": (
            "epistemic-graph/docs/benchmarks-soak.md#phase-a (2026-07-11 soak run, "
            "shared/contended box — upper bound per source doc)"
        ),
    },
    "ClaimNext": {
        "source": (
            "epistemic-graph/docs/benchmarks-soak.md (queue/claim primitive is "
            "measured in the same soak run; approximates AgentBus queue latency "
            "per that doc's own caveat — see the doc for the current number, not "
            "duplicated here to avoid a second copy drifting from the source)"
        ),
    },
}


@dataclass
class Provenance:
    """Where every field in one CPD came from, and when it was generated."""

    generator_version: str = "1.0.0"
    source_repo_au: str = "agent-utilities"
    source_module_au: str = "agent_utilities.mcp.kg_server"
    source_method_eg: str | None = None
    eg_ledger_path: str | None = None
    eg_ledger_available: bool = False
    generated_at: str = ""

    def __post_init__(self) -> None:
        if not self.generated_at:
            self.generated_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    def to_dict(self) -> dict[str, Any]:
        return {
            "generator_version": self.generator_version,
            "source_repo_au": self.source_repo_au,
            "source_module_au": self.source_module_au,
            "source_method_eg": self.source_method_eg,
            "eg_ledger_path": self.eg_ledger_path,
            "eg_ledger_available": self.eg_ledger_available,
            "generated_at": self.generated_at,
        }


@dataclass
class CapabilityPowerDescriptor:
    """One capability's full "power" record — see module docstring for sources."""

    id: str
    title: str
    one_line: str
    intent_verbs: list[str] = field(default_factory=list)
    does: list[dict[str, Any]] = field(default_factory=list)
    typed_io: dict[str, Any] = field(default_factory=dict)
    side_effects: dict[str, Any] = field(default_factory=dict)
    scopes: list[str] = field(default_factory=list)
    policy: dict[str, Any] = field(default_factory=dict)
    cost: dict[str, Any] = field(default_factory=dict)
    latency: dict[str, Any] = field(default_factory=dict)
    reliability: dict[str, Any] = field(default_factory=dict)
    preconditions: list[str] = field(default_factory=list)
    when_to_use: list[str] = field(default_factory=list)
    when_not: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    eligibility_predicates: dict[str, Any] = field(default_factory=dict)
    calibrated_outcomes: dict[str, Any] = field(default_factory=dict)
    provenance: Provenance = field(default_factory=Provenance)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "one_line": self.one_line,
            "intent_verbs": list(self.intent_verbs),
            "does": list(self.does),
            "typed_io": dict(self.typed_io),
            "side_effects": dict(self.side_effects),
            "scopes": list(self.scopes),
            "policy": dict(self.policy),
            "cost": dict(self.cost),
            "latency": dict(self.latency),
            "reliability": dict(self.reliability),
            "preconditions": list(self.preconditions),
            "when_to_use": list(self.when_to_use),
            "when_not": list(self.when_not),
            "examples": list(self.examples),
            "eligibility_predicates": dict(self.eligibility_predicates),
            "calibrated_outcomes": dict(self.calibrated_outcomes),
            "provenance": self.provenance.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CapabilityPowerDescriptor:
        prov = d.get("provenance") or {}
        return cls(
            id=str(d["id"]),
            title=str(d.get("title", "")),
            one_line=str(d.get("one_line", "")),
            intent_verbs=list(d.get("intent_verbs") or []),
            does=list(d.get("does") or []),
            typed_io=dict(d.get("typed_io") or {}),
            side_effects=dict(d.get("side_effects") or {}),
            scopes=list(d.get("scopes") or []),
            policy=dict(d.get("policy") or {}),
            cost=dict(d.get("cost") or {}),
            latency=dict(d.get("latency") or {}),
            reliability=dict(d.get("reliability") or {}),
            preconditions=list(d.get("preconditions") or []),
            when_to_use=list(d.get("when_to_use") or []),
            when_not=list(d.get("when_not") or []),
            examples=list(d.get("examples") or []),
            eligibility_predicates=dict(d.get("eligibility_predicates") or {}),
            calibrated_outcomes=dict(d.get("calibrated_outcomes") or {}),
            provenance=Provenance(
                generator_version=str(prov.get("generator_version", "1.0.0")),
                source_repo_au=str(prov.get("source_repo_au", "agent-utilities")),
                source_module_au=str(
                    prov.get("source_module_au", "agent_utilities.mcp.kg_server")
                ),
                source_method_eg=prov.get("source_method_eg"),
                eg_ledger_path=prov.get("eg_ledger_path"),
                eg_ledger_available=bool(prov.get("eg_ledger_available", False)),
                generated_at=str(prov.get("generated_at", "")),
            ),
        )


# ---------------------------------------------------------------------------
# EG ledger parsing
# ---------------------------------------------------------------------------

_LEDGER_ROW_RE = re.compile(
    r"^\|\s*`([^`]+)`\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*`([^`]+)`\s*\|\s*"
    r"([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*(.*?)\s*\|\s*$"
)


@dataclass
class LedgerRow:
    method: str
    mutates: str
    durability: str
    authz_action: str
    idempotent: str
    audited: str
    emits_cdc: str
    txn_participation: str
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "mutates": self.mutates,
            "durability": self.durability,
            "authz_action": self.authz_action,
            "idempotent": self.idempotent,
            "audited": self.audited,
            "emits_cdc": self.emits_cdc,
            "txn_participation": self.txn_participation,
            "note": self.note,
        }


def parse_eg_ledger_markdown(text: str) -> dict[str, LedgerRow]:
    """Parse the EG-P0-1 generated capability ledger table into ``{Method: LedgerRow}``.

    Tolerant of the surrounding prose/headers — only matches the strict
    9-column ``| `Method` | ... |`` data rows, skipping the header/separator
    rows (which don't backtick-quote the first cell).
    """
    out: dict[str, LedgerRow] = {}
    for line in text.splitlines():
        m = _LEDGER_ROW_RE.match(line.strip())
        if not m:
            continue
        method = m.group(1)
        out[method] = LedgerRow(
            method=method,
            mutates=m.group(2),
            durability=m.group(3),
            authz_action=m.group(4),
            idempotent=m.group(5),
            audited=m.group(6),
            emits_cdc=m.group(7),
            txn_participation=m.group(8),
            note=m.group(9),
        )
    return out


# ---------------------------------------------------------------------------
# Name-derived fuzzy matching: AU action name -> EG Method
# ---------------------------------------------------------------------------

_CAMEL_SPLIT_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z0-9]+|[A-Z]+")

# tool-name (post "engine_"/"graph_" prefix strip) -> extra domain tokens that
# help the matcher without hand-mapping the whole surface: just a plural/
# singular + a couple of EG-side synonyms actually seen in the ledger's Method
# names. Missing an entry here just means a slightly lower match rate for that
# tool's actions, never a wrong/fabricated one (unmatched stays honestly unmatched).
_DOMAIN_SYNONYMS: dict[str, tuple[str, ...]] = {
    "nodes": ("node",),
    "edges": ("edge",),
    "channels": ("channel",),
    "tenants": ("graph",),  # engine_tenants ~ EG's per-graph registry Methods
    "ledger": ("ledger",),
    "blob": ("blob",),
    "rdf": ("rdf", "triples", "triple"),
    "streaming": ("stream",),
    "resharding": ("reshard", "catalog", "rebalance"),
    "txn": ("txn",),
    "analytics": ("centrality", "pagerank"),
    "graph": ("graph",),
    "lifecycle": ("node", "graph"),
    "consensus": ("multisig", "identity"),
    "reasoning": ("reasoning", "datalog"),
    "mining": ("mine",),
    "mine": ("mine",),
    "graphlearn": ("graph", "learn"),
    "learn": ("graph", "learn"),
}


# A handful of verb pairs AU and EG each spell differently for the same
# operation (observed directly by eyeballing ledger vs manifest names, not
# guessed) — expanding both spellings into the token set turns a spelling
# difference into a match instead of an honest-but-avoidable non-match.
_VERB_SYNONYMS: dict[str, tuple[str, ...]] = {
    "delete": ("remove",),
    "remove": ("delete",),
    "fetch": ("get",),
}


def camel_tokens(name: str) -> set[str]:
    """Split a PascalCase EG ``Method`` name into lowercase word tokens."""
    return {t.lower() for t in _CAMEL_SPLIT_RE.findall(name) if t}


def snake_tokens(name: str, *extra: str) -> set[str]:
    """Split a snake_case AU action name (plus any extra domain hints) into tokens.

    Each token is also expanded with its known :data:`_VERB_SYNONYMS` (e.g.
    ``delete`` also yields ``remove``) so a same-operation spelling difference
    between AU and EG doesn't read as a non-match.
    """
    base = {t for t in re.split(r"[_\-]+", name.lower()) if t}
    toks = set(base)
    for t in base:
        toks.update(_VERB_SYNONYMS.get(t, ()))
    toks.update(t.lower() for t in extra if t)
    return toks


def match_action_to_method(
    tool_name: str, action: str, ledger: dict[str, LedgerRow]
) -> tuple[LedgerRow | None, float, list[str]]:
    """Best-effort, name-derived match of one AU ``(tool, action)`` to an EG ``Method``.

    Two-factor check (precision over recall — a wrong match is worse than an
    honest non-match): a candidate Method must share a token with the ACTION
    name itself (``core_overlap``). If the action tokens alone do not fully
    explain the Method's name (``full_core``), a token with the tool's DOMAIN
    word must land in the Method too (``domain_overlap``) — so
    ``engine_nodes.list`` cannot match ``ListGraphs`` merely because both
    happen to contain "list"; the domain word "node" must also appear. When
    the action tokens alone DO fully explain the Method's name (e.g.
    ``graph_write.add_node`` against ``AddNode``, or ``engine_admin.backup``
    against ``Backup``, where the domain word itself isn't repeated in the
    Method name), the domain check is skipped — a full name match needs no
    corroborating domain word. Score is coverage of the candidate Method's own
    tokens. Returns ``(row_or_None, coverage 0..1, matched_tokens)`` —
    ``(None, 0.0, [])`` when no candidate clears the checks and a minimum
    coverage, meaning no confident match, never a guess.
    """
    domain = tool_name.split("engine_", 1)[-1].split("graph_", 1)[-1]
    domain_tokens = {domain, domain.rstrip("s")} | set(_DOMAIN_SYNONYMS.get(domain, ()))
    domain_tokens.discard("")
    action_tokens = snake_tokens(action)

    best_row: LedgerRow | None = None
    best_coverage = 0.0
    best_overlap: set[str] = set()
    for method, row in ledger.items():
        method_tokens = camel_tokens(method)
        core_overlap = action_tokens & method_tokens
        if not core_overlap:
            continue
        full_core = core_overlap == method_tokens
        if full_core:
            overlap = core_overlap
        else:
            domain_overlap = domain_tokens & method_tokens
            if domain_tokens and not domain_overlap:
                continue
            overlap = core_overlap | domain_overlap
        coverage = len(overlap) / len(method_tokens) if method_tokens else 0.0
        if coverage > best_coverage:
            best_coverage = coverage
            best_row = row
            best_overlap = overlap

    if best_row is None or best_coverage < 0.5:
        return None, 0.0, []
    return best_row, round(best_coverage, 3), sorted(best_overlap)


# ---------------------------------------------------------------------------
# does[] assembly + side-effect aggregation
# ---------------------------------------------------------------------------


def build_does_items(
    tool_name: str, actions: list[str], ledger: dict[str, LedgerRow]
) -> list[dict[str, Any]]:
    """One ``does[]`` entry per action, each carrying its own EG ledger match (if any)."""
    items: list[dict[str, Any]] = []
    for action in actions:
        row, score, tokens = match_action_to_method(tool_name, action, ledger)
        item: dict[str, Any] = {"action": action}
        if row is not None:
            item["eg_method"] = row.method
            item["match_confidence"] = score
            item["matched_tokens"] = tokens
            item["mutates"] = row.mutates
            item["durability"] = row.durability
            item["authz_action"] = row.authz_action
            item["idempotent"] = row.idempotent
            item["audited"] = row.audited
            item["emits_cdc"] = row.emits_cdc
            item["txn_participation"] = row.txn_participation
            if row.note:
                item["eg_note"] = row.note
        else:
            item["eg_method"] = None
            item["note"] = (
                "no confident EG-P0-1 ledger match by name — likely an AU-level "
                "orchestration action with no 1:1 raw engine Method, not a gap in "
                "matching effort"
            )
        items.append(item)
    return items


def aggregate_side_effects(does_items: list[dict[str, Any]]) -> dict[str, Any]:
    """Roll up per-action ledger facts into one tool-level side-effects summary."""
    matched = [d for d in does_items if d.get("eg_method")]
    mutates_values = {d.get("mutates") for d in matched}
    durability_values = {d.get("durability") for d in matched if d.get("durability")}
    txn_values = {
        d.get("txn_participation") for d in matched if d.get("txn_participation")
    }
    any_mutates = any(v in ("true", "~true") for v in mutates_values)
    all_mutates = bool(matched) and all(v in ("true", "~true") for v in mutates_values)
    return {
        "matched_action_count": len(matched),
        "unmatched_action_count": len(does_items) - len(matched),
        "any_action_mutates": any_mutates,
        "all_matched_actions_mutate": all_mutates,
        "durability_values_seen": sorted(durability_values),
        "txn_participation_values_seen": sorted(txn_values),
        "note": (
            "derived by matching each action to its EG-P0-1 ledger Method (see "
            "each does[] item for the per-action match + confidence); a tool with "
            "0 matched actions has no rollup here (see does[] notes) rather than a "
            "guessed value"
            if matched
            else "no action on this tool matched an EG ledger Method by name — no "
            "side-effect rollup is stated rather than guessed"
        ),
    }


# ---------------------------------------------------------------------------
# Intent-verb inference (2a) — which of ask/write/act/find/manage/why this
# capability would resolve under, from its own MCP tags + name (derived, not
# a hand-authored per-tool table).
# ---------------------------------------------------------------------------

_VERB_TAG_HINTS: dict[str, tuple[str, ...]] = {
    "ask": ("query", "search", "nl", "table", "reach", "data-analysis", "synthesis"),
    "write": ("write", "write_ingest", "ingestion", "mutation", "writeback", "etl"),
    "act": (
        "orchestrate",
        "governance",
        "sandbox",
        "runvcs",
        "scheduler",
        "loops",
        "goals",
    ),
    "find": ("retrieval",),
    "manage": (
        "state",
        "tenancy",
        "tenants",
        "rbac",
        "secret",
        "resharding",
        "lifecycle",
        "security",
    ),
    "why": ("explain", "observe", "eval", "evaluate"),
}


def infer_intent_verbs(tool_name: str, tags: set[str]) -> list[str]:
    """Which 2a intent verb(s) this capability would resolve under, from its tags/name."""
    verbs: list[str] = []
    for verb, hints in _VERB_TAG_HINTS.items():
        if tags & set(hints):
            verbs.append(verb)
    name = tool_name.lower()
    if not verbs:
        if any(k in name for k in ("query", "search", "ask", "table", "reach")):
            verbs.append("ask")
        elif any(k in name for k in ("write", "ingest", "etl")):
            verbs.append("write")
        elif any(k in name for k in ("explain", "observe")):
            verbs.append("why")
        elif any(
            k in name for k in ("configure", "secret", "tenant", "lifecycle", "reshard")
        ):
            verbs.append("manage")
        elif any(
            k in name for k in ("orchestrate", "loop", "goal", "sandbox", "runvcs")
        ):
            verbs.append("act")
        else:
            verbs.append(
                "ask"
            )  # conservative default: read-shaped until proven otherwise
    if tool_name in ("find_tools", "list_catalog", "load_tools", "unload_tools"):
        verbs = ["find"]
    return sorted(set(verbs))


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_markdown(cpds: list[CapabilityPowerDescriptor], *, generated_at: str) -> str:
    lines: list[str] = []
    lines.append("# graph-os Capability Power Descriptors (generated)")
    lines.append("")
    lines.append(
        "> **GENERATED — do not edit by hand.** Regenerate with "
        "`python3 scripts/gen_capability_power.py --write`; "
        "`scripts/check_cpd.py` gates drift in CI/pre-commit "
        "(CONCEPT:AU-KG.retrieval.capability-power-descriptor). "
        "Seam 8 Phase 1 — "
        "`plans/program-design-2026-07-11-epistemic-tool-routing.md` section 2b."
    )
    lines.append(">")
    lines.append(
        f"> {len(cpds)} capabilities · generated {generated_at}. Every field is "
        "derived from a live source (the MCP tool registry, the generated "
        "graph-os action manifest, the EG-P0-1 capability ledger, transcribed "
        "measured benchmarks) — an empty field means the source had no answer, "
        "never a fabricated one."
    )
    lines.append("")
    lines.append("## Index")
    lines.append("")
    lines.append("| Capability | Intent verbs | One-line power | Actions | REST |")
    lines.append("|---|---|---|---:|---|")
    for c in cpds:
        rest = c.typed_io.get("rest_route", "")
        lines.append(
            f"| [`{c.id}`](#{c.id.replace('_', '')}) | {', '.join(c.intent_verbs)} "
            f"| {c.one_line[:100]} | {len(c.does)} | `{rest}` |"
        )
    lines.append("")
    lines.append("## Capabilities")
    lines.append("")
    for c in cpds:
        lines.append(f"### `{c.id}`")
        lines.append("")
        lines.append(f"**{c.title}**")
        lines.append("")
        lines.append(c.one_line)
        lines.append("")
        lines.append(
            f"- **Intent verbs:** {', '.join(c.intent_verbs) or '(none inferred)'}"
        )
        lines.append(f"- **REST route:** `{c.typed_io.get('rest_route', '(none)')}`")
        lines.append(f"- **MCP tags:** {', '.join(c.typed_io.get('tags', []))}")
        se = c.side_effects
        lines.append(
            f"- **Side effects:** {se.get('matched_action_count', 0)}/"
            f"{se.get('matched_action_count', 0) + se.get('unmatched_action_count', 0)} "
            f"actions matched an EG ledger Method; any_mutates="
            f"{se.get('any_action_mutates')}; durability={se.get('durability_values_seen')}; "
            f"txn={se.get('txn_participation_values_seen')}"
        )
        if c.cost or c.latency:
            lines.append(f"- **Cost:** {c.cost or '(unmeasured)'}")
            lines.append(f"- **Latency:** {c.latency or '(unmeasured)'}")
        else:
            lines.append(
                "- **Cost/Latency:** unmeasured for this capability (no benchmark source)"
            )
        lines.append(
            f"- **Reliability:** {c.reliability or '(unmeasured — no live engine reward reachable at generation time)'}"
        )
        if c.does:
            lines.append("")
            lines.append("**Does:**")
            lines.append("")
            for d in c.does[:60]:
                eg = (
                    f" → EG `{d['eg_method']}` (confidence {d.get('match_confidence')})"
                    if d.get("eg_method")
                    else " → (no EG ledger match)"
                )
                lines.append(f"- `{d['action']}`{eg}")
            if len(c.does) > 60:
                lines.append(f"- ... and {len(c.does) - 60} more actions")
        if c.typed_io.get("input_params"):
            lines.append("")
            lines.append("**Typed input:**")
            lines.append("")
            for p in c.typed_io["input_params"]:
                lines.append(
                    f"- `{p['name']}` ({p.get('type', 'any')}"
                    f"{', required' if p.get('required') else ''}): {p.get('description', '')}"
                )
        lines.append("")
        lines.append(
            f"**Eligibility predicates:** {c.eligibility_predicates.get('formula', '(n/a)')}"
        )
        lines.append("")
        lines.append(
            f"**Calibrated outcomes:** {c.calibrated_outcomes or '(empty — no live bandit reward reachable at generation time)'}"
        )
        lines.append("")
        lines.append(f"*Provenance: {c.provenance.to_dict()}*")
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)


def render_json(cpds: list[CapabilityPowerDescriptor], *, generated_at: str) -> str:
    import json

    return json.dumps(
        {
            "generated_at": generated_at,
            "count": len(cpds),
            "capabilities": [c.to_dict() for c in cpds],
        },
        indent=2,
        sort_keys=False,
        default=str,
    )
