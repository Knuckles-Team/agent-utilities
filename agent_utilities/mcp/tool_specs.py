#!/usr/bin/python
"""Immutable, profile-aware Graph-OS capability universe.

The generated action manifest is the distribution-owned inventory of Graph-OS
tool families and their actions.  Runtime registration is deliberately *not* a
source of truth here: ``REGISTERED_TOOLS`` is an execution table and can change
as servers are built, tests install fakes, or optional features are loaded.

This module projects the manifest into frozen :class:`ToolSpec` values.  The
core profile is always present, the six intent verbs form the bounded default
surface, and optional families declare the feature that enables them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from agent_utilities.mcp._graphos_action_manifest import GRAPHOS_ACTIONS
from agent_utilities.mcp.optional_tool_features import (
    OPTIONAL_TOOL_FEATURES,
    SUPPORTED_FEATURES,
)

INTENT_VERBS: tuple[str, ...] = ("ask", "find", "write", "act", "manage", "why")

#: Every granular tool's allowed intent verb(s); the first entry is primary.
#: CPD generation copies this order exactly and runtime routing rejects drift.
TOOL_VERBS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        # ── reads / NL / search / analysis ──
        "ask_data": ("ask",),
        "nl_query": ("ask",),
        "graph_query": ("ask",),
        "graph_ask": ("ask",),
        "graph_search": ("ask",),
        "graph_search_synthesis": ("ask",),
        "graph_analyze": ("ask", "why"),
        "graph_context": ("ask",),
        "graph_document_tree": ("ask", "write"),
        "graph_engineering": ("ask", "write"),
        "graph_table": ("ask", "write"),
        "graph_promql": ("ask",),
        "graph_federated_search": ("ask",),
        "graph_code": ("ask",),
        "graph_code_nav": ("ask",),
        "graph_reach": ("act",),
        "graph_gis": ("ask",),
        "usage_query": ("ask",),
        "concept_registry": ("find", "ask"),
        "object_index": ("ask", "find"),
        "object_set": ("ask", "write"),
        "research_artifact": ("ask", "write"),
        "quant": ("ask", "act"),
        "engine_query": ("ask",),
        "engine_analytics": ("ask",),
        "engine_datascience": ("ask",),
        "engine_finance": ("ask", "act"),
        "engine_mining": ("ask",),
        "engine_graph": ("ask", "write"),
        "graph_mine": ("ask",),
        "graph_mine_deep": ("act", "ask"),
        "graph_learn": ("act", "ask"),
        "engine_graphlearn": ("act", "ask"),
        "graph_ops_causal": ("why", "ask"),
        "graph_traces": ("ask", "why"),
        "graph_audit": ("why", "ask"),
        "graph_compliance": ("ask", "why"),
        "graph_epistemic": ("why", "ask"),
        "graph_incident": ("ask", "act"),
        "graph_argument": ("why", "write", "ask"),
        # ── writes / ingest / persist ──
        "graph_write": ("write",),
        "graph_ingest": ("write",),
        "graph_writeback": ("write",),
        "graph_etl": ("write",),
        "source_sync": ("write",),
        "source_connector": ("manage", "write"),
        "source_drain": ("write",),
        "ingest_sessions": ("write",),
        "object_edits": ("write",),
        "ontology_derive": ("write",),
        "ontology_link_materialize": ("write",),
        "ontology_leanix_sync": ("write",),
        "document_process": ("write",),
        "spec_ticket": ("write", "ask"),
        "engine_nodes": ("write", "ask"),
        "engine_edges": ("write",),
        "engine_blob": ("write",),
        "engine_rdf": ("write", "ask"),
        "engine_timeseries": ("write", "ask"),
        "graph_share": ("write", "manage"),
        "graph_feedback": ("why", "write"),
        "ontology_function": ("act", "write"),
        # ── act / orchestrate / execute / schedule ──
        "graph_orchestrate": ("act",),
        "graph_jobs": ("act", "ask"),
        "graph_agents": ("act",),
        "graph_workflows": ("act", "ask", "manage"),
        "graph_evolution": ("act",),
        "graph_rlm": ("act",),
        "graph_governance": ("act", "manage"),
        "graph_domain_ops": ("act",),
        "graph_loops": ("act",),
        "graph_goals": ("act",),
        "graph_sandbox": ("act",),
        "graph_runvcs": ("act",),
        "graph_fork": ("act",),
        "graph_bus": ("act",),
        "graph_broker": ("act",),
        "graph_message": ("act",),
        "graph_feeds": ("act", "ask"),
        "graph_research": ("ask", "act"),
        "engine_txn": ("act",),
        "engine_consensus": ("act",),
        "engine_channels": ("act",),
        "engine_streaming": ("act",),
        "engine_broker": ("act", "manage", "ask"),
        "engine_ledger": ("write", "act"),
        # ── manage / configure / admin ──
        "graph_configure": ("manage",),
        "graph_secret": ("manage",),
        "graph_sessions": ("manage", "ask"),
        "graph_kvcache": ("manage",),
        "graph_schedules": ("manage", "act"),
        "graph_ontology": ("write", "ask", "manage"),
        "ontology_property_types": ("manage", "ask"),
        "ontology_value_types": ("manage", "ask"),
        "ontology_interface": ("manage", "ask"),
        "ontology_sampling_profile": ("manage", "ask"),
        "object_permissioning": ("manage",),
        "engine_tenants": ("manage",),
        "engine_lifecycle": ("manage", "act"),
        "engine_resharding": ("manage",),
        "engine_rbac": ("manage",),
        "engine_admin": ("manage",),
        # ── why / explain / evaluate / observe ──
        "graph_explain": ("why",),
        "graph_evaluate": ("why",),
        "graph_observe": ("why", "ask"),
        "graph_memory": ("write", "ask"),
        "engine_reasoning": ("why",),
    }
)


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """One canonical Graph-OS tool family.

    ``actions`` is empty for a single-operation family.  ``feature`` is
    ``None`` for required capabilities and names the package feature for an
    optional family.  Intent verbs are descriptors and entry points, not
    resolver targets.
    """

    name: str
    actions: tuple[str, ...]
    feature: str | None
    surface: Literal["granular", "intent"]

    @property
    def required(self) -> bool:
        return self.feature is None


def _build_specs() -> tuple[ToolSpec, ...]:
    actions_by_tool: dict[str, set[str]] = {}
    for entry in GRAPHOS_ACTIONS:
        name = entry["tool"]
        action = entry["action"]
        actions = actions_by_tool.setdefault(name, set())
        if action is not None:
            actions.add(action)

    specs = tuple(
        ToolSpec(
            name=name,
            actions=tuple(sorted(actions)),
            feature=OPTIONAL_TOOL_FEATURES.get(name),
            surface="intent" if name in INTENT_VERBS else "granular",
        )
        for name, actions in sorted(actions_by_tool.items())
    )
    names = {spec.name for spec in specs}
    missing_intents = sorted(set(INTENT_VERBS) - names)
    if missing_intents:
        raise RuntimeError(
            f"generated Graph-OS manifest is missing intent verbs: {missing_intents}"
        )
    unknown_feature_tools = sorted(set(OPTIONAL_TOOL_FEATURES) - names)
    if unknown_feature_tools:
        raise RuntimeError(
            "generated Graph-OS manifest is missing optional tool families: "
            f"{unknown_feature_tools}"
        )
    granular_names = names - set(INTENT_VERBS)
    missing_verb_authority = sorted(granular_names - set(TOOL_VERBS))
    stale_verb_authority = sorted(set(TOOL_VERBS) - granular_names)
    invalid_verb_authority = sorted(
        name
        for name, verbs in TOOL_VERBS.items()
        if not verbs
        or len(set(verbs)) != len(verbs)
        or not set(verbs) <= set(INTENT_VERBS)
    )
    if missing_verb_authority or stale_verb_authority or invalid_verb_authority:
        raise RuntimeError(
            "Graph-OS intent-verb authority disagrees with the canonical tool "
            "universe: "
            f"missing={missing_verb_authority}; stale={stale_verb_authority}; "
            f"invalid={invalid_verb_authority}"
        )
    return specs


TOOL_SPECS: tuple[ToolSpec, ...] = _build_specs()
TOOL_SPECS_BY_NAME = MappingProxyType({spec.name: spec for spec in TOOL_SPECS})


def canonical_tool_specs(
    *,
    features: frozenset[str] = frozenset(),
    include_intent: bool = True,
) -> tuple[ToolSpec, ...]:
    """Return the immutable tool universe for one explicit feature profile."""
    unknown = features - SUPPORTED_FEATURES
    if unknown:
        raise ValueError(f"unknown Graph-OS features: {sorted(unknown)}")
    return tuple(
        spec
        for spec in TOOL_SPECS
        if (spec.feature is None or spec.feature in features)
        and (include_intent or spec.surface != "intent")
    )


def canonical_tool_names(
    *,
    features: frozenset[str] = frozenset(),
    include_intent: bool = True,
) -> frozenset[str]:
    """Return canonical family names for one explicit feature profile."""
    return frozenset(
        spec.name
        for spec in canonical_tool_specs(
            features=features, include_intent=include_intent
        )
    )


def feature_for_tool(name: str) -> str | None:
    """Return the enabling feature for ``name``, or ``None`` for core tools."""
    spec = TOOL_SPECS_BY_NAME.get(name)
    if spec is None:
        raise KeyError(name)
    return spec.feature


__all__ = [
    "INTENT_VERBS",
    "SUPPORTED_FEATURES",
    "TOOL_VERBS",
    "TOOL_SPECS",
    "TOOL_SPECS_BY_NAME",
    "ToolSpec",
    "canonical_tool_names",
    "canonical_tool_specs",
    "feature_for_tool",
]
