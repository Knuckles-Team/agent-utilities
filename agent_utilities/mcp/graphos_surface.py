"""The graph-os MCP surface: intent tools only (CONCEPT:AU-ECO.mcp.intent-surface-condensed-collapse).

graph-os serves the ecosystem's one tool contract
(:mod:`agent_utilities.mcp.intent_contract`): the intent tools ``ask``,
``find``, ``write``, ``act``, ``manage`` and ``why``, each taking ``action`` +
``params`` (+ optional natural-language ``intent``). Their ``action`` values are
operation ids from the generated action manifest
(``_graphos_action_manifest.GRAPHOS_ACTIONS``): ``"<tool>.<op>"`` or, for a
single-operation tool, ``"<tool>"``.

:data:`ROUTING_GROUPS` is the router's internal, job-based routing table: every
non-intent manifest family belongs to exactly one group (read, write, ingest,
code, run, …, admin). Groups organize ``describe`` output and the coverage
gates; they are not MCP tools.

The action-routed tools that implement the operations register on a private
FastMCP instance (:func:`backing_server`). They populate the shared
``kg_server.REGISTERED_TOOLS`` dispatch core — which the intent router, the REST
gateway and every operation dispatch through — and supply the per-operation
schemas ``describe`` serves. Host-native operations a server adds beside the
manifest (graph-os's browser control, A2A, RLM) register on the same backing
server and are reached through ``act``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from types import MappingProxyType
from typing import Any

from agent_utilities.mcp._graphos_action_manifest import GRAPHOS_ACTIONS
from agent_utilities.mcp.intent_contract import operation_id
from agent_utilities.mcp.tool_specs import INTENT_VERBS

#: Routing group whose operations are the raw engine namespaces.
ADMIN_GROUP = "admin"
#: Capabilities that satisfy an ``admin``-group operation.
ADMIN_CAPABILITIES = frozenset({"admin", "kg:admin", "mcp:admin"})


@dataclass(frozen=True)
class RoutingGroup:
    """One job-based group of the router's operation table."""

    name: str
    summary: str
    families: tuple[str, ...]


def _group(name: str, summary: str, families: str) -> RoutingGroup:
    return RoutingGroup(name=name, summary=summary, families=tuple(families.split()))


_ENGINE_FAMILIES = " ".join(
    sorted({op["tool"] for op in GRAPHOS_ACTIONS if op["tool"].startswith("engine_")})
)

#: The router's job-based routing table, in ``describe`` order. Every
#: non-intent family of the generated action manifest belongs to exactly one
#: group (gated by ``tests/unit/mcp/test_graphos_surface.py``).
ROUTING_GROUPS: tuple[RoutingGroup, ...] = (
    _group(
        "read",
        "Query, search, explain, NL/tabular/data questions, tables, documents, catalogs.",
        "graph_query graph_search graph_federated_search graph_search_synthesis "
        "graph_explain nl_query tabular_query ask_data graph_ask graph_table "
        "graph_catalog graph_document_tree graph_gis research_artifact",
    ),
    _group(
        "write",
        "Nodes, edges, memory writes, writeback, object edits, forks, sharing, tickets.",
        "graph_write graph_writeback object_edits graph_fork graph_share spec_ticket",
    ),
    _group(
        "ingest",
        "Ingestion, source sync and connectors, ETL, feeds, pipelines, media, data prep.",
        "graph_ingest source_sync source_connector source_drain graph_etl "
        "graph_feeds graph_pipeline document_process graph_media_sidecar "
        "ingest_sessions graph_data_prep",
    ),
    _group(
        "code",
        "Code intelligence, navigation, run version control, sandboxes.",
        "graph_code graph_code_nav graph_runvcs graph_sandbox",
    ),
    _group(
        "run",
        "Agents, workflows, goals, jobs, schedules, loops, durable runs, sessions.",
        "graph_orchestrate graph_agents graph_workflows graph_goals graph_jobs "
        "graph_schedules graph_loops graph_durable graph_sessions",
    ),
    _group(
        "message",
        "Conversations, reaching users on channels, the event bus and broker.",
        "graph_message graph_reach graph_bus graph_broker",
    ),
    _group(
        "reason",
        "Claims, arguments, epistemic status, evaluation, analysis, quant models.",
        "graph_claims graph_candidate_claims graph_argument graph_epistemic "
        "graph_evaluate graph_analyze quant graph_domain_ops",
    ),
    _group(
        "improve",
        "Feedback, learning, evolution proposals, graph engineering.",
        "graph_feedback graph_learn graph_evolution graph_engineering",
    ),
    _group(
        "ontology",
        "Ontology types, interfaces, derivations, object sets, indexes, permissions.",
        "graph_ontology ontology_classification_claims ontology_derive "
        "ontology_function ontology_interface ontology_leanix_sync "
        "ontology_link_materialize ontology_model_profile ontology_property_types "
        "ontology_repository_provenance ontology_sampling_profile "
        "ontology_value_types concept_registry skill_classify object_set "
        "object_index object_permissioning",
    ),
    _group(
        "observe",
        "Logs, traces, PromQL, audit, usage, failure analysis, charts.",
        "graph_logs graph_traces graph_observe graph_promql graph_audit "
        "usage_query graph_viz",
    ),
    _group(
        "govern",
        "Governance, compliance, secrets, configuration, connections, incidents.",
        "graph_governance graph_compliance graph_secret graph_config "
        "graph_configure graph_incident",
    ),
    _group(
        "memory",
        "Memory, context store, KV cache, KV checkpoints, projections.",
        "graph_memory graph_context graph_kvcache graph_kv_checkpoint "
        "graph_projection",
    ),
    _group(
        ADMIN_GROUP,
        "Raw engine namespaces (nodes, edges, rdf, txn, ...).",
        _ENGINE_FAMILIES,
    ),
)

ROUTING_GROUPS_BY_NAME: Mapping[str, RoutingGroup] = MappingProxyType(
    {group.name: group for group in ROUTING_GROUPS}
)


@cache
def group_for_tool_map() -> Mapping[str, str]:
    """``{backing tool: routing group}`` for every non-intent manifest family."""
    return MappingProxyType(
        {family: group.name for group in ROUTING_GROUPS for family in group.families}
    )


def group_for_tool(tool: str) -> str | None:
    """The routing group of backing ``tool`` (``None`` for an intent verb)."""
    return group_for_tool_map().get(tool)


@cache
def manifest_operations() -> Mapping[str, tuple[str, str | None]]:
    """``{operation id: (tool, action)}`` for every non-intent manifest operation.

    Raises ``RuntimeError`` on an operation no routing group claims, so a
    manifest change can never silently drop an operation from the router.
    """
    owner = group_for_tool_map()
    table: dict[str, tuple[str, str | None]] = {}
    unowned: set[str] = set()
    for op in GRAPHOS_ACTIONS:
        tool, action = op["tool"], op["action"]
        if tool in INTENT_VERBS:
            continue
        if tool not in owner:
            unowned.add(tool)
            continue
        table[operation_id(tool, action)] = (tool, action)
    if unowned:
        raise RuntimeError(
            "graph-os manifest families without a routing group: "
            + ", ".join(sorted(unowned))
        )
    return MappingProxyType(table)


def resolve_operation(op_id: str) -> tuple[str, str | None] | None:
    """``(tool, action)`` of a manifest operation id, or ``None``."""
    return manifest_operations().get(op_id)


def requires_admin(tool: str) -> bool:
    """Whether ``tool``'s operations need an administrative capability."""
    return group_for_tool(tool) == ADMIN_GROUP


def backing_server(mcp: Any) -> Any:
    """The private FastMCP instance holding the operations' backing tools."""
    backing = getattr(mcp, "_graphos_backing", None)
    if backing is None:
        raise RuntimeError("graph-os tool surface has not been registered")
    return backing


def backing_tools(mcp: Any) -> dict[str, Any]:
    """``{name: FastMCP tool}`` registered on ``mcp``'s backing server."""
    from agent_utilities.mcp.verbose_tools import _provider_tools

    return _provider_tools(backing_server(mcp))


def host_operations(mcp: Any) -> frozenset[str]:
    """Backing tools a host added beside the manifest (reached through ``act``)."""
    return frozenset(backing_tools(mcp)) - frozenset(
        tool for tool, _ in manifest_operations().values()
    )


def graphos_registrars() -> list[Any]:
    """The action-routed registrars that populate the dispatch core."""
    from agent_utilities.mcp import tools as au_tools

    return [
        au_tools.register_query_tools,
        au_tools.register_write_ingest_tools,
        au_tools.register_analysis_tools,
        au_tools.register_agent_execution_tools,
        au_tools.register_analyze_suite_tools,
        au_tools.register_state_tools,
        au_tools.register_ontology_tools,
        au_tools.register_reach_tools,
        au_tools.register_bus_tools,
        au_tools.register_candidate_claim_tools,
        au_tools.register_claim_tools,
        au_tools.register_secret_tools,
        au_tools.register_config_tools,
        au_tools.register_data_prep_tools,
        au_tools.register_engine_tools,
        lambda server: au_tools.register_engine_surface_tools(
            server, include_unserved_mining=False
        ),
        au_tools.register_domain_ops_tools,
        au_tools.register_evolution_tools,
        au_tools.register_governance_tools,
        au_tools.register_graph_engineering_tools,
        au_tools.register_audit_tools,
        au_tools.register_epistemic_tools,
        au_tools.register_incident_tools,
        au_tools.register_job_tools,
        au_tools.register_media_sidecar_tools,
        au_tools.register_compliance_tools,
        au_tools.register_workflow_tools,
        au_tools.register_argument_tools,
        au_tools.register_durable_tools,
    ]


def register_graphos_surface(mcp: Any, *, canonical: bool = False) -> Any:
    """Register graph-os's intent tools on ``mcp``; return the backing server.

    ``canonical=True`` registers every backing family regardless of the
    per-family ``<FAMILY>TOOL`` deployment toggles (catalog generation only).
    """
    from fastmcp import FastMCP

    from agent_utilities.mcp.tools import register_mcp_apps_tools
    from agent_utilities.mcp.tools.intent_tools import register_intent_tools
    from agent_utilities.mcp.verbose_tools import register_backing_tools

    manifest_operations()  # fail closed on an unowned operation
    backing = FastMCP("graph-os-backing")
    register_backing_tools(
        backing,
        registrars=graphos_registrars(),
        force_registration=canonical,
    )
    mcp._graphos_backing = backing
    register_intent_tools(mcp)
    register_mcp_apps_tools(mcp)
    return backing


__all__ = [
    "ADMIN_CAPABILITIES",
    "ADMIN_GROUP",
    "ROUTING_GROUPS",
    "ROUTING_GROUPS_BY_NAME",
    "RoutingGroup",
    "backing_server",
    "backing_tools",
    "graphos_registrars",
    "group_for_tool",
    "host_operations",
    "manifest_operations",
    "register_graphos_surface",
    "requires_admin",
    "resolve_operation",
]
