#!/usr/bin/python
from __future__ import annotations

"""Enterprise Operations Causal Graph — connector crosswalk (Codex X-2).

CONCEPT:AU-KG.ontology.ops-causal-crosswalk

X-2 joins entities ALREADY ingested by six independent connectors into one
operations causal chain::

    Langfuse Trace/Generation
        -> Agent / Tool / Model
        -> System (service/application)
        -> Container / ContainerStack (deployment)
        -> Commit / MergeRequest (the change)
        -> Incident / ChangeRequest (the ticket)
        -> Capability + Owner (LeanIX)
        -> Policy / ComplianceControl + Evidence (governance)

Every one of those connectors already ships a **Connector Ontology Manifest**
(``ontology/connector_manifests/<connector>/connector_manifest.yml``) whose
``schema_mappings[resource].ontology_class`` crosswalks a resource onto an OWL
class wired into the hub ``ontology.ttl`` (via
:data:`connector_manifest.HUB_NAME_HEURISTIC_CROSSWALK` / D16). That system is
deliberately narrow — it only crosswalks onto the small, OWL-reasoning-gated
class set that's actually declared ``owl:Class`` in ``ontology.ttl`` (governed
by ``manifest_compiler.AntiSprawlError``), and several of the resources this
causal chain needs are either unresolved there (``Commit``, ``MergeRequest``,
``Deployment``, ``Change``, ``ConfigurationItem``, ``BusinessCapability`` are
all ``null`` in their shipped manifests today) or crosswalk to something too
generic to join on (LeanIX's fact-sheet subtypes all collapse to the bare
``FactSheet`` OWL class).

This module is the crosswalk **X-2 actually needs to join on** — it targets
the broader, already-populated :class:`~agent_utilities.models.knowledge_graph.RegistryNodeType`
/ :class:`~agent_utilities.models.knowledge_graph.RegistryEdgeType` LPG type
system that every traversal primitive (``engine.get_blast_radius``,
``graph_mine``, ``StructuralCausalModel``) actually operates over, rather than
the stricter OWL-manifest crosswalk. It reuses the SAME resolution discipline
(name-heuristic table + explicit per-connector override, "unresolved stays
unresolved — never guessed") documented in ``connector_manifest.py``.

Only ONE hub type was genuinely missing to complete the chain — ServiceNow's
``Change`` (a planned change ticket) has no home distinct from ``Incident``
(an unplanned one) — added as :attr:`RegistryNodeType.CHANGE_REQUEST`. Every
other stage reuses an existing hub type/edge (``System``, ``Container``,
``ContainerStack``, ``MergeRequest``, ``Capability``, ``ComplianceControl``,
``Policy``, ``Evidence`` node types; ``AFFECTS``, ``CAUSED_INCIDENT``,
``RESOLVED_INCIDENT``, ``DEPLOYS_SOFTWARE``, ``OWNS_SYSTEM``, ``GOVERNS``,
``HAS_EVIDENCE``, ``SUPPORTS``, ``PART_OF``, ``EXECUTED_BY``, ``USED_TOOL``
edge types — see :data:`OPS_CAUSAL_EDGE_CHAIN` for the exact, causally-
directed spine) — plus one new edge, :attr:`RegistryEdgeType.USED_MODEL`, for
the one Langfuse leg ("this generation invoked this model") nothing else
already covers.
"""

from dataclasses import dataclass

from ...models.knowledge_graph import RegistryEdgeType, RegistryNodeType

__all__ = [
    "OpsCausalStage",
    "OPS_CAUSAL_STAGES",
    "OpsCausalHubType",
    "OPS_CAUSAL_NODE_CROSSWALK",
    "OPS_CAUSAL_EDGE_CHAIN",
    "resolve_ops_causal_node_type",
    "stage_of",
]


@dataclass(frozen=True)
class OpsCausalHubType:
    """One crosswalk target: the graph LABEL a node actually carries, plus its
    (optional) companion :class:`RegistryNodeType` for callers that write
    through the typed registry-graph path rather than a raw label.

    Two vocabularies coexist in this KG: the PascalCase graph LABEL every
    Cypher traversal/manifest ``ontology_class`` matches on (``labels(t)[0]``
    in ``engine.get_blast_radius``; ``Incident``/``System``/``Commit`` in the
    connector manifests), and the snake_case :class:`RegistryNodeType` enum
    the registry-graph dataclasses (``RegistryNode.type``) use. Most causal-
    chain stages have both; ``label`` is always populated, ``registry_type``
    is ``None`` where the entity is only ever written as a raw label (e.g.
    ``git_history.py``'s ``:Commit`` nodes never go through the enum).
    """

    label: str
    stage: str
    registry_type: RegistryNodeType | None = None


class OpsCausalStage:
    """Ordered stage names of the operations causal spine (root cause -> governance)."""

    OBSERVABILITY = "observability"  # Langfuse Trace/Generation/Score
    EXECUTOR = "executor"  # Agent / Tool / Model
    SERVICE = "service"  # System (application/service)
    DEPLOYMENT = "deployment"  # Container / ContainerStack
    CHANGE = "change"  # Commit / MergeRequest
    TICKET = "ticket"  # Incident / ChangeRequest
    OWNERSHIP = "ownership"  # Capability + owning Person/Team/Organization
    GOVERNANCE = "governance"  # Policy / ComplianceControl / Evidence


#: Ordered — matches the causal chain's intended traversal direction
#: (root cause, upstream, first -> blast radius, downstream, last).
OPS_CAUSAL_STAGES: tuple[str, ...] = (
    OpsCausalStage.OBSERVABILITY,
    OpsCausalStage.EXECUTOR,
    OpsCausalStage.SERVICE,
    OpsCausalStage.DEPLOYMENT,
    OpsCausalStage.CHANGE,
    OpsCausalStage.TICKET,
    OpsCausalStage.OWNERSHIP,
    OpsCausalStage.GOVERNANCE,
)

#: connector -> connector resource name -> :class:`OpsCausalHubType`.
#: Only the resources the causal chain actually joins on are listed here — this
#: is NOT a replacement for each connector's full OWL manifest crosswalk, it is
#: the narrower LPG-label crosswalk X-2's join layer resolves against.
OPS_CAUSAL_NODE_CROSSWALK: dict[str, dict[str, OpsCausalHubType]] = {
    "langfuse-agent": {
        "Trace": OpsCausalHubType(
            "Trace", OpsCausalStage.OBSERVABILITY, RegistryNodeType.TRACE
        ),
        "Generation": OpsCausalHubType(
            "Generation", OpsCausalStage.OBSERVABILITY, RegistryNodeType.GENERATION
        ),
        "Observation": OpsCausalHubType(
            "Observation", OpsCausalStage.OBSERVABILITY, RegistryNodeType.OBSERVATION
        ),
        "Score": OpsCausalHubType(
            "Score", OpsCausalStage.OBSERVABILITY, RegistryNodeType.ONLINE_SCORE
        ),
        "Model": OpsCausalHubType(
            "Model", OpsCausalStage.EXECUTOR, RegistryNodeType.MODEL
        ),
    },
    "container-manager-mcp": {
        # k8s Deployment/StatefulSet/DaemonSet are all "a managed set of pods" —
        # the same shape a ContainerStack already models (docker-compose stack).
        "Deployment": OpsCausalHubType(
            "ContainerStack",
            OpsCausalStage.DEPLOYMENT,
            RegistryNodeType.CONTAINER_STACK,
        ),
        "StatefulSet": OpsCausalHubType(
            "ContainerStack",
            OpsCausalStage.DEPLOYMENT,
            RegistryNodeType.CONTAINER_STACK,
        ),
        "DaemonSet": OpsCausalHubType(
            "ContainerStack",
            OpsCausalStage.DEPLOYMENT,
            RegistryNodeType.CONTAINER_STACK,
        ),
        "Pod": OpsCausalHubType(
            "Container", OpsCausalStage.DEPLOYMENT, RegistryNodeType.CONTAINER
        ),
        "SwarmService": OpsCausalHubType(
            "System", OpsCausalStage.SERVICE, RegistryNodeType.SYSTEM
        ),
        "K8sService": OpsCausalHubType(
            "System", OpsCausalStage.SERVICE, RegistryNodeType.SYSTEM
        ),
    },
    "gitlab-api": {
        # Same label the first-party git-history ingest already writes
        # (agent_utilities.knowledge_graph.enrichment.git_history: ``:Commit``
        # nodes keyed ``commit:<sha>``, written as a raw label — never through
        # RegistryNodeType). GitLab's own Commit resource IS that same commit
        # (identity join on sha), so ``registry_type`` is deliberately None.
        "Commit": OpsCausalHubType("Commit", OpsCausalStage.CHANGE),
        "MergeRequest": OpsCausalHubType(
            "MergeRequest", OpsCausalStage.CHANGE, RegistryNodeType.MERGE_REQUEST
        ),
        "Project": OpsCausalHubType(
            "SoftwareProject", OpsCausalStage.SERVICE, RegistryNodeType.SOFTWARE_PROJECT
        ),
        "Issue": OpsCausalHubType(
            "Incident", OpsCausalStage.TICKET, RegistryNodeType.INCIDENT
        ),
    },
    "repository-manager": {
        "GitRepository": OpsCausalHubType(
            "SoftwareProject", OpsCausalStage.SERVICE, RegistryNodeType.SOFTWARE_PROJECT
        ),
        "Project": OpsCausalHubType(
            "SoftwareProject", OpsCausalStage.SERVICE, RegistryNodeType.SOFTWARE_PROJECT
        ),
    },
    "servicenow-api": {
        "Incident": OpsCausalHubType(
            "Incident", OpsCausalStage.TICKET, RegistryNodeType.INCIDENT
        ),
        # The one node truly missing before X-2 (see module docstring).
        "Change": OpsCausalHubType(
            "ChangeRequest", OpsCausalStage.TICKET, RegistryNodeType.CHANGE_REQUEST
        ),
        # A CMDB CI IS the service/application/asset the chain reasons about.
        "ConfigurationItem": OpsCausalHubType(
            "System", OpsCausalStage.SERVICE, RegistryNodeType.SYSTEM
        ),
        "Person": OpsCausalHubType(
            "Person", OpsCausalStage.OWNERSHIP, RegistryNodeType.PERSON
        ),
    },
    "atlassian-agent": {
        "Issue": OpsCausalHubType(
            "Incident", OpsCausalStage.TICKET, RegistryNodeType.INCIDENT
        ),
    },
    "leanix-agent": {
        # More precise than the manifest's blanket ``FactSheet`` crosswalk —
        # CAPABILITY is the hub type the causal chain's OWNERSHIP stage joins on.
        "BusinessCapability": OpsCausalHubType(
            "Capability", OpsCausalStage.OWNERSHIP, RegistryNodeType.CAPABILITY
        ),
        "Application": OpsCausalHubType(
            "System", OpsCausalStage.SERVICE, RegistryNodeType.SYSTEM
        ),
    },
}


def resolve_ops_causal_node_type(
    connector: str, resource: str
) -> OpsCausalHubType | None:
    """Resolve ``(connector, resource)`` to its :class:`OpsCausalHubType`.

    Returns ``None`` (never a guess) when the pair isn't in the crosswalk —
    same "unresolved stays unresolved" discipline as
    :func:`connector_manifest.nearest_hub_class`.
    """
    return OPS_CAUSAL_NODE_CROSSWALK.get(connector, {}).get(resource)


def stage_of(label: str) -> str | None:
    """Best-effort reverse lookup: which causal stage a hub graph LABEL belongs to."""
    for mapping in OPS_CAUSAL_NODE_CROSSWALK.values():
        for hub in mapping.values():
            if hub.label == label:
                return hub.stage
    return None


#: The causal spine itself: ``(from_stage, edge_type, to_stage, mechanism)``,
#: where ``from_stage -> to_stage`` is the CAUSAL direction (cause -> effect —
#: so ``get_causal_ancestors(failure)`` walks toward root causes and
#: ``get_causal_descendants(change)`` walks toward blast radius). Every edge
#: type here is REUSED from the existing hub vocabulary except ``USED_MODEL``
#: (see module docstring). Note the causal direction is sometimes the LOGICAL
#: REVERSE of the KG relationship's usual write direction (e.g. an ingested
#: ``Agent -[:PART_OF]-> Service`` edge records STRUCTURE — "the agent is part
#: of the service" — but the CAUSAL direction for root-cause purposes is
#: "the service's failure causes the agent's failure", i.e. ``SERVICE ->
#: EXECUTOR``); those hops are flagged ``reversed=True`` and the join layer
#: (``enrichment.ops_causal_graph``) is responsible for adding the
#: :class:`~agent_utilities.knowledge_graph.core.formal_reasoning_core.CausalEdge`
#: source/target swapped accordingly when it materializes causal links from
#: raw KG edges of that type.
OPS_CAUSAL_EDGE_CHAIN: tuple[tuple[str, RegistryEdgeType, str, str, bool], ...] = (
    (
        OpsCausalStage.CHANGE,
        RegistryEdgeType.AFFECTS,
        OpsCausalStage.DEPLOYMENT,
        "commit/merge-request shipped in this deployment",
        False,
    ),
    (
        OpsCausalStage.CHANGE,
        RegistryEdgeType.AFFECTS,
        OpsCausalStage.SERVICE,
        "commit/merge-request touched this service directly",
        False,
    ),
    (
        OpsCausalStage.DEPLOYMENT,
        RegistryEdgeType.DEPLOYS_SOFTWARE,
        OpsCausalStage.SERVICE,
        "the deployment/container runs this service",
        False,
    ),
    (
        OpsCausalStage.SERVICE,
        RegistryEdgeType.PART_OF,
        OpsCausalStage.EXECUTOR,
        "service hosts the agent/tool — a service incident causes the "
        "agent/tool that runs on it to misbehave (KG edge is written "
        "agent -[:PART_OF]-> service; causal direction is its reverse)",
        True,
    ),
    (
        OpsCausalStage.EXECUTOR,
        RegistryEdgeType.EXECUTED_BY,
        OpsCausalStage.OBSERVABILITY,
        "the agent's execution produces the trace/generation (KG edge is "
        "trace -[:EXECUTED_BY]-> agent; causal direction is its reverse)",
        True,
    ),
    (
        OpsCausalStage.EXECUTOR,
        RegistryEdgeType.USED_TOOL,
        OpsCausalStage.OBSERVABILITY,
        "the tool invocation is recorded in the trace/generation (reverse "
        "of the recorded trace -[:USED_TOOL]-> tool edge)",
        True,
    ),
    (
        OpsCausalStage.EXECUTOR,
        RegistryEdgeType.USED_MODEL,
        OpsCausalStage.OBSERVABILITY,
        "the model run is recorded in the generation (reverse of the "
        "recorded generation -[:USED_MODEL]-> model edge)",
        True,
    ),
    (
        OpsCausalStage.CHANGE,
        RegistryEdgeType.CAUSED_INCIDENT,
        OpsCausalStage.TICKET,
        "the change caused this incident/ticket",
        False,
    ),
    (
        OpsCausalStage.TICKET,
        RegistryEdgeType.RESOLVED_INCIDENT,
        OpsCausalStage.TICKET,
        "a change-request ticket resolved this incident",
        False,
    ),
    (
        OpsCausalStage.SERVICE,
        RegistryEdgeType.SUPPORTS,
        OpsCausalStage.OWNERSHIP,
        "service supports this business capability",
        False,
    ),
    (
        OpsCausalStage.TICKET,
        RegistryEdgeType.APPLIES_TO,
        OpsCausalStage.OWNERSHIP,
        "incident/ticket applies to this capability",
        False,
    ),
    (
        OpsCausalStage.OWNERSHIP,
        RegistryEdgeType.OWNS_SYSTEM,
        OpsCausalStage.OWNERSHIP,
        "owner (person/team) owns this capability/system",
        False,
    ),
    (
        OpsCausalStage.GOVERNANCE,
        RegistryEdgeType.GOVERNS,
        OpsCausalStage.OWNERSHIP,
        "policy/control governs this capability/system",
        False,
    ),
    (
        OpsCausalStage.GOVERNANCE,
        RegistryEdgeType.HAS_EVIDENCE,
        OpsCausalStage.GOVERNANCE,
        "policy/control has this evidence",
        False,
    ),
)
