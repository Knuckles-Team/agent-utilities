"""Tests for the Enterprise Operations Causal Graph crosswalk (Codex X-2).

CONCEPT:AU-KG.ontology.ops-causal-crosswalk
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.ontology.ops_causal_crosswalk import (
    OPS_CAUSAL_EDGE_CHAIN,
    OPS_CAUSAL_NODE_CROSSWALK,
    OPS_CAUSAL_STAGES,
    OpsCausalStage,
    resolve_ops_causal_node_type,
    stage_of,
)
from agent_utilities.models.knowledge_graph import RegistryEdgeType, RegistryNodeType


def test_change_request_node_type_exists():
    """ServiceNow's planned 'Change' has its own hub type, distinct from Incident."""
    assert RegistryNodeType.CHANGE_REQUEST == "change_request"
    assert RegistryNodeType.CHANGE_REQUEST != RegistryNodeType.INCIDENT


def test_used_model_edge_type_exists():
    assert RegistryEdgeType.USED_MODEL == "used_model"


def test_resolve_known_connector_resources():
    langfuse_trace = resolve_ops_causal_node_type("langfuse-agent", "Trace")
    assert langfuse_trace is not None
    assert langfuse_trace.label == "Trace"
    assert langfuse_trace.stage == OpsCausalStage.OBSERVABILITY
    assert langfuse_trace.registry_type == RegistryNodeType.TRACE

    servicenow_change = resolve_ops_causal_node_type("servicenow-api", "Change")
    assert servicenow_change is not None
    assert servicenow_change.label == "ChangeRequest"
    assert servicenow_change.registry_type == RegistryNodeType.CHANGE_REQUEST

    gitlab_commit = resolve_ops_causal_node_type("gitlab-api", "Commit")
    assert gitlab_commit is not None
    assert gitlab_commit.label == "Commit"
    # Deliberately no RegistryNodeType companion — identity join on the same
    # ``:Commit`` label the first-party git-history ingest already writes.
    assert gitlab_commit.registry_type is None

    leanix_capability = resolve_ops_causal_node_type("leanix-agent", "BusinessCapability")
    assert leanix_capability is not None
    assert leanix_capability.registry_type == RegistryNodeType.CAPABILITY


def test_resolve_unknown_pair_returns_none():
    """Unresolved stays unresolved — never guessed (same discipline as
    ``connector_manifest.nearest_hub_class``)."""
    assert resolve_ops_causal_node_type("gitlab-api", "NoSuchResource") is None
    assert resolve_ops_causal_node_type("no-such-connector", "Trace") is None


def test_stage_of_reverse_lookup():
    assert stage_of("Commit") == OpsCausalStage.CHANGE
    assert stage_of("Incident") == OpsCausalStage.TICKET
    assert stage_of("Capability") == OpsCausalStage.OWNERSHIP
    assert stage_of("nonexistent-label") is None


def test_every_crosswalk_entry_has_a_valid_stage():
    for connector, resources in OPS_CAUSAL_NODE_CROSSWALK.items():
        for resource, hub in resources.items():
            assert hub.stage in OPS_CAUSAL_STAGES, (
                f"{connector}.{resource} -> unknown stage {hub.stage!r}"
            )


def test_edge_chain_only_new_edge_type_is_used_model():
    """Every edge type in the spine is reused from the existing hub vocabulary
    except USED_MODEL — the one genuinely missing edge (see module docstring)."""
    reused = {
        RegistryEdgeType.AFFECTS,
        RegistryEdgeType.DEPLOYS_SOFTWARE,
        RegistryEdgeType.PART_OF,
        RegistryEdgeType.EXECUTED_BY,
        RegistryEdgeType.USED_TOOL,
        RegistryEdgeType.CAUSED_INCIDENT,
        RegistryEdgeType.RESOLVED_INCIDENT,
        RegistryEdgeType.SUPPORTS,
        RegistryEdgeType.APPLIES_TO,
        RegistryEdgeType.OWNS_SYSTEM,
        RegistryEdgeType.GOVERNS,
        RegistryEdgeType.HAS_EVIDENCE,
    }
    seen_edge_types = {edge for _, edge, _, _, _ in OPS_CAUSAL_EDGE_CHAIN}
    assert seen_edge_types == reused | {RegistryEdgeType.USED_MODEL}


def test_edge_chain_stages_are_all_declared():
    for from_stage, _edge, to_stage, _mechanism, _reversed in OPS_CAUSAL_EDGE_CHAIN:
        assert from_stage in OPS_CAUSAL_STAGES
        assert to_stage in OPS_CAUSAL_STAGES


# ── Governance-process connector crosswalk (G17) ────────────────────────────


def test_governance_process_node_types_resolve_to_governance_stage():
    """The descriptive process-definition nodes each connector actually writes
    (confirmed against their own kg_ingest.py) join the GOVERNANCE stage —
    same causal role as Policy/ComplianceControl/Evidence."""
    governance_pairs = [
        ("camunda-mcp", "BusinessProcess"),
        ("aris-mcp", "ProcessModel"),
        ("aris-mcp", "EPCFunction"),
        ("aris-mcp", "EPCEvent"),
        ("aris-mcp", "EPCRule"),
        ("aris-mcp", "ProcessConnection"),
        ("archimate-mcp", "BusinessProcess"),
        ("archimate-mcp", "BusinessFunction"),
        ("archimate-mcp", "BusinessInteraction"),
        ("archimate-mcp", "ApplicationProcess"),
        ("egeria-mcp", "GovernanceRule"),
        ("onetrust-api", "AssessmentTemplate"),
        ("governance-import", "WorkflowDefinition"),
        ("governance-import", "WorkflowStep"),
    ]
    for connector, resource in governance_pairs:
        hub = resolve_ops_causal_node_type(connector, resource)
        assert hub is not None, f"{connector}.{resource} unresolved"
        assert hub.stage == OpsCausalStage.GOVERNANCE


def test_camunda_deployment_is_a_change_event():
    hub = resolve_ops_causal_node_type("camunda-mcp", "Deployment")
    assert hub is not None
    assert hub.stage == OpsCausalStage.CHANGE


def test_camunda_process_instance_and_onetrust_assessment_are_tickets():
    """A running process instance / an in-flight DPIA assessment are the
    tracked units of work being approved, not the governing procedure itself."""
    instance = resolve_ops_causal_node_type("camunda-mcp", "ProcessInstance")
    assert instance is not None
    assert instance.stage == OpsCausalStage.TICKET

    assessment = resolve_ops_causal_node_type("onetrust-api", "Assessment")
    assert assessment is not None
    assert assessment.stage == OpsCausalStage.TICKET


def test_erpnext_agent_has_no_guessed_workflow_entry():
    """erpnext-agent's kg_ingest.py doesn't ingest a Workflow/WorkflowState
    doctype today — unresolved stays unresolved, never guessed."""
    assert resolve_ops_causal_node_type("erpnext-agent", "Workflow") is None


def test_workflow_definition_reverse_lookup():
    assert stage_of("WorkflowDefinition") == OpsCausalStage.GOVERNANCE
    assert stage_of("GovernanceRule") == OpsCausalStage.GOVERNANCE


def test_new_governance_entries_do_not_disturb_existing_crosswalk():
    """The pre-existing connectors/resources are byte-for-byte unaffected by
    the G17 extension (additive-only)."""
    langfuse_trace = resolve_ops_causal_node_type("langfuse-agent", "Trace")
    assert langfuse_trace is not None
    assert langfuse_trace.stage == OpsCausalStage.OBSERVABILITY
    servicenow_change = resolve_ops_causal_node_type("servicenow-api", "Change")
    assert servicenow_change is not None
    assert servicenow_change.label == "ChangeRequest"
    assert stage_of("Commit") == OpsCausalStage.CHANGE
    assert stage_of("Incident") == OpsCausalStage.TICKET
