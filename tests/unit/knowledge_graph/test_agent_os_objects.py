"""Current Agent-OS model contract: WorkItem is the sole task authority."""

from agent_utilities.models.knowledge_graph import (
    AgentMailboxNode,
    AgentProcessNode,
    OutcomeEvaluationNode,
    RegistryNodeType,
    SessionCheckpointNode,
    WorkItemNode,
)


def test_removed_task_authorities_are_not_registry_members() -> None:
    names = {member.name for member in RegistryNodeType}
    assert "AGENT_TASK" not in names
    assert "AGENT_LEASE" not in names


def test_work_item_round_trip_carries_fencing_and_dependencies() -> None:
    node = WorkItemNode(
        id="workitem:opaque",
        name="WorkItem",
        tenant="tenant-ref",
        kind="agent_execution",
        status="leased",
        depends_on=["workitem:parent"],
        dep_count=1,
        lease_owner="worker-ref",
        lease_epoch=3,
        fencing_token=7,
        lease_expires_at=100.0,
    )
    restored = WorkItemNode.model_validate_json(node.model_dump_json())
    assert restored == node
    assert restored.type == RegistryNodeType.WORK_ITEM


def test_checkpoint_references_work_item_and_engine_lease() -> None:
    node = SessionCheckpointNode(
        id="checkpoint:opaque",
        name="checkpoint",
        session_id="session-ref",
        work_item_id="workitem:opaque",
        lease_id="lease-ref",
    )
    assert node.work_item_id == "workitem:opaque"
    assert node.lease_id == "lease-ref"


def test_outcome_is_evidence_not_a_task_status_projection() -> None:
    node = OutcomeEvaluationNode(
        id="outcome:opaque",
        name="outcome",
        reward=0.9,
        feedback_text="accepted",
        lease_id="lease-ref",
        dag_id="dag-ref",
    )
    assert node.lease_id == "lease-ref"
    assert node.dag_id == "dag-ref"


def test_non_task_agent_models_remain_available() -> None:
    process = AgentProcessNode(id="process:opaque", name="process")
    mailbox = AgentMailboxNode(id="mailbox:opaque", name="mailbox")
    assert process.spent_usd == 0.0
    assert mailbox.unread_count == 0
