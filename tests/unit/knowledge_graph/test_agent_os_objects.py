#!/usr/bin/python
from __future__ import annotations

"""Graph-Native Agent-OS Objects — C3 (Phase 3a) node models.

CONCEPT:AU-OS.state.cognitive-scheduler-preemption — Graph-Native Agent-OS Objects (C3/Phase 3a)

Covers the 3 genuinely NEW ``RegistryNode`` subclasses this phase introduces
(``AgentLeaseNode``, ``AgentTaskNode``, ``AgentMailboxNode``) plus the fields
added to 3 EXISTING node types (``AgentProcessNode.cost_ceiling_usd`` /
``spent_usd``, ``SessionCheckpointNode.agent_task_id`` / ``lease_id``,
``OutcomeEvaluationNode.lease_id`` / ``dag_id``) instead of sprawling new
node types for those concerns (the reuse audit this phase requires).

``TeamComposition.to_durable_task_dag()``'s round trip through
``WorkflowStore`` is covered in ``tests/unit/test_agent_task_dag.py``
(engine-shaped fake needed there, not here).

@pytest.mark.concept("AU-OS.state.cognitive-scheduler-preemption")
"""

import pytest

from agent_utilities.models.knowledge_graph import (
    AgentLeaseNode,
    AgentMailboxNode,
    AgentProcessNode,
    AgentTaskNode,
    OutcomeEvaluationNode,
    RegistryNodeType,
    SessionCheckpointNode,
)

pytestmark = pytest.mark.concept("AU-OS.state.cognitive-scheduler-preemption")


# ---------------------------------------------------------------------------
# New node type enum members
# ---------------------------------------------------------------------------


def test_new_node_enum_members_present() -> None:
    names = {m.name for m in RegistryNodeType}
    for expected in ("AGENT_LEASE", "AGENT_TASK", "AGENT_MAILBOX"):
        assert expected in names


# ---------------------------------------------------------------------------
# AgentLeaseNode
# ---------------------------------------------------------------------------


def test_agent_lease_node_defaults() -> None:
    node = AgentLeaseNode(id="lease:task-1:abc", name="Lease: task-1")
    assert node.type == RegistryNodeType.AGENT_LEASE
    assert node.owner_token == ""
    assert node.resource_id == ""
    assert node.acquired_at == 0.0
    assert node.lease_expires_at == 0.0


def test_agent_lease_node_explicit_fields() -> None:
    node = AgentLeaseNode(
        id="lease:task-1:abc",
        name="Lease: task-1",
        owner_token="hostA:1:agent-dispatch",
        resource_id="task-1",
        acquired_at=100.0,
        lease_expires_at=3700.0,
    )
    assert node.owner_token == "hostA:1:agent-dispatch"
    assert node.resource_id == "task-1"
    assert node.lease_expires_at > node.acquired_at


def test_agent_lease_node_round_trips_json() -> None:
    node = AgentLeaseNode(
        id="lease:x",
        name="Lease: x",
        owner_token="hostB:2:agent-dispatch",
        resource_id="x",
        acquired_at=1.0,
        lease_expires_at=2.0,
    )
    restored = AgentLeaseNode.model_validate_json(node.model_dump_json())
    assert restored == node


# ---------------------------------------------------------------------------
# AgentTaskNode
# ---------------------------------------------------------------------------


def test_agent_task_node_defaults() -> None:
    node = AgentTaskNode(id="task-1", name="Task: task-1")
    assert node.type == RegistryNodeType.AGENT_TASK
    assert node.dag_id == ""
    assert node.depends_on_task_ids == []
    assert node.status == "pending"
    assert node.checkpoint_id is None


def test_agent_task_node_with_dependencies() -> None:
    node = AgentTaskNode(
        id="dag-1:task:b",
        name="Task: b",
        dag_id="dag-1",
        depends_on_task_ids=["dag-1:task:a"],
        status="blocked",
        checkpoint_id="checkpoint:1",
    )
    assert node.dag_id == "dag-1"
    assert node.depends_on_task_ids == ["dag-1:task:a"]
    assert node.status == "blocked"
    assert node.checkpoint_id == "checkpoint:1"


def test_agent_task_node_round_trips_json() -> None:
    node = AgentTaskNode(
        id="dag-1:task:c",
        name="Task: c",
        dag_id="dag-1",
        depends_on_task_ids=["dag-1:task:a", "dag-1:task:b"],
        status="running",
    )
    restored = AgentTaskNode.model_validate_json(node.model_dump_json())
    assert restored == node


# ---------------------------------------------------------------------------
# AgentMailboxNode
# ---------------------------------------------------------------------------


def test_agent_mailbox_node_defaults() -> None:
    node = AgentMailboxNode(id="mailbox:agent-1", name="Mailbox: agent-1")
    assert node.type == RegistryNodeType.AGENT_MAILBOX
    assert node.recipient_agent_id == ""
    assert node.messages == []
    assert node.unread_count == 0


def test_agent_mailbox_node_with_messages() -> None:
    node = AgentMailboxNode(
        id="mailbox:agent-1",
        name="Mailbox: agent-1",
        recipient_agent_id="agent-1",
        messages=[
            {"sender": "agent-2", "body": "hello", "sent_at": "2026-07-09T00:00:00Z"}
        ],
        unread_count=1,
    )
    assert node.recipient_agent_id == "agent-1"
    assert len(node.messages) == 1
    assert node.messages[0]["sender"] == "agent-2"
    assert node.unread_count == 1


# ---------------------------------------------------------------------------
# Extended EXISTING node types (reuse audit: no new node types for these)
# ---------------------------------------------------------------------------


def test_agent_process_node_cost_fields_default_unbounded() -> None:
    proc = AgentProcessNode(id="proc-1", name="Process: agent-1")
    assert proc.cost_ceiling_usd is None
    assert proc.spent_usd == 0.0


def test_agent_process_node_cost_fields_explicit() -> None:
    proc = AgentProcessNode(
        id="proc-1",
        name="Process: agent-1",
        cost_ceiling_usd=5.0,
        spent_usd=1.25,
    )
    assert proc.cost_ceiling_usd == 5.0
    assert proc.spent_usd == 1.25


def test_session_checkpoint_node_links_agent_task_and_lease() -> None:
    node = SessionCheckpointNode(
        id="checkpoint:1",
        name="Checkpoint: session-1",
        session_id="session-1",
        agent_task_id="dag-1:task:a",
        lease_id="lease:dag-1:task:a:abc",
    )
    assert node.agent_task_id == "dag-1:task:a"
    assert node.lease_id == "lease:dag-1:task:a:abc"


def test_session_checkpoint_node_agent_task_fields_default_empty() -> None:
    node = SessionCheckpointNode(id="checkpoint:2", name="c2", session_id="s2")
    assert node.agent_task_id == ""
    assert node.lease_id == ""


def test_outcome_evaluation_node_links_lease_and_dag() -> None:
    node = OutcomeEvaluationNode(
        id="outcome:1",
        name="Outcome: task a",
        reward=0.9,
        feedback_text="good result",
        lease_id="lease:dag-1:task:a:abc",
        dag_id="dag-1",
    )
    assert node.lease_id == "lease:dag-1:task:a:abc"
    assert node.dag_id == "dag-1"


def test_outcome_evaluation_node_defaults_empty() -> None:
    node = OutcomeEvaluationNode(
        id="outcome:2", name="Outcome: task b", reward=0.5, feedback_text="ok"
    )
    assert node.lease_id == ""
    assert node.dag_id == ""
