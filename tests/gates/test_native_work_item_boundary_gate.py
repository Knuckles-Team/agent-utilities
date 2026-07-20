"""Meta-tests for the engine-native WorkItem authority gate."""

from __future__ import annotations

from pathlib import Path

from scripts.check_native_work_item_boundary import check


def _write(root: Path, relative: str, source: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def test_repository_satisfies_native_work_item_boundary() -> None:
    assert check(Path(__file__).resolve().parents[2]) == []


def test_gate_rejects_parallel_lifecycle_and_direct_write(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "agent_utilities/orchestration/org_runtime.py",
        """
class OrgPhase:
    RUNNING = "running"

class WorkItem:
    def transition(self, target):
        self.phase = target

def persist(backend, item):
    _write_node(
        backend,
        item.work_item_id,
        "WorkItem",
        {"workItemPhase": item.phase, "status": "running"},
    )
""",
    )
    findings = check(tmp_path)
    assert any(
        "parallel WorkItem lifecycle class 'OrgPhase'" in item for item in findings
    )
    assert any(
        "parallel WorkItem lifecycle class 'WorkItem'" in item for item in findings
    )
    assert any("direct WorkItem persistence" in item for item in findings)


def test_gate_rejects_tagged_lifecycle_batch(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "agent_utilities/orchestration/queue.py",
        """
def persist(engine):
    engine.ingest_graph_slice([
        {"type": "WorkItem", "status": "running", "fencing_token": 4}
    ])
""",
    )
    findings = check(tmp_path)
    assert any("direct WorkItem persistence" in item for item in findings)
    assert any("WorkItem lifecycle payload" in item for item in findings)


def test_gate_allows_authority_schemas_and_read_only_labels(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "agent_utilities/orchestration/work_item.py",
        """
class WorkItemStatus:
    READY = "ready"

def transition_work_item(backend):
    backend.add_node("id", "WorkItem", properties={"status": "ready"})
""",
    )
    _write(
        tmp_path,
        "agent_utilities/models/knowledge_graph.py",
        "class WorkItem:\n    status: str\n",
    )
    _write(
        tmp_path,
        "agent_utilities/orchestration/fleet_autoscaler.py",
        'WORK_ITEM_LABEL = "WorkItem"\n\ndef read(engine):\n'
        "    return engine.get_nodes_by_type(WORK_ITEM_LABEL)\n",
    )
    assert check(tmp_path) == []


def test_gate_limits_bus_to_initial_materialization(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "agent_utilities/messaging/bus_inbox.py",
        """
def _properties():
    work = WorkItemNode(status="ready")
    work["type"] = "WorkItem"
    return work
""",
    )
    assert check(tmp_path) == []

    _write(
        tmp_path,
        "agent_utilities/messaging/bus_inbox.py",
        """
def _properties():
    return WorkItemNode(status="ready")

def consume(engine, item):
    claim_specific(engine, item)
""",
    )
    findings = check(tmp_path)
    assert any(
        "AgentBus inbox invokes WorkItem transition" in item for item in findings
    )


def test_gate_rejects_parallel_transition_function(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "agent_utilities/orchestration/alternate_queue.py",
        "def renew_work_item(engine, item):\n    return engine.update_node(item)\n",
    )
    findings = check(tmp_path)
    assert any(
        "parallel WorkItem transition 'renew_work_item'" in item for item in findings
    )
