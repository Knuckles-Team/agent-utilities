import os
import json
import pytest
from pathlib import Path
from agent_utilities.sdd import SDDManager
from agent_utilities.models import (
    ProjectConstitution,
    FeatureSpec,
    ImplementationPlan,
    TaskList,
    Task,
    TaskPhase,
    TaskStatus,
)

@pytest.fixture
def sdd_manager(tmp_path):
    """Fixture to provide a clean SDDManager with a temporary workspace."""
    return SDDManager(workspace_path=tmp_path)

def test_sdd_manager_serialization(sdd_manager):
    """Verify that ProjectConstitution can be saved and loaded correctly."""
    constitution = ProjectConstitution(
        vision="Test Vision",
        mission="Test Mission",
        core_principles=["Clean Architecture"],
        tech_stack={"backend": "Python"},
    )

    path = sdd_manager.save(constitution)
    assert path.exists()
    assert "constitution.json" in str(path)

    loaded = sdd_manager.load(ProjectConstitution)
    assert loaded is not None
    assert loaded.vision == "Test Vision"
    assert "Clean Architecture" in loaded.core_principles

def test_feature_spec_persistence(sdd_manager):
    """Verify that FeatureSpec requires feature_id and persists correctly."""
    spec = FeatureSpec(
        name="Test Feature",
        description="A feature for testing",
        functional_requirements=["Req 1"],
    )

    # Save/Load with feature_id
    path = sdd_manager.save(spec, feature_id="F1")
    assert "specs/F1.json" in str(path).replace("\\", "/")

    loaded = sdd_manager.load(FeatureSpec, feature_id="F1")
    assert loaded.name == "Test Feature"

    # Verify error on missing feature_id
    with pytest.raises(ValueError, match="feature_id is required"):
        sdd_manager.save(spec)

def test_parallel_opportunities_logic(sdd_manager):
    """Verify the dependency and collision detection in SDDManager."""
    t1 = Task(id="T1", title="Task 1", description="D1", status=TaskStatus.PENDING, file_paths=["f1.py"])
    t2 = Task(id="T2", title="Task 2", description="D2", status=TaskStatus.PENDING, file_paths=["f2.py"])
    t3 = Task(id="T3", title="Task 3", description="D3", status=TaskStatus.PENDING, dependencies=["T1"], file_paths=["f3.py"])
    t4 = Task(id="T4", title="Task 4", description="D4", status=TaskStatus.PENDING, file_paths=["f1.py"])

    task_list = TaskList(
        phases=[
            TaskPhase(name="Phase 1", tasks=[t1, t2, t3, t4])
        ]
    )

    # Batch calculation logic in SDDManager:
    # 1. T1, T2 -> Added to Batch 0
    # 2. T3 -> Dependency T1 not met -> Skipped
    # 3. T4 -> Collision with f1.py (T1) -> Not in current batch, but added to NEXT batch
    groups = sdd_manager.get_parallel_opportunities(task_list)
    assert len(groups) >= 2
    assert "T1" in groups[0]
    assert "T2" in groups[0]
    assert "T3" not in groups[0]
    assert "T4" not in groups[0]

    assert "T4" in groups[1]

def test_task_list_loading_nonexistent(sdd_manager):
    """Verify that loading a missing artifact returns None."""
    loaded = sdd_manager.load(TaskList, feature_id="missing")
    assert loaded is None
