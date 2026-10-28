"""Track the current state of the recorded relocation inventories."""

from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[2] / "agent_utilities"
INVENTORY: dict[str, tuple[str, ...]] = {
    "R036": ("usage/backend.py", "usage/authorization.py", "usage/backends"),
    "R037": (
        "governance/concept_allocator.py",
        "governance/concept_hierarchy.py",
        "governance/concept_lineage.py",
    ),
    "R040": tuple(
        f"knowledge_graph/memory/{name}.py"
        for name in (
            "agent_context",
            "learning_engine",
            "memory_engine",
            "rlm_memory",
            "media_store",
            "distillation",
            "lifecycle",
            "observer",
        )
    ),
    "R044": (
        "orchestration/agent_activation.py",
        "orchestration/agent_digital_twin.py",
        "orchestration/action_policy.py",
    ),
    "R045": tuple(
        f"knowledge_graph/{name}"
        for name in (
            "distillation",
            "adaptation",
            "pipeline",
            "search_synthesis",
            "shapes",
            "live_artifacts",
        )
    ),
    "R047": ("kvcache", "numeric", "caching", "media"),
}


def _check(root: Path, paths: tuple[str, ...], recorded: set[str]) -> None:
    actual = {name for name in paths if (root / name).exists()}
    assert actual == recorded


@pytest.mark.spec("AU-BOUNDARY-R036.1")
def test_usage_inventory_matches_tree() -> None:
    _check(PACKAGE, INVENTORY["R036"], set())


@pytest.mark.spec("AU-BOUNDARY-R037.1")
def test_governance_inventory_matches_tree() -> None:
    _check(PACKAGE, INVENTORY["R037"], set())


@pytest.mark.spec("AU-BOUNDARY-R040.1")
def test_memory_inventory_matches_tree() -> None:
    _check(PACKAGE, INVENTORY["R040"], set())


@pytest.mark.spec("AU-BOUNDARY-R044.1")
def test_orchestration_inventory_matches_tree() -> None:
    _check(PACKAGE, INVENTORY["R044"], set())


@pytest.mark.spec("AU-BOUNDARY-R045.1")
def test_knowledge_graph_inventory_matches_tree() -> None:
    _check(PACKAGE, INVENTORY["R045"], set())


@pytest.mark.spec("AU-BOUNDARY-R047.1")
def test_adapter_inventory_matches_tree() -> None:
    _check(PACKAGE, INVENTORY["R047"], set())


def test_inventoried_relocation_still_present(tmp_path: Path) -> None:
    path = tmp_path / "usage/backend.py"
    path.parent.mkdir()
    path.write_text("")
    recorded = {"usage/backend.py"}
    _check(tmp_path, INVENTORY["R036"], recorded)
    with pytest.raises(AssertionError):
        _check(tmp_path, INVENTORY["R036"], set())
    path.unlink()
    with pytest.raises(AssertionError):
        _check(tmp_path, INVENTORY["R036"], recorded)
