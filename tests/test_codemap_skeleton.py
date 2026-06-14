"""Tests for the repo-map skeleton renderer (CONCEPT:ORCH-1.48)."""

from __future__ import annotations

from agent_utilities.models.codemap import CodemapArtifact, CodemapNode
from agent_utilities.models.codemap import _estimate_tokens


def _artifact(n: int) -> CodemapArtifact:
    nodes = [
        CodemapNode(
            id=f"s{i}",
            label=f"symbol_{i}",
            type="function",
            file=f"pkg/mod_{i % 3}.py",
            line=i + 1,
            importance=round(i / n, 4),  # higher index = higher importance
        )
        for i in range(n)
    ]
    return CodemapArtifact(id="x", prompt="map it", mode="fast", nodes=nodes)


def test_empty_artifact_renders_empty():
    art = CodemapArtifact(id="x", prompt="p", mode="fast", nodes=[])
    assert art.to_skeleton(100) == ""


def test_skeleton_respects_token_budget():
    art = _artifact(80)
    skeleton = art.to_skeleton(max_tokens=60)
    assert _estimate_tokens(skeleton) <= 60
    # Something was rendered, and truncation was noted.
    assert "symbol_" in skeleton
    assert "omitted" in skeleton


def test_high_importance_symbols_survive_truncation():
    art = _artifact(40)
    skeleton = art.to_skeleton(max_tokens=40)
    # The most important symbol (highest index) must be present...
    assert "symbol_39" in skeleton
    # ...while a low-importance one is dropped.
    assert "symbol_0 " not in skeleton


def test_generous_budget_includes_everything():
    art = _artifact(10)
    skeleton = art.to_skeleton(max_tokens=100_000)
    for i in range(10):
        assert f"symbol_{i}" in skeleton
    assert "omitted" not in skeleton


def test_skeleton_groups_by_file_and_is_deterministic():
    art = _artifact(9)
    a = art.to_skeleton(max_tokens=100_000)
    b = art.to_skeleton(max_tokens=100_000)
    assert a == b
    # Files appear as headers.
    assert "pkg/mod_0.py" in a
    # Symbols are indented under their file.
    assert "\n  symbol_" in a
