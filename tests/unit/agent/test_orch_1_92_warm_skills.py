"""CONCEPT:ORCH-1.92 — dispatch-tier warm-fork: SkillsToolset is built once and warm-shared."""

from __future__ import annotations

import pytest

from agent_utilities.agent.warm_skills import get_or_build_skills_toolset
from agent_utilities.runtime.warm_registry import WarmParentRegistry


@pytest.fixture
def clean_registry():
    WarmParentRegistry._instance = None  # noqa: SLF001 - test isolation
    yield
    WarmParentRegistry._instance = None  # noqa: SLF001


def test_same_dir_set_reused_order_independent(clean_registry):
    builds = {"n": 0}

    def factory():
        builds["n"] += 1
        return ("toolset", builds["n"])

    dirs = ["/a/skills", "/b/skills"]
    t1 = get_or_build_skills_toolset(dirs, factory)
    t2 = get_or_build_skills_toolset(
        list(reversed(dirs)), factory
    )  # same set, diff order

    assert t1 is t2, "the same skill-dir set must reuse the warm toolset"
    assert builds["n"] == 1, "built once, reused thereafter"
    assert WarmParentRegistry.get().stats()["by_kind"].get("skills_toolset") == 1


def test_distinct_dir_sets_build_separately(clean_registry):
    builds = {"n": 0}

    def factory():
        builds["n"] += 1
        return builds["n"]

    get_or_build_skills_toolset(["/a"], factory)
    get_or_build_skills_toolset(["/b"], factory)
    assert builds["n"] == 2


def test_empty_dirs_bypasses_cache(clean_registry):
    builds = {"n": 0}

    def factory():
        builds["n"] += 1
        return builds["n"]

    get_or_build_skills_toolset([], factory)
    get_or_build_skills_toolset([], factory)
    # No caching when there are no dirs — each call builds fresh (nothing to amortise).
    assert builds["n"] == 2
    assert WarmParentRegistry.get().stats()["warm_parents"] == 0


def test_factory_used_in_create_agent():
    """Wire-First: the live agent factory routes SkillsToolset through the warm cache."""
    import inspect

    from agent_utilities.agent import factory as agent_factory

    src = inspect.getsource(agent_factory.create_agent)
    assert "get_or_build_skills_toolset" in src
