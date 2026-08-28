"""CONCEPT:AU-ORCH.dispatch.dispatch-warm-fork — dispatch-tier warm-fork: SkillsToolset is built once and warm-shared."""

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
    """Wire-First: the live agent factory routes SkillsToolset through the warm cache.

    Resolved through the module's CALL GRAPH rather than by grepping
    ``create_agent``'s own source text. The previous form asserted the literal
    ``"get_or_build_skills_toolset" in inspect.getsource(create_agent)``, which
    broke the moment the call moved one frame down into an extracted helper --
    while the wiring it exists to protect was completely intact. A source-text
    assertion answers "does this name appear in these bytes", which is never the
    same question as "is the warm cache on the live path".
    """
    import ast
    import inspect

    from agent_utilities.agent import factory as agent_factory

    module = ast.parse(inspect.getsource(agent_factory))
    defs = {
        n.name: n
        for n in module.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    def called_names(node):
        out = set()
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                fn = sub.func
                if isinstance(fn, ast.Name):
                    out.add(fn.id)
                elif isinstance(fn, ast.Attribute):
                    out.add(fn.attr)
            elif isinstance(sub, ast.Name):
                out.add(sub.id)
        return out

    # Breadth-first over module-level helpers reachable from create_agent.
    seen: set[str] = set()
    frontier = ["create_agent"]
    reachable: set[str] = set()
    while frontier:
        name = frontier.pop()
        if name in seen:
            continue
        seen.add(name)
        node = defs.get(name)
        if node is None:
            continue
        names = called_names(node)
        reachable |= names
        frontier.extend(n for n in names if n in defs and n not in seen)

    assert "get_or_build_skills_toolset" in reachable, (
        "the warm skills cache is no longer reachable from create_agent; "
        f"reachable helpers were {sorted(seen)}"
    )
