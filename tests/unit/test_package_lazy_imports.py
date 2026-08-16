"""CONCEPT:AU-OS.deployment.agent-factory-autoload

Regression test for B-22: ``agent_utilities/__init__.py``'s module-level ``__getattr__``
(PEP 562 lazy-import registry) had a stale entry -- ``SemanticCompactor`` pointed at
``.knowledge_graph.memory.memory_compaction``, a module that does not exist, so
``agent_utilities.SemanticCompactor`` raised ``ModuleNotFoundError`` at attribute-access
time for anyone who imported it that way (the real location is
``.knowledge_graph.memory.agent_context``).

Rather than pinning that one entry, this walks every ``from <module> import <names>``
statement inside ``__getattr__`` via the AST (so it needs no knowledge of the branch
structure) and asserts each resolved module imports cleanly and actually defines every
name it re-exports -- so a *future* lazy-import entry that drifts from its target (a
rename, a moved module) fails this test instead of only failing for whichever caller
happens to touch that specific attribute first.
"""

from __future__ import annotations

import ast
import importlib
import inspect

import agent_utilities


def _getattr_import_targets() -> list[tuple[str, list[str]]]:
    """Every ``from <module> import <names>`` inside ``agent_utilities.__getattr__``."""
    source = inspect.getsource(agent_utilities)
    tree = ast.parse(
        source, filename=agent_utilities.__file__ or "agent_utilities/__init__.py"
    )

    func = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "__getattr__"
    )

    targets: list[tuple[str, list[str]]] = []
    for node in ast.walk(func):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                module = "agent_utilities" + (f".{node.module}" if node.module else "")
            else:
                module = node.module or ""
            targets.append((module, [alias.name for alias in node.names]))
    return targets


def test_every_lazy_import_target_module_exists_and_exports_its_names():
    targets = _getattr_import_targets()
    assert len(targets) > 30, (
        "sanity check: expected __getattr__ to contain the full lazy-import surface; "
        f"only found {len(targets)} `from ... import ...` statements -- did the AST walk "
        "or the function name change?"
    )

    failures: list[str] = []
    for module_name, names in targets:
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            failures.append(
                f"{module_name!r} (importing {names}): {type(exc).__name__}: {exc}"
            )
            continue
        for name in names:
            if not hasattr(module, name):
                failures.append(f"{module_name!r} has no attribute {name!r}")

    assert not failures, (
        "broken lazy-import entries in agent_utilities.__getattr__:\n"
        + "\n".join(failures)
    )


def test_semantic_compactor_resolves_via_the_package_lazy_import():
    """Pin the exact B-20/B-22 regression: ``agent_utilities.SemanticCompactor`` must
    resolve to the real class, not raise ``ModuleNotFoundError``."""
    from agent_utilities.knowledge_graph.memory.agent_context import (
        SemanticCompactor as direct,
    )

    assert agent_utilities.SemanticCompactor is direct
