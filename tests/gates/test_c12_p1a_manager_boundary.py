"""C12 P1a — orchestration manager composition boundary.

These tests pin the architectural change at the module edge.  The manager may
consume an injected graph capability and compiler protocol, but it must not
resolve the concrete Knowledge Graph engine/compiler itself (eagerly,
deferred, type-only, or through a dynamic selector).
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent_utilities.orchestration.manager import (
    Orchestrator,
    WorkflowCompilerNotBoundError,
)

_MANAGER = Path(__file__).parents[2] / "agent_utilities/orchestration/manager.py"
_CONCRETE_TARGETS = frozenset(
    {
        "agent_utilities.knowledge_graph.core.engine",
        "agent_utilities.knowledge_graph.workflow_compiler",
    }
)


def _module_targets(node: ast.Import | ast.ImportFrom) -> set[str]:
    if isinstance(node, ast.Import):
        return {alias.name for alias in node.names}
    if node.module:
        return {node.module}
    return set()


def _is_concrete_target(module: str) -> bool:
    return module in _CONCRETE_TARGETS


class _ImportModes(ast.NodeVisitor):
    def __init__(self) -> None:
        self.mode = "eager"
        self.found: dict[str, list[tuple[int, str]]] = {
            "eager": [],
            "deferred": [],
            "type": [],
        }
        self._function_depth = 0
        self._type_guard_depth = 0

    def _record(self, node: ast.Import | ast.ImportFrom) -> None:
        mode = (
            "type"
            if self._type_guard_depth
            else "deferred"
            if self._function_depth
            else "eager"
        )
        self.found[mode].extend(
            (node.lineno, target)
            for target in _module_targets(node)
            if _is_concrete_target(target)
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._function_depth += 1
        self.generic_visit(node)
        self._function_depth -= 1

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._function_depth += 1
        self.generic_visit(node)
        self._function_depth -= 1

    def visit_If(self, node: ast.If) -> None:
        is_type_guard = (
            isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING"
        )
        if is_type_guard:
            self._type_guard_depth += 1
        self.generic_visit(node)
        if is_type_guard:
            self._type_guard_depth -= 1

    def visit_Import(self, node: ast.Import) -> None:
        self._record(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self._record(node)


def _imports() -> _ImportModes:
    tree = ast.parse(_MANAGER.read_text(encoding="utf-8"), filename=str(_MANAGER))
    imports = _ImportModes()
    imports.visit(tree)
    return imports


def test_manager_has_no_eager_concrete_graph_edges() -> None:
    assert _imports().found["eager"] == []


def test_manager_has_no_deferred_concrete_graph_edges() -> None:
    assert _imports().found["deferred"] == []


def test_manager_has_no_type_only_concrete_graph_edges() -> None:
    assert _imports().found["type"] == []


def test_manager_has_no_dynamic_concrete_graph_selector() -> None:
    tree = ast.parse(_MANAGER.read_text(encoding="utf-8"), filename=str(_MANAGER))
    dynamic_imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            getattr(node.func, "attr", None) in {"import_module", "__import__"}
            or getattr(node.func, "id", None) in {"import_module", "__import__"}
        )
    ]
    assert dynamic_imports == []


class _Compiler:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def compile_and_store(
        self,
        name: str,
        description: str,
        domain: str = "general",
    ) -> str:
        self.calls.append({"name": name, "description": description, "domain": domain})
        return "workflow:injected"


@pytest.mark.asyncio
async def test_manager_uses_the_injected_compiler_capability() -> None:
    compiler = _Compiler()
    engine = object()
    orchestrator = Orchestrator(engine=engine, compiler=compiler)

    workflow_id = await orchestrator.compile_workflow(
        name="demo", task="summarize the bounded result"
    )

    assert orchestrator.engine is engine
    assert workflow_id == "workflow:injected"
    assert compiler.calls == [
        {
            "name": "demo",
            "description": "summarize the bounded result",
            "domain": "general",
        }
    ]


@pytest.mark.asyncio
async def test_manager_rejects_unbound_compiler_at_the_application_boundary() -> None:
    orchestrator = Orchestrator(engine=object())

    with pytest.raises(
        WorkflowCompilerNotBoundError,
        match="workflow_compiler_not_bound",
    ):
        await orchestrator.compile_workflow(name="demo", task="summarize results")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "compiler",
    [None, SimpleNamespace(compile_and_store=None)],
    ids=["none", "non-callable"],
)
async def test_manager_rejects_non_callable_compiler_capability(
    compiler: Any,
) -> None:
    orchestrator = Orchestrator(engine=object(), compiler=compiler)

    with pytest.raises(
        WorkflowCompilerNotBoundError,
        match="workflow_compiler_not_bound",
    ):
        await orchestrator.compile_workflow(name="demo", task="summarize results")


def test_prefixed_loop_skill_does_not_resolve_unused_compiler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stored workflow id needs execution only, never NL compiler setup."""
    from agent_utilities.knowledge_graph.research import loop_controller

    class _Orchestrator:
        def __init__(self, engine: Any) -> None:
            self.engine = engine

        async def execute_workflow(
            self, workflow_id: str, task: str = ""
        ) -> dict[str, str]:
            return {"workflow_id": workflow_id, "task": task}

    monkeypatch.setattr(
        "agent_utilities.orchestration.manager.Orchestrator", _Orchestrator
    )
    # If the loop attempts even to resolve the concrete compiler import, this
    # sentinel makes the test fail.  The prefixed path must only use execution.
    monkeypatch.setitem(
        sys.modules,
        "agent_utilities.knowledge_graph.workflow_compiler",
        None,
    )

    ok, output = loop_controller._default_skill_runner(
        "workflow:stored", "execute it", engine=object()
    )

    assert ok is True
    assert "workflow:stored" in output
