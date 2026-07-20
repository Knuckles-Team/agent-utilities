#!/usr/bin/env python3
from __future__ import annotations

"""Architecture gate for the mandatory ContextCompiler model boundary."""

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "agent_utilities"
SCAN_ROOTS = (PACKAGE, ROOT / "scripts", ROOT / "tests", ROOT / "examples")
CANONICAL_AGENT_BOUNDARY = "agent_utilities/core/contextual_model.py"
RAW_PROVIDER_ALLOWLIST = {
    "agent_utilities/core/model_factory.py",
    "agent_utilities/knowledge_graph/retrieval/context_compiler_serving.py",
}


def _dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _function_ancestors(tree: ast.AST) -> dict[ast.AST, str]:
    """Map every descendant node to its nearest function name."""

    owners: dict[ast.AST, str] = {}

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.stack: list[str] = []

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def generic_visit(self, node: ast.AST) -> None:
            if self.stack:
                owners[node] = self.stack[-1]
            super().generic_visit(node)

    Visitor().visit(tree)
    return owners


def _source_violations(relative: str, source: str) -> list[str]:
    """Return boundary violations for one Python source unit."""

    failures: list[str] = []
    try:
        tree = ast.parse(source, filename=relative)
    except SyntaxError as exc:
        return [f"{relative}: parse failed: {exc}"]

    canonical = relative == CANONICAL_AGENT_BOUNDARY
    owners = _function_ancestors(tree)
    pydantic_modules: set[str] = set()
    pydantic_agent_aliases: set[str] = {"Agent", "PydanticAgent"}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "pydantic_ai":
                    pydantic_modules.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and (
            node.module == "pydantic_ai"
            or bool(node.module and node.module.startswith("pydantic_ai."))
        ):
            for alias in node.names:
                if alias.name == "*" and not canonical:
                    failures.append(
                        f"{relative}:{node.lineno}: wildcard PydanticAI import may "
                        "re-export Agent; import the required non-Agent symbols"
                    )
                    continue
                if alias.name != "Agent":
                    continue
                local_name = alias.asname or alias.name
                pydantic_agent_aliases.add(local_name)
                if not canonical:
                    failures.append(
                        f"{relative}:{node.lineno}: direct PydanticAI Agent import "
                        f"{local_name!r}; use create_context_agent"
                    )

    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if value is None or not _dotted(value).endswith(".Agent"):
            continue
        if any(isinstance(target, ast.Name) and target.id == "Agent" for target in targets):
            failures.append(
                f"{relative}:{node.lineno}: direct Agent re-export is forbidden; "
                "use create_context_agent"
            )

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        called = _dotted(node.func)
        raw_call = called.endswith("chat.completions.create") or called.endswith(
            "responses.create"
        )
        raw_constructor = called in {"OpenAI", "AsyncOpenAI", "Anthropic"}
        if (
            relative.startswith("agent_utilities/")
            and (raw_call or raw_constructor)
            and relative not in RAW_PROVIDER_ALLOWLIST
        ):
            failures.append(
                f"{relative}:{node.lineno}: raw model provider bypass {called}"
            )

        module_agent_call = any(
            called == f"{module_name}.Agent" for module_name in pydantic_modules
        )
        named_agent_call = called in pydantic_agent_aliases
        if not (module_agent_call or named_agent_call):
            continue
        if canonical and owners.get(node) == "create_context_agent":
            continue
        failures.append(
            f"{relative}:{node.lineno}: direct Agent construction {called}; "
            "use create_context_agent"
        )

    if relative.startswith("agent_utilities/server/routers/"):
        raw_http_transport = any(
            marker in source
            for marker in (
                "api.openai.com",
                "api.anthropic.com",
                "generativelanguage.googleapis.com",
                'client.stream("POST"',
                'client.post("POST"',
            )
        )
        if raw_http_transport and "compile_model_context(" not in source:
            failures.append(
                f"{relative}: raw HTTP model transport bypasses ContextCompiler"
            )
    return failures


def violations() -> list[str]:
    failures: list[str] = []
    for scan_root in SCAN_ROOTS:
        for path in sorted(scan_root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            relative = path.relative_to(ROOT).as_posix()
            try:
                source = path.read_text(encoding="utf-8")
            except OSError as exc:
                failures.append(f"{relative}: read failed: {exc}")
                continue
            failures.extend(_source_violations(relative, source))

    factory = (PACKAGE / "core" / "model_factory.py").read_text(encoding="utf-8")
    if "wrap_model_with_context(model)" not in factory:
        failures.append("model_factory.py does not install mandatory context wrapper")
    wrapper = (PACKAGE / "core" / "contextual_model.py").read_text(encoding="utf-8")
    if "def create_context_agent(" not in wrapper:
        failures.append("contextual_model.py does not own the Agent constructor")
    if "model=wrap_model_with_context(model)" not in wrapper:
        failures.append("create_context_agent does not govern the supplied model")
    for transport in (
        "async def request(",
        "async def count_tokens(",
        "async def compact_messages(",
        "async def request_stream(",
    ):
        if transport not in wrapper:
            failures.append(
                f"contextual_model.py does not govern {transport.removesuffix('(')}"
            )
    graph_state = (PACKAGE / "graph" / "state.py").read_text(encoding="utf-8")
    if (
        "def __post_init__" not in graph_state
        or "governed(self.agent_model)" not in graph_state
    ):
        failures.append("GraphDeps leaves direct model-id construction ungoverned")
    serving = (
        PACKAGE
        / "knowledge_graph"
        / "retrieval"
        / "context_compiler_serving.py"
    ).read_text(encoding="utf-8")
    if "compile_model_context(" not in serving:
        failures.append("context serving does not compile governed evidence")
    return failures


def main() -> int:
    failures = violations()
    if failures:
        print("ContextCompiler boundary gate failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print("ContextCompiler boundary gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
