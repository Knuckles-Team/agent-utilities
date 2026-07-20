"""Static regression gate for exception leakage at served boundaries."""

from __future__ import annotations

import ast
from pathlib import Path

import agent_utilities


def _unsafe_exception_uses(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.name)
    findings: list[str] = []
    for handler in (
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ExceptHandler) and node.name
    ):
        exception_name = handler.name
        body = ast.Module(body=handler.body, type_ignores=[])
        for node in ast.walk(body):
            unsafe = False
            if isinstance(node, ast.JoinedStr):
                for value in node.values:
                    expression = (
                        value.value if isinstance(value, ast.FormattedValue) else None
                    )
                    if isinstance(expression, ast.Name):
                        unsafe |= expression.id == exception_name
                    elif (
                        isinstance(expression, ast.Call)
                        and isinstance(expression.func, ast.Name)
                        and expression.func.id in {"str", "repr"}
                    ):
                        unsafe |= any(
                            isinstance(arg, ast.Name) and arg.id == exception_name
                            for arg in expression.args
                        )
            elif isinstance(node, ast.Call):
                if (
                    isinstance(node.func, ast.Name)
                    and node.func.id in {"str", "repr"}
                ):
                    unsafe |= any(
                        isinstance(arg, ast.Name) and arg.id == exception_name
                        for arg in node.args
                    )
                if isinstance(node.func, ast.Attribute):
                    log_methods = {
                        "critical",
                        "debug",
                        "error",
                        "exception",
                        "info",
                        "log",
                        "warn",
                        "warning",
                    }
                    unsafe |= node.func.attr in {
                        "format_exc",
                        "format_exception",
                        "print_exc",
                    }
                    unsafe |= node.func.attr == "exception"
                    unsafe |= node.func.attr in log_methods and any(
                        isinstance(arg, ast.Name) and arg.id == exception_name
                        for arg in node.args
                    )
                    unsafe |= node.func.attr in log_methods and any(
                        isinstance(keyword.value, ast.Name)
                        and keyword.value.id == exception_name
                        for keyword in node.keywords
                    )
                    unsafe |= any(
                        keyword.arg == "exc_info"
                        and isinstance(keyword.value, ast.Constant)
                        and keyword.value.value is True
                        for keyword in node.keywords
                    )
            if unsafe:
                findings.append(f"{path.name}:{getattr(node, 'lineno', 0)}")
    return findings


def test_served_packages_do_not_expose_raw_exception_text() -> None:
    package_root = Path(agent_utilities.__file__).resolve().parent
    findings: list[str] = []
    for relative_root in ("mcp", "gateway", "server"):
        for path in (package_root / relative_root).rglob("*.py"):
            findings.extend(_unsafe_exception_uses(path))
    assert findings == []
