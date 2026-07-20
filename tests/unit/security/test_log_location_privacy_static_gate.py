"""Static gate preventing runtime locations and caller IDs in log arguments."""

from __future__ import annotations

import ast
from pathlib import Path

_LOG_METHODS = {
    "critical",
    "debug",
    "error",
    "exception",
    "info",
    "log",
    "warning",
}
_SENSITIVE_NAMES = {
    "actor",
    "agent_id",
    "backup_path",
    "caller",
    "config_path",
    "dsn",
    "endpoint",
    "file_path",
    "filepath",
    "host",
    "hostname",
    "input_path",
    "output_path",
    "path",
    "source_path",
    "target_path",
    "uri",
    "url",
    "user",
    "username",
    "workspace",
}


def _terminal_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def test_logs_do_not_receive_location_or_caller_values() -> None:
    package_root = Path(__file__).resolve().parents[3] / "agent_utilities"
    findings: list[str] = []
    for path in package_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.name)
        for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
            if not isinstance(call.func, ast.Attribute):
                continue
            if call.func.attr not in _LOG_METHODS:
                continue
            # Positional values after the format string are interpolated into the
            # record. F-string values can occur in the first argument itself.
            values = list(call.args[1:])
            for argument in call.args:
                values.extend(
                    formatted.value
                    for formatted in ast.walk(argument)
                    if isinstance(formatted, ast.FormattedValue)
                )
            for value in values:
                name = _terminal_name(value)
                if name in _SENSITIVE_NAMES:
                    findings.append(f"{path.relative_to(package_root)}:{call.lineno}")
    assert findings == []
