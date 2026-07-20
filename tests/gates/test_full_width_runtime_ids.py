"""Strict-current collision-resistance contract for production identifiers."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_ROOTS = (ROOT / "agent_utilities", ROOT / "scripts")


def _is_uuid4_hex_slice(node: ast.AST) -> bool:
    if not isinstance(node, ast.Subscript):
        return False
    value = node.value
    if not isinstance(value, ast.Attribute) or value.attr != "hex":
        return False
    call = value.value
    if not isinstance(call, ast.Call):
        return False
    function = call.func
    return (isinstance(function, ast.Name) and function.id == "uuid4") or (
        isinstance(function, ast.Attribute) and function.attr == "uuid4"
    )


def test_production_identifiers_use_full_width_uuid4_tokens() -> None:
    offenders: list[str] = []
    for source_root in PRODUCTION_ROOTS:
        for path in source_root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if _is_uuid4_hex_slice(node):
                    relative = path.relative_to(ROOT).as_posix()
                    offenders.append(f"{relative}:{node.lineno}")

    assert offenders == [], (
        "Production identifiers must retain the complete uuid4().hex token; "
        f"truncated sites: {offenders}"
    )
