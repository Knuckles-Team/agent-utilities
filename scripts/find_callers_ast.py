"""Typed AST matching context used by :mod:`scripts.find_callers`.

The command-line wrapper owns repository traversal and parsing.  This module
owns the per-file matching state so aliases and dynamic-dispatch heuristics
are passed as one explicit, typed context rather than a positional argument
list that is easy to reorder.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class Hit:
    file: str
    line: int
    kind: str  # "call" | "reference" | "monkeypatch" | "getattr"
    snippet: str


@dataclass
class _FileImports:
    # local name -> fully-qualified dotted path it refers to
    aliases: dict[str, str] = field(default_factory=dict)
    # local name -> module it was imported *from* (for from-imports of the
    # symbol's containing module itself, e.g. "from pkg import module as m")
    module_aliases: dict[str, str] = field(default_factory=dict)


@dataclass
class SearchContext:
    """All immutable symbol state plus per-file duplicate suppression."""

    tree: ast.AST
    symbol: str
    simple_name: str
    module_path: str
    imports: _FileImports
    relative_file: str
    direct_alias: str | None
    module_alias: str | None
    existing_hits: tuple[Hit, ...] = ()
    call_lines: set[int] = field(default_factory=set)

    def resolve_node(self, node: ast.AST) -> tuple[str | None, str | None]:
        """Return the local dotted spelling and its imported resolution."""
        parts: list[str] = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if not isinstance(current, ast.Name):
            return None, None
        parts.append(current.id)
        dotted = ".".join(reversed(parts))
        head, _, rest = dotted.partition(".")
        base = self.imports.aliases.get(head)
        resolved = f"{base}.{rest}" if base and rest else base
        return dotted, resolved


def _call_hit(node: ast.AST, context: SearchContext) -> Hit | None:
    if not isinstance(node, ast.Call):
        return None
    function = node.func
    if isinstance(function, ast.Name):
        return (
            Hit(context.relative_file, node.lineno, "call", f"{function.id}(...)")
            if context.direct_alias and function.id == context.direct_alias
            else None
        )
    if not isinstance(function, ast.Attribute):
        return None
    dotted, resolved = context.resolve_node(function)
    if dotted is None:
        return None
    matches = resolved == context.symbol or bool(
        context.module_alias
        and dotted == f"{context.module_alias}.{context.simple_name}"
    )
    return (
        Hit(context.relative_file, node.lineno, "call", dotted + "(...)")
        if matches
        else None
    )


def _call_hits(context: SearchContext) -> list[Hit]:
    hits: list[Hit] = []
    for node in ast.walk(context.tree):
        hit = _call_hit(node, context)
        if hit is not None:
            hits.append(hit)
    return hits


def _attribute_reference_hit(node: ast.Attribute, context: SearchContext) -> Hit | None:
    dotted, resolved = context.resolve_node(node)
    matches = resolved == context.symbol or bool(
        context.module_alias
        and dotted == f"{context.module_alias}.{context.simple_name}"
    )
    return (
        Hit(context.relative_file, node.lineno, "reference", dotted)
        if dotted and matches and node.lineno not in context.call_lines
        else None
    )


def _reference_hit(node: ast.AST, context: SearchContext) -> Hit | None:
    if isinstance(node, ast.Name):
        return (
            Hit(context.relative_file, node.lineno, "reference", node.id)
            if context.direct_alias
            and node.id == context.direct_alias
            and node.lineno not in context.call_lines
            else None
        )
    if not isinstance(node, ast.Attribute):
        return None
    return _attribute_reference_hit(node, context)


def _reference_hits(context: SearchContext) -> list[Hit]:
    hits: list[Hit] = []
    for node in ast.walk(context.tree):
        hit = _reference_hit(node, context)
        if hit is not None:
            hits.append(hit)
    return hits


def _dynamic_dispatch_target(
    node: ast.AST, context: SearchContext
) -> tuple[int, str | None] | None:
    if not isinstance(node, ast.Call) or len(node.args) < 2:
        return None
    name_arg = node.args[1]
    if not isinstance(name_arg, ast.Constant) or name_arg.value != context.simple_name:
        return None
    target_dotted, target_resolved = context.resolve_node(node.args[0])
    if target_resolved != context.module_path and target_dotted != context.module_alias:
        return None
    return node.lineno, target_dotted


def _monkeypatch_hit(node: ast.AST, context: SearchContext) -> Hit | None:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return None
    if node.func.attr != "setattr":
        return None
    target = _dynamic_dispatch_target(node, context)
    if target is None:
        return None
    line, target_dotted = target
    return Hit(
        context.relative_file,
        line,
        "monkeypatch",
        f"setattr({target_dotted}, {context.simple_name!r}, ...)",
    )


def _getattr_hit(node: ast.AST, context: SearchContext) -> Hit | None:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        return None
    if node.func.id != "getattr":
        return None
    target = _dynamic_dispatch_target(node, context)
    if target is None:
        return None
    line, target_dotted = target
    return Hit(
        context.relative_file,
        line,
        "getattr",
        f"getattr({target_dotted}, {context.simple_name!r})",
    )


def _collect_dynamic_hits(
    context: SearchContext,
    matcher: Callable[[ast.AST, SearchContext], Hit | None],
) -> list[Hit]:
    hits: list[Hit] = []
    for node in ast.walk(context.tree):
        hit = matcher(node, context)
        if hit is not None:
            hits.append(hit)
    return hits


def _file_hits(context: SearchContext) -> list[Hit]:
    hits = _call_hits(context)
    context.call_lines.update(
        hit.line for hit in context.existing_hits if hit.file == context.relative_file
    )
    context.call_lines.update(hit.line for hit in hits)
    hits.extend(_reference_hits(context))
    hits.extend(_collect_dynamic_hits(context, _monkeypatch_hit))
    hits.extend(_collect_dynamic_hits(context, _getattr_hit))
    return hits
