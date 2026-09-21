"""AST import collection and relative-name primitives."""

from __future__ import annotations

import ast
from dataclasses import dataclass

_PRODUCT_ROOTS = frozenset({"agent_connector_sdk", "epistemic_graph", "graph_os"})
_DYNAMIC_IMPORT_NAMES = frozenset(
    {"__import__", "find_spec", "import_client", "import_module"}
)


@dataclass(frozen=True, slots=True)
class ProductImport:
    """A literal import of one of the four product packages."""

    target: str
    lineno: int


def is_type_checking_guard(test: ast.expr) -> bool:
    """Recognize bare and module-qualified ``TYPE_CHECKING`` guards."""
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    if isinstance(test, ast.Attribute):
        return test.attr == "TYPE_CHECKING"
    return False


class ImportWalker(ast.NodeVisitor):
    """Collect every import and whether it executes at module load."""

    def __init__(self) -> None:
        self.found: list[tuple[ast.Import | ast.ImportFrom, bool]] = []
        self._function_depth = 0
        self._type_checking_depth = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._function_depth += 1
        self.generic_visit(node)
        self._function_depth -= 1

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._function_depth += 1
        self.generic_visit(node)
        self._function_depth -= 1

    def visit_If(self, node: ast.If) -> None:
        if not is_type_checking_guard(node.test):
            self.generic_visit(node)
            return
        self.visit(node.test)
        self._type_checking_depth += 1
        for child in node.body:
            self.visit(child)
        self._type_checking_depth -= 1
        for child in node.orelse:
            self.visit(child)

    def visit_Import(self, node: ast.Import) -> None:
        self.found.append((node, self.is_eager()))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.found.append((node, self.is_eager()))

    def is_eager(self) -> bool:
        """Return whether the current AST position executes at module load."""
        return self._function_depth == 0 and self._type_checking_depth == 0


def relative_base(current_package: str, level: int) -> str:
    """Resolve the dotted package selected by leading relative-import dots."""
    segments = current_package.split(".") if current_package else []
    retained = max(len(segments) - max(level - 1, 0), 0)
    return ".".join(segments[:retained])


def from_base(node: ast.ImportFrom, current_package: str) -> str:
    """Return the absolute dotted base of one ``from`` import."""
    if node.level == 0:
        return node.module or ""
    base = relative_base(current_package, node.level)
    if base and node.module:
        return f"{base}.{node.module}"
    return node.module or base


def product_imports(tree: ast.AST) -> list[ProductImport]:
    """Collect static and literal-dynamic cross-product imports.

    Lower-case names imported from a product package root are treated as
    submodules (``from agent_connector_sdk import runner``); exported public
    types remain package-root imports.  Literal ``import_module``/``find_spec``
    probes are dependencies too and cannot bypass the direction gate.
    """
    found: list[ProductImport] = []
    dynamic_names = _dynamic_import_aliases(tree)
    for node in ast.walk(tree):
        for target in _node_import_targets(node, dynamic_names):
            if target.split(".", 1)[0] in _PRODUCT_ROOTS:
                found.append(
                    ProductImport(target=target, lineno=int(getattr(node, "lineno", 0)))
                )
    return sorted(set(found), key=lambda item: (item.lineno, item.target))


def _node_import_targets(
    node: ast.AST, dynamic_names: frozenset[str]
) -> tuple[str, ...]:
    """Return the statically knowable module targets of one AST node."""
    if isinstance(node, ast.Import):
        return tuple(alias.name for alias in node.names)
    if isinstance(node, ast.ImportFrom):
        return _from_import_targets(node)
    if not isinstance(node, ast.Call):
        return ()
    literal = _literal_dynamic_import(node, dynamic_names)
    return (literal,) if literal is not None else ()


def _from_import_targets(node: ast.ImportFrom) -> tuple[str, ...]:
    """Resolve product submodules named by an absolute ``from`` import."""
    if node.level != 0 or not node.module:
        return ()
    if node.module not in _PRODUCT_ROOTS:
        return (node.module,)
    submodules = tuple(
        f"{node.module}.{alias.name}"
        for alias in node.names
        if alias.name[:1].islower() or alias.name.startswith("_")
    )
    return (node.module, *submodules)


def _dynamic_import_aliases(tree: ast.AST) -> frozenset[str]:
    """Return direct-call names bound to known dynamic import functions."""
    aliases = set(_DYNAMIC_IMPORT_NAMES)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        for alias in node.names:
            if alias.name in _DYNAMIC_IMPORT_NAMES:
                aliases.add(alias.asname or alias.name)
    return frozenset(aliases)


def _literal_dynamic_import(
    node: ast.Call, dynamic_names: frozenset[str]
) -> str | None:
    """Return the literal module from a recognized dynamic-import call."""
    if not node.args:
        return None
    argument = node.args[0]
    if not isinstance(argument, ast.Constant) or not isinstance(argument.value, str):
        return None
    function = node.func
    recognized = (isinstance(function, ast.Name) and function.id in dynamic_names) or (
        isinstance(function, ast.Attribute) and function.attr in _DYNAMIC_IMPORT_NAMES
    )
    return argument.value if recognized else None
