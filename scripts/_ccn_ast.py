"""Stdlib-only cyclomatic complexity for Python, so the fleet gate has NO dependency.

Why not just use lizard everywhere: pre-commit must install it into a managed venv,
which cost 6m04s on first run in a pilot repo. Across 79 repos, on every developer
machine and CI image, that is a tax large enough that the hook would get disabled --
and a disabled gate measures nothing. 78 of the 79 repos are Python, so those get
this zero-dependency path and `language: system`; only epistemic-graph (Rust) needs
lizard, and it is a single repo.

Definition follows McCabe as ruff/mccabe implement it: start at 1, add one per
decision point. Cross-validated against lizard 1.24.0 -- see check_complexity.py's
--validate mode, which must be re-run if this counter is ever edited.

Nested functions are scored SEPARATELY and are not folded into their parent, which
is why this disagrees with ruff's C901 on decorated registration functions: ruff
charges a closure's branches to the enclosing def (scoring `register_analysis_tools`
at 320), while this and lizard charge them to the nested function that actually
holds the branching (`_run_analysis_action` at 290).
"""

from __future__ import annotations

import ast


class _Counter(ast.NodeVisitor):
    """Count decision points for ONE function, not descending into nested defs."""

    def __init__(self) -> None:
        self.score = 1

    # Each of these is one branch.
    def visit_If(self, node: ast.If) -> None:
        self.score += 1
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self.score += 1
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self.score += 1
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.score += 1
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self.score += 1
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        self.score += 1
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> None:
        self.score += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        # `a and b and c` is two extra branches, not one.
        self.score += len(node.values) - 1
        self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        # The comprehension itself is a loop; each `if` filter adds another.
        self.score += 1 + len(node.ifs)
        self.generic_visit(node)

    def visit_Match(self, node: ast.Match) -> None:
        # A wildcard `case _` is the default arm and is not itself a decision.
        for case in node.cases:
            if not (isinstance(case.pattern, ast.MatchAs) and case.pattern.pattern is None):
                self.score += 1
        self.generic_visit(node)

    # Do NOT descend into nested definitions; they are scored on their own.
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return


_FUNC = (ast.FunctionDef, ast.AsyncFunctionDef)


def analyze_python(path: str) -> list[tuple[str, int]] | None:
    """Return [(qualified_name, ccn), ...], or None if the file cannot be parsed."""
    try:
        with open(path, "rb") as fh:
            tree = ast.parse(fh.read(), filename=path)
    except (SyntaxError, ValueError, OSError, UnicodeDecodeError):
        return None

    out: list[tuple[str, int]] = []

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, _FUNC):
                name = f"{prefix}{child.name}"
                counter = _Counter()
                for sub in ast.iter_child_nodes(child):
                    counter.visit(sub)
                out.append((name, counter.score))
                walk(child, f"{name}.")
            elif isinstance(child, ast.ClassDef):
                walk(child, f"{prefix}{child.name}.")
            else:
                walk(child, prefix)

    walk(tree, "")
    return out
