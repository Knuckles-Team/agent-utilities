"""Governed code-execution preflight (CONCEPT:AU-ECO.messaging.native-backend-abstraction).

AST screening is defense in depth, never a sandbox. The underlying adapter
fails closed unless a genuinely isolated execution runtime is provisioned.
"""

import ast
from typing import Any

from .jupyter_adapter import JupyterKernelAdapter


class SandboxExecutor:
    """Preflight untrusted code before the governed execution boundary."""

    def __init__(self):
        self.kernel = JupyterKernelAdapter()

    def _validate_invariants(self, code: str) -> bool:
        """Reject structurally dangerous constructs without claiming isolation."""
        if (
            not isinstance(code, str)
            or not code.strip()
            or len(code.encode("utf-8")) > 65_536
        ):
            return False
        try:
            tree = ast.parse(code, mode="exec")
        except (SyntaxError, ValueError, MemoryError):
            return False
        forbidden_nodes = (
            ast.AsyncFor,
            ast.AsyncFunctionDef,
            ast.AsyncWith,
            ast.ClassDef,
            ast.Global,
            ast.Import,
            ast.ImportFrom,
            ast.Nonlocal,
            ast.Raise,
            ast.Try,
            ast.While,
            ast.With,
            ast.Yield,
            ast.YieldFrom,
        )
        forbidden_names = {
            "__builtins__",
            "__import__",
            "breakpoint",
            "compile",
            "eval",
            "exec",
            "globals",
            "help",
            "input",
            "locals",
            "open",
            "vars",
        }
        for node in ast.walk(tree):
            if isinstance(node, forbidden_nodes):
                return False
            if isinstance(node, ast.Name) and node.id in forbidden_names:
                return False
            if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
                return False
        return True

    def run_safe(self, code: str) -> dict[str, Any]:
        """Runs code only if structural invariants pass."""
        if not self._validate_invariants(code):
            return {
                "status": "error",
                "error": "InvariantViolation: Unsafe code patterns detected.",
            }

        return self.kernel.execute(code)
