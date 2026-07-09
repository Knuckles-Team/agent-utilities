"""CONCEPT:AU-ORCH.sandbox.tiered-rlm-sandbox — Hot-path code analysis for sandbox routing.

The router needs to know, for one LLM-generated snippet: does it import third-party libs?
define classes/dataclasses? use async? call the RLM host helpers? — to pick the cheapest
capable backend. This is computed in-process with the stdlib :mod:`ast` (~50-200µs/snippet).

Why not the epistemic-graph ``ParseFile`` kernel? It is a UDS round-trip (~0.2ms+ floor),
the engine isn't guaranteed running, and its own guidance is "batch, never per-element" —
all wrong for a per-snippet hot path. So :class:`AstAnalyzer` is the default. The
:class:`Analyzer` Protocol leaves the door open for an engine-backed *batch* analyzer
(``ParseFiles``) in a future offline pre-classification pass, without touching the hot path.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .base import HELPER_NAMES

# Names importable in the monty/CPython-WASI tiers without a third-party package.
# ``sys.stdlib_module_names`` is the authoritative CPython set; we treat anything outside
# it (and outside the project's own helper namespace) as third-party.
_STDLIB: frozenset[str] = frozenset(sys.stdlib_module_names)


@dataclass
class CodeRequirements:
    """What a snippet needs from a backend (the router matches this to capabilities).

    ``syntax_ok=False`` means the snippet does not even parse as Python — the router can skip
    straight to the floor backend, which will surface the SyntaxError to the model to retry.
    """

    syntax_ok: bool = True
    third_party_imports: set[str] = field(default_factory=set)
    defines_classes: bool = (
        False  # ``class`` or ``@dataclass`` — needs a ``classes`` backend
    )
    uses_async: bool = False
    helper_calls: set[str] = field(default_factory=set)  # subset of HELPER_NAMES called

    @property
    def needs_third_party(self) -> bool:
        return bool(self.third_party_imports)

    @property
    def needs_host_callbacks(self) -> bool:
        return bool(self.helper_calls)


@runtime_checkable
class Analyzer(Protocol):
    """Anything that can turn a snippet into :class:`CodeRequirements`."""

    def analyze(self, code: str) -> CodeRequirements:
        ...


class AstAnalyzer:
    """Default in-process analyzer: one ``ast.walk`` over the snippet, no IPC, no deps."""

    def analyze(self, code: str) -> CodeRequirements:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return CodeRequirements(syntax_ok=False)

        req = CodeRequirements()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self._note_import(alias.name, req)
            elif isinstance(node, ast.ImportFrom):
                # level>0 is a relative import (project-local); module None on bare ``from . import``.
                if node.level == 0 and node.module:
                    self._note_import(node.module, req)
            elif isinstance(node, ast.ClassDef):
                req.defines_classes = True
            elif isinstance(
                node, ast.AsyncFunctionDef | ast.AsyncFor | ast.AsyncWith | ast.Await
            ):
                req.uses_async = True
            elif isinstance(node, ast.Call):
                target = self._call_name(node.func)
                if target in HELPER_NAMES:
                    req.helper_calls.add(target)
        return req

    @staticmethod
    def _note_import(dotted: str, req: CodeRequirements) -> None:
        """Record a top-level import module, classifying stdlib vs third-party."""
        top = dotted.split(".", 1)[0]
        if top and top not in _STDLIB:
            req.third_party_imports.add(top)

    @staticmethod
    def _call_name(func: ast.expr) -> str | None:
        """Best-effort callee name: ``f(...)`` -> 'f', ``obj.m(...)`` -> 'm'."""
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            return func.attr
        return None
