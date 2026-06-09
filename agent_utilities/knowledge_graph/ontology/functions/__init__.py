#!/usr/bin/python
from __future__ import annotations

"""Ontology Functions runtime (CONCEPT:KG-2.41).

Palantir Foundry ``functions/overview`` parity for the ontology layer: typed,
versioned, releasable user *Functions* with declared inputs/outputs, the three
Foundry function kinds (PLAIN | ON_OBJECTS | QUERY), Functions-on-Objects graph
reads, and a single governed runtime that validates I/O and audits every call.

Building blocks:
  - :class:`FunctionSpec` / :class:`FunctionParameter` / :class:`FunctionKind` —
    the typed, versioned signature (``registry.py``).
  - :class:`FunctionRegistry` — registration, duplicate/version handling,
    release/publish, lookup-by-name(+version) (``registry.py``).
  - :class:`ObjectFunctionContext` — Functions-on-Objects: read properties,
    traverse N-hop links, aggregate over object sets via the live
    :class:`KnowledgeGraph` facade (``objects.py``).
  - :class:`FunctionRuntime` — the single invocation entry that validates typed
    inputs, runs the handler, coerces the typed output, and audits
    (``runtime.py``).

Module-level live paths (populated/bound at import, never empty shells):
  - :data:`DEFAULT_FUNCTION_REGISTRY` — carries built-in ``object.summarize``
    (ON_OBJECTS) and ``numeric.aggregate`` (QUERY) functions.
  - :data:`DEFAULT_FUNCTION_RUNTIME` — a runtime bound to that registry; the
    contract Actions (CONCEPT:KG-2.25) and derived-properties consume.
"""

from .objects import ObjectFunctionContext
from .registry import (
    DEFAULT_FUNCTION_REGISTRY,
    FunctionHandler,
    FunctionKind,
    FunctionParameter,
    FunctionRegistry,
    FunctionSpec,
    register_builtins,
    resolve_type,
)
from .runtime import (
    DEFAULT_FUNCTION_RUNTIME,
    FUNCTION_AUDIT,
    FunctionResult,
    FunctionRuntime,
    coerce_output,
)

__all__ = [
    "DEFAULT_FUNCTION_REGISTRY",
    "DEFAULT_FUNCTION_RUNTIME",
    "FUNCTION_AUDIT",
    "FunctionHandler",
    "FunctionKind",
    "FunctionParameter",
    "FunctionRegistry",
    "FunctionResult",
    "FunctionRuntime",
    "FunctionSpec",
    "ObjectFunctionContext",
    "coerce_output",
    "register_builtins",
    "resolve_type",
]
