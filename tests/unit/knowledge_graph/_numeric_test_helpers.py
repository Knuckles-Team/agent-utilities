"""Shared helpers for tests driving ``agent_utilities.numeric`` (``xp``).

``agent_utilities.numeric`` crosses the boundary as bounded builtin lists,
never an array-like object with operator overloading (its own module
docstring: "deliberately does not provide an array object"), so plain
``+``/``*`` on two of its results is either a TypeError (float multiplier) or
silent list concatenation/repetition (int multiplier), never elementwise
math. Several test modules build a signal + independent noise sample and need
real elementwise addition; share one helper instead of repeating it.
"""

from __future__ import annotations

from typing import Any

__all__ = ["elementwise_add"]


def elementwise_add(a: Any, b: Any) -> Any:
    if isinstance(a, list):
        return [elementwise_add(x, y) for x, y in zip(a, b, strict=True)]
    return a + b
