"""Read the row set out of EG's ``kind``-tagged UQL result.

CONCEPT:AU-KG.query.top-nodes-by-degree — EG's ``QueryClient.uql`` returns one
dict whose ``"kind"`` says what ran: ``"rows"`` (``columns``/``rows``/
``warnings``), ``"profile"`` or ``"explain"``. The engine's row surface
(:meth:`IntelligenceGraphEngine.uql`) serves rows only; an ``EXPLAIN`` or
``PROFILE`` statement belongs on the raw ``engine_query action=uql`` tool,
which forwards the whole result. Refusing here is the correctness fix: listing
an explain dict yields its KEYS, which used to flow on as if they were rows.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ["uql_rows"]


def uql_rows(result: object) -> list[Any]:
    """The ``rows`` of a ``kind == "rows"`` UQL result; refuse anything else."""
    if not isinstance(result, Mapping):
        raise TypeError(f"UQL returned {type(result).__name__}, not a result dict")
    kind = result.get("kind")
    if kind != "rows":
        raise ValueError(
            f"UQL {kind!r} results are not rows; run EXPLAIN/PROFILE through "
            "engine_query action='uql', which returns the whole result"
        )
    return list(result.get("rows") or [])
