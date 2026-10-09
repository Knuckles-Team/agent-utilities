"""Typed delegation seam for AU-CONTEXT-R007.1.

AU-CONTEXT-R007 moves finance math to epistemic-graph's finance core, but
AU's local finance math (``agent_utilities/domains/finance/**`` and
``knowledge_graph/orchestration/engine_finance.py``) is still the live
implementation today. This module gives callers one typed seam that prefers
EG's served finance primitives when they exist and otherwise calls the
caller-supplied local implementation -- it never silently invents a result
of its own.

AU-CONTEXT-R007.2 deletes the local finance-math modules and the fallback
branch here, once EG's finance core is confirmed to cover every migrated
calculation (per the boundary test in
``tests/unit/finance/test_r007_finance_module_boundary.py``, which pins the
current module set so it can only shrink until then).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

__all__ = [
    "KellySizingRequest",
    "KellySizingResponse",
    "FinancePrimitivesClient",
    "resolve_finance_primitives_client",
    "kelly_size_via_eg_or_local",
]


@dataclass(frozen=True, slots=True)
class KellySizingRequest:
    """Typed request mirroring ``engine_finance.calculate_kelly_size``'s inputs."""

    win_probability: float
    win_loss_ratio: float


@dataclass(frozen=True, slots=True)
class KellySizingResponse:
    """Typed response mirroring the local Kelly-sizing calculation's output."""

    suggested_fraction: float


@runtime_checkable
class FinancePrimitivesClient(Protocol):
    """Contract for EG's served finance-core primitives, once they exist."""

    def kelly_sizing(self, request: KellySizingRequest) -> KellySizingResponse: ...


def resolve_finance_primitives_client() -> FinancePrimitivesClient | None:
    """Return EG's generated finance-primitives client, or ``None``.

    Unlike a fail-closed refusal seam, this returns ``None`` rather than
    raising: AU's local finance math (``engine_finance.py``,
    ``domains/finance/**``) remains the live, authorized implementation
    until AU-CONTEXT-R007.2 deletes it, so a missing EG client is an
    expected, handled state here -- not an error.
    """
    try:
        from epistemic_graph.generated.finance_core import (  # type: ignore[import-not-found]
            FinanceCoreClient,
        )
    except ImportError:
        return None

    if not hasattr(FinanceCoreClient, "kelly_sizing"):
        return None

    return None  # pragma: no cover - AU-CONTEXT-R007.2 wires the real instance.


def kelly_size_via_eg_or_local(
    request: KellySizingRequest,
    *,
    local_fallback: Callable[[KellySizingRequest], KellySizingResponse],
) -> KellySizingResponse:
    """Delegate Kelly sizing to EG's finance core when available, else local.

    This is the one call site callers should use instead of calling
    ``engine_finance.calculate_kelly_size`` (or an equivalent local
    calculation) directly, so that AU-CONTEXT-R007.2 can swap the
    implementation out from under callers without touching their code.
    """
    client = resolve_finance_primitives_client()
    if client is not None:
        return client.kelly_sizing(request)
    return local_fallback(request)
