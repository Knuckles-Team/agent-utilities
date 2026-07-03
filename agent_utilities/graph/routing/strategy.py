"""Router composition framework (Plan 03 Step 3).

Defines the ``RoutingStrategy`` protocol and a ``Router`` that runs an ordered
pipeline of strategies, returning the first non-``None`` decision. This is the
target home for the R1–R13 behaviours currently in ``graph/_router_impl.py``;
strategies are migrated into it incrementally (R1 fast-path is done).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class RoutingStrategy(Protocol):
    """A composable routing step.

    ``decide`` inspects the routing context and returns a decision (any truthy
    value the caller understands — a sentinel, a RoutingDecision, an End node)
    or ``None`` to defer to the next strategy in the pipeline.
    """

    name: str

    async def decide(self, ctx: Any) -> Any | None: ...


@dataclass
class RoutingConfig:
    """Ordered pipeline of strategy names plus a terminal fallback name.

    Reproduces the *sequence* of behaviours from the monolith (e.g. fast_path →
    team_reuse → kg_materialization → enrich → llm_planner → optimization →
    fallback) as named, reorderable steps.
    """

    pipeline: list[str] = field(
        default_factory=lambda: [
            "fast_path",
            "team_reuse",
            "kg_materialization",
            "self_model",
            "llm_planner",
            "optimization",
        ]
    )
    fallback: str = "fallback"


class Router:
    """Compose strategies into a single routing entrypoint.

    Strategies are keyed by ``.name`` and executed in ``config.pipeline`` order;
    the first non-``None`` decision wins. If none decide, ``route`` returns
    ``None`` and the caller applies its fallback chain (R13).
    """

    def __init__(
        self,
        strategies: list[RoutingStrategy] | None = None,
        config: RoutingConfig | None = None,
    ) -> None:
        self.config = config or RoutingConfig()
        self._by_name: dict[str, RoutingStrategy] = {}
        for s in strategies or []:
            self.register(s)

    def register(self, strategy: RoutingStrategy) -> None:
        self._by_name[strategy.name] = strategy

    @property
    def strategies(self) -> dict[str, RoutingStrategy]:
        return dict(self._by_name)

    async def route(self, ctx: Any) -> Any | None:
        """Run the configured pipeline; return the first non-None decision."""
        for name in self.config.pipeline:
            strategy = self._by_name.get(name)
            if strategy is None:
                continue
            decision = await strategy.decide(ctx)
            if decision is not None:
                return decision
        return None


def default_router() -> Router:
    """Construct a Router with the strategies extracted so far."""
    from .strategies.fast_path import FastPathStrategy

    return Router(strategies=[FastPathStrategy()])
