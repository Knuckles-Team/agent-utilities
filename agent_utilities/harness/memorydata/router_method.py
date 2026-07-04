#!/usr/bin/python
from __future__ import annotations

"""Family-aware retrieval router over the graph-os memory configs (CONCEPT:AU-AHE.harness.callers-feed-back-per).

A single retrieval surface is rarely best across MemoryData's heterogeneous families: an
update/conflict family wants bi-temporal point-in-time recall, a code family wants
context-plane synthesis, a long-document family wants latent retrieval. :class:`GraphOSRouterMethod`
picks the retrieval config per query from the family tag (a default prior table), then answers
through that config's spec — so the *router* competes against every single config in the
bake-off. ``record_outcome`` updates a per-config EMA weight, an online-learning hook the
caller can use to bias future routing toward configs that paid off.
"""

from typing import Any

from agent_utilities.harness.memorydata.adapter import (
    RETRIEVAL_CONFIGS,
    GraphOSMemoryMethod,
)

__all__ = ["GraphOSRouterMethod", "DEFAULT_FAMILY_PRIORS"]


# Default family-tag → retrieval-config priors. Substring-matched against the family tag so a
# preset like ``membench-update`` or ``locomo-singlehop`` resolves without an exact key.
DEFAULT_FAMILY_PRIORS: dict[str, str] = {
    "membench-update": "graphos_bitemporal_asof",
    "locomo": "graphos_context_plane",
    "longbench": "graphos_latent",
    "conflict": "graphos_bitemporal_asof",
    "memoryagentbench": "graphos_bitemporal_asof",
    "membench-recall": "graphos_graph_rerank",
}
_DEFAULT_CONFIG = "graphos_semantic_hnsw"


class GraphOSRouterMethod(GraphOSMemoryMethod):
    """Routes each query to a retrieval config by family tag (CONCEPT:AU-AHE.harness.callers-feed-back-per).

    Subclasses :class:`GraphOSMemoryMethod`, so it satisfies the same MemoryData
    ``send_message`` contract; it only overrides *which* retrieval spec a query uses. Keep it
    dependency-free — the EMA hook is plain arithmetic.
    """

    def __init__(
        self,
        agent_config: dict[str, Any],
        dataset_config: dict[str, Any] | None = None,
        load_agent_from: str | None = None,
        *,
        family_tag: str | None = None,
        priors: dict[str, float] | None = None,
        ema_alpha: float = 0.3,
    ) -> None:
        # The base method needs a concrete retrieval config; seed it with the routed choice so
        # ``__init__`` validation passes, then keep routing per query in ``send_message``.
        self.family_tag = (
            family_tag or (dataset_config or {}).get("sub_dataset") or ""
        ).lower()
        seed_config = self._select_config({"family_tag": self.family_tag})
        seeded = dict(agent_config or {})
        seeded.setdefault("retrieval", seed_config)
        seeded["retrieval"] = seeded.get("retrieval") or seed_config
        # Force the seed so an unset/invalid inherited value never breaks construction.
        seeded["retrieval"] = seed_config
        super().__init__(seeded, dataset_config, load_agent_from)

        self.ema_alpha = ema_alpha
        # Per-config EMA weight (online reward signal); seeded from optional priors.
        self.weights: dict[str, float] = {name: 1.0 for name in RETRIEVAL_CONFIGS}
        if priors:
            for name, weight in priors.items():
                if name in self.weights:
                    self.weights[name] = float(weight)

    def _select_config(self, eval_metadata: dict[str, Any] | None) -> str:
        """Pick a retrieval config from the family tag via the default prior table."""
        tag = ""
        if eval_metadata:
            tag = str(
                eval_metadata.get("family_tag") or eval_metadata.get("family") or ""
            ).lower()
        if not tag:
            tag = getattr(self, "family_tag", "") or ""
        for needle, config in DEFAULT_FAMILY_PRIORS.items():
            if needle in tag:
                return config
        return _DEFAULT_CONFIG

    def send_message(
        self,
        message: Any,
        memorizing: bool = False,
        query_id: Any | None = None,
        context_id: Any | None = None,
        eval_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Route the query to its family's retrieval spec, then defer to the base method."""
        if not memorizing:
            selected = self._select_config(
                eval_metadata or {"family_tag": self.family_tag}
            )
            self.retrieval = selected
            self.spec = RETRIEVAL_CONFIGS[selected]
        return super().send_message(
            message,
            memorizing=memorizing,
            query_id=query_id,
            context_id=context_id,
            eval_metadata=eval_metadata,
        )

    def record_outcome(self, config: str, reward: float) -> float:
        """Update the EMA weight for ``config`` with ``reward`` and return the new weight.

        Online-learning hook (CONCEPT:AU-AHE.harness.callers-feed-back-per): callers feed back a per-query reward so the
        router can bias future selection toward configs that performed well.
        """
        if config not in self.weights:
            self.weights[config] = 1.0
        prior = self.weights[config]
        updated = (1.0 - self.ema_alpha) * prior + self.ema_alpha * float(reward)
        self.weights[config] = updated
        return updated
