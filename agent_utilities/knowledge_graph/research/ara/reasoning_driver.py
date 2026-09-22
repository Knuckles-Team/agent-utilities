#!/usr/bin/python
from __future__ import annotations

"""Native OWL reasoning boundary for ARA (CONCEPT:AU-KG.research.best-effort-lightweight-never).

The epistemic-graph OwlReason operation is read-only and returns a typed class
closure. It does not promote arbitrary AU nodes into an OWL store or downfeed
inferred property edges into the graph. ARA's former cross-domain edge harvest
therefore remains unavailable until EG exposes that exact result contract.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class InferenceHarvest:
    """Native reasoner status; no inferred graph edges are synthesized."""

    stats: dict[str, Any] = field(default_factory=dict)
    inferred_edges: list[dict[str, Any]] = field(default_factory=list)
    new_topics: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""


class OntologyReasoningDriver:
    """Run EG-native OwlReason without local promotion or graph downfeed."""

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    def extrapolate(
        self,
        *,
        persist: bool = True,
    ) -> InferenceHarvest:
        """Run native OWL classification and report unsupported edge harvesting.

        ``persist`` cannot enable the former behavior: the
        current EG contract does not return inferred property edges or proof-linked
        cross-domain relations to materialize as research topics.
        """
        graph = getattr(self._engine, "graph", None)
        reason = getattr(graph, "owl_reason", None)
        if not callable(reason):
            return InferenceHarvest(
                error="native GraphComputeEngine.owl_reason is unavailable"
            )

        try:
            result = reason(class_base="http://agent-utilities.dev/ontology#")
        except Exception as exc:  # noqa: BLE001 — surface engine/schema failure explicitly
            logger.error("Native OWL reasoning failed: %s", exc)
            return InferenceHarvest(error=str(exc))

        if not isinstance(result, dict):
            return InferenceHarvest(error="native OwlReason returned an invalid result")

        stats = {
            "mode": "read_only",
            "consistent": result.get("consistent"),
            "schema_digests": result.get("schema_digests", []),
            "subclass_entailments": len(result.get("subclasses", [])),
            "direct_subclass_entailments": len(result.get("direct_subclasses", [])),
            "instance_entailments": len(result.get("instances", [])),
        }
        error = self._reasoning_result_error(result, stats)
        if error is not None:
            return InferenceHarvest(stats=stats, error=error)

        return InferenceHarvest(
            stats=stats,
            error=(
                "native OwlReason is read-only and does not return inferred property "
                "edges; ARA cross-domain edge harvesting/downfeed is unsupported"
            ),
        )

    @staticmethod
    def _reasoning_result_error(
        result: dict[str, Any], stats: dict[str, Any]
    ) -> str | None:
        if result.get("consistent") is not True:
            return "native OwlReason did not prove committed ontology consistency"
        if not stats["schema_digests"]:
            return "native OwlReason omitted committed GraphSchema digests"
        return None


__all__ = ["InferenceHarvest", "OntologyReasoningDriver"]
