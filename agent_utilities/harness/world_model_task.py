#!/usr/bin/python
from __future__ import annotations

"""World-model prediction as a SAI specialization domain (CONCEPT:KG-2.73).

The second concrete specialization track for the SAI factory (AHE-3.29), alongside
GPU kernels — a *non-LLM, learned-dynamics* track that honours the paper's
architectural-diversity pillar (world models, latent prediction). A candidate is a
JSON config for the learned :class:`~agent_utilities.knowledge_graph.core.world_model.LatentDynamicsModel`
(embedding dim, ridge alpha, neighbours); the :class:`WorldModelVerifier` trains
that model on a task's transition corpus and rewards it by **held-out next-state
prediction accuracy** — so the SAI factory searches/distills configs that build a
*better forward model of environment dynamics*, measured by adaptation speed.

This makes "improve the world model" a first-class, machine-verifiable
specialization the same controller drives, with no LLM in the loop.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from agent_utilities.harness.sai_task import SpecializationTask, VerifierResult
from agent_utilities.knowledge_graph.core.world_model import LatentDynamicsModel

#: A transition is ``(state, action, next_state)``.
Transition = tuple[str, str, str]


@dataclass
class WorldModelVerifier:
    """Reward a learned-dynamics config by held-out next-state prediction accuracy.

    ``reward`` = top-1 hit-rate over the held-out transitions (fraction whose true
    next-state is the model's most-likely prediction); ``passed`` when the model
    predicts better than the empirical majority-class baseline (so a degenerate
    "always predict the most common next-state" config does not pass).
    """

    train: Sequence[Transition]
    holdout: Sequence[Transition]
    pass_margin: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def _majority_baseline(self) -> float:
        from collections import Counter

        if not self.holdout:
            return 0.0
        counts = Counter(ns for _, _, ns in self.holdout)
        return counts.most_common(1)[0][1] / len(self.holdout)

    def verify(self, candidate: str) -> VerifierResult:
        try:
            cfg = json.loads(candidate) if candidate.strip() else {}
            if not isinstance(cfg, dict):
                raise ValueError("config must be a JSON object")
        except (ValueError, AttributeError) as exc:
            return VerifierResult(
                reward=0.0, passed=False, detail={"error": f"bad config: {exc!r}"}
            )

        try:
            model = LatentDynamicsModel(
                dim=int(cfg.get("dim", 64)),
                alpha=float(cfg.get("alpha", 1.0)),
                neighbors=int(cfg.get("neighbors", 1)),
            )
        except (TypeError, ValueError) as exc:
            return VerifierResult(
                reward=0.0, passed=False, detail={"error": f"bad params: {exc!r}"}
            )

        for s, a, ns in self.train:
            model.observe(s, a, ns)
        model.fit()

        if not self.holdout:
            return VerifierResult(
                reward=0.0, passed=False, detail={"error": "no holdout"}
            )

        hits = 0
        for s, a, ns in self.holdout:
            preds = model.predict(s, a, k=1)
            if preds and preds[0][0] == ns:
                hits += 1
        accuracy = hits / len(self.holdout)
        baseline = self._majority_baseline()
        return VerifierResult(
            reward=accuracy,
            passed=accuracy > baseline + self.pass_margin,
            detail={
                "accuracy": accuracy,
                "majority_baseline": baseline,
                "holdout": len(self.holdout),
            },
        )


def _default_config_scaffolds() -> list[str]:
    """A spread of learned-dynamics configs the factory's scaffold arm searches."""
    return [
        json.dumps({"dim": 16, "alpha": 1.0, "neighbors": 1}),
        json.dumps({"dim": 64, "alpha": 0.1, "neighbors": 1}),
        json.dumps({"dim": 128, "alpha": 0.01, "neighbors": 1}),
    ]


def build_world_model_task(
    train: Sequence[Transition],
    holdout: Sequence[Transition],
    *,
    target_tau: float = 0.8,
    human_baseline: float | None = None,
    scaffolds: Sequence[str] | None = None,
) -> SpecializationTask:
    """Build a SAI :class:`SpecializationTask` that specializes a learned world model."""
    return SpecializationTask(
        task_id="world-model:latent-dynamics",
        prompt_corpus=list(scaffolds) if scaffolds else _default_config_scaffolds(),
        verifier=WorldModelVerifier(train=list(train), holdout=list(holdout)),
        target_tau=target_tau,
        human_baseline=human_baseline,
        metadata={"domain": "world-model", "backend": "latent"},
    )
