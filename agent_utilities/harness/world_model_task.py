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


#: The engine node label whose committed changes drive reactive re-specialization.
WORLD_MODEL_TRANSITION_LABEL = "WorldModelTransition"


def world_model_subscription(engine: Any) -> Any:
    """Build the reactive change-feed subscription over ``WorldModelTransition``.

    CONCEPT:KG-2.251 — the poll→push seam that replaces the every-tick full
    history re-scan: the daemon polls this on its tick and only re-specializes the
    world model when the engine pushed a NEW transition (or on cold-start
    catch-up), instead of re-querying the entire transition corpus each time.

    Resolves the engine's content-graph compute (``__commons__`` — where
    :meth:`WorldModel.record_observation` writes the nodes) and subscribes to its
    label-filtered CDC feed. The handler bumps a counter; the daemon reads
    ``sub.pending_state["pending"]`` to decide whether the expensive
    specialization is worth running. Returns a
    :class:`~agent_utilities.graph.reactive.EngineSubscription` (its ``.available``
    is ``False`` — a permanent no-op — when no engine streaming surface exists, so
    the caller's periodic catch-up stays correct).
    """
    from agent_utilities.graph.reactive import subscribe

    source = getattr(engine, "graph_compute", None) or engine

    state = {"pending": 0}

    def _on_transition(_event: dict[str, Any]) -> None:
        state["pending"] += 1

    sub = subscribe(source, WORLD_MODEL_TRANSITION_LABEL, _on_transition)
    # Expose the live counter on the subscription so the tick can read/reset it
    # without re-implementing handler bookkeeping.
    sub.pending_state = state  # type: ignore[attr-defined]
    return sub


def transitions_from_engine(engine: Any, *, limit: int = 2000) -> list[Transition]:
    """Read persisted ``WorldModelTransition`` rows from the engine as tuples.

    Mirrors :meth:`WorldModel.from_engine`'s query; best-effort (empty on no
    engine/query support). These are the observations the daemon tick specializes
    a learned dynamics model against.
    """
    try:
        rows = engine.query_cypher(
            "MATCH (t:WorldModelTransition) "
            "RETURN t.state AS state, t.action AS action, t.next_state AS next_state "
            f"LIMIT {int(limit)}"
        )
    except Exception:  # noqa: BLE001 — no engine/query support ⇒ no transitions
        return []
    out: list[Transition] = []
    for r in rows or []:
        if isinstance(r, dict) and r.get("state"):
            out.append(
                (
                    str(r["state"]),
                    str(r.get("action") or ""),
                    str(r.get("next_state") or ""),
                )
            )
    return out


def specialize_world_model_from_engine(
    engine: Any,
    *,
    min_transitions: int = 20,
    holdout_frac: float = 0.2,
    rounds: int = 1,
    certifier: Any = None,
) -> dict[str, Any] | None:
    """Run one world-model specialization cycle grounded in the engine's history.

    The AU-native live path the SAI-factory daemon tick (AHE-3.29) drives: ground a
    learned dynamics model in persisted ``WorldModelTransition`` history, specialize
    its config via :class:`SaiFactoryController`, optionally certify it, and persist
    a queryable ``SaiFactoryCycle`` node. Returns the cycle summary, or ``None`` when
    there is too little history to split train/holdout.
    """
    from agent_utilities.knowledge_graph.research.sai_factory import (
        SaiFactoryController,
    )

    transitions = transitions_from_engine(engine)
    if len(transitions) < min_transitions:
        return None
    cut = max(1, int(len(transitions) * (1.0 - holdout_frac)))
    train, holdout = transitions[:cut], transitions[cut:]
    if not holdout:
        return None

    task = build_world_model_task(train, holdout)
    # The candidate IS the config scaffold for this non-LLM track (no generator LLM).
    result = SaiFactoryController(task, generate_fn=lambda scaffold: scaffold).run(
        rounds=rounds
    )
    summary: dict[str, Any] = result.metrics()
    summary["transitions"] = len(transitions)

    if certifier is not None and task.human_baseline is not None:
        samples = [task.score(result.specialist.generate()).reward for _ in range(3)]
        summary["certification"] = certifier.certify(
            samples, task.human_baseline
        ).to_dict()

    if engine is not None:
        import json
        import time
        import uuid

        try:
            engine.add_node(
                f"sai_factory_cycle:{uuid.uuid4().hex[:12]}",
                "SaiFactoryCycle",
                properties={
                    "task_id": task.task_id,
                    "metrics_json": json.dumps(summary),
                    "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
            )
        except Exception:  # noqa: BLE001 — persistence is best-effort
            pass
    return summary
