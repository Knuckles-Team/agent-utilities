#!/usr/bin/python
from __future__ import annotations

"""Action-conditioned world model over the graph's transition kernels.

CONCEPT:KG-2.67 — a first-class action-conditioned world model wrapping the graph's Markov transition kernel so an agent can roll an action policy forward over predicted next-states and rewards instead of only retrieving the past

The epistemic graph holds rich *descriptive* state, but every "model" is an LLM, a
scikit regressor (AHE-3.3) or a symbolic OWL closure — there is no manipulable
predictive model of *how an action changes state* that an agent can roll forward.
This adds that first-class abstraction:

    state x action -> next_state + expected_reward

The symbolic backend wraps the existing :class:`MarkovTransitionModel`
(`formal_reasoning_core`) by keying transitions on a composite ``state|action`` so the
same kernel becomes action-conditioned, plus a reward table learned from observed
transitions. ``from_engine`` grounds it in persisted outcome history; ``rollout``
forward-simulates a policy and ``persist_rollout`` writes the imagined trajectory back
as graph nodes (graph-native). A parametric/learned backend can later implement the
same ``step`` contract without changing callers.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

_SEP = "\x1f"  # composite "state|action" key separator (unit-separator, collision-safe)


def _key(state: str, action: str) -> str:
    return f"{state}{_SEP}{action}"


@dataclass
class Transition:
    """One predicted ``state × action → next_state`` step with its reward."""

    state: str
    action: str
    next_state: str
    probability: float = 0.0
    reward: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "action": self.action,
            "next_state": self.next_state,
            "probability": round(self.probability, 4),
            "reward": round(self.reward, 4),
        }


class WorldModel:
    """An action-conditioned, forward-simulatable model of graph state dynamics."""

    def __init__(self, engine: Any = None) -> None:
        self.engine = engine
        from agent_utilities.knowledge_graph.core.formal_reasoning_core import (
            MarkovTransitionModel,
        )

        self._transitions = MarkovTransitionModel()
        self._rewards: dict[tuple[str, str], list[float]] = {}

    # ── learning ─────────────────────────────────────────────────────
    def observe(
        self, state: str, action: str, next_state: str, reward: float | None = None
    ) -> None:
        """Record one observed transition (and its reward, if known)."""
        self._transitions.ingest_trace([_key(state, action), next_state])
        if reward is not None:
            self._rewards.setdefault((state, action), []).append(float(reward))

    def learn(self, transitions: list[tuple]) -> int:
        """Batch-observe ``(state, action, next_state[, reward])`` tuples."""
        n = 0
        for t in transitions:
            state, action, next_state = t[0], t[1], t[2]
            reward = t[3] if len(t) > 3 else None
            self.observe(state, action, next_state, reward)
            n += 1
        return n

    @classmethod
    def from_engine(cls, engine: Any, *, limit: int = 2000) -> WorldModel:
        """Ground a world model in persisted transition history from the KG.

        Reads ``WorldModelTransition`` observation nodes (state/action/next_state/
        reward) if any have been recorded; best-effort, returns an empty model when
        none exist or the engine has no query support.
        """
        wm = cls(engine)
        try:
            rows = engine.query_cypher(
                "MATCH (t:WorldModelTransition) "
                "RETURN t.state AS state, t.action AS action, "
                "t.next_state AS next_state, t.reward AS reward "
                f"LIMIT {int(limit)}"
            )
        except Exception as exc:  # noqa: BLE001 — no engine/query support ⇒ empty model
            logger.debug("[KG-2.67] from_engine found no transition history: %s", exc)
            rows = []
        for r in rows or []:
            if not isinstance(r, dict) or not r.get("state"):
                continue
            reward = r.get("reward")
            wm.observe(
                str(r["state"]),
                str(r.get("action") or ""),
                str(r.get("next_state") or ""),
                float(reward) if reward is not None else None,
            )
        return wm

    # ── prediction ───────────────────────────────────────────────────
    def predict(self, state: str, action: str, k: int = 1) -> list[tuple[str, float]]:
        """Top-``k`` ``(next_state, probability)`` for taking ``action`` in ``state``."""
        return self._transitions.predict_next_states(_key(state, action), k=k)

    def expected_reward(self, state: str, action: str) -> float:
        rs = self._rewards.get((state, action), [])
        return sum(rs) / len(rs) if rs else 0.0

    def step(self, state: str, action: str) -> Transition:
        """The most-likely single transition (absorbing in ``state`` if unseen)."""
        preds = self.predict(state, action, k=1)
        if preds:
            next_state, prob = preds[0]
        else:
            next_state, prob = state, 0.0  # unknown ⇒ assume no-op
        return Transition(
            state=state,
            action=action,
            next_state=next_state,
            probability=float(prob),
            reward=self.expected_reward(state, action),
        )

    def rollout(
        self, state: str, policy: Callable[[str], str], horizon: int
    ) -> list[Transition]:
        """Forward-simulate ``policy`` (``state -> action``) for ``horizon`` steps."""
        traj: list[Transition] = []
        cur = state
        for _ in range(max(0, int(horizon))):
            action = policy(cur)
            t = self.step(cur, action)
            traj.append(t)
            cur = t.next_state
        return traj

    def expected_return(self, rollout: list[Transition], *, gamma: float = 0.95) -> float:
        """Discounted sum of predicted rewards over a rollout."""
        return sum((gamma**i) * t.reward for i, t in enumerate(rollout))

    # ── persistence ──────────────────────────────────────────────────
    def record_observation(
        self, state: str, action: str, next_state: str, reward: float | None = None
    ) -> None:
        """Observe AND persist a ``WorldModelTransition`` node so ``from_engine``
        can re-ground a future model from the durable history."""
        self.observe(state, action, next_state, reward)
        if self.engine is None:
            return
        import uuid

        try:
            self.engine.add_node(
                f"wm_transition:{uuid.uuid4().hex[:12]}",
                "WorldModelTransition",
                properties={
                    "state": state,
                    "action": action,
                    "next_state": next_state,
                    "reward": "" if reward is None else float(reward),
                    "recorded_at": _now_iso(),
                },
            )
        except Exception as exc:  # noqa: BLE001 — persistence is best-effort
            logger.debug("[KG-2.67] could not persist transition: %s", exc)

    def persist_rollout(self, rollout: list[Transition], *, gamma: float = 0.95) -> str | None:
        """Write an imagined trajectory back as a graph-native ``WorldModelRollout``."""
        if self.engine is None:
            return None
        import json
        import uuid

        rollout_id = f"wm_rollout:{uuid.uuid4().hex[:12]}"
        try:
            self.engine.add_node(
                rollout_id,
                "WorldModelRollout",
                properties={
                    "horizon": len(rollout),
                    "expected_return": round(self.expected_return(rollout, gamma=gamma), 4),
                    "steps_json": json.dumps([t.as_dict() for t in rollout]),
                    "recorded_at": _now_iso(),
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("[KG-2.67] could not persist rollout: %s", exc)
            return None
        return rollout_id


def _now_iso() -> str:
    import time

    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
