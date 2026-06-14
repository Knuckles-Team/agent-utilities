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
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

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


def _hash_embed(text: str, dim: int) -> Any:
    """Deterministic, dependency-free embedding of ``text`` into a ``dim``-vector.

    Hashes character 3-grams into ``dim`` buckets with signed counts (the hashing
    trick) and L2-normalizes. Deterministic across processes (uses ``blake2b``, not
    the salted built-in ``hash``), so the learned backend and its verifier produce
    reproducible curves. Pass a real embedder (``create_embedding_model``) for
    production grounding; this default keeps the backend importable + CPU-testable
    with no model download.
    """
    import hashlib

    import numpy as np

    v = np.zeros(dim, dtype=np.float64)
    s = f"  {text} "
    for i in range(len(s) - 2):
        gram = s[i : i + 3]
        h = int.from_bytes(
            hashlib.blake2b(gram.encode(), digest_size=8).digest(), "little"
        )
        v[h % dim] += 1.0 if (h >> 7) & 1 else -1.0
    norm = float(np.linalg.norm(v))
    return v / norm if norm else v


class LatentDynamicsModel:
    """Learned parametric backend for :class:`WorldModel` (CONCEPT:KG-2.73).

    The parametric counterpart to the symbolic Markov kernel: it embeds states and
    actions, fits a ridge-regression map ``[embed(state) ; embed(action)] → embed(next_state)``
    over observed transitions, and predicts the next state by nearest-neighbour over
    the known next-state embeddings. Unlike the exact-match Markov model it
    **generalizes to unseen ``(state, action)`` pairs** — the property the paper's
    "compressed, manipulable representation of environment dynamics" requires — while
    honouring the same ``predict`` contract so :class:`WorldModel` callers are unchanged.

    Pure numpy + a dependency-free default embedder ⇒ CPU-trainable and unit-testable;
    inject ``embed_fn`` to ground it in the real graph embedder.
    """

    def __init__(
        self,
        embed_fn: Callable[[str], Any] | None = None,
        *,
        dim: int = 64,
        alpha: float = 1.0,
        neighbors: int = 3,
    ) -> None:
        self.dim = int(dim)
        self.alpha = float(alpha)
        self.neighbors = int(neighbors)
        self._embed = embed_fn or (lambda t: _hash_embed(t, self.dim))
        self._w: Any = None
        self._known: dict[str, Any] = {}
        self._buffer: list[tuple[str, str, str]] = []
        self._dirty = False

    def observe(self, state: str, action: str, next_state: str) -> None:
        self._buffer.append((state, action, next_state))
        self._dirty = True

    def _feature(self, state: str, action: str) -> Any:
        import numpy as np

        return np.concatenate([self._embed(state), self._embed(action)])

    def fit(self) -> None:
        """Solve the ridge map from the observed-transition buffer."""
        import numpy as np

        if not self._buffer:
            self._w = None
            self._dirty = False
            return
        x = np.stack([self._feature(s, a) for s, a, _ in self._buffer])
        y = np.stack([self._embed(ns) for _, _, ns in self._buffer])
        d = x.shape[1]
        self._w = np.linalg.solve(x.T @ x + self.alpha * np.eye(d), x.T @ y)
        self._known = {ns: self._embed(ns) for _, _, ns in self._buffer}
        self._dirty = False

    def predict(self, state: str, action: str, k: int = 1) -> list[tuple[str, float]]:
        """Top-``k`` ``(next_state, score∈[0,1])`` by cosine to the predicted latent."""
        import numpy as np

        if self._dirty or self._w is None:
            self.fit()
        if self._w is None or not self._known:
            return []
        y_hat = self._feature(state, action) @ self._w
        ny = float(np.linalg.norm(y_hat)) or 1.0
        scored = [
            (ns, float(emb @ y_hat) / ((float(np.linalg.norm(emb)) or 1.0) * ny))
            for ns, emb in self._known.items()
        ]
        scored.sort(key=lambda p: p[1], reverse=True)
        return [(ns, max(0.0, (sim + 1.0) / 2.0)) for ns, sim in scored[: max(1, k)]]


class WorldModel:
    """An action-conditioned, forward-simulatable model of graph state dynamics."""

    def __init__(
        self,
        engine: Any = None,
        *,
        backend: str = "symbolic",
        embed_fn: Callable[[str], Any] | None = None,
        latent_dim: int = 64,
        ridge_alpha: float = 1.0,
        neighbors: int = 3,
    ) -> None:
        self.engine = engine
        self.backend = backend
        from agent_utilities.knowledge_graph.core.formal_reasoning_core import (
            MarkovTransitionModel,
        )

        self._transitions = MarkovTransitionModel()
        self._rewards: dict[tuple[str, str], list[float]] = {}
        # Learned backend (KG-2.73): generalizes to unseen (state, action). The
        # symbolic Markov kernel is always kept for reward bookkeeping + the default
        # path; ``backend="latent"`` routes prediction through the parametric model.
        self._latent: LatentDynamicsModel | None = (
            LatentDynamicsModel(
                embed_fn, dim=latent_dim, alpha=ridge_alpha, neighbors=neighbors
            )
            if backend == "latent"
            else None
        )

    # ── learning ─────────────────────────────────────────────────────
    def observe(
        self, state: str, action: str, next_state: str, reward: float | None = None
    ) -> None:
        """Record one observed transition (and its reward, if known)."""
        self._transitions.ingest_trace([_key(state, action), next_state])
        if self._latent is not None:
            self._latent.observe(state, action, next_state)
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
        """Top-``k`` ``(next_state, probability)`` for taking ``action`` in ``state``.

        Routes through the learned latent-dynamics backend (KG-2.73) when
        ``backend="latent"`` (generalizes to unseen pairs), else the symbolic
        Markov kernel (exact-match).
        """
        if self._latent is not None:
            return self._latent.predict(state, action, k=k)
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

    def expected_return(
        self, rollout: list[Transition], *, gamma: float = 0.95
    ) -> float:
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

    def persist_rollout(
        self, rollout: list[Transition], *, gamma: float = 0.95
    ) -> str | None:
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
                    "expected_return": round(
                        self.expected_return(rollout, gamma=gamma), 4
                    ),
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
