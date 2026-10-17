#!/usr/bin/python
from __future__ import annotations

"""Pluggable reasoning paradigms + an outcome-learning paradigm router.

CONCEPT:AU-KG.compute.first-class-reasoner-paradigm — a first-class Reasoner paradigm abstraction with an outcome-learning router that selects a reasoning paradigm per task by capability tags blended with a learned reward EMA and feeds the scored result back so routing self-improves

AU is model-agnostic (any provider backs any role, ORCH-1.27) and symbolic-backend-
pluggable (KG-2.23) but NOT *paradigm*-agnostic: there was no seam at which an
alternative way of *thinking* slots in behind the orchestrator. This adds it — and,
crucially, the router that consumes it, so it is a live loop rather than a dead seam.

Design (why this is robust, not an if/else switch):

* A :class:`Reasoner` advertises ``capability_tags`` and implements
  ``reason(task) -> ReasoningResult`` that self-reports a ``score`` in ``[0, 1]``.
* The :class:`ReasonerRouter` receives paradigm designation from EG through the
  capability-designation port (``EG-DECISION-ENGINE-R172``) — EG gates by required
  capability tags and ranks with its **learned reward**; without a port the
  router uses the declared tag-overlap rule. After running, the router forwards
  the result's score to EG via ``record_outcome``, so the router *learns which paradigm works for which task
  class* (the recursive-improvement pathway applied to thinking itself).

The built-in paradigms wire the rest of the seam together: inductive program
synthesis (KG-2.69), model-based planning over the world model (KG-2.67), and pure
deductive forward-chaining. A generative paradigm slots in behind an injected
completion fn. New paradigms register without touching any caller.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

_ROUTING_DIM = 8  # tag-primary routing: a uniform embedding ⇒ reward-EMA orders


@dataclass
class ReasoningTask:
    """One reasoning request. ``tags`` route it; ``payload`` carries paradigm input."""

    goal: str
    tags: tuple[str, ...] = ()
    payload: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None  # optional semantic routing vector


@dataclass
class ReasoningResult:
    """A paradigm's answer plus the self-reported score that trains the router."""

    answer: Any
    reasoner: str
    score: float = 0.0  # correctness/confidence in [0, 1] — fed back as routing reward
    trace: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Reasoner(Protocol):
    name: str
    capability_tags: tuple[str, ...]

    def reason(self, task: ReasoningTask) -> ReasoningResult: ...


# ── built-in paradigms ───────────────────────────────────────────────


class ProgramSynthesisReasoner:
    """Inductive paradigm: synthesize the shortest program fitting examples (KG-2.69)."""

    name = "program_synthesis"
    capability_tags: tuple[str, ...] = ("induction", "examples", "symbolic")

    def reason(self, task: ReasoningTask) -> ReasoningResult:
        from agent_utilities.harness.program_synthesis import synthesize

        prog = synthesize(
            task.payload["primitives"],
            task.payload["examples"],
            max_depth=task.payload.get("max_depth", 3),
            mdl_weight=task.payload.get("mdl_weight", 0.5),
        )
        if prog is None:
            return ReasoningResult(None, self.name, 0.0, {"fit": "none"})
        return ReasoningResult(
            answer=prog,
            reasoner=self.name,
            score=prog.score,
            trace={"ops": list(prog.ops)},
        )


class DeductiveReasoner:
    """Symbolic paradigm: forward-chain rules to a fixpoint over a fact set."""

    name = "deductive"
    capability_tags: tuple[str, ...] = ("symbolic", "logic", "deduction")

    def reason(self, task: ReasoningTask) -> ReasoningResult:
        facts = set(task.payload.get("facts", ()))
        rules = task.payload.get("rules", ())  # list of (premises, conclusion)
        derived = set(facts)
        changed = True
        while changed:
            changed = False
            for premises, conclusion in rules:
                if set(premises) <= derived and conclusion not in derived:
                    derived.add(conclusion)
                    changed = True
        goal = task.payload.get("goal_fact")
        if goal is not None:
            score = 1.0 if goal in derived else 0.0
        else:
            score = 1.0 if len(derived) > len(facts) else 0.0
        return ReasoningResult(
            answer=derived,
            reasoner=self.name,
            score=score,
            trace={"derived": len(derived)},
        )


class GenerativeReasoner:
    """Generative paradigm: an injected completion fn, optionally verifier-scored."""

    name = "generative"
    capability_tags: tuple[str, ...] = ("generative", "qa", "language")

    def __init__(
        self,
        llm_fn: Callable[[str], str] | None = None,
        verifier: Callable[[str, ReasoningTask], float] | None = None,
    ) -> None:
        self._llm_fn = llm_fn
        self._verifier = verifier

    def reason(self, task: ReasoningTask) -> ReasoningResult:
        fn = self._llm_fn or task.payload.get("llm_fn")
        if fn is None:
            from agent_utilities.knowledge_graph.enrichment.cards import (
                make_lite_llm_fn,
            )

            fn = make_lite_llm_fn()
        prompt = task.payload.get("prompt", task.goal)
        answer = str(fn(prompt) or "")
        verifier = self._verifier or task.payload.get("verifier")
        score = float(verifier(answer, task)) if verifier else (0.5 if answer else 0.0)
        return ReasoningResult(
            answer=answer, reasoner=self.name, score=max(0.0, min(1.0, score)), trace={}
        )


# ── the learning router ──────────────────────────────────────────────


class ReasonerRouter:
    """Route a task to a reasoning paradigm and learn from the outcome (KG-2.68)."""

    def __init__(
        self,
        *,
        reward_weight: float = 0.3,
        harvester: Any = None,
        designator: Any = None,
    ) -> None:
        # EG owns paradigm designation and its reward learning
        # (EG-DECISION-ENGINE-R172). The router is GIVEN that port; AU keeps no
        # index and no reward EMA. Without a port it uses the declared tag rule.
        self._designator = designator
        self._reasoners: dict[str, Reasoner] = {}
        self._reward_weight = float(reward_weight)
        # CONCEPT:AU-AHE.harness.search-distillation-harvester — optional search-distillation harvester. When attached,
        # the router's verified high-scoring results are distilled into a training
        # corpus (test-time compute → training data); off by default.
        self._harvester = harvester

    @staticmethod
    def _uniform_embedding() -> list[float]:
        # Uniform vector ⇒ equal cosine for all paradigms, so designate() orders by
        # the capability-tag filter + the learned reward EMA (tag-primary routing).
        return [1.0 / _ROUTING_DIM] * _ROUTING_DIM

    def register(self, reasoner: Reasoner) -> None:
        self._reasoners[reasoner.name] = reasoner

    def _designate(
        self, task: ReasoningTask, k: int, *, tags: bool = True
    ) -> list[Any]:
        """EG-designated paradigms (empty without a port or on port failure)."""
        if self._designator is None:
            return []
        from agent_utilities.knowledge_graph.retrieval.capability_designation_port import (
            CapabilityDesignationRequest,
        )

        try:
            return list(
                self._designator.designate(
                    CapabilityDesignationRequest(
                        query_vector=list(task.embedding or self._uniform_embedding()),
                        required_capabilities=tuple(task.tags) if tags else (),
                        k=k,
                    )
                )
            )
        except Exception as exc:  # noqa: BLE001 — the declared tag rule below is the documented fallback
            logger.debug("[KG-2.68] EG designation unavailable: %s", exc)
            return []

    def _record(self, name: str, reward: float) -> None:
        """Forward a paradigm outcome to EG (no-op without a port)."""
        if self._designator is None:
            return
        try:
            self._designator.record_outcome(name, reward=max(0.0, min(1.0, reward)))
        except Exception as exc:  # noqa: BLE001 — learning must never break reasoning
            logger.debug("[KG-2.68] outcome forward skipped: %s", exc)

    def route(self, task: ReasoningTask) -> Reasoner | None:
        """The best paradigm for this task (tag-gated, reward-EMA-ranked)."""
        if not self._reasoners:
            return None
        des = [d for d in self._designate(task, 1) if d.id in self._reasoners]
        if des:
            return self._reasoners.get(des[0].id)
        # No tag match ⇒ fall back to a tag-overlap reasoner, else the first.
        for r in self._reasoners.values():
            if set(task.tags) & set(r.capability_tags):
                return r
        return next(iter(self._reasoners.values()))

    def reason(self, task: ReasoningTask) -> ReasoningResult | None:
        """Route, run, and feed the score back so routing self-improves."""
        reasoner = self.route(task)
        if reasoner is None:
            return None
        try:
            result = reasoner.reason(task)
        except Exception as exc:  # noqa: BLE001 — a failing paradigm scores 0, never crashes routing
            logger.warning("[KG-2.68] reasoner %s failed: %s", reasoner.name, exc)
            self._record(reasoner.name, 0.0)
            return ReasoningResult(None, reasoner.name, 0.0, {"error": str(exc)})
        # Close the loop: the self-reported score trains the paradigm's routing reward.
        self._record(reasoner.name, result.score)
        result.trace.setdefault("routed_to", reasoner.name)
        # CONCEPT:AU-AHE.harness.search-distillation-harvester — distil a verified win into training data (test-time
        # compute → better data), collapse-guarded by SAFE-1.4. Best-effort.
        if self._harvester is not None:
            try:
                if self._harvester.harvest_result(task, result) is not None:
                    result.trace["distilled"] = True
            except Exception as exc:  # noqa: BLE001 — harvesting never breaks reasoning
                logger.debug("[OS-5.36] harvest skipped: %s", exc)
        return result

    def reason_adaptive(
        self, task: ReasoningTask, governor: Any = None
    ) -> ReasoningResult | None:
        """Try paradigms in learned-reward order, stopping when more compute is not
        worth it (CONCEPT:AU-OS.scaling.value-test-time-compute). Returns the best result, value-allocated: a
        satisficing or diminishing-returns verdict from the compute governor halts the
        search so test-time compute is spent only where the marginal return is high.
        """
        if governor is None:
            from agent_utilities.harness.compute_governor import ComputeGovernor

            governor = ComputeGovernor()
        des = self._designate(task, len(self._reasoners))
        ranked = [self._reasoners[d.id] for d in des if d.id in self._reasoners]
        if not ranked:
            r = self.route(task)
            ranked = [r] if r else []

        results: list[ReasoningResult] = []
        scores: list[float] = []
        for reasoner in ranked:
            if not governor.should_continue(scores):
                break
            try:
                result = reasoner.reason(task)
            except Exception as exc:  # noqa: BLE001 — a failing paradigm scores 0
                logger.warning("[KG-2.68] reasoner %s failed: %s", reasoner.name, exc)
                self._record(reasoner.name, 0.0)
                continue
            self._record(reasoner.name, result.score)
            if self._harvester is not None:
                try:
                    self._harvester.harvest_result(task, result)
                except Exception as exc:  # noqa: BLE001 — trajectory-harvesting is a secondary learning-signal capture AFTER result.score has already been recorded via record_outcome above and appended to results/scores below regardless
                    logger.debug("[OS-5.36] harvest skipped: %s", exc)
            results.append(result)
            scores.append(result.score)

        if not results:
            return None
        best = max(results, key=lambda r: r.score)
        best.trace["attempts"] = len(results)
        best.trace["governor"] = governor.report(scores)
        return best

    def reward(self, reasoner_name: str) -> float:
        """EG's learned routing reward for a paradigm (0.5 = neutral / no port)."""
        probe = ReasoningTask(goal="", tags=())
        for d in self._designate(probe, len(self._reasoners), tags=False):
            if d.id == reasoner_name:
                return float(d.provenance.get("reward", 0.5))
        return 0.5


_DEFAULT_ROUTER: ReasonerRouter | None = None


def get_reasoner_router() -> ReasonerRouter:
    """The process-wide router, pre-registered with the built-in paradigms."""
    global _DEFAULT_ROUTER
    if _DEFAULT_ROUTER is None:
        router = ReasonerRouter()
        router.register(ProgramSynthesisReasoner())
        router.register(DeductiveReasoner())
        router.register(GenerativeReasoner())
        _DEFAULT_ROUTER = router
    return _DEFAULT_ROUTER


def reason(task: ReasoningTask) -> ReasoningResult | None:
    """Entry point: route ``task`` to the best-learned paradigm and run it."""
    return get_reasoner_router().reason(task)
