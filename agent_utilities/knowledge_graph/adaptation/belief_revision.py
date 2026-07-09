#!/usr/bin/python
from __future__ import annotations

"""Confidence propagation + a light truth-maintenance surface over BeliefNode graphs.

CONCEPT:AU-KG.maintenance.confidence-propagation-belief-revision

Sibling to :mod:`.contradiction_detector` in the same night-shift-Critic
tradition (Building a Second Brain): where the ``ContradictionDetector``
only *flags* that two beliefs oppose each other, this module turns that
friction — plus a belief's already-recorded ``supported_by_node_ids`` /
``contradicted_by_node_ids`` edges — into a recomputed, EXPLAINABLE
confidence. It is deliberately **propose-only**: :class:`BeliefRevisionPass`
returns :class:`BeliefRevision` records and never mutates a
:class:`~agent_utilities.models.knowledge_graph.BeliefNode`, calls
``engine.add_node``/``update_node``, or otherwise writes to the KG. The
Critic flags, it does not arbitrate — resolution/write-back is the caller's
job (see ``loop_controller._run_belief_revision``, which routes any
persisted proposal the same way the existing ``TeamSpec``/``SearchTask``
stages do: new proposal-status nodes, never a mutation of the canonical
belief).

SCOPE — this is confidence **propagation**, NOT a full paraconsistent Truth
Maintenance System (TMS). A real TMS keeps justification sets, supports
dependency-directed backtracking, and can hold multiple mutually
inconsistent belief states in superposition (De Kleer's ATMS, Doyle's JTMS).
None of that lives here. What this module gives you is the light-weight
piece the golden loop needs today: a deterministic, single-pass,
explainable confidence update per belief, run each cycle. A full TMS is a
separate engine-side effort (see the module TODO below for the delegation
seam that effort would plug into).

The default formula is **zero-infra**: a Bayesian-ish log-odds update using
only the stdlib (``math.log``/``math.exp``) — no embeddings, model, or
network, exactly like :func:`.contradiction_detector.lexical_similarity`.

Concept: belief-revision
"""

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from agent_utilities.models.knowledge_graph import BeliefNode

from .contradiction_detector import Claim, ContradictionDetector

logger = logging.getLogger(__name__)

__all__ = [
    "BeliefRevision",
    "recompute_confidence",
    "explain_revision",
    "apply_revision",
    "BeliefRevisionPass",
]

# Coarse severity ordering shared with FrictionFinding.severity bands
# ("low" | "medium" | "high") — kept local (contradiction_detector's is
# private) since it is two lines of shared vocabulary, not logic.
_SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2}

# Log-odds gain per unit of the OTHER belief's own confidence. Contradiction
# is weighted higher than support on purpose — a conservative, "extraordinary
# claims" epistemic bias so a single strongly-held contradicting belief moves
# confidence down faster than an equally strong supporting belief moves it up.
# Both are tunable per pass via BeliefRevisionPass(support_gain=, contradict_gain=).
_SUPPORT_GAIN = 0.4
_CONTRADICT_GAIN = 0.55

# Keeps math.log/math.exp finite at the [0, 1] boundary.
_LOGIT_EPS = 1e-6


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _logit(p: float) -> float:
    """Log-odds of a probability, clamped away from the 0/1 singularities."""
    p = min(max(p, _LOGIT_EPS), 1.0 - _LOGIT_EPS)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    """Numerically stable logistic function, inverse of :func:`_logit`."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _contribution_terms(
    supporting_beliefs: Sequence[BeliefNode],
    contradicting_beliefs: Sequence[BeliefNode],
    *,
    support_gain: float,
    contradict_gain: float,
) -> list[dict[str, Any]]:
    """Per-edge log-odds contributions — the shared inner loop for both
    :func:`recompute_confidence` (which only needs the sum) and
    :func:`explain_revision` (which needs the itemized rows). Each
    contributing belief's OWN confidence modulates how strongly it pushes:
    a supporting/contradicting belief the KG itself is unsure about (near
    0.5 confidence) barely moves the needle; one the KG is confident about
    moves it more. Deterministic order: supporting first, then contradicting,
    each in input order.
    """
    terms: list[dict[str, Any]] = []
    for s in supporting_beliefs:
        contribution = support_gain * _clamp01(s.confidence)
        terms.append(
            {
                "node_id": s.id,
                "role": "support",
                "node_confidence": round(_clamp01(s.confidence), 6),
                "log_odds_contribution": round(contribution, 6),
            }
        )
    for c in contradicting_beliefs:
        contribution = contradict_gain * _clamp01(c.confidence)
        terms.append(
            {
                "node_id": c.id,
                "role": "contradict",
                "node_confidence": round(_clamp01(c.confidence), 6),
                "log_odds_contribution": round(-contribution, 6),
            }
        )
    return terms


def recompute_confidence(
    belief: BeliefNode,
    supporting_beliefs: Sequence[BeliefNode],
    contradicting_beliefs: Sequence[BeliefNode],
    *,
    support_gain: float = _SUPPORT_GAIN,
    contradict_gain: float = _CONTRADICT_GAIN,
) -> float:
    """Explainable, deterministic confidence-propagation update, in ``[0, 1]``.

    A simple Bayesian-ish **log-odds** (logit) update: the belief's prior
    confidence is converted to log-odds, each supporting belief adds
    ``support_gain * that_belief's_own_confidence`` to the log-odds, each
    contradicting belief subtracts ``contradict_gain *
    that_belief's_own_confidence``, and the result is squashed back through
    the logistic function. This is the textbook additive log-odds update
    used for combining independent evidence (naive-Bayes-style), simplified
    to treat every support/contradiction edge as one unit of evidence whose
    strength is the OTHER belief's own confidence:

        posterior_logit = logit(prior) + Σ gains(supporting) − Σ gains(contradicting)
        new_confidence  = sigmoid(posterior_logit)

    Monotonic by construction: adding another supporting belief (of any
    positive confidence) can only raise the result; adding another
    contradicting belief can only lower it. The logistic squash keeps the
    result in the open interval ``(0, 1)`` regardless of how much evidence
    piles up (diminishing returns near the bounds, exactly like a real
    Bayesian update) and it is symmetric/deterministic — same inputs, same
    output, every time. See :func:`explain_revision` for the itemized trace
    of exactly which edges drove the shift and by how much.

    NOT a full paraconsistent TMS — see the module docstring. This is one
    deterministic pass, not a justification-maintaining belief revision
    calculus.
    """
    terms = _contribution_terms(
        supporting_beliefs,
        contradicting_beliefs,
        support_gain=support_gain,
        contradict_gain=contradict_gain,
    )
    log_odds = _logit(_clamp01(belief.confidence)) + sum(
        t["log_odds_contribution"] for t in terms
    )
    return _clamp01(_sigmoid(log_odds))


def explain_revision(
    belief: BeliefNode,
    supporting_beliefs: Sequence[BeliefNode],
    contradicting_beliefs: Sequence[BeliefNode],
    new_confidence: float,
    *,
    support_gain: float = _SUPPORT_GAIN,
    contradict_gain: float = _CONTRADICT_GAIN,
) -> list[dict[str, Any]]:
    """Build the reasoning-trace for one :func:`recompute_confidence` call.

    Returns a list of dicts: one row per contributing support/contradiction
    edge (``node_id``, ``role``, that node's own ``node_confidence``, and the
    ``log_odds_contribution`` it made — signed, so contradictions show
    negative), followed by one trailing ``role: "summary"`` row carrying the
    belief id, old confidence, new confidence, and the net delta. This is the
    explainability record the golden-loop stage persists alongside each
    :class:`BeliefRevision` proposal so a reviewer (human or agent) can see
    exactly *why* a confidence moved, not just that it did.
    """
    terms = _contribution_terms(
        supporting_beliefs,
        contradicting_beliefs,
        support_gain=support_gain,
        contradict_gain=contradict_gain,
    )
    trace = list(terms)
    trace.append(
        {
            "node_id": belief.id,
            "role": "summary",
            "old_confidence": round(_clamp01(belief.confidence), 6),
            "new_confidence": round(_clamp01(new_confidence), 6),
            "delta": round(_clamp01(new_confidence) - _clamp01(belief.confidence), 6),
        }
    )
    return trace


@dataclass
class BeliefRevision:
    """One proposed confidence-propagation outcome for a single belief.

    Propose-only: a *record* of what a revision would look like — it never
    mutates anything. Pass it to :func:`apply_revision` to materialize the
    (still in-memory, still unwritten) revised :class:`BeliefNode`, and to
    the caller's own write/governance path to actually persist it.
    """

    belief_id: str
    old_confidence: float
    new_confidence: float
    new_contradicted_by_node_ids: list[str]
    last_reviewed: str
    reasoning_trace: list[dict[str, Any]] = field(default_factory=list)

    @property
    def delta(self) -> float:
        return round(self.new_confidence - self.old_confidence, 6)

    def to_dict(self) -> dict[str, Any]:
        """JSON-able projection for reports / persisted proposal nodes."""
        return {
            "belief_id": self.belief_id,
            "old_confidence": self.old_confidence,
            "new_confidence": self.new_confidence,
            "delta": self.delta,
            "new_contradicted_by_node_ids": list(self.new_contradicted_by_node_ids),
            "last_reviewed": self.last_reviewed,
            "reasoning_trace": list(self.reasoning_trace),
        }


def apply_revision(belief: BeliefNode, revision: BeliefRevision) -> BeliefNode:
    """Return a NEW, revised ``BeliefNode`` — never mutates ``belief`` in place.

    This is a pure projection for callers that want the would-be revised
    node in hand (e.g. to diff, or to hand to a write/governance path); it
    does not touch any engine. The support/contradict mutex
    (``BeliefNode._validate_support_contradict_mutex``) is preserved by
    construction: any id landing in the revision's
    ``new_contradicted_by_node_ids`` is dropped from ``supported_by_node_ids``
    first, so the returned node always passes that validator.
    """
    contradicted = list(dict.fromkeys(revision.new_contradicted_by_node_ids))
    contradicted_set = set(contradicted)
    supported = [
        node_id
        for node_id in belief.supported_by_node_ids
        if node_id not in contradicted_set
    ]
    return belief.model_copy(
        update={
            "confidence": revision.new_confidence,
            "contradicted_by_node_ids": contradicted,
            "supported_by_node_ids": supported,
            "last_reviewed": revision.last_reviewed,
        }
    )


def _invoke_engine_propagate(
    belief: BeliefNode,
    supporting_beliefs: Sequence[BeliefNode],
    contradicting_beliefs: Sequence[BeliefNode],
) -> tuple[float, list[dict[str, Any]]] | None:
    """Delegate confidence recomputation to the engine's propagation surface.

    TODO(C2-engine-delegation): mirrors ``loop_controller._mine_association_
    rules``'s call boundary — the SAME ``_invoke`` helper the ``graph_mine``/
    ``graph_learn`` MCP tools use
    (:func:`agent_utilities.mcp.tools.engine_surface_tools._invoke`) — so this
    degrades exactly like those tools do on an engine build that doesn't ship
    an ``epistemic``/``propagate`` surface yet: ``_invoke`` returns a
    ``degraded``/``error`` JSON payload as data (never raises), and this
    function returns ``None`` so the caller falls back to the local
    :func:`recompute_confidence` + :func:`explain_revision` formula. No such
    engine surface exists today — this is inert scaffolding for when the
    engine-side TMS/propagation effort ships one; the Python formula above is
    the real implementation until then.
    """
    import json as _json

    try:
        from agent_utilities.mcp.tools.engine_surface_tools import _invoke
    except Exception as e:  # noqa: BLE001 — optional import, never fatal
        logger.debug("belief_revision: engine_surface_tools unavailable: %s", e)
        return None

    try:
        raw = _invoke(
            surface="epistemic",
            action="propagate",
            graph="",
            candidates=(("epistemic", "propagate"),),
            params={
                "belief_id": belief.id,
                "prior_confidence": belief.confidence,
                "supporting": [
                    {"id": s.id, "confidence": s.confidence} for s in supporting_beliefs
                ],
                "contradicting": [
                    {"id": c.id, "confidence": c.confidence}
                    for c in contradicting_beliefs
                ],
            },
        )
        payload = _json.loads(raw)
    except Exception as e:  # noqa: BLE001 — delegation is best-effort only
        logger.debug("belief_revision: engine propagate invoke failed: %s", e)
        return None

    if not (isinstance(payload, dict) and "error" not in payload):
        return None
    result = payload.get("result") or {}
    new_confidence = result.get("new_confidence")
    trace = result.get("reasoning_trace")
    if new_confidence is None or not isinstance(trace, list):
        return None
    try:
        return float(new_confidence), trace
    except (TypeError, ValueError):
        return None


class BeliefRevisionPass:
    """Confidence-propagation + light-TMS pass over a ``BeliefNode`` set.

    CONCEPT:AU-KG.maintenance.confidence-propagation-belief-revision

    The night-shift Critic's sibling to :class:`.contradiction_detector.
    ContradictionDetector`: given a belief and its current
    support/contradiction neighborhood, recompute its confidence and explain
    why. Propose-only — like ``ContradictionDetector``, this returns
    :class:`BeliefRevision` records and never mutates a node or writes to an
    engine; resolution/persistence is the caller's job.

    Args:
        contradiction_detector: injected friction detector (defaults to a
            fresh zero-infra :class:`~.contradiction_detector.
            ContradictionDetector`) used by :meth:`scan` to discover NEW
            friction between belief statements.
        severity_threshold: minimum :class:`~.contradiction_detector.
            FrictionFinding` severity (``"low"``/``"medium"``/``"high"``)
            for a fresh friction finding to populate
            ``contradicted_by_node_ids``. Existing recorded contradictions
            are always honored regardless of this threshold.
        support_gain / contradict_gain: passed through to
            :func:`recompute_confidence` / :func:`explain_revision`.
        use_engine: when ``True``, :meth:`check` first tries
            :func:`_invoke_engine_propagate` (the TODO delegation seam) and
            only falls back to the local formula when that degrades. Default
            ``False`` — the local formula is the primary path today.
        now_fn: injectable clock (defaults to ``datetime.now(UTC).
            isoformat()``) so ``last_reviewed`` bumps are deterministic in
            tests.
    """

    def __init__(
        self,
        *,
        contradiction_detector: ContradictionDetector | None = None,
        severity_threshold: str = "medium",
        support_gain: float = _SUPPORT_GAIN,
        contradict_gain: float = _CONTRADICT_GAIN,
        use_engine: bool = False,
        now_fn: Callable[[], str] | None = None,
    ) -> None:
        self.detector = contradiction_detector or ContradictionDetector()
        self.severity_threshold = severity_threshold
        self.support_gain = support_gain
        self.contradict_gain = contradict_gain
        self.use_engine = use_engine
        self._now_fn = now_fn or (lambda: datetime.now(UTC).isoformat())

    def check(
        self,
        belief: BeliefNode,
        supporting_beliefs: Sequence[BeliefNode],
        contradicting_beliefs: Sequence[BeliefNode],
    ) -> BeliefRevision:
        """Recompute one belief's confidence from an explicit support/contradiction set.

        Never mutates ``belief``. Bumps ``last_reviewed`` unconditionally
        (per docs/KG_V2_DESIGN.md §2.2.8, ``last_reviewed`` "bumps on
        evidence update" — a review happened even if confidence didn't move).
        """
        delegated = (
            _invoke_engine_propagate(belief, supporting_beliefs, contradicting_beliefs)
            if self.use_engine
            else None
        )
        if delegated is not None:
            new_confidence, trace = delegated
        else:
            new_confidence = recompute_confidence(
                belief,
                supporting_beliefs,
                contradicting_beliefs,
                support_gain=self.support_gain,
                contradict_gain=self.contradict_gain,
            )
            trace = explain_revision(
                belief,
                supporting_beliefs,
                contradicting_beliefs,
                new_confidence,
                support_gain=self.support_gain,
                contradict_gain=self.contradict_gain,
            )
        contradicted_ids = sorted({c.id for c in contradicting_beliefs})
        return BeliefRevision(
            belief_id=belief.id,
            old_confidence=belief.confidence,
            new_confidence=new_confidence,
            new_contradicted_by_node_ids=contradicted_ids,
            last_reviewed=self._now_fn(),
            reasoning_trace=trace,
        )

    def scan(self, beliefs: Sequence[BeliefNode]) -> list[BeliefRevision]:
        """Revise every belief in ``beliefs`` from friction + recorded edges.

        For each belief: (1) run the detector's all-pairs
        :meth:`~.contradiction_detector.ContradictionDetector.scan` over every
        belief's ``statement`` to discover NEW friction at/above
        ``severity_threshold``; (2) union those fresh conflict ids with the
        belief's already-recorded ``contradicted_by_node_ids`` (restricted to
        ids present in this set — an id this pass can't resolve to an actual
        confidence can't be weighed, so it is left untouched rather than
        guessed at); (3) resolve ``supported_by_node_ids`` (likewise
        restricted to this set) as the support side, with anything also
        flagged as contradicting taking precedence (preserves the mutex);
        (4) recompute via :meth:`check`. Deterministic: sorted by belief id.
        """
        by_id = {b.id: b for b in beliefs}
        claims = [Claim(id=b.id, text=b.statement) for b in beliefs]
        friction = self.detector.scan(claims)
        threshold_rank = _SEVERITY_RANK.get(self.severity_threshold, 1)

        fresh_conflicts: dict[str, set[str]] = {b.id: set() for b in beliefs}
        for finding in friction:
            if _SEVERITY_RANK.get(finding.severity, 0) < threshold_rank:
                continue
            fresh_conflicts.setdefault(finding.new_id, set()).add(finding.conflict_id)
            fresh_conflicts.setdefault(finding.conflict_id, set()).add(finding.new_id)

        revisions: list[BeliefRevision] = []
        for belief in beliefs:
            contradicting_ids = (
                set(belief.contradicted_by_node_ids)
                | fresh_conflicts.get(belief.id, set())
            ) & by_id.keys()
            contradicting_ids.discard(belief.id)

            supporting_ids = {
                i for i in belief.supported_by_node_ids if i in by_id
            } - contradicting_ids
            supporting_ids.discard(belief.id)

            supporting = [by_id[i] for i in sorted(supporting_ids)]
            contradicting = [by_id[i] for i in sorted(contradicting_ids)]
            revisions.append(self.check(belief, supporting, contradicting))

        revisions.sort(key=lambda r: r.belief_id)
        return revisions
