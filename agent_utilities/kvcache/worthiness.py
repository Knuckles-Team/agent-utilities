"""Checkpoint worthiness — is *now* a good moment to checkpoint the KV cache?

CONCEPT:AU-KG.memory.checkpoint-worthiness-scoring — the intelligence half of the
KV-checkpoint layer. :mod:`agent_utilities.kvcache.checkpoint` already answers *how* to
store a checkpoint durably and safely; this module answers **when one is worth taking**,
and :mod:`agent_utilities.kvcache.tiering` acts on the answer.

A framework, not a heuristic
----------------------------
The deliverable is the **scoring framework plus a default signal set**, never one
fixed built-in rule. Every signal is a :class:`CheckpointScorer` — an independent object with
a name, a weight, and one method — registered in a :class:`CheckpointScorerRegistry` an
operator can add to, remove from, or replace wholesale without touching the advisor, the
tiering layer, or the MCP surface. Adding a deployment-specific signal is a `register()`
call; deleting a default is an `unregister()` call.

Abstention is a first-class answer
----------------------------------
Every scorer may return :meth:`CheckpointScorer.abstain`, and several defaults do so
routinely because this platform genuinely cannot compute their input cheaply today
(predicted reuse, notably — see :class:`PredictedReuseScorer`). An abstaining scorer
contributes **nothing**: it does not dilute the aggregate toward zero, and it is listed
by name on the recommendation so a reader can see what evidence was missing. A scorer
that guesses is worse than one that abstains.

Uniform polarity
----------------
Every scorer emits a value in ``[0, 1]`` where **1 means "more checkpoint-worthy"**,
including the ones that are conceptually negative: :class:`ContradictionScorer` returns
1.0 for a clean context and 0.0 for a contradicted one. That keeps aggregation a plain
weighted mean while :attr:`CheckpointSignal.kind` still records whether a signal is an
opportunity, a risk, or a hard gate, so the rendered rationale reads correctly.

A signal may additionally :attr:`CheckpointSignal.veto`, which forces
:attr:`CheckpointTier.NONE` regardless of the aggregate — used for the state where the
context is demonstrably *not* good (unresolved high-severity contradictions), which no
amount of expensive-to-rebuild-ness should override.

The model's own opinion is evidence, not authority
---------------------------------------------------
:class:`ModelSelfReportScorer` lets an LLM flag "I have a strong working understanding of
X". It carries the smallest default weight of any scorer, and the advisor additionally
refuses to recommend a checkpoint when the self-report is the *only* non-abstaining
signal (:attr:`CheckpointRecommendation.blockers` records why). A model claiming
confidence is evidence; it is not proof.
"""

from __future__ import annotations

import contextvars
import logging
import threading
from enum import StrEnum
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from agent_utilities.kvcache.rebuild_cost import (
    RebuildCostInputs,
    RebuildCostScale,
    estimate_rebuild_cost,
)

logger = logging.getLogger(__name__)

__all__ = [
    "CheckpointAdvisor",
    "CheckpointObservation",
    "CheckpointRecommendation",
    "CheckpointScorer",
    "CheckpointScorerRegistry",
    "CheckpointSignal",
    "CheckpointTier",
    "ContextStabilityScorer",
    "ContradictionScorer",
    "DiskPromotionRule",
    "DiskPromotionVerdict",
    "GroundingDensityScorer",
    "ModelSelfReport",
    "ModelSelfReportScorer",
    "PhaseBoundaryScorer",
    "PredictedReuseScorer",
    "RebuildCostScorer",
    "RetrievalSaturationScorer",
    "build_default_scorers",
    "clear_checkpoint_advisory",
    "current_checkpoint_advisory",
    "default_scorer_registry",
    "publish_checkpoint_advisory",
    "render_checkpoint_advisory_instructions",
]


class CheckpointTier(StrEnum):
    """Where a checkpoint should live. Ordered by increasing commitment."""

    #: Not worth checkpointing at all.
    NONE = "none"
    #: Keep it in RAM for the rest of the session. The default for anything worth
    #: keeping — cheap, reversible, and it makes no data-at-rest commitment.
    RAM = "ram"
    #: Persist durably so it survives the session. Requires a materially higher bar
    #: AND an eligibility grant (see :mod:`agent_utilities.kvcache.eligibility`).
    DISK = "disk"


#: How a signal should be read in the rendered rationale. Purely descriptive — it does
#: not change how the value aggregates (see the module docstring on uniform polarity).
SignalKind = Literal["opportunity", "risk", "gate"]


class ModelSelfReport(BaseModel):
    """An LLM's own claim about the state of its understanding.

    Deliberately structured rather than free text so it can be weighed rather than
    merely believed: a ``subject`` the model claims to understand, its stated
    ``confidence``, and its own reasoning. Recorded verbatim on the recommendation, so
    an operator reviewing a checkpoint can see exactly what the model asserted.
    """

    subject: str = Field(
        default="", description="What the model claims to have a grip on."
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="The model's stated confidence. None when it declined to quantify "
        "— which is not the same as low confidence.",
    )
    rationale: str = Field(
        default="", description="The model's own justification, recorded verbatim."
    )

    model_config = ConfigDict(extra="forbid", frozen=True)


class CheckpointObservation(BaseModel):
    """Everything the scorers may look at, as measured by the caller.

    Every measurable field is ``None`` by default, meaning **not measured** — never
    "zero". Scorers abstain on ``None`` and score on ``0``. Populate whatever the call
    site cheaply knows and leave the rest alone; the aggregate is computed over what
    was actually measured.

    :attr:`extras` is the pluggability escape hatch: a deployment-specific scorer reads
    its input from there, so adding a signal never requires editing this model.
    """

    # -- identity / provenance ------------------------------------------------
    run_id: str = Field(default="", description="Run this observation belongs to.")
    tenant: str = Field(default="", description="Owning tenant.")
    point: str = Field(default="", description="Label for the moment being considered.")

    # -- rebuild cost (the economic signal) -----------------------------------
    rebuild: RebuildCostInputs = Field(
        default_factory=RebuildCostInputs,
        description="What was spent assembling this context. See "
        ":mod:`agent_utilities.kvcache.rebuild_cost`.",
    )

    # -- predicted reuse ------------------------------------------------------
    sibling_task_count: int | None = Field(
        default=None,
        ge=0,
        description="Related tasks that would hit this same context.",
    )
    queued_task_count: int | None = Field(
        default=None, ge=0, description="Queued tasks that would hit this context."
    )

    # -- retrieval saturation / novelty decay ---------------------------------
    retrieved_items: int | None = Field(
        default=None, ge=0, description="Items returned by the recent retrievals."
    )
    novel_items: int | None = Field(
        default=None,
        ge=0,
        description="Of those, how many were NOT already present in the context. "
        "Falling novelty means understanding has converged; rising novelty means the "
        "run is still exploring, which is a bad moment to freeze.",
    )

    # -- grounding density ----------------------------------------------------
    claim_count: int | None = Field(
        default=None, ge=0, description="Atomic claims the current answer rests on."
    )
    evidence_span_count: int | None = Field(
        default=None, ge=0, description="Cited evidence spans backing those claims."
    )

    # -- contradictions (the strong negative gate) ----------------------------
    unresolved_contradictions: int | None = Field(
        default=None, ge=0, description="Open contradictions in the current context."
    )
    high_severity_contradictions: int | None = Field(
        default=None,
        ge=0,
        description="Of those, how many are high severity. Any of these vetoes a "
        "checkpoint outright — a contradicted context is not in a good state.",
    )

    # -- churn / stability ----------------------------------------------------
    context_rewrites: int | None = Field(
        default=None,
        ge=0,
        description="Times the context was rewritten/compacted in the recent window.",
    )
    evicted_items: int | None = Field(
        default=None, ge=0, description="Items evicted from context recently."
    )
    turns_since_context_change: int | None = Field(
        default=None,
        ge=0,
        description="Turns since the last rewrite/eviction. Higher means settled.",
    )

    # -- phase boundary -------------------------------------------------------
    phase: str = Field(
        default="",
        description="The phase just finished (e.g. 'understand', 'plan', 'execute'). "
        "Empty means the caller tracks no phase and the scorer abstains.",
    )
    phase_completed: bool = Field(
        default=False, description="True iff `phase` actually completed."
    )

    # -- the model's own flag -------------------------------------------------
    model_self_report: ModelSelfReport | None = Field(
        default=None, description="The LLM's own claim, when it made one."
    )

    # -- pluggability ---------------------------------------------------------
    extras: dict[str, Any] = Field(
        default_factory=dict,
        description="Free-form inputs for deployment-specific scorers, so a new signal "
        "never requires changing this model.",
    )

    model_config = ConfigDict(extra="forbid")

    @staticmethod
    def _read(bundle: Any, name: str) -> Any:
        """Read ``name`` off a bundle that may be an object OR a plain mapping.

        The adapters below are reached two ways — in-process with a real
        ``EvidenceBundle``/``ContextBundle``, and over the MCP surface with the same
        thing already decoded from JSON into a dict. Supporting both here means the
        adapters have exactly one implementation instead of an object version and a
        near-identical dict version that could drift.
        """
        if isinstance(bundle, dict):
            return bundle.get(name)
        return getattr(bundle, name, None)

    @classmethod
    def from_evidence_bundle(
        cls, bundle: Any, /, **overrides: Any
    ) -> CheckpointObservation:
        """Populate the grounding + contradiction axes from an
        :class:`~agent_utilities.models.evidence_bundle.EvidenceBundle`.

        Duck-typed on purpose (never imported) so this module stays importable without
        pulling the evidence/model layer onto the KV-cache import path, and accepts a
        decoded JSON mapping as readily as the object. Anything the bundle does not
        carry is left unmeasured.
        """
        claims = list(cls._read(bundle, "claims") or [])
        spans = list(cls._read(bundle, "evidence_spans") or [])
        contradictions = list(cls._read(bundle, "contradictions") or [])
        high = sum(
            1
            for c in contradictions
            if str(cls._read(c, "severity") or "").lower() == "high"
        )
        fields: dict[str, Any] = {
            "claim_count": len(claims),
            "evidence_span_count": len(spans),
            "unresolved_contradictions": len(contradictions),
            "high_severity_contradictions": high,
        }
        fields.update(overrides)
        return cls(**fields)

    @classmethod
    def from_context_bundle(
        cls, bundle: Any, /, **overrides: Any
    ) -> CheckpointObservation:
        """Populate the retrieval-saturation axis from a
        :class:`~agent_utilities.knowledge_graph.retrieval.context_compiler.ContextBundle`.

        ``ContextBundle.dropped_redundant`` is the compiler's own count of candidates
        discarded as already-covered, so ``novel = selected`` and
        ``retrieved = selected + dropped_redundant`` is a real (not inferred) novelty
        ratio. Duck-typed for the same import-hygiene reason as
        :meth:`from_evidence_bundle`.
        """
        items = list(cls._read(bundle, "items") or [])
        dropped_redundant = cls._read(bundle, "dropped_redundant")
        fields: dict[str, Any] = {}
        if dropped_redundant is not None:
            fields["novel_items"] = len(items)
            fields["retrieved_items"] = len(items) + int(dropped_redundant)
        citations = cls._read(bundle, "citations")
        if citations is not None:
            fields["claim_count"] = len(items)
            fields["evidence_span_count"] = len(list(citations))
        fields.update(overrides)
        return cls(**fields)


class CheckpointSignal(BaseModel):
    """One scorer's contribution — a value, or an explicit abstention."""

    name: str
    value: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Checkpoint-worthiness in [0, 1], 1 = most worthy. None = the "
        "scorer abstained (no evidence), and contributes nothing to the aggregate.",
    )
    weight: float = Field(default=1.0, ge=0.0)
    kind: SignalKind = Field(
        default="opportunity",
        description="How to read this signal in the rationale: an 'opportunity' argues "
        "for checkpointing, a 'risk' argues against, a 'gate' can veto.",
    )
    veto: bool = Field(
        default=False,
        description="True forces CheckpointTier.NONE regardless of the aggregate.",
    )
    rationale: str = Field(default="", description="Why this scorer said what it said.")
    detail: dict[str, Any] = Field(
        default_factory=dict, description="The raw numbers behind the value."
    )

    model_config = ConfigDict(extra="forbid", frozen=True)

    @property
    def abstained(self) -> bool:
        return self.value is None


@runtime_checkable
class CheckpointScorer(Protocol):
    """The pluggable signal contract.

    Implementations MUST be cheap: no LLM call, no network round trip, no embedding.
    This runs on the decision path for every candidate moment. A signal that needs an
    expensive computation should be measured upstream and passed in through
    :attr:`CheckpointObservation.extras`.
    """

    #: Stable identifier, unique within a registry.
    name: str
    #: Relative influence on the aggregate. Only compared against other
    #: non-abstaining scorers, so removing a scorer never rescales the rest.
    weight: float

    def score(self, observation: CheckpointObservation) -> CheckpointSignal:
        """Return this signal for ``observation``. MUST NOT raise for a well-formed
        observation; return an abstention instead."""
        ...  # pragma: no cover - protocol


class _BaseScorer:
    """Shared plumbing for the default scorers — name, weight, and abstain/emit."""

    name: str = "scorer"
    kind: SignalKind = "opportunity"

    def __init__(self, *, weight: float) -> None:
        if weight < 0.0:
            raise ValueError(f"{self.name}: weight must be >= 0, got {weight}")
        self.weight = weight

    def _abstain(self, reason: str) -> CheckpointSignal:
        return CheckpointSignal(
            name=self.name,
            value=None,
            weight=self.weight,
            kind=self.kind,
            rationale=f"abstained: {reason}",
        )

    def _emit(
        self,
        value: float,
        rationale: str,
        detail: dict[str, Any] | None = None,
        *,
        veto: bool = False,
    ) -> CheckpointSignal:
        return CheckpointSignal(
            name=self.name,
            value=round(min(1.0, max(0.0, value)), 4),
            weight=self.weight,
            kind=self.kind,
            veto=veto,
            rationale=rationale,
            detail=detail or {},
        )


class RebuildCostScorer(_BaseScorer):
    """Expensive-to-rebuild context is worth keeping — the strongest economic signal.

    Delegates entirely to :func:`~agent_utilities.kvcache.rebuild_cost.estimate_rebuild_cost`
    so this and the engine-side pressure-aware eviction consumer share one definition of
    "expensive" rather than drifting apart.
    """

    name = "rebuild_cost"
    kind = "opportunity"

    def __init__(
        self, *, weight: float = 0.30, scale: RebuildCostScale | None = None
    ) -> None:
        super().__init__(weight=weight)
        self.scale = scale

    def score(self, observation: CheckpointObservation) -> CheckpointSignal:
        estimate = estimate_rebuild_cost(observation.rebuild, scale=self.scale)
        if not estimate.known:
            return self._abstain(
                "no rebuild-cost component was measured (tokens/tool calls/retrievals/"
                "wall time all absent)"
            )
        return self._emit(
            estimate.normalized,
            estimate.summary(),
            {
                "components": estimate.components,
                "measured": list(estimate.measured),
                "usd": estimate.usd,
            },
        )


class PredictedReuseScorer(_BaseScorer):
    """How many other tasks would hit this same context.

    **Honest limitation:** this platform has no reuse-probability model today — the task
    graph carries dependencies (``TASK_DEPENDS_ON``) but nothing predicts context
    overlap between siblings, and deriving it would need work this scorer cannot do
    cheaply on the decision path. So the scorer consumes counts the *caller* supplies
    from whatever task view it has, and **abstains** when neither is supplied rather
    than inventing a reuse estimate. When a real predictor lands, it populates these
    fields and nothing here changes.
    """

    name = "predicted_reuse"
    kind = "opportunity"

    def __init__(self, *, weight: float = 0.15, saturating_tasks: int = 6) -> None:
        super().__init__(weight=weight)
        if saturating_tasks <= 0:
            raise ValueError("saturating_tasks must be > 0")
        self.saturating_tasks = saturating_tasks

    def score(self, observation: CheckpointObservation) -> CheckpointSignal:
        siblings = observation.sibling_task_count
        queued = observation.queued_task_count
        if siblings is None and queued is None:
            return self._abstain(
                "no sibling/queued task counts supplied — this platform has no "
                "predicted-reuse model to fall back on"
            )
        total = (siblings or 0) + (queued or 0)
        return self._emit(
            min(1.0, total / self.saturating_tasks),
            f"{total} related task(s) would reuse this context",
            {"sibling_task_count": siblings, "queued_task_count": queued},
        )


class RetrievalSaturationScorer(_BaseScorer):
    """Novelty decay: retrievals returning already-present material mean convergence.

    High saturation (little new material coming back) is a *good* moment — understanding
    has stabilized. Rising novelty means the run is still exploring, which is a bad
    moment to freeze a context that is about to change.
    """

    name = "retrieval_saturation"
    kind = "opportunity"

    def __init__(self, *, weight: float = 0.15) -> None:
        super().__init__(weight=weight)

    def score(self, observation: CheckpointObservation) -> CheckpointSignal:
        retrieved = observation.retrieved_items
        novel = observation.novel_items
        if retrieved is None or novel is None:
            return self._abstain("retrieved/novel item counts not measured")
        if retrieved == 0:
            return self._abstain(
                "no retrievals in the window — nothing to infer convergence from"
            )
        novelty_ratio = min(1.0, novel / retrieved)
        saturation = 1.0 - novelty_ratio
        return self._emit(
            saturation,
            f"retrieval saturation {saturation:.2f} "
            f"({novel}/{retrieved} items were new)",
            {
                "retrieved_items": retrieved,
                "novel_items": novel,
                "novelty_ratio": round(novelty_ratio, 4),
            },
        )


class GroundingDensityScorer(_BaseScorer):
    """Proportion of the context that is cited evidence rather than speculation.

    Measured as evidence spans per claim, saturating at
    :attr:`spans_per_claim_target` — one well-cited claim is grounded; a claim with six
    citations is not six times better.
    """

    name = "grounding_density"
    kind = "opportunity"

    def __init__(
        self, *, weight: float = 0.15, spans_per_claim_target: float = 1.0
    ) -> None:
        super().__init__(weight=weight)
        if spans_per_claim_target <= 0:
            raise ValueError("spans_per_claim_target must be > 0")
        self.spans_per_claim_target = spans_per_claim_target

    def score(self, observation: CheckpointObservation) -> CheckpointSignal:
        claims = observation.claim_count
        spans = observation.evidence_span_count
        if claims is None or spans is None:
            return self._abstain("claim/evidence-span counts not measured")
        if claims == 0:
            return self._abstain("no claims in the context — nothing to ground")
        density = min(1.0, (spans / claims) / self.spans_per_claim_target)
        return self._emit(
            density,
            f"grounding density {density:.2f} ({spans} evidence span(s) "
            f"for {claims} claim(s))",
            {"claim_count": claims, "evidence_span_count": spans},
        )


class ContradictionScorer(_BaseScorer):
    """Unresolved contradictions mean the context is NOT in a good state.

    The strongest negative in the default set, and the only one that vetoes: any
    high-severity unresolved contradiction forces :attr:`CheckpointTier.NONE` outright,
    because freezing a self-contradictory context and restoring it later propagates the
    contradiction into every run seeded from it.
    """

    name = "contradictions"
    kind = "gate"

    def __init__(self, *, weight: float = 0.15, saturating_count: int = 4) -> None:
        super().__init__(weight=weight)
        if saturating_count <= 0:
            raise ValueError("saturating_count must be > 0")
        self.saturating_count = saturating_count

    def score(self, observation: CheckpointObservation) -> CheckpointSignal:
        unresolved = observation.unresolved_contradictions
        high = observation.high_severity_contradictions
        if unresolved is None and high is None:
            return self._abstain("contradiction counts not measured")
        high_count = high or 0
        unresolved_count = unresolved if unresolved is not None else high_count
        if high_count > 0:
            return self._emit(
                0.0,
                f"VETO: {high_count} unresolved high-severity contradiction(s) — the "
                "context is not in a good state to freeze",
                {"unresolved": unresolved_count, "high_severity": high_count},
                veto=True,
            )
        cleanliness = 1.0 - min(1.0, unresolved_count / self.saturating_count)
        return self._emit(
            cleanliness,
            (
                "no unresolved contradictions"
                if unresolved_count == 0
                else f"{unresolved_count} unresolved contradiction(s)"
            ),
            {"unresolved": unresolved_count, "high_severity": high_count},
        )


class ContextStabilityScorer(_BaseScorer):
    """Recently rewritten or heavily evicted context is not stable — don't freeze it.

    Emits 1.0 for a context that has been settled for a while and decays toward 0 as
    rewrites/evictions pile up in the recent window.
    """

    name = "context_stability"
    kind = "risk"

    def __init__(
        self,
        *,
        weight: float = 0.10,
        saturating_churn: int = 5,
        settled_turns: int = 3,
    ) -> None:
        super().__init__(weight=weight)
        if saturating_churn <= 0:
            raise ValueError("saturating_churn must be > 0")
        if settled_turns <= 0:
            raise ValueError("settled_turns must be > 0")
        self.saturating_churn = saturating_churn
        self.settled_turns = settled_turns

    def score(self, observation: CheckpointObservation) -> CheckpointSignal:
        rewrites = observation.context_rewrites
        evicted = observation.evicted_items
        settled = observation.turns_since_context_change
        if rewrites is None and evicted is None and settled is None:
            return self._abstain("no churn/stability measurement supplied")
        churn = (rewrites or 0) + (evicted or 0)
        stability = 1.0 - min(1.0, churn / self.saturating_churn)
        if settled is not None:
            # A context that has not changed for a while is stable regardless of how
            # much churn preceded that quiet period — take the more favourable of the
            # two readings, since both describe the SAME window.
            stability = max(stability, min(1.0, settled / self.settled_turns))
        return self._emit(
            stability,
            f"context stability {stability:.2f} "
            f"(churn={churn}, settled_turns={settled if settled is not None else 'n/a'})",
            {
                "context_rewrites": rewrites,
                "evicted_items": evicted,
                "turns_since_context_change": settled,
            },
        )


class PhaseBoundaryScorer(_BaseScorer):
    """A completed understand/plan phase before execution is a natural seam.

    The end of a phase is the moment the context is most "finished" and least likely to
    churn — and the moment a sibling run branching off it would want to start from.
    """

    name = "phase_boundary"
    kind = "opportunity"

    #: Phases whose completion is a strong seam. A deployment with different phase
    #: names constructs the scorer with its own set.
    DEFAULT_SEAM_PHASES: frozenset[str] = frozenset(
        {"understand", "understanding", "research", "plan", "planning", "design"}
    )

    def __init__(
        self, *, weight: float = 0.10, seam_phases: frozenset[str] | None = None
    ) -> None:
        super().__init__(weight=weight)
        self.seam_phases = seam_phases or self.DEFAULT_SEAM_PHASES

    def score(self, observation: CheckpointObservation) -> CheckpointSignal:
        phase = (observation.phase or "").strip().lower()
        if not phase:
            return self._abstain("caller tracks no phase")
        if not observation.phase_completed:
            return self._emit(
                0.0,
                f"mid-phase ({phase!r}) — not a boundary",
                {"phase": phase, "completed": False},
            )
        is_seam = phase in self.seam_phases
        return self._emit(
            1.0 if is_seam else 0.5,
            f"completed phase {phase!r}"
            + (" — a natural checkpoint seam" if is_seam else ""),
            {"phase": phase, "completed": True, "seam": is_seam},
        )


class ModelSelfReportScorer(_BaseScorer):
    """The model's own "I have a strong working understanding of X" flag.

    Carries the smallest default weight in the set, and the advisor separately refuses
    to act on it alone (see :meth:`CheckpointAdvisor.evaluate`). A model with no stated
    confidence abstains on the number but still counts as having flagged: the flag
    itself is treated as a moderate signal rather than being silently upgraded to
    certainty.
    """

    name = "model_self_report"
    kind = "opportunity"

    #: Value used when the model flagged but declined to quantify its confidence. A
    #: deliberate middle reading — "it said something" is not "it said 1.0".
    UNQUANTIFIED_CONFIDENCE = 0.6

    def __init__(self, *, weight: float = 0.05) -> None:
        super().__init__(weight=weight)

    def score(self, observation: CheckpointObservation) -> CheckpointSignal:
        report = observation.model_self_report
        if report is None:
            return self._abstain("the model made no self-report")
        confidence = report.confidence
        value = self.UNQUANTIFIED_CONFIDENCE if confidence is None else confidence
        return self._emit(
            value,
            f"model self-reported understanding of {report.subject!r}"
            + (
                " (no confidence stated)"
                if confidence is None
                else f" at confidence {confidence:.2f}"
            ),
            {
                "subject": report.subject,
                "stated_confidence": confidence,
                "rationale": report.rationale,
            },
        )


def build_default_scorers() -> list[CheckpointScorer]:
    """The default signal set, freshly constructed.

    Returned as a list so a caller can filter/reorder/re-weight before registering
    without mutating anyone else's registry.
    """
    return [
        RebuildCostScorer(),
        PredictedReuseScorer(),
        RetrievalSaturationScorer(),
        GroundingDensityScorer(),
        ContradictionScorer(),
        ContextStabilityScorer(),
        PhaseBoundaryScorer(),
        ModelSelfReportScorer(),
    ]


class CheckpointScorerRegistry:
    """A mutable, thread-safe set of scorers keyed by name.

    This is the pluggability surface: ``register`` a deployment's own signal,
    ``unregister`` a default that does not apply, ``replace`` one whose weighting is
    wrong for the workload. Nothing downstream — the advisor, the tiering layer, the MCP
    actions — needs to know which scorers are present.
    """

    def __init__(self, scorers: list[CheckpointScorer] | None = None) -> None:
        self._lock = threading.Lock()
        self._scorers: dict[str, CheckpointScorer] = {}
        for scorer in scorers or []:
            self.register(scorer)

    def register(self, scorer: CheckpointScorer, *, replace: bool = False) -> None:
        """Add ``scorer``. Raises on a duplicate name unless ``replace=True``."""
        if not isinstance(scorer, CheckpointScorer):
            raise TypeError(
                "a checkpoint scorer must expose name/weight/score(observation); got "
                f"{type(scorer).__name__}"
            )
        name = scorer.name
        if not name:
            raise ValueError("a checkpoint scorer must have a non-empty name")
        with self._lock:
            if name in self._scorers and not replace:
                raise ValueError(
                    f"a scorer named {name!r} is already registered; pass replace=True "
                    "to override it"
                )
            self._scorers[name] = scorer

    def unregister(self, name: str) -> CheckpointScorer | None:
        """Remove the scorer called ``name``; returns it, or ``None`` if absent."""
        with self._lock:
            return self._scorers.pop(name, None)

    def scorers(self) -> tuple[CheckpointScorer, ...]:
        """A stable snapshot of the registered scorers, in registration order."""
        with self._lock:
            return tuple(self._scorers.values())

    def reset_to_defaults(self) -> None:
        """Restore exactly :func:`build_default_scorers` — mainly for tests."""
        with self._lock:
            self._scorers = {s.name: s for s in build_default_scorers()}

    def __contains__(self, name: object) -> bool:
        with self._lock:
            return name in self._scorers

    def __len__(self) -> int:
        with self._lock:
            return len(self._scorers)


_DEFAULT_REGISTRY = CheckpointScorerRegistry(build_default_scorers())


def default_scorer_registry() -> CheckpointScorerRegistry:
    """The process-wide registry the advisor uses when none is supplied."""
    return _DEFAULT_REGISTRY


class DiskPromotionRule(BaseModel):
    """The materially higher bar durable persistence must clear.

    Encodes the operator's rule literally: high rebuild cost **AND** high predicted
    reuse **AND** stability — plus an aggregate floor. Each required signal must have
    actually produced a value: a scorer that abstained cannot satisfy a requirement,
    because "we don't know" is not "it's high". Every field is inspectable and the
    verdict names precisely which requirement failed.
    """

    min_aggregate: float = Field(default=0.80, ge=0.0, le=1.0)
    required_signals: dict[str, float] = Field(
        default_factory=lambda: {
            "rebuild_cost": 0.70,
            "predicted_reuse": 0.50,
            "context_stability": 0.70,
        },
        description="Signal name -> the minimum value it must reach. Every one must be "
        "present AND meet its minimum; an abstention fails the requirement.",
    )

    model_config = ConfigDict(extra="forbid", frozen=True)

    def evaluate(
        self, aggregate: float, signals: dict[str, CheckpointSignal]
    ) -> DiskPromotionVerdict:
        """Check ``aggregate`` + ``signals`` against this rule."""
        failures: list[str] = []
        if aggregate < self.min_aggregate:
            failures.append(
                f"aggregate {aggregate:.2f} < required {self.min_aggregate:.2f}"
            )
        for name, minimum in self.required_signals.items():
            signal = signals.get(name)
            if signal is None:
                failures.append(f"{name}: no such scorer registered")
            elif signal.abstained:
                failures.append(
                    f"{name}: abstained ({signal.rationale}) — durable persistence "
                    "needs positive evidence, not absent evidence"
                )
            elif (signal.value or 0.0) < minimum:
                failures.append(f"{name}: {signal.value:.2f} < required {minimum:.2f}")
        return DiskPromotionVerdict(
            satisfied=not failures, failures=tuple(failures), rule=self
        )


class DiskPromotionVerdict(BaseModel):
    """Whether the worthiness half of disk promotion is satisfied, and why not.

    Note this is only the *worthiness* half. Even a satisfied verdict does not persist
    anything: :mod:`agent_utilities.kvcache.eligibility` still has to permit it.
    """

    satisfied: bool
    failures: tuple[str, ...] = ()
    rule: DiskPromotionRule

    model_config = ConfigDict(extra="forbid", frozen=True)


class CheckpointRecommendation(BaseModel):
    """The advisor's verdict — a score, a tier, and a fully inspectable rationale."""

    score: float = Field(ge=0.0, le=1.0)
    recommended_tier: CheckpointTier
    signals: list[CheckpointSignal] = Field(default_factory=list)
    drivers: list[str] = Field(
        default_factory=list,
        description="The signals that most argued FOR checkpointing, strongest first.",
    )
    blockers: list[str] = Field(
        default_factory=list,
        description="The reasons a higher tier was not recommended, including vetoes.",
    )
    abstained: list[str] = Field(
        default_factory=list, description="Scorers that had no evidence to offer."
    )
    rationale: str = Field(default="")
    ram_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    disk_verdict: DiskPromotionVerdict | None = None
    observation: CheckpointObservation | None = Field(
        default=None,
        description="The observation this verdict was computed from, retained so a "
        "recommendation is reproducible from its own record.",
    )

    model_config = ConfigDict(extra="forbid")

    def as_advisory(self) -> str:
        """Render the recommendation as the structured line surfaced to an LLM.

        CONCEPT:AU-ORCH.optimization.checkpoint-recommendation-surface. Deliberately
        compact and machine-parseable-ish, and it always states the drivers AND the
        blockers — a model told only the score cannot judge whether to trust it.
        """
        head = (
            f"checkpoint-worthy: score {self.score:.2f} "
            f"(recommended tier: {self.recommended_tier.value})"
        )
        lines = [head]
        if self.drivers:
            lines.append("  drivers: " + "; ".join(self.drivers))
        if self.blockers:
            lines.append("  blockers: " + "; ".join(self.blockers))
        if self.abstained:
            lines.append("  no evidence from: " + ", ".join(self.abstained))
        return "\n".join(lines)


class CheckpointAdvisor:
    """Aggregates the registered scorers into a :class:`CheckpointRecommendation`.

    CONCEPT:AU-KG.memory.checkpoint-worthiness-scoring. Construct-then-evaluate is cheap
    (no I/O, no model), so a fresh advisor per decision is fine.
    """

    #: Aggregate at/above which a RAM checkpoint is recommended. RAM is the cheap,
    #: reversible tier, so the bar is deliberately moderate.
    DEFAULT_RAM_THRESHOLD = 0.55

    #: How many drivers/blockers to surface in the rendered rationale.
    DEFAULT_TOP_N = 3

    def __init__(
        self,
        *,
        registry: CheckpointScorerRegistry | None = None,
        ram_threshold: float | None = None,
        disk_rule: DiskPromotionRule | None = None,
        top_n: int | None = None,
    ) -> None:
        # ``is None``, never truthiness: a registry an operator has emptied on purpose
        # is falsy (it defines __len__), and silently swapping it for the default set
        # would resurrect exactly the scorers they removed.
        self.registry = default_scorer_registry() if registry is None else registry
        self.ram_threshold = (
            self.DEFAULT_RAM_THRESHOLD if ram_threshold is None else ram_threshold
        )
        self.disk_rule = disk_rule or DiskPromotionRule()
        self.top_n = self.DEFAULT_TOP_N if top_n is None else top_n

    def evaluate(self, observation: CheckpointObservation) -> CheckpointRecommendation:
        """Score ``observation`` against every registered scorer."""
        signals = [
            self._score_one(scorer, observation) for scorer in self.registry.scorers()
        ]
        by_name = {s.name: s for s in signals}
        scored = [s for s in signals if not s.abstained and s.weight > 0.0]
        abstained = [s.name for s in signals if s.abstained]

        weight_total = sum(s.weight for s in scored)
        aggregate = (
            sum((s.value or 0.0) * s.weight for s in scored) / weight_total
            if weight_total > 0.0
            else 0.0
        )
        aggregate = round(min(1.0, max(0.0, aggregate)), 4)

        blockers: list[str] = []
        vetoed = [s for s in signals if s.veto]
        for signal in vetoed:
            blockers.append(f"{signal.name}: {signal.rationale}")

        # A model's own flag is evidence, never proof: if it is the ONLY thing arguing
        # for a checkpoint, there is no independent corroboration and we decline.
        self_report_only = bool(scored) and all(
            s.name == ModelSelfReportScorer.name for s in scored
        )
        if self_report_only:
            blockers.append(
                "model_self_report was the only non-abstaining signal — a model "
                "claiming understanding is evidence, not proof; no independent signal "
                "corroborated it"
            )

        drivers = [
            f"{s.name}={s.value:.2f} ({s.rationale})"
            for s in sorted(
                (s for s in scored if (s.value or 0.0) > 0.0),
                key=lambda s: (s.value or 0.0) * s.weight,
                reverse=True,
            )[: self.top_n]
        ]

        disk_verdict = self.disk_rule.evaluate(aggregate, by_name)

        if vetoed or self_report_only or not scored:
            tier = CheckpointTier.NONE
            if not scored and not vetoed:
                blockers.append(
                    "every scorer abstained — no evidence either way, so no "
                    "checkpoint is recommended"
                )
        elif disk_verdict.satisfied:
            tier = CheckpointTier.DISK
        elif aggregate >= self.ram_threshold:
            tier = CheckpointTier.RAM
            blockers.extend(
                f"disk not recommended — {f}" for f in disk_verdict.failures
            )
        else:
            tier = CheckpointTier.NONE
            blockers.append(
                f"aggregate {aggregate:.2f} < RAM threshold {self.ram_threshold:.2f}"
            )

        rationale = self._render_rationale(tier, aggregate, drivers, blockers)
        recommendation = CheckpointRecommendation(
            score=aggregate,
            recommended_tier=tier,
            signals=signals,
            drivers=drivers,
            blockers=blockers,
            abstained=abstained,
            rationale=rationale,
            ram_threshold=self.ram_threshold,
            disk_verdict=disk_verdict,
            observation=observation,
        )
        _record_recommendation(tier.value)
        return recommendation

    def _score_one(
        self, scorer: CheckpointScorer, observation: CheckpointObservation
    ) -> CheckpointSignal:
        """Run one scorer, converting a misbehaving one into a loud abstention.

        A third-party scorer that raises must not take down the decision path, but the
        failure is never silent: it is logged at WARNING with the cause preserved and
        surfaced as an abstention naming the exception on the recommendation itself.
        """
        try:
            return scorer.score(observation)
        except Exception as exc:  # noqa: BLE001 — a plugin must not break the path
            logger.warning(
                "[CONCEPT:AU-KG.memory.checkpoint-worthiness-scoring] scorer %r raised "
                "%s: %s — treating it as an abstention",
                getattr(scorer, "name", type(scorer).__name__),
                type(exc).__name__,
                exc,
                exc_info=exc,
            )
            return CheckpointSignal(
                name=getattr(scorer, "name", type(scorer).__name__),
                value=None,
                weight=getattr(scorer, "weight", 0.0),
                rationale=f"abstained: scorer raised {type(exc).__name__}: {exc}",
            )

    @staticmethod
    def _render_rationale(
        tier: CheckpointTier,
        aggregate: float,
        drivers: list[str],
        blockers: list[str],
    ) -> str:
        parts = [f"tier={tier.value} score={aggregate:.2f}"]
        if drivers:
            parts.append("driven by " + "; ".join(drivers))
        if blockers:
            parts.append("held back by " + "; ".join(blockers))
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# The recommendation surface — how the advisory reaches the model
# ---------------------------------------------------------------------------
# CONCEPT:AU-ORCH.optimization.checkpoint-recommendation-surface. The operator's
# request is that the system *recommend*, and the model decide. So whenever anything
# scores a moment, the resulting verdict is published into a context-local slot, and
# every agent built by the standard factory reads that slot through pydantic-ai's
# dynamic-instructions mechanism on its next call. Nothing published means nothing
# injected — an untouched run is byte-identical to before.
#
# A context variable (not a global) because the advisory is per-run state and agent
# runs interleave: two concurrent runs must never read each other's verdict.

_CURRENT_ADVISORY: contextvars.ContextVar[CheckpointRecommendation | None] = (
    contextvars.ContextVar("_kv_checkpoint_advisory", default=None)
)

#: Instruction text wrapped around the rendered advisory. Deliberately tells the model
#: what the number *is* and what it may do with it — a bare score with no framing is an
#: invitation to over-read it.
_ADVISORY_PREAMBLE = (
    "\n\nKV-CHECKPOINT ADVISORY (from the system's checkpoint-worthiness scorers — "
    "advisory only, you decide):\n"
)
_ADVISORY_EPILOGUE = (
    "\nIf you judge this context worth preserving, call graph_kv_checkpoint with "
    "action='checkpoint_now'. Durable (cross-session) persistence additionally "
    "requires an explicit operator grant and is refused without one."
)


def publish_checkpoint_advisory(
    recommendation: CheckpointRecommendation | None,
) -> contextvars.Token[CheckpointRecommendation | None]:
    """Make ``recommendation`` the advisory the next model call sees.

    Returns the context token so a caller that wants strict scoping can reset it;
    callers that simply want the latest verdict to be visible can ignore it.
    """
    return _CURRENT_ADVISORY.set(recommendation)


def current_checkpoint_advisory() -> CheckpointRecommendation | None:
    """The advisory in force for this context, if any."""
    return _CURRENT_ADVISORY.get()


def clear_checkpoint_advisory() -> None:
    """Drop the current advisory (e.g. once the model has acted on it)."""
    _CURRENT_ADVISORY.set(None)


def render_checkpoint_advisory_instructions() -> str:
    """The dynamic-instruction fragment for the current advisory, or ``""``.

    Returns the empty string when no advisory is published, or when the verdict is
    :attr:`CheckpointTier.NONE` — telling a model "this is not a good moment to
    checkpoint" every single turn is prompt noise that teaches it to ignore the whole
    channel. The advisory appears only when there is something to act on.
    """
    recommendation = _CURRENT_ADVISORY.get()
    if recommendation is None:
        return ""
    if recommendation.recommended_tier is CheckpointTier.NONE:
        return ""
    return _ADVISORY_PREAMBLE + recommendation.as_advisory() + _ADVISORY_EPILOGUE


def _record_recommendation(tier: str) -> None:
    """Best-effort worthiness telemetry (the established WS-4 idiom in this package)."""
    try:
        from agent_utilities.observability.gateway_metrics import (
            KVCACHE_CHECKPOINT_RECOMMENDATIONS,
        )

        KVCACHE_CHECKPOINT_RECOMMENDATIONS.labels(tier=tier).inc()
    except Exception as exc:  # noqa: BLE001 — metrics must never break the caller
        logger.debug("checkpoint recommendation telemetry recording failed: %s", exc)
