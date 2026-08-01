"""RAM vs disk tiering — the layer that ACTS on a checkpoint-worthiness verdict.

CONCEPT:AU-KG.memory.checkpoint-worthiness-scoring (the acting half).
:mod:`agent_utilities.kvcache.worthiness` decides *whether* now is a good moment;
this module takes the checkpoint, decides *where it lives*, and records *why*.

Three trigger paths, one implementation
---------------------------------------
All three ways a checkpoint can happen route through :class:`TieredCheckpointManager`,
so they share one tiering rule, one eligibility gate, and one audit record:

1. **User-invoked** — :meth:`TieredCheckpointManager.checkpoint_now`. An explicit
   "checkpoint now". Skips the worthiness threshold entirely (a human asking is
   sufficient reason) but does NOT skip the eligibility gate for disk.
2. **Agent-invoked** — :meth:`TieredCheckpointManager.recommend` produces the structured
   advisory the model reads, and the model then calls
   :meth:`TieredCheckpointManager.checkpoint_now` with ``trigger="agent"`` if it
   decides to act. The system recommends; the model chooses *whether*; the authority the
   model is running under decides *whether it may*.
3. **System-autonomous** — :meth:`TieredCheckpointManager.observe`. No LLM in the loop:
   the scorers run, and if the verdict clears the threshold the checkpoint is taken on
   the spot. The payload is passed as a callable so a run that is *not* checkpoint-worthy
   never pays to serialize its KV state.

Two tiers, and the second is not a bigger version of the first
---------------------------------------------------------------
* **RAM (the default)** — :class:`RAMCheckpointStore`, a bounded, tenant-isolated,
  process-local store. Lives for the session, evaporates with the process, makes no
  data-at-rest commitment. Anything worth keeping goes here.
* **DISK** — the existing :class:`~agent_utilities.kvcache.checkpoint.KVCheckpointStore`,
  which writes a content-addressed blob plus a ``:KVCheckpoint`` node that survives the
  session. This is data-at-rest and a retention decision, so it requires BOTH a
  materially higher worthiness bar (:class:`~agent_utilities.kvcache.worthiness.DiskPromotionRule`)
  AND an affirmative verdict from the registered
  :mod:`~agent_utilities.kvcache.eligibility` gate, which derives eligibility from the
  caller's verified authority and the labels of every source that contributed to the
  context (CONCEPT:AU-OS.governance.authority-derived-persistence-eligibility).

**RAM never implies disk consent.** Promotion re-runs the full eligibility check even
for a checkpoint that has already been sitting in RAM for hours — and re-derives the
authority at that moment, so an expired credential or a revoked delegation refuses a
promotion that would have been permitted an hour earlier.

**Which trigger fired is provenance, not authority.** All three paths are gated
identically; ``trigger`` is recorded and never read by the eligibility decision.

Inspectability
--------------
Every checkpoint carries the recommendation that produced it and the eligibility
decision that permitted (or refused) it — including the full
:class:`~agent_utilities.kvcache.eligibility.PersistenceDerivation` (whose authority,
through which delegation chain, against which sources' composed labels, and every
check's verdict). :meth:`TieredCheckpointManager.explain` returns that record for a RAM
checkpoint, and for a persisted one the same material is flattened into the
``:KVCheckpoint`` node's ``provenance`` — so "why was this persisted?" is answerable
from the graph long after the session ended.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent_utilities.kvcache.checkpoint import (
    CrossTenantCheckpointError,
    KVCheckpointError,
    KVCheckpointKey,
    KVCheckpointRecord,
)
from agent_utilities.kvcache.eligibility import (
    ContributingSource,
    EligibilityDecision,
    Initiator,
    PersistenceEligibilityGate,
    PersistenceRequest,
    derive_caller_authority,
    get_persistence_eligibility_gate,
)
from agent_utilities.kvcache.worthiness import (
    CheckpointAdvisor,
    CheckpointObservation,
    CheckpointRecommendation,
    CheckpointTier,
    clear_checkpoint_advisory,
    publish_checkpoint_advisory,
)

logger = logging.getLogger(__name__)

__all__ = [
    "CheckpointExplanation",
    "CheckpointOutcome",
    "RAMCheckpointRecord",
    "RAMCheckpointStore",
    "RAMTierStats",
    "TieredCheckpointManager",
    "prefix_digest",
]

#: Default RAM-tier bounds. Module constants rather than env flags (*Configuration
#: discipline*): a deployment that needs different bounds constructs the store with
#: them, and the correct value is a property of the workload, not of the host.
DEFAULT_MAX_RAM_CHECKPOINTS = 32
DEFAULT_MAX_RAM_BYTES = 512 * 1024 * 1024


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _record_tier_op(trigger: str, tier: str, outcome: str) -> None:
    """Best-effort tiering telemetry (the established WS-4 idiom in this package)."""
    try:
        from agent_utilities.observability.gateway_metrics import (
            KVCACHE_CHECKPOINT_TIER_OPS,
        )

        KVCACHE_CHECKPOINT_TIER_OPS.labels(
            trigger=trigger, tier=tier, outcome=outcome
        ).inc()
    except Exception as exc:  # noqa: BLE001 — metrics must never break the caller
        logger.debug("checkpoint tier telemetry recording failed: %s", exc)


class RAMCheckpointRecord(BaseModel):
    """Provenance for a RAM-tier checkpoint (never the bytes — see
    :meth:`RAMCheckpointStore.fetch`)."""

    checkpoint_id: str
    key: KVCheckpointKey
    run_id: str = ""
    point: str = ""
    size_bytes: int = 0
    created_at: str = ""
    trigger: Initiator = "system"
    recommendation: CheckpointRecommendation | None = None
    #: Every source that contributed material to the checkpointed context. Recorded at
    #: RAM-tier time because that is when the context is in hand; ``promote`` re-reads
    #: it rather than re-deriving it, so the labels a checkpoint is judged against are
    #: the ones its content actually came from. EMPTY DENIES at promotion — an
    #: unprovenanced RAM checkpoint is fine, an unprovenanced durable one is not.
    sources: tuple[ContributingSource, ...] = ()
    #: Set once this RAM checkpoint has been promoted to the durable tier, so the same
    #: context is never persisted twice and ``explain`` can follow the promotion.
    promoted: bool = False
    #: The eligibility verdict from the most recent promotion ATTEMPT — present and
    #: ``permitted=False`` when a promotion was refused, which is exactly the record an
    #: operator asking "why wasn't this persisted?" needs.
    eligibility: EligibilityDecision | None = None

    model_config = ConfigDict(extra="forbid")


class RAMTierStats(BaseModel):
    """RAM-tier occupancy. A model rather than a bare dict because these keys cross a
    surface boundary (the ``ram_stats`` MCP action) — a producer writing ``resident``
    while a consumer reads ``resident_bytes`` is the exact silent-drift failure typed
    seams exist to prevent."""

    entries: int = Field(ge=0)
    resident_bytes: int = Field(ge=0)
    max_entries: int = Field(gt=0)
    max_bytes: int = Field(gt=0)
    evictions: int = Field(ge=0)

    model_config = ConfigDict(extra="forbid", frozen=True)


class CheckpointExplanation(BaseModel):
    """The answer to "why does this checkpoint exist, and why is it where it is?".

    Typed for the same reason as :class:`RAMTierStats`: this is the inspectability
    contract an operator (and the ``explain`` MCP action) reads key-by-key, so the keys
    are part of the API and must not be able to drift silently.
    """

    checkpoint_id: str
    tier: CheckpointTier
    trigger: Initiator = "system"
    created_at: str = ""
    size_bytes: int = 0
    eligibility_gate_in_force: str = ""
    #: Present for a checkpoint still tracked in the RAM tier (where the full verdict
    #: lives). A checkpoint known only from the durable tier carries ``provenance``
    #: instead — the same material, as persisted onto the ``:KVCheckpoint`` node.
    recommendation: CheckpointRecommendation | None = None
    eligibility: EligibilityDecision | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class CheckpointOutcome(BaseModel):
    """What actually happened for one checkpoint request, on any of the three paths."""

    taken: bool = Field(description="True iff a checkpoint was actually stored.")
    tier: CheckpointTier = CheckpointTier.NONE
    trigger: Initiator = "system"
    checkpoint_id: str = ""
    reason: str = Field(
        default="", description="Why this outcome — always populated, permit or refuse."
    )
    recommendation: CheckpointRecommendation | None = None
    eligibility: EligibilityDecision | None = None
    ram_record: RAMCheckpointRecord | None = None
    disk_record: KVCheckpointRecord | None = None

    model_config = ConfigDict(extra="forbid")


class RAMCheckpointStore:
    """The default tier: a bounded, tenant-isolated, process-local checkpoint store.

    Bounded by both entry count and total bytes, evicting least-recently-used first —
    a KV blob is large, and an unbounded in-process cache of them is a memory leak with
    extra steps.

    Tenant isolation mirrors :class:`~agent_utilities.kvcache.checkpoint.KVCheckpointStore`
    exactly: :meth:`fetch` re-checks the stored tenant against the requesting one and
    raises :class:`~agent_utilities.kvcache.checkpoint.CrossTenantCheckpointError` on a
    mismatch. The RAM tier being cheap and transient does not make it a weaker security
    boundary — a checkpoint id can be handed around, so the check has to live at the
    load primitive on both tiers.
    """

    def __init__(
        self,
        *,
        max_entries: int = DEFAULT_MAX_RAM_CHECKPOINTS,
        max_bytes: int = DEFAULT_MAX_RAM_BYTES,
    ) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be > 0")
        if max_bytes <= 0:
            raise ValueError("max_bytes must be > 0")
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, tuple[RAMCheckpointRecord, bytes]] = (
            OrderedDict()
        )
        self._bytes = 0
        self.evictions = 0

    def put(
        self,
        data: bytes,
        *,
        key: KVCheckpointKey,
        run_id: str = "",
        point: str = "",
        trigger: Initiator = "system",
        recommendation: CheckpointRecommendation | None = None,
        sources: tuple[ContributingSource, ...] = (),
    ) -> RAMCheckpointRecord:
        """Store ``data`` in the RAM tier under ``key``'s deterministic checkpoint id."""
        if not data:
            raise KVCheckpointError("refusing to checkpoint an empty KV payload")
        if not key.tenant:
            raise KVCheckpointError(
                "KVCheckpointKey.tenant is mandatory — a checkpoint with no tenant can "
                "never be safely loaded back"
            )
        record = RAMCheckpointRecord(
            checkpoint_id=key.checkpoint_id,
            key=key,
            run_id=run_id,
            point=point,
            size_bytes=len(data),
            created_at=_now(),
            trigger=trigger,
            recommendation=recommendation,
            sources=tuple(sources),
        )
        with self._lock:
            existing = self._entries.pop(record.checkpoint_id, None)
            if existing is not None:
                self._bytes -= len(existing[1])
            self._entries[record.checkpoint_id] = (record, data)
            self._bytes += len(data)
            self._evict_locked()
        logger.info(
            "[CONCEPT:AU-KG.memory.checkpoint-worthiness-scoring] RAM checkpoint %s "
            "stored (run=%s, point=%s, tenant=%s, %d bytes, trigger=%s)",
            record.checkpoint_id[:12],
            run_id,
            point,
            key.tenant,
            len(data),
            trigger,
        )
        return record

    def fetch(
        self, checkpoint_id: str, *, requesting_tenant: str
    ) -> tuple[RAMCheckpointRecord, bytes]:
        """Load a RAM checkpoint, fail-closed on tenant mismatch.

        Raises :class:`KVCheckpointError` when absent and
        :class:`CrossTenantCheckpointError` when the stored tenant differs — never a
        silent ``None``, exactly like the durable tier.
        """
        if not requesting_tenant:
            raise KVCheckpointError(
                "requesting_tenant is mandatory for a checkpoint load"
            )
        with self._lock:
            entry = self._entries.get(checkpoint_id)
            if entry is None:
                raise KVCheckpointError(
                    f"no RAM checkpoint found for id {checkpoint_id}"
                )
            record, data = entry
            if record.key.tenant != requesting_tenant:
                logger.warning(
                    "[CONCEPT:AU-KG.memory.checkpoint-worthiness-scoring] REFUSED "
                    "cross-tenant RAM checkpoint load: checkpoint=%s owner_tenant=%s "
                    "requester_tenant=%s",
                    checkpoint_id,
                    record.key.tenant,
                    requesting_tenant,
                )
                raise CrossTenantCheckpointError(
                    f"RAM checkpoint {checkpoint_id} belongs to tenant "
                    f"{record.key.tenant!r}, refusing load for {requesting_tenant!r}"
                )
            self._entries.move_to_end(checkpoint_id)
            return record, data

    def record(
        self, checkpoint_id: str, *, requesting_tenant: str
    ) -> RAMCheckpointRecord:
        """Provenance only, with the same fail-closed tenant check as :meth:`fetch`."""
        return self.fetch(checkpoint_id, requesting_tenant=requesting_tenant)[0]

    def update(self, record: RAMCheckpointRecord) -> None:
        """Replace a stored record's provenance in place (bytes untouched)."""
        with self._lock:
            entry = self._entries.get(record.checkpoint_id)
            if entry is None:
                return
            self._entries[record.checkpoint_id] = (record, entry[1])

    def drop(self, checkpoint_id: str, *, requesting_tenant: str) -> bool:
        """Remove a RAM checkpoint. Cross-tenant drops are refused like loads."""
        self.record(checkpoint_id, requesting_tenant=requesting_tenant)
        with self._lock:
            entry = self._entries.pop(checkpoint_id, None)
            if entry is None:
                return False
            self._bytes -= len(entry[1])
            return True

    def records(self, *, tenant: str = "") -> list[RAMCheckpointRecord]:
        """Every held record, newest-used last; filtered to ``tenant`` when supplied."""
        with self._lock:
            items = [record for record, _ in self._entries.values()]
        if tenant:
            return [r for r in items if r.key.tenant == tenant]
        return items

    def stats(self) -> RAMTierStats:
        with self._lock:
            return RAMTierStats(
                entries=len(self._entries),
                resident_bytes=self._bytes,
                max_entries=self.max_entries,
                max_bytes=self.max_bytes,
                evictions=self.evictions,
            )

    def _evict_locked(self) -> None:
        """LRU-evict until both bounds hold. Caller must hold the lock."""
        while self._entries and (
            len(self._entries) > self.max_entries or self._bytes > self.max_bytes
        ):
            _, (record, data) = self._entries.popitem(last=False)
            self._bytes -= len(data)
            self.evictions += 1
            logger.debug(
                "[CONCEPT:AU-KG.memory.checkpoint-worthiness-scoring] evicted RAM "
                "checkpoint %s (%d bytes) under pressure",
                record.checkpoint_id[:12],
                len(data),
            )


class TieredCheckpointManager:
    """The one entry point for all three checkpoint trigger paths.

    CONCEPT:AU-KG.memory.checkpoint-worthiness-scoring. Binds a
    :class:`~agent_utilities.kvcache.worthiness.CheckpointAdvisor` (which signals fire),
    a :class:`RAMCheckpointStore` (the default tier), an optional
    :class:`~agent_utilities.kvcache.checkpoint.KVCheckpointStore` (the durable tier),
    and a :mod:`~agent_utilities.kvcache.eligibility` gate (whether durable is allowed
    at all).

    ``disk_store=None`` is a supported, safe configuration: the durable tier is simply
    unavailable and every promotion is refused with that reason.
    """

    def __init__(
        self,
        *,
        advisor: CheckpointAdvisor | None = None,
        ram_store: RAMCheckpointStore | None = None,
        disk_store: Any | None = None,
        eligibility_gate: PersistenceEligibilityGate | None = None,
        durable_region: str = "",
    ) -> None:
        self.advisor = advisor or CheckpointAdvisor()
        self.ram_store = ram_store or RAMCheckpointStore()
        self.disk_store = disk_store
        self._eligibility_gate = eligibility_gate
        #: The region ``disk_store`` physically writes into. A fact about deployment
        #: topology, not a policy value, and it is only consulted when a contributing
        #: source actually restricts residency — where it is unknown, such a source
        #: denies.
        self.durable_region = durable_region

    @property
    def eligibility_gate(self) -> PersistenceEligibilityGate:
        """The gate this manager consults — its own, or the process-wide default.

        Resolved on every access rather than captured at construction, so a deployment
        that installs its real policy with
        :func:`~agent_utilities.kvcache.eligibility.set_persistence_eligibility_gate`
        takes effect on already-constructed managers.
        """
        return self._eligibility_gate or get_persistence_eligibility_gate()

    # -- path 2 (agent): recommend, don't decide -----------------------------
    def recommend(self, observation: CheckpointObservation) -> CheckpointRecommendation:
        """Score the current moment and return the advisory.

        This is the surface the **agent** path reads: the system computes the signals
        and hands the model a structured verdict
        (:meth:`~agent_utilities.kvcache.worthiness.CheckpointRecommendation.as_advisory`)
        so the model can decide. It takes no checkpoint.

        The verdict is also **published** to the context-local advisory slot
        (CONCEPT:AU-ORCH.optimization.checkpoint-recommendation-surface), so every agent
        built by the standard factory renders it into its instructions on the next call
        without the caller having to plumb it through — that publish is what makes this
        a recommendation *to the model* rather than just a return value.
        """
        recommendation = self.advisor.evaluate(observation)
        publish_checkpoint_advisory(recommendation)
        return recommendation

    # -- path 1 (user) and the agent's follow-through ------------------------
    def checkpoint_now(
        self,
        data: bytes,
        *,
        key: KVCheckpointKey,
        run_id: str = "",
        point: str = "",
        trigger: Initiator = "user",
        persist: bool = False,
        observation: CheckpointObservation | None = None,
        sources: tuple[ContributingSource, ...] = (),
        session: Any | None = None,
    ) -> CheckpointOutcome:
        """Take a checkpoint **now**, because someone explicitly asked.

        Deliberately does NOT consult the worthiness threshold: an explicit request is
        its own justification, and second-guessing a human who says "checkpoint this"
        is the wrong behaviour. When ``observation`` is supplied the recommendation is
        still computed and attached, so the record shows what the scorers thought at the
        time even though they did not gate the decision.

        ``persist=True`` additionally attempts a durable write, which DOES go through
        the full eligibility gate. A refused persistence is not an error: the RAM
        checkpoint still exists, and the outcome carries the refusal reason.

        ``sources`` are the labelled sources that contributed to this context. They are
        the material half of the eligibility derivation (the authority half is read from
        the caller's verified session), and an empty set refuses durable persistence.
        """
        recommendation = (
            self.advisor.evaluate(observation) if observation is not None else None
        )
        try:
            ram_record = self.ram_store.put(
                data,
                key=key,
                run_id=run_id,
                point=point,
                trigger=trigger,
                recommendation=recommendation,
                sources=sources,
            )
        except KVCheckpointError as exc:
            _record_tier_op(trigger, "ram", "error")
            return CheckpointOutcome(
                taken=False,
                trigger=trigger,
                reason=f"RAM checkpoint refused: {exc}",
                recommendation=recommendation,
            )
        _record_tier_op(trigger, "ram", "taken")
        # The advisory is spent: someone acted on it. Leaving it published would keep
        # telling the model "checkpoint-worthy, consider checkpointing" on every
        # subsequent turn and invite a duplicate of the checkpoint just taken.
        clear_checkpoint_advisory()
        outcome = CheckpointOutcome(
            taken=True,
            tier=CheckpointTier.RAM,
            trigger=trigger,
            checkpoint_id=ram_record.checkpoint_id,
            reason=f"explicit {trigger}-invoked checkpoint stored in RAM",
            recommendation=recommendation,
            ram_record=ram_record,
        )
        if not persist:
            return outcome
        return self.promote(
            ram_record.checkpoint_id,
            requesting_tenant=key.tenant,
            trigger=trigger,
            session=session,
            base_outcome=outcome,
        )

    # -- path 3 (system): decide and act, no LLM in the loop -----------------
    def observe(
        self,
        observation: CheckpointObservation,
        *,
        key: KVCheckpointKey,
        payload: bytes | Callable[[], bytes],
        run_id: str = "",
        point: str = "",
        sources: tuple[ContributingSource, ...] = (),
        session: Any | None = None,
    ) -> CheckpointOutcome:
        """Score the moment and checkpoint autonomously if it clears the bar.

        ``payload`` may be a callable, and it is invoked **only** when the verdict is to
        checkpoint — so a moment that is not worth freezing never pays to serialize KV
        state. That laziness is the reason this path can be called often.

        The autonomous path can reach the RAM tier on its own. Whether it reaches disk
        is **not** decided by it being autonomous: the gate reads the authority the run
        is executing under, so a scheduled run carrying a verified session whose
        authority dominates the contributing sources' labels persists, and one that does
        not produces a RAM checkpoint plus a recorded, refused verdict — visible evidence
        that the system *wanted* to persist and was not permitted to. A daemon tick with
        no session at all has no authority and is therefore always refused.

        Unlike :meth:`recommend`, this deliberately does **not** publish an advisory to
        the model: this path *acts*, so telling the model "checkpoint-worthy, consider
        checkpointing" right after the system already checkpointed would only invite a
        duplicate. The agent path is for when the decision is the model's to make.
        """
        recommendation = self.advisor.evaluate(observation)
        if recommendation.recommended_tier is CheckpointTier.NONE:
            _record_tier_op("system", "none", "declined")
            return CheckpointOutcome(
                taken=False,
                trigger="system",
                reason=f"not checkpoint-worthy: {recommendation.rationale}",
                recommendation=recommendation,
            )

        data = payload() if callable(payload) else payload
        try:
            ram_record = self.ram_store.put(
                data,
                key=key,
                run_id=run_id or observation.run_id,
                point=point or observation.point,
                trigger="system",
                recommendation=recommendation,
                sources=sources,
            )
        except KVCheckpointError as exc:
            _record_tier_op("system", "ram", "error")
            return CheckpointOutcome(
                taken=False,
                trigger="system",
                reason=f"RAM checkpoint refused: {exc}",
                recommendation=recommendation,
            )
        _record_tier_op("system", "ram", "taken")
        outcome = CheckpointOutcome(
            taken=True,
            tier=CheckpointTier.RAM,
            trigger="system",
            checkpoint_id=ram_record.checkpoint_id,
            reason=f"autonomous checkpoint: {recommendation.rationale}",
            recommendation=recommendation,
            ram_record=ram_record,
        )
        if recommendation.recommended_tier is not CheckpointTier.DISK:
            return outcome
        return self.promote(
            ram_record.checkpoint_id,
            requesting_tenant=key.tenant,
            trigger="system",
            session=session,
            base_outcome=outcome,
        )

    # -- RAM -> disk promotion ------------------------------------------------
    def promote(
        self,
        checkpoint_id: str,
        *,
        requesting_tenant: str,
        trigger: Initiator = "user",
        session: Any | None = None,
        base_outcome: CheckpointOutcome | None = None,
    ) -> CheckpointOutcome:
        """Promote a RAM checkpoint to the durable tier, through the eligibility gate.

        The gate is consulted on **every** promotion, including one for a checkpoint
        that has been resident in RAM since the session began: having been kept in
        memory is not consent to write the same context to disk.

        **The authority is derived here, at promotion time, not at checkpoint time.**
        :func:`~agent_utilities.kvcache.eligibility.derive_caller_authority` reads the
        caller's verified ``GraphSession`` and any active ``SpawnDelegation`` at the
        moment of the write, so a credential that has expired or a delegation that has
        been revoked since the RAM checkpoint was taken refuses the promotion. Nothing
        the caller passes in can substitute for that: there is no grant flag, and a
        payload-supplied tenant is checked *against* the verified one rather than
        trusted.

        A refusal leaves the RAM checkpoint intact and returns an outcome whose
        ``eligibility`` explains why — including the full
        :class:`~agent_utilities.kvcache.eligibility.PersistenceDerivation`, recorded on
        the RAM record too, so :meth:`explain` can answer "why wasn't this persisted?"
        later.
        """
        ram_record, data = self.ram_store.fetch(
            checkpoint_id, requesting_tenant=requesting_tenant
        )
        outcome = base_outcome or CheckpointOutcome(
            taken=True,
            tier=CheckpointTier.RAM,
            trigger=trigger,
            checkpoint_id=checkpoint_id,
            reason="already resident in the RAM tier",
            recommendation=ram_record.recommendation,
            ram_record=ram_record,
        )

        request = PersistenceRequest(
            tenant=requesting_tenant,
            run_id=ram_record.run_id,
            point=ram_record.point,
            size_bytes=ram_record.size_bytes,
            trigger=trigger,
            worthiness_score=(
                ram_record.recommendation.score if ram_record.recommendation else 0.0
            ),
            authority=derive_caller_authority(session=session),
            sources=ram_record.sources,
            target_region=self.durable_region,
        )
        gate = self.eligibility_gate
        decision = gate.evaluate(request)
        ram_record = ram_record.model_copy(update={"eligibility": decision})
        self.ram_store.update(ram_record)
        outcome = outcome.model_copy(
            update={"eligibility": decision, "ram_record": ram_record}
        )

        if not decision.permitted:
            _record_tier_op(trigger, "disk", "eligibility_denied")
            logger.info(
                "[CONCEPT:AU-OS.governance.checkpoint-persistence-eligibility] disk "
                "promotion of checkpoint %s REFUSED by gate %s: %s",
                checkpoint_id[:12],
                decision.gate,
                decision.reason,
            )
            return outcome.model_copy(
                update={
                    "reason": (
                        f"{outcome.reason}; disk promotion refused by "
                        f"{decision.gate}: {decision.reason}"
                    )
                }
            )

        if self.disk_store is None:
            _record_tier_op(trigger, "disk", "error")
            return outcome.model_copy(
                update={
                    "reason": (
                        f"{outcome.reason}; disk promotion permitted by "
                        f"{decision.gate} but no durable store is bound to this manager"
                    )
                }
            )

        disk_record = self.disk_store.create_checkpoint(
            data,
            key=ram_record.key,
            run_id=ram_record.run_id,
            point=ram_record.point,
            session=session,
            provenance=self._persistence_provenance(ram_record, decision, trigger),
        )
        if disk_record is None:
            _record_tier_op(trigger, "disk", "error")
            return outcome.model_copy(
                update={
                    "reason": (
                        f"{outcome.reason}; disk promotion permitted by "
                        f"{decision.gate} but the durable write failed"
                    )
                }
            )

        self.ram_store.update(ram_record.model_copy(update={"promoted": True}))
        _record_tier_op(trigger, "disk", "taken")
        logger.info(
            "[CONCEPT:AU-OS.governance.checkpoint-persistence-eligibility] checkpoint "
            "%s persisted to disk under gate %s (trigger=%s, score=%.2f)",
            checkpoint_id[:12],
            decision.gate,
            trigger,
            request.worthiness_score,
        )
        return outcome.model_copy(
            update={
                "tier": CheckpointTier.DISK,
                "disk_record": disk_record,
                "reason": (
                    f"{outcome.reason}; persisted to disk, permitted by "
                    f"{decision.gate}: {decision.reason}"
                ),
            }
        )

    @staticmethod
    def _persistence_provenance(
        record: RAMCheckpointRecord,
        decision: EligibilityDecision,
        trigger: Initiator,
    ) -> dict[str, Any]:
        """The "why was this persisted?" record written onto the ``:KVCheckpoint`` node.

        Kept small and value-typed on purpose: the durable node is provenance, not a
        second copy of the scoring engine's internals. It carries the aggregate, the
        drivers, the deciding gate and its unresolved policy questions — everything an
        operator auditing a durable checkpoint needs, and nothing that would leak the
        run's content.
        """
        provenance: dict[str, Any] = {
            "trigger": trigger,
            "eligibility_gate": decision.gate,
            "eligibility_reason": decision.reason,
            "eligibility_unresolved": list(decision.unresolved),
            "eligibility_policy_ref": decision.policy_ref,
            "promoted_from_tier": CheckpointTier.RAM.value,
            "promoted_at": _now(),
        }
        if decision.derivation is not None:
            # The derivation is what makes a durable checkpoint auditable long after the
            # session ended: WHOSE authority permitted it, through WHICH delegation
            # chain, and against WHICH sources' composed labels. Flattened to value
            # types because the node stores properties, not nested models.
            derivation = decision.derivation
            provenance.update(
                {
                    "authorized_actor": derivation.authority.actor_id,
                    "authorized_tenant": derivation.authority.tenant,
                    "authorized_clearance": derivation.authority.clearance_label,
                    "delegation_chain": list(derivation.authority.delegation_chain),
                    "delegation_principal": derivation.authority.delegation_principal,
                    "composed_classification": derivation.label.classification,
                    "composed_residency_regions": sorted(
                        derivation.label.residency_regions
                    ),
                    "composed_retention_days": derivation.label.retention_days,
                    "composed_markings": sorted(derivation.label.markings),
                    "contributing_sources": [
                        c.source_id for c in derivation.label.contributions
                    ],
                    "eligibility_rule": derivation.rule,
                    "eligibility_checks": [
                        f"{c.name}={'pass' if c.passed else 'FAIL'}: {c.detail}"
                        for c in derivation.checks
                    ],
                }
            )
        if record.recommendation is not None:
            provenance.update(
                {
                    "worthiness_score": record.recommendation.score,
                    "worthiness_drivers": list(record.recommendation.drivers),
                    "worthiness_blockers": list(record.recommendation.blockers),
                    "worthiness_abstained": list(record.recommendation.abstained),
                }
            )
        return provenance

    # -- inspectability -------------------------------------------------------
    def explain(
        self, checkpoint_id: str, *, requesting_tenant: str
    ) -> CheckpointExplanation:
        """Answer "why does this checkpoint exist, and why is it where it is?".

        Looks in the RAM tier first (where the full recommendation lives), then falls
        back to the durable tier's ``provenance``. Fail-closed on tenant, like every
        other load.
        """
        try:
            record = self.ram_store.record(
                checkpoint_id, requesting_tenant=requesting_tenant
            )
        except CrossTenantCheckpointError:
            raise
        except KVCheckpointError:
            record = None
        if record is not None:
            return CheckpointExplanation(
                checkpoint_id=checkpoint_id,
                tier=(CheckpointTier.DISK if record.promoted else CheckpointTier.RAM),
                trigger=record.trigger,
                created_at=record.created_at,
                size_bytes=record.size_bytes,
                recommendation=record.recommendation,
                eligibility=record.eligibility,
                eligibility_gate_in_force=self.eligibility_gate.name,
            )
        if self.disk_store is None:
            raise KVCheckpointError(
                f"no checkpoint {checkpoint_id} in the RAM tier and no durable store "
                "is bound to this manager"
            )
        disk_record = self.disk_store.get_checkpoint(
            checkpoint_id, requesting_tenant=requesting_tenant
        )
        return CheckpointExplanation(
            checkpoint_id=checkpoint_id,
            tier=CheckpointTier.DISK,
            trigger=disk_record.provenance.get("trigger", "") or "system",
            created_at=disk_record.created_at,
            size_bytes=disk_record.size_bytes,
            provenance=disk_record.provenance,
            eligibility_gate_in_force=self.eligibility_gate.name,
        )


def prefix_digest(text: str) -> str:
    """Content digest for a prompt prefix, for building a :class:`KVCheckpointKey`.

    A small shared helper so every call site derives ``prefix_digest`` the same way —
    two callers hashing the same prefix differently would mint different checkpoint ids
    for identical context and silently defeat reuse.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
