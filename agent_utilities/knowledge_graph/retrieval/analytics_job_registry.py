#!/usr/bin/python
from __future__ import annotations

"""Analytics-job feature/model/experiment registries (CONCEPT:INT-P2-1b, L41).

The engine's durable analytics-job plane (``epistemic-graph`` ``eg-jobs``,
INT-P2-1) commits every job result as a provenance'd ``:Claim``/``:Evidence``
pair stamped with full **AlgoVersion lineage** (``family``, ``algorithm``,
``params_digest``, ``code_version``, ``env_version``) plus the immutable
input-snapshot handle (``graph`` + OCC ``version``) it ran against — see
``epistemic-graph`` ``crates/eg-jobs/src/claim.rs``. That crate's own docs name
the follow-up this module is: *"an AU-side feature/model/experiment REGISTRY
that indexes committed claims by ``AlgoVersion`` lineage across jobs ...
building a queryable registry ON TOP of that is out of scope [for eg-jobs]."*

**The engine is the store of record.** This module never writes a claim/evidence
node — it only reads them (via the same ``engine.query_cypher`` /
``backend.execute`` Cypher surface every other read-only index in this codebase
uses, e.g. ``retrieval/evaluation_corpus.py``) and builds a **queryable,
rebuildable in-process index** over what it read. Exactly the same
authority split ``retrieval/capability_index.py`` documents for AU-P1-3: the
engine owns the data, this class is a bounded cache/view a caller refreshes
(:meth:`AnalyticsJobRegistry.refresh_from_engine`) — never a second persistence
layer.

Why not extend ``agent_utilities.models.model_registry.ModelRegistry``: that
class is the **LLM routing registry** (provider/model_id/api_key_env/cost/tier
for picking a chat model at runtime) — a different domain from an **analytics
job's produced model artifact** (a trained estimator/rule-set/cluster
assignment with an evaluation score and an approval/deployment state). Folding
job-artifact fields into the LLM-routing schema would conflate two unrelated
concepts under one name; this module's :class:`ModelArtifactRecord` is
deliberately named to avoid any import/identity collision with that class.

## The one grouping key: :class:`AlgoVersionLineage`

Mirrors ``eg_jobs::model::AlgoVersion`` field-for-field. Two job-committed
claims with the SAME lineage are, by the engine's own determinism contract
(``AnalyticsJob::result_ref`` — a pure function of ``(input_snapshot, algo)``),
the same computation re-run — over the SAME snapshot (an idempotent duplicate,
collapsed by the engine to one claim) or over a LATER snapshot (a fresh
``result_ref``/claim: the same model/feature-set/algorithm, re-evaluated as the
underlying graph evolved). Grouping by lineage is what answers "which jobs
produced model X" instead of "which jobs share a `job_id`" (a job_id is
one-shot; a lineage is the model/feature-set/experiment's *identity*).

## Three views, one index

- **Feature registry** (:meth:`AnalyticsJobRegistry.features`) — lineages whose
  ``family`` root is ``feature``/``embedding`` (:func:`is_feature_family`).
  Surfaces whatever of {model, dimension, tokenizer, drift, re-embedding
  schedule} the claim/evidence properties actually carry (optional; the base
  ``eg-jobs`` claim schema does not define these — a future embedding-refresh
  ``JobKind`` would stamp them as extra claim properties, read here generically
  via :attr:`JobResultRecord.extra` so no engine change is required for this
  module to pick them up).
- **Model registry** (:meth:`AnalyticsJobRegistry.models`) — every OTHER
  lineage (the default bucket: today's one shipped job kind,
  ``mining.association``, lands here — an association-rule set is a model
  artifact). Surfaces evaluation/calibration/approval/deployment state the same
  way: from ``extra`` when present, with the job's own ``confidence`` as the
  always-available evaluation signal.
- **Experiment registry** (:meth:`AnalyticsJobRegistry.experiments` /
  :meth:`AnalyticsJobRegistry.jobs_for_experiment`) — grouped by
  ``JobPolicy.purpose`` (the field ``eg-jobs`` already carries on every job,
  e.g. the crate's own test fixture ``purpose="quarterly-mining"``), bundling
  the jobs + their AlgoVersion + input-snapshot + result claim so a run is
  reproducible.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "AlgoVersionLineage",
    "JobResultRecord",
    "FeatureRecord",
    "ModelArtifactRecord",
    "ExperimentRecord",
    "AnalyticsJobRegistry",
    "is_feature_family",
]

#: Family "root" tokens (the segment before the first ``.``) that classify a
#: job-result lineage as a FEATURE job rather than a model job. Everything else
#: falls to the model registry's default bucket.
_FEATURE_FAMILY_ROOTS = frozenset({"feature", "embedding"})

#: Claim/evidence properties already modeled explicitly on :class:`JobResultRecord`
#: — everything else lands in :attr:`JobResultRecord.extra` verbatim.
_MODELED_CLAIM_KEYS = frozenset(
    {
        "type",
        "family",
        "about",
        "confidence",
        "validation_state",
        "job_id",
        "input_snapshot_graph",
        "input_snapshot_version",
        "algo_family",
        "algo_algorithm",
        "algo_params_digest",
        "algo_code_version",
        "algo_env_version",
        "result_ref",
    }
)
_MODELED_EVIDENCE_KEYS = frozenset(
    {
        "type",
        "family",
        "about",
        "provenance",
        "confidence",
        "validation_state",
        "job_id",
        "tenant",
        "actor",
        "purpose",
    }
)


def is_feature_family(family: str) -> bool:
    """Whether ``family`` (e.g. ``"feature.embedding.bge_m3"``) is a feature-set family.

    The root token (before the first ``.``) decides it — ``"feature"`` /
    ``"embedding"`` route to the feature registry; every other family (including
    today's only shipped kind, ``"mining.association"``) is a model-artifact
    family by default.
    """
    root = (family or "").split(".", 1)[0].strip().lower()
    return root in _FEATURE_FAMILY_ROOTS


@dataclass(frozen=True)
class AlgoVersionLineage:
    """Mirrors ``eg_jobs::model::AlgoVersion`` — the identity of a computation.

    Two job-result claims sharing every field here are, by the engine's own
    ``result_ref`` determinism, the SAME algorithm/model/feature-set build —
    the grouping key every registry view indexes on.
    """

    family: str
    algorithm: str
    params_digest: str
    code_version: str
    env_version: str

    def key(self) -> tuple[str, str, str, str, str]:
        """The hashable/sortable tuple form used as the internal dict key."""
        return (
            self.family,
            self.algorithm,
            self.params_digest,
            self.code_version,
            self.env_version,
        )

    def is_feature(self) -> bool:
        """Whether this lineage's ``family`` classifies as a feature-set family."""
        return is_feature_family(self.family)


@dataclass
class JobResultRecord:
    """One job's committed result, as read back from its ``:Claim``/``:Evidence`` pair.

    Built from a claim row alone (``tenant``/``actor``/``purpose`` are ``None``)
    when no matching evidence row was indexed, and enriched in place once the
    evidence row arrives — so a caller that only queried claims (skipping the
    evidence round-trip) still gets a usable record.
    """

    job_id: str
    result_ref: str
    claim_id: str
    lineage: AlgoVersionLineage
    input_snapshot_graph: str
    input_snapshot_version: int
    confidence: float
    validation_state: str
    tenant: str | None = None
    actor: str | None = None
    purpose: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureRecord:
    """A feature-set identity (one per feature lineage) — CONCEPT:INT-P2-1b."""

    lineage: AlgoVersionLineage
    jobs: list[JobResultRecord]
    source_snapshots: list[tuple[str, int]]
    policy_purposes: list[str]
    validity: str
    model: str | None = None
    dimension: int | None = None
    tokenizer: str | None = None
    drift_score: float | None = None
    reembedding_schedule: str | None = None


@dataclass
class ModelArtifactRecord:
    """A model-artifact identity (one per non-feature lineage) — CONCEPT:INT-P2-1b.

    Deliberately NOT named ``ModelRegistry``/``ModelRecord`` to avoid any
    confusion with :class:`agent_utilities.models.model_registry.ModelRegistry`
    (the unrelated LLM-routing registry) — see the module docstring.
    """

    lineage: AlgoVersionLineage
    jobs: list[JobResultRecord]
    version: str
    source_snapshots: list[tuple[str, int]]
    evaluation_metrics: dict[str, Any]
    approval_state: str
    deployment_state: str
    calibration: Any | None = None
    cost_usd: float | None = None
    risk_score: float | None = None


@dataclass
class ExperimentRecord:
    """A reproducible run: every job sharing one ``JobPolicy.purpose`` — CONCEPT:INT-P2-1b."""

    purpose: str
    tenant: str | None
    jobs: list[JobResultRecord]

    @property
    def lineages(self) -> list[AlgoVersionLineage]:
        """Distinct AlgoVersion lineages this experiment's jobs ran, in first-seen order."""
        seen: dict[tuple[str, str, str, str, str], AlgoVersionLineage] = {}
        for j in self.jobs:
            seen.setdefault(j.lineage.key(), j.lineage)
        return list(seen.values())


class AnalyticsJobRegistry:
    """Queryable index over the engine's job-result claims, by AlgoVersion lineage.

    Not the store of record — the engine's ``:Claim``/``:Evidence`` nodes are.
    This is a rebuildable cache: call :meth:`refresh_from_engine` (or feed it
    directly via :meth:`index_claim`/:meth:`index_evidence`/:meth:`index_rows`
    for tests / offline building) to (re)populate it, then use the query
    methods below.
    """

    def __init__(self) -> None:
        self._records: dict[str, JobResultRecord] = {}  # keyed by result_ref
        self._by_lineage: dict[tuple[str, str, str, str, str], list[str]] = {}
        self._by_experiment: dict[tuple[str, str], list[str]] = {}

    def __len__(self) -> int:
        return len(self._records)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    @staticmethod
    def _lineage_from_claim(props: dict[str, Any]) -> AlgoVersionLineage:
        return AlgoVersionLineage(
            family=str(props.get("algo_family", props.get("family", ""))),
            algorithm=str(props.get("algo_algorithm", "")),
            params_digest=str(props.get("algo_params_digest", "")),
            code_version=str(props.get("algo_code_version", "")),
            env_version=str(props.get("algo_env_version", "")),
        )

    def _index(
        self, lineage_key: tuple[str, str, str, str, str], result_ref: str
    ) -> None:
        ids = self._by_lineage.setdefault(lineage_key, [])
        if result_ref not in ids:
            ids.append(result_ref)

    def index_claim(self, props: dict[str, Any]) -> JobResultRecord | None:
        """Ingest one ``:Claim`` node's properties (job-committed claims only).

        A claim with no ``job_id`` (e.g. the synchronous ``mining.rs``
        writeback claim shape, which shares the convention but not the
        job-lineage fields) is not a job-result claim and is skipped —
        returns ``None``.
        """
        job_id = props.get("job_id")
        if not job_id:
            return None
        result_ref = str(props.get("result_ref", props.get("about", "")))
        if not result_ref:
            return None

        lineage = self._lineage_from_claim(props)
        extra = {k: v for k, v in props.items() if k not in _MODELED_CLAIM_KEYS}
        record = self._records.get(result_ref)
        if record is None:
            record = JobResultRecord(
                job_id=str(job_id),
                result_ref=result_ref,
                claim_id=f"jobclaim:{result_ref}",
                lineage=lineage,
                input_snapshot_graph=str(props.get("input_snapshot_graph", "")),
                input_snapshot_version=int(props.get("input_snapshot_version", 0) or 0),
                confidence=float(props.get("confidence", 0.0) or 0.0),
                validation_state=str(props.get("validation_state", "")),
                extra=extra,
            )
            self._records[result_ref] = record
        else:
            record.lineage = lineage
            record.input_snapshot_graph = str(props.get("input_snapshot_graph", ""))
            record.input_snapshot_version = int(
                props.get("input_snapshot_version", 0) or 0
            )
            record.confidence = float(props.get("confidence", record.confidence) or 0.0)
            record.validation_state = str(props.get("validation_state", ""))
            record.extra.update(extra)

        self._index(lineage.key(), result_ref)
        if record.purpose:
            self._index_experiment(record)
        return record

    def index_evidence(self, props: dict[str, Any]) -> JobResultRecord | None:
        """Ingest one ``:Evidence`` node's properties, enriching (or seeding) its record."""
        job_id = props.get("job_id")
        result_ref = str(props.get("about", ""))
        if not job_id or not result_ref:
            return None

        record = self._records.get(result_ref)
        if record is None:
            lineage = self._lineage_from_claim(props)
            record = JobResultRecord(
                job_id=str(job_id),
                result_ref=result_ref,
                claim_id=f"jobclaim:{result_ref}",
                lineage=lineage,
                input_snapshot_graph="",
                input_snapshot_version=0,
                confidence=float(props.get("confidence", 0.0) or 0.0),
                validation_state=str(props.get("validation_state", "")),
            )
            self._records[result_ref] = record
            self._index(lineage.key(), result_ref)

        record.job_id = str(job_id)
        record.tenant = props.get("tenant") or record.tenant
        record.actor = props.get("actor") or record.actor
        record.purpose = props.get("purpose") or record.purpose
        extra = {k: v for k, v in props.items() if k not in _MODELED_EVIDENCE_KEYS}
        record.extra.update(extra)

        if record.purpose:
            self._index_experiment(record)
        return record

    def _index_experiment(self, record: JobResultRecord) -> None:
        exp_key = (record.tenant or "", record.purpose or "")
        if not exp_key[1]:
            return
        ids = self._by_experiment.setdefault(exp_key, [])
        if record.result_ref not in ids:
            ids.append(record.result_ref)

    def index_rows(
        self,
        claim_rows: list[dict[str, Any]] | None = None,
        evidence_rows: list[dict[str, Any]] | None = None,
    ) -> int:
        """Bulk-ingest claim + evidence rows (e.g. Cypher ``execute()`` results).

        Each row may be the flat property dict, or ``{"c": {...}}`` /
        ``{"e": {...}}`` (the shape ``engine.query_cypher(... RETURN c ...)``
        returns) — unwrapped the same way ``evaluation_corpus.py`` does
        (``row.get("c", row)``).

        Returns the number of distinct job-result records indexed.
        """
        for row in claim_rows or []:
            props = row.get("c", row) if isinstance(row, dict) else row
            self.index_claim(props)
        for row in evidence_rows or []:
            props = row.get("e", row) if isinstance(row, dict) else row
            self.index_evidence(props)
        return len(self._records)

    def refresh_from_engine(self, engine: Any, *, limit: int = 10_000) -> int:
        """(Re)populate the index from the live engine's job-result claims.

        Runs the same read-only ``query_cypher`` surface every other AU
        retrieval index uses (mirrors ``retrieval/evaluation_corpus.py``): a
        property-filtered ``MATCH`` (not label matching, which not every
        backend indexes) for every ``:Claim``/``:Evidence`` node carrying a
        ``job_id`` — i.e. only job-committed claims, never the synchronous
        mining-writeback claim shape. Never raises — a query failure (engine
        unreachable, or a build without a live claim/evidence path) logs and
        leaves the index unchanged, exactly like the other bounded-cache
        indexes in this package.

        Returns the number of distinct job-result records indexed after the
        refresh (cumulative — call on a fresh instance for an exact count).
        """
        try:
            claim_rows = engine.query_cypher(
                "MATCH (c) WHERE c.type = $ctype AND c.job_id IS NOT NULL "
                "RETURN c LIMIT $limit",
                {"ctype": "Claim", "limit": limit},
            )
        except Exception as exc:  # noqa: BLE001 - best-effort refresh, never fatal
            logger.warning("analytics_job_registry: claim refresh failed: %s", exc)
            claim_rows = []
        try:
            evidence_rows = engine.query_cypher(
                "MATCH (e) WHERE e.type = $etype AND e.job_id IS NOT NULL "
                "RETURN e LIMIT $limit",
                {"etype": "Evidence", "limit": limit},
            )
        except Exception as exc:  # noqa: BLE001 - best-effort refresh, never fatal
            logger.warning("analytics_job_registry: evidence refresh failed: %s", exc)
            evidence_rows = []
        return self.index_rows(claim_rows, evidence_rows)

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------
    def lineages(self) -> list[AlgoVersionLineage]:
        """Every distinct AlgoVersion lineage indexed so far."""
        out = []
        for key, ids in self._by_lineage.items():
            if ids:
                out.append(self._records[ids[0]].lineage)
        return out

    def job(self, job_id: str) -> JobResultRecord | None:
        """The record for a given ``job_id``, or ``None`` (last-indexed wins if reused)."""
        for record in self._records.values():
            if record.job_id == job_id:
                return record
        return None

    def jobs_for_result_ref(self, result_ref: str) -> JobResultRecord | None:
        """The record for an exact ``result_ref``, or ``None``."""
        return self._records.get(result_ref)

    def jobs_for_lineage(
        self,
        family: str,
        algorithm: str,
        params_digest: str | None = None,
        code_version: str | None = None,
        env_version: str | None = None,
    ) -> list[JobResultRecord]:
        """Every job-result record sharing ``(family, algorithm, ...)``.

        Answers "which jobs produced model/feature-set X" — the reason this
        registry groups by lineage rather than by ``job_id``. Unset
        ``params_digest``/``code_version``/``env_version`` match ANY value for
        that field (a coarser lookup, e.g. "every build of this algorithm").
        Sorted by ascending ``input_snapshot_version`` (oldest first — the
        evolution of one model/feature-set over graph time).
        """
        matches: list[JobResultRecord] = []
        for record in self._records.values():
            lineage = record.lineage
            if lineage.family != family or lineage.algorithm != algorithm:
                continue
            if params_digest is not None and lineage.params_digest != params_digest:
                continue
            if code_version is not None and lineage.code_version != code_version:
                continue
            if env_version is not None and lineage.env_version != env_version:
                continue
            matches.append(record)
        matches.sort(key=lambda r: (r.input_snapshot_version, r.job_id))
        return matches

    def experiments(self) -> list[str]:
        """Every distinct experiment (``JobPolicy.purpose``) indexed, sorted."""
        return sorted({purpose for _, purpose in self._by_experiment if purpose})

    def jobs_for_experiment(
        self, purpose: str, tenant: str | None = None
    ) -> ExperimentRecord | None:
        """The reproducible run bundle for one experiment, or ``None`` if unknown.

        When ``tenant`` is omitted, jobs across every tenant that ran this
        ``purpose`` are merged (a single-tenant deployment's common case).
        """
        keys = (
            [(tenant or "", purpose)]
            if tenant is not None
            else [k for k in self._by_experiment if k[1] == purpose]
        )
        result_refs: list[str] = []
        for key in keys:
            result_refs.extend(self._by_experiment.get(key, []))
        if not result_refs:
            return None
        jobs = [self._records[rid] for rid in result_refs if rid in self._records]
        jobs.sort(key=lambda r: (r.input_snapshot_version, r.job_id))
        resolved_tenant = tenant if tenant is not None else jobs[0].tenant
        return ExperimentRecord(purpose=purpose, tenant=resolved_tenant, jobs=jobs)

    def features(self) -> list[FeatureRecord]:
        """One :class:`FeatureRecord` per feature-family lineage, newest job first."""
        out: list[FeatureRecord] = []
        for lineage in self.lineages():
            if not lineage.is_feature():
                continue
            jobs = self.jobs_for_lineage(
                lineage.family,
                lineage.algorithm,
                lineage.params_digest,
                lineage.code_version,
                lineage.env_version,
            )
            if not jobs:
                continue
            latest = jobs[-1]
            merged_extra: dict[str, Any] = {}
            for j in jobs:
                merged_extra.update(j.extra)
            out.append(
                FeatureRecord(
                    lineage=lineage,
                    jobs=jobs,
                    source_snapshots=sorted(
                        {
                            (j.input_snapshot_graph, j.input_snapshot_version)
                            for j in jobs
                        }
                    ),
                    policy_purposes=sorted({j.purpose for j in jobs if j.purpose}),
                    validity=latest.validation_state or "unvalidated",
                    model=merged_extra.get("feature_model"),
                    dimension=merged_extra.get("feature_dimension"),
                    tokenizer=merged_extra.get("feature_tokenizer"),
                    drift_score=merged_extra.get("drift_score"),
                    reembedding_schedule=merged_extra.get("reembedding_schedule"),
                )
            )
        return out

    def models(self) -> list[ModelArtifactRecord]:
        """One :class:`ModelArtifactRecord` per non-feature lineage, newest job first."""
        out: list[ModelArtifactRecord] = []
        for lineage in self.lineages():
            if lineage.is_feature():
                continue
            jobs = self.jobs_for_lineage(
                lineage.family,
                lineage.algorithm,
                lineage.params_digest,
                lineage.code_version,
                lineage.env_version,
            )
            if not jobs:
                continue
            latest = jobs[-1]
            merged_extra = {}
            for j in jobs:
                merged_extra.update(j.extra)
            eval_metrics = dict(merged_extra.get("eval_metrics") or {})
            eval_metrics.setdefault("confidence", latest.confidence)
            out.append(
                ModelArtifactRecord(
                    lineage=lineage,
                    jobs=jobs,
                    version=f"{lineage.code_version}:{lineage.params_digest[:12]}",
                    source_snapshots=sorted(
                        {
                            (j.input_snapshot_graph, j.input_snapshot_version)
                            for j in jobs
                        }
                    ),
                    evaluation_metrics=eval_metrics,
                    approval_state=merged_extra.get("approval_state", "unreviewed"),
                    deployment_state=merged_extra.get(
                        "deployment_state", "not_deployed"
                    ),
                    calibration=merged_extra.get("calibration"),
                    cost_usd=merged_extra.get("cost_usd"),
                    risk_score=merged_extra.get("risk_score"),
                )
            )
        return out
