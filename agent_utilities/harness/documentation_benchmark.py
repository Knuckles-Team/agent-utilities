"""Deterministic agent-documentation benchmark and evidence contract.

NE-145 measures the documentation interface we own rather than importing a
third-party readiness score. A fixed, versioned question corpus is answered
through two injected read-only adapters: an HTML baseline and the generated
agent path. The runner records correctness, exact source selection, request
count, elapsed time, input/output/reasoning-token usage, stale-answer rate,
cache mode, repeat identity, and the revisions that made the result possible.

The adapters are deliberately outside this module. A live adapter may fetch a
site or call a model, while the offline fixtures can use a small mapping. The
runner never writes, mutates a cache, or stores answer/source content. Clearing
a cache is an explicit callback supplied by the caller; durable evidence is
written only when :meth:`DocumentationBenchmarkEvidence.persist` is called.

Promotion is intentionally conservative: a candidate must not regress
correctness, source selection, or stale-answer rate; it must show a measured
latency or token improvement; repeated answers must be identical; and a human
caller must explicitly mark that improvement as reviewed. No external
benchmark percentage is used as an acceptance threshold.

CONCEPT:AU-AHE.evaluation.capability-benchmark-regression-ratchet
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "BenchmarkBounds",
    "BenchmarkBoundExceeded",
    "CacheMode",
    "DocumentationAdapter",
    "DocumentationBenchmarkError",
    "DocumentationBenchmarkEvidence",
    "DocumentationBenchmarkRunner",
    "DocumentationCorpus",
    "DocumentationPath",
    "DocumentationQuestion",
    "MetricSet",
    "MetricStats",
    "PathAnswer",
    "PathComparison",
    "PathObservation",
    "PathRevision",
    "PathSummary",
    "ProjectType",
    "RegressionThresholds",
    "RepeatIdentity",
    "RequestRecorder",
    "TokenUsage",
    "DEFAULT_DOCUMENTATION_CORPUS",
    "compare_documentation_paths",
]


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/+-]{0,127}$")
_SAFE_REVISION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/+@=-]{0,255}$")
_UNTRUSTED_REVISION_NAMES = {"head", "main", "master", "latest", "unknown"}
_SCHEMA_VERSION = "docs-agent-benchmark.v1"


def _is_safe_opaque_id(value: str) -> bool:
    """Reject identifiers that could smuggle a URL or local path into evidence."""

    return bool(
        isinstance(value, str)
        and _SAFE_ID.fullmatch(value)
        and "://" not in value
        and not value.startswith(("/", "~"))
    )


class DocumentationPath(StrEnum):
    """The two documentation delivery paths compared by the benchmark."""

    HTML_BASELINE = "html_baseline"
    AGENT_GENERATED = "agent_generated"


class CacheMode(StrEnum):
    """Whether an observation is the clean-cache pass or a repeated pass."""

    CLEAN = "clean"
    REPEAT = "repeat"


class ProjectType(StrEnum):
    """Project families represented in the fixed corpus."""

    FRAMEWORK = "framework"
    CONNECTOR = "connector"
    FRONTEND = "frontend"
    ENGINE = "engine"


class DocumentationBenchmarkError(RuntimeError):
    """Base error for a bounded or malformed benchmark execution."""


class BenchmarkBoundExceeded(DocumentationBenchmarkError):
    """Raised when an adapter exceeds a declared benchmark bound."""


class TokenUsage(BaseModel):
    """Provider-reported token counts; unavailable values remain ``None``.

    Zero is a real measured count, not a stand-in for an adapter that did not
    expose usage. This distinction keeps a missing reasoning-token field from
    becoming a false efficiency win.
    """

    model_config = ConfigDict(frozen=True)

    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _validate_finite(self) -> TokenUsage:
        if any(
            value is not None and not math.isfinite(value)
            for value in (
                self.input_tokens,
                self.output_tokens,
                self.reasoning_tokens,
            )
        ):
            raise ValueError("token counts must be finite")
        return self


class DocumentationQuestion(BaseModel):
    """One fixed question and its independently reviewable answer contract."""

    model_config = ConfigDict(frozen=True)

    question_id: str = Field(pattern=_SAFE_ID.pattern)
    project_type: ProjectType
    prompt: str = Field(min_length=1, max_length=2_000)
    required_facts: tuple[str, ...]
    expected_source_ids: tuple[str, ...]
    stale_source_ids: tuple[str, ...] = ()
    stale_facts: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate_contract(self) -> DocumentationQuestion:
        if not _is_safe_opaque_id(self.question_id):
            raise ValueError("question ids must be bounded opaque identifiers")
        if not self.required_facts:
            raise ValueError("documentation questions need at least one required fact")
        if not self.expected_source_ids:
            raise ValueError("documentation questions need an expected source")
        all_ids = (*self.expected_source_ids, *self.stale_source_ids)
        if any(not _is_safe_opaque_id(value) for value in all_ids):
            raise ValueError("question source ids must be bounded opaque identifiers")
        if len(set(all_ids)) != len(all_ids):
            raise ValueError("question source ids must be unique")
        if any(not fact.strip() for fact in (*self.required_facts, *self.stale_facts)):
            raise ValueError("question facts must not be blank")
        return self


class DocumentationCorpus(BaseModel):
    """Versioned corpus; the runner executes every question, without sampling."""

    model_config = ConfigDict(frozen=True)

    corpus_id: str = Field(pattern=_SAFE_ID.pattern)
    version: str = Field(pattern=_SAFE_REVISION.pattern)
    questions: tuple[DocumentationQuestion, ...]

    @model_validator(mode="after")
    def _validate_corpus(self) -> DocumentationCorpus:
        if not _is_safe_opaque_id(self.corpus_id):
            raise ValueError("corpus ids must be bounded opaque identifiers")
        if (
            not _SAFE_REVISION.fullmatch(self.version)
            or "://" in self.version
            or self.version.startswith(("/", "~"))
        ):
            raise ValueError("corpus versions must be exact bounded identifiers")
        if len(self.questions) < 4:
            raise ValueError("the documentation corpus must cover at least four questions")
        ids = [question.question_id for question in self.questions]
        if len(set(ids)) != len(ids):
            raise ValueError("documentation question ids must be unique")
        project_types = {question.project_type for question in self.questions}
        if len(project_types) < 3:
            raise ValueError("the corpus must cover at least three project types")
        return self


class PathRevision(BaseModel):
    """Exact revisions needed to reproduce one delivery path."""

    model_config = ConfigDict(frozen=True)

    path: DocumentationPath
    model_revision: str = Field(min_length=1, max_length=256)
    tool_revision: str = Field(min_length=1, max_length=256)
    site_revision: str = Field(min_length=1, max_length=256)
    source_revisions: dict[str, str]
    harness_revision: str = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def _validate_revisions(self) -> PathRevision:
        revisions = (
            self.model_revision,
            self.tool_revision,
            self.site_revision,
            self.harness_revision,
            *self.source_revisions.keys(),
            *self.source_revisions.values(),
        )
        if any(
            not _SAFE_REVISION.fullmatch(value)
            or "://" in value
            or value.startswith(("/", "~"))
            or value.casefold() in _UNTRUSTED_REVISION_NAMES
            for value in revisions
        ):
            raise ValueError(
                "model/tool/site/source/harness revisions must be exact bounded "
                "identifiers, not mutable names such as latest or HEAD"
            )
        return self


class PathAnswer(BaseModel):
    """Ephemeral adapter output; raw answer text never enters durable evidence."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(max_length=200_000)
    selected_source_ids: tuple[str, ...]
    token_usage: TokenUsage = Field(default_factory=TokenUsage)


class BenchmarkBounds(BaseModel):
    """Hard resource bounds for one benchmark invocation."""

    model_config = ConfigDict(frozen=True)

    max_questions: int = Field(default=24, ge=4, le=100)
    max_repetitions: int = Field(default=5, ge=2, le=10)
    max_requests_per_answer: int = Field(default=32, ge=1, le=256)
    max_answer_chars: int = Field(default=200_000, ge=1, le=1_000_000)


class RequestRecorder:
    """Bounded source-request counter passed to each read-only adapter."""

    def __init__(self, *, limit: int) -> None:
        self._limit = limit
        self._source_ids: list[str] = []

    def record(self, source_id: str) -> None:
        """Record one opaque source request, failing before the bound is crossed."""

        if not _is_safe_opaque_id(source_id):
            raise DocumentationBenchmarkError("source request id is not a safe opaque id")
        if len(self._source_ids) >= self._limit:
            raise BenchmarkBoundExceeded(
                f"documentation adapter exceeded {self._limit} requests for one answer"
            )
        self._source_ids.append(source_id)

    @property
    def count(self) -> int:
        return len(self._source_ids)

    @property
    def source_ids(self) -> tuple[str, ...]:
        return tuple(self._source_ids)


class DocumentationAdapter(Protocol):
    """Read-only adapter for one path; no network/client is owned by the runner."""

    def answer(
        self, question: DocumentationQuestion, requests: RequestRecorder
    ) -> PathAnswer:
        """Answer *question*, using ``requests.record`` for every source read."""


class MetricStats(BaseModel):
    """Mean plus explicit sampling uncertainty for one measured dimension."""

    model_config = ConfigDict(frozen=True)

    sample_count: int = Field(ge=0)
    missing_count: int = Field(ge=0)
    mean: float | None = None
    standard_deviation: float | None = None
    ci95_half_width: float | None = None


class MetricSet(BaseModel):
    """All metrics collected for one path/cache slice."""

    model_config = ConfigDict(frozen=True)

    correctness: MetricStats
    source_selection: MetricStats
    stale_answer: MetricStats
    request_count: MetricStats
    elapsed_ms: MetricStats
    input_tokens: MetricStats
    output_tokens: MetricStats
    reasoning_tokens: MetricStats


class PathObservation(BaseModel):
    """Privacy-safe durable record for one question/path/repetition."""

    model_config = ConfigDict(frozen=True)

    path: DocumentationPath
    cache_mode: CacheMode
    repetition: int = Field(ge=0)
    question_id: str = Field(pattern=_SAFE_ID.pattern)
    correct: bool
    source_selection_correct: bool
    stale_answer: bool
    requested_source_ids: tuple[str, ...]
    selected_source_ids: tuple[str, ...]
    request_count: int = Field(ge=0)
    elapsed_ms: float = Field(ge=0)
    token_usage: TokenUsage
    answer_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    answer_chars: int = Field(ge=0)


class RepeatIdentity(BaseModel):
    """Whether repeated observations produced the same semantic answer/evidence."""

    model_config = ConfigDict(frozen=True)

    path: DocumentationPath
    comparisons: int = Field(ge=0)
    identical: int = Field(ge=0)
    identity_rate: float = Field(ge=0, le=1)
    mismatched_question_ids: tuple[str, ...] = ()
    identity_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class PathSummary(BaseModel):
    """Aggregate metrics for one path over all and cache-specific slices."""

    model_config = ConfigDict(frozen=True)

    path: DocumentationPath
    all_metrics: MetricSet
    clean_metrics: MetricSet
    repeat_metrics: MetricSet
    repeat_identity: RepeatIdentity


class DocumentationBenchmarkEvidence(BaseModel):
    """Complete structured evidence emitted by a benchmark run."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = _SCHEMA_VERSION
    run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")
    corpus_id: str = Field(pattern=_SAFE_ID.pattern)
    corpus_version: str = Field(pattern=_SAFE_REVISION.pattern)
    question_ids: tuple[str, ...]
    path_revisions: dict[DocumentationPath, PathRevision]
    observations: tuple[PathObservation, ...]
    summaries: tuple[PathSummary, ...]
    uncertainty: tuple[str, ...] = ()

    def to_json(self) -> str:
        """Return canonical JSON without raw prompts, answers, URLs, or paths."""

        return self.model_dump_json(indent=2, exclude_none=False)

    def persist(self, path: str | Path) -> Path:
        """Explicitly persist this privacy-safe evidence atomically.

        The runner never calls this method. The caller owns the destination and
        the decision to retain the evidence; the temporary file is mode 0600 and
        is replaced only after the complete JSON payload is written.
        """

        target = Path(path)
        if not target.parent.is_dir():
            raise FileNotFoundError(f"evidence directory does not exist: {target.parent}")
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        temporary = Path(temporary_name)
        try:
            os.chmod(temporary, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(self.to_json())
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return target


class RegressionThresholds(BaseModel):
    """Reviewed, local regression limits; no external benchmark values."""

    model_config = ConfigDict(frozen=True)

    max_correctness_drop: float = Field(default=0.0, ge=0.0, le=1.0)
    max_source_selection_drop: float = Field(default=0.0, ge=0.0, le=1.0)
    max_stale_answer_increase: float = Field(default=0.0, ge=0.0, le=1.0)
    max_request_count_increase: float = Field(default=0.0, ge=0.0)
    max_elapsed_ms_increase: float = Field(default=0.0, ge=0.0)
    max_token_count_increase: float = Field(default=0.0, ge=0.0)
    min_repeat_identity: float = Field(default=1.0, ge=0.0, le=1.0)


class PathComparison(BaseModel):
    """Baseline/candidate comparison and promotion decision."""

    model_config = ConfigDict(frozen=True)

    baseline_path: DocumentationPath
    candidate_path: DocumentationPath
    deltas: dict[str, float | None]
    violations: tuple[str, ...]
    uncertainty: tuple[str, ...]
    efficiency_improved: bool
    reviewed: bool
    promotion_ready: bool


@dataclass(frozen=True)
class _RunClock:
    """Default clock wrapper; tests can inject a deterministic callable instead."""

    read: Callable[[], float] = time.perf_counter


def _metric_stats(values: Sequence[float | None]) -> MetricStats:
    present = [float(value) for value in values if value is not None]
    missing = len(values) - len(present)
    if not present:
        return MetricStats(sample_count=0, missing_count=missing)
    mean = fmean(present)
    deviation = stdev(present) if len(present) > 1 else 0.0
    half_width = 1.96 * deviation / math.sqrt(len(present))
    return MetricStats(
        sample_count=len(present),
        missing_count=missing,
        mean=round(mean, 6),
        standard_deviation=round(deviation, 6),
        ci95_half_width=round(half_width, 6),
    )


def _metrics(observations: Sequence[PathObservation]) -> MetricSet:
    return MetricSet(
        correctness=_metric_stats([float(item.correct) for item in observations]),
        source_selection=_metric_stats(
            [float(item.source_selection_correct) for item in observations]
        ),
        stale_answer=_metric_stats([float(item.stale_answer) for item in observations]),
        request_count=_metric_stats(
            [float(item.request_count) for item in observations]
        ),
        elapsed_ms=_metric_stats([item.elapsed_ms for item in observations]),
        input_tokens=_metric_stats(
            [item.token_usage.input_tokens for item in observations]
        ),
        output_tokens=_metric_stats(
            [item.token_usage.output_tokens for item in observations]
        ),
        reasoning_tokens=_metric_stats(
            [item.token_usage.reasoning_tokens for item in observations]
        ),
    )


def _normalise(text: str) -> str:
    return " ".join(text.casefold().split())


def _contains_facts(text: str, facts: Sequence[str]) -> bool:
    normalised = _normalise(text)
    return bool(facts) and all(_normalise(fact) in normalised for fact in facts)


def _answer_digest(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _identity_material(observation: PathObservation) -> dict[str, Any]:
    """Fields whose equality means the answer/evidence was semantically stable."""

    return {
        "question_id": observation.question_id,
        "correct": observation.correct,
        "source_selection_correct": observation.source_selection_correct,
        "stale_answer": observation.stale_answer,
        "requested_source_ids": observation.requested_source_ids,
        "selected_source_ids": observation.selected_source_ids,
        "request_count": observation.request_count,
        "answer_digest": observation.answer_digest,
        "token_usage": observation.token_usage.model_dump(mode="json"),
    }


def _identity_digest(observations: Sequence[PathObservation]) -> str:
    payload = json.dumps(
        [_identity_material(item) for item in observations],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _repeat_identity(
    path: DocumentationPath,
    observations: Sequence[PathObservation],
    *,
    repetitions: int,
) -> RepeatIdentity:
    by_key = {
        (item.repetition, item.question_id): item
        for item in observations
    }
    mismatches: list[str] = []
    comparisons = 0
    identical = 0
    for question_id in sorted(
        item.question_id for item in observations if item.repetition == 0
    ):
        first = by_key[(0, question_id)]
        for repetition in range(1, repetitions):
            current = by_key[(repetition, question_id)]
            comparisons += 1
            if _identity_material(first) == _identity_material(current):
                identical += 1
            elif question_id not in mismatches:
                mismatches.append(question_id)
    rate = identical / comparisons if comparisons else 0.0
    return RepeatIdentity(
        path=path,
        comparisons=comparisons,
        identical=identical,
        identity_rate=round(rate, 6),
        mismatched_question_ids=tuple(mismatches[:32]),
        identity_digest=_identity_digest(observations),
    )


def _default_run_id(
    corpus: DocumentationCorpus,
    revisions: Mapping[DocumentationPath, PathRevision],
    repetitions: int,
) -> str:
    payload = json.dumps(
        {
            "corpus": corpus.model_dump(mode="json"),
            "revisions": {
                path.value: value.model_dump(mode="json")
                for path, value in sorted(revisions.items(), key=lambda pair: pair[0].value)
            },
            "repetitions": repetitions,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "run-" + hashlib.sha256(payload).hexdigest()[:24]


class DocumentationBenchmarkRunner:
    """Execute the fixed corpus through HTML and generated-path adapters."""

    def __init__(
        self,
        *,
        bounds: BenchmarkBounds | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.bounds = bounds or BenchmarkBounds()
        self._clock = _RunClock(clock or time.perf_counter)

    def run(
        self,
        *,
        corpus: DocumentationCorpus | None = None,
        adapters: Mapping[DocumentationPath, DocumentationAdapter],
        revisions: Mapping[DocumentationPath, PathRevision],
        repetitions: int = 2,
        reset_cache: Callable[[DocumentationPath], None] | None,
        run_id: str | None = None,
    ) -> DocumentationBenchmarkEvidence:
        """Run every corpus question on both paths, bounded and without persistence."""

        corpus = corpus or DEFAULT_DOCUMENTATION_CORPUS
        if len(corpus.questions) > self.bounds.max_questions:
            raise BenchmarkBoundExceeded(
                f"corpus has {len(corpus.questions)} questions, over the "
                f"{self.bounds.max_questions}-question bound"
            )
        if not 2 <= repetitions <= self.bounds.max_repetitions:
            raise BenchmarkBoundExceeded(
                f"repetitions must be between 2 and {self.bounds.max_repetitions}"
            )
        expected_paths = set(DocumentationPath)
        normalized_adapters = {
            DocumentationPath(path): adapter for path, adapter in adapters.items()
        }
        adapter_paths = set(normalized_adapters)
        revision_paths = {DocumentationPath(path) for path in revisions}
        if adapter_paths != expected_paths or revision_paths != expected_paths:
            raise DocumentationBenchmarkError(
                "the benchmark requires exactly html_baseline and agent_generated paths"
            )
        if reset_cache is None:
            raise DocumentationBenchmarkError(
                "reset_cache callback is required to label a clean-cache pass honestly"
            )

        all_source_ids = {
            source_id
            for question in corpus.questions
            for source_id in (
                *question.expected_source_ids,
                *question.stale_source_ids,
            )
        }
        checked_revisions: dict[DocumentationPath, PathRevision] = {}
        for raw_path, revision in revisions.items():
            path = DocumentationPath(raw_path)
            if revision.path != path:
                raise DocumentationBenchmarkError(
                    f"revision path mismatch for {path.value}"
                )
            missing_sources = sorted(all_source_ids - set(revision.source_revisions))
            if missing_sources:
                raise DocumentationBenchmarkError(
                    f"{path.value} revision is missing source revision(s): {missing_sources}"
                )
            checked_revisions[path] = revision

        observations: list[PathObservation] = []
        for path in (DocumentationPath.HTML_BASELINE, DocumentationPath.AGENT_GENERATED):
            adapter = normalized_adapters[path]
            try:
                reset_cache(path)
            except Exception as exc:  # noqa: BLE001 - add bounded context, then fail
                raise DocumentationBenchmarkError(
                    f"{path.value} cache reset failed: {type(exc).__name__}"
                ) from exc
            for repetition in range(repetitions):
                cache_mode = CacheMode.CLEAN if repetition == 0 else CacheMode.REPEAT
                for question in corpus.questions:
                    recorder = RequestRecorder(
                        limit=self.bounds.max_requests_per_answer
                    )
                    started = self._clock.read()
                    try:
                        answer = adapter.answer(question, recorder)
                    except BenchmarkBoundExceeded:
                        raise
                    except Exception as exc:  # noqa: BLE001 - add bounded context, then fail
                        raise DocumentationBenchmarkError(
                            f"{path.value}/{question.question_id} adapter failed: "
                            f"{type(exc).__name__}"
                        ) from exc
                    if not isinstance(answer, PathAnswer):
                        raise DocumentationBenchmarkError(
                            f"{path.value}/{question.question_id} adapter returned "
                            "an invalid answer type"
                        )
                    elapsed = self._clock.read() - started
                    if not math.isfinite(elapsed) or elapsed < 0:
                        raise DocumentationBenchmarkError(
                            "benchmark clock moved backwards"
                        )
                    if len(answer.text) > self.bounds.max_answer_chars:
                        raise BenchmarkBoundExceeded(
                            f"{path.value}/{question.question_id} answer exceeds "
                            f"{self.bounds.max_answer_chars} characters"
                        )
                    selected = tuple(answer.selected_source_ids)
                    if len(set(selected)) != len(selected):
                        raise DocumentationBenchmarkError(
                            f"{path.value}/{question.question_id} selected duplicate sources"
                        )
                    if any(not _is_safe_opaque_id(source_id) for source_id in selected):
                        raise DocumentationBenchmarkError(
                            f"{path.value}/{question.question_id} selected unsafe source id"
                        )
                    if not set(selected).issubset(set(recorder.source_ids)):
                        raise DocumentationBenchmarkError(
                            f"{path.value}/{question.question_id} cited a source it did not read"
                        )
                    correct = _contains_facts(answer.text, question.required_facts)
                    stale = bool(
                        set(selected) & set(question.stale_source_ids)
                    ) or _contains_facts(answer.text, question.stale_facts)
                    source_selection_correct = set(selected) == set(
                        question.expected_source_ids
                    )
                    observations.append(
                        PathObservation(
                            path=path,
                            cache_mode=cache_mode,
                            repetition=repetition,
                            question_id=question.question_id,
                            correct=correct,
                            source_selection_correct=source_selection_correct,
                            stale_answer=stale,
                            requested_source_ids=recorder.source_ids,
                            selected_source_ids=selected,
                            request_count=recorder.count,
                            elapsed_ms=round(elapsed * 1_000, 6),
                            token_usage=answer.token_usage,
                            answer_digest=_answer_digest(answer.text),
                            answer_chars=len(answer.text),
                        )
                    )

        summaries: list[PathSummary] = []
        uncertainty: list[str] = []
        for path in (DocumentationPath.HTML_BASELINE, DocumentationPath.AGENT_GENERATED):
            path_observations = [item for item in observations if item.path == path]
            clean = [item for item in path_observations if item.cache_mode == CacheMode.CLEAN]
            repeat = [item for item in path_observations if item.cache_mode == CacheMode.REPEAT]
            identity = _repeat_identity(
                path, path_observations, repetitions=repetitions
            )
            summaries.append(
                PathSummary(
                    path=path,
                    all_metrics=_metrics(path_observations),
                    clean_metrics=_metrics(clean),
                    repeat_metrics=_metrics(repeat),
                    repeat_identity=identity,
                )
            )
            if identity.identity_rate < 1.0:
                uncertainty.append(f"{path.value}: repeated semantic output is not identical")
            for metric_name in (
                "input_tokens",
                "output_tokens",
                "reasoning_tokens",
            ):
                metric = getattr(summaries[-1].all_metrics, metric_name)
                if metric.missing_count:
                    uncertainty.append(
                        f"{path.value}: {metric_name} unavailable for "
                        f"{metric.missing_count} observation(s)"
                    )

        return DocumentationBenchmarkEvidence(
            run_id=run_id or _default_run_id(corpus, checked_revisions, repetitions),
            corpus_id=corpus.corpus_id,
            corpus_version=corpus.version,
            question_ids=tuple(question.question_id for question in corpus.questions),
            path_revisions=checked_revisions,
            observations=tuple(observations),
            summaries=tuple(summaries),
            uncertainty=tuple(dict.fromkeys(uncertainty)),
        )


def _summary_by_path(
    evidence: DocumentationBenchmarkEvidence, path: DocumentationPath
) -> PathSummary:
    for summary in evidence.summaries:
        if summary.path == path:
            return summary
    raise DocumentationBenchmarkError(f"evidence has no summary for {path.value}")


def compare_documentation_paths(
    evidence: DocumentationBenchmarkEvidence,
    *,
    thresholds: RegressionThresholds | None = None,
    reviewed: bool = False,
) -> PathComparison:
    """Compare generated docs to HTML without importing external score claims."""

    thresholds = thresholds or RegressionThresholds()
    baseline = _summary_by_path(evidence, DocumentationPath.HTML_BASELINE)
    candidate = _summary_by_path(evidence, DocumentationPath.AGENT_GENERATED)
    base_metrics = baseline.all_metrics
    candidate_metrics = candidate.all_metrics

    def delta(name: str) -> float | None:
        left = getattr(base_metrics, name).mean
        right = getattr(candidate_metrics, name).mean
        return None if left is None or right is None else round(right - left, 6)

    deltas = {
        "correctness": delta("correctness"),
        "source_selection": delta("source_selection"),
        "stale_answer": delta("stale_answer"),
        "request_count": delta("request_count"),
        "elapsed_ms": delta("elapsed_ms"),
    }
    base_token_stats = (
        base_metrics.input_tokens,
        base_metrics.output_tokens,
        base_metrics.reasoning_tokens,
    )
    candidate_token_stats = (
        candidate_metrics.input_tokens,
        candidate_metrics.output_tokens,
        candidate_metrics.reasoning_tokens,
    )
    if all(stat.mean is not None for stat in (*base_token_stats, *candidate_token_stats)):
        base_tokens = sum(stat.mean or 0.0 for stat in base_token_stats)
        candidate_tokens = sum(stat.mean or 0.0 for stat in candidate_token_stats)
        deltas["token_count"] = round(candidate_tokens - base_tokens, 6)
    else:
        deltas["token_count"] = None

    violations: list[str] = []
    if deltas["correctness"] is None:
        violations.append("correctness metric unavailable")
    elif deltas["correctness"] < -thresholds.max_correctness_drop:
        violations.append("correctness regression")
    if deltas["source_selection"] is None:
        violations.append("source-selection metric unavailable")
    elif deltas["source_selection"] < -thresholds.max_source_selection_drop:
        violations.append("source-selection regression")
    if deltas["stale_answer"] is None:
        violations.append("stale-answer metric unavailable")
    elif deltas["stale_answer"] > thresholds.max_stale_answer_increase:
        violations.append("stale-answer regression")
    if deltas["request_count"] is not None and deltas["request_count"] > thresholds.max_request_count_increase:
        violations.append("request-count regression")
    if deltas["elapsed_ms"] is not None and deltas["elapsed_ms"] > thresholds.max_elapsed_ms_increase:
        violations.append("elapsed-time regression")
    if deltas["token_count"] is not None and deltas["token_count"] > thresholds.max_token_count_increase:
        violations.append("token-count regression")

    latency_improved = (
        deltas["elapsed_ms"] is not None and deltas["elapsed_ms"] < 0
    )
    tokens_improved = (
        deltas["token_count"] is not None and deltas["token_count"] < 0
    )
    efficiency_improved = latency_improved or tokens_improved
    if not efficiency_improved:
        violations.append("no measured latency or token improvement")
    if candidate.repeat_identity.identity_rate < thresholds.min_repeat_identity:
        violations.append("candidate repeat identity below threshold")

    uncertainty = list(evidence.uncertainty)
    if deltas["token_count"] is None:
        uncertainty.append("combined token count unavailable; latency is the only efficiency axis")
    promotion_ready = not violations and reviewed
    if not reviewed:
        uncertainty.append("efficiency improvement has not been explicitly reviewed")

    return PathComparison(
        baseline_path=DocumentationPath.HTML_BASELINE,
        candidate_path=DocumentationPath.AGENT_GENERATED,
        deltas=deltas,
        violations=tuple(dict.fromkeys(violations)),
        uncertainty=tuple(dict.fromkeys(uncertainty)),
        efficiency_improved=efficiency_improved,
        reviewed=reviewed,
        promotion_ready=promotion_ready,
    )


DEFAULT_DOCUMENTATION_CORPUS = DocumentationCorpus(
    corpus_id="agent-documentation",
    version="2026.08.19.v1",
    questions=(
        DocumentationQuestion(
            question_id="framework.authority",
            project_type=ProjectType.FRAMEWORK,
            prompt="Which file is the authoritative contributor and agent working contract?",
            required_facts=("AGENTS.md",),
            expected_source_ids=("framework:agents-md",),
            stale_source_ids=("framework:legacy-readme",),
            stale_facts=("legacy readme only",),
        ),
        DocumentationQuestion(
            question_id="framework.discovery",
            project_type=ProjectType.FRAMEWORK,
            prompt="Where should an agent start discovering the framework documentation?",
            required_facts=("docs/start-here.md", "llms.txt"),
            expected_source_ids=("framework:docs-start",),
            stale_source_ids=("framework:legacy-readme",),
            stale_facts=("legacy readme only",),
        ),
        DocumentationQuestion(
            question_id="connector.acl",
            project_type=ProjectType.CONNECTOR,
            prompt="What is the default ACL posture for an unknown connector?",
            required_facts=("quarantined", "deny-by-default"),
            expected_source_ids=("connector:acl-contract",),
            stale_source_ids=("connector:legacy-acl",),
            stale_facts=("public by default",),
        ),
        DocumentationQuestion(
            question_id="connector.ingest",
            project_type=ProjectType.CONNECTOR,
            prompt="Which governed path must a connector use to apply graph mutations?",
            required_facts=("ChangeEnvelope",),
            expected_source_ids=("connector:ingest-contract",),
            stale_source_ids=("connector:legacy-ingest",),
            stale_facts=("raw graph mutation",),
        ),
        DocumentationQuestion(
            question_id="frontend.markdown",
            project_type=ProjectType.FRONTEND,
            prompt="What is the static fallback and negotiated media type for agent documentation?",
            required_facts=("index.md", "text/markdown"),
            expected_source_ids=("frontend:markdown-delivery",),
            stale_source_ids=("frontend:legacy-html",),
            stale_facts=("html only",),
        ),
        DocumentationQuestion(
            question_id="engine.authority",
            project_type=ProjectType.ENGINE,
            prompt="Which system is the authoritative knowledge-graph compute and persistence plane?",
            required_facts=("epistemic-graph", "single authority"),
            expected_source_ids=("engine:authority",),
            stale_source_ids=("engine:mirror-authority",),
            stale_facts=("mirror is authoritative",),
        ),
    ),
)
