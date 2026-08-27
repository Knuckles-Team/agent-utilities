"""Characterization tests for CX-AU-10 targets in documentation_benchmark.py:

- DocumentationBenchmarkRunner.run (CCN 42)
- compare_documentation_paths (CCN 28)

Pins OBSERVED behaviour of the unmodified functions before a
complexity-reduction refactor. Must stay byte-identical across the refactor
commit.
"""

from __future__ import annotations

import pytest

from agent_utilities.harness.documentation_benchmark import (
    BenchmarkBoundExceeded,
    BenchmarkBounds,
    DocumentationBenchmarkError,
    DocumentationBenchmarkRunner,
    DocumentationCorpus,
    DocumentationPath,
    DocumentationQuestion,
    PathAnswer,
    PathRevision,
    ProjectType,
    RegressionThresholds,
    TokenUsage,
    compare_documentation_paths,
)

_PROJECT_TYPES = (
    ProjectType.FRAMEWORK,
    ProjectType.CONNECTOR,
    ProjectType.FRONTEND,
    ProjectType.ENGINE,
)


def _question(
    qid: str, *, project_type: ProjectType, stale: bool = True
) -> DocumentationQuestion:
    return DocumentationQuestion(
        question_id=qid,
        project_type=project_type,
        prompt=f"prompt for {qid}?",
        required_facts=("fact-a",),
        expected_source_ids=(f"{qid}:good",),
        stale_source_ids=(f"{qid}:stale",) if stale else (),
        stale_facts=("stale-fact",) if stale else (),
    )


def _corpus(n: int = 4, *, corpus_id: str = "fixture-corpus") -> DocumentationCorpus:
    # The corpus contract requires >= 3 distinct project types; cycle through
    # the enum so any n >= 4 still satisfies it.
    return DocumentationCorpus(
        corpus_id=corpus_id,
        version="fixture.v1",
        questions=tuple(
            _question(f"q{i}", project_type=_PROJECT_TYPES[i % len(_PROJECT_TYPES)])
            for i in range(n)
        ),
    )


def _revisions_for(
    corpus: DocumentationCorpus, path: DocumentationPath
) -> PathRevision:
    source_ids = {
        source_id
        for question in corpus.questions
        for source_id in (*question.expected_source_ids, *question.stale_source_ids)
    }
    return PathRevision(
        path=path,
        model_revision="fixture:model-1",
        tool_revision="fixture:tool-1",
        site_revision=f"fixture:site-{path.value}",
        source_revisions={sid: "fixture:source-1" for sid in source_ids},
        harness_revision="fixture:harness-1",
    )


class _Clock:
    def __init__(self, *, ticks: list[float] | None = None) -> None:
        self._ticks = ticks
        self._i = 0
        self._value = 0.0

    def __call__(self) -> float:
        if self._ticks is not None:
            v = self._ticks[min(self._i, len(self._ticks) - 1)]
            self._i += 1
            return v
        self._value += 0.001
        return self._value


class _Adapter:
    def __init__(
        self,
        *,
        stale: bool = False,
        extra_requests: int = 0,
        token_usage: TokenUsage | None = None,
        raise_on: Exception | None = None,
        return_bad_type: bool = False,
        text: str | None = None,
        selected_override: tuple[str, ...] | None = None,
        record_extra_unread: bool = False,
    ) -> None:
        self.stale = stale
        self.extra_requests = extra_requests
        self.token_usage = token_usage or TokenUsage(
            input_tokens=5, output_tokens=2, reasoning_tokens=1
        )
        self.raise_on = raise_on
        self.return_bad_type = return_bad_type
        self.text = text
        self.selected_override = selected_override
        self.record_extra_unread = record_extra_unread

    def answer(self, question, requests):
        if self.raise_on is not None:
            raise self.raise_on
        source_id = (
            question.stale_source_ids[0]
            if self.stale
            else question.expected_source_ids[0]
        )
        requests.record(source_id)
        for i in range(self.extra_requests):
            requests.record(f"{question.question_id}:extra-{i}")
        if self.return_bad_type:
            return {"not": "a PathAnswer"}
        facts = list(question.stale_facts if self.stale else question.required_facts)
        selected = self.selected_override or (source_id,)
        return PathAnswer(
            text=self.text if self.text is not None else " ".join(facts),
            selected_source_ids=selected,
            token_usage=self.token_usage,
        )


def _run(
    *,
    corpus=None,
    baseline=None,
    generated=None,
    clock=None,
    bounds=None,
    repetitions=2,
    reset_cache=lambda _path: None,
    revisions=None,
):
    corpus = corpus or _corpus()
    baseline = baseline or _Adapter()
    generated = generated or _Adapter(
        token_usage=TokenUsage(input_tokens=3, output_tokens=1, reasoning_tokens=1)
    )
    if revisions is None:
        revisions = {
            DocumentationPath.HTML_BASELINE: _revisions_for(
                corpus, DocumentationPath.HTML_BASELINE
            ),
            DocumentationPath.AGENT_GENERATED: _revisions_for(
                corpus, DocumentationPath.AGENT_GENERATED
            ),
        }
    return DocumentationBenchmarkRunner(bounds=bounds, clock=clock or _Clock()).run(
        corpus=corpus,
        adapters={
            DocumentationPath.HTML_BASELINE: baseline,
            DocumentationPath.AGENT_GENERATED: generated,
        },
        revisions=revisions,
        repetitions=repetitions,
        reset_cache=reset_cache,
    )


# ---------------------------------------------------------------------------
# DocumentationBenchmarkRunner.run
# ---------------------------------------------------------------------------


def test_golden_path_produces_expected_observation_count_and_summaries():
    evidence = _run(corpus=_corpus(4), repetitions=3)
    # 4 questions * 3 repetitions * 2 paths = 24 observations
    assert len(evidence.observations) == 24
    assert len(evidence.summaries) == 2
    assert evidence.corpus_id == "fixture-corpus"
    assert set(evidence.path_revisions) == {
        DocumentationPath.HTML_BASELINE,
        DocumentationPath.AGENT_GENERATED,
    }


def test_corpus_over_max_questions_bound_rejected():
    with pytest.raises(BenchmarkBoundExceeded, match="question bound"):
        _run(corpus=_corpus(5), bounds=BenchmarkBounds(max_questions=4))


def test_repetitions_out_of_range_rejected():
    with pytest.raises(BenchmarkBoundExceeded, match="repetitions must be"):
        _run(repetitions=1)
    with pytest.raises(BenchmarkBoundExceeded, match="repetitions must be"):
        _run(repetitions=99, bounds=BenchmarkBounds(max_repetitions=5))


def test_adapter_path_set_mismatch_rejected():
    corpus = _corpus()
    with pytest.raises(
        DocumentationBenchmarkError, match="html_baseline and agent_generated"
    ):
        DocumentationBenchmarkRunner().run(
            corpus=corpus,
            adapters={DocumentationPath.HTML_BASELINE: _Adapter()},
            revisions={
                DocumentationPath.HTML_BASELINE: _revisions_for(
                    corpus, DocumentationPath.HTML_BASELINE
                ),
                DocumentationPath.AGENT_GENERATED: _revisions_for(
                    corpus, DocumentationPath.AGENT_GENERATED
                ),
            },
            reset_cache=lambda _p: None,
        )


def test_reset_cache_none_rejected():
    with pytest.raises(
        DocumentationBenchmarkError, match="reset_cache callback is required"
    ):
        _run(reset_cache=None)


def test_revision_path_mismatch_rejected():
    corpus = _corpus()
    bad_revision = _revisions_for(corpus, DocumentationPath.HTML_BASELINE).model_copy(
        update={"path": DocumentationPath.HTML_BASELINE}
    )
    revisions = {
        # key says AGENT_GENERATED but the revision's own .path is HTML_BASELINE
        DocumentationPath.AGENT_GENERATED: bad_revision,
        DocumentationPath.HTML_BASELINE: _revisions_for(
            corpus, DocumentationPath.HTML_BASELINE
        ),
    }
    with pytest.raises(DocumentationBenchmarkError, match="revision path mismatch"):
        _run(corpus=corpus, revisions=revisions)


def test_missing_source_revision_rejected():
    corpus = _corpus()
    incomplete = _revisions_for(corpus, DocumentationPath.HTML_BASELINE).model_copy(
        update={"source_revisions": {}}
    )
    revisions = {
        DocumentationPath.HTML_BASELINE: incomplete,
        DocumentationPath.AGENT_GENERATED: _revisions_for(
            corpus, DocumentationPath.AGENT_GENERATED
        ),
    }
    with pytest.raises(DocumentationBenchmarkError, match="missing source revision"):
        _run(corpus=corpus, revisions=revisions)


def test_reset_cache_exception_wrapped():
    def boom(_path):
        raise RuntimeError("cache is on fire")

    with pytest.raises(
        DocumentationBenchmarkError, match="cache reset failed: RuntimeError"
    ):
        _run(reset_cache=boom)


def test_adapter_bound_exceeded_propagates_unwrapped():
    with pytest.raises(BenchmarkBoundExceeded, match="too many"):
        _run(baseline=_Adapter(raise_on=BenchmarkBoundExceeded("too many requests")))


def test_adapter_generic_exception_wrapped():
    with pytest.raises(DocumentationBenchmarkError, match="adapter failed: ValueError"):
        _run(baseline=_Adapter(raise_on=ValueError("boom")))


def test_adapter_invalid_return_type_rejected():
    with pytest.raises(DocumentationBenchmarkError, match="invalid answer type"):
        _run(baseline=_Adapter(return_bad_type=True))


def test_clock_moving_backwards_rejected():
    with pytest.raises(DocumentationBenchmarkError, match="clock moved backwards"):
        _run(clock=_Clock(ticks=[10.0, 1.0]))


def test_answer_exceeding_max_chars_rejected():
    with pytest.raises(BenchmarkBoundExceeded, match="exceeds"):
        _run(
            baseline=_Adapter(text="x" * 50),
            bounds=BenchmarkBounds(max_answer_chars=10),
        )


def test_duplicate_selected_sources_rejected():
    corpus = _corpus()
    q0 = corpus.questions[0]
    with pytest.raises(DocumentationBenchmarkError, match="selected duplicate sources"):
        _run(
            corpus=corpus,
            baseline=_Adapter(
                selected_override=(q0.expected_source_ids[0], q0.expected_source_ids[0])
            ),
        )


def test_unsafe_selected_source_id_rejected():
    with pytest.raises(DocumentationBenchmarkError, match="unsafe source id"):
        _run(baseline=_Adapter(selected_override=("../etc/passwd",)))


def test_selecting_unread_source_rejected():
    with pytest.raises(
        DocumentationBenchmarkError, match="cited a source it did not read"
    ):
        _run(baseline=_Adapter(selected_override=("never-requested-id",)))


def test_stale_answer_and_uncertainty_are_recorded():
    corpus = _corpus()
    evidence = _run(
        corpus=corpus,
        baseline=_Adapter(stale=True),
        repetitions=2,
    )
    baseline_obs = [
        o for o in evidence.observations if o.path == DocumentationPath.HTML_BASELINE
    ]
    assert all(o.stale_answer for o in baseline_obs)
    assert all(not o.source_selection_correct for o in baseline_obs)


def test_different_answer_on_repeat_lowers_identity_and_adds_uncertainty():
    class _FlakyAdapter(_Adapter):
        def __init__(self):
            super().__init__()
            self._seen: dict[str, int] = {}

        def answer(self, question, requests):
            self._seen[question.question_id] = (
                self._seen.get(question.question_id, 0) + 1
            )
            source_id = question.expected_source_ids[0]
            requests.record(source_id)
            text = " ".join(question.required_facts)
            if self._seen[question.question_id] > 1:
                text += " different"
            return PathAnswer(text=text, selected_source_ids=(source_id,))

    evidence = _run(generated=_FlakyAdapter(), repetitions=2)
    assert any(
        "repeated semantic output is not identical" in u for u in evidence.uncertainty
    )


def test_missing_token_metric_adds_uncertainty():
    evidence = _run(
        baseline=_Adapter(token_usage=TokenUsage()),  # all None
    )
    assert any("unavailable for" in u for u in evidence.uncertainty)


# ---------------------------------------------------------------------------
# compare_documentation_paths
# ---------------------------------------------------------------------------


def _evidence_with(*, baseline=None, generated=None, reviewed_corpus_size=4):
    corpus = _corpus(reviewed_corpus_size)
    return _run(corpus=corpus, baseline=baseline, generated=generated)


def test_comparison_promotion_ready_when_improved_and_reviewed():
    evidence = _evidence_with(
        baseline=_Adapter(extra_requests=3),
        generated=_Adapter(
            token_usage=TokenUsage(input_tokens=1, output_tokens=1, reasoning_tokens=1)
        ),
    )
    comparison = compare_documentation_paths(evidence, reviewed=True)
    assert comparison.efficiency_improved is True
    assert comparison.promotion_ready is True
    assert comparison.violations == ()


def test_comparison_not_reviewed_adds_uncertainty_and_blocks_promotion():
    evidence = _evidence_with()
    comparison = compare_documentation_paths(evidence, reviewed=False)
    assert comparison.promotion_ready is False
    assert any("has not been explicitly reviewed" in u for u in comparison.uncertainty)


def test_comparison_correctness_regression_is_a_violation():
    evidence = _evidence_with(
        generated=_Adapter(stale=True)
    )  # wrong facts -> incorrect
    comparison = compare_documentation_paths(evidence, reviewed=True)
    assert "correctness regression" in comparison.violations
    assert comparison.promotion_ready is False


def test_comparison_no_efficiency_improvement_is_a_violation():
    same_tokens = TokenUsage(input_tokens=5, output_tokens=5, reasoning_tokens=5)
    evidence = _evidence_with(
        baseline=_Adapter(token_usage=same_tokens),
        generated=_Adapter(token_usage=same_tokens),
    )
    comparison = compare_documentation_paths(
        evidence,
        reviewed=True,
        thresholds=RegressionThresholds(
            max_elapsed_ms_increase=10_000, max_token_count_increase=10_000
        ),
    )
    assert "no measured latency or token improvement" in comparison.violations


def test_comparison_repeat_identity_below_threshold_is_a_violation():
    class _FlakyAdapter(_Adapter):
        def __init__(self):
            super().__init__()
            self._seen: dict[str, int] = {}

        def answer(self, question, requests):
            self._seen[question.question_id] = (
                self._seen.get(question.question_id, 0) + 1
            )
            source_id = question.expected_source_ids[0]
            requests.record(source_id)
            text = " ".join(question.required_facts)
            if self._seen[question.question_id] > 1:
                text += " different"
            return PathAnswer(text=text, selected_source_ids=(source_id,))

    evidence = _evidence_with(generated=_FlakyAdapter())
    comparison = compare_documentation_paths(evidence, reviewed=True)
    assert "candidate repeat identity below threshold" in comparison.violations


def test_comparison_missing_token_metrics_marks_token_delta_none():
    evidence = _evidence_with(baseline=_Adapter(token_usage=TokenUsage()))
    comparison = compare_documentation_paths(evidence, reviewed=True)
    assert comparison.deltas["token_count"] is None
    assert any("combined token count unavailable" in u for u in comparison.uncertainty)
