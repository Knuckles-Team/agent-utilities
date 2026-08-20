"""NE-145 documentation benchmark contract and evidence tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_utilities.harness.documentation_benchmark import (
    DEFAULT_DOCUMENTATION_CORPUS,
    BenchmarkBoundExceeded,
    BenchmarkBounds,
    DocumentationBenchmarkError,
    DocumentationBenchmarkRunner,
    DocumentationPath,
    PathAnswer,
    PathRevision,
    TokenUsage,
    compare_documentation_paths,
)


def _revisions(path: DocumentationPath) -> PathRevision:
    source_ids = {
        source_id
        for question in DEFAULT_DOCUMENTATION_CORPUS.questions
        for source_id in (*question.expected_source_ids, *question.stale_source_ids)
    }
    return PathRevision(
        path=path,
        model_revision="fixture:model-1",
        tool_revision="fixture:tool-1",
        site_revision=f"fixture:site-{path.value}",
        source_revisions={source_id: "fixture:source-1" for source_id in source_ids},
        harness_revision="fixture:harness-1",
    )


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        self.value += 0.001
        return self.value


class _FixtureAdapter:
    def __init__(
        self,
        *,
        stale: bool = False,
        extra_requests: int = 0,
        different_on_repeat: bool = False,
        token_usage: TokenUsage | None = None,
    ) -> None:
        self.stale = stale
        self.extra_requests = extra_requests
        self.different_on_repeat = different_on_repeat
        self.token_usage = token_usage or TokenUsage(
            input_tokens=20, output_tokens=8, reasoning_tokens=4
        )
        self.calls = 0
        self._calls_by_question = {}

    def answer(self, question, requests):
        self.calls += 1
        self._calls_by_question[question.question_id] = (
            self._calls_by_question.get(question.question_id, 0) + 1
        )
        source_id = (
            question.stale_source_ids[0]
            if self.stale
            else question.expected_source_ids[0]
        )
        requests.record(source_id)
        for index in range(self.extra_requests):
            requests.record(f"fixture:extra-{index}")
        facts = list(question.required_facts)
        if self.stale and question.stale_facts:
            facts = list(question.stale_facts)
        if (
            self.different_on_repeat
            and self._calls_by_question[question.question_id] > 1
        ):
            facts.append("changed answer")
        return PathAnswer(
            text=" ".join(facts),
            selected_source_ids=(source_id,),
            token_usage=self.token_usage,
        )


def _run(*, baseline=None, generated=None, clock=None):
    baseline = baseline or _FixtureAdapter(extra_requests=3)
    generated = generated or _FixtureAdapter(
        token_usage=TokenUsage(input_tokens=10, output_tokens=4, reasoning_tokens=2)
    )
    return DocumentationBenchmarkRunner(clock=clock or _Clock()).run(
        adapters={
            DocumentationPath.HTML_BASELINE: baseline,
            DocumentationPath.AGENT_GENERATED: generated,
        },
        revisions={
            DocumentationPath.HTML_BASELINE: _revisions(
                DocumentationPath.HTML_BASELINE
            ),
            DocumentationPath.AGENT_GENERATED: _revisions(
                DocumentationPath.AGENT_GENERATED
            ),
        },
        reset_cache=lambda _path: None,
    )


def test_default_corpus_is_versioned_and_project_type_diverse() -> None:
    assert DEFAULT_DOCUMENTATION_CORPUS.version == "2026.08.19.v1"
    assert len(DEFAULT_DOCUMENTATION_CORPUS.questions) >= 4
    assert len({q.project_type for q in DEFAULT_DOCUMENTATION_CORPUS.questions}) >= 3


def test_runner_measures_both_paths_and_repeated_identity() -> None:
    evidence = _run()

    assert evidence.schema_version == "docs-agent-benchmark.v1"
    assert set(evidence.path_revisions) == set(DocumentationPath)
    assert len(evidence.observations) == len(DEFAULT_DOCUMENTATION_CORPUS.questions) * 4
    assert all(item.correct for item in evidence.observations)
    for summary in evidence.summaries:
        assert summary.repeat_identity.identity_rate == 1.0
        assert summary.all_metrics.correctness.mean == 1.0
        expected_reasoning_tokens = (
            4.0 if summary.path == DocumentationPath.HTML_BASELINE else 2.0
        )
        assert summary.all_metrics.reasoning_tokens.mean == expected_reasoning_tokens


def test_evidence_is_privacy_safe_and_persistence_is_explicit(tmp_path: Path) -> None:
    evidence = _run()
    serialized = evidence.to_json()
    assert "required fact" not in serialized
    assert "changed answer" not in serialized
    assert "https://" not in serialized
    assert "answer_digest" in serialized

    destination = tmp_path / "docs-benchmark.json"
    assert not destination.exists()
    evidence.persist(destination)
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "docs-agent-benchmark.v1"
    assert payload["observations"][0]["answer_digest"].startswith("sha256:")


def test_clean_cache_callback_is_required() -> None:
    with pytest.raises(DocumentationBenchmarkError, match="reset_cache"):
        DocumentationBenchmarkRunner().run(
            adapters={
                DocumentationPath.HTML_BASELINE: _FixtureAdapter(),
                DocumentationPath.AGENT_GENERATED: _FixtureAdapter(),
            },
            revisions={
                DocumentationPath.HTML_BASELINE: _revisions(
                    DocumentationPath.HTML_BASELINE
                ),
                DocumentationPath.AGENT_GENERATED: _revisions(
                    DocumentationPath.AGENT_GENERATED
                ),
            },
            reset_cache=None,
        )


def test_request_bound_fails_loudly() -> None:
    with pytest.raises(BenchmarkBoundExceeded, match="exceeded 2 requests"):
        DocumentationBenchmarkRunner(
            bounds=BenchmarkBounds(max_requests_per_answer=2)
        ).run(
            adapters={
                DocumentationPath.HTML_BASELINE: _FixtureAdapter(extra_requests=2),
                DocumentationPath.AGENT_GENERATED: _FixtureAdapter(),
            },
            revisions={
                DocumentationPath.HTML_BASELINE: _revisions(
                    DocumentationPath.HTML_BASELINE
                ),
                DocumentationPath.AGENT_GENERATED: _revisions(
                    DocumentationPath.AGENT_GENERATED
                ),
            },
            reset_cache=lambda _path: None,
        )


def test_stale_source_and_answer_are_measured_separately() -> None:
    evidence = _run(
        generated=_FixtureAdapter(stale=True),
    )
    generated = next(
        summary
        for summary in evidence.summaries
        if summary.path == DocumentationPath.AGENT_GENERATED
    )
    assert generated.all_metrics.stale_answer.mean == 1.0
    assert generated.all_metrics.correctness.mean == 0.0
    assert generated.all_metrics.source_selection.mean == 0.0


def test_repeat_identity_failure_is_reported() -> None:
    evidence = _run(generated=_FixtureAdapter(different_on_repeat=True))
    generated = next(
        summary
        for summary in evidence.summaries
        if summary.path == DocumentationPath.AGENT_GENERATED
    )
    assert generated.repeat_identity.identity_rate < 1.0
    assert any("agent_generated" in value for value in evidence.uncertainty)


def test_comparison_requires_reviewed_measured_improvement() -> None:
    evidence = _run()
    comparison = compare_documentation_paths(evidence, reviewed=False)
    assert comparison.efficiency_improved is True
    assert comparison.promotion_ready is False
    assert comparison.reviewed is False
    assert (
        "efficiency improvement has not been explicitly reviewed"
        in comparison.uncertainty
    )

    reviewed = compare_documentation_paths(evidence, reviewed=True)
    assert reviewed.promotion_ready is True
    assert reviewed.violations == ()


def test_missing_revision_is_rejected_before_execution() -> None:
    generated = _revisions(DocumentationPath.AGENT_GENERATED).model_copy(
        update={
            "source_revisions": {
                source_id: revision
                for source_id, revision in _revisions(
                    DocumentationPath.AGENT_GENERATED
                ).source_revisions.items()
                if source_id != "engine:authority"
            }
        }
    )
    with pytest.raises(DocumentationBenchmarkError, match="missing source revision"):
        DocumentationBenchmarkRunner().run(
            adapters={
                DocumentationPath.HTML_BASELINE: _FixtureAdapter(),
                DocumentationPath.AGENT_GENERATED: _FixtureAdapter(),
            },
            revisions={
                DocumentationPath.HTML_BASELINE: _revisions(
                    DocumentationPath.HTML_BASELINE
                ),
                DocumentationPath.AGENT_GENERATED: generated,
            },
            reset_cache=lambda _path: None,
        )
