"""Langfuse-as-evolution-signal adapter (CONCEPT:AU-AHE.reward.langfuse-evolution-signal).

Our deployed Langfuse instance is production ground truth; this suite proves it
actually drives the AHE reward/rollout primitives, not just that the adapter
functions work in isolation:

1. :class:`~agent_utilities.knowledge_graph.research.claim_flywheel.ClaimFlywheel`'s
   REAL ``record_outcome`` reads a (mocked) Langfuse SCORE and blends it into the
   reward it persists — the flywheel/promotion reward-signal proof.
2. :class:`~agent_utilities.harness.eval_corpus.EvalCorpus`'s REAL
   ``add_from_langfuse_dataset`` → ``load_cases`` → ``run_corpus`` pulls a (mocked)
   Langfuse dataset and executes it as a rollout — the dataset-as-benchmark proof.

No live Langfuse instance is touched anywhere in this file — every test mocks at
the ``LangfuseApi`` client boundary (``LangfuseTraceBackend._get_api``) or the
``create_trace_backend`` factory, exactly like the existing
``tests/harness/test_evaluators.py`` Langfuse suite.

@pytest.mark.concept("AU-AHE.reward.langfuse-evolution-signal")
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

import pytest

from agent_utilities.core.config import config
from agent_utilities.harness import langfuse_signal
from agent_utilities.harness.eval_corpus import EvalCorpus
from agent_utilities.harness.trace_backend import LangfuseTraceBackend
from agent_utilities.knowledge_graph.research.claim_flywheel import ClaimFlywheel

pytestmark = pytest.mark.concept("AU-AHE.reward.langfuse-evolution-signal")


@pytest.fixture
def mock_langfuse_config(monkeypatch):
    """Credentials configured — the gate every adapter function checks first."""
    monkeypatch.setattr(config, "langfuse_secret_key_ref", "env://TEST_LANGFUSE_SECRET")
    monkeypatch.setattr(config, "langfuse_public_key_ref", "env://TEST_LANGFUSE_PUBLIC")


class _FakeLangfuseApi:
    """Stands in for ``langfuse_agent.api_client.LangfuseApi`` — the Langfuse HTTP
    client boundary. Shaped exactly like real ``/api/public/v3/scores`` and
    ``/api/public/dataset-items`` responses so the normalization logic in
    ``LangfuseTraceBackend`` is exercised for real."""

    def __init__(
        self,
        scores: dict[str, Any] | None = None,
        datasets: dict[str, Any] | None = None,
    ):
        self._scores = scores or {"data": []}
        self._datasets = datasets or {"data": []}
        self.scores_calls: list[dict[str, Any]] = []
        self.dataset_calls: list[dict[str, Any]] = []

    def scores_get_many(self, **kwargs: Any) -> dict[str, Any]:
        self.scores_calls.append(kwargs)
        return self._scores

    def dataset_items_list(self, **kwargs: Any) -> dict[str, Any]:
        self.dataset_calls.append(kwargs)
        return self._datasets


def _backend_with_fake_api(fake_api: _FakeLangfuseApi) -> LangfuseTraceBackend:
    backend = LangfuseTraceBackend()
    backend._api = fake_api  # bypass credential resolution; the fake IS the API
    return backend


# ---------------------------------------------------------------------------
# LangfuseTraceBackend: the two new read methods, against the fake API client
# ---------------------------------------------------------------------------


def test_backend_get_trace_reward_averages_numeric_scores():
    fake_api = _FakeLangfuseApi(
        scores={"data": [{"value": 1.0}, {"value": 0.0}, {"value": 0.5}]}
    )
    backend = _backend_with_fake_api(fake_api)
    reward = asyncio.run(backend.get_trace_reward("trace-1"))
    assert reward == pytest.approx(0.5)
    assert fake_api.scores_calls[0]["trace_id"] == "trace-1"


def test_backend_get_trace_reward_coerces_boolean_and_categorical_values():
    fake_api = _FakeLangfuseApi(
        scores={"data": [{"value": True}, {"value": "fail"}, {"value": "unusable"}]}
    )
    backend = _backend_with_fake_api(fake_api)
    reward = asyncio.run(backend.get_trace_reward("trace-1"))
    # True -> 1.0, "fail" -> 0.0, "unusable" contributes nothing (unrecognized).
    assert reward == pytest.approx(0.5)


def test_backend_get_trace_reward_none_when_no_scores():
    backend = _backend_with_fake_api(_FakeLangfuseApi(scores={"data": []}))
    assert asyncio.run(backend.get_trace_reward("trace-1")) is None


def test_backend_get_trace_reward_unsafe_trace_id_short_circuits():
    fake_api = _FakeLangfuseApi(scores={"data": [{"value": 1.0}]})
    backend = _backend_with_fake_api(fake_api)
    assert asyncio.run(backend.get_trace_reward("../etc/passwd")) is None
    assert fake_api.scores_calls == []


def test_backend_list_dataset_items_normalizes_rows():
    fake_api = _FakeLangfuseApi(
        datasets={
            "data": [
                {"id": "item-1", "input": "2+2?", "expectedOutput": "4"},
                {
                    "id": "item-2",
                    "input": "capital of France?",
                    "expectedOutput": "Paris",
                },
            ]
        }
    )
    backend = _backend_with_fake_api(fake_api)
    items = asyncio.run(backend.list_dataset_items("math-bench", limit=10))
    assert items == [
        {"id": "item-1", "input": "2+2?", "expected_output": "4"},
        {"id": "item-2", "input": "capital of France?", "expected_output": "Paris"},
    ]
    assert fake_api.dataset_calls[0]["dataset_name"] == "math-bench"


def test_backend_list_dataset_items_empty_dataset_returns_empty_list():
    backend = _backend_with_fake_api(_FakeLangfuseApi(datasets={"data": []}))
    assert asyncio.run(backend.list_dataset_items("empty-bench")) == []


# ---------------------------------------------------------------------------
# The adapter module: fetch_reward / fetch_dataset_tasks / blend_reward
# ---------------------------------------------------------------------------


@patch("agent_utilities.harness.trace_backend.create_trace_backend")
def test_fetch_reward_delegates_to_the_backend(mock_create, mock_langfuse_config):
    fake_api = _FakeLangfuseApi(scores={"data": [{"value": 0.8}]})
    mock_create.return_value = _backend_with_fake_api(fake_api)

    reward = langfuse_signal.fetch_reward(trace_id="trace-1", name="correctness")
    assert reward == pytest.approx(0.8)
    mock_create.assert_called_once_with(backend_type="langfuse")


def test_fetch_reward_none_when_langfuse_unconfigured(monkeypatch):
    monkeypatch.setattr(config, "langfuse_secret_key_ref", None)
    assert langfuse_signal.fetch_reward(trace_id="trace-1") is None


@patch("agent_utilities.harness.trace_backend.create_trace_backend")
def test_fetch_dataset_tasks_delegates_to_the_backend(
    mock_create, mock_langfuse_config
):
    fake_api = _FakeLangfuseApi(
        datasets={"data": [{"id": "i1", "input": "q", "expectedOutput": "a"}]}
    )
    mock_create.return_value = _backend_with_fake_api(fake_api)

    tasks = langfuse_signal.fetch_dataset_tasks("bench-1")
    assert tasks == [{"id": "i1", "input": "q", "expected_output": "a"}]


def test_fetch_dataset_tasks_empty_when_langfuse_unconfigured(monkeypatch):
    monkeypatch.setattr(config, "langfuse_secret_key_ref", None)
    assert langfuse_signal.fetch_dataset_tasks("bench-1") == []


def test_blend_reward_even_split():
    assert langfuse_signal.blend_reward(0.2, 1.0, weight=0.5) == pytest.approx(0.6)


def test_blend_reward_passthrough_when_no_langfuse_signal():
    assert langfuse_signal.blend_reward(0.42, None) == pytest.approx(0.42)


def test_blend_reward_clamps_to_unit_interval():
    assert langfuse_signal.blend_reward(1.0, 1.0, weight=2.0) == pytest.approx(1.0)
    assert langfuse_signal.blend_reward(0.0, 0.0, weight=-1.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# LIVE PATH 1 — ClaimFlywheel.record_outcome reads a Langfuse score
# ---------------------------------------------------------------------------


class _FlywheelStubEngine:
    """Minimal engine double — mirrors ``test_claim_flywheel.py``'s stub."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.backend = None

    def add_node(self, node_id, node_type, properties=None) -> None:
        self.nodes[node_id] = {"id": node_id, "type": node_type, **(properties or {})}

    def query_cypher(self, query, params=None):
        return []

    def by_type(self, node_type: str) -> list[dict[str, Any]]:
        return [n for n in self.nodes.values() if n["type"] == node_type]


@patch("agent_utilities.harness.trace_backend.create_trace_backend")
def test_flywheel_record_outcome_blends_a_langfuse_score_live_path(
    mock_create, mock_langfuse_config
):
    """The REAL ``ClaimFlywheel.record_outcome`` — not a fresh helper — reads a
    Langfuse score through the adapter and blends it into the persisted reward.
    Proves: "flywheel reward reads a Langfuse score"."""
    fake_api = _FakeLangfuseApi(scores={"data": [{"value": 1.0}]})
    mock_create.return_value = _backend_with_fake_api(fake_api)

    eng = _FlywheelStubEngine()
    fw = ClaimFlywheel(eng)
    fw.propose("claim:1")
    fw.validate("claim:1", True)
    fw.accept("claim:1")

    outcome = fw.record_outcome(
        "claim:1",
        reward=0.2,
        langfuse_trace_id="prod-trace-42",
        langfuse_score_name="correctness",
    )

    # blend_reward(0.2, 1.0, weight=0.5) == 0.6, not the raw internal 0.2.
    assert outcome["reward"] == pytest.approx(0.6)
    assert outcome["langfuse_reward"] == pytest.approx(1.0)
    persisted = eng.by_type("ClaimOutcome")[0]
    assert persisted["reward"] == pytest.approx(0.6)
    assert persisted["langfuse_trace_id"] == "prod-trace-42"
    assert fake_api.scores_calls[0]["trace_id"] == "prod-trace-42"
    assert fake_api.scores_calls[0]["name"] == "correctness"


def test_flywheel_record_outcome_unchanged_without_langfuse_trace_id():
    """Zero behavior change (the prior default): omitting ``langfuse_trace_id``
    never touches Langfuse and never blends."""
    eng = _FlywheelStubEngine()
    fw = ClaimFlywheel(eng)
    fw.propose("claim:1")
    fw.validate("claim:1", True)
    fw.accept("claim:1")

    outcome = fw.record_outcome("claim:1", reward=0.9, note="unchanged")
    assert outcome["reward"] == pytest.approx(0.9)
    assert outcome["langfuse_reward"] is None


@patch("agent_utilities.harness.trace_backend.create_trace_backend")
def test_flywheel_record_outcome_no_matching_score_keeps_internal_reward(
    mock_create, mock_langfuse_config
):
    mock_create.return_value = _backend_with_fake_api(
        _FakeLangfuseApi(scores={"data": []})
    )

    eng = _FlywheelStubEngine()
    fw = ClaimFlywheel(eng)
    fw.propose("claim:1")
    fw.validate("claim:1", True)
    fw.accept("claim:1")

    outcome = fw.record_outcome(
        "claim:1", reward=0.77, langfuse_trace_id="prod-trace-no-score"
    )
    assert outcome["reward"] == pytest.approx(0.77)
    assert outcome["langfuse_reward"] is None


def test_flywheel_record_outcome_langfuse_lookup_failure_degrades_to_internal_reward():
    """A Langfuse-side exception must never break outcome recording."""
    eng = _FlywheelStubEngine()
    fw = ClaimFlywheel(eng)
    fw.propose("claim:1")
    fw.validate("claim:1", True)
    fw.accept("claim:1")

    with patch(
        "agent_utilities.harness.langfuse_signal.fetch_reward",
        side_effect=RuntimeError("langfuse unreachable"),
    ):
        outcome = fw.record_outcome(
            "claim:1", reward=0.55, langfuse_trace_id="prod-trace-down"
        )
    assert outcome["reward"] == pytest.approx(0.55)


# ---------------------------------------------------------------------------
# LIVE PATH 2 — a rollout pulls a Langfuse dataset
# ---------------------------------------------------------------------------


@patch("agent_utilities.harness.trace_backend.create_trace_backend")
def test_eval_corpus_rollout_pulls_a_langfuse_dataset_live_path(
    mock_create, mock_langfuse_config
):
    """The REAL ``EvalCorpus`` pulls a (mocked) Langfuse dataset, ``load_cases``
    returns it, and ``run_corpus`` actually executes it as a rollout. Proves:
    "a rollout pulls a Langfuse dataset"."""
    fake_api = _FakeLangfuseApi(
        datasets={
            "data": [
                {"id": "item-1", "input": "2+2?", "expectedOutput": "4"},
                {
                    "id": "item-2",
                    "input": "capital of France?",
                    "expectedOutput": "Paris",
                },
            ]
        }
    )
    mock_create.return_value = _backend_with_fake_api(fake_api)

    corpus = EvalCorpus()
    case_ids = corpus.add_from_langfuse_dataset("geo-math-bench", tags=["smoke"])
    assert len(case_ids) == 2

    cases = corpus.load_cases()
    assert {c.query for c in cases} == {"2+2?", "capital of France?"}
    for c in cases:
        assert "from_langfuse_dataset" in c.tags
        assert c.metadata["dataset_name"] == "geo-math-bench"

    # The rollout: run every pulled task through a trivial "perfect" agent.
    results = corpus.run_corpus(actual_output_fn=lambda case: case.expected_output)
    assert len(results) == 2
    assert all(r.passed for r in results)


def test_eval_corpus_add_from_langfuse_dataset_empty_when_unconfigured(monkeypatch):
    monkeypatch.setattr(config, "langfuse_secret_key_ref", None)
    corpus = EvalCorpus()
    assert corpus.add_from_langfuse_dataset("bench-1") == []
    assert corpus.load_cases() == []
