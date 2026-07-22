"""``LangfuseSignalProvider`` — SkillOpt's ``SkillSignalProvider`` protocol, Langfuse-
backed (CONCEPT:AU-AHE.optimization.skillopt-langfuse-signal).

Proves this actually closes the loop, not just that the class exists:

1. ``LangfuseSignalProvider.train_tasks``/``holdout_tasks`` feed a MOCKED Langfuse
   DATASET into the REAL ``skill_evolution.run_reflact_cycle`` as Rollout/Evaluate
   tasks.
2. ``LangfuseSignalProvider.score`` blends a MOCKED Langfuse SCORE into the
   candidate/incumbent's held-out score with an exact, arithmetic-checkable value
   (``blend_reward``'s weighted average) — proof the Langfuse signal, not just the
   internal check, drove the outcome.
3. ``select_skill_signal_provider`` selects ``LangfuseSignalProvider`` when a
   dataset is configured AND Langfuse credentials are configured, and falls back
   to ``InternalCorpusSignalProvider`` when either is missing.

No live Langfuse instance is touched: every test mocks at the
``create_trace_backend``/``LangfuseApi`` boundary, exactly like
``tests/unit/harness/test_langfuse_signal.py``.

@pytest.mark.concept("AU-AHE.optimization.skillopt-langfuse-signal")
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from agent_utilities.core.config import config
from agent_utilities.harness.langfuse_skill_signal import (
    LangfuseSignalProvider,
    is_langfuse_configured,
    select_skill_signal_provider,
)
from agent_utilities.harness.trace_backend import LangfuseTraceBackend
from agent_utilities.knowledge_graph.research.skill_evolution import (
    InternalCorpusSignalProvider,
    SkillSignalProvider,
    SkillTask,
    run_reflact_cycle,
)

pytestmark = pytest.mark.concept("AU-AHE.optimization.skillopt-langfuse-signal")

_INCUMBENT = "Base skill instructions.\n"


# ---------------------------------------------------------------------------
# fake Langfuse API client — shaped like the real ``/api/public/dataset-items``
# and ``/api/public/v3/scores`` responses (mirrors test_langfuse_signal.py's
# ``_FakeLangfuseApi``, extended to disambiguate the two ``scores_get_many``
# call shapes ``get_trace_reward`` (by ``trace_id``) and ``get_low_score_traces``
# (by ``operator``) both make).
# ---------------------------------------------------------------------------


class _FakeLangfuseApi:
    def __init__(
        self,
        *,
        dataset_items: dict[str, list[dict[str, Any]]] | None = None,
        reward_scores: dict[str, list[dict[str, Any]]] | None = None,
        low_score_rows: list[dict[str, Any]] | None = None,
    ) -> None:
        self._dataset_items = dataset_items or {}
        self._reward_scores = reward_scores or {}
        self._low_score_rows = low_score_rows or []
        self.scores_calls: list[dict[str, Any]] = []
        self.dataset_calls: list[dict[str, Any]] = []

    def dataset_items_list(self, **kwargs: Any) -> dict[str, Any]:
        self.dataset_calls.append(kwargs)
        name = kwargs.get("dataset_name")
        return {"data": self._dataset_items.get(name, [])}

    def scores_get_many(self, **kwargs: Any) -> dict[str, Any]:
        self.scores_calls.append(kwargs)
        if "trace_id" in kwargs:
            return {"data": self._reward_scores.get(kwargs["trace_id"], [])}
        # get_low_score_traces() path: operator/value-filtered scan.
        return {"data": self._low_score_rows}


def _backend_with_fake_api(fake_api: _FakeLangfuseApi) -> LangfuseTraceBackend:
    backend = LangfuseTraceBackend()
    backend._api = fake_api  # bypass credential resolution; the fake IS the API
    return backend


@pytest.fixture
def mock_langfuse_config(monkeypatch):
    """Credentials configured — the gate every adapter function checks first."""
    monkeypatch.setattr(config, "langfuse_secret_key_ref", "env://TEST_LANGFUSE_SECRET")
    monkeypatch.setattr(config, "langfuse_public_key_ref", "env://TEST_LANGFUSE_PUBLIC")


class _SkillEvoStubEngine:
    """Minimal engine double — same shape as ``test_skill_evolution.py``'s stub."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.backend = object()
        self.edges: list[tuple[str, str, dict[str, Any]]] = []

    def add_node(self, node_id, node_type, properties=None) -> None:
        self.nodes[node_id] = {"id": node_id, "type": node_type, **(properties or {})}

    def add_edge(self, source, target, rel_type: str = "", **properties: Any) -> None:
        self.edges.append((source, target, {"rel_type": rel_type, **properties}))

    def query_cypher(self, q, params=None):
        return []

    def by_type(self, node_type: str) -> list[dict[str, Any]]:
        return [n for n in self.nodes.values() if n["type"] == node_type]


# ---------------------------------------------------------------------------
# (a) protocol conformance
# ---------------------------------------------------------------------------


def test_provider_satisfies_the_skillsignalprovider_protocol():
    provider = LangfuseSignalProvider(train_dataset="bench")
    assert isinstance(provider, SkillSignalProvider)


# ---------------------------------------------------------------------------
# (b) train_tasks/holdout_tasks actually call fetch_dataset_tasks (grep-provable:
# the provider goes THROUGH langfuse_signal.fetch_dataset_tasks -> the backend's
# real list_dataset_items -> the fake API's dataset_items_list).
# ---------------------------------------------------------------------------


@patch("agent_utilities.harness.trace_backend.create_trace_backend")
def test_train_and_holdout_tasks_pull_the_configured_langfuse_datasets(
    mock_create, mock_langfuse_config
):
    fake_api = _FakeLangfuseApi(
        dataset_items={
            "skill-bench-train": [
                {"id": "t1", "input": "p1", "expectedOutput": "nonexistent_term"}
            ],
            "skill-bench-holdout": [
                {"id": "h1", "input": "h1", "expectedOutput": "Known failure patterns"}
            ],
        }
    )
    mock_create.return_value = _backend_with_fake_api(fake_api)

    provider = LangfuseSignalProvider(
        train_dataset="skill-bench-train", holdout_dataset="skill-bench-holdout"
    )
    train = provider.train_tasks()
    holdout = provider.holdout_tasks()

    assert [t.id for t in train] == ["t1"]
    assert train[0].prompt == "p1"
    assert train[0].metadata["expected_output"] == "nonexistent_term"
    assert [t.id for t in holdout] == ["h1"]
    assert {c["dataset_name"] for c in fake_api.dataset_calls} == {
        "skill-bench-train",
        "skill-bench-holdout",
    }


@patch("agent_utilities.harness.trace_backend.create_trace_backend")
def test_train_tasks_degrade_to_fallback_when_dataset_is_empty(
    mock_create, mock_langfuse_config
):
    mock_create.return_value = _backend_with_fake_api(_FakeLangfuseApi())
    fallback = InternalCorpusSignalProvider(
        [SkillTask(id="fallback-1", prompt="p")],
        [],
    )
    provider = LangfuseSignalProvider(train_dataset="empty-bench", fallback=fallback)
    tasks = provider.train_tasks()
    assert [t.id for t in tasks] == ["fallback-1"]


# ---------------------------------------------------------------------------
# (c) score() blends a mocked Langfuse SCORE — exact arithmetic proof
# ---------------------------------------------------------------------------


@patch("agent_utilities.harness.trace_backend.create_trace_backend")
def test_score_blends_langfuse_reward_with_internal_check(
    mock_create, mock_langfuse_config
):
    fake_api = _FakeLangfuseApi(reward_scores={"h1": [{"value": 0.4}]})
    mock_create.return_value = _backend_with_fake_api(fake_api)

    from agent_utilities.knowledge_graph.research.skill_evolution import SkillTask

    provider = LangfuseSignalProvider(train_dataset="bench", weight=0.5)
    task = SkillTask(
        id="h1",
        prompt="h1",
        metadata={"expected_output": "Known failure patterns", "trace_id": "h1"},
    )

    # Output does NOT contain the expected substring -> internal_reward=0.0;
    # blend_reward(0.0, 0.4, weight=0.5) == 0.2 -> below the 0.5 success line.
    success, score, reason = provider.score(task, "Base skill instructions.\n")
    assert score == pytest.approx(0.2)
    assert success is False
    assert reason == "langfuse_low_score"
    assert fake_api.scores_calls[0]["trace_id"] == "h1"

    # Output DOES contain it -> internal_reward=1.0;
    # blend_reward(1.0, 0.4, weight=0.5) == 0.7 -> above the line.
    success2, score2, reason2 = provider.score(
        task, "Base skill instructions.\n\n## Known failure patterns\n- x\n"
    )
    assert score2 == pytest.approx(0.7)
    assert success2 is True
    assert reason2 == ""


def test_score_passes_through_internal_reward_without_a_trace_id():
    """No ``trace_id`` metadata -> Langfuse is never queried; zero behavior change."""
    from agent_utilities.knowledge_graph.research.skill_evolution import SkillTask

    provider = LangfuseSignalProvider(train_dataset="bench")
    task = SkillTask(id="t1", prompt="p1", metadata={})
    success, score, reason = provider.score(task, "some output")
    assert success is True
    assert score == pytest.approx(1.0)
    assert reason == ""


# ---------------------------------------------------------------------------
# (d) failure_traces() actually calls fetch_low_score_traces
# ---------------------------------------------------------------------------


@patch("agent_utilities.harness.trace_backend.create_trace_backend")
def test_failure_traces_pulls_low_score_langfuse_traces(
    mock_create, mock_langfuse_config
):
    fake_api = _FakeLangfuseApi(
        low_score_rows=[
            {"traceId": "bad-1", "name": "correctness", "value": 0.1},
        ]
    )
    mock_create.return_value = _backend_with_fake_api(fake_api)

    provider = LangfuseSignalProvider(train_dataset="bench")
    traces = provider.failure_traces("skill:demo")

    assert len(traces) == 1
    assert traces[0]["trace_id"] == "bad-1"
    assert traces[0]["source"] == "langfuse"
    assert traces[0]["skill_id"] == "skill:demo"
    assert any("operator" in c for c in fake_api.scores_calls)


def test_failure_traces_degrade_to_empty_when_langfuse_unconfigured(monkeypatch):
    monkeypatch.setattr(config, "langfuse_secret_key_ref", None)
    provider = LangfuseSignalProvider(train_dataset="bench")
    assert provider.failure_traces("skill:demo") == []


# ---------------------------------------------------------------------------
# (e) LIVE PATH — the mocked Langfuse dataset + score actually drive the REAL
# ReflACT cycle (run_reflact_cycle), not a standalone helper.
# ---------------------------------------------------------------------------


@patch("agent_utilities.harness.trace_backend.create_trace_backend")
def test_langfuse_signal_drives_the_real_reflact_cycle_live_path(
    mock_create, mock_langfuse_config
):
    """A mocked Langfuse DATASET seeds Rollout/Evaluate tasks and a mocked
    Langfuse SCORE blends into the Evaluate outcome of the REAL
    ``run_reflact_cycle`` — proving the wiring reaches the cycle, not just that
    ``LangfuseSignalProvider`` computes numbers in isolation."""
    fake_api = _FakeLangfuseApi(
        dataset_items={
            "skill-bench-train": [
                {"id": "t1", "input": "p1", "expectedOutput": "nonexistent_term"}
            ],
            "skill-bench-holdout": [
                {"id": "h1", "input": "h1", "expectedOutput": "Known failure patterns"}
            ],
        },
        # h1's dataset-item id doubles as the SCORES lookup key (Foundation
        # simplification documented on ``_task_from_dataset_row``).
        reward_scores={"h1": [{"value": 0.4}]},
    )
    mock_create.return_value = _backend_with_fake_api(fake_api)

    provider = LangfuseSignalProvider(
        train_dataset="skill-bench-train",
        holdout_dataset="skill-bench-holdout",
        weight=0.5,
    )
    eng = _SkillEvoStubEngine()
    rep = run_reflact_cycle(eng, "skill:demo", _INCUMBENT, signal=provider)

    assert rep["train_n"] == 1
    assert rep["holdout_n"] == 1
    assert rep["failure_patterns"], (
        "Reflect must extract a pattern from the Langfuse-sourced train failure"
    )

    # incumbent never contains the marker -> internal 0.0, blended with the
    # mocked 0.4 Langfuse score -> 0.2. candidate DOES (default_edit_fn appends
    # the marker) -> internal 1.0, blended -> 0.7. Neither is 0.0/1.0 -- proof
    # the mocked Langfuse score, not just the internal check, produced these.
    assert rep["incumbent_score"] == pytest.approx(0.2)
    assert rep["candidate_score"] == pytest.approx(0.7)
    assert rep["gate_action"] == "accept"  # 0.7 > 0.2 strictly beats it
    assert rep["candidate_version_id"] is not None

    # scored via fetch_reward -> backend.get_trace_reward -> the fake API: t1 once
    # (train Rollout) plus h1 twice (holdout Evaluate: incumbent + candidate).
    reward_calls = [c for c in fake_api.scores_calls if "trace_id" in c]
    assert [c["trace_id"] for c in reward_calls] == ["t1", "h1", "h1"]

    # a winning candidate is persisted as a proposal (never auto-promoted by default).
    versions = eng.by_type("skill_version")
    assert len(versions) == 1
    assert versions[0]["status"] == "proposal"
    assert versions[0]["benchmark_score"] == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# (f) select_skill_signal_provider — the config/param opt-in seam
# ---------------------------------------------------------------------------


@patch("agent_utilities.harness.trace_backend.create_trace_backend")
def test_selector_picks_langfuse_when_dataset_and_credentials_are_configured(
    mock_create, mock_langfuse_config
):
    from agent_utilities.knowledge_graph.research.skill_evolution import SkillTask

    mock_create.return_value = _backend_with_fake_api(_FakeLangfuseApi())
    provider = select_skill_signal_provider(
        "skill:demo",
        [SkillTask(id="t", prompt="p")],
        [SkillTask(id="h", prompt="h")],
        train_dataset="skill-bench-train",
    )
    assert isinstance(provider, LangfuseSignalProvider)
    assert provider.train_dataset == "skill-bench-train"


def test_selector_falls_back_to_internal_when_langfuse_unconfigured(monkeypatch):
    monkeypatch.setattr(config, "langfuse_secret_key_ref", None)
    from agent_utilities.knowledge_graph.research.skill_evolution import SkillTask

    train = [SkillTask(id="t", prompt="p")]
    holdout = [SkillTask(id="h", prompt="h")]
    provider = select_skill_signal_provider(
        "skill:demo", train, holdout, train_dataset="skill-bench-train"
    )
    assert isinstance(provider, InternalCorpusSignalProvider)
    assert not isinstance(provider, LangfuseSignalProvider)
    assert provider.train_tasks() == train


def test_selector_falls_back_to_internal_when_no_dataset_configured(
    mock_langfuse_config,
):
    from agent_utilities.knowledge_graph.research.skill_evolution import SkillTask

    train = [SkillTask(id="t", prompt="p")]
    holdout = [SkillTask(id="h", prompt="h")]
    # Credentials ARE configured, but no dataset name given (arg or config) ->
    # internal stays the default. Proves Langfuse is opt-IN, not opt-out.
    provider = select_skill_signal_provider("skill:demo", train, holdout)
    assert isinstance(provider, InternalCorpusSignalProvider)


@patch("agent_utilities.harness.trace_backend.create_trace_backend")
def test_selector_uses_config_default_dataset_when_no_explicit_arg(
    mock_create, mock_langfuse_config, monkeypatch
):
    """The config-field half of "config/param": an operator can wire this
    system-wide via ``KG_SKILL_EVOLUTION_LANGFUSE_TRAIN_DATASET`` without every
    caller passing ``train_dataset=`` explicitly."""
    from agent_utilities.knowledge_graph.research.skill_evolution import SkillTask

    monkeypatch.setattr(
        config, "kg_skill_evolution_langfuse_train_dataset", "cfg-bench"
    )
    mock_create.return_value = _backend_with_fake_api(_FakeLangfuseApi())
    provider = select_skill_signal_provider(
        "skill:demo", [SkillTask(id="t", prompt="p")], [SkillTask(id="h", prompt="h")]
    )
    assert isinstance(provider, LangfuseSignalProvider)
    assert provider.train_dataset == "cfg-bench"


def test_is_langfuse_configured_reflects_config(monkeypatch):
    monkeypatch.setattr(config, "langfuse_secret_key_ref", None)
    assert is_langfuse_configured() is False
    monkeypatch.setattr(config, "langfuse_secret_key_ref", "env://X")
    assert is_langfuse_configured() is True
