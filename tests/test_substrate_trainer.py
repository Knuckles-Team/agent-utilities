#!/usr/bin/python
from __future__ import annotations

"""Tests for the substrate trainer (CONCEPT:AU-ORCH.execution.substrate-training-job-emission).

Deterministic stubs only — no GPU, no DSM, no network. Exercises the GRPO corpus
build (asserting it matches the real ``batch_normalized_advantage``), the
record-only default dispatch, an accepting dispatch (``dispatched``), a raising
dispatch (graceful ``skipped_no_substrate``), the ``FastSlowController`` TrainerFn
adapter, the ``min_group`` floor, idempotent job ids, and determinism.
"""

from types import SimpleNamespace

from agent_utilities.graph.training_signals import batch_normalized_advantage
from agent_utilities.harness.fast_slow_controller import FastSlowController, Trace
from agent_utilities.harness.substrate_trainer import (
    GrpoSample,
    SubstrateTrainer,
    TrainingJobSpec,
)


def _traces(rewards, prompt="p"):
    """Deterministic stub traces with .reward / .prompt."""
    return [
        SimpleNamespace(reward=r, prompt=f"{prompt}-{i}") for i, r in enumerate(rewards)
    ]


def test_build_corpus_uses_real_batch_normalized_advantage():
    rewards = [1.0, 2.0, 3.0, 4.0]
    trainer = SubstrateTrainer()
    corpus = trainer.build_corpus("taskA", _traces(rewards))

    expected = batch_normalized_advantage(rewards)
    assert [s.advantage for s in corpus] == expected
    assert all(isinstance(s, GrpoSample) for s in corpus)
    assert [s.reward for s in corpus] == rewards
    assert [s.prompt for s in corpus] == [f"p-{i}" for i in range(4)]
    assert all(s.task_key == "taskA" for s in corpus)


def test_train_default_dispatch_records_job():
    trainer = SubstrateTrainer()
    spec = trainer.train("taskA", _traces([1.0, 2.0, 3.0]))

    assert isinstance(spec, TrainingJobSpec)
    assert spec.status == "recorded"
    assert spec.method == "grpo"
    assert spec.n_samples == 3
    assert spec.corpus
    assert trainer.jobs() == [spec]


def test_train_dispatch_true_marks_dispatched():
    seen = []

    def dispatch(spec: TrainingJobSpec) -> bool:
        seen.append(spec.job_id)
        return True

    trainer = SubstrateTrainer(dispatch_fn=dispatch)
    spec = trainer.train("taskA", _traces([1.0, 2.0, 3.0]))

    assert spec.status == "dispatched"
    assert seen == [spec.job_id]
    assert trainer.jobs() == [spec]


def test_train_dispatch_returning_false_stays_recorded():
    trainer = SubstrateTrainer(dispatch_fn=lambda spec: False)
    spec = trainer.train("taskA", _traces([1.0, 2.0, 3.0]))
    assert spec.status == "recorded"


def test_train_dispatch_raises_is_graceful():
    def boom(spec: TrainingJobSpec) -> bool:
        raise RuntimeError("substrate unreachable")

    trainer = SubstrateTrainer(dispatch_fn=boom)
    spec = trainer.train("taskA", _traces([1.0, 2.0, 3.0]))

    # No crash; degraded gracefully and still queued for audit.
    assert spec.status == "skipped_no_substrate"
    assert trainer.jobs() == [spec]


def test_min_group_gating():
    trainer = SubstrateTrainer(min_group=3)
    spec = trainer.train("taskA", _traces([1.0, 2.0]))

    assert spec.status == "skipped_no_substrate"
    assert spec.n_samples == 2
    assert spec.corpus == []
    # Above the floor it builds and records.
    spec2 = trainer.train("taskB", _traces([1.0, 2.0, 3.0]))
    assert spec2.status == "recorded"
    assert spec2.corpus


def test_as_trainer_fn_usable_as_controller_trainer():
    trainer = SubstrateTrainer()
    controller = FastSlowController(
        lambda recent: "h",
        trainer_fn=trainer.as_trainer_fn(),
        recurrence_threshold=3,
    )
    for _ in range(3):
        controller.observe(Trace(task_key="taskA", reward=1.0, prompt="x"))
    updates = controller.slow_step()

    assert len(updates) == 1
    jobs = trainer.jobs()
    assert len(jobs) == 1
    assert jobs[0].task_key == "taskA"
    assert jobs[0].n_samples == 3
    assert jobs[0].status == "recorded"


def test_idempotent_job_id():
    trainer = SubstrateTrainer()
    a = trainer.train("taskA", _traces([1.0, 2.0, 3.0]))
    b = trainer.train("taskA", _traces([4.0, 5.0, 6.0]))
    # Same task_key + corpus size -> same id.
    assert a.job_id == b.job_id == "job-taskA-n3"
    c = trainer.train("taskA", _traces([1.0, 2.0]))
    assert c.job_id == "job-taskA-n2"


def test_determinism():
    rewards = [0.5, 1.5, 2.5, 0.0]
    s1 = SubstrateTrainer().train("taskA", _traces(rewards))
    s2 = SubstrateTrainer().train("taskA", _traces(rewards))

    assert s1.mean_advantage == s2.mean_advantage
    assert [g.advantage for g in s1.corpus] == [g.advantage for g in s2.corpus]
    assert s1.job_id == s2.job_id
