"""Cross-harness GRPO co-evolution (CONCEPT:AHE-3.55/3.56)."""

from __future__ import annotations

from agent_utilities.harness.co_evolution import CrossHarnessCoEvolution, Trajectory


def _seed(co):
    # Two tasks, each rolled out under two harness versions (H0 weak, H1 strong).
    co.observe(Trajectory(task="A", harness_version="H0", model_ckpt="M0", reward=0.2))
    co.observe(Trajectory(task="A", harness_version="H1", model_ckpt="M0", reward=0.8))
    co.observe(Trajectory(task="B", harness_version="H0", model_ckpt="M0", reward=0.5))
    co.observe(Trajectory(task="B", harness_version="H1", model_ckpt="M0", reward=0.9))


def test_cross_harness_grouping_is_per_task():
    co = CrossHarnessCoEvolution()
    _seed(co)
    adv = dict(
        ((t.task, t.harness_version), a) for t, a in co.cross_harness_advantages()
    )
    # Within each task, advantage is group-relative (mean ~0 across harness versions):
    # the stronger scaffold gets positive advantage, the weaker negative.
    assert adv[("A", "H1")] > 0 > adv[("A", "H0")]
    assert adv[("B", "H1")] > 0 > adv[("B", "H0")]
    # Grouping is by task, not global: task B's higher rewards don't dominate task A.
    assert abs(adv[("A", "H0")] + adv[("A", "H1")]) < 1e-6


def test_grpo_corpus_carries_advantage():
    co = CrossHarnessCoEvolution()
    _seed(co)
    corpus = co.grpo_corpus()
    assert len(corpus) == 4
    assert {s.task_key for s in corpus} == {"A", "B"}
    assert all(hasattr(s, "advantage") for s in corpus)


def test_held_out_certification_gate():
    co = CrossHarnessCoEvolution()
    # A clearly-superhuman held-out result certifies; a borderline one does not.
    strong = co.certify_promotion([0.95] * 20, human_baseline=0.6)
    assert strong.certified
    weak = co.certify_promotion([0.61, 0.59, 0.6, 0.58, 0.62], human_baseline=0.6)
    assert not weak.certified
