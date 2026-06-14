"""Recursive distillation loop (CONCEPT:AHE-3.31).

corpus -> fine-tune -> capability-gate -> promote. The trainer and model-evaluator are
external (GPU) and injected; the loop, monotone gate and promotion are pure and tested
end-to-end with stubs.
"""

from __future__ import annotations

import pytest

from agent_utilities.harness.recursive_distill import RecursiveDistiller

pytestmark = pytest.mark.concept("AHE-3.31")


class _Engine:
    def __init__(self):
        self.nodes = {}

    def add_node(self, nid, ntype, properties=None):
        self.nodes[nid] = {"id": nid, "type": ntype, **(properties or {})}

    def by_type(self, t):
        return [n for n in self.nodes.values() if n["type"] == t]

    def query_cypher(self, query, params=None):
        if "DistilledModelBaseline" in query:
            return [
                {"scores": n["scores_json"], "ts": n["recorded_at"]}
                for n in self.by_type("DistilledModelBaseline")
            ]
        return []


def _distiller(
    engine, corpus, *, scores, promoted, trainer_model="model-v1", min_rows=3
):
    return RecursiveDistiller(
        engine,
        corpus_source=lambda: corpus,
        trainer=lambda c: trainer_model,
        evaluate_model=lambda m: scores,
        promote=lambda m: promoted.append(m),
        min_rows=min_rows,
    )


class TestRecursiveDistill:
    def test_skips_when_corpus_too_small(self):
        d = _distiller(_Engine(), ["r1"], scores={"cap": 1.0}, promoted=[], min_rows=3)
        assert d.maybe_distill().status == "skipped"

    def test_bootstrap_establishes_baseline_and_promotes(self):
        eng, promoted = _Engine(), []
        d = _distiller(eng, ["r1", "r2", "r3"], scores={"cap": 0.8}, promoted=promoted)
        report = d.maybe_distill()
        assert report.status == "bootstrap" and promoted == ["model-v1"]
        assert eng.by_type("DistilledModelBaseline")

    def test_promotes_on_improvement(self):
        eng, promoted = _Engine(), []
        corpus = ["r1", "r2", "r3"]
        _distiller(
            eng, corpus, scores={"cap": 0.8}, promoted=promoted
        ).maybe_distill()  # bootstrap
        d = _distiller(
            eng,
            corpus,
            scores={"cap": 0.9},
            promoted=promoted,
            trainer_model="model-v2",
        )
        report = d.maybe_distill()
        assert report.status == "promoted" and "model-v2" in promoted

    def test_rejects_regression(self):
        eng, promoted = _Engine(), []
        corpus = ["r1", "r2", "r3"]
        _distiller(
            eng, corpus, scores={"cap": 0.9}, promoted=promoted
        ).maybe_distill()  # bootstrap @ 0.9
        d = _distiller(
            eng, corpus, scores={"cap": 0.5}, promoted=promoted, trainer_model="bad"
        )
        report = d.maybe_distill()
        assert report.status == "rejected" and report.regressions == ["cap"]
        assert "bad" not in promoted  # never promoted a regressing model
        assert eng.by_type("RecursiveDistillationResult")

    def test_trainer_unavailable_is_skipped(self):
        d = RecursiveDistiller(
            None,
            corpus_source=lambda: ["a", "b", "c"],
            trainer=lambda c: None,  # no GPU/trainer
            evaluate_model=lambda m: {"cap": 1.0},
            min_rows=2,
        )
        assert d.maybe_distill().status == "skipped"
