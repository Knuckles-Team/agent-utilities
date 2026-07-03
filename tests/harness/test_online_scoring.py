"""Online-scoring pipeline over live traces (CONCEPT:AHE-3.64).

Production automation rules AND regression assertions run through ONE judge path; verdicts
persist as OnlineScore/AssertionResult nodes linked SCORED_BY the trace, and a FAILED
assertion feeds the failing trace back into the eval corpus.
"""

from __future__ import annotations

from agent_utilities.harness.online_scoring import AutomationRule, OnlineScoringSampler
from agent_utilities.harness.trace_backend import KGTraceBackend


class _FakeKG:
    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}
        self.edges: list[tuple[str, str, str]] = []

    def add_node(self, node_id: str, **props) -> None:
        self.nodes[node_id] = props

    def link_nodes(self, src: str, dst: str, rel, **_kw) -> None:
        self.edges.append((src, dst, str(rel)))


class _FakeCorpus:
    def __init__(self, cases=None) -> None:
        self._cases = cases or []
        self.added: list[dict] = []

    def load_cases(self):
        return self._cases

    def add_case(self, **kw):
        self.added.append(kw)
        return f"eval_case:{len(self.added)}"


class _Case:
    def __init__(self, cid, assertion, tags=None):
        self.id, self.assertion, self.tags = cid, assertion, tags or []
        self.query = ""
        self.expected_output = ""


def _trace(backend: KGTraceBackend, tid: str, output: str, tags=None):
    # Synthesize a finished trace: a child generation + the root (with output text).
    backend.record_event(
        trace_id=tid,
        span_id=f"{tid}:g",
        name="llm",
        is_root=False,
        kind="llm",
        model="vllm-x",
        input_tokens=10,
        output_tokens=5,
    )
    backend.record_event(
        trace_id=tid,
        span_id=f"{tid}:root",
        name="agent.run",
        is_root=True,
        tags=tags or [],
        input_text="what is 2+2?",
        output_text=output,
    )


def _judge_contains(expected_word):
    # Deterministic stand-in for the live LLM judge: pass iff the word is in the output.
    def judge(assertion, query, actual):
        ok = expected_word.lower() in (actual or "").lower()
        return (1.0 if ok else 0.0, f"contains({expected_word})={ok}")

    return judge


def test_automation_rules_write_online_scores_linked_to_trace():
    kg = _FakeKG()
    be = KGTraceBackend(backend=kg)
    _trace(be, "t1", "2 plus 2 equals 4")
    sampler = OnlineScoringSampler(
        backend=be,
        rules=[AutomationRule(dimension="correctness", criteria="answer is 4")],
        judge=_judge_contains("4"),
    )
    written = sampler.score_trace("t1")
    assert len(written) == 1 and written[0].dimension == "correctness"
    assert written[0].score == 1.0
    # persisted + SCORED_BY edge from the trace.
    assert any(p.get("type") == "online_score" for p in kg.nodes.values())
    assert any(s == "t1" and str(rel).endswith("scored_by") for s, _d, rel in kg.edges)


def test_regression_assertion_failure_feeds_corpus():
    kg = _FakeKG()
    be = KGTraceBackend(backend=kg)
    _trace(be, "t2", "the answer is five", tags=["math"])
    corpus = _FakeCorpus(cases=[_Case("c1", "answer is 4", tags=["regression"])])
    sampler = OnlineScoringSampler(
        backend=be, rules=[], eval_corpus=corpus, judge=_judge_contains("4")
    )
    written = sampler.score_trace("t2")
    ar = [w for w in written if w.type == "assertion_result"]
    assert ar and ar[0].status == "failed"
    # The failing prod trace was fed back into the corpus (regression loop).
    assert corpus.added and corpus.added[0]["metadata"]["source_trace_id"] == "t2"


def test_install_defers_scoring_via_completion_hook():
    kg = _FakeKG()
    be = KGTraceBackend(backend=kg)
    sampler = OnlineScoringSampler(
        backend=be,
        rules=[AutomationRule("correctness", "answer is 4")],
        judge=_judge_contains("4"),
    )
    sampler._pool = (
        None  # force inline scoring in the hook (deterministic for the test)
    )
    sampler.install()
    assert be.on_trace_complete is not None
    _trace(
        be, "t3", "the answer is 4"
    )  # root completion fires the hook → scores inline
    assert any(p.get("type") == "online_score" for p in kg.nodes.values())


# ── B5: sandboxed user-defined Python metrics (CONCEPT:AHE-3.67) ──
from agent_utilities.harness.online_scoring import Metric  # noqa: E402


def test_sandboxed_metric_scores_trace():
    kg = _FakeKG()
    be = KGTraceBackend(backend=kg)
    _trace(be, "m1", "a fairly long answer with several words here")
    # metric: normalized word count (capped at 1.0 by the runner clamp).
    metric = Metric(
        name="verbosity",
        source="def metric(trace):\n    return len(trace['output'].split()) / 10.0\n",
    )
    sampler = OnlineScoringSampler(backend=be, metrics=[metric])
    written = sampler.score_trace("m1")
    score_nodes = [w for w in written if w.evaluator == "metric:verbosity"]
    assert score_nodes and 0.0 < score_nodes[0].score <= 1.0


def test_sandboxed_metric_error_is_contained():
    kg = _FakeKG()
    be = KGTraceBackend(backend=kg)
    _trace(be, "m2", "x")
    bad = Metric(
        name="boom", source="def metric(trace):\n    raise RuntimeError('x')\n"
    )
    sampler = OnlineScoringSampler(backend=be, metrics=[bad])
    written = sampler.score_trace("m2")
    n = next(w for w in written if w.evaluator == "metric:boom")
    assert n.score == 0.0 and "error" in n.reasoning
