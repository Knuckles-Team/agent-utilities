"""Unit tests for the unified ingestion profiler (CONCEPT:AU-OS.observability.ingestion-profile-report/70/71)."""

from __future__ import annotations

import json

from agent_utilities.knowledge_graph.core.ingest_profile import (
    IngestProfile,
    profile_ingest,
    record_embed_usage,
    record_llm_usage,
    record_offqueue_span,
    stage,
)


def test_stage_timing_and_record():
    p = IngestProfile(label="paper-1")
    with p.stage("read"):
        pass
    with p.stage("extract"):
        pass
    with p.stage("read"):  # re-entrant: sums into the same stage
        pass
    assert set(p.stages) == {"read", "extract"}
    assert all(v >= 0.0 for v in p.stages.values())
    p.record_llm(120, 40)
    p.record_llm(80, 10)
    p.record_embed(200)
    assert p.prompt_tokens == 200
    assert p.completion_tokens == 50
    assert p.embed_tokens == 200
    assert p.total_tokens == 450
    assert p.llm_calls == 2
    assert p.embed_calls == 1
    d = p.to_dict()
    assert d["total_tokens"] == 450
    assert d["llm_calls"] == 2
    assert set(d["stages_ms"]) == {"read", "extract"}


def test_contextvar_wrappers_accumulate_when_active():
    """The llm/embed wrappers call record_* and must land on the active profile."""
    with profile_ingest("p") as prof:
        record_llm_usage(100, 25)  # what make_llm_fn calls
        record_embed_usage(texts=["x" * 400])  # what make_embed_fn calls (estimates)
        with stage("write"):
            pass
    assert prof.prompt_tokens == 100
    assert prof.completion_tokens == 25
    assert prof.embed_tokens == 100  # 400 chars / 4 chars-per-token
    assert "write" in prof.stages


def test_inactive_is_a_safe_noop():
    """Outside any ingest, the wrappers must no-op without error (zero cost)."""
    assert IngestProfile.active() is None
    record_llm_usage(10, 5)  # no active profile → no-op
    record_embed_usage(texts=["abc"])
    with stage("read"):  # no-op contextmanager
        pass
    assert IngestProfile.active() is None


def test_profile_ingest_is_reentrant():
    """A nested activation reuses the outer profile (never double-counts by stacking)."""
    with profile_ingest("outer") as outer:
        record_llm_usage(10, 0)
        with profile_ingest("inner") as inner:
            assert inner is outer  # same object
            record_llm_usage(5, 0)
        record_llm_usage(1, 0)
    assert outer.prompt_tokens == 16
    assert outer.label == "outer"


def test_cost_is_derived_and_jsonable():
    p = IngestProfile()
    p.record_llm(1000, 1000)
    p.record_embed(1000)
    d = p.to_dict()
    # cost is a float (derived from token counts) and the record is JSON-safe.
    assert isinstance(d["cost"], float)
    import json

    assert json.loads(json.dumps(d))["total_tokens"] == 3000


# ---------------------------------------------------------------------------
# BUG-059 — record_offqueue_span is a JUSTIFIED chokepoint bypass, pinned.
# ---------------------------------------------------------------------------
#
# It writes a fresh EpistemicGraphBackend().for_graph("__control__").add_node(...)
# directly, never through IntelligenceGraphEngine._upsert_node/
# GraphComputeEngine.add_node, so it never reaches stamp_ownership. That is
# deliberate: every caller is a background maintenance pass with no
# request/actor context, the payload is pure system telemetry, and the
# whole write is already wrapped in "except Exception: pass" (never raises
# into the pass it is measuring). This test pins that the span write keeps
# working with zero actor bound.


def test_record_offqueue_span_works_with_no_actor_bound(monkeypatch):
    import contextvars

    captured: dict[str, dict] = {}

    class _FakeGraphView:
        def add_node(self, node_id, **props):
            captured[node_id] = props

    class _FakeEpistemicGraphBackend:
        def __init__(self, *a, **kw):
            pass

        def for_graph(self, graph_name):
            assert graph_name == "__control__"
            return _FakeGraphView()

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.backends.epistemic_graph_backend.EpistemicGraphBackend",
        _FakeEpistemicGraphBackend,
    )

    p = IngestProfile(label="embed-backfill")

    def isolated():
        # No actor bound anywhere in this fresh context — must not raise.
        record_offqueue_span(None, "embed_backfill", p)

    contextvars.Context().run(isolated)

    (props,) = captured.values()
    # ``record_offqueue_span`` writes a work-shaped envelope: the span carries
    # ``node_type='ProfileSpan'`` plus a JSON ``metadata`` blob, and ``kind``
    # lives INSIDE that blob — it is never a top-level property. Asserting
    # ``props["kind"]`` therefore raised ``KeyError`` against a shape the code
    # has never produced. Caught by the wave-2 checkpoint run; the test was
    # written but never executed (BUG-063 blocked its author's suite), so the
    # mismatch was invisible at authoring time.
    assert props["node_type"] == "ProfileSpan"
    assert json.loads(props["metadata"])["kind"] == "embed_backfill"
    # No governance stamp was applied — this is the pinned, deliberate gap.
    assert "_owner_id" not in props
