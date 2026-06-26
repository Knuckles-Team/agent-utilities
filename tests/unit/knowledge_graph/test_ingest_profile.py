"""Unit tests for the unified ingestion profiler (CONCEPT:OS-5.69/70/71)."""

from __future__ import annotations

from agent_utilities.knowledge_graph.core.ingest_profile import (
    IngestProfile,
    profile_ingest,
    record_embed_usage,
    record_llm_usage,
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
