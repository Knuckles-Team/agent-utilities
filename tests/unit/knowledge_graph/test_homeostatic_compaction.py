#!/usr/bin/python
"""Tests for quality-budget / fidelity-gated context compaction (b7-01).

CONCEPT:AU-KG.memory.tiered-memory-caching
"""

import pytest

from agent_utilities.knowledge_graph.memory.agent_context import ContextCompactor

pytestmark = pytest.mark.concept("AU-KG.memory.tiered-memory-caching")


def _msg(words: int) -> list[dict]:
    return [{"role": "user", "content": "word " * words}]


# --- fidelity curve F(P) ----------------------------------------------------


def test_fidelity_curve():
    c = ContextCompactor(degradation_factor=1.0, quality_budget=100_000)
    assert c.fidelity(0) == 1.0
    assert c.fidelity(50_000) == pytest.approx(0.5)
    assert c.fidelity(200_000) == 0.0  # clamped, never negative


def test_record_processed_decreases_fidelity():
    c = ContextCompactor(quality_budget=1000, degradation_factor=1.0)
    assert c.fidelity() == 1.0
    f = c.record_processed(_msg(100))
    assert c._processed_tokens > 0
    assert f < 1.0


# --- should_compact: capacity gate unchanged --------------------------------


def test_capacity_gate_still_fires():
    c = ContextCompactor(max_tokens=100, auto_compaction_ratio=0.5)
    assert c.should_compact(_msg(60)) is True  # > 50-token threshold
    assert c.should_compact(_msg(1)) is False  # under capacity, no fidelity pressure


# --- should_compact: fidelity gate (homeostatic) ----------------------------


def test_fidelity_gate_dormant_until_recorded():
    # large budget so capacity won't fire; no processed tokens → fidelity 1.0 → no compaction
    c = ContextCompactor(max_tokens=1_000_000, fidelity_floor=0.5, quality_budget=100)
    assert c.should_compact(_msg(1)) is False


def test_fidelity_gate_fires_below_floor():
    c = ContextCompactor(
        max_tokens=1_000_000,
        fidelity_floor=0.5,
        degradation_factor=1.0,
        quality_budget=100,
    )
    c._processed_tokens = 60  # fidelity = 1 - 60/100 = 0.4 < 0.5
    assert c.fidelity() == pytest.approx(0.4)
    assert c.should_compact(_msg(1)) is True  # capacity fine, but fidelity gate trips


# --- divergence telemetry ---------------------------------------------------


def test_divergence_report_fields():
    c = ContextCompactor(quality_budget=1000, degradation_factor=1.0)
    c._processed_tokens = 1000
    rep = c.divergence_report(_msg(2))  # tiny current footprint vs big cumulative
    assert set(rep) == {
        "processed_tokens",
        "current_footprint",
        "fidelity",
        "below_fidelity_floor",
        "divergence",
    }
    assert rep["divergence"] > 0.9  # window far smaller than naive append
    assert rep["below_fidelity_floor"] is True  # fidelity 0.0 < floor
