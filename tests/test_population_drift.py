#!/usr/bin/python
"""Tests for the W1 population-drift monitor and VariantPool wiring.

CONCEPT:AHE-3.2
"""

from unittest.mock import MagicMock

import pytest

from agent_utilities.graph.population_drift import (
    PopulationDriftMonitor,
    population_spread,
    wasserstein1,
)

pytestmark = pytest.mark.concept("AHE-3.2")


# --- wasserstein1 ----------------------------------------------------------


def test_wasserstein1_identical_is_zero():
    assert wasserstein1([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0


def test_wasserstein1_shift_equals_shift():
    # A uniform shift of c moves every quantile by c → W1 == c.
    assert wasserstein1([0.0, 1.0, 2.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_wasserstein1_empty_is_zero():
    assert wasserstein1([], [1.0]) == 0.0
    assert wasserstein1([1.0], []) == 0.0


def test_wasserstein1_monotone_in_separation():
    near = wasserstein1([0.0, 0.0], [0.1, 0.1])
    far = wasserstein1([0.0, 0.0], [1.0, 1.0])
    assert far > near


# --- population_spread -----------------------------------------------------


def test_population_spread_constant_is_zero():
    assert population_spread([0.5, 0.5, 0.5]) == 0.0
    assert population_spread([0.5]) == 0.0


def test_population_spread_varied_positive():
    assert population_spread([0.0, 1.0]) > 0.0


# --- PopulationDriftMonitor ------------------------------------------------


def test_monitor_first_reading_has_no_drift():
    r = PopulationDriftMonitor().update([0.1, 0.9])
    assert r.drift is None
    assert r.population == 2
    assert r.spread > 0


def test_monitor_diverse_population_not_collapsed():
    mon = PopulationDriftMonitor(collapse_threshold=0.05, patience=2)
    for _ in range(4):
        r = mon.update([0.1, 0.5, 0.9])
    assert r.collapsed is False


def test_monitor_detects_collapse_after_patience():
    mon = PopulationDriftMonitor(collapse_threshold=0.05, patience=2)
    r1 = mon.update([0.5, 0.5, 0.5])  # low_streak 1
    assert r1.collapsed is False
    r2 = mon.update([0.5, 0.5, 0.5])  # low_streak 2 → collapsed
    assert r2.collapsed is True
    assert r2.drift == 0.0  # identical consecutive distributions


def test_monitor_reset_clears_state():
    mon = PopulationDriftMonitor(collapse_threshold=0.05, patience=1)
    mon.update([0.5, 0.5])
    mon.reset()
    r = mon.update([0.5, 0.5])
    assert r.drift is None  # prev cleared → first reading again


# --- live path: VariantPool.population_health ------------------------------


def _pool(variants_by_call):
    from agent_utilities.harness.variant_pool import VariantPool

    pool = VariantPool(MagicMock())
    calls = iter(variants_by_call)
    pool.get_variants = lambda base_id: next(calls)  # type: ignore[assignment]
    return pool


def test_population_health_reports_distributional_fields():
    pool = _pool([[{"id": "a", "fitness": 0.2}, {"id": "b", "fitness": 0.8}]])
    health = pool.population_health("base")
    assert set(health) == {"population", "spread", "drift", "collapsed", "low_streak"}
    assert health["population"] == 2
    assert health["spread"] > 0
    assert health["collapsed"] is False


def test_population_health_flags_collapse_across_calls():
    flat = [{"id": "a", "fitness": 0.5}, {"id": "b", "fitness": 0.5}]
    pool = _pool([flat, flat, flat])
    pool.population_health("base", collapse_threshold=0.05, patience=2)
    pool.population_health("base")
    third = pool.population_health("base")
    assert third["collapsed"] is True
