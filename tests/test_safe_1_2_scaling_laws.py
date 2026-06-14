"""Multi-agent scaling-law harness (CONCEPT:SAFE-1.2).

Hold a task fixed, sweep collective size, fit capability ~ N^alpha to recover whether
adding agents helps super- or sub-linearly — instead of assuming it does.
"""

from __future__ import annotations

import pytest

from agent_utilities.harness.scaling_laws import (
    MultiAgentScalingHarness,
    ScalingLaw,
    fit_scaling_law,
)

pytestmark = pytest.mark.concept("SAFE-1.2")


class _Engine:
    def __init__(self):
        self.nodes = {}

    def add_node(self, nid, ntype, properties=None):
        self.nodes[nid] = {"id": nid, "type": ntype, **(properties or {})}


class TestFit:
    def test_recovers_sublinear_exponent(self):
        # quality = N^0.5 ⇒ alpha ≈ 0.5, sublinear regime
        law = fit_scaling_law([(n, n**0.5) for n in (1, 2, 4, 8, 16)])
        assert law is not None
        assert law.alpha == pytest.approx(0.5, abs=1e-6)
        assert law.regime == "sublinear" and law.r_squared > 0.999

    def test_recovers_superlinear(self):
        law = fit_scaling_law([(n, n**1.4) for n in (1, 2, 4, 8)])
        assert law.alpha == pytest.approx(1.4, abs=1e-6) and law.regime == "superlinear"

    def test_linear_regime(self):
        law = fit_scaling_law([(n, float(n)) for n in (1, 2, 4, 8)])
        assert law.regime == "linear" and law.alpha == pytest.approx(1.0, abs=1e-6)

    def test_flat_when_no_synergy(self):
        # homogeneous collective with zero synergy: quality constant ⇒ alpha ≈ 0
        law = fit_scaling_law([(n, 5.0) for n in (1, 2, 4, 8)])
        assert law.regime == "flat" and law.alpha == pytest.approx(0.0, abs=1e-6)

    def test_needs_two_distinct_sizes(self):
        assert fit_scaling_law([(4, 1.0)]) is None
        assert fit_scaling_law([(4, 1.0), (4, 2.0)]) is None

    def test_predict(self):
        law = fit_scaling_law([(n, float(n)) for n in (1, 2, 4)])
        assert law.predict(8) == pytest.approx(8.0, rel=1e-6)


class TestHarness:
    def test_measure_sweeps_and_persists(self):
        eng = _Engine()
        h = MultiAgentScalingHarness(eng)
        law = h.measure(lambda n: n**0.7, sizes=[1, 2, 4, 8])
        assert isinstance(law, ScalingLaw) and law.alpha == pytest.approx(0.7, abs=1e-6)
        node = [n for n in eng.nodes.values() if n["type"] == "ScalingLawMeasurement"][0]
        assert node["regime"] == law.regime

    def test_failed_sizes_skipped(self):
        def flaky(n):
            if n == 4:
                raise RuntimeError("boom")
            return float(n)

        law = MultiAgentScalingHarness().measure(flaky, sizes=[1, 2, 4, 8])
        assert law is not None  # still fits from the 3 good points
        assert 4 not in {n for n, _ in law.points}

    def test_zero_scores_excluded(self):
        law = MultiAgentScalingHarness().measure(lambda n: 0.0, sizes=[1, 2, 4])
        assert law is None  # nothing positive to fit
