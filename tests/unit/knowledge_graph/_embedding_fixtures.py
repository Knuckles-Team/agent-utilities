"""Seeded unit-vector fixtures on the numpy-free numeric shim (EH-380).

``agent_utilities.numeric.xp`` returns builtin lists, not ndarrays, so the
old ``RandomState(...).randn(...).astype(...)`` fixtures raised
AttributeError at setup.
"""

from __future__ import annotations

import math

from agent_utilities.numeric import xp


def _unit(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values))
    return [value / norm for value in values] if norm > 0 else values


def random_unit_embedding(dim: int, seed: int | None) -> list[float]:
    """A reproducible random unit-norm embedding."""
    rng = xp.random.default_rng(0 if seed is None else seed)
    return _unit([float(v) for v in rng.standard_normal(dim)])


def similar_unit_embedding(base: list[float], noise: float, seed: int) -> list[float]:
    """``base`` plus seeded Gaussian ``noise``, renormalized."""
    rng = xp.random.default_rng(seed)
    jitter = rng.standard_normal(len(base))
    return _unit([b + noise * float(j) for b, j in zip(base, jitter, strict=True)])
