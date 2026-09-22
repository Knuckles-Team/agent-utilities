#!/usr/bin/env python3
"""Bind the legacy concept-marker detector into fast contract discovery.

Execution is delegated without modifying the detector's repository-root contract.
"""

try:
    from ._fast_tier_forward import bind_gate
except ImportError:
    from _fast_tier_forward import bind_gate

main = bind_gate(__file__, __name__)
