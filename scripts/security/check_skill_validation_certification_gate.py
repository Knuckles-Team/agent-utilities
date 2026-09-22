#!/usr/bin/env python3
"""Make bundled-skill certification visible to the fast-tier runner.

All validation remains owned by the canonical certification implementation.
"""

try:
    from ._fast_tier_forward import bind_gate
except ImportError:
    from _fast_tier_forward import bind_gate

main = bind_gate(__file__, __name__)
