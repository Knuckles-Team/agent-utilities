#!/usr/bin/env python3
"""Expose the exact-artifact closure check to the fast security tier.

The shared forwarder preserves the canonical script's arguments and exit status.
"""

try:
    from ._fast_tier_forward import bind_gate
except ImportError:
    from _fast_tier_forward import bind_gate

main = bind_gate(__file__, __name__)
