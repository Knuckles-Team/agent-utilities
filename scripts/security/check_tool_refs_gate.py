#!/usr/bin/env python3
"""Expose tool-reference integrity through the shared fast-tier adapter.

Default advisory semantics are preserved; this wrapper does not add ``--strict``.
"""

try:
    from ._fast_tier_forward import bind_gate
except ImportError:
    from _fast_tier_forward import bind_gate

main = bind_gate(__file__, __name__)
