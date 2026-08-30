#!/usr/bin/env python3
"""Fast-tier forwarder: universal external-graph connector contract (D-ML-1).

Delegates, unmodified, to ``scripts/check_external_graph_contract.py``
— dark since 2026-07-28 because it was wired only into the push-triggered
``.github/workflows/guardrails.yml`` ("Universal external-graph connector contract" step). See
``scripts/security/_fast_tier_forward.py`` for why a forwarder lives here
instead of moving or duplicating that script.

Usage:
  python3 scripts/security/check_external_graph_contract_gate.py
  python3 scripts/security/check_external_graph_contract_gate.py --repository-root DIR
  python3 scripts/security/check_external_graph_contract_gate.py --self-check

Exit semantics follow the shared forwarder.
"""

try:
    from ._fast_tier_forward import bind_gate
except ImportError:
    from _fast_tier_forward import bind_gate

main = bind_gate(__file__, __name__)
