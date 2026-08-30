#!/usr/bin/env python3
"""Fast-tier forwarder: one engine-native WorkItem lifecycle authority (D-MQ2-4).

Delegates, unmodified, to ``scripts/check_native_work_item_boundary.py`` —
dark under the merge queue (D-ORC-5's audit): wired into neither
``.github/workflows/guardrails.yml`` nor ``.pre-commit-config.yaml``, and
``CONTRACT_CHECK_GLOB`` only discovers ``scripts/security/check_*.py``, one
directory up from where the canonical script lives — the SAME gap its
sibling ``check_native_change_envelope_boundary.py`` already had a
forwarder for; this one was simply missed. See
``scripts/security/_fast_tier_forward.py`` for why a forwarder lives here
instead of moving or duplicating that script. Currently green; measured
~28s for a full-package AST walk on a loaded host, comfortably inside the
forwarder's 55s subprocess timeout (itself under
``merge_queue.py``'s 60s ``CONTRACT_CHECK_BUDGET_SECONDS``) — safe to
broaden discovery immediately per D-MW-9's caution against reddening every
merge.

Usage:
  python3 scripts/security/check_native_work_item_boundary_gate.py
  python3 scripts/security/check_native_work_item_boundary_gate.py --repository-root DIR
  python3 scripts/security/check_native_work_item_boundary_gate.py --self-check

Exit semantics follow the shared forwarder.
"""

try:
    from ._fast_tier_forward import bind_gate
except ImportError:
    from _fast_tier_forward import bind_gate

main = bind_gate(__file__, __name__)
