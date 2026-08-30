#!/usr/bin/env python3
"""Fast-tier forwarder: current-only (no retired-surface) contract (D-ML-1).

Delegates, unmodified, to ``scripts/check_current_only_contract.py``
— dark since 2026-07-28 because it was wired only into the push-triggered
``.github/workflows/guardrails.yml`` ("Current-only contract" step). See
``scripts/security/_fast_tier_forward.py`` for why a forwarder lives here
instead of moving or duplicating that script.

w5-ci-gates-local (D-W5CG-1): this forwarder was the ONE gap in an otherwise
complete set — every other ``guardrails.yml`` script already had a
``scripts/security/check_*.py`` forwarder wiring it into the merge queue's
``contract-checks`` fast-tier gate (``.mergequeue.yaml``, which already
carries a comment anticipating this exact check's ~403 (mergequeue.yaml's own comment estimated ~490; re-measured live this lane) pre-existing
"retired surface" violations under its ``compare: lines`` differential
rationale) except this one. Adding it closes that gap: the queue's
differential compare means main's pre-existing debt is not newly refused,
only a NEW retired-surface reference on a candidate is. NOT added to this
repo's per-commit pre-commit stage (unlike the other, already-clean
forwarders) — see the `contract-checks-current-only-debt` pre-commit hook's
own comment for why.

Usage:
  python3 scripts/security/check_current_only_contract_gate.py
  python3 scripts/security/check_current_only_contract_gate.py --repository-root DIR
  python3 scripts/security/check_current_only_contract_gate.py --self-check

Exit semantics follow the shared forwarder.
"""

try:
    from ._fast_tier_forward import bind_gate
except ImportError:
    from _fast_tier_forward import bind_gate

main = bind_gate(__file__, __name__)
