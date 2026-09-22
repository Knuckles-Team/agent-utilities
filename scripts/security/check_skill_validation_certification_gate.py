#!/usr/bin/env python3
"""Fast-tier forwarder: bundled-skill certification source contract (D-MQ2-2).

Delegates, unmodified, to ``scripts/check_skill_validation_certification.py``
— dark under the merge queue (D-ORC-5's audit): wired into neither
``.github/workflows/guardrails.yml`` nor ``.config/pre-commit.yaml``, and
``CONTRACT_CHECK_GLOB`` only discovers ``scripts/security/check_*.py``, one
directory up from where the canonical script lives. See
``scripts/security/_fast_tier_forward.py`` for why a forwarder lives here
instead of moving or duplicating that script. Currently green (fast,
self-contained, no external dependencies) — safe to broaden discovery
immediately per D-MW-9's caution against reddening every merge.

Usage:
  python3 scripts/security/check_skill_validation_certification_gate.py
  python3 scripts/security/check_skill_validation_certification_gate.py --repository-root DIR
  python3 scripts/security/check_skill_validation_certification_gate.py --self-check

Exit semantics follow the shared forwarder.
"""

try:
    from ._fast_tier_forward import bind_gate
except ImportError:
    from _fast_tier_forward import bind_gate

main = bind_gate(__file__, __name__)
