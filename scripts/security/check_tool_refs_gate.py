#!/usr/bin/env python3
"""Fast-tier forwarder: tool/skill reference-integrity gate (D-MQ2-5).

Delegates, unmodified, to ``scripts/check_tool_refs.py`` — dark under the
merge queue (D-ORC-5's audit): wired into neither
``.github/workflows/guardrails.yml`` nor ``.pre-commit-config.yaml``, and
``CONTRACT_CHECK_GLOB`` only discovers ``scripts/security/check_*.py``, one
directory up from where the canonical script lives. See
``scripts/security/_fast_tier_forward.py`` for why a forwarder lives here
instead of moving or duplicating that script.

Forwards WITHOUT ``--strict``: the canonical script's default mode reports
drift as advisory output but only exits non-zero with ``--strict`` passed
(``check_tool_refs.py:152``). Currently green in default mode (fast,
self-contained) — safe to broaden discovery immediately per D-MW-9's
caution against reddening every merge. ``--strict`` is left for a future,
deliberate tightening once the existing drift this script already reports
is burned down — not silently opted into here.

Usage:
  python3 scripts/security/check_tool_refs_gate.py
  python3 scripts/security/check_tool_refs_gate.py --repository-root DIR
  python3 scripts/security/check_tool_refs_gate.py --self-check

Exit semantics follow the shared forwarder.
"""

try:
    from ._fast_tier_forward import bind_gate
except ImportError:
    from _fast_tier_forward import bind_gate

main = bind_gate(__file__, __name__)
