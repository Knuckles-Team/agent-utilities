#!/usr/bin/env python3
"""Fast-tier forwarder: mandatory ContextCompiler model boundary (D-CIM-3).

Delegates, unmodified, to ``scripts/check_context_compiler_boundary.py``
— dark since 2026-07-28 because it was wired only into the push-triggered
``.github/workflows/guardrails.yml`` ("ContextCompiler boundary gate" step).
See ``scripts/security/_fast_tier_forward.py`` for why a forwarder lives here
instead of moving or duplicating that script.

**Why this one needed work before it could be ported (D-CIM-3).** Two
problems, both fixed on this branch:

1. Four real pre-existing violations: four test files under
   ``tests/unit/capabilities/`` constructed ``pydantic_ai.Agent`` directly
   (18 call sites total across the four) instead of through
   ``create_context_agent``, the repo's sole governed constructor. Fixed by
   migrating every site to ``create_context_agent(..., default_capabilities=False)``
   (preserving each test's original capability list exactly — no default
   capability set was added, so the tests still exercise the SAME isolated
   behavior they did before), plus wrapping every live ``agent.run``/
   ``run_sync`` call in ``with use_grounding_policy("none"):`` — governed
   construction wraps the model in the mandatory ContextCompiler transport,
   which by default requires live evidence compilation before a request may
   proceed; the existing composition test in ``test_content_guardrails.py``
   already established this exact opt-out pattern for a ``FunctionModel``
   test double with no live KG behind it.
2. Runtime: profiled at ~34s over the ~3100-file scan (``agent_utilities`` +
   ``scripts`` + ``tests`` + ``examples``) — too slow to add to the merge
   queue's 180s fast tier alongside its tests. ``scripts/check_context_compiler_boundary.py``
   itself now documents the three algorithmic fixes (skip the
   canonical-file-only ``_function_ancestors`` walk for every other file;
   merge two of the three full-tree ``ast.walk`` passes into one; skip both
   walks entirely for a file that mentions none of the substrings any
   violation requires) that brought it to ~13s with an identical violation
   set (verified against the original 3-pass/4-walk implementation and a
   battery of synthetic before/after fixtures covering every violation kind
   plus the syntax-error and clean-file paths).

Usage:
  python3 scripts/security/check_context_compiler_boundary_gate.py
  python3 scripts/security/check_context_compiler_boundary_gate.py --repository-root DIR
  python3 scripts/security/check_context_compiler_boundary_gate.py --self-check

Exit semantics follow the shared forwarder.
"""

try:
    from ._fast_tier_forward import bind_gate
except ImportError:
    from _fast_tier_forward import bind_gate

main = bind_gate(__file__, __name__)
