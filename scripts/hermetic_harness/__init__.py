"""GOC-38 hermetic test collection, runner control, and evidence-envelope harness.

CONCEPT:AU-GOC.harness.evidence-envelope

This package is the **interface freeze** for GOC-39 (agent-utilities full-suite
green) and GOC-40 (epistemic-graph full-suite green): those lanes consume the
manifest/envelope schemas and the ``build_envelope``/``compute_verdict`` contract
here rather than inventing a parallel evidence format. See
``plans/graph-os-completion-program/lanes/GOC-38-hermetic-test-evidence-harness.md``
for the design this implements, and
``plans/graph-os-completion-program/decisions/GOC-PG-GREEN-TIER-AUTHORIZATION-2026-08-16.md``
for the authorization this was built under.

Design principle: **falsifiability**. An envelope must let a reader distinguish
a real pass from a vacuous one (zero tests collected, wrong interpreter, a
truncated capture with no digest, or contamination that invalidates the
attempt) without re-running anything. ``compute_verdict`` is the single place
that decides ``green`` -- no caller may treat an absent field, a non-zero exit
code alone, or a "PASSED" adapter summary as sufficient.

Submodules:

* ``manifest`` -- build/validate the run manifest (candidate, selection,
  digests, resource limits, temp root).
* ``launcher`` -- process-group launcher: deadline, grace interval, group
  kill, survivor check. Never relies on signal-based timeouts alone.
* ``envelope`` -- build/validate the evidence envelope and compute the
  falsifiable verdict.
* ``pytest_adapter`` -- first concrete adapter (GOC-38-W04); normalizes a
  pytest invocation into the shared envelope while preserving native evidence
  (collection count via an authoritative ``--collect-only`` pass, terminal
  summary counts, raw stdout/stderr).
"""

from __future__ import annotations

__all__ = ["SCHEMA_DIR"]

from pathlib import Path

SCHEMA_DIR = Path(__file__).parent / "schemas"
