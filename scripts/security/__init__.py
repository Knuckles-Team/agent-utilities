"""Repo-invariant contract checks discovered by the merge queue's fast gate.

CONCEPT:AU-OS.governance.tiered-merge-gate — every ``check_*.py`` script here is
auto-discovered by ``agent_utilities.governance.merge_queue``'s
``CONTRACT_CHECK_GLOB`` and run, differentially against the base ref, inside the
merge queue's fast tier. This is the *only* mechanism that runs them at merge
time; adding a new invariant is a new script here, not a change to
``merge_queue.py``.
"""
