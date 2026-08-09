"""Governance subpackage: concept-ID allocation and multi-session coordination.

This package owns the concept-ID coordination protocols.  The legacy
(:mod:`agent_utilities.governance.concept_allocator`) path arbitrates linked
worktrees on one host; :mod:`agent_utilities.governance.concept_reservation`
uses the graph's existing atomic create/CAS primitives as the authority port
for separate hosts and fails closed when the native surface or authority-owned
policy is unavailable.
"""
