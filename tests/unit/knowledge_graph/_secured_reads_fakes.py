"""Shared no-op secured_reads monkeypatches for engine_query routing tests.

Several ``engine_query`` wiring tests isolate the routing/dispatch logic
under test from the separate per-node ACL/visibility layer (covered
elsewhere, e.g. ``test_engine_query_aggregate_governance.py``) by patching
``secured_reads.scope``/``filter_rows``/``visible``/``audit_read`` to no-ops.
Share one patcher instead of repeating the monkeypatch calls per fixture.
"""

from __future__ import annotations

import pytest

__all__ = ["bypass_secured_reads"]


def bypass_secured_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent_utilities.knowledge_graph.core import secured_reads

    # D-W2T-2: secured_reads.scope() returns (query, extra_params) now.
    monkeypatch.setattr(secured_reads, "scope", lambda query, _actor: (query, {}))
    # b76116143 ("fix(kg): push authorization down instead of raising on
    # id-less rows") added the keyword-only trust_pushdown parameter every
    # real caller now passes to filter_rows/row_node_ids.
    monkeypatch.setattr(
        secured_reads,
        "filter_rows",
        lambda rows, _actor, trust_pushdown=False: rows,
    )
    monkeypatch.setattr(secured_reads, "visible", lambda rows, _actor: rows)
    monkeypatch.setattr(secured_reads, "audit_read", lambda *_args, **_kwargs: None)
