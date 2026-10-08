"""CA-28: the Loop-engine lakehouse-maintenance propose-only hook.

``state_tools.propose_lakehouse_maintenance_gap`` is what
``core.schedule_engine``'s ``lakehouse-maintenance`` schedule dispatch targets
(``debezium_lag_check``/``opensearch_reindex_staleness_check``/
``lineage_sweep`` — CA-21/24/15/25's real detection logic, stubbed today) will
call on a genuine finding, instead of inventing a second execution path: it
reuses the SAME canonical ``:Gap`` -> SpecProposal -> review lifecycle every
other discovery track files into (``research/gaps.py``'s ``submit_gap``,
exercised the same way ``tests/unit/mcp/test_state_tools_gap_lifecycle.py``
already covers for the operator-facing ``graph_loops`` MCP tool).

Deliberately propose-only: this module proves the hook has NO develop/apply
path of its own, matching ``graph_loops`` ``run``'s existing
``mine_discovery`` default-ON-but-propose-only contract.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_utilities.mcp.tools.state_tools import propose_lakehouse_maintenance_gap
from tests.unit.fleet_autonomy_fakes import verified_fleet_session
from tests.unit.work_market_fakes import FakeWorkMarket, attach_market


@pytest.fixture(autouse=True)
def _verified_session():
    """Every Gap call binds the ambient verified tenant (EG's rule)."""
    with verified_fleet_session():
        yield


def _engine() -> tuple[SimpleNamespace, FakeWorkMarket]:
    """An engine whose only surface is EG's typed Gap/work-market namespaces --
    the hook cannot write anything else."""
    engine = SimpleNamespace()
    return engine, attach_market(engine)


def test_propose_lakehouse_maintenance_gap_files_a_canonical_gap() -> None:
    eng, market = _engine()

    gap = propose_lakehouse_maintenance_gap(
        eng,
        source="lakehouse-maintenance:opensearch_reindex_staleness_check",
        statement="OpenSearch index eg.homelab.Document is 3 CDC generations behind.",
        domain="lakehouse-maintenance",
        severity=0.6,
    )

    assert gap is not None
    assert gap["source"] == "lakehouse-maintenance:opensearch_reindex_staleness_check"
    assert gap["status"] == "open"
    ((_, stored_id),) = market.gap_rows
    assert stored_id == gap["id"]


def test_propose_lakehouse_maintenance_gap_is_propose_only_no_apply_path() -> None:
    """The hook's only durable effect is the ONE ``GapUpsert`` every discovery
    track files: EG's canonical Gap plus the WorkItem EG pairs with it in the
    same transaction (the canonical gap-lifecycle's plumbing, not a second
    execution path). Nothing that would apply/execute a lakehouse change lives
    on this path -- the engine double exposes no other surface to reach."""
    eng, market = _engine()
    gap = propose_lakehouse_maintenance_gap(
        eng,
        source="lakehouse-maintenance:debezium_lag_check",
        statement="Debezium consumer for source=erp lagging 900s over SLO.",
    )
    assert gap is not None
    assert len(market.gap_rows) == 1
    assert list(market.items) == [gap["work_item_id"]]
    assert market.items[gap["work_item_id"]]["status"] == "ready"


def test_propose_lakehouse_maintenance_gap_idempotent_on_repeated_identical_finding() -> (
    None
):
    """The same finding re-detected on the next tick must not file a second
    :Gap (nor a second WorkItem) -- ``signature`` defaults to a stable hash of
    source+statement, and the repeat carries no evidence the Gap has not seen,
    so EG answers ``unchanged``."""
    eng, market = _engine()
    statement = "OpenSearch index eg.homelab.Concept is 5 CDC generations behind."

    first = propose_lakehouse_maintenance_gap(
        eng,
        source="lakehouse-maintenance:opensearch_reindex_staleness_check",
        statement=statement,
    )
    second = propose_lakehouse_maintenance_gap(
        eng,
        source="lakehouse-maintenance:opensearch_reindex_staleness_check",
        statement=statement,
    )

    assert first is not None and second is not None
    assert first["id"] == second["id"]
    assert second["revision"] == first["revision"]
    assert len(market.gap_rows) == 1 and len(market.items) == 1


def test_propose_lakehouse_maintenance_gap_blank_statement_is_a_noop() -> None:
    eng, market = _engine()
    assert (
        propose_lakehouse_maintenance_gap(
            eng, source="lakehouse-maintenance:lineage_sweep", statement="   "
        )
        is None
    )
    assert market.gap_rows == {}
