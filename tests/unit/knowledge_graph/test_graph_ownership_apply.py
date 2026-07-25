"""Tests for the W2.8 grant-APPLICATION half (CONCEPT:AU-KG.audit.graph-ownership-disposition).

Covers:
- plan_grants: auto-apply-UNAMBIGUOUS / hold-ambiguous filtering (the program
  decision), and already-covered graphs never re-planned
- rollback_for / rollback_to_json: the exact inverse op list
- apply_plan dry-run (default): NEVER calls the client, matches rollback_for
- apply_plan live-mode: idempotent add_role, stops at first failure, and —
  the key correctness property — a partial failure still returns/carries
  exactly what succeeded so far (never silently strands an un-rollbackable
  partial mutation)
- LiveRbacAdminClient normalizes every failure to GrantApplicationError
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.maintenance import graph_ownership as go
from agent_utilities.knowledge_graph.maintenance import graph_ownership_apply as goa

# ---------------------------------------------------------------------------
# plan_grants
# ---------------------------------------------------------------------------


def _disposition(
    graph: str,
    *,
    status: str,
    covered: bool,
    recommended: tuple[go.RecommendedGrant, ...] = (),
) -> go.OwnershipDisposition:
    grants = (
        (go.GrantRecord(role="r", resource="All", action="Read", effect="Allow"),)
        if covered
        else ()
    )
    return go.OwnershipDisposition(
        graph=graph,
        status=status,
        owner="someone",
        reasons=("test",),
        explicit_public=False,
        existing_grants=grants,
        recommended_grants=recommended,
        sampled_node_count=0,
        ledger_op_count=None,
    )


def _grant(graph: str, action: str) -> go.RecommendedGrant:
    return go.RecommendedGrant(
        role="owner:someone", resource={"Graph": graph}, action=action, effect="Allow"
    )


def _report(dispositions: list[go.OwnershipDisposition]) -> go.OwnershipReport:
    return go.OwnershipReport(
        generated_at="2026-07-24T00:00:00+00:00",
        mode="template",
        source_note="unit test",
        node_sample_limit=50,
        dispositions=dispositions,
    )


def test_plan_grants_includes_only_unambiguous_uncovered() -> None:
    report = _report(
        [
            _disposition(
                "code_a",
                status="UNAMBIGUOUS",
                covered=False,
                recommended=(_grant("code_a", "Read"), _grant("code_a", "Write")),
            ),
            _disposition(
                "code_b",
                status="UNAMBIGUOUS",
                covered=True,  # already granted -> nothing to plan
                recommended=(),
            ),
            _disposition(
                "scratch_c",
                status="AMBIGUOUS",
                covered=False,
                recommended=(),  # compute_disposition never populates this for AMBIGUOUS
            ),
        ]
    )
    plan = goa.plan_grants(report)
    assert {(e.graph, e.action) for e in plan} == {
        ("code_a", "Read"),
        ("code_a", "Write"),
    }


def test_plan_grants_include_ambiguous_requires_explicit_opt_in() -> None:
    # Simulates a human who manually reviewed one AMBIGUOUS row and hand-built its
    # recommended grant (compute_disposition itself never does this automatically —
    # see graph_ownership.py's "do not guess" rule).
    report = _report(
        [
            _disposition(
                "scratch_c",
                status="AMBIGUOUS",
                covered=False,
                recommended=(_grant("scratch_c", "Read"),),
            ),
        ]
    )
    assert goa.plan_grants(report) == []
    forced = goa.plan_grants(report, include_ambiguous=True)
    assert [e.graph for e in forced] == ["scratch_c"]


def test_plan_grants_empty_when_nothing_needs_it() -> None:
    report = _report([_disposition("code_a", status="UNAMBIGUOUS", covered=True)])
    assert goa.plan_grants(report) == []


# ---------------------------------------------------------------------------
# rollback
# ---------------------------------------------------------------------------


def test_rollback_for_is_the_exact_inverse_of_the_plan() -> None:
    plan = [
        goa.GrantPlanEntry(
            graph="code_a",
            owner="system:code-ingestion",
            role="owner:system-code-ingestion",
            resource={"Graph": "code_a"},
            action="Read",
            effect="Allow",
        )
    ]
    rollback = goa.rollback_for(plan)
    assert rollback == [
        goa.RollbackEntry(
            role="owner:system-code-ingestion",
            resource={"Graph": "code_a"},
            action="Read",
            effect="Allow",
        )
    ]


def test_rollback_to_json_shape() -> None:
    rollback = [
        goa.RollbackEntry(
            role="r", resource={"Graph": "g"}, action="Read", effect="Allow"
        )
    ]
    assert goa.rollback_to_json(rollback) == [
        {
            "op": "remove_grant",
            "role": "r",
            "resource": {"Graph": "g"},
            "action": "Read",
            "effect": "Allow",
        }
    ]


# ---------------------------------------------------------------------------
# apply_plan — dry-run (the default; must NEVER touch the client)
# ---------------------------------------------------------------------------


def _plan_for(*graphs: str) -> list[goa.GrantPlanEntry]:
    return [
        goa.GrantPlanEntry(
            graph=g,
            owner="system:code-ingestion",
            role="owner:system-code-ingestion",
            resource={"Graph": g},
            action=action,
            effect="Allow",
        )
        for g in graphs
        for action in ("Read", "Write")
    ]


def test_apply_plan_dry_run_never_calls_the_client() -> None:
    client = goa.FixtureRbacAdminClient()
    plan = _plan_for("code_a", "code_b")
    applied, rollback = goa.apply_plan(plan, client, dry_run=True)
    assert client.calls == []
    assert len(applied) == len(plan) == len(rollback)
    assert all(a.grant_result == "DRY-RUN (not applied)" for a in applied)
    assert rollback == goa.rollback_for(plan)


# ---------------------------------------------------------------------------
# apply_plan — live mode (fixture client, never a real engine)
# ---------------------------------------------------------------------------


def test_apply_plan_creates_role_once_even_for_many_grants_on_it() -> None:
    client = goa.FixtureRbacAdminClient()
    plan = _plan_for("code_a")  # Read + Write -> same role, two add_grant calls
    applied, rollback = goa.apply_plan(plan, client, dry_run=False)
    assert len(applied) == 2
    assert applied[0].role_created is True
    assert applied[1].role_created is False  # already seen within this same run
    add_role_calls = [c for c in client.calls if c[0] == "add_role"]
    assert len(add_role_calls) == 1
    assert len(rollback) == 2


def test_apply_plan_skips_add_role_when_role_preexists() -> None:
    client = goa.FixtureRbacAdminClient(roles={"owner:system-code-ingestion"})
    plan = _plan_for("code_a")
    goa.apply_plan(plan, client, dry_run=False)
    assert [c for c in client.calls if c[0] == "add_role"] == []


class _FlakyRbacAdminClient:
    """A fixture-shaped client that fails on the Nth add_grant call (1-indexed) —
    used only to prove apply_plan's partial-failure recovery contract."""

    def __init__(self, *, fail_on_call: int) -> None:
        self.roles: set[str] = set()
        self.fail_on_call = fail_on_call
        self._grant_calls = 0
        self.calls: list[tuple[str, tuple]] = []

    def existing_roles(self) -> set[str]:
        return set(self.roles)

    def add_role(self, role: str) -> str:
        self.calls.append(("add_role", (role,)))
        self.roles.add(role)
        return "role_added"

    def add_grant(self, role: str, resource, action: str, effect: str) -> str:
        self._grant_calls += 1
        self.calls.append(("add_grant", (role, resource, action, effect)))
        if self._grant_calls == self.fail_on_call:
            raise RuntimeError("simulated engine failure")
        return "grant_added"

    def remove_grant(self, role: str, resource, action: str, effect: str) -> dict:
        self.calls.append(("remove_grant", (role, resource, action, effect)))
        return {"removed": True}


def test_apply_plan_stops_at_first_failure_and_preserves_partial_state() -> None:
    client = _FlakyRbacAdminClient(fail_on_call=3)
    plan = _plan_for("code_a", "code_b")  # 4 entries: a/Read, a/Write, b/Read, b/Write
    with pytest.raises(goa.GrantApplicationError) as excinfo:
        goa.apply_plan(plan, client, dry_run=False)

    exc = excinfo.value
    # Exactly the first 2 entries succeeded before the 3rd add_grant call failed —
    # the exception must carry that partial state, not lose it.
    assert len(exc.applied) == 2
    assert [a.entry.graph for a in exc.applied] == ["code_a", "code_a"]
    assert len(exc.rollback) == 2
    assert exc.rollback == goa.rollback_for(plan[:2])
    # add_grant was attempted a 3rd time (the failing call) but no 4th.
    assert len([c for c in client.calls if c[0] == "add_grant"]) == 3


def test_apply_plan_cause_is_preserved_on_failure() -> None:
    client = _FlakyRbacAdminClient(fail_on_call=1)
    plan = _plan_for("code_a")
    with pytest.raises(goa.GrantApplicationError) as excinfo:
        goa.apply_plan(plan, client, dry_run=False)
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert "simulated engine failure" in str(excinfo.value.__cause__)


# ---------------------------------------------------------------------------
# LiveRbacAdminClient — failure normalization
# ---------------------------------------------------------------------------


class _FailingGraphComputeEngine:
    @staticmethod
    def get_or_create(graph_name: str | None = None, **kwargs: object) -> object:
        raise RuntimeError("no engine configured")


def test_live_rbac_admin_client_wraps_every_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.core.graph_compute.GraphComputeEngine",
        _FailingGraphComputeEngine,
    )
    client = goa.LiveRbacAdminClient()
    with pytest.raises(goa.GrantApplicationError):
        client.existing_roles()
    with pytest.raises(goa.GrantApplicationError):
        client.add_role("owner:x")
    with pytest.raises(goa.GrantApplicationError):
        client.add_grant("owner:x", {"Graph": "g"}, "Read", "Allow")
    with pytest.raises(goa.GrantApplicationError):
        client.remove_grant("owner:x", {"Graph": "g"}, "Read", "Allow")


def test_resolve_rbac_admin_client_returns_live_client() -> None:
    assert isinstance(goa.resolve_rbac_admin_client(), goa.LiveRbacAdminClient)


# ---------------------------------------------------------------------------
# Fixture client sanity (the class every test above/CLI dry-run relies on)
# ---------------------------------------------------------------------------


def test_fixture_rbac_admin_client_remove_grant_reports_removed() -> None:
    client = goa.FixtureRbacAdminClient()
    client.add_grant("r", {"Graph": "g"}, "Read", "Allow")
    result = client.remove_grant("r", {"Graph": "g"}, "Read", "Allow")
    assert result == {"removed": True}
    # a second removal of the same (now-absent) grant reports False, not an error
    result_again = client.remove_grant("r", {"Graph": "g"}, "Read", "Allow")
    assert result_again == {"removed": False}
