"""Tests for the W2.8 per-graph ownership/provenance pass (CONCEPT:AU-KG.audit.graph-ownership-disposition).

Covers:
- name-convention inference (tenant__/code_/agent:/team: + the unrecognized default)
- structural "explicit public" detection (commons + GraphType Commons/Global)
- grant-resource matching (All/Graph/Pattern/Label, Allow/Deny)
- disposition fusion: UNAMBIGUOUS from name alone, from a decisive sample, from a
  corroborated medium-confidence hint; AMBIGUOUS on conflict/no-signal/uncorroborated
- end-to-end report assembly + markdown rendering (live vs template banner)
- the architecture invariant: report-only by default, hard-fail when enforced
- LiveEngineCatalogClient normalizes every failure to EngineUnreachableError and
  correctly translates a working fake engine's responses
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_utilities.knowledge_graph.maintenance import graph_ownership as go

# ---------------------------------------------------------------------------
# Name-convention inference
# ---------------------------------------------------------------------------


def test_name_hint_tenant_graph() -> None:
    hint = go.infer_owner_from_name("tenant__homelab__default")
    assert hint.owner_hint == "tenant:homelab"
    assert hint.confidence == "high"


def test_name_hint_tenant_graph_double_underscore_base() -> None:
    # The exact real-world example from reports/HANDOFF-2026-07-22.md §8: the
    # commons base itself starts with "__", producing 4 underscores in a row.
    hint = go.infer_owner_from_name("tenant__homelab____commons__")
    assert hint.owner_hint == "tenant:homelab"
    assert hint.confidence == "high"


def test_name_hint_code_legacy_underscore_form() -> None:
    hint = go.infer_owner_from_name("code_agent-utilities")
    assert hint.owner_hint == "system:code-ingestion"
    assert hint.confidence == "high"
    assert "agent-utilities" in hint.basis


def test_name_hint_code_legacy_colon_form() -> None:
    hint = go.infer_owner_from_name("code:agent-utilities")
    assert hint.owner_hint == "system:code-ingestion"
    assert hint.confidence == "high"


def test_name_hint_agent_graph_is_medium_confidence() -> None:
    hint = go.infer_owner_from_name("agent:alice-graph")
    assert hint.owner_hint == "agent:alice-graph"
    assert hint.confidence == "medium"


def test_name_hint_team_graph_is_medium_confidence() -> None:
    hint = go.infer_owner_from_name("team:platform")
    assert hint.owner_hint == "team:platform"
    assert hint.confidence == "medium"


def test_name_hint_unrecognized_name() -> None:
    hint = go.infer_owner_from_name("scratch_import_2026")
    assert hint.owner_hint is None
    assert hint.confidence == "none"


# ---------------------------------------------------------------------------
# Structural "explicit public"
# ---------------------------------------------------------------------------


def test_is_structurally_public_commons_name() -> None:
    assert go.is_structurally_public("__commons__") is True


def test_is_structurally_public_graph_type_commons() -> None:
    record = go.GraphRecord(name="weird-name", graph_type="Commons")
    assert go.is_structurally_public("weird-name", record=record) is True


def test_is_structurally_public_graph_type_global() -> None:
    record = go.GraphRecord(name="weird-name", graph_type="Global")
    assert go.is_structurally_public("weird-name", record=record) is True


def test_is_not_structurally_public_agent_graph() -> None:
    record = go.GraphRecord(name="agent:alice-graph", graph_type="Agent")
    assert go.is_structurally_public("agent:alice-graph", record=record) is False


# ---------------------------------------------------------------------------
# Grant-resource matching
# ---------------------------------------------------------------------------


def test_grant_covers_graph_all_selector() -> None:
    grant = go.GrantRecord(role="r", resource="All", action="Read", effect="Allow")
    assert grant.covers_graph("anything") is True


def test_grant_covers_graph_exact_selector() -> None:
    grant = go.GrantRecord(
        role="r", resource={"Graph": "code_foo"}, action="Read", effect="Allow"
    )
    assert grant.covers_graph("code_foo") is True
    assert grant.covers_graph("code_bar") is False


def test_grant_covers_graph_pattern_selector() -> None:
    grant = go.GrantRecord(
        role="r", resource={"Pattern": "code_*"}, action="Write", effect="Allow"
    )
    assert grant.covers_graph("code_foo") is True
    assert grant.covers_graph("tenant__x__y") is False


def test_grant_label_selector_never_covers_a_graph() -> None:
    grant = go.GrantRecord(
        role="r", resource={"Label": "Person"}, action="Read", effect="Allow"
    )
    assert grant.covers_graph("code_foo") is False


def test_grant_deny_effect_never_covers() -> None:
    grant = go.GrantRecord(role="r", resource="All", action="Read", effect="Deny")
    assert grant.covers_graph("code_foo") is False


def test_grant_admin_action_never_covers() -> None:
    grant = go.GrantRecord(role="r", resource="All", action="Admin", effect="Allow")
    assert grant.covers_graph("code_foo") is False


def test_grants_covering_graph_filters_correctly() -> None:
    grants = [
        go.GrantRecord(
            role="a", resource={"Graph": "code_foo"}, action="Read", effect="Allow"
        ),
        go.GrantRecord(
            role="b", resource={"Graph": "code_bar"}, action="Read", effect="Allow"
        ),
    ]
    covering = go.grants_covering_graph(grants, "code_foo")
    assert [g.role for g in covering] == ["a"]


# ---------------------------------------------------------------------------
# Disposition fusion
# ---------------------------------------------------------------------------


def _signals(
    name: str,
    *,
    graph_type: str | None = "Agent",
    owner_id_rows: list[str] | None = None,
    source_system_rows: list[str] | None = None,
    grants: tuple[go.GrantRecord, ...] = (),
) -> go.GraphOwnershipSignals:
    record = go.GraphRecord(name=name, graph_type=graph_type)
    signals = go.GraphOwnershipSignals(
        record=record, name_hint=go.infer_owner_from_name(name), grants=grants
    )
    for value in owner_id_rows or []:
        signals.owner_id_votes[value] += 1
    for value in source_system_rows or []:
        signals.source_system_votes[value] += 1
    signals.sampled_node_count = len(owner_id_rows or []) + len(
        source_system_rows or []
    )
    return signals


def test_disposition_unambiguous_from_high_confidence_name_alone() -> None:
    d = go.compute_disposition(_signals("code_agent-utilities"))
    assert d.status == "UNAMBIGUOUS"
    assert d.owner == "system:code-ingestion"
    assert d.recommended_grants  # not already covered/public -> a grant is proposed


def test_disposition_unambiguous_corroborated_by_decisive_sample() -> None:
    signals = _signals(
        "code_agent-utilities",
        source_system_rows=["code:agent-utilities"] * 10,
    )
    d = go.compute_disposition(signals)
    assert d.status == "UNAMBIGUOUS"
    assert any("sampled source_system majority" in r for r in d.reasons)


def test_disposition_ambiguous_on_conflicting_sample() -> None:
    signals = _signals(
        "code_agent-utilities",
        source_system_rows=["code:agent-utilities"] * 5 + ["code:unrelated"] * 5,
    )
    d = go.compute_disposition(signals)
    assert d.status == "AMBIGUOUS"
    assert d.owner is None
    assert any("conflicting sampled" in r for r in d.reasons)


def test_disposition_ambiguous_no_signal_at_all() -> None:
    d = go.compute_disposition(_signals("scratch_import_2026", graph_type="Agent"))
    assert d.status == "AMBIGUOUS"
    assert d.owner is None
    assert not d.recommended_grants


def test_disposition_medium_confidence_uncorroborated_is_ambiguous() -> None:
    # "do not guess grants": a bare agent: name hint with zero sample is NOT
    # auto-appliable.
    d = go.compute_disposition(_signals("agent:cold-case"))
    assert d.status == "AMBIGUOUS"
    assert not d.recommended_grants


def test_disposition_medium_confidence_corroborated_is_unambiguous() -> None:
    signals = _signals("agent:known-writer", owner_id_rows=["known-writer"] * 3)
    d = go.compute_disposition(signals)
    assert d.status == "UNAMBIGUOUS"
    assert d.owner == "agent:known-writer"


def test_disposition_name_conflicts_with_sample_is_ambiguous() -> None:
    # tenant__ names a specific slug; a decisive but UNRELATED sampled owner_id
    # must not be silently trusted over (or blended with) the name.
    signals = _signals(
        "tenant__homelab__default",
        graph_type="Global",
        owner_id_rows=["totally-unrelated-actor"] * 10,
    )
    d = go.compute_disposition(signals)
    assert d.status == "AMBIGUOUS"
    assert any("unresolved conflict" in r for r in d.reasons)


def test_disposition_explicit_public_needs_no_grant() -> None:
    record = go.GraphRecord(name="__commons__", graph_type="Commons")
    signals = go.GraphOwnershipSignals(
        record=record, name_hint=go.infer_owner_from_name("__commons__")
    )
    d = go.compute_disposition(signals)
    assert d.status == "UNAMBIGUOUS"
    assert d.explicit_public is True
    assert d.already_covered is True
    assert d.recommended_grants == ()


def test_disposition_already_granted_yields_no_recommendation() -> None:
    existing = (
        go.GrantRecord(
            role="owner:code-ingestion",
            resource={"Graph": "code_agent-utilities"},
            action="Read",
            effect="Allow",
        ),
    )
    d = go.compute_disposition(_signals("code_agent-utilities", grants=existing))
    assert d.status == "UNAMBIGUOUS"
    assert d.already_covered is True
    assert d.recommended_grants == ()
    assert d.existing_grants == existing


def test_role_slug_is_stable_and_coarse_for_code_ingestion() -> None:
    a = go.compute_disposition(_signals("code_repo-one"))
    b = go.compute_disposition(_signals("code_repo-two"))
    # Same coarse owner -> same reusable role across many graphs (never N roles
    # for one system actor).
    assert a.recommended_grants[0].role == b.recommended_grants[0].role
    assert a.recommended_grants[0].role == "owner:system-code-ingestion"
    assert a.recommended_grants[0].resource == {"Graph": "code_repo-one"}
    assert {g.action for g in a.recommended_grants} == {"Read", "Write"}


# ---------------------------------------------------------------------------
# End-to-end report assembly + rendering
# ---------------------------------------------------------------------------


def _small_fixture_client() -> go.FixtureCatalogClient:
    graphs = [
        {"name": "__commons__", "type": "Commons", "valid": True},
        {"name": "code_agent-utilities", "type": "Agent", "valid": True},
        {"name": "scratch_import_2026", "type": "Agent", "valid": True},
        {"name": "tenant__homelab__default", "type": "Global", "valid": True},
    ]
    grants = [
        {
            "role": "owner:tenant-homelab",
            "resource": {"Graph": "tenant__homelab__default"},
            "action": "Read",
            "effect": "Allow",
        },
        {
            "role": "owner:tenant-homelab",
            "resource": {"Graph": "tenant__homelab__default"},
            "action": "Write",
            "effect": "Allow",
        },
    ]
    return go.FixtureCatalogClient(
        graphs,
        grants=grants,
        node_samples={
            "code_agent-utilities": [{"source_system": "code:agent-utilities"}] * 5
        },
        ledgers={"code_agent-utilities": ["ADD_NODE|n1", "ADD_NODE|n2"]},
    )


def test_build_ownership_report_end_to_end() -> None:
    report = go.build_ownership_report(
        _small_fixture_client(), mode="template", source_note="unit-test fixture"
    )
    names = [d.graph for d in report.dispositions]
    assert names == sorted(names)  # deterministic ordering
    by_name = {d.graph: d for d in report.dispositions}

    assert by_name["__commons__"].status == "UNAMBIGUOUS"
    assert by_name["__commons__"].explicit_public is True

    assert by_name["code_agent-utilities"].status == "UNAMBIGUOUS"
    assert by_name["code_agent-utilities"].ledger_op_count == 2

    assert by_name["scratch_import_2026"].status == "AMBIGUOUS"

    assert by_name["tenant__homelab__default"].already_covered is True
    assert by_name["tenant__homelab__default"].existing_grants

    counts = report.counts
    assert counts["total"] == 4
    assert counts["ambiguous"] == 1
    assert counts["unambiguous"] == 3


def test_render_markdown_template_mode_banner() -> None:
    report = go.build_ownership_report(
        _small_fixture_client(), mode="template", source_note="unit-test fixture"
    )
    text = go.render_markdown(report)
    assert "MODE: TEMPLATE" in text
    assert "code_agent-utilities" in text
    assert "UNAMBIGUOUS" in text and "AMBIGUOUS" in text


def test_render_markdown_live_mode_banner() -> None:
    report = go.build_ownership_report(
        _small_fixture_client(), mode="live", source_note="unit-test fixture"
    )
    text = go.render_markdown(report)
    assert "MODE: LIVE" in text


# ---------------------------------------------------------------------------
# Architecture invariant
# ---------------------------------------------------------------------------


def _disposition(graph: str, *, covered: bool) -> go.OwnershipDisposition:
    grants = (
        (go.GrantRecord(role="r", resource="All", action="Read", effect="Allow"),)
        if covered
        else ()
    )
    return go.OwnershipDisposition(
        graph=graph,
        status="UNAMBIGUOUS" if covered else "AMBIGUOUS",
        owner="someone" if covered else None,
        reasons=("test",),
        explicit_public=False,
        existing_grants=grants,
        recommended_grants=(),
        sampled_node_count=0,
        ledger_op_count=None,
    )


def test_find_invariant_violations_flags_uncovered_only() -> None:
    dispositions = [_disposition("a", covered=True), _disposition("b", covered=False)]
    violations = go.find_invariant_violations(dispositions)
    assert [v.graph for v in violations] == ["b"]


def test_check_invariant_is_report_only_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KG_GRAPH_OWNERSHIP_ENFORCED", raising=False)
    dispositions = [_disposition("b", covered=False)]
    violations = go.check_invariant(dispositions)  # no raise
    assert len(violations) == 1


def test_check_invariant_hard_fails_when_enforced_explicitly() -> None:
    dispositions = [_disposition("b", covered=False)]
    with pytest.raises(go.GraphOwnershipInvariantViolation):
        go.check_invariant(dispositions, enforced=True)


def test_check_invariant_hard_fails_when_enforced_via_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KG_GRAPH_OWNERSHIP_ENFORCED", "true")
    dispositions = [_disposition("b", covered=False)]
    with pytest.raises(go.GraphOwnershipInvariantViolation):
        go.check_invariant(dispositions)


def test_check_invariant_passes_when_everything_covered() -> None:
    dispositions = [_disposition("a", covered=True), _disposition("b", covered=True)]
    violations = go.check_invariant(dispositions, enforced=True)
    assert violations == []


def test_invariant_enforced_defaults_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KG_GRAPH_OWNERSHIP_ENFORCED", raising=False)
    assert go.invariant_enforced() is False


# ---------------------------------------------------------------------------
# LiveEngineCatalogClient — failure normalization + response translation
# ---------------------------------------------------------------------------


class _FakeQueryClient:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows
        self.last_query: str | None = None

    def cypher_read(self, query: str) -> list[dict]:
        self.last_query = query
        return self._rows


class _FakeEngineClient:
    def __init__(
        self,
        *,
        tenants_rows: list[dict] | None = None,
        rbac_policy: dict | None = None,
        node_rows: list[dict] | None = None,
        ledger_ops: list[str] | None = None,
    ) -> None:
        self.tenants = SimpleNamespace(list=lambda: tenants_rows or [])
        self.rbac = SimpleNamespace(
            list=lambda: rbac_policy or {"roles": [], "grants": []}
        )
        self.query = _FakeQueryClient(node_rows or [])
        self.ledger = SimpleNamespace(get=lambda: ledger_ops or [])


def _install_fake_engine(monkeypatch: pytest.MonkeyPatch, client: object) -> list:
    """Patch GraphComputeEngine so LiveEngineCatalogClient's lazy imports resolve
    to a fake whose `.get_or_create(...)` returns an object exposing `.client`.
    Returns the list of graph_name args each get_or_create call was made with."""
    requested: list[str | None] = []

    class _FakeGraphComputeEngine:
        @staticmethod
        def get_or_create(graph_name: str | None = None, **kwargs: object) -> object:
            requested.append(graph_name)
            return SimpleNamespace(client=client)

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.core.graph_compute.GraphComputeEngine",
        _FakeGraphComputeEngine,
    )
    return requested


class _FailingGraphComputeEngine:
    @staticmethod
    def get_or_create(graph_name: str | None = None, **kwargs: object) -> object:
        raise RuntimeError("no engine configured")


def test_live_client_list_graphs_wraps_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.core.graph_compute.GraphComputeEngine",
        _FailingGraphComputeEngine,
    )
    client = go.LiveEngineCatalogClient()
    with pytest.raises(go.EngineUnreachableError):
        client.list_graphs()


def test_live_client_rbac_policy_wraps_non_dict_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bad_client = SimpleNamespace(rbac=SimpleNamespace(list=lambda: ["not-a-dict"]))
    _install_fake_engine(monkeypatch, bad_client)
    client = go.LiveEngineCatalogClient()
    with pytest.raises(go.EngineUnreachableError):
        client.rbac_policy()


def test_live_client_sample_nodes_wraps_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Boom:
        def cypher_read(self, query: str) -> list[dict]:
            raise RuntimeError("boom")

    _install_fake_engine(monkeypatch, SimpleNamespace(query=_Boom()))
    client = go.LiveEngineCatalogClient()
    with pytest.raises(go.EngineUnreachableError):
        client.sample_nodes("code_foo", 10)


def test_live_client_translates_real_response_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeEngineClient(
        tenants_rows=[{"name": "code_foo", "type": "Agent"}],
        rbac_policy={"roles": [{"name": "r"}], "grants": []},
        node_rows=[{"owner_id": "alice", "source_system": None, "tenant_id": None}],
        ledger_ops=["ADD_NODE|n1"],
    )
    requested = _install_fake_engine(monkeypatch, fake)
    client = go.LiveEngineCatalogClient()

    assert client.list_graphs() == [{"name": "code_foo", "type": "Agent"}]
    assert client.rbac_policy()["roles"] == [{"name": "r"}]
    rows = client.sample_nodes("code_foo", 25)
    assert rows == [{"owner_id": "alice", "source_system": None, "tenant_id": None}]
    assert "LIMIT 25" in fake.query.last_query
    assert "code_foo" in requested
    assert client.graph_ledger("code_foo") == ["ADD_NODE|n1"]


def test_resolve_catalog_client_returns_live_client() -> None:
    assert isinstance(go.resolve_catalog_client(), go.LiveEngineCatalogClient)
