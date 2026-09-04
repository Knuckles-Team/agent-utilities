"""Tests for CA-26's bundle applier (`permissioning_external_sync.py`).

CONCEPT:AU-KG.ontology.policy-external-sync

Fixture-based throughout -- CA-16's live `/policy/export` endpoint was not
reachable from this lane's environment (see the module's own "Live-integration
status" doc section), so every test here exercises the bundle SHAPE CA-16's
Rust suite proves it emits (`epistemic-graph`'s `policy_export::mod::tests::
known_marking_set_produces_the_expected_bundle_json`), not a live fetch.
"""

from __future__ import annotations

import json
import re

import pytest

from agent_utilities.knowledge_graph.ontology import permissioning_external_sync as sync


def _predicate_json(name: str, column: str = sync.RESERVED_MARKING_COLUMN) -> str:
    return json.dumps(
        {"kind": "requires_role", "role": f"{sync.MARKING_ROLE_PREFIX}{name}", "column": column}
    )


def _fixture_bundle(
    *,
    governs: list[str] | None = None,
    markings: dict[str, str] | None = None,
    subject: str = "svc:restricted",
    effective_roles: list[str] | None = None,
) -> sync.PolicyBundle:
    """A DEC-CA-04-shaped bundle -- byte-for-byte the JSON CA-16's Rust
    `generate_bundle` emits (matching field names/types exactly).

    The default caller is NOT cleared for `confidential`, i.e. the ordinary
    case: a bundle whose one subject lacks the marking role.
    """
    markings = markings if markings is not None else {"confidential": _predicate_json("confidential")}
    raw = {
        "version": "policy-bundle-v1",
        "generated_from": "epoch:sha256:deadbeef",
        "governs": governs if governs is not None else ["M1"],
        "tenant": "tenant-a",
        "graphs": ["tenant:tenant-a", "__commons__"],
        "caller": {
            "subject": subject,
            "effective_roles": (
                effective_roles if effective_roles is not None else ["kg:read"]
            ),
        },
        "markings": {name: {"predicate": pred} for name, pred in markings.items()},
    }
    return sync.PolicyBundle.from_json(raw)


def _reasons(rendered: sync.RenderedTarget) -> str:
    """`not_applicable` joined into one searchable string.

    A test that searched it with ``any(... for ...)`` would add a branch to a
    function the complexity regression gate measures; this keeps the search
    branch-free and in exactly one place.
    """
    return " | ".join(rendered.not_applicable)


def _cleared_bundle(**kwargs) -> sync.PolicyBundle:
    """A bundle whose caller DOES hold `marking:confidential`."""
    kwargs.setdefault("subject", "svc:cleared")
    kwargs.setdefault("effective_roles", ["kg:read", "marking:confidential"])
    return _fixture_bundle(**kwargs)


# ── Schema round-trip (W02 acceptance evidence) ─────────────────────────────


def test_schema_round_trip_matches_eg_bundle_shape():
    bundle = _fixture_bundle()
    assert bundle.version == "policy-bundle-v1"
    assert bundle.governs == ("M1",)
    assert bundle.tenant == "tenant-a"
    assert bundle.graphs == ("tenant:tenant-a", "__commons__")
    assert bundle.caller == sync.BundleCaller(
        subject="svc:restricted", effective_roles=("kg:read",)
    )
    assert bundle.caller.holds("kg:read") is True
    assert bundle.caller.holds("marking:confidential") is False
    assert not hasattr(bundle, "principals"), (
        "the forgeable `principals` map is deleted, not renamed -- a consumer must "
        "not be able to reach a population type on this bundle (RF-RULING-001)"
    )
    assert not hasattr(bundle, "role_holders"), (
        "`role_holders` answered 'who holds role R?', which a single-caller bundle "
        "cannot answer; it is deleted, not adapted"
    )
    predicate = bundle.markings["confidential"].decode()
    assert predicate == sync.RequiresRolePredicate(role="marking:confidential", column="_markings")
    assert predicate.marking_name == "confidential"


def test_missing_required_field_raises():
    raw = {"version": "policy-bundle-v1", "generated_from": "x", "governs": ["M1"], "tenant": "t"}
    with pytest.raises(ValueError):
        sync.PolicyBundle.from_json(raw)


def test_a_bundle_carrying_the_deleted_principals_map_is_refused():
    """The other half of the atomic cutover, asserted from this side: a bundle
    in the OLD shape (a `principals` map, no `caller`) does not decode. It is
    refused loudly by `from_json`, so `BundleFetcher.fetch` turns it into
    `FetchedBundle(bundle=None, ...)` and `Applier.apply_all` denies every
    target -- never a silent partial read of a stale contract."""
    raw = {
        "version": "policy-bundle-v1",
        "generated_from": "epoch:sha256:deadbeef",
        "governs": ["M1"],
        "tenant": "tenant-a",
        "graphs": ["tenant:tenant-a"],
        "principals": {"svc:a": ["kg:read"]},
        "markings": {},
    }
    with pytest.raises(ValueError):
        sync.PolicyBundle.from_json(raw)


def test_unrecognized_predicate_kind_raises():
    bundle = _fixture_bundle(markings={"x": json.dumps({"kind": "row_content_match"})})
    with pytest.raises(ValueError):
        bundle.markings["x"].decode()


# ── `governs` recognition -- DEC-CA-04's binding "unrecognized governs denies" rule ──


@pytest.mark.parametrize(
    ("governs", "expected"),
    [
        (["M1"], True),
        ([], False),
        (["M1", "M7"], False),
        (["M6"], False),
    ],
)
def test_governs_recognized(governs, expected):
    assert _fixture_bundle(governs=governs).governs_recognized() is expected


# ── TrinoRenderer ────────────────────────────────────────────────────────────


def test_trino_renderer_emits_pushdown_filter_only_for_columned_tables():
    bundle = _fixture_bundle()
    tables = (
        sync.TrinoTargetTable("lakehouse", "analytics", "docs", has_markings_column=True),
        sync.TrinoTargetTable("lakehouse", "analytics", "no_column_yet", has_markings_column=False),
    )
    rendered = sync.TrinoRenderer(tables=tables).render(bundle)

    # The un-columned table is reported, never silently dropped or faked.
    assert "no_column_yet" in _reasons(rendered)

    # Exactly one filter row, for the bundle's caller, which LACKS
    # marking:confidential.
    docs_rows = [r for r in rendered.payload if r["table"] == "docs"]
    assert len(docs_rows) == 1
    row = docs_rows[0]
    assert row["user"] == "svc:restricted"
    assert row["catalog"] == "lakehouse"
    assert row["schema"] == "analytics"
    # Trino's real file-based-access-control key is `filter`, not the bundle
    # contract's engine-neutral `row_filter` name -- this asserts the
    # translation, not the bundle's own field name.
    assert "filter" in row
    assert "_markings" in row["filter"] and "confidential" in row["filter"]


# ── C-7 regression: the catch-all fail-open is closed ───────────────────────
#
# This block replaces `test_trino_renderer_everyone_filtered_when_no_principal_
# holds_role`, which PINNED the defect: it asserted that a bundle in which no
# principal held the marking role renders `{"user": ".*"}`. See
# `TrinoRenderer`'s class doc and `plans/refactor/evidence/reports/
# ADVERSARIAL-REVIEW-THREE-DESIGNS.md` C-7.


def test_trino_renderer_never_emits_a_catch_all_user_rule():
    """C-7. No principal holding the marking role must NOT produce a
    `"user": ".*"` rule. A catch-all matches every user, and Trino evaluates
    table rules first-match-wins, so one such rule silently disables every rule
    appended after it."""
    bundle = _fixture_bundle()  # caller holds kg:read only -- holds no marking role
    tables = (sync.TrinoTargetTable("lakehouse", "analytics", "docs", has_markings_column=True),)
    rendered = sync.TrinoRenderer(tables=tables).render(bundle)

    assert rendered.payload, "the caller lacks the role, so it must still be filtered"
    assert all(row["user"] != ".*" for row in rendered.payload)
    # Nothing that MATCHES every user, however it is spelled.
    for row in rendered.payload:
        assert re.fullmatch(row["user"], "some:unrelated:subject") is None, (
            f"rule user pattern {row['user']!r} matches a subject this bundle does "
            "not describe"
        )
    # The population it cannot describe is REPORTED, not approximated.
    assert "single-caller bundle" in _reasons(rendered)


def test_trino_renderer_cleared_caller_emits_no_rule_but_reports_the_gap():
    """The other side of C-7: when the caller IS cleared there is nothing to
    filter for it, and still nothing may be asserted about anyone else."""
    bundle = _cleared_bundle()
    tables = (sync.TrinoTargetTable("lakehouse", "analytics", "docs", has_markings_column=True),)
    rendered = sync.TrinoRenderer(tables=tables).render(bundle)
    assert rendered.payload == []
    assert "single-caller bundle" in _reasons(rendered)


def test_trino_renderer_one_marking_never_shadows_another():
    """The concrete harm the old catch-all caused: with two markings on one
    table, a `.*` rule emitted for the first shadowed the second under Trino's
    first-match-wins rule order, so rows carrying the second marking became
    visible to everyone. Every rule now names exactly one subject, so no rule
    can shadow a later one for a different user."""
    bundle = _fixture_bundle(
        markings={
            "confidential": _predicate_json("confidential"),
            "restricted": _predicate_json("restricted"),
        }
    )
    tables = (sync.TrinoTargetTable("lakehouse", "analytics", "docs", has_markings_column=True),)
    rendered = sync.TrinoRenderer(tables=tables).render(bundle)

    marking_names = {
        name for name in ("confidential", "restricted")
        for row in rendered.payload
        if f"'{name}'" in row["filter"]
    }
    assert marking_names == {"confidential", "restricted"}, (
        "both markings must still be filtered -- neither may be shadowed"
    )
    assert all(row["user"] == "svc:restricted" for row in rendered.payload)


def test_trino_renderer_escapes_regex_metacharacters_in_the_subject():
    """`user` is a REGEX in Trino's file-based access control. An unescaped
    subject would widen a per-subject rule to other users -- the same defect
    class as the catch-all, one order of magnitude smaller."""
    bundle = _fixture_bundle(subject="svc.planner+eu")
    tables = (sync.TrinoTargetTable("c", "s", "t", has_markings_column=True),)
    rendered = sync.TrinoRenderer(tables=tables).render(bundle)
    pattern = rendered.payload[0]["user"]
    assert re.fullmatch(pattern, "svc.planner+eu") is not None
    assert re.fullmatch(pattern, "svcxplanner+eu") is None


def test_trino_renderer_pushdown_is_role_based_not_row_content_based():
    """The filter expression is a function of (marking name, column) only --
    never a value derived from OTHER columns already returned to the caller,
    which is exactly why Trino can evaluate it during query planning
    (pushdown) regardless of what the caller projects (`SELECT id` vs
    `SELECT *`): the expression references only the governed marking column,
    the same reasoning CA-24's `dls.py` module doc gives for OpenSearch DLS
    and DEC-CA-04/CA-63 requires every renderer to state explicitly."""
    bundle = _fixture_bundle()
    tables = (sync.TrinoTargetTable("c", "s", "t", has_markings_column=True),)
    rendered = sync.TrinoRenderer(tables=tables).render(bundle)
    for row in rendered.payload:
        assert row["filter"].startswith(f"NOT contains({sync.RESERVED_MARKING_COLUMN}")


def test_trino_renderer_rejects_unrecognized_governs_bundle():
    bundle = _fixture_bundle(governs=["M7"])
    with pytest.raises(ValueError):
        sync.TrinoRenderer(tables=()).render(bundle)


# ── OpenSearchRenderer -- reuses CA-24's dls.py, does not re-derive it ──────


def test_opensearch_renderer_reuses_ca24_dls_query_shape():
    restricted = sync.OpenSearchRenderer(index_patterns=("kg-tenant-a-*",)).render(
        _fixture_bundle()
    )
    cleared = sync.OpenSearchRenderer(index_patterns=("kg-tenant-a-*",)).render(
        _cleared_bundle()
    )
    # One row per index pattern, for the ONE subject the bundle describes.
    assert len(restricted.payload) == 1
    assert len(cleared.payload) == 1
    assert restricted.payload[0]["role"] == "ca26-svc:restricted"
    assert cleared.payload[0]["role"] == "ca26-svc:cleared"
    assert cleared.payload[0]["dls_query"] == {"match_all": {}}
    assert restricted.payload[0]["dls_query"] == {
        "bool": {"must_not": [{"term": {"marking": "confidential"}}]}
    }
    assert cleared.payload[0]["index_pattern"] == "kg-tenant-a-*"
    # DLS is an allow-list, so one subject under-grants rather than
    # over-granting -- but the un-described population is still reported.
    assert "only its caller" in _reasons(restricted)


def test_opensearch_renderer_query_matches_ca24_render_dls_query_for_role():
    """Cross-check against the actual CA-24 function this module wraps, so a
    future refactor that stops reusing it (re-deriving the shape locally)
    fails this test rather than silently drifting."""
    from agent_utilities.knowledge_graph.search.dls import render_dls_query_for_role

    bundle = _fixture_bundle()
    rendered = sync.OpenSearchRenderer(index_patterns=("kg-*",)).render(bundle)
    restricted_row = next(r for r in rendered.payload if r["role"] == "ca26-svc:restricted")
    expected = render_dls_query_for_role(["kg:read"], ["confidential"])
    assert restricted_row["dls_query"] == expected


# ── LakekeeperRenderer -- the granularity-mismatch finding ──────────────────


def test_lakekeeper_renderer_row_scoped_marking_is_not_applicable():
    """No table_scope entry for 'confidential' -> reported not-applicable,
    never fabricated as a tuple that wouldn't actually gate row content."""
    bundle = _fixture_bundle()
    rendered = sync.LakekeeperRenderer(table_scope={}).render(bundle)
    assert rendered.payload == []
    assert "row-level relation" in _reasons(rendered)


def test_lakekeeper_renderer_table_scoped_marking_renders_real_tuples():
    bundle = _cleared_bundle()
    ref = sync.LakekeeperTableRef(namespace="analytics", table="restricted_table")
    rendered = sync.LakekeeperRenderer(table_scope={"confidential": ref}).render(bundle)
    assert len(rendered.payload) == 1
    entry = rendered.payload[0]
    assert entry["namespace"] == "analytics"
    assert entry["table"] == "restricted_table"
    # The caller holds marking:confidential, so it gets the grant tuple.
    users = {t["user"] for t in entry["openfga_tuples"]}
    assert users == {"oidc~svc:cleared"}
    assert entry["openfga_tuples"][0]["relation"] == "select"
    # An OpenFGA tuple set is an allow-list: an un-described subject simply
    # gets no grant (fail-closed), and that limit is reported.
    assert "only the bundle's caller" in _reasons(rendered)


def test_lakekeeper_renderer_uncleared_caller_gets_no_grant_tuple():
    bundle = _fixture_bundle()
    ref = sync.LakekeeperTableRef(namespace="analytics", table="restricted_table")
    rendered = sync.LakekeeperRenderer(table_scope={"confidential": ref}).render(bundle)
    assert rendered.payload == [
        {"namespace": "analytics", "table": "restricted_table", "openfga_tuples": []}
    ]


# ── Applier: fail-closed + idempotent double-apply ──────────────────────────


@pytest.fixture(autouse=True)
def _enable_sync(monkeypatch):
    monkeypatch.setenv("CA26_EXTERNAL_POLICY_SYNC_ENABLED", "true")
    yield


def _applier() -> tuple[sync.Applier, dict[str, sync.InMemoryTargetClient]]:
    bundle_tables = (sync.TrinoTargetTable("c", "s", "t", has_markings_column=True),)
    renderers = {
        "trino": sync.TrinoRenderer(tables=bundle_tables),
        "opensearch": sync.OpenSearchRenderer(index_patterns=("kg-*",)),
        "lakekeeper": sync.LakekeeperRenderer(table_scope={}),
    }
    clients = {name: sync.InMemoryTargetClient(name=name) for name in renderers}
    return sync.Applier(renderers, clients), clients


def test_applier_disabled_by_default_flag(monkeypatch):
    monkeypatch.delenv("CA26_EXTERNAL_POLICY_SYNC_ENABLED", raising=False)
    applier, clients = _applier()
    fetched = sync.FetchedBundle(bundle=_fixture_bundle(), fetched_at_monotonic=0.0)
    report = applier.apply_all(fetched)
    assert report["disabled"] is True
    assert report["outcomes"] == []
    for client in clients.values():
        assert client.applied == []
        assert client.deny_all_calls == 0


def test_applier_denies_all_targets_on_missing_bundle():
    applier, clients = _applier()
    fetched = sync.FetchedBundle(bundle=None, fetched_at_monotonic=0.0, error="fetch failed")
    report = applier.apply_all(fetched)
    assert {o["target"] for o in report["outcomes"]} == {"trino", "opensearch", "lakekeeper"}
    for outcome in report["outcomes"]:
        assert outcome["action"] == "denied-all"
    for client in clients.values():
        assert client.deny_all_calls == 1
        assert client.applied == []


def test_applier_denies_all_targets_on_unrecognized_governs():
    applier, clients = _applier()
    fetched = sync.FetchedBundle(bundle=_fixture_bundle(governs=["M6"]), fetched_at_monotonic=0.0)
    report = applier.apply_all(fetched)
    for outcome in report["outcomes"]:
        assert outcome["action"] == "denied-all"
    for client in clients.values():
        assert client.deny_all_calls == 1


def test_applier_double_apply_of_unchanged_bundle_makes_zero_downstream_calls():
    applier, clients = _applier()
    fetched = sync.FetchedBundle(bundle=_fixture_bundle(), fetched_at_monotonic=0.0)

    first = applier.apply_all(fetched)
    assert all(o["action"] == "applied" for o in first["outcomes"])
    for client in clients.values():
        assert len(client.applied) == 1

    second = applier.apply_all(fetched)
    assert all(o["action"] == "skipped-unchanged" for o in second["outcomes"])
    for client in clients.values():
        # Zero ADDITIONAL downstream calls on the second, unchanged apply.
        assert len(client.applied) == 1


def test_applier_reapplies_when_bundle_content_changes():
    applier, clients = _applier()
    fetched1 = sync.FetchedBundle(bundle=_fixture_bundle(), fetched_at_monotonic=0.0)
    applier.apply_all(fetched1)

    fetched2 = sync.FetchedBundle(
        bundle=_fixture_bundle(markings={"restricted": _predicate_json("restricted")}),
        fetched_at_monotonic=1.0,
    )
    second = applier.apply_all(fetched2)
    outcomes_by_target = {o["target"]: o for o in second["outcomes"]}
    # trino/opensearch payloads changed (marking name flipped) -> re-applied.
    assert outcomes_by_target["trino"]["action"] == "applied"
    assert outcomes_by_target["opensearch"]["action"] == "applied"
    assert len(clients["trino"].applied) == 2
    assert len(clients["opensearch"].applied) == 2
    # lakekeeper's rendered payload is EMPTY both times (no table_scope
    # configured for either marking name) -- same digest, correctly skipped;
    # this is the "not_applicable, not fabricated" behaviour, not a bug.
    assert outcomes_by_target["lakekeeper"]["action"] == "skipped-unchanged"
    assert len(clients["lakekeeper"].applied) == 1


def test_applier_rejects_renderer_client_name_mismatch():
    with pytest.raises(ValueError):
        sync.Applier({"trino": sync.TrinoRenderer(tables=())}, {"opensearch": sync.InMemoryTargetClient("opensearch")})


# ── invalidate_cache / dirty-tenant bookkeeping ─────────────────────────────


def test_invalidate_cache_marks_tenant_dirty():
    sync.clear_dirty_state("tenant-x")
    assert sync.is_dirty("tenant-x") is False
    sync.invalidate_cache("tenant-x", "node-1")
    assert sync.is_dirty("tenant-x") is True
    sync.clear_dirty_state("tenant-x")
    assert sync.is_dirty("tenant-x") is False


def test_invalidate_cache_ignores_empty_tenant():
    sync.clear_dirty_state("")
    sync.invalidate_cache("", "node-1")
    assert sync.is_dirty("") is False


def test_external_policy_sync_enabled_default_false(monkeypatch):
    monkeypatch.delenv("CA26_EXTERNAL_POLICY_SYNC_ENABLED", raising=False)
    assert sync.external_policy_sync_enabled() is False


def test_external_policy_sync_enabled_true(monkeypatch):
    monkeypatch.setenv("CA26_EXTERNAL_POLICY_SYNC_ENABLED", "true")
    assert sync.external_policy_sync_enabled() is True
