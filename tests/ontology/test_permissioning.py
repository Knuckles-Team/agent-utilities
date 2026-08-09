"""Current-contract tests for fail-closed object permissioning."""

from __future__ import annotations

import json

import pytest

from agent_utilities.knowledge_graph.core.company_brain_runtime import (
    get_company_brain,
    reset_company_brain,
)
from agent_utilities.knowledge_graph.ontology import permissioning as p
from agent_utilities.models.company_brain import ActorType, DataClassification, NodeACL
from agent_utilities.security.brain_context import ActorContext


class FakeMarkingStore:
    def __init__(self) -> None:
        self.persisted: dict[str, dict] = {}

    def execute(self, query, params):
        if query.startswith("MATCH"):
            return list(self.persisted.values())
        self.persisted[params["id"]] = {
            "node_id": params["n"],
            "tenant_id": params["tenant"],
            "markings": params["marks"],
        }
        return []


def _actor(*roles: str, tenant: str = "tenant-a") -> ActorContext:
    return ActorContext(
        actor_id="principal:verified",
        actor_type=ActorType.AUTOMATED_SERVICE,
        roles=roles,
        tenant_id=tenant,
        authenticated=True,
    )


@pytest.fixture(autouse=True)
def clean_permissioning():
    reset_company_brain()
    p.clear_markings()
    p.set_marking_store(FakeMarkingStore())
    yield
    p.clear_markings()
    reset_company_brain()


def _acl(node_id: str, **kwargs) -> None:
    get_company_brain().permissions.set_acl(
        NodeACL(
            node_id=node_id,
            classification=kwargs.pop("classification", DataClassification.PUBLIC),
            **kwargs,
        )
    )


def test_markings_are_tenant_keyed_and_persisted():
    store = FakeMarkingStore()
    p.set_marking_store(store)
    p.apply_marking("node-a", "export-controlled", tenant="tenant-a")
    assert p.markings_for("node-a", tenant="tenant-a") == {"export-controlled"}
    assert p.markings_for("node-a", tenant="tenant-b") == set()
    row = store.persisted["marking::tenant-a::node-a"]
    assert json.loads(row["markings"]) == ["export-controlled"]


def test_propagation_preserves_mandatory_markings():
    p.apply_marking("source", "restricted", tenant="tenant-a")
    assert p.propagate_markings(
        "source", "derived", tenant="tenant-a", propagate_classification=False
    ) == {"restricted"}


class _CountingMarkingStore(FakeMarkingStore):
    """A ``FakeMarkingStore`` that also counts every ``execute`` call, so a
    test can assert a redundant durable write did NOT happen — not just that
    the end state is correct."""

    def __init__(self) -> None:
        super().__init__()
        self.execute_calls = 0

    def execute(self, query, params):
        self.execute_calls += 1
        return super().execute(query, params)


def test_apply_marking_skips_redundant_persist_for_an_already_recorded_marking():
    """BUG-047: ``apply_marking`` used to re-run the durable ``MERGE ... SET``
    on EVERY call, even when the marking was already recorded for that
    (tenant, node) key. The fleet's per-tool ``connector-unconfigured-acl``
    backfill (:mod:`...source_connectors.connectors.mcp_tool` ->
    :func:`...source_connectors.permission_sync.sync_access`) calls this once
    per tool on every source-sync pass regardless of whether that tool's
    markings already matched — re-issuing the SAME MERGE for the SAME
    tenant/node/marking every cycle, measured as one of the two dominant
    sustained-slow-query sources behind the engine circuit breaker tripping.

    Markings are strictly additive (this module exposes no removal path), so
    once a name is present for a key it is durably present for good — a
    repeat call is by construction a no-op and must not touch the store."""
    store = _CountingMarkingStore()
    p.set_marking_store(store)

    # First-ever call in a fresh process: one hydration READ (populates the
    # empty cache from the — empty — durable store) plus one MERGE WRITE
    # (the marking is genuinely new).
    p.apply_marking("tool-1", "connector-unconfigured-acl", tenant="homelab")
    assert store.execute_calls == 2
    assert p.markings_for("tool-1", tenant="homelab") == {"connector-unconfigured-acl"}

    # A second sync pass re-classifies the SAME unconfigured tool with the
    # SAME marking — hydration is already cached (a process-wide, hydrate-once
    # flag), and the marking is already present, so this must be a genuine
    # no-op against the durable store: no new calls at all.
    p.apply_marking("tool-1", "connector-unconfigured-acl", tenant="homelab")
    assert store.execute_calls == 2, (
        "a repeat apply_marking for an already-recorded marking must not "
        "re-issue the durable MERGE"
    )

    # A GENUINELY NEW marking on the same node must still persist — this is
    # not a blanket "never write again" regression, only a same-marking skip.
    # Hydration is already cached, so this adds exactly one MERGE write.
    p.apply_marking("tool-1", "another-marking", tenant="homelab")
    assert store.execute_calls == 3
    assert p.markings_for("tool-1", tenant="homelab") == {
        "connector-unconfigured-acl",
        "another-marking",
    }


def test_apply_marking_idempotency_survives_process_restart_via_hydration():
    """The skip is keyed off the HYDRATED registry, not just this-process
    call history — a marking persisted by an earlier process (or an earlier
    run that has since re-hydrated) must be recognized too, not just one
    applied via this exact process's own prior ``apply_marking`` call."""
    store = _CountingMarkingStore()
    store.persisted["marking::homelab::tool-1"] = {
        "node_id": "tool-1",
        "tenant_id": "homelab",
        "markings": json.dumps(["connector-unconfigured-acl"]),
    }
    # Fresh registry/hydration state (simulates a new process that must
    # hydrate from the durable store before it can know this).
    p.clear_markings()
    p.set_marking_store(store)

    p.apply_marking("tool-1", "connector-unconfigured-acl", tenant="homelab")

    # Exactly one call: the hydration MATCH read. No MERGE write followed it,
    # because hydration already showed this marking as durably present.
    assert store.execute_calls == 1
    assert store.persisted["marking::homelab::tool-1"]["markings"] == json.dumps(
        ["connector-unconfigured-acl"]
    )


def test_missing_marking_store_fails_closed():
    p.clear_markings()
    p.set_marking_store(None)
    with pytest.raises(PermissionError, match="store is unavailable"):
        p.markings_for("node-a", tenant="tenant-a")


def test_missing_acl_is_denied():
    assert p.enforce([{"id": "node-a", "value": 1}], _actor("reader")) == []


def test_explicit_public_acl_allows_row():
    _acl("node-a")
    assert p.enforce([{"id": "node-a", "value": 1}], _actor("reader")) == [
        {"id": "node-a", "value": 1}
    ]


def test_acl_and_marking_both_must_permit():
    _acl("node-a", read_roles=["analyst"])
    p.apply_marking("node-a", "compartment-a", tenant="tenant-a")
    assert p.enforce([{"id": "node-a"}], _actor("analyst")) == []
    assert p.enforce(
        [{"id": "node-a"}], _actor("analyst", "marking:compartment-a")
    ) == [{"id": "node-a"}]


def test_property_redaction_requires_clearance_and_marking():
    row = {
        "id": "node-a",
        "public": "visible",
        "secret": "hidden",
        "compartment": "hidden",
        "__classification__": {"secret": "restricted"},
        "__markings__": {"compartment": ["a"]},
    }
    assert p.redact_object(row, _actor("reader")) == {
        "id": "node-a",
        "public": "visible",
    }
    assert p.redact_object(row, _actor("kg:admin"))["secret"] == "hidden"


def test_generic_admin_role_is_not_graph_privileged():
    row = {
        "id": "node-a",
        "secret": "hidden",
        "__classification__": {"secret": "restricted"},
    }
    assert "secret" not in p.redact_object(row, _actor("admin"))


def test_role_only_system_is_not_privileged():
    row = {
        "id": "node-a",
        "secret": "hidden",
        "__classification__": {"secret": "restricted"},
    }
    assert "secret" not in p.redact_object(row, _actor("system"))


@pytest.mark.parametrize(
    "row",
    [
        {"id": "node-a", "value": 1, "__classification__": "restricted"},
        {"id": "node-a", "value": 1, "__markings__": ["a"]},
        {
            "id": "node-a",
            "value": 1,
            "__classification__": {"value": "unknown-level"},
        },
    ],
)
def test_malformed_policy_metadata_fails_closed(row):
    with pytest.raises(PermissionError):
        p.redact_object(row, _actor("reader"))


def test_missing_governed_id_raises_instead_of_leaking_projection():
    with pytest.raises(PermissionError, match="governed node id"):
        p.enforce([{"value": "opaque"}], _actor("reader"))


def test_unverified_or_tenantless_actor_is_rejected():
    _acl("node-a")
    for actor in (
        ActorContext(actor_id="principal", tenant_id="tenant-a"),
        _actor("reader", tenant=""),
    ):
        with pytest.raises(PermissionError, match="verified tenant"):
            p.enforce([{"id": "node-a"}], actor)


def test_acl_infrastructure_failure_denies(monkeypatch):
    _acl("node-a")
    permissions = get_company_brain().permissions
    monkeypatch.setattr(
        permissions,
        "check_permission",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError()),
    )
    assert p.enforce([{"id": "node-a"}], _actor("reader")) == []
