"""RMDD-16 concept authority contract and compatibility tests."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any

import pytest
import yaml

from agent_utilities.governance.concept_reservation import (
    AuthorityUnavailable,
    ConceptNamespacePolicy,
    ConceptReservationConflict,
    ConceptReservationError,
    ConceptReservationFenceConflict,
    ConceptReservationIdUnavailable,
    ConceptReservationService,
    ConceptReservationState,
    ConceptReservationUnauthorized,
    ConceptReservationVisibility,
    FixtureConceptReservationAuthority,
    NativeConceptReservationAuthority,
    reconcile_projection,
    reservation_request,
    reserve_next_numeric,
)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "agent_utilities").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "concepts.yaml").write_text(
        yaml.safe_dump({"concepts": []}), encoding="utf-8"
    )
    return tmp_path


def request(
    concept_id: str = "AU-OS.governance.authority-test",
    *,
    tenant: str = "tenant-a",
    owner: str = "owner-a",
    key: str = "request-a",
    purpose: str = "reserve for a test",
    range_start: int | None = None,
    range_end: int | None = None,
    now: datetime | None = None,
):
    return reservation_request(
        concept_id,
        tenant_id=tenant,
        repository="agent-utilities",
        lane="rmdd-16",
        owner=owner,
        request_key=key,
        purpose=purpose,
        range_start=range_start,
        range_end=range_end,
        now=now or datetime(2026, 8, 9, tzinfo=UTC),
    )


class NativeStore:
    """Small durable-client contract fixture for existing native primitives."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.lock = RLock()
        self.always_lose_cas = False

    def create_node_if_absent(self, node_id: str, properties: dict[str, Any]) -> bool:
        with self.lock:
            if node_id in self.nodes:
                return False
            self.nodes[node_id] = dict(properties)
            return True

    def compare_and_set_node_fields(
        self, node_id: str, conditions: dict[str, Any], updates: dict[str, Any]
    ) -> bool:
        with self.lock:
            if self.always_lose_cas:
                return False
            current = self.nodes.get(node_id)
            if current is None or any(
                current.get(field) != expected for field, expected in conditions.items()
            ):
                return False
            current.update(updates)
            return True

    def get_node_properties(self, node_id: str) -> dict[str, Any] | None:
        with self.lock:
            value = self.nodes.get(node_id)
            return dict(value) if value is not None else None

    def list_nodes_by_label(
        self, label: str, limit: int, *, after: str | None = None
    ) -> list[tuple[str, dict[str, Any]]]:
        with self.lock:
            ids = sorted(
                node_id
                for node_id, props in self.nodes.items()
                if props.get("node_type") == label
                and (after is None or node_id > after)
            )
            return [(node_id, dict(self.nodes[node_id])) for node_id in ids[:limit]]


def native_authority(store: NativeStore):
    return NativeConceptReservationAuthority(
        store,
        policies=(
            ConceptNamespacePolicy("AU-OS.governance", policy_version="policy-1"),
        ),
    )


def test_fixture_is_explicitly_not_a_global_authority() -> None:
    authority = FixtureConceptReservationAuthority()
    assert authority.authoritative is False
    assert not getattr(authority, "cross_host_safe", False)


def test_reserve_is_idempotent_but_changed_input_conflicts() -> None:
    authority = FixtureConceptReservationAuthority()
    first = authority.reserve(request())
    assert authority.reserve(request()) == first
    assert (
        authority.reserve(request(now=datetime(2026, 8, 9, 0, 0, 5, tzinfo=UTC)))
        == first
    )
    with pytest.raises(ConceptReservationConflict, match="different immutable"):
        authority.reserve(request(purpose="a changed design"))


def test_concurrent_clients_have_one_winner_for_one_id() -> None:
    authority = FixtureConceptReservationAuthority()
    requests = [request(key=f"request-{index}") for index in range(50)]

    def reserve(item):
        try:
            return authority.reserve(item)
        except ConceptReservationConflict:
            return None

    with ThreadPoolExecutor(max_workers=50) as pool:
        results = list(pool.map(reserve, requests))
    winners = [result for result in results if result is not None]
    assert len(winners) == 1
    assert winners[0].concept_id == requests[0].concept_id


def test_namespace_policy_partitions_ranges() -> None:
    authority = FixtureConceptReservationAuthority(
        policies=(
            ConceptNamespacePolicy(
                "AU-OS.governance",
                range_start=10,
                range_end=20,
                concept_prefixes=("claim",),
            ),
        )
    )
    # The semantic grammar permits a numeric suffix, while policy decides its range.
    authority.reserve(request("AU-OS.governance.claim-10"))
    with pytest.raises(ValueError, match="outside the namespace/range policy"):
        authority.reserve(request("AU-OS.governance.claim-21", key="outside"))
    with pytest.raises(ValueError, match="outside the namespace/range policy"):
        authority.reserve(request("AU-OS.governance.other-10.extra", key="prefix"))


def test_fence_owner_and_terminal_visibility_are_enforced() -> None:
    authority = FixtureConceptReservationAuthority()
    record = authority.reserve(request())
    with pytest.raises(ConceptReservationFenceConflict):
        authority.transition(
            record.reservation_id,
            tenant_ref=record.tenant_ref,
            owner_ref=record.owner_ref,
            expected_fence=record.fence + 1,
            target=ConceptReservationState.MATERIALIZED,
        )
    with pytest.raises(ConceptReservationUnauthorized):
        authority.get(
            record.reservation_id, tenant_ref=request(tenant="tenant-b").tenant_ref
        )
    materialized = authority.transition(
        record.reservation_id,
        tenant_ref=record.tenant_ref,
        owner_ref=record.owner_ref,
        expected_fence=record.fence,
        target=ConceptReservationState.MATERIALIZED,
    )
    landed = authority.transition(
        materialized.reservation_id,
        tenant_ref=materialized.tenant_ref,
        owner_ref=materialized.owner_ref,
        expected_fence=materialized.fence,
        target=ConceptReservationState.LANDED,
    )
    tombstoned = authority.transition(
        landed.reservation_id,
        tenant_ref=landed.tenant_ref,
        owner_ref=landed.owner_ref,
        expected_fence=landed.fence,
        target=ConceptReservationState.TOMBSTONED,
    )
    assert tombstoned.visibility.value == "external"
    with pytest.raises(ConceptReservationConflict):
        authority.reserve(request(key="new-request"))


def test_released_id_can_be_reclaimed_but_landed_id_cannot() -> None:
    authority = FixtureConceptReservationAuthority()
    record = authority.reserve(request())
    released = authority.transition(
        record.reservation_id,
        tenant_ref=record.tenant_ref,
        owner_ref=record.owner_ref,
        expected_fence=record.fence,
        target=ConceptReservationState.RELEASED,
    )
    assert released.state is ConceptReservationState.RELEASED
    replacement = authority.reserve(request(key="replacement"))
    assert replacement.concept_id == record.concept_id


def test_native_adapter_refuses_missing_native_primitives() -> None:
    class Engine:
        def _send_wire(self, method, payload=None):
            return {"ok": True}

    with pytest.raises(AuthorityUnavailable, match="required native primitive"):
        native_authority(Engine()).reserve(request())


def test_native_create_if_absent_is_idempotent_and_policy_bound() -> None:
    store = NativeStore()
    authority = native_authority(store)
    first = authority.reserve(request())
    assert authority.reserve(request()) == first
    assert first.request.policy_version == "policy-1"
    with pytest.raises(ConceptReservationConflict, match="different immutable"):
        authority.reserve(request(purpose="changed native purpose"))
    with pytest.raises(ConceptReservationError, match="outside"):
        authority.reserve(request("AU-OS.identity.not-in-policy", key="outside"))


@pytest.mark.parametrize(
    "authority_factory", [FixtureConceptReservationAuthority, native_authority]
)
def test_request_key_idempotency_is_scoped_to_the_canonical_concept_node(
    authority_factory,
) -> None:
    store = NativeStore()
    authority = (
        authority_factory(store)
        if authority_factory is native_authority
        else authority_factory()
    )
    first = authority.reserve(request(key="same-key"))
    second = authority.reserve(
        request("AU-OS.governance.other-concept", key="same-key")
    )
    assert first.reservation_id != second.reservation_id
    assert authority.reserve(request(key="same-key")) == first


def test_native_cross_tenant_read_and_mutation_are_denied() -> None:
    store = NativeStore()
    authority = native_authority(store)
    record = authority.reserve(request())
    with pytest.raises(ConceptReservationUnauthorized):
        authority.get(
            record.reservation_id,
            tenant_ref=request(tenant="tenant-b").tenant_ref,
        )
    with pytest.raises(ConceptReservationIdUnavailable):
        authority.reserve(request(tenant="tenant-b", key="tenant-b-request"))


def test_native_50_adapters_share_one_native_engine_winner() -> None:
    """Exercise shared-engine atomicity, not a separate-host deployment."""

    store = NativeStore()
    clients = [native_authority(store) for _ in range(50)]
    requests = [request(key=f"native-request-{index}") for index in range(50)]

    def reserve(item):
        try:
            return clients[item[0]].reserve(item[1])
        except ConceptReservationConflict:
            return None

    with ThreadPoolExecutor(max_workers=50) as pool:
        results = list(pool.map(reserve, enumerate(requests)))
    winners = [result for result in results if result is not None]
    assert len(winners) == 1
    assert len(store.nodes) == 1


def test_native_reclaim_cas_loss_is_bounded_and_fails_closed() -> None:
    store = NativeStore()
    authority = native_authority(store)
    reserved = authority.reserve(request())
    released = authority.transition(
        reserved.reservation_id,
        tenant_ref=reserved.tenant_ref,
        owner_ref=reserved.owner_ref,
        expected_fence=reserved.fence,
        target=ConceptReservationState.RELEASED,
    )
    assert released.state is ConceptReservationState.RELEASED
    store.always_lose_cas = True
    with pytest.raises(AuthorityUnavailable, match="retry bound"):
        native_authority(store).reserve(request(key="bounded-reclaim"))


def test_native_reads_fail_closed_on_a_repeating_full_cursor_page() -> None:
    class RepeatingPageStore(NativeStore):
        def list_nodes_by_label(
            self, label: str, limit: int, *, after: str | None = None
        ) -> list[tuple[str, dict[str, Any]]]:
            rows = super().list_nodes_by_label(label, limit, after=None)
            if not rows:
                return []
            return [rows[0]] * limit

    store = RepeatingPageStore()
    authority = native_authority(store)
    authority.reserve(request())
    missing = "pref_concept_reservation_" + ("a" * 64)
    with pytest.raises(AuthorityUnavailable, match="cursor|strictly ordered"):
        authority.get(missing, tenant_ref=request().tenant_ref)
    with pytest.raises(AuthorityUnavailable, match="cursor|strictly ordered"):
        authority.list(
            tenant_ref=request().tenant_ref,
            concept_prefix="never-matches",
            limit=1,
        )


def test_native_cursor_input_is_bounded_and_control_safe() -> None:
    authority = native_authority(NativeStore())
    with pytest.raises(ConceptReservationError, match="cursor is invalid"):
        authority.list(tenant_ref=request().tenant_ref, cursor="\x00")
    with pytest.raises(ConceptReservationError, match="cursor is invalid"):
        authority.list(tenant_ref=request().tenant_ref, cursor="x" * 513)


def test_native_filtered_pages_resume_after_last_scanned_node_without_loss() -> None:
    store = NativeStore()
    authority = native_authority(store)
    expected_ids = []
    for index in range(40):
        if index % 3 == 0:
            concept_id = f"AU-OS.governance.target-{index}"
            expected_ids.append(concept_id)
        else:
            concept_id = f"AU-OS.governance.noise-{index}"
        authority.reserve(request(concept_id, key=f"paged-{index}"))
    expected_ids.sort()

    received_ids: list[str] = []
    cursor = None
    while True:
        page, cursor = authority.list(
            tenant_ref=request().tenant_ref,
            concept_prefix="AU-OS.governance.target",
            limit=1,
            cursor=cursor,
        )
        received_ids.extend(record.concept_id for record in page)
        if cursor is None:
            break

    assert received_ids == expected_ids
    assert len(received_ids) == len(set(received_ids))


@pytest.mark.parametrize(
    "authority_factory", [FixtureConceptReservationAuthority, native_authority]
)
def test_same_owner_transition_retry_is_idempotent(authority_factory) -> None:
    store = NativeStore()
    authority = (
        authority_factory(store)
        if authority_factory is native_authority
        else authority_factory()
    )
    reserved = authority.reserve(request())
    materialized = authority.transition(
        reserved.reservation_id,
        tenant_ref=reserved.tenant_ref,
        owner_ref=reserved.owner_ref,
        expected_fence=reserved.fence,
        target=ConceptReservationState.MATERIALIZED,
    )
    assert (
        authority.transition(
            materialized.reservation_id,
            tenant_ref=materialized.tenant_ref,
            owner_ref=materialized.owner_ref,
            expected_fence=reserved.fence,
            target=ConceptReservationState.MATERIALIZED,
        )
        == materialized
    )


@pytest.mark.parametrize(
    "authority_factory", [FixtureConceptReservationAuthority, native_authority]
)
def test_materialized_and_landed_visibility_never_downgrade_external(
    authority_factory,
) -> None:
    store = NativeStore()
    authority = (
        authority_factory(store)
        if authority_factory is native_authority
        else authority_factory()
    )
    reserved = authority.reserve(request())
    materialized = authority.transition(
        reserved.reservation_id,
        tenant_ref=reserved.tenant_ref,
        owner_ref=reserved.owner_ref,
        expected_fence=reserved.fence,
        target=ConceptReservationState.MATERIALIZED,
        visibility=ConceptReservationVisibility.EXTERNAL,
    )
    assert materialized.state is ConceptReservationState.TOMBSTONED
    assert materialized.visibility is ConceptReservationVisibility.EXTERNAL
    assert materialized.tombstoned_at is not None

    landed_request = request("AU-OS.governance.authority-landed", key="landed-external")
    landed_reserved = authority.reserve(landed_request)
    landed_materialized = authority.transition(
        landed_reserved.reservation_id,
        tenant_ref=landed_reserved.tenant_ref,
        owner_ref=landed_reserved.owner_ref,
        expected_fence=landed_reserved.fence,
        target=ConceptReservationState.MATERIALIZED,
    )
    landed = authority.transition(
        landed_materialized.reservation_id,
        tenant_ref=landed_materialized.tenant_ref,
        owner_ref=landed_materialized.owner_ref,
        expected_fence=landed_materialized.fence,
        target=ConceptReservationState.LANDED,
        visibility=ConceptReservationVisibility.EXTERNAL,
    )
    assert landed.state is ConceptReservationState.TOMBSTONED
    assert landed.visibility is ConceptReservationVisibility.EXTERNAL


def test_overlapping_policy_versions_are_rejected_not_lexically_selected() -> None:
    policies = (
        ConceptNamespacePolicy("AU-OS.governance", policy_version="2"),
        ConceptNamespacePolicy("AU-OS.governance", policy_version="10"),
    )
    with pytest.raises(ConceptReservationError, match="overlap"):
        NativeConceptReservationAuthority(NativeStore(), policies=policies)
    with pytest.raises(ConceptReservationError, match="overlap"):
        FixtureConceptReservationAuthority(policies=policies)


def test_native_numeric_allocator_probes_bounded_candidates() -> None:
    store = NativeStore()
    authority = NativeConceptReservationAuthority(
        store,
        policies=(
            ConceptNamespacePolicy(
                "AU-OS.governance",
                range_start=1,
                range_end=3,
                concept_prefixes=("claim",),
                policy_version="policy-2",
            ),
        ),
    )
    authority.reserve(request("AU-OS.governance.claim-1", key="occupied-1"))
    authority.reserve(request("AU-OS.governance.claim-2", key="occupied-2"))
    with pytest.raises(ConceptReservationConflict, match="policy"):
        authority.reserve(
            request(
                "AU-OS.governance.claim-1",
                key="occupied-1",
                range_start=2,
            )
        )
    allocated = reserve_next_numeric(
        authority,
        request("AU-OS.governance.placeholder", key="auto-claim"),
        concept_prefix="claim",
        range_start=1,
        range_end=3,
    )
    assert allocated.concept_id == "AU-OS.governance.claim-3"
    assert allocated.request.policy_version == "policy-2"


def test_native_restart_and_fenced_lifecycle_reclaim() -> None:
    store = NativeStore()
    first_authority = native_authority(store)
    reserved = first_authority.reserve(request())
    # A new adapter instance represents a process restart over the same durable
    # engine; point lookup and fenced materialization must survive it.
    restarted = native_authority(store)
    assert (
        restarted.get(reserved.reservation_id, tenant_ref=reserved.tenant_ref)
        == reserved
    )
    materialized = restarted.transition(
        reserved.reservation_id,
        tenant_ref=reserved.tenant_ref,
        owner_ref=reserved.owner_ref,
        expected_fence=reserved.fence,
        target=ConceptReservationState.MATERIALIZED,
    )
    assert materialized.materialized_at is not None
    with pytest.raises(ConceptReservationFenceConflict):
        first_authority.transition(
            reserved.reservation_id,
            tenant_ref=reserved.tenant_ref,
            owner_ref=reserved.owner_ref,
            expected_fence=reserved.fence,
            target=ConceptReservationState.LANDED,
        )
    landed = restarted.transition(
        materialized.reservation_id,
        tenant_ref=materialized.tenant_ref,
        owner_ref=materialized.owner_ref,
        expected_fence=materialized.fence,
        target=ConceptReservationState.LANDED,
    )
    tombstoned = restarted.transition(
        landed.reservation_id,
        tenant_ref=landed.tenant_ref,
        owner_ref=landed.owner_ref,
        expected_fence=landed.fence,
        target=ConceptReservationState.TOMBSTONED,
    )
    assert tombstoned.visibility.value == "external"
    with pytest.raises(
        ConceptReservationConflict, match="already reserved|never reusable"
    ):
        restarted.reserve(request(key="native-reuse"))


def test_native_restart_between_reserve_and_projection_is_reconciled_once(
    repo: Path,
) -> None:
    store = NativeStore()
    first = native_authority(store)
    reserved = first.reserve(request())
    restarted_service = ConceptReservationService(
        native_authority(store), repo_root=repo
    )
    materialized = restarted_service.materialize(
        reserved.reservation_id,
        tenant_ref=reserved.tenant_ref,
        owner_ref=reserved.owner_ref,
        expected_fence=reserved.fence,
    )
    before = (repo / "docs" / "concept_reservations.yaml").read_bytes()
    retried = restarted_service.materialize(
        materialized.reservation_id,
        tenant_ref=materialized.tenant_ref,
        owner_ref=materialized.owner_ref,
        expected_fence=reserved.fence,
    )
    assert retried == materialized
    assert (repo / "docs" / "concept_reservations.yaml").read_bytes() == before
    report = reconcile_projection(
        native_authority(store), tenant_ref=reserved.tenant_ref, repo_root=repo
    )
    assert report.matches == (reserved.concept_id,)


def test_external_visibility_on_release_forces_tombstone_and_no_reuse() -> None:
    authority = FixtureConceptReservationAuthority()
    record = authority.reserve(request())
    tombstoned = authority.transition(
        record.reservation_id,
        tenant_ref=record.tenant_ref,
        owner_ref=record.owner_ref,
        expected_fence=record.fence,
        target=ConceptReservationState.RELEASED,
        visibility=ConceptReservationVisibility.EXTERNAL,
    )
    assert tombstoned.state is ConceptReservationState.TOMBSTONED
    assert tombstoned.tombstoned_at is not None
    with pytest.raises(ConceptReservationConflict):
        authority.reserve(request(key="external-reuse"))


def test_expiry_requires_time_and_records_expiry_timestamp() -> None:
    authority = FixtureConceptReservationAuthority()
    record = authority.reserve(request(now=datetime(2020, 1, 1, tzinfo=UTC)))
    expired = authority.transition(
        record.reservation_id,
        tenant_ref=record.tenant_ref,
        owner_ref=record.owner_ref,
        expected_fence=record.fence,
        target=ConceptReservationState.EXPIRED,
    )
    assert expired.expired_at is not None


def test_reconciliation_is_bounded_and_rejects_stuck_cursor(tmp_path: Path) -> None:
    repo_root = tmp_path
    (repo_root / "agent_utilities").mkdir()
    (repo_root / "docs").mkdir()
    (repo_root / "docs" / "concepts.yaml").write_text(
        yaml.safe_dump({"concepts": []}), encoding="utf-8"
    )

    class StuckCursorAuthority(FixtureConceptReservationAuthority):
        def list(self, **kwargs):  # noqa: ANN003 - contract fixture
            kwargs.pop("cursor", None)
            rows, _cursor = super().list(**kwargs)
            return rows, "same-cursor"

    authority = StuckCursorAuthority()
    authority.reserve(request())
    with pytest.raises(AuthorityUnavailable, match="cursor did not advance"):
        reconcile_projection(
            authority,
            tenant_ref=request().tenant_ref,
            repo_root=repo_root,
            max_records=2,
        )


def test_materialization_is_retryable_and_reconciliation_is_read_only(
    repo: Path,
) -> None:
    authority = FixtureConceptReservationAuthority()
    service = ConceptReservationService(authority, repo_root=repo)
    reserved = service.reserve(request())
    materialized = service.materialize(
        reserved.reservation_id,
        tenant_ref=reserved.tenant_ref,
        owner_ref=reserved.owner_ref,
        expected_fence=reserved.fence,
    )
    before = (repo / "docs" / "concept_reservations.yaml").read_bytes()
    retried = service.materialize(
        materialized.reservation_id,
        tenant_ref=materialized.tenant_ref,
        owner_ref=materialized.owner_ref,
        expected_fence=reserved.fence,
    )
    assert retried == materialized
    assert (repo / "docs" / "concept_reservations.yaml").read_bytes() == before
    result = reconcile_projection(
        authority, tenant_ref=reserved.tenant_ref, repo_root=repo
    )
    assert result.matches == (reserved.concept_id,)
    assert result.missing_projection == ()

    marker = repo / "agent_utilities" / "feature.py"
    marker.write_text(f"# CONCEPT:{reserved.concept_id}\n", encoding="utf-8")
    after_marker = reconcile_projection(
        authority, tenant_ref=reserved.tenant_ref, repo_root=repo
    )
    assert after_marker.state_mismatch == (reserved.concept_id,)
    assert (repo / "docs" / "concept_reservations.yaml").read_bytes() == before
