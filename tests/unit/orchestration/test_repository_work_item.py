"""Repository-development WorkItem authority contract tests."""

from __future__ import annotations

import threading
from collections.abc import Callable
from copy import deepcopy
from datetime import UTC, datetime
from typing import Any

import pytest

from agent_utilities.orchestration import repository_work_item as rwi
from agent_utilities.orchestration import work_item as wi
from agent_utilities.orchestration.operation_payload import (
    RepositoryBuildExecutionPayloadV1,
    cache_key_digest_from_components,
)
from agent_utilities.orchestration.repository_work_item import (
    RepositoryConsentPolicy,
    RepositoryJobState,
    RepositoryOperation,
    RepositoryWorkItemConflict,
    RepositoryWorkItemError,
    RepositoryWorkItemKind,
    RepositoryWorkItemRequest,
    cancel_repository_work_item,
    checkpoint_repository_work_item,
    claim_repository_work_item,
    commit_repository_work_item,
    get_repository_operation_payload,
    get_repository_operation_payload_for_claim,
    get_repository_work_item,
    heartbeat_repository_work_item,
    list_repository_work_items,
    reconcile_repository_work_items,
    repository_job_id,
    repository_result_from_view,
    repository_work_item_id,
    repository_work_item_kind,
    submit_repository_work_item,
)


class RepositoryEngine:
    """Durable-shaped in-memory engine with atomic create and native verbs."""

    def __init__(self, *, create_barrier: threading.Barrier | None = None) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: list[tuple[str, str, str]] = []
        self._lock = threading.Lock()
        self.create_barrier = create_barrier
        self.on_create: Callable[[], None] | None = None
        self.on_link: Callable[[], None] | None = None
        self.cas_hook: Callable[[str, dict[str, Any], dict[str, Any]], None] | None = (
            None
        )

    def add_node(
        self,
        node_id: str,
        node_type: str,
        properties: dict[str, Any] | None = None,
        ephemeral: bool = False,
    ) -> None:
        with self._lock:
            self.nodes[node_id] = {
                **self.nodes.get(node_id, {}),
                **dict(properties or {}),
                "label": node_type,
            }

    def create_node_if_absent(
        self, node_id: str, properties: dict[str, Any] | None = None
    ) -> bool:
        if self.create_barrier is not None:
            self.create_barrier.wait()
        with self._lock:
            if node_id in self.nodes:
                return False
            self.nodes[node_id] = {
                **dict(properties or {}),
                "label": "WorkItem",
            }
            callback = self.on_create
            self.on_create = None
            if callback is not None:
                callback()
            return True

    def link_nodes(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: dict[str, Any] | None = None,
        ephemeral: bool = False,
    ) -> None:
        edge = (source_id, target_id, str(rel_type))
        if edge not in self.edges:
            self.edges.append(edge)
        callback = self.on_link
        self.on_link = None
        if callback is not None:
            callback()

    def compare_and_set_node_fields(
        self, node_id: str, conditions: dict[str, Any], updates: dict[str, Any]
    ) -> bool:
        hook = self.cas_hook
        if hook is not None:
            hook(node_id, conditions, updates)
        with self._lock:
            node = self.nodes.get(node_id)
            if node is None or any(node.get(k) != v for k, v in conditions.items()):
                return False
            node.update(updates)
            return True

    @staticmethod
    def _negative() -> dict[str, Any]:
        return {
            "schema_version": "1",
            "claimed": False,
            "reason": "empty",
            "work_item_id": None,
            "kind": None,
            "payload_ref": None,
            "lease_holder_ref": None,
            "lease_epoch": None,
            "fencing_token": None,
            "lease_expires_at_ms": None,
            "attempt": None,
            "max_attempts": None,
            "tenant_in_flight": 0,
            "changed_work_item_ids": [],
        }

    @staticmethod
    def _owns(node: dict[str, Any] | None, request: dict[str, Any]) -> bool:
        return bool(
            node
            and node.get("lease_owner") == request.get("worker_ref")
            and node.get("lease_epoch") == request.get("expected_epoch")
            and node.get("fencing_token") == request.get("fencing_token")
        )

    def _candidate(self, request: Any) -> tuple[str, dict[str, Any]] | None:
        if request.work_item_id:
            candidates = [
                (str(request.work_item_id), self.nodes.get(str(request.work_item_id)))
            ]
        else:
            candidates = sorted(
                self.nodes.items(),
                key=lambda pair: (
                    int(pair[1].get("prio_bucket") or 0),
                    float(pair[1].get("created_at") or 0),
                ),
            )
        now = float(request.now_ms) / 1000.0
        for item_id, node in candidates:
            if not node or node.get("label") != "WorkItem":
                continue
            if node.get("tenant") != request.tenant_ref:
                continue
            if request.queue_ref and node.get("queue") != request.queue_ref:
                continue
            if (
                request.resource_class
                and node.get("resource_class") != request.resource_class
            ):
                continue
            status = node.get("status")
            if status == "ready":
                pass
            elif status in {"leased", "running"}:
                # Native claim recovery may reclaim an expired lease after a
                # process restart; an unexpired owner remains fenced out.
                if float(node.get("lease_expires_at") or 0) >= now:
                    continue
            else:
                continue
            if float(node.get("next_retry_at") or 0) > now:
                continue
            return item_id, node
        return None

    def claim_work_item(self, request: Any) -> dict[str, Any]:
        with self._lock:
            selected = self._candidate(request)
            if selected is None:
                return self._negative()
            item_id, node = selected
            attempt = int(node.get("attempt") or 0) + 1
            if attempt > int(node.get("max_attempts") or 1):
                node["status"] = "dead_letter"
                return self._negative()
            epoch = int(node.get("lease_epoch") or 0) + 1
            node.update(
                status="leased",
                lease_owner=request.worker_ref,
                lease_epoch=epoch,
                fencing_token=epoch,
                attempt=attempt,
                lease_expires_at=float(request.now_ms + request.lease_ms) / 1000.0,
            )
            return {
                "schema_version": "1",
                "claimed": True,
                "reason": "claimed",
                "work_item_id": item_id,
                "kind": node.get("kind"),
                "payload_ref": node.get("payload_ref"),
                "lease_holder_ref": request.worker_ref,
                "lease_epoch": epoch,
                "fencing_token": epoch,
                "lease_expires_at_ms": request.now_ms + request.lease_ms,
                "attempt": attempt,
                "max_attempts": node["max_attempts"],
                "tenant_in_flight": 1,
                "changed_work_item_ids": [item_id],
            }

    def renew_work_item_lease(self, request: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            node = self.nodes.get(request["work_item_id"])
            if not self._owns(node, request):
                return {"renewed": False}
            if float(node.get("lease_expires_at") or 0) < float(request["now_unix"]):
                return {"renewed": False}
            node["lease_expires_at"] = float(request["now_unix"]) + float(
                request["lease_ttl"]
            )
            node["heartbeat_at"] = float(request["now_unix"])
            return {"renewed": True}

    def commit_work_item_result(self, request: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            node = self.nodes.get(request["work_item_id"])
            if node is None:
                return {"status": "missing"}
            if node.get("status") in wi.TERMINAL_WORK_ITEM_STATUSES:
                return {"status": "noop"}
            if not self._owns(node, request):
                return {"status": "fenced"}
            if request["outcome"] == "failed" and request["retryable"]:
                if int(node["attempt"]) >= int(node["max_attempts"]):
                    node.update(
                        status="dead_letter",
                        error_ref=request.get("error_ref"),
                        lease_owner=None,
                    )
                    return {"status": "dead_letter"}
                node.update(
                    status="ready",
                    next_retry_at=float(request["now_unix"]),
                    lease_epoch=int(node["lease_epoch"]) + 1,
                    fencing_token=int(node["fencing_token"]) + 1,
                    lease_owner=None,
                    lease_expires_at=None,
                )
                return {"status": "retry_scheduled"}
            node.update(
                status=request["outcome"],
                result_ref=request.get("result_ref"),
                error_ref=request.get("error_ref"),
                completed_at=request["now_unix"],
                lease_owner=None,
                lease_expires_at=None,
            )
            if request["outcome"] == "succeeded":
                for child_id in node.get("downstream_ids") or []:
                    child = self.nodes[child_id]
                    child["dep_count"] = max(0, int(child["dep_count"]) - 1)
                    if child["dep_count"] == 0:
                        child["status"] = "ready"
            return {"status": "committed"}

    def cancel_work_item(self, request: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            node = self.nodes.get(request["work_item_id"])
            if node is None:
                return {"status": "missing"}
            if node.get("status") == "cancelled":
                return {"status": "cancelled"}
            if node.get("status") in wi.TERMINAL_WORK_ITEM_STATUSES:
                return {"status": "not_cancellable"}
            node["status"] = "cancelled"
            return {"status": "cancelled"}

    def defer_work_item(self, request: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            node = self.nodes.get(request["work_item_id"])
            if not self._owns(node, request):
                return {"status": "fenced"}
            node.update(
                status="ready",
                next_retry_at=request["next_retry_at"],
                lease_owner=None,
                lease_expires_at=None,
                lease_epoch=int(node["lease_epoch"]) + 1,
                fencing_token=int(node["fencing_token"]) + 1,
            )
            return {"status": "deferred"}

    def _row(self, item_id: str, node: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": item_id,
            **{field: node.get(field) for field in wi._FIELDS},
        }

    def query_cypher(
        self, cypher: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        params = params or {}
        query = " ".join(cypher.split())
        if query.startswith("MATCH (w:WorkItem {id: $id}) RETURN w.id"):
            node = self.nodes.get(str(params["id"]))
            return (
                []
                if not node or node.get("label") != "WorkItem"
                else [self._row(str(params["id"]), node)]
            )
        if query.startswith("MATCH (w:WorkItem) WHERE w.kind IN $kinds"):
            rows = []
            statuses = params.get("statuses")
            cursor_created_at = params.get("cursor_created_at")
            cursor_id = params.get("cursor_id")
            for item_id, node in self.nodes.items():
                if node.get("label") != "WorkItem":
                    continue
                if (
                    node.get("kind") not in params["kinds"]
                    or node.get("tenant") != params["tenant"]
                ):
                    continue
                if statuses and node.get("status") not in statuses:
                    continue
                row_created_at = float(node.get("created_at") or 0.0)
                if cursor_created_at is not None and (
                    row_created_at < float(cursor_created_at)
                    or (
                        row_created_at == float(cursor_created_at)
                        and str(item_id) <= str(cursor_id)
                    )
                ):
                    continue
                rows.append(self._row(item_id, node))
            rows.sort(
                key=lambda row: (
                    float(row.get("created_at") or 0.0),
                    str(row.get("id") or ""),
                )
            )
            return rows[: int(params["limit"])]
        if query.startswith(
            "MATCH (w:WorkItem {tenant: $tenant}) WHERE NOT w.status IN $terminal"
        ):
            return [
                {
                    "c": sum(
                        node.get("label") == "WorkItem"
                        and node.get("tenant") == params["tenant"]
                        and node.get("status") not in params["terminal"]
                        for node in self.nodes.values()
                    )
                }
            ]
        raise AssertionError(f"unrecognized query: {query}")


class AuthorityProxy:
    """Host wrapper proving repository reads use the native WorkItem view."""

    def __init__(self, inner: RepositoryEngine, *, tenant: str | None = None) -> None:
        self.inner = inner
        self.tenant = tenant
        self.query_calls = 0

    def query_cypher(
        self, cypher: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        self.query_calls += 1
        return self.inner.query_cypher(cypher, params)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


def _insert_repository_row(
    engine: RepositoryEngine,
    request: RepositoryWorkItemRequest,
    *,
    created_at: float,
    dependencies: tuple[str, ...] = (),
) -> str:
    """Install a minimal durable-shaped row for pagination fixtures."""

    job_id = repository_job_id(request.tenant_id, request.idempotency_key)
    work_item_id = repository_work_item_id(job_id)
    kind = repository_work_item_kind(request.operation).value
    engine.nodes[work_item_id] = {
        "label": "WorkItem",
        "node_type": "WorkItem",
        "kind": kind,
        "queue": kind,
        "tenant": request.tenant_id,
        "status": "submitted" if dependencies else "submitted",
        "created_at": created_at,
        "depends_on": list(dependencies),
        "dep_count": len(dependencies),
        "downstream_ids": [],
        "metadata": rwi._request_metadata(
            request,
            job_id=job_id,
            input_digest=request.immutable_digest(),
            dependencies=dependencies,
        ),
    }
    return work_item_id


def _request(
    *, tenant: str = "tenant-a", key: str = "request-1", operation: str = "build"
) -> RepositoryWorkItemRequest:
    return RepositoryWorkItemRequest(
        request_id=key + ":request",
        idempotency_key=key,
        operation=operation,
        repository_id="agent-utilities",
        base_ref="main",
        base_sha="a" * 40,
        owner_id="actor-a",
        session_id="session-a",
        tenant_id=tenant,
        resource_class="light-check",
        concurrency_key="light-check",
    )


def _operation_payload(
    *,
    repository_id: str = "agent-utilities",
    base_sha: str = "a" * 40,
) -> RepositoryBuildExecutionPayloadV1:
    components = {
        "key_version": "v2",
        "repo": repository_id,
        "spec": "test-build",
        "tree_sha": base_sha,
        "feature_set": "cargo build",
        "toolchain_fingerprint": "unpinned",
        "target_triple": "x86_64-unknown-linux-gnu",
        "config_digest": "b" * 64,
        "spec_digest": "c" * 64,
        "generation_id": "",
        "generation_digest": "",
    }
    return RepositoryBuildExecutionPayloadV1(
        repository_id=repository_id,
        base_sha=base_sha,
        tree_sha=base_sha,
        build_spec_name="test-build",
        spec_digest=components["spec_digest"],
        config_digest=components["config_digest"],
        toolchain_digest="d" * 64,
        artifact_contract_digest="e" * 64,
        feature_set=components["feature_set"],
        target_triple=components["target_triple"],
        cache_key_components=components,
        cache_key_digest=cache_key_digest_from_components(components),
        argv=("cargo", "build"),
        workdir=".",
        timeout_seconds=60,
        artifact_patterns=("target/**",),
        environment_refs=("CI",),
        execution_policy_ref="repository.build-policy:v1",
        profile_ref="repository_manager:resource_profile:light-check:v1",
        cacheable=True,
    )


def test_registered_kinds_and_nested_contract_projection() -> None:
    request = RepositoryWorkItemRequest.from_contract(
        {
            "request_id": "req-1",
            "idempotency_key": "idem-1",
            "operation": "validation",
            "repository": {"repository_id": "repository-manager"},
            "base_ref": "main",
            "base_sha": "b" * 40,
            "owner_id": "actor",
            "session_id": "session",
            "tenant_id": "tenant",
            "resources": {"resource_class": "pre-commit", "priority": 8},
            "target": {"kind": "local"},
            "validation_policy": {"stages": ["feedback", "integration"]},
        }
    )
    assert request.operation == RepositoryOperation.VALIDATION
    assert request.resource_class == "pre-commit"
    assert RepositoryWorkItemKind.VALIDATION.value == "repository.validation"
    with pytest.raises(RepositoryWorkItemError, match="contract version"):
        RepositoryWorkItemRequest.from_contract({"contract_version": "2"})


def test_nested_c03_resource_projection_survives_view_and_result_round_trip() -> None:
    engine = RepositoryEngine()
    deadline = datetime(2030, 1, 2, 3, 4, 5, tzinfo=UTC)
    opaque = "1234-5678-9012-3456"
    request = RepositoryWorkItemRequest.from_contract(
        {
            "request_id": opaque + "-request",
            "idempotency_key": opaque + "-idempotency",
            "operation": "build",
            "repository": {"repository_id": opaque},
            "base_ref": opaque,
            "base_sha": "c" * 40,
            "owner_id": opaque,
            "session_id": opaque,
            "tenant_id": "tenant-a",
            "target": {
                "kind": "inventory_alias",
                "alias": opaque,
            },
            "lane_id": opaque,
            "candidate_id": opaque,
            "generation_id": opaque,
            "retry_class": opaque,
            "correlation_id": opaque,
            "resources": {
                "resource_class": opaque,
                "concurrency_key": opaque,
                "fairness_group": opaque,
                "priority": 8,
                "cpu_weight": 7,
                "memory_mib": 2048,
                "disk_mib": 4096,
                "process_slots": 3,
                "host_labels": [opaque, "nodejs"],
                "preferred_target": {
                    "kind": "inventory_alias",
                    "alias": opaque,
                    "capability_labels": [opaque, "pnpm"],
                },
                "required_target": {
                    "kind": "inventory_alias",
                    "alias": opaque,
                    "capability_labels": [opaque],
                },
                "anti_affinity": [opaque, "agent-webui"],
                "queue_deadline": deadline.isoformat(),
                "disk_low_watermark_mib": 100,
                "disk_high_watermark_mib": 1000,
            },
        }
    )

    assert request.cpu_weight == 7
    assert request.memory_mib == 2048
    assert request.disk_mib == 4096
    assert request.process_slots == 3
    assert request.host_labels == (opaque, "nodejs")
    assert request.preferred_target.kind == "inventory_alias"
    assert request.preferred_target.alias == opaque
    assert request.preferred_target.capability_labels == (opaque, "pnpm")
    assert request.required_target is not None
    assert request.required_target.alias == opaque
    assert request.anti_affinity == (opaque, "agent-webui")
    assert request.queue_deadline == deadline
    assert request.disk_low_watermark_mib == 100
    assert request.disk_high_watermark_mib == 1000

    handle = submit_repository_work_item(engine, request)
    row = engine.nodes[handle.work_item_id]
    assert row["deadline_unix"] == deadline.timestamp()
    assert row["prio_bucket"] == 3
    metadata = row["metadata"]["repository_work_item"]
    assert metadata["cpu_weight"] == 7
    assert metadata["memory_mib"] == 2048
    assert metadata["disk_mib"] == 4096
    assert metadata["process_slots"] == 3
    assert all(value.startswith("opaque:v1:") for value in metadata["host_labels"])
    assert metadata["base_ref"].startswith("opaque:v1:")
    assert metadata["target_alias"].startswith("opaque:v1:")
    assert metadata["preferred_target"]["alias"].startswith("opaque:v1:")
    assert metadata["required_target"]["alias"].startswith("opaque:v1:")
    assert all(value.startswith("opaque:v1:") for value in metadata["anti_affinity"])
    assert metadata["queue_deadline"] == "2030-01-02T03:04:05Z"
    assert metadata["disk_low_watermark_mib"] == 100
    assert metadata["disk_high_watermark_mib"] == 1000

    view = get_repository_work_item(engine, handle.job_id, tenant="tenant-a")
    assert view is not None
    assert view.request_id == request.request_id
    assert view.repository_id == request.repository_id
    assert view.base_ref == opaque
    assert view.owner_id == request.owner_id
    assert view.session_id == request.session_id
    assert view.target_alias == opaque
    assert view.lane_id == opaque
    assert view.candidate_id == opaque
    assert view.generation_id == opaque
    assert view.retry_class == opaque
    assert view.correlation_id == opaque
    assert view.resource_class == opaque
    assert view.concurrency_key == opaque
    assert view.fairness_group == opaque
    assert view.priority == 8
    assert view.cpu_weight == request.cpu_weight
    assert view.memory_mib == request.memory_mib
    assert view.disk_mib == request.disk_mib
    assert view.process_slots == request.process_slots
    assert view.host_labels == request.host_labels
    assert view.preferred_target == request.preferred_target
    assert view.required_target == request.required_target
    assert view.anti_affinity == request.anti_affinity
    assert view.queue_deadline == request.queue_deadline
    assert view.disk_low_watermark_mib == request.disk_low_watermark_mib
    assert view.disk_high_watermark_mib == request.disk_high_watermark_mib

    claim = claim_repository_work_item(
        engine, handle.job_id, tenant="tenant-a", token="opaque-worker", now=10.0
    )
    assert claim is not None
    assert (
        commit_repository_work_item(
            engine,
            handle.job_id,
            claim,
            tenant="tenant-a",
            outcome="succeeded",
            result_ref="artifact:opaque",
            now=11.0,
        )
        == "committed"
    )
    terminal_view = get_repository_work_item(engine, handle.job_id, tenant="tenant-a")
    assert terminal_view is not None
    result = repository_result_from_view(terminal_view)
    assert result.request_id == request.request_id
    assert result.repository_id == request.repository_id
    assert result.target_alias == opaque
    assert result.resource_class == opaque
    assert result.concurrency_key == opaque
    assert result.fairness_group == opaque
    assert result.priority == 8
    assert result.cpu_weight == request.cpu_weight
    assert result.memory_mib == request.memory_mib
    assert result.disk_mib == request.disk_mib
    assert result.process_slots == request.process_slots
    assert result.host_labels == request.host_labels
    assert result.preferred_target == request.preferred_target
    assert result.required_target == request.required_target
    assert result.anti_affinity == request.anti_affinity
    assert result.queue_deadline == request.queue_deadline
    assert result.disk_low_watermark_mib == request.disk_low_watermark_mib
    assert result.disk_high_watermark_mib == request.disk_high_watermark_mib

    changed = request.model_copy(update={"cpu_weight": 8})
    assert changed.immutable_digest() != request.immutable_digest()
    with pytest.raises(RepositoryWorkItemConflict, match="idempotency key"):
        submit_repository_work_item(engine, changed)
    with pytest.raises(ValueError):
        RepositoryWorkItemRequest.model_validate(
            {**request.model_dump(), "cpu_weight": "7"}
        )
    with pytest.raises(ValueError):
        RepositoryWorkItemRequest.model_validate(
            {**request.model_dump(), "priority": "8"}
        )
    with pytest.raises(ValueError):
        RepositoryWorkItemRequest.model_validate(
            {**request.model_dump(), "priority": True}
        )


def _trusted_resolved_request(
    *, key: str = "resolved-profile"
) -> RepositoryWorkItemRequest:
    return _request(key=key).model_copy(
        update={
            "profile_version": "1",
            "resolved_profile_authority": (
                "repository_manager:resource_profile_registry:v1"
            ),
            "disk_policy_key": "light-check-v1",
            "fairness_cost": 1,
            # These profiles intentionally have no concurrency or disk
            # hysteresis limits; explicit nulls are still part of the trusted
            # resolved projection and must not be confused with missing data.
            "concurrency_limit": None,
            "disk_low_watermark_mib": None,
            "disk_high_watermark_mib": None,
        }
    )


def test_public_submission_cannot_self_stamp_resolved_profile_authority() -> None:
    engine = RepositoryEngine()
    request = _trusted_resolved_request()
    with pytest.raises(
        RepositoryWorkItemError, match="reserved for the trusted RM projection"
    ):
        submit_repository_work_item(engine, request)


def test_trusted_resolved_projection_preserves_explicit_null_policy_fields() -> None:
    engine = RepositoryEngine()
    request = _trusted_resolved_request(key="resolved-null-policy")
    handle = submit_repository_work_item(
        engine, request, resolved_profile_projection=True
    )
    extension = engine.nodes[handle.work_item_id]["metadata"]["repository_work_item"][
        "resource_reservation"
    ]
    assert extension["resolved_profile_authority"] == (
        "repository_manager:resource_profile_registry:v1"
    )
    assert extension["profile_version"].startswith("opaque:v1:")
    assert extension["concurrency_limit"] is None
    assert extension["disk_low_watermark_mib"] is None
    assert extension["disk_high_watermark_mib"] is None


def test_trusted_resolved_projection_rejects_partial_authority_marker() -> None:
    engine = RepositoryEngine()
    raw = _request(key="partial-resolved-profile").model_dump(mode="python")
    raw["resolved_profile_authority"] = (
        "repository_manager:resource_profile_registry:v1"
    )
    with pytest.raises(RepositoryWorkItemError, match="projection is incomplete"):
        submit_repository_work_item(engine, raw, resolved_profile_projection=True)


def test_c01_consent_is_projected_without_broadening_branch_land_or_repair() -> None:
    engine = RepositoryEngine()
    release_values = _request(key="release-consent").model_dump(mode="python")
    release_values["operation"] = RepositoryOperation.RELEASE
    with pytest.raises(ValueError, match="explicit push consent"):
        RepositoryWorkItemRequest.model_validate(release_values)

    consent = RepositoryConsentPolicy(
        allow_push=True,
        risk_acknowledged=True,
        risk_marker="1234-5678-9012-3456",
    )
    release_values["consent"] = consent
    release = RepositoryWorkItemRequest.model_validate(release_values)
    release_handle = submit_repository_work_item(engine, release, now=42.0)
    release_node = engine.nodes[release_handle.work_item_id]
    assert release_node["consent_required"] is True
    assert release_node["consent_scope"] == "repository:release"
    assert release_node["consent_subject"] == release.repository_id
    assert release_node["consent_granted_at"] == 42.0

    release_view = get_repository_work_item(
        engine, release_handle.job_id, tenant="tenant-a"
    )
    assert release_view is not None
    assert release_view.consent == consent
    with pytest.raises(ValueError, match="must be terminal"):
        repository_result_from_view(release_view)
    claim = claim_repository_work_item(
        engine,
        release_handle.job_id,
        tenant="tenant-a",
        token="release-worker",
        now=43.0,
    )
    assert claim is not None
    assert (
        commit_repository_work_item(
            engine,
            release_handle.job_id,
            claim,
            tenant="tenant-a",
            outcome="succeeded",
            result_ref="artifact:release",
            now=44.0,
        )
        == "committed"
    )
    terminal = get_repository_work_item(
        engine, release_handle.job_id, tenant="tenant-a"
    )
    assert terminal is not None
    assert repository_result_from_view(terminal).consent == consent

    branch_land_values = _request(key="branch-land-consent").model_dump(mode="python")
    branch_land_values["operation"] = RepositoryOperation.BRANCH_LAND
    branch_land = RepositoryWorkItemRequest.model_validate(branch_land_values)
    assert branch_land.consent == RepositoryConsentPolicy()

    repair_values = _request(key="repair-no-consent").model_dump(mode="python")
    repair_values["operation"] = RepositoryOperation.REPAIR
    repair = RepositoryWorkItemRequest.model_validate(repair_values)
    repair_handle = submit_repository_work_item(engine, repair)
    assert engine.nodes[repair_handle.work_item_id]["consent_required"] is False


def test_repository_reads_route_through_bound_work_item_authority() -> None:
    engine = RepositoryEngine()
    submitted = submit_repository_work_item(engine, _request(key="authority-route"))
    authority = AuthorityProxy(engine, tenant="tenant-a")
    engine._work_item_engine = authority

    listed = list_repository_work_items(engine, tenant="tenant-a")
    assert [item.job_id for item in listed] == [submitted.job_id]
    assert authority.query_calls >= 1
    assert (
        get_repository_work_item(engine, submitted.job_id, tenant="tenant-a")
        is not None
    )

    with pytest.raises(RepositoryWorkItemError, match="does not match"):
        get_repository_work_item(engine, submitted.job_id, tenant="tenant-b")
    with pytest.raises(RepositoryWorkItemError, match="does not match"):
        list_repository_work_items(engine, tenant="tenant-b")


def test_cursor_pages_past_old_submitted_prefix_for_filters_and_reconcile() -> None:
    engine = RepositoryEngine()
    parent = submit_repository_work_item(engine, _request(key="cursor-parent"))
    child_request = _request(key="cursor-child", operation="validation").model_copy(
        update={
            "repository_id": "target-repository",
            "lane_id": "target-lane",
            "dependencies": (parent.work_item_id,),
        }
    )
    child_id = _insert_repository_row(
        engine,
        child_request,
        created_at=10_010.0,
        dependencies=(parent.work_item_id,),
    )

    # More than one requested page of older submitted rows deliberately does
    # not match the target repository. Every query remains page-bounded, while
    # keyset pagination must reach the newer child rather than returning an
    # empty filtered result from the first prefix.
    for index in range(1001):
        old_request = _request(key=f"cursor-old-{index}").model_copy(
            update={"repository_id": "historical-repository"}
        )
        _insert_repository_row(engine, old_request, created_at=float(index))

    listed = list_repository_work_items(
        engine,
        tenant="tenant-a",
        repository_id="target-repository",
        lane_id="target-lane",
        limit=1,
    )
    assert [item.work_item_id for item in listed] == [child_id]
    assert child_id in engine.nodes[parent.work_item_id]["downstream_ids"]
    assert engine.nodes[child_id]["status"] == "submitted"
    assert reconcile_repository_work_items(engine, tenant="tenant-a", limit=100) == 0


def test_submission_cannot_bypass_bound_tenant_authority() -> None:
    engine = RepositoryEngine()
    engine._work_item_engine = AuthorityProxy(engine, tenant="tenant-a")
    with pytest.raises(RepositoryWorkItemError, match="does not match"):
        submit_repository_work_item(engine, _request(tenant="tenant-b", key="bound"))


def test_submission_is_atomic_idempotent_and_changed_input_refused() -> None:
    engine = RepositoryEngine()
    first = submit_repository_work_item(engine, _request())
    second = submit_repository_work_item(engine, _request())
    assert first.job_id == second.job_id
    assert first.work_item_id == second.work_item_id
    assert not first.deduplicated
    assert second.deduplicated
    assert len(engine.nodes) == 1
    assert engine.nodes[first.work_item_id]["label"] == "WorkItem"
    assert engine.nodes[first.work_item_id]["node_type"] == "WorkItem"

    with pytest.raises(RepositoryWorkItemConflict, match="idempotency key"):
        submit_repository_work_item(engine, _request(operation="validation"))

    other = submit_repository_work_item(engine, _request(tenant="tenant-b"))
    assert other.job_id != first.job_id
    assert get_repository_work_item(engine, first.job_id, tenant="tenant-b") is None

    # This deterministic UUID has an all-numeric prefix that the generic
    # privacy guard can conservatively classify as a credit-card-shaped
    # substring.  Durable identity must still round-trip from the WorkItem ID.
    privacy_edge = _request(key="historical-terminal-288")
    edge_first = submit_repository_work_item(engine, privacy_edge)
    edge_second = submit_repository_work_item(engine, privacy_edge)
    assert edge_second.deduplicated
    assert edge_first.job_id == edge_second.job_id


def test_repository_dependencies_cannot_cross_tenant_or_reference_unknown_jobs() -> (
    None
):
    engine = RepositoryEngine()
    foreign_parent = submit_repository_work_item(
        engine, _request(tenant="tenant-b", key="foreign-parent")
    )
    with pytest.raises(RepositoryWorkItemError, match="missing or outside"):
        submit_repository_work_item(
            engine,
            _request(key="cross-tenant-child", operation="validation").model_copy(
                update={"dependencies": (foreign_parent.job_id,)}
            ),
        )

    missing_parent = repository_job_id("tenant-a", "missing-parent")
    with pytest.raises(RepositoryWorkItemError, match="missing or outside"):
        submit_repository_work_item(
            engine,
            _request(key="missing-parent-child", operation="validation").model_copy(
                update={"dependencies": (missing_parent,)}
            ),
        )
    assert engine.nodes[foreign_parent.work_item_id]["downstream_ids"] == []
    assert len(engine.nodes) == 1


def test_full_work_item_ids_round_trip_across_dependency_and_lifecycle_apis() -> None:
    engine = RepositoryEngine()
    parent = submit_repository_work_item(engine, _request(key="full-id-parent"))
    assert parent.work_item_id.count("-") == 4

    child = submit_repository_work_item(
        engine,
        _request(key="full-id-child", operation="validation").model_copy(
            update={"dependencies": (parent.work_item_id,)}
        ),
    )
    parent_view = get_repository_work_item(
        engine, parent.work_item_id, tenant="tenant-a"
    )
    assert parent_view is not None
    assert parent_view.job_id == parent.job_id
    assert child.work_item_id in engine.nodes[parent.work_item_id]["downstream_ids"]
    assert (
        claim_repository_work_item(
            engine,
            child.work_item_id,
            tenant="tenant-a",
            token="worker-child",
            now=100.0,
        )
        is None
    )

    parent_claim = claim_repository_work_item(
        engine,
        parent.work_item_id,
        tenant="tenant-a",
        token="worker-parent",
        now=100.0,
    )
    assert parent_claim is not None
    assert (
        commit_repository_work_item(
            engine,
            parent.work_item_id,
            parent_claim,
            tenant="tenant-a",
            outcome="succeeded",
            result_ref="artifact:full-id-parent",
            now=101.0,
        )
        == "committed"
    )
    child_claim = claim_repository_work_item(
        engine,
        child.work_item_id,
        tenant="tenant-a",
        token="worker-child",
        now=102.0,
    )
    assert child_claim is not None

    cancellable = submit_repository_work_item(engine, _request(key="full-id-cancel"))
    assert cancel_repository_work_item(
        engine,
        cancellable.work_item_id,
        tenant="tenant-a",
        reason="operator",
        now=103.0,
    )
    cancelled = get_repository_work_item(
        engine, cancellable.work_item_id, tenant="tenant-a"
    )
    assert cancelled is not None
    assert cancelled.state == RepositoryJobState.CANCELLED


def test_concurrent_first_writers_share_one_atomic_work_item() -> None:
    # Both callers complete their pre-read and arrive at the native create
    # barrier before either can inspect/insert the node.  This makes the
    # losing-writer result deterministic rather than depending on scheduling.
    engine = RepositoryEngine(create_barrier=threading.Barrier(2))
    request = _request(key="racing-submit")
    handles: list[Any] = []
    errors: list[BaseException] = []

    def submit() -> None:
        try:
            handles.append(submit_repository_work_item(engine, request))
        except BaseException as exc:  # pragma: no cover - diagnostic guard
            errors.append(exc)

    workers = [threading.Thread(target=submit) for _ in range(2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=5.0)
        assert not worker.is_alive(), "atomic submission barrier did not release"

    assert not errors
    assert len(handles) == 2
    assert {handle.work_item_id for handle in handles} == {handles[0].work_item_id}
    assert {handle.deduplicated for handle in handles} == {False, True}
    assert len(engine.nodes) == 1


def test_dependency_waits_without_lease_then_releases_on_parent_success() -> None:
    engine = RepositoryEngine()
    parent = submit_repository_work_item(engine, _request(key="parent"))
    child = submit_repository_work_item(
        engine,
        _request(key="child", operation="validation").model_copy(
            update={"dependencies": (parent.job_id,)}
        ),
    )
    assert child.state == RepositoryJobState.SUBMITTED
    assert (
        claim_repository_work_item(
            engine, child.job_id, tenant="tenant-a", token="worker-child", now=100.0
        )
        is None
    )

    parent_claim = claim_repository_work_item(
        engine, parent.job_id, tenant="tenant-a", token="worker-parent", now=100.0
    )
    assert parent_claim is not None
    assert (
        commit_repository_work_item(
            engine,
            parent.job_id,
            parent_claim,
            tenant="tenant-a",
            outcome="succeeded",
            result_ref="artifact:parent",
            now=101.0,
        )
        == "committed"
    )
    child_claim = claim_repository_work_item(
        engine, child.job_id, tenant="tenant-a", token="worker-child", now=102.0
    )
    assert child_claim is not None


def test_dependency_parent_success_between_create_and_index_is_reconciled() -> None:
    engine = RepositoryEngine()
    parent = submit_repository_work_item(engine, _request(key="interleave-parent"))

    def commit_parent_before_child_index() -> None:
        parent_claim = wi.claim_specific(
            engine,
            parent.work_item_id,
            token="worker-parent",
            now=100.0,
        )
        assert parent_claim is not None
        assert (
            commit_repository_work_item(
                engine,
                parent.job_id,
                parent_claim,
                tenant="tenant-a",
                outcome="succeeded",
                result_ref="artifact:interleave-parent",
                now=101.0,
            )
            == "committed"
        )

    # ``submit_work_item`` creates the child, links it, then appends it to the
    # parent's reverse index.  Fire the parent commit after the graph edge but
    # before that index append to model the production race exactly.
    engine.on_link = commit_parent_before_child_index
    child = submit_repository_work_item(
        engine,
        _request(key="interleave-child", operation="validation").model_copy(
            update={"dependencies": (parent.job_id,)}
        ),
    )
    child_view = get_repository_work_item(engine, child.job_id, tenant="tenant-a")
    assert child_view is not None
    assert child_view.state == RepositoryJobState.READY
    child_claim = claim_repository_work_item(
        engine, child.job_id, tenant="tenant-a", token="worker-child", now=102.0
    )
    assert child_claim is not None


def test_multiple_dependency_interleave_reconciles_missed_parent_before_indexed_commit() -> (
    None
):
    engine = RepositoryEngine()
    parents = [
        submit_repository_work_item(engine, _request(key="multi-parent-a")),
        submit_repository_work_item(engine, _request(key="multi-parent-b")),
    ]
    missed_parent, indexed_parent = sorted(
        parents, key=lambda handle: handle.work_item_id
    )
    indexed_claim = claim_repository_work_item(
        engine,
        indexed_parent.job_id,
        tenant="tenant-a",
        token="worker-indexed-parent",
        now=100.0,
    )
    assert indexed_claim is not None

    def commit_missed_parent_before_reverse_index() -> None:
        missed_claim = wi.claim_specific(
            engine,
            missed_parent.work_item_id,
            token="worker-missed-parent",
            now=100.0,
        )
        assert missed_claim is not None
        assert (
            commit_repository_work_item(
                engine,
                missed_parent.job_id,
                missed_claim,
                tenant="tenant-a",
                outcome="succeeded",
                result_ref="artifact:missed-parent",
                now=101.0,
            )
            == "committed"
        )

    # The first dependency edge is deliberately the parent that commits
    # before its reverse index append. The other parent stays leased until
    # after admission, so reconciliation must repair 2 -> 1 before its
    # indexed commit performs the final decrement.
    engine.on_link = commit_missed_parent_before_reverse_index
    child = submit_repository_work_item(
        engine,
        _request(key="multi-child", operation="validation").model_copy(
            update={
                "dependencies": (
                    missed_parent.job_id,
                    indexed_parent.job_id,
                )
            }
        ),
    )
    blocked_view = get_repository_work_item(engine, child.job_id, tenant="tenant-a")
    assert blocked_view is not None
    assert blocked_view.state == RepositoryJobState.SUBMITTED
    assert engine.nodes[child.work_item_id]["dep_count"] == 1

    assert (
        commit_repository_work_item(
            engine,
            indexed_parent.job_id,
            indexed_claim,
            tenant="tenant-a",
            outcome="succeeded",
            result_ref="artifact:indexed-parent",
            now=102.0,
        )
        == "committed"
    )
    ready_view = get_repository_work_item(engine, child.job_id, tenant="tenant-a")
    assert ready_view is not None
    assert ready_view.state == RepositoryJobState.READY
    assert ready_view.attempt == 0


def test_dependency_reconciliation_retries_after_competing_count_cas() -> None:
    engine = RepositoryEngine()
    parent = submit_repository_work_item(engine, _request(key="cas-parent"))
    child_request = _request(key="cas-child", operation="validation").model_copy(
        update={"dependencies": (parent.job_id,)}
    )
    child_id = repository_work_item_id(
        repository_job_id(child_request.tenant_id, child_request.idempotency_key)
    )
    injected = False

    def commit_parent_and_inject_competing_reconcile() -> None:
        parent_claim = wi.claim_specific(
            engine,
            parent.work_item_id,
            token="worker-cas-parent",
            now=100.0,
        )
        assert parent_claim is not None
        assert (
            commit_repository_work_item(
                engine,
                parent.job_id,
                parent_claim,
                tenant="tenant-a",
                outcome="succeeded",
                result_ref="artifact:cas-parent",
                now=101.0,
            )
            == "committed"
        )

        def compete(
            node_id: str, conditions: dict[str, Any], updates: dict[str, Any]
        ) -> None:
            nonlocal injected
            if node_id != child_id or updates.get("status") != "ready":
                return
            injected = True
            engine.cas_hook = None
            assert engine.compare_and_set_node_fields(
                child_id,
                {"status": "submitted", "dep_count": 1},
                {"status": "submitted", "dep_count": 0, "updated_at": 101.5},
            )

        engine.cas_hook = compete

    # The parent commits after the child edge exists but before the reverse
    # index append. The competing CAS changes only the dependency count, so
    # the first readiness CAS loses and must reread/retry to publish ready.
    engine.on_link = commit_parent_and_inject_competing_reconcile
    child = submit_repository_work_item(engine, child_request)
    assert child.work_item_id == child_id
    assert injected
    child_view = get_repository_work_item(engine, child.job_id, tenant="tenant-a")
    assert child_view is not None
    assert child_view.state == RepositoryJobState.READY


def test_fresh_adapter_reconciles_child_created_before_reverse_index() -> None:
    engine = RepositoryEngine()
    parent = submit_repository_work_item(engine, _request(key="restart-parent"))
    child_request = _request(key="restart-child", operation="validation").model_copy(
        update={"dependencies": (parent.job_id,)}
    )

    def crash_after_durable_create() -> None:
        raise RuntimeError("simulated process stop after WorkItem create")

    engine.on_create = crash_after_durable_create
    with pytest.raises(RuntimeError, match="process stop"):
        submit_repository_work_item(engine, child_request)
    child_job_id = repository_job_id(
        child_request.tenant_id, child_request.idempotency_key
    )
    child_id = repository_work_item_id(child_job_id)
    assert child_id in engine.nodes
    assert child_id not in engine.nodes[parent.work_item_id]["downstream_ids"]

    # Simulate a fresh adapter/process: listing is a durable restart repair
    # entry point and backfills the missing reverse index idempotently.
    listed = list_repository_work_items(engine, tenant="tenant-a")
    assert {item.job_id for item in listed} == {parent.job_id, child_job_id}
    assert child_id in engine.nodes[parent.work_item_id]["downstream_ids"]
    assert (
        child_id,
        parent.work_item_id,
        wi._task_depends_on_edge_type(),
    ) in engine.edges
    # Listing already performed the idempotent repair; the mutation count is
    # therefore zero on the second pass.
    assert reconcile_repository_work_items(engine, tenant="tenant-a") == 0

    parent_claim = claim_repository_work_item(
        engine, parent.job_id, tenant="tenant-a", token="worker-restart", now=100.0
    )
    assert parent_claim is not None
    assert (
        commit_repository_work_item(
            engine,
            parent.job_id,
            parent_claim,
            tenant="tenant-a",
            outcome="succeeded",
            result_ref="artifact:restart-parent",
            now=101.0,
        )
        == "committed"
    )
    child_view = get_repository_work_item(engine, child_job_id, tenant="tenant-a")
    assert child_view is not None
    assert child_view.state == RepositoryJobState.READY


def test_claim_target_repair_bypasses_terminal_history_prefix() -> None:
    engine = RepositoryEngine()
    # A global oldest-first query must not hide a newer crash-window child
    # behind historical terminal jobs.  The production repair query narrows
    # to submitted candidates, while a specific claim repairs its target
    # directly regardless of any list cap.
    historical = 1001
    for index in range(historical):
        historical_id = f"workitem:repository_manager:{index:032x}"
        engine.nodes[historical_id] = {
            "label": "WorkItem",
            "kind": RepositoryWorkItemKind.BUILD.value,
            "tenant": "tenant-a",
            "status": "succeeded",
            "created_at": float(index),
        }

    parent = submit_repository_work_item(engine, _request(key="history-parent"))
    child_request = _request(key="history-child", operation="validation").model_copy(
        update={"dependencies": (parent.job_id,)}
    )

    def crash_after_durable_create() -> None:
        raise RuntimeError("simulated process stop after WorkItem create")

    engine.on_create = crash_after_durable_create
    with pytest.raises(RuntimeError, match="process stop"):
        submit_repository_work_item(engine, child_request)
    child_job_id = repository_job_id(
        child_request.tenant_id, child_request.idempotency_key
    )
    child_id = repository_work_item_id(child_job_id)
    assert child_id not in engine.nodes[parent.work_item_id]["downstream_ids"]

    # Claim-specific repair must find and index this child even though a
    # historical terminal prefix is larger than the normal reconciliation
    # limit.  It remains blocked until the parent succeeds.
    assert (
        claim_repository_work_item(
            engine, child_job_id, tenant="tenant-a", token="worker-child", now=100.0
        )
        is None
    )
    assert child_id in engine.nodes[parent.work_item_id]["downstream_ids"]
    assert engine.nodes[child_id]["status"] == "submitted"

    parent_claim = claim_repository_work_item(
        engine, parent.job_id, tenant="tenant-a", token="worker-parent", now=101.0
    )
    assert parent_claim is not None
    assert (
        commit_repository_work_item(
            engine,
            parent.job_id,
            parent_claim,
            tenant="tenant-a",
            outcome="succeeded",
            result_ref="artifact:history-parent",
            now=102.0,
        )
        == "committed"
    )
    child_claim = claim_repository_work_item(
        engine, child_job_id, tenant="tenant-a", token="worker-child", now=103.0
    )
    assert child_claim is not None


def test_failed_dependency_keeps_downstream_blocked_for_reconciliation() -> None:
    engine = RepositoryEngine()
    parent = submit_repository_work_item(engine, _request(key="failed-parent"))
    child = submit_repository_work_item(
        engine,
        _request(key="blocked-child", operation="validation").model_copy(
            update={"dependencies": (parent.job_id,)}
        ),
    )
    parent_claim = claim_repository_work_item(
        engine, parent.job_id, tenant="tenant-a", token="worker-parent", now=100.0
    )
    assert parent_claim is not None
    assert (
        commit_repository_work_item(
            engine,
            parent.job_id,
            parent_claim,
            tenant="tenant-a",
            outcome="failed",
            failure_class="validation_candidate_failure",
            retryable=False,
            now=101.0,
        )
        == "committed"
    )
    child_view = get_repository_work_item(engine, child.job_id, tenant="tenant-a")
    assert child_view is not None
    assert child_view.state == RepositoryJobState.SUBMITTED
    assert child_view.attempt == 0
    assert (
        claim_repository_work_item(
            engine, child.job_id, tenant="tenant-a", token="worker-child", now=102.0
        )
        is None
    )


def test_cancellation_before_lease_and_terminal_race_are_durable() -> None:
    engine = RepositoryEngine()
    before_lease = submit_repository_work_item(engine, _request(key="cancel-before"))
    assert cancel_repository_work_item(
        engine, before_lease.job_id, tenant="tenant-a", reason="operator", now=10.0
    )
    cancelled_view = get_repository_work_item(
        engine, before_lease.job_id, tenant="tenant-a"
    )
    assert cancelled_view is not None
    assert cancelled_view.state == RepositoryJobState.CANCELLED
    assert (
        claim_repository_work_item(
            engine, before_lease.job_id, tenant="tenant-a", token="worker", now=11.0
        )
        is None
    )

    running = submit_repository_work_item(engine, _request(key="cancel-running"))
    claim = claim_repository_work_item(
        engine, running.job_id, tenant="tenant-a", token="worker", now=20.0
    )
    assert claim is not None
    assert cancel_repository_work_item(
        engine, running.job_id, tenant="tenant-a", reason="operator", now=21.0
    )
    assert (
        commit_repository_work_item(
            engine,
            running.job_id,
            claim,
            tenant="tenant-a",
            outcome="succeeded",
            result_ref="artifact:late",
            now=22.0,
        )
        == "noop"
    )
    running_view = get_repository_work_item(engine, running.job_id, tenant="tenant-a")
    assert (
        running_view is not None and running_view.state == RepositoryJobState.CANCELLED
    )


def test_checkpoint_and_retry_retain_identity_and_fence_attempts() -> None:
    engine = RepositoryEngine()
    submitted = submit_repository_work_item(
        engine, _request(key="retry-two"), max_attempts=2
    )
    first = claim_repository_work_item(
        engine, submitted.job_id, tenant="tenant-a", token="worker-a", now=100.0
    )
    assert first is not None
    assert checkpoint_repository_work_item(
        engine,
        submitted.job_id,
        first,
        "checkpoint:after-feedback",
        tenant="tenant-a",
        now=101.0,
    )
    assert (
        checkpoint_repository_work_item(
            engine,
            submitted.job_id,
            {**first, "fencing_token": int(first["fencing_token"]) - 1},
            "checkpoint:stale",
            tenant="tenant-a",
            now=102.0,
        )
        is False
    )
    assert (
        commit_repository_work_item(
            engine,
            submitted.job_id,
            first,
            tenant="tenant-a",
            outcome="failed",
            failure_class="worker_environment_failure",
            retryable=True,
            now=103.0,
        )
        == "retry_scheduled"
    )
    second = claim_repository_work_item(
        engine, submitted.job_id, tenant="tenant-a", token="worker-b", now=104.0
    )
    assert second is not None
    assert second["work_item_id"] == first["work_item_id"] == submitted.work_item_id
    assert second["attempt"] == 2


def test_claim_cannot_be_replayed_against_another_work_item() -> None:
    engine = RepositoryEngine()
    first = submit_repository_work_item(engine, _request(key="claim-owner-a"))
    second = submit_repository_work_item(engine, _request(key="claim-owner-b"))
    claim = claim_repository_work_item(
        engine, first.job_id, tenant="tenant-a", token="worker", now=100.0
    )
    assert claim is not None

    with pytest.raises(RepositoryWorkItemError, match="does not belong"):
        heartbeat_repository_work_item(
            engine,
            second.job_id,
            claim,
            tenant="tenant-a",
            now=101.0,
        )
    with pytest.raises(RepositoryWorkItemError, match="does not belong"):
        checkpoint_repository_work_item(
            engine,
            second.job_id,
            claim,
            "checkpoint:cross-item",
            tenant="tenant-a",
            now=101.0,
        )
    with pytest.raises(RepositoryWorkItemError, match="does not belong"):
        commit_repository_work_item(
            engine,
            second.job_id,
            claim,
            tenant="tenant-a",
            outcome="succeeded",
            result_ref="artifact:cross-item",
            now=101.0,
        )
    second_view = get_repository_work_item(engine, second.job_id, tenant="tenant-a")
    assert second_view is not None
    assert second_view.state == RepositoryJobState.READY


def test_restart_view_filters_tenant_and_stale_fence_cannot_publish() -> None:
    engine = RepositoryEngine()
    submitted = submit_repository_work_item(engine, _request())
    # A fresh adapter/process reads only durable graph state.
    view = get_repository_work_item(engine, submitted.job_id, tenant="tenant-a")
    assert view is not None
    claim = claim_repository_work_item(
        engine,
        submitted.job_id,
        tenant="tenant-a",
        token="worker-a",
        now=10.0,
        lease_ttl_s=5.0,
    )
    assert claim is not None
    stale = dict(claim)
    current = claim_repository_work_item(
        engine, submitted.job_id, tenant="tenant-a", token="worker-b", now=20.0
    )
    assert current is not None
    assert (
        commit_repository_work_item(
            engine,
            submitted.job_id,
            stale,
            tenant="tenant-a",
            outcome="succeeded",
            result_ref="artifact:stale",
            now=21.0,
        )
        == "fenced"
    )
    assert get_repository_work_item(engine, submitted.job_id, tenant="tenant-b") is None
    assert [
        item.job_id for item in list_repository_work_items(engine, tenant="tenant-a")
    ] == [submitted.job_id]


def test_checkpoint_retry_dead_letter_and_typed_terminal_result() -> None:
    engine = RepositoryEngine()
    submitted = submit_repository_work_item(
        engine, _request(key="retry"), max_attempts=1
    )
    claim = claim_repository_work_item(
        engine, submitted.job_id, tenant="tenant-a", token="worker", now=100.0
    )
    assert claim is not None
    assert (
        commit_repository_work_item(
            engine,
            submitted.job_id,
            claim,
            tenant="tenant-a",
            outcome="failed",
            failure_class="validation_candidate_failure",
            retryable=True,
            now=101.0,
        )
        == "dead_letter"
    )
    view = get_repository_work_item(engine, submitted.job_id, tenant="tenant-a")
    assert view is not None and view.state == RepositoryJobState.DEAD_LETTER
    result = repository_result_from_view(view)
    assert result.state == RepositoryJobState.DEAD_LETTER
    assert result.failure_class == "validation_candidate_failure"
    assert result.target_kind == "local"
    assert result.retry_class is None


def test_operation_payload_is_atomic_private_summary_and_exact_owner_read() -> None:
    engine = RepositoryEngine()
    payload = _operation_payload()
    request = _request(key="typed-payload").model_copy(
        update={"operation_payload": payload}
    )
    submitted = submit_repository_work_item(engine, request)
    row = engine.nodes[submitted.work_item_id]
    record = row["metadata"]["repository_work_item"]
    assert record["resource_reservation"]["schema_version"] == "1"
    assert record["operation_payload"]["kind"] == "repository.build-execution/v1"
    assert record["operation_payload_digest"] == payload.payload_digest
    assert row["correlation_id"] == request.request_id
    assert "argv" not in row["correlation_id"]

    view = get_repository_work_item(engine, submitted.job_id, tenant="tenant-a")
    assert view is not None
    assert view.operation_payload_kind == "repository.build-execution/v1"
    assert view.operation_payload_version == "1"
    assert view.operation_payload_digest == payload.payload_digest
    assert "operation_payload" not in view.model_dump(mode="json")
    listed = list_repository_work_items(engine, tenant="tenant-a")
    assert len(listed) == 1
    assert "operation_payload" not in listed[0].model_dump(mode="json")

    assert (
        get_repository_operation_payload(
            engine,
            submitted.job_id,
            tenant="tenant-a",
            owner_id="actor-a",
        )
        == payload
    )
    restarted = RepositoryEngine()
    restarted.nodes = deepcopy(engine.nodes)
    assert (
        get_repository_operation_payload(
            restarted,
            submitted.job_id,
            tenant="tenant-a",
            owner_id="actor-a",
        )
        == payload
    )
    assert (
        get_repository_operation_payload(
            engine,
            submitted.job_id,
            tenant="tenant-b",
            owner_id="actor-a",
        )
        is None
    )
    claim = claim_repository_work_item(
        engine, submitted.job_id, tenant="tenant-a", token="typed-worker", now=10.0
    )
    assert claim is not None
    assert (
        commit_repository_work_item(
            engine,
            submitted.job_id,
            claim,
            tenant="tenant-a",
            outcome="succeeded",
            result_ref="artifact:typed",
            now=11.0,
        )
        == "committed"
    )
    terminal = get_repository_work_item(engine, submitted.job_id, tenant="tenant-a")
    assert terminal is not None
    result = repository_result_from_view(terminal)
    assert result.operation_payload_kind == "repository.build-execution/v1"
    assert result.operation_payload_version == "1"
    assert result.operation_payload_digest == payload.payload_digest
    assert "operation_payload" not in result.model_dump(mode="json")
    assert (
        get_repository_operation_payload(
            engine,
            submitted.job_id,
            tenant="tenant-a",
            owner_id="another-owner",
        )
        is None
    )


def test_claim_bound_exact_read_rejects_public_and_stale_worker_scopes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = RepositoryEngine()
    clock = 100.0
    monkeypatch.setattr(rwi.time, "time", lambda: clock)
    payload = _operation_payload()
    submitted = submit_repository_work_item(
        engine,
        _request(key="claim-bound-payload").model_copy(
            update={"operation_payload": payload}
        ),
    )
    with pytest.raises(RepositoryWorkItemError, match="native worker claim"):
        get_repository_operation_payload_for_claim(
            engine,
            submitted.job_id,
            tenant="tenant-a",
            claim={"work_item_id": submitted.work_item_id},
        )

    first = claim_repository_work_item(
        engine,
        submitted.job_id,
        tenant="tenant-a",
        token="typed-worker-a",
        now=clock,
    )
    assert first is not None
    assert (
        get_repository_operation_payload_for_claim(
            engine,
            submitted.job_id,
            tenant="tenant-a",
            claim=first,
        )
        == payload
    )

    first_epoch = int(first["lease_epoch"])
    second = {
        **first,
        "lease_owner": "typed-worker-b",
        "lease_epoch": first_epoch + 1,
        "fence_token": first_epoch + 1,
        "fencing_token": first_epoch + 1,
        "attempt": int(first["attempt"]) + 1,
    }
    engine.nodes[submitted.work_item_id].update(
        status="running",
        lease_owner=second["lease_owner"],
        lease_epoch=second["lease_epoch"],
        fencing_token=second["fencing_token"],
        attempt=second["attempt"],
        lease_expires_at=clock + 300.0,
    )
    with pytest.raises(RepositoryWorkItemError, match="stale"):
        get_repository_operation_payload_for_claim(
            engine,
            submitted.job_id,
            tenant="tenant-a",
            claim=first,
        )
    assert (
        get_repository_operation_payload_for_claim(
            engine,
            submitted.job_id,
            tenant="tenant-a",
            claim=second,
        )
        == payload
    )


def test_claim_bound_exact_read_rejects_reclaim_between_authority_and_row_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = RepositoryEngine()
    clock = 100.0
    monkeypatch.setattr(rwi.time, "time", lambda: clock)
    payload = _operation_payload()
    submitted = submit_repository_work_item(
        engine,
        _request(key="claim-read-race").model_copy(
            update={"operation_payload": payload}
        ),
    )
    claim = claim_repository_work_item(
        engine,
        submitted.job_id,
        tenant="tenant-a",
        token="typed-worker-a",
        now=clock,
    )
    assert claim is not None

    original_get = rwi.get_work_item
    reads = 0

    def reclaim_before_latest_row(
        engine_arg: Any, item_id: str
    ) -> dict[str, Any] | None:
        nonlocal reads
        reads += 1
        if reads == 2:
            node = engine.nodes[item_id]
            next_epoch = int(node["lease_epoch"]) + 1
            node.update(
                status="running",
                lease_owner="typed-worker-b",
                lease_epoch=next_epoch,
                fencing_token=next_epoch,
                attempt=int(node["attempt"]) + 1,
            )
        return original_get(engine_arg, item_id)

    monkeypatch.setattr(rwi, "get_work_item", reclaim_before_latest_row)
    with pytest.raises(RepositoryWorkItemError, match="stale"):
        get_repository_operation_payload_for_claim(
            engine,
            submitted.job_id,
            tenant="tenant-a",
            claim=claim,
        )
    assert reads == 2


def test_claim_bound_exact_read_enforces_expiry_boundaries_and_malformed_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = RepositoryEngine()
    clock = 100.0
    monkeypatch.setattr(rwi.time, "time", lambda: clock)
    payload = _operation_payload()
    submitted = submit_repository_work_item(
        engine,
        _request(key="claim-expiry-payload").model_copy(
            update={"operation_payload": payload}
        ),
    )
    claim = claim_repository_work_item(
        engine,
        submitted.job_id,
        tenant="tenant-a",
        token="typed-worker",
        now=clock,
    )
    assert claim is not None
    lease_expires_at = float(engine.nodes[submitted.work_item_id]["lease_expires_at"])
    monkeypatch.setattr(rwi.time, "time", lambda: lease_expires_at - 0.001)
    assert (
        get_repository_operation_payload_for_claim(
            engine,
            submitted.job_id,
            tenant="tenant-a",
            claim=claim,
        )
        == payload
    )
    monkeypatch.setattr(rwi.time, "time", lambda: lease_expires_at)
    with pytest.raises(RepositoryWorkItemError, match="expired"):
        get_repository_operation_payload_for_claim(
            engine,
            submitted.job_id,
            tenant="tenant-a",
            claim=claim,
        )
    for malformed in ("not-a-time", float("nan"), float("inf")):
        engine.nodes[submitted.work_item_id]["lease_expires_at"] = malformed
        with pytest.raises(RepositoryWorkItemError, match="(?:malformed|expired)"):
            get_repository_operation_payload_for_claim(
                engine,
                submitted.job_id,
                tenant="tenant-a",
                claim=claim,
            )


def test_payload_digest_conflict_tamper_and_explicit_copy_preservation() -> None:
    engine = RepositoryEngine()
    payload = _operation_payload()
    request = _request(key="typed-conflict").model_copy(
        update={"operation_payload": payload}
    )
    first = submit_repository_work_item(engine, request)
    duplicate = submit_repository_work_item(engine, request)
    assert duplicate.deduplicated is True
    changed_payload = payload.model_copy(update={"argv": ("cargo", "test")})
    changed = request.model_copy(update={"operation_payload": changed_payload})
    with pytest.raises(RepositoryWorkItemConflict, match="input_conflict"):
        submit_repository_work_item(engine, changed)
    assert (
        engine.nodes[first.work_item_id]["metadata"]["repository_work_item"][
            "operation_payload_digest"
        ]
        == payload.payload_digest
    )

    tampered = engine.nodes[first.work_item_id]["metadata"]["repository_work_item"]
    tampered["operation_payload"]["argv"] = ["cargo", "clean"]
    with pytest.raises(RepositoryWorkItemConflict, match="input_conflict"):
        get_repository_work_item(engine, first.job_id, tenant="tenant-a")

    retry_request = request.model_copy(
        update={
            "request_id": "typed-conflict-retry:request",
            "idempotency_key": "typed-conflict-retry",
        }
    )
    # An explicit new WorkItem copies the exact typed input; correlation is
    # still only the new request relationship.
    second = submit_repository_work_item(engine, retry_request)
    assert (
        get_repository_operation_payload(
            engine,
            second.job_id,
            tenant="tenant-a",
            owner_id="actor-a",
        )
        == payload
    )


def test_typed_build_repair_preserves_exact_payload_and_replays_idempotently() -> None:
    engine = RepositoryEngine()
    payload = _operation_payload()
    source_request = _request(key="typed-repair-source").model_copy(
        update={"operation_payload": payload}
    )
    source = submit_repository_work_item(engine, source_request)

    repair_values = source_request.model_dump(mode="python", exclude_none=False)
    repair_values.update(
        {
            "request_id": "repair:typed-repair:request",
            "idempotency_key": "repair:typed-repair",
            "correlation_id": source.job_id,
            "retry_class": "reconciliation",
            "input_digest": "f" * 64,
        }
    )
    repaired = submit_repository_work_item(engine, repair_values)
    view = get_repository_work_item(engine, repaired.job_id, tenant="tenant-a")
    assert view is not None
    assert view.operation == RepositoryOperation.BUILD.value
    assert view.correlation_id == source.job_id
    assert view.retry_class == "reconciliation"
    assert (
        get_repository_operation_payload(
            engine,
            repaired.job_id,
            tenant="tenant-a",
            owner_id="actor-a",
        )
        == payload
    )

    restarted = RepositoryEngine()
    restarted.nodes = deepcopy(engine.nodes)
    assert (
        get_repository_operation_payload(
            restarted,
            repaired.job_id,
            tenant="tenant-a",
            owner_id="actor-a",
        )
        == payload
    )
    replay = submit_repository_work_item(engine, repair_values)
    assert replay.deduplicated is True

    changed_payload = payload.model_copy(update={"argv": ("cargo", "test")})
    changed_values = dict(repair_values)
    changed_values["operation_payload"] = changed_payload.model_dump(
        mode="json", exclude_none=False
    )
    with pytest.raises(RepositoryWorkItemConflict, match="input_conflict"):
        submit_repository_work_item(engine, changed_values)


def test_payloadless_legacy_build_is_readable_but_exact_input_fails_closed() -> None:
    engine = RepositoryEngine()
    legacy = submit_repository_work_item(engine, _request(key="legacy-build"))
    view = get_repository_work_item(engine, legacy.job_id, tenant="tenant-a")
    assert view is not None
    assert view.operation_payload_kind is None
    assert view.operation_payload_digest is None
    with pytest.raises(
        RepositoryWorkItemError, match="typed_execution_payload_required"
    ):
        get_repository_operation_payload(
            engine,
            legacy.job_id,
            tenant="tenant-a",
            owner_id="actor-a",
        )


def test_unknown_variant_refuses_before_work_item_creation() -> None:
    engine = RepositoryEngine()
    payload = _operation_payload()
    raw = _request(key="unknown-payload").model_dump(mode="python")
    raw["operation_payload"] = {
        **payload.model_dump(mode="json"),
        "kind": "repository.future/v1",
    }
    with pytest.raises(RepositoryWorkItemError, match="invalid"):
        submit_repository_work_item(engine, raw)
    job_id = repository_job_id("tenant-a", "unknown-payload")
    assert repository_work_item_id(job_id) not in engine.nodes


def test_existing_request_instances_revalidate_typed_payload_at_authority() -> None:
    engine = RepositoryEngine()
    payload = _operation_payload()
    request = _request(key="invalid-copy").model_copy(
        update={
            "operation_payload": {
                **payload.model_dump(mode="python"),
                "argv": ("sh", "-c", "echo secret"),
            }
        }
    )
    with pytest.raises(RepositoryWorkItemError, match="invalid"):
        submit_repository_work_item(engine, request)
    job_id = repository_job_id("tenant-a", "invalid-copy")
    assert repository_work_item_id(job_id) not in engine.nodes


def test_automatic_retry_keeps_exact_operation_payload_and_digest() -> None:
    engine = RepositoryEngine()
    payload = _operation_payload()
    request = _request(key="typed-auto-retry").model_copy(
        update={"operation_payload": payload}
    )
    submitted = submit_repository_work_item(engine, request, max_attempts=2)
    first = claim_repository_work_item(
        engine,
        submitted.job_id,
        tenant="tenant-a",
        token="typed-worker-a",
        now=100.0,
    )
    assert first is not None
    assert (
        commit_repository_work_item(
            engine,
            submitted.job_id,
            first,
            tenant="tenant-a",
            outcome="failed",
            failure_class="worker_environment_failure",
            retryable=True,
            now=101.0,
        )
        == "retry_scheduled"
    )
    assert (
        get_repository_operation_payload(
            engine,
            submitted.job_id,
            tenant="tenant-a",
            owner_id="actor-a",
        )
        == payload
    )
    second = claim_repository_work_item(
        engine,
        submitted.job_id,
        tenant="tenant-a",
        token="typed-worker-b",
        now=102.0,
    )
    assert second is not None
    assert second["work_item_id"] == submitted.work_item_id
    assert (
        get_repository_operation_payload(
            engine,
            submitted.job_id,
            tenant="tenant-a",
            owner_id="actor-a",
        )
        == payload
    )
