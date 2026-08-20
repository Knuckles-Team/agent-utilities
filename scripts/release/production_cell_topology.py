"""Measured-input contract for the Kubernetes GraphOS production cell.

The checked-in manifests under ``deploy/k8s/production-cell`` are templates.
This module is the small, dependency-light authority used by the release
renderer to bind those templates to one measured resource pool, one engine
identity, and one set of retained external authorities.  It deliberately does
not contact Kubernetes or an external provider.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any
from urllib.parse import urlparse


class TopologyInputError(ValueError):
    """A production-cell input is absent, inconsistent, or unsafe."""


_DIGEST_RE = re.compile(r"^.+@sha256:[0-9a-f]{64}$")
_DNS_RE = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")
_HOST_RE = re.compile(r"^[a-z0-9]([-a-z0-9.]*[a-z0-9])?$")
_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_RFC3339_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:Z|[+-]\d{2}:\d{2})$")
_METRIC_RE = re.compile(r"^[a-z][a-z0-9_]{2,127}$")
_TARGET_RE = re.compile(r"^[0-9]+(?:m)?$")
_ENGINE_CONTRACT_REF = "services/epistemic-graph/k8s/production/engine-identity-contract.v1.json"
_ENGINE_CONTRACT_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_EXPECTED_METRICS = {
    "gateway": "agent_utilities_gateway_in_flight_requests",
    "dispatch": "agent_utilities_dispatch_queue_depth",
    "ingest": "agent_utilities_kg_ingest_queue_depth",
}
_EXPECTED_ENGINE = {
    "statefulset_name": "epistemic-graph-raft",
    "client_service_name": "epistemic-graph-coordinator",
    "peer_service_name": "epistemic-graph-raft",
    "namespace": "graphos-cell",
    "client_port": 9101,
    "raft_port": 9100,
    "metrics_port": 9102,
}
_EXPECTED_WORKLOADS = {
    "gateway": ("graphos-front", "graphos-control", "graph-os"),
    "dispatch": ("graphos-dispatch-worker", "graphos-cell", "agent-dispatch-worker"),
    "ingest": ("graphos-ingest-worker", "graphos-cell", "kg-ingest-worker"),
    "mining": ("graphos-analytics-worker", "graphos-cell", "graph-os-analytics-worker"),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TopologyInputError(message)


def _closed(value: Any, field: str, allowed: set[str]) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{field} must be an object")
    unknown = sorted(set(value) - allowed)
    _require(not unknown, f"{field} has unknown fields: {', '.join(unknown)}")
    return value


def _string(value: Any, field: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()), f"{field} must be a non-empty string")
    _require("\x00" not in value and "\n" not in value and "\r" not in value, f"{field} contains control characters")
    return value


def _positive(value: Any, field: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool) and value > 0, f"{field} must be a positive integer")
    return value


def _dns(value: Any, field: str) -> str:
    value = _string(value, field)
    _require(_DNS_RE.fullmatch(value) is not None, f"{field} must be a DNS label")
    return value


def _safe_ref(value: Any, field: str) -> str:
    value = _string(value, field)
    lowered = value.lower()
    _require(len(value) <= 256, f"{field} is too long")
    _require(
        not any(secret in lowered for secret in ("password", "token", "bearer", "secret", "private-key")),
        f"{field} must not contain secret material",
    )
    _require(
        not re.search(r"(?:^[a-z]:\\|^/|^file://|/home/|/users/|/mnt/[a-z]/)", lowered),
        f"{field} must be an opaque non-filesystem reference",
    )
    return value


def _digest_image(value: Any, field: str) -> str:
    value = _string(value, field)
    _require(_DIGEST_RE.fullmatch(value) is not None, f"{field} must be an immutable @sha256 image")
    return value


def _release_image(manifest: dict[str, Any] | None, component: str) -> str | None:
    if manifest is None:
        return None
    item = ((manifest.get("components") or {}).get(component) or {})
    artifact = _string(item.get("artifact"), f"release.components.{component}.artifact")
    digest = _string(item.get("digest"), f"release.components.{component}.digest")
    name, separator, suffix = artifact.rpartition("@")
    _require(separator and name and suffix == digest, f"release component {component} has mismatched artifact/digest")
    return artifact


def _canonical_engine_contract_digest(contract: dict[str, Any]) -> str:
    payload = {key: value for key, value in contract.items() if key != "digest"}
    return "sha256:" + hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _validate_engine_contract(value: Any) -> dict[str, Any]:
    contract = _closed(
        value,
        "engine_identity_contract",
        {
            "$schema", "apiVersion", "kind", "authority", "contract_version", "namespace",
            "statefulset_name", "peer_service_name", "client_service_name", "peer_port",
            "client_port", "metrics_port", "tls_server_name", "tls_sans", "discovery_identity",
            "transport", "compatibility_aliases", "digest",
        },
    )
    _require(contract.get("apiVersion") == "agent-utilities.io/engine-identity/v1", "engine identity apiVersion is not supported")
    _require(contract.get("$schema") == "https://agent-utilities.invalid/schemas/engine-identity-contract-v1.json", "engine identity schema is not canonical")
    _require(contract.get("kind") == "EngineIdentityContract", "engine identity kind is not supported")
    _require(contract.get("authority") == _ENGINE_CONTRACT_REF, "engine identity authority is not canonical")
    _require(contract.get("contract_version") == "engine-identity.v1", "engine identity contract version is not supported")
    digest = _string(contract.get("digest"), "engine_identity_contract.digest")
    _require(_ENGINE_CONTRACT_DIGEST_RE.fullmatch(digest) is not None, "engine identity digest must be sha256")
    _require(digest == _canonical_engine_contract_digest(contract), "engine identity digest does not match canonical content")
    namespace = _dns(contract.get("namespace"), "engine_identity_contract.namespace")
    statefulset = _dns(contract.get("statefulset_name"), "engine_identity_contract.statefulset_name")
    peer = _dns(contract.get("peer_service_name"), "engine_identity_contract.peer_service_name")
    client = _dns(contract.get("client_service_name"), "engine_identity_contract.client_service_name")
    peer_port = _positive(contract.get("peer_port"), "engine_identity_contract.peer_port")
    client_port = _positive(contract.get("client_port"), "engine_identity_contract.client_port")
    metrics_port = _positive(contract.get("metrics_port"), "engine_identity_contract.metrics_port")
    _require(len({peer_port, client_port, metrics_port}) == 3, "engine identity ports must be distinct")
    _require(all(port <= 65535 for port in (peer_port, client_port, metrics_port)), "engine identity ports must be <= 65535")
    tls_name = _string(contract.get("tls_server_name"), "engine_identity_contract.tls_server_name")
    expected_tls = f"{peer}.{namespace}.svc.cluster.local"
    _require(tls_name == expected_tls, "engine identity TLS server name must identify the canonical peer Service")
    sans = contract.get("tls_sans")
    _require(isinstance(sans, list) and sans and all(isinstance(item, str) for item in sans), "engine identity tls_sans must be a non-empty string list")
    _require(tls_name in sans and len(sans) == len(set(sans)), "engine identity tls_sans must contain the canonical name without duplicates")
    _require(contract.get("discovery_identity") == tls_name, "engine identity discovery identity must equal TLS server name")
    _require(contract.get("transport") == "length-prefixed-msgpack-tls", "engine identity transport is not supported")
    aliases = contract.get("compatibility_aliases")
    _require(isinstance(aliases, list) and aliases, "engine identity compatibility aliases must be explicit")
    for index, alias in enumerate(aliases):
        alias = _closed(alias, f"engine_identity_contract.compatibility_aliases[{index}]", {"peer_service_name", "client_service_name", "namespace", "status", "live_authority"})
        _require(alias.get("status") == "migration-only" and alias.get("live_authority") is False, "engine identity aliases cannot be live authority")
        _require((alias.get("peer_service_name"), alias.get("client_service_name"), alias.get("namespace")) != (peer, client, namespace), "engine identity alias must differ from canonical identity")
    return {
        "ref": _ENGINE_CONTRACT_REF,
        "digest": digest,
        "namespace": namespace,
        "statefulset_name": statefulset,
        "peer_service_name": peer,
        "client_service_name": client,
        "peer_port": peer_port,
        "client_port": client_port,
        "metrics_port": metrics_port,
        "tls_server_name": tls_name,
        "discovery_identity": tls_name,
    }


def _validate_engine_binding(value: Any, contract: dict[str, Any]) -> None:
    binding = _closed(value, "engine_identity_contract_binding", {"ref", "digest"})
    _require(binding.get("ref") == contract["ref"], "engine identity binding ref does not match the canonical contract")
    _require(binding.get("digest") == contract["digest"], "engine identity binding digest does not match the canonical contract")


def _validate_metric(raw: Any, workload: str) -> dict[str, Any]:
    metric = _closed(raw, f"workloads.{workload}.metric", {"mode", "name", "target", "adapter_ref", "reason"})
    mode = _string(metric.get("mode"), f"workloads.{workload}.metric.mode")
    if mode == "fixed":
        _require(workload == "mining", "only mining may use fixed scaling until its signal is bounded")
        return {"mode": mode, "reason": _string(metric.get("reason"), f"workloads.{workload}.metric.reason")}
    _require(mode == "hpa", f"workloads.{workload}.metric.mode must be hpa or fixed")
    name = _string(metric.get("name"), f"workloads.{workload}.metric.name")
    _require(_METRIC_RE.fullmatch(name) is not None, f"workloads.{workload}.metric.name is unsafe")
    _require(name == _EXPECTED_METRICS[workload], f"workloads.{workload} must use {_EXPECTED_METRICS[workload]!r}")
    target = _string(metric.get("target"), f"workloads.{workload}.metric.target")
    _require(_TARGET_RE.fullmatch(target) is not None, f"workloads.{workload}.metric.target is unbounded")
    return {"mode": mode, "name": name, "target": target, "adapter_ref": _safe_ref(metric.get("adapter_ref"), f"workloads.{workload}.metric.adapter_ref")}


def validate(
    document: dict[str, Any],
    *,
    release_manifest: dict[str, Any] | None = None,
    engine_identity_contract: dict[str, Any],
) -> dict[str, Any]:
    """Validate and normalize a measured production-cell input."""

    _closed(
        document,
        "input",
        {"$schema", "apiVersion", "kind", "authority", "environment", "engine_identity_contract", "resource_pool", "engine", "oidc", "authorities", "release", "workloads"},
    )
    for field, expected in {
        "apiVersion": "agent-utilities.io/v1",
        "kind": "GraphOSProductionCellTopologyInput",
        "authority": "agent-utilities/deploy/k8s/production-cell",
        "environment": "production",
    }.items():
        _require(document.get(field) == expected, f"{field} must be {expected!r}")

    contract = _validate_engine_contract(engine_identity_contract)
    _validate_engine_binding(document.get("engine_identity_contract"), contract)

    pool = _closed(document.get("resource_pool"), "resource_pool", {"id", "measured_at", "evidence_ref", "nodes"})
    pool_id = _string(pool.get("id"), "resource_pool.id")
    measured_at = _string(pool.get("measured_at"), "resource_pool.measured_at")
    _require(_RFC3339_RE.fullmatch(measured_at) is not None, "resource_pool.measured_at must be RFC3339")
    pool_evidence = _safe_ref(pool.get("evidence_ref"), "resource_pool.evidence_ref")
    raw_nodes = pool.get("nodes")
    _require(isinstance(raw_nodes, list) and raw_nodes, "resource_pool.nodes must be non-empty")
    nodes: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_hosts: set[str] = set()
    ordinals: set[int] = set()
    zones: set[str] = set()
    for index, raw in enumerate(raw_nodes):
        raw = _closed(raw, f"resource_pool.nodes[{index}]", {"ordinal", "id", "hostname", "zone", "allocatable"})
        ordinal = raw.get("ordinal")
        _require(isinstance(ordinal, int) and not isinstance(ordinal, bool) and ordinal >= 0, f"resource_pool.nodes[{index}].ordinal must be non-negative")
        node_id = _string(raw.get("id"), f"resource_pool.nodes[{index}].id")
        hostname = _string(raw.get("hostname"), f"resource_pool.nodes[{index}].hostname")
        zone = _string(raw.get("zone"), f"resource_pool.nodes[{index}].zone")
        _require(_HOST_RE.fullmatch(hostname) is not None, f"resource_pool.nodes[{index}].hostname is unsafe")
        _require(node_id not in seen_ids and hostname not in seen_hosts and ordinal not in ordinals, f"duplicate node identity at resource_pool.nodes[{index}]")
        alloc = _closed(raw.get("allocatable"), f"resource_pool.nodes[{index}].allocatable", {"cpu_millicores", "memory_bytes"})
        node = {
            "ordinal": ordinal,
            "id": node_id,
            "hostname": hostname,
            "zone": zone,
            "cpu": _positive(alloc.get("cpu_millicores"), f"resource_pool.nodes[{index}].allocatable.cpu_millicores"),
            "memory": _positive(alloc.get("memory_bytes"), f"resource_pool.nodes[{index}].allocatable.memory_bytes"),
        }
        nodes.append(node)
        seen_ids.add(node_id)
        seen_hosts.add(hostname)
        ordinals.add(ordinal)
        zones.add(zone)
    nodes.sort(key=lambda value: value["ordinal"])
    _require([node["ordinal"] for node in nodes] == list(range(len(nodes))), "resource_pool.node ordinals must be contiguous")

    engine = _closed(
        document.get("engine"),
        "engine",
        {"statefulset_name", "client_service_name", "peer_service_name", "namespace", "client_port", "raft_port", "metrics_port", "tls_server_name", "tls_verified", "discovery_ref", "discovery_verified", "replicas", "raft_groups", "current_image", "rollback_image", "resources", "pdb_max_unavailable", "termination_grace_seconds", "pre_stop_seconds"},
    )
    engine_statefulset = _dns(engine.get("statefulset_name"), "engine.statefulset_name")
    engine_client = _dns(engine.get("client_service_name"), "engine.client_service_name")
    engine_peer = _dns(engine.get("peer_service_name"), "engine.peer_service_name")
    engine_namespace = _dns(engine.get("namespace"), "engine.namespace")
    engine_port = _positive(engine.get("client_port"), "engine.client_port")
    raft_port = _positive(engine.get("raft_port"), "engine.raft_port")
    metrics_port = _positive(engine.get("metrics_port"), "engine.metrics_port")
    actual_engine = {
        "statefulset_name": engine_statefulset,
        "client_service_name": engine_client,
        "peer_service_name": engine_peer,
        "namespace": engine_namespace,
        "client_port": engine_port,
        "raft_port": raft_port,
        "metrics_port": metrics_port,
    }
    for field, expected in _EXPECTED_ENGINE.items():
        _require(
            actual_engine[field] == expected,
            f"engine.{field} must be the canonical production-cell identity {expected!r}",
        )
    for field in ("statefulset_name", "client_service_name", "peer_service_name", "namespace", "client_port", "raft_port", "metrics_port"):
        _require(actual_engine[field] == contract[field], f"engine.{field} disagrees with the canonical engine identity contract")
    _require(engine_port <= 65535 and raft_port <= 65535 and metrics_port <= 65535, "engine ports must be <= 65535")
    tls_name = _string(engine.get("tls_server_name"), "engine.tls_server_name")
    _require(tls_name == contract["tls_server_name"], "engine.tls_server_name must match the canonical engine identity contract")
    _require(engine.get("tls_verified") is True, "engine.tls_verified must be true after preflight")
    engine_discovery = _safe_ref(engine.get("discovery_ref"), "engine.discovery_ref")
    _require(engine.get("discovery_verified") is True, "engine.discovery_verified must be true after preflight")
    engine_replicas = _positive(engine.get("replicas"), "engine.replicas")
    _require(engine_replicas == 3, "the production cell requires exactly three engine members")
    _require(engine_replicas <= len(zones), "engine replicas exceed measured failure domains")
    raft_groups = _positive(engine.get("raft_groups"), "engine.raft_groups")
    engine_current = _digest_image(engine.get("current_image"), "engine.current_image")
    engine_rollback = _digest_image(engine.get("rollback_image"), "engine.rollback_image")
    expected_engine = _release_image(release_manifest, "epistemic-graph")
    if expected_engine is not None:
        _require(engine_current == expected_engine, "engine.current_image does not match the exact release manifest")
    engine_resources = _closed(engine.get("resources"), "engine.resources", {"cpu_request_millicores", "cpu_limit_millicores", "memory_request_bytes", "memory_limit_bytes"})
    engine_cpu_request = _positive(engine_resources.get("cpu_request_millicores"), "engine.resources.cpu_request_millicores")
    engine_cpu_limit = _positive(engine_resources.get("cpu_limit_millicores"), "engine.resources.cpu_limit_millicores")
    engine_memory_request = _positive(engine_resources.get("memory_request_bytes"), "engine.resources.memory_request_bytes")
    engine_memory_limit = _positive(engine_resources.get("memory_limit_bytes"), "engine.resources.memory_limit_bytes")
    _require(engine_cpu_request <= engine_cpu_limit and engine_memory_request <= engine_memory_limit, "engine resource request must not exceed limit")
    _require(all(node["cpu"] >= engine_cpu_limit and node["memory"] >= engine_memory_limit for node in nodes), "measured node capacity cannot host engine limits")
    engine_pdb = engine.get("pdb_max_unavailable")
    _require(isinstance(engine_pdb, int) and 0 <= engine_pdb < engine_replicas, "engine.pdb_max_unavailable must preserve quorum")
    engine_termination = _positive(engine.get("termination_grace_seconds"), "engine.termination_grace_seconds")
    engine_pre_stop = engine.get("pre_stop_seconds")
    _require(isinstance(engine_pre_stop, int) and 0 <= engine_pre_stop < engine_termination, "engine.pre_stop_seconds must be below termination grace")

    oidc = _closed(document.get("oidc"), "oidc", {"issuer", "jwks_url", "audience", "client_id", "verified"})
    issuer = _string(oidc.get("issuer"), "oidc.issuer")
    jwks_url = _string(oidc.get("jwks_url"), "oidc.jwks_url")
    issuer_url = urlparse(issuer)
    jwks = urlparse(jwks_url)
    _require(issuer_url.scheme == "https" and issuer_url.netloc, "oidc.issuer must be HTTPS")
    _require(jwks.scheme == "https" and jwks.netloc == issuer_url.netloc, "oidc.jwks_url must share the HTTPS authority")
    _require(not issuer_url.username and not issuer_url.password and not jwks.username and not jwks.password, "oidc URLs must not contain userinfo")
    oidc_audience = _string(oidc.get("audience"), "oidc.audience")
    oidc_client_id = _string(oidc.get("client_id"), "oidc.client_id")
    _require(oidc.get("verified") is True, "oidc.verified must be true after preflight")

    authorities = _closed(document.get("authorities"), "authorities", {"control_configmap_name", "cell_configmap_name", "configmap_retention", "control_configmap_ref", "cell_configmap_ref", "secret_name", "secret_retention", "secret_ref", "session_store", "action_audit"})
    control_config = _dns(authorities.get("control_configmap_name"), "authorities.control_configmap_name")
    cell_config = _dns(authorities.get("cell_configmap_name"), "authorities.cell_configmap_name")
    _require(authorities.get("configmap_retention") == "Retain", "ConfigMap authority retention must be Retain")
    control_config_ref = _safe_ref(authorities.get("control_configmap_ref"), "authorities.control_configmap_ref")
    cell_config_ref = _safe_ref(authorities.get("cell_configmap_ref"), "authorities.cell_configmap_ref")
    secret_name = _dns(authorities.get("secret_name"), "authorities.secret_name")
    _require(authorities.get("secret_retention") == "Retain", "Secret authority retention must be Retain")
    secret_ref = _safe_ref(authorities.get("secret_ref"), "authorities.secret_ref")
    session = _closed(authorities.get("session_store"), "authorities.session_store", {"kind", "service_name", "namespace", "authority_ref", "retention"})
    _require(session.get("kind") == "external", "authorities.session_store.kind must be external")
    session_service = _dns(session.get("service_name"), "authorities.session_store.service_name")
    session_namespace = _dns(session.get("namespace"), "authorities.session_store.namespace")
    session_ref = _safe_ref(session.get("authority_ref"), "authorities.session_store.authority_ref")
    _require(session.get("retention") == "Retain", "session store authority retention must be Retain")
    action_audit = _closed(authorities.get("action_audit"), "authorities.action_audit", {"authority_ref", "retention", "graph", "kind"})
    action_audit_ref = _safe_ref(action_audit.get("authority_ref"), "authorities.action_audit.authority_ref")
    _require(action_audit.get("retention") == "Retain", "action audit authority retention must be Retain")
    action_audit_graph = _string(action_audit.get("graph"), "authorities.action_audit.graph")
    action_audit_kind = _string(action_audit.get("kind"), "authorities.action_audit.kind")

    release = _closed(document.get("release"), "release", {"current_manifest_ref", "rollback_manifest_ref", "rollout_evidence_ref", "rollback_evidence_ref"})
    current_manifest_ref = _safe_ref(release.get("current_manifest_ref"), "release.current_manifest_ref")
    rollback_manifest_ref = _safe_ref(release.get("rollback_manifest_ref"), "release.rollback_manifest_ref")
    rollout_evidence = _safe_ref(release.get("rollout_evidence_ref"), "release.rollout_evidence_ref")
    rollback_evidence = _safe_ref(release.get("rollback_evidence_ref"), "release.rollback_evidence_ref")

    workloads = _closed(document.get("workloads"), "workloads", {"gateway", "dispatch", "ingest", "mining"})
    expected_components = {"gateway": "agent-utilities", "dispatch": "agent-utilities", "ingest": "agent-utilities", "mining": "agent-utilities"}
    normalized: dict[str, dict[str, Any]] = {}
    names: set[str] = set()
    total_cpu_request = engine_replicas * engine_cpu_request
    total_cpu_limit = engine_replicas * engine_cpu_limit
    total_memory_request = engine_replicas * engine_memory_request
    total_memory_limit = engine_replicas * engine_memory_limit
    for workload_key in ("gateway", "dispatch", "ingest", "mining"):
        raw = _closed(workloads.get(workload_key), f"workloads.{workload_key}", {"name", "namespace", "entrypoint", "current_image", "rollback_image", "replicas", "resources", "pdb_max_unavailable", "termination_grace_seconds", "pre_stop_seconds", "metric", "component"})
        name = _dns(raw.get("name"), f"workloads.{workload_key}.name")
        namespace = _dns(raw.get("namespace"), f"workloads.{workload_key}.namespace")
        _require(name not in names, f"duplicate workload name {name}")
        names.add(name)
        entrypoint = _string(raw.get("entrypoint"), f"workloads.{workload_key}.entrypoint")
        _require(_TOKEN_RE.fullmatch(entrypoint) is not None, f"workloads.{workload_key}.entrypoint is unsafe")
        component = _string(raw.get("component") or expected_components[workload_key], f"workloads.{workload_key}.component")
        _require(component == expected_components[workload_key], f"workloads.{workload_key}.component must be {expected_components[workload_key]!r}")
        current_image = _digest_image(raw.get("current_image"), f"workloads.{workload_key}.current_image")
        rollback_image = _digest_image(raw.get("rollback_image"), f"workloads.{workload_key}.rollback_image")
        expected_image = _release_image(release_manifest, component)
        if expected_image is not None:
            _require(current_image == expected_image, f"workloads.{workload_key}.current_image does not match the exact release manifest")
        replicas = _closed(raw.get("replicas"), f"workloads.{workload_key}.replicas", {"min", "desired", "max", "scale_up_step", "scale_down_step", "up_stabilization_seconds", "down_stabilization_seconds"})
        minimum = _positive(replicas.get("min"), f"workloads.{workload_key}.replicas.min")
        desired = _positive(replicas.get("desired"), f"workloads.{workload_key}.replicas.desired")
        maximum = _positive(replicas.get("max"), f"workloads.{workload_key}.replicas.max")
        _require(minimum <= desired <= maximum and maximum <= len(zones), f"workloads.{workload_key}.replicas must fit min <= desired <= max <= measured zones")
        scale_up = _positive(replicas.get("scale_up_step"), f"workloads.{workload_key}.replicas.scale_up_step")
        scale_down = _positive(replicas.get("scale_down_step"), f"workloads.{workload_key}.replicas.scale_down_step")
        up_window = replicas.get("up_stabilization_seconds")
        down_window = replicas.get("down_stabilization_seconds")
        _require(isinstance(up_window, int) and up_window >= 0 and isinstance(down_window, int) and down_window >= 0, f"workloads.{workload_key}.stabilization windows must be non-negative")
        resources = _closed(raw.get("resources"), f"workloads.{workload_key}.resources", {"cpu_request_millicores", "cpu_limit_millicores", "memory_request_bytes", "memory_limit_bytes"})
        cpu_request = _positive(resources.get("cpu_request_millicores"), f"workloads.{workload_key}.resources.cpu_request_millicores")
        cpu_limit = _positive(resources.get("cpu_limit_millicores"), f"workloads.{workload_key}.resources.cpu_limit_millicores")
        memory_request = _positive(resources.get("memory_request_bytes"), f"workloads.{workload_key}.resources.memory_request_bytes")
        memory_limit = _positive(resources.get("memory_limit_bytes"), f"workloads.{workload_key}.resources.memory_limit_bytes")
        _require(cpu_request <= cpu_limit and memory_request <= memory_limit, f"workloads.{workload_key}.resource request must not exceed limit")
        _require(all(node["cpu"] >= cpu_limit and node["memory"] >= memory_limit for node in nodes), f"measured node capacity cannot host {workload_key} limits")
        total_cpu_request += maximum * cpu_request
        total_cpu_limit += maximum * cpu_limit
        total_memory_request += maximum * memory_request
        total_memory_limit += maximum * memory_limit
        pdb = raw.get("pdb_max_unavailable")
        _require(isinstance(pdb, int) and 0 <= pdb < minimum, f"workloads.{workload_key}.pdb_max_unavailable must preserve one replica")
        termination = _positive(raw.get("termination_grace_seconds"), f"workloads.{workload_key}.termination_grace_seconds")
        pre_stop = raw.get("pre_stop_seconds")
        _require(isinstance(pre_stop, int) and 0 <= pre_stop < termination, f"workloads.{workload_key}.pre_stop_seconds must be below termination grace")
        metric = _validate_metric(raw.get("metric"), workload_key)
        expected_name, expected_namespace, expected_entrypoint = _EXPECTED_WORKLOADS[workload_key]
        _require(name == expected_name, f"workloads.{workload_key}.name must be the canonical production-cell identity {expected_name!r}")
        _require(namespace == expected_namespace, f"workloads.{workload_key}.namespace must be the canonical production-cell namespace {expected_namespace!r}")
        _require(entrypoint == expected_entrypoint, f"workloads.{workload_key}.entrypoint must be {expected_entrypoint!r}")
        normalized[workload_key] = {
            "name": name, "namespace": namespace, "entrypoint": entrypoint,
            "component": component, "current_image": current_image, "rollback_image": rollback_image,
            "min": minimum, "desired": desired, "max": maximum,
            "scale_up": scale_up, "scale_down": scale_down,
            "up_window": up_window, "down_window": down_window,
            "cpu_request": str(cpu_request), "cpu_limit": str(cpu_limit),
            "memory_request": str(memory_request), "memory_limit": str(memory_limit),
            "pdb_min_available": minimum - pdb, "termination": termination,
            "pre_stop": pre_stop, "metric": metric,
        }
    _require(total_cpu_request <= sum(node["cpu"] for node in nodes), "maximum workload CPU requests exceed measured pool")
    _require(total_cpu_limit <= sum(node["cpu"] for node in nodes), "maximum workload CPU limits exceed measured pool")
    _require(total_memory_request <= sum(node["memory"] for node in nodes), "maximum workload memory requests exceed measured pool")
    _require(total_memory_limit <= sum(node["memory"] for node in nodes), "maximum workload memory limits exceed measured pool")
    gateway_images = (normalized["gateway"]["current_image"], normalized["gateway"]["rollback_image"])
    for workload_key, item in normalized.items():
        _require(
            (item["current_image"], item["rollback_image"]) == gateway_images,
            f"workloads.{workload_key} images must match gateway because graph-os-image is one shared kustomize pin",
        )

    topology = {
        "pool_id": pool_id, "measured_at": measured_at, "pool_evidence": pool_evidence,
        "nodes": nodes, "zones": sorted(zones),
        "engine_statefulset": engine_statefulset, "engine_client": engine_client, "engine_peer": engine_peer,
        "engine_namespace": engine_namespace, "engine_port": engine_port, "raft_port": raft_port, "engine_metrics_port": metrics_port,
        "engine_endpoint": f"tls://{engine_client}.{engine_namespace}.svc.cluster.local:{engine_port}",
        "engine_tls_name": tls_name, "engine_discovery": engine_discovery,
        "engine_identity_ref": contract["ref"], "engine_identity_digest": contract["digest"],
        "engine_replicas": engine_replicas, "engine_raft_groups": raft_groups,
        "engine_current_image": engine_current, "engine_rollback_image": engine_rollback,
        "engine_cpu_request": str(engine_cpu_request), "engine_cpu_limit": str(engine_cpu_limit),
        "engine_memory_request": str(engine_memory_request), "engine_memory_limit": str(engine_memory_limit),
        "engine_pdb_max_unavailable": engine_pdb, "engine_termination": engine_termination, "engine_pre_stop": engine_pre_stop,
        "oidc_issuer": issuer, "oidc_jwks": jwks_url, "oidc_audience": oidc_audience, "oidc_client_id": oidc_client_id,
        "control_config": control_config, "cell_config": cell_config,
        "control_config_ref": control_config_ref, "cell_config_ref": cell_config_ref,
        "secret_name": secret_name, "secret_ref": secret_ref,
        "session_service": session_service, "session_namespace": session_namespace, "session_ref": session_ref,
        "action_audit_ref": action_audit_ref, "action_audit_graph": action_audit_graph, "action_audit_kind": action_audit_kind,
        "current_manifest_ref": current_manifest_ref, "rollback_manifest_ref": rollback_manifest_ref,
        "rollout_evidence": rollout_evidence, "rollback_evidence": rollback_evidence,
        "workloads": normalized,
    }
    topology["input_digest"] = hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return topology


def canonical_contract(topology: dict[str, Any], *, rollback: bool) -> dict[str, Any]:
    """Return a bounded, non-secret contract ConfigMap payload."""

    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": "graphos-topology-contract",
            "namespace": "graphos-control",
            "annotations": {
                "graphos.io/authority-ref": "agent-utilities/deploy/k8s/production-cell",
                "graphos.io/authority-retention": "Retain",
            },
        },
        "immutable": True,
        "data": {
            "TOPOLOGY_INPUT_DIGEST": topology["input_digest"],
            "TOPOLOGY_RENDER_MODE": "rollback" if rollback else "forward",
            "RESOURCE_POOL_ID": topology["pool_id"],
            "RESOURCE_POOL_MEASURED_AT": topology["measured_at"],
            "RESOURCE_POOL_EVIDENCE_REF": topology["pool_evidence"],
            "ENGINE_SERVICE": f"{topology['engine_client']}.{topology['engine_namespace']}.svc.cluster.local",
            "ENGINE_PEER_SERVICE": f"{topology['engine_peer']}.{topology['engine_namespace']}.svc.cluster.local",
            "ENGINE_ENDPOINT": topology["engine_endpoint"],
            "ENGINE_IDENTITY_CONTRACT_REF": topology["engine_identity_ref"],
            "ENGINE_IDENTITY_CONTRACT_DIGEST": topology["engine_identity_digest"],
            "ENGINE_TLS_SERVER_NAME": topology["engine_tls_name"],
            "ENGINE_TLS_VERIFIED": "true",
            "ENGINE_DISCOVERY_REF": topology["engine_discovery"],
            "ENGINE_DISCOVERY_VERIFIED": "true",
            "ENGINE_REPLICAS": str(topology["engine_replicas"]),
            "ENGINE_RAFT_GROUPS": str(topology["engine_raft_groups"]),
            "OIDC_ISSUER": topology["oidc_issuer"],
            "OIDC_JWKS_URI": topology["oidc_jwks"],
            "OIDC_AUDIENCE": topology["oidc_audience"],
            "OIDC_CLIENT_ID": topology["oidc_client_id"],
            "OIDC_DISCOVERY_VERIFIED": "true",
            "CONFIG_AUTHORITY_REF": topology["control_config_ref"],
            "CELL_CONFIG_AUTHORITY_REF": topology["cell_config_ref"],
            "CONTROL_CONFIG_AUTHORITY_NAME": topology["control_config"],
            "CELL_CONFIG_AUTHORITY_NAME": topology["cell_config"],
            "SECRET_AUTHORITY_REF": topology["secret_ref"],
            "SECRET_AUTHORITY_NAME": topology["secret_name"],
            "SESSION_STORE_SERVICE": f"{topology['session_service']}.{topology['session_namespace']}.svc.cluster.local",
            "SESSION_STORE_AUTHORITY_REF": topology["session_ref"],
            "SESSION_STORE_RETENTION": "Retain",
            "ACTION_AUDIT_AUTHORITY_REF": topology["action_audit_ref"],
            "ACTION_AUDIT_GRAPH": topology["action_audit_graph"],
            "ACTION_AUDIT_KIND": topology["action_audit_kind"],
            "CURRENT_MANIFEST_REF": topology["current_manifest_ref"],
            "ROLLBACK_MANIFEST_REF": topology["rollback_manifest_ref"],
            "ROLLOUT_EVIDENCE_REF": topology["rollout_evidence"],
            "ROLLBACK_EVIDENCE_REF": topology["rollback_evidence"],
            **{
                f"WORKLOAD_{key.upper()}_CURRENT_IMAGE": item["current_image"]
                for key, item in topology["workloads"].items()
            },
            **{
                f"WORKLOAD_{key.upper()}_ROLLBACK_IMAGE": item["rollback_image"]
                for key, item in topology["workloads"].items()
            },
            **{
                f"WORKLOAD_{key.upper()}_METRIC_MODE": item["metric"]["mode"]
                for key, item in topology["workloads"].items()
            },
            **{
                f"WORKLOAD_{key.upper()}_METRIC_NAME": item["metric"].get("name", "")
                for key, item in topology["workloads"].items()
            },
            **{
                f"WORKLOAD_{key.upper()}_ADAPTER_REF": item["metric"].get("adapter_ref", "")
                for key, item in topology["workloads"].items()
            },
        },
    }


_WORKLOAD_DOCS = {
    "gateway": ("Deployment", "graphos-front", "graphos-control"),
    "dispatch": ("Deployment", "graphos-dispatch-worker", "graphos-cell"),
    "ingest": ("Deployment", "graphos-ingest-worker", "graphos-cell"),
    "mining": ("Deployment", "graphos-analytics-worker", "graphos-cell"),
}


def _identity(document: dict[str, Any]) -> tuple[str, str, str]:
    metadata = document.get("metadata") or {}
    return (str(document.get("kind") or ""), str(metadata.get("name") or ""), str(metadata.get("namespace") or ""))


def _resource_value(value: str) -> str:
    """Keep resource quantities explicit and portable across YAML emitters."""

    return value


def _set_container_contract(container: dict[str, Any], item: dict[str, Any], *, rollback: bool) -> None:
    resources = container.setdefault("resources", {})
    resources["requests"] = {"cpu": _resource_value(item["cpu_request"]), "memory": _resource_value(item["memory_request"])}
    resources["limits"] = {"cpu": _resource_value(item["cpu_limit"]), "memory": _resource_value(item["memory_limit"])}
    container["lifecycle"] = {"preStop": {"exec": {"command": ["/bin/sh", "-c", f"sleep {item['pre_stop']}"]}}}
    if rollback:
        container["image"] = item["rollback_image"]


def _set_named_port(document: dict[str, Any], name: str, port: int) -> None:
    ports = document.setdefault("spec", {}).setdefault("ports", [])
    for entry in ports:
        if isinstance(entry, dict) and entry.get("name") == name:
            entry["port"] = port
            return
    raise TopologyInputError(f"template is missing named port {name!r} for {_identity(document)}")


def apply_to_documents(documents: list[dict[str, Any]], topology: dict[str, Any], *, rollback: bool) -> list[dict[str, Any]]:
    """Bind template documents to validated topology values.

    The source template remains reviewable and deliberately conservative.  A
    rendered directory receives measured replica/resource bounds, governed
    HPA selectors, retained-authority references, and explicit rollout/drain
    annotations.  No Kubernetes API or external authority is contacted.
    """

    by_identity = {_identity(document): document for document in documents}
    _require(
        by_identity.get(("ConfigMap", topology["control_config"], "graphos-control")) is not None,
        "template is missing the declared control ConfigMap authority",
    )
    _require(
        by_identity.get(("ConfigMap", topology["cell_config"], "graphos-cell")) is not None,
        "template is missing the declared cell ConfigMap authority",
    )
    for workload_key, identity in _WORKLOAD_DOCS.items():
        document = by_identity.get(identity)
        _require(document is not None, f"template is missing workload {identity}")
        item = topology["workloads"][workload_key]
        spec = document.setdefault("spec", {})
        spec["replicas"] = item["desired"]
        spec["strategy"] = {"type": "RollingUpdate", "rollingUpdate": {"maxSurge": item["scale_up"], "maxUnavailable": 0}}
        template = spec.setdefault("template", {})
        pod = template.setdefault("spec", {})
        pod["terminationGracePeriodSeconds"] = item["termination"]
        containers = pod.get("containers") or []
        _require(len(containers) == 1, f"{identity} must have one application container")
        _set_container_contract(containers[0], item, rollback=rollback)
        template.setdefault("metadata", {}).setdefault("labels", {})["graphos_workload"] = workload_key
        metadata = document.setdefault("metadata", {})
        annotations = metadata.setdefault("annotations", {})
        annotations.update(
            {
                "graphos.io/topology-input-digest": topology["input_digest"],
                "graphos.io/resource-pool-evidence": topology["pool_evidence"],
                "graphos.io/current-image-digest": item["current_image"],
                "graphos.io/rollback-image-digest": item["rollback_image"],
                "graphos.io/rollout-evidence": topology["rollout_evidence"],
                "graphos.io/rollback-evidence": topology["rollback_evidence"],
            }
        )
        pod_annotations = template.setdefault("metadata", {}).setdefault("annotations", {})
        pod_annotations.update({"graphos.io/topology-input-digest": topology["input_digest"], "graphos.io/rollout-evidence": topology["rollout_evidence"]})

    engine_identity = ("StatefulSet", topology["engine_statefulset"], topology["engine_namespace"])
    engine = by_identity.get(engine_identity)
    _require(engine is not None, f"template is missing engine {engine_identity}")
    engine_spec = engine.setdefault("spec", {})
    engine_spec["replicas"] = topology["engine_replicas"]
    engine_spec["updateStrategy"] = {"type": "OnDelete"}
    engine_template = engine_spec.setdefault("template", {})
    engine_pod = engine_template.setdefault("spec", {})
    engine_pod["terminationGracePeriodSeconds"] = topology["engine_termination"]
    engine_containers = engine_pod.get("containers") or []
    _require(len(engine_containers) == 1, "engine must have one application container")
    engine_container = engine_containers[0]
    engine_container["resources"] = {
        "requests": {"cpu": topology["engine_cpu_request"], "memory": topology["engine_memory_request"]},
        "limits": {"cpu": topology["engine_cpu_limit"], "memory": topology["engine_memory_limit"]},
    }
    engine_container["lifecycle"] = {"preStop": {"exec": {"command": ["/bin/sh", "-c", f"sleep {topology['engine_pre_stop']}"]}}}
    if rollback:
        engine_container["image"] = topology["engine_rollback_image"]
    engine_env = {str(entry.get("name")): entry for entry in engine_container.get("env") or () if isinstance(entry, dict)}
    tls_entry = engine_env.get("GRAPH_SERVICE_TLS_SERVER_NAME")
    if tls_entry is not None:
        tls_entry["value"] = topology["engine_tls_name"]
    engine["metadata"].setdefault("annotations", {}).update(
        {
            "graphos.io/topology-input-digest": topology["input_digest"],
            "graphos.io/resource-pool-evidence": topology["pool_evidence"],
            "graphos.io/current-image-digest": topology["engine_current_image"],
            "graphos.io/rollback-image-digest": topology["engine_rollback_image"],
            "graphos.io/rollout-evidence": topology["rollout_evidence"],
            "graphos.io/rollback-evidence": topology["rollback_evidence"],
            "graphos.io/authority-retention": "Retain",
        }
    )
    engine_template.setdefault("metadata", {}).setdefault("annotations", {}).update(
        {
            "graphos.io/topology-input-digest": topology["input_digest"],
            "graphos.io/resource-pool-evidence": topology["pool_evidence"],
            "graphos.io/rollout-evidence": topology["rollout_evidence"],
            "graphos.io/rollback-evidence": topology["rollback_evidence"],
        }
    )
    for document in documents:
        if _identity(document) == ("Service", topology["engine_client"], topology["engine_namespace"]):
            _set_named_port(document, "rpc", topology["engine_port"])
        if _identity(document) == ("Service", topology["engine_peer"], topology["engine_namespace"]):
            _set_named_port(document, "rpc", topology["engine_port"])
            _set_named_port(document, "raft", topology["raft_port"])
            _set_named_port(document, "metrics", topology["engine_metrics_port"])
        if _identity(document) == ("Service", topology["engine_statefulset"], topology["engine_namespace"]):
            _set_named_port(document, "rpc", topology["engine_port"])
            _set_named_port(document, "raft", topology["raft_port"])
            _set_named_port(document, "metrics", topology["engine_metrics_port"])

    for workload_key, identity in _WORKLOAD_DOCS.items():
        item = topology["workloads"][workload_key]
        pdb_name = identity[1]
        pdb = next((document for document in documents if _identity(document) == ("PodDisruptionBudget", pdb_name, identity[2])), None)
        _require(pdb is not None, f"template is missing PDB {pdb_name}")
        pdb_spec = pdb.setdefault("spec", {})
        pdb_spec.pop("maxUnavailable", None)
        pdb_spec["minAvailable"] = item["pdb_min_available"]
    engine_pdb = next((document for document in documents if _identity(document) == ("PodDisruptionBudget", topology["engine_statefulset"], topology["engine_namespace"])), None)
    _require(engine_pdb is not None, "template is missing engine PDB")
    engine_pdb.setdefault("spec", {}).pop("minAvailable", None)
    engine_pdb.setdefault("spec", {})["maxUnavailable"] = topology["engine_pdb_max_unavailable"]

    hpa_names = {"gateway": "graphos-front", "dispatch": "graphos-dispatch-worker", "ingest": "graphos-ingest-worker", "mining": "graphos-analytics-worker"}
    for workload_key, name in hpa_names.items():
        item = topology["workloads"][workload_key]
        hpa = next((document for document in documents if _identity(document) == ("HorizontalPodAutoscaler", name, _WORKLOAD_DOCS[workload_key][2])), None)
        if item["metric"]["mode"] == "fixed":
            if hpa is not None:
                documents.remove(hpa)
            continue
        _require(hpa is not None, f"template is missing HPA {name}")
        hpa_spec = hpa.setdefault("spec", {})
        hpa_spec["minReplicas"] = item["min"]
        hpa_spec["maxReplicas"] = item["max"]
        hpa_spec["behavior"] = {
            "scaleUp": {"stabilizationWindowSeconds": item["up_window"], "selectPolicy": "Max", "policies": [{"type": "Pods", "value": item["scale_up"], "periodSeconds": 60}]},
            "scaleDown": {"stabilizationWindowSeconds": item["down_window"], "selectPolicy": "Max", "policies": [{"type": "Pods", "value": item["scale_down"], "periodSeconds": 60}]},
        }
        # autoscaling/v2 ExternalMetricSource: `metric` and `target` are SIBLINGS
        # under `external` (see deploy/k8s/production-cell/autoscaling.yaml). The
        # previous one-liner both nested `target` inside `metric` and left the
        # metric item dict unclosed, so this module did not even parse.
        hpa_spec["metrics"] = [
            {
                "type": "External",
                "external": {
                    "metric": {
                        "name": item["metric"]["name"],
                        "selector": {
                            "matchLabels": {"graphos_workload": workload_key}
                        },
                    },
                    "target": {
                        "type": "AverageValue",
                        "averageValue": item["metric"]["target"],
                    },
                },
            }
        ]
    config_values = {
        "GRAPH_SERVICE_ENDPOINTS": topology["engine_endpoint"],
        "ENGINE_IDENTITY_CONTRACT_REF": topology["engine_identity_ref"],
        "ENGINE_IDENTITY_CONTRACT_DIGEST": topology["engine_identity_digest"],
        "ENGINE_CLIENT_PORT": str(topology["engine_port"]),
        "ENGINE_PEER_PORT": str(topology["raft_port"]),
        "ENGINE_METRICS_PORT": str(topology["engine_metrics_port"]),
        "ENGINE_TLS_SERVER_NAME": topology["engine_tls_name"],
        "ENGINE_DISCOVERY_REF": topology["engine_discovery"],
        "ENGINE_DISCOVERY_VERIFIED": "true",
        "OIDC_ISSUER": topology["oidc_issuer"],
        "OIDC_JWKS_URI": topology["oidc_jwks"],
        "OIDC_AUDIENCE": topology["oidc_audience"],
        "OIDC_CLIENT_ID": topology["oidc_client_id"],
        "OIDC_DISCOVERY_VERIFIED": "true",
        "CONFIG_AUTHORITY_REF": topology["control_config_ref"],
        "CELL_CONFIG_AUTHORITY_REF": topology["cell_config_ref"],
        "CONTROL_CONFIG_AUTHORITY_NAME": topology["control_config"],
        "CELL_CONFIG_AUTHORITY_NAME": topology["cell_config"],
        "SECRET_AUTHORITY_REF": topology["secret_ref"],
        "SECRET_AUTHORITY_NAME": topology["secret_name"],
        "SESSION_STORE_SERVICE": f"{topology['session_service']}.{topology['session_namespace']}.svc.cluster.local",
        "SESSION_STORE_AUTHORITY_REF": topology["session_ref"],
        "SESSION_STORE_RETENTION": "Retain",
        "ACTION_AUDIT_AUTHORITY_REF": topology["action_audit_ref"],
        "ACTION_AUDIT_GRAPH": topology["action_audit_graph"],
        "ACTION_AUDIT_KIND": topology["action_audit_kind"],
        "ACTION_AUDIT_RETENTION": "Retain",
        "TOPOLOGY_INPUT_DIGEST": topology["input_digest"],
        "RESOURCE_POOL_EVIDENCE_REF": topology["pool_evidence"],
        "ROLLOUT_EVIDENCE_REF": topology["rollout_evidence"],
        "ROLLBACK_EVIDENCE_REF": topology["rollback_evidence"],
    }
    for document in documents:
        identity = _identity(document)
        if identity in {
            ("ConfigMap", topology["control_config"], "graphos-control"),
            ("ConfigMap", topology["cell_config"], "graphos-cell"),
        }:
            document.setdefault("metadata", {}).setdefault("annotations", {}).update(
                {
                    "graphos.io/authority-ref": topology["control_config_ref"] if identity[1] == topology["control_config"] else topology["cell_config_ref"],
                    "graphos.io/authority-retention": "Retain",
                    "graphos.io/topology-input-digest": topology["input_digest"],
                }
            )
            data = document.setdefault("data", {})
            data.update(config_values)
            if identity[1] == topology["cell_config"]:
                data["EPISTEMIC_GRAPH_RAFT_GROUPS"] = str(topology["engine_raft_groups"])
        if identity == ("ConfigMap", "graphos-prometheus-adapter-rules", "graphos-control"):
            document.setdefault("metadata", {}).setdefault("annotations", {}).update(
                {
                    "graphos.io/authority-ref": "agent-utilities/deploy/k8s/production-cell",
                    "graphos.io/topology-input-digest": topology["input_digest"],
                }
            )
            document.setdefault("data", {})["external-rules.yaml"] = _adapter_rules(topology)
    return documents


def _adapter_rules(topology: dict[str, Any]) -> str:
    """Return external metric rules bound to the same HPA names/selectors."""

    rules = []
    for key in ("gateway", "dispatch", "ingest"):
        metric = topology["workloads"][key]["metric"]
        rules.append(
            "      - seriesQuery: '{name}'\n"
            "        name: {{matches: '^{name}$'}}\n"
            "        metricsQuery: 'sum({name}{{<<.LabelMatchers>>}})'".format(name=metric["name"])
        )
    return "externalRules:\n" + "\n".join(rules) + "\n"
