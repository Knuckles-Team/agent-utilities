#!/usr/bin/env python3
"""Fail-closed static gate for the GraphOS production-cell deployment assets.

This checker is deliberately independent of a cluster.  It rejects incomplete
templates before release rendering, then adds exact-image checks for rendered
output.  Runtime secrets, endpoints, identities and certificate material are
never accepted as committed manifest data.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml


class ProductionAssetError(ValueError):
    """A production deployment invariant is absent or unsafe."""


_WORKLOAD_KINDS = {"Deployment", "StatefulSet", "DaemonSet", "Job", "CronJob"}
_SERVING_WORKLOADS = {"Deployment", "StatefulSet", "DaemonSet"}
_REQUIRED_OBJECTS = {
    ("Deployment", "graphos-front", "graphos-control"),
    ("StatefulSet", "epistemic-graph-raft", "graphos-cell"),
    ("Service", "epistemic-graph-coordinator", "graphos-cell"),
    ("Deployment", "graphos-dispatch-worker", "graphos-cell"),
    ("Deployment", "graphos-ingest-worker", "graphos-cell"),
    ("Deployment", "graphos-analytics-worker", "graphos-cell"),
    ("CronJob", "graphos-backup", "graphos-cell"),
    ("CronJob", "graphos-restore-validation", "graphos-cell"),
    ("HorizontalPodAutoscaler", "graphos-front", "graphos-control"),
    ("HorizontalPodAutoscaler", "graphos-dispatch-worker", "graphos-cell"),
    ("HorizontalPodAutoscaler", "graphos-ingest-worker", "graphos-cell"),
    ("HorizontalPodAutoscaler", "graphos-analytics-worker", "graphos-cell"),
    ("PodDisruptionBudget", "graphos-front", "graphos-control"),
    ("PodDisruptionBudget", "epistemic-graph-raft", "graphos-cell"),
    ("PodDisruptionBudget", "graphos-dispatch-worker", "graphos-cell"),
    ("PodDisruptionBudget", "graphos-ingest-worker", "graphos-cell"),
    ("PodDisruptionBudget", "graphos-analytics-worker", "graphos-cell"),
    ("NetworkPolicy", "default-deny", "graphos-control"),
    ("NetworkPolicy", "default-deny", "graphos-cell"),
    ("NetworkPolicy", "graphos-front-ingress", "graphos-control"),
    ("NetworkPolicy", "graphos-front-egress", "graphos-control"),
    ("NetworkPolicy", "engine-ingress", "graphos-cell"),
    ("NetworkPolicy", "cell-egress", "graphos-cell"),
    ("NetworkPolicy", "engine-raft-egress", "graphos-cell"),
    ("PrometheusRule", "graphos-slo", "graphos-control"),
    ("ServiceMonitor", "graphos-front", "graphos-control"),
    ("ServiceMonitor", "epistemic-graph", "graphos-cell"),
    ("PeerAuthentication", "default-strict", "graphos-control"),
    ("PeerAuthentication", "default-strict", "graphos-cell"),
    ("DestinationRule", "epistemic-graph-mtls", "graphos-control"),
    ("DestinationRule", "epistemic-graph-mtls", "graphos-cell"),
    ("DestinationRule", "epistemic-graph-coordinator-mtls", "graphos-cell"),
    ("AuthorizationPolicy", "epistemic-graph-authority", "graphos-cell"),
}
_REQUIRED_CONFIG = {
    "APP_PROFILE": "production",
    "AGENT_BUS_LOG_BACKEND": "engine",
    "TRACE_EXPORT_ENABLED": "true",
    "LANGFUSE_CAPTURE_CONTENT": "false",
    "USAGE_DB_BACKEND": "postgres",
    "USAGE_TRACKING_ENABLED": "true",
    "USAGE_CONTENT_RETENTION": "metadata",
    "ENABLE_OTEL": "true",
    "PERSISTENCE_IDENTITY_HMAC_KEY_REF": "env://PERSISTENCE_IDENTITY_HMAC_KEY",
}
_FORBIDDEN_TEXT = (
    re.compile(r"(?:[A-Za-z]:\\|/home/|/Users/|/mnt/[a-z]/|file://)"),
    re.compile(r"\b(?:sk-lf|pk-lf|Bearer\s+)[A-Za-z0-9_-]+", re.IGNORECASE),
    re.compile(r"(?:\.internal|\.corp)(?=[:/\s\"']|$)", re.IGNORECASE),
)
_CERTIFICATION_SCENARIOS = {
    "identity-tls-policy-trace",
    "kill-commit-phases",
    "worker-process-loss",
    "raft-leader-loss",
    "broker-leader-loss",
    "node-loss",
    "zone-isolation",
    "broker-rebalance",
    "online-reshard",
    "atomic-exact-release-cutover",
    "one-time-index-migration",
    "one-time-ontology-migration",
    "backup-restore",
    "regional-recovery",
    "policy-and-deletion-propagation",
}


def _documents(directory: Path) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.yaml")):
        for value in yaml.safe_load_all(path.read_text(encoding="utf-8")):
            if value is None:
                continue
            if not isinstance(value, dict):
                raise ProductionAssetError(
                    f"{path.name} contains a non-object document"
                )
            value["__source"] = path.name
            documents.append(value)
    return documents


def _identity(value: dict[str, Any]) -> tuple[str, str, str]:
    metadata = value.get("metadata") or {}
    return (
        str(value.get("kind") or ""),
        str(metadata.get("name") or ""),
        str(metadata.get("namespace") or ""),
    )


def _pod_spec(value: dict[str, Any]) -> dict[str, Any]:
    kind = value.get("kind")
    spec = value.get("spec") or {}
    if kind == "CronJob":
        return spec["jobTemplate"]["spec"]["template"]["spec"]
    if kind == "Job":
        return spec["template"]["spec"]
    return spec["template"]["spec"]


def _pod_metadata(value: dict[str, Any]) -> dict[str, Any]:
    spec = value.get("spec") or {}
    if value.get("kind") == "CronJob":
        return spec["jobTemplate"]["spec"]["template"].get("metadata") or {}
    return (spec.get("template") or {}).get("metadata") or {}


def _all_containers(pod: dict[str, Any]) -> Iterable[dict[str, Any]]:
    yield from pod.get("initContainers") or ()
    yield from pod.get("containers") or ()


def _validate_workload(value: dict[str, Any]) -> None:
    kind, name, _ = _identity(value)
    pod = _pod_spec(value)
    annotations = _pod_metadata(value).get("annotations") or {}
    if (
        annotations.get("graphos.io/mtls-mode") != "strict"
        or annotations.get("sidecar.istio.io/nativeSidecar") != "true"
        or "holdApplicationUntilProxyStarts"
        not in str(annotations.get("proxy.istio.io/config") or "")
    ):
        raise ProductionAssetError(f"{kind}/{name} is not bound to strict mesh startup")
    if pod.get("automountServiceAccountToken") is not False:
        raise ProductionAssetError(
            f"{kind}/{name} must disable ambient service-account tokens"
        )
    if not pod.get("serviceAccountName"):
        raise ProductionAssetError(f"{kind}/{name} has no workload identity")
    projected = [
        volume
        for volume in pod.get("volumes") or ()
        if isinstance(volume, dict)
        and isinstance(volume.get("projected"), dict)
        and any(
            "serviceAccountToken" in source
            for source in volume["projected"].get("sources") or ()
        )
    ]
    if not projected:
        raise ProductionAssetError(
            f"{kind}/{name} has no audience-bound projected token"
        )
    containers = list(_all_containers(pod))
    if not containers:
        raise ProductionAssetError(f"{kind}/{name} has no containers")
    for container in containers:
        container_name = str(container.get("name") or "unnamed")
        image = str(container.get("image") or "")
        if not image or "latest" in image.casefold():
            raise ProductionAssetError(
                f"{kind}/{name}/{container_name} has a mutable image"
            )
        resources = container.get("resources") or {}
        if not (resources.get("requests") and resources.get("limits")):
            raise ProductionAssetError(
                f"{kind}/{name}/{container_name} has no resource bounds"
            )
        security = container.get("securityContext") or {}
        if (
            security.get("allowPrivilegeEscalation") is not False
            or security.get("readOnlyRootFilesystem") is not True
            or security.get("runAsNonRoot") is not True
            or (security.get("capabilities") or {}).get("drop") != ["ALL"]
        ):
            raise ProductionAssetError(
                f"{kind}/{name}/{container_name} is not restricted"
            )
        env_from = container.get("envFrom") or ()
        if not any(
            (entry.get("secretRef") or {}).get("name") == "graphos-runtime-secrets"
            and (entry.get("secretRef") or {}).get("optional") is False
            for entry in env_from
            if isinstance(entry, dict)
        ):
            raise ProductionAssetError(
                f"{kind}/{name}/{container_name} has no required runtime secret ref"
            )
        if kind in _SERVING_WORKLOADS and not all(
            probe in container
            for probe in ("startupProbe", "readinessProbe", "livenessProbe")
        ):
            raise ProductionAssetError(
                f"{kind}/{name}/{container_name} is missing serving probes"
            )


def _validate_config(documents: list[dict[str, Any]]) -> None:
    configs = {
        _identity(value)[1]: value.get("data") or {}
        for value in documents
        if value.get("kind") == "ConfigMap"
    }
    for name in ("graphos-security-contract", "graphos-cell-contract"):
        data = configs.get(name)
        if not isinstance(data, dict):
            raise ProductionAssetError(f"required ConfigMap/{name} is absent")
        for key, expected in _REQUIRED_CONFIG.items():
            if str(data.get(key)) != expected:
                raise ProductionAssetError(
                    f"ConfigMap/{name} must set {key}={expected}"
                )
        if "OTEL_EXPORTER_OTLP_ENDPOINT" in data:
            raise ProductionAssetError(
                "OTEL exporter endpoint must come from the runtime Secret"
            )
        if "PERSISTENCE_IDENTITY_HMAC_KEY" in data:
            raise ProductionAssetError(
                "identity HMAC key material must come from the runtime Secret"
            )
    cell = configs["graphos-cell-contract"]
    control = configs["graphos-security-contract"]
    for key, expected in {
        "MCP_TOOL_MODE": "intent",
        "MCP_CLIENT_AUTH": "oidc-client-credentials",
        "LANGFUSE_MCP_ENABLED": "true",
        "KG_LOOP": "true",
        "KG_LOOP_BREADTH": "true",
        "KG_LOOP_MINE_DISCOVERY": "true",
        "KG_LOOP_BELIEF_REVISION": "true",
        "KG_LOOP_INSIGHT_VALIDATION": "true",
        "KG_LOOP_TRACE_MINING": "true",
        "KG_OPTIMIZATION_ENABLED": "true",
        "KG_FAILURE_EVOLUTION": "true",
        "KG_FAILURE_REGRESSION_DATASET": "false",
        "KG_LOOP_DISCOVER": "false",
        "KG_LOOP_DISTILL": "false",
        "KG_LOOP_STANDARDIZE": "false",
        "KG_GOLDEN_AUTO_MERGE": "false",
        "KG_AGENT_AUTO_APPLY": "false",
        "KG_LOOP_AUTO_DEVELOP": "false",
        "KG_INSIGHT_AUTONOMY": "false",
    }.items():
        if str(control.get(key)) != expected:
            raise ProductionAssetError(f"control contract must set {key}={expected}")
    expected_cell = {
        "EPISTEMIC_GRAPH_RAFT_GROUPS": "20",
        "EPISTEMIC_GRAPH_REQUIRE_VERIFIED_CONTEXT": "1",
        "EPISTEMIC_GRAPH_REQUIRE_SIGNED": "1",
        "EPISTEMIC_GRAPH_RLS_DEFAULT_DENY": "1",
        "EG_ANALYTICS_WORKERS": "0",
        "GRAPHOS_BACKUP_RETENTION_COUNT": "2",
    }
    for key, expected in expected_cell.items():
        if str(cell.get(key)) != expected:
            raise ProductionAssetError(f"cell contract must set {key}={expected}")
    coordinator = "tls://epistemic-graph-coordinator.graphos-cell.svc:9100"
    if cell.get("GRAPH_SERVICE_ENDPOINTS") != coordinator:
        raise ProductionAssetError(
            "all clients must use the replicated TLS graph service"
        )
    if any(
        key in cell
        for key in {
            "GRAPH_SERVICE_TCP_" + "ADDR",
            "EG_ANALYTICS_COORDINATOR_ENDPOINT",
            "EG_ANALYTICS_COORDINATOR_TCP_ADDR",
        }
    ):
        raise ProductionAssetError(
            "production must not retain plaintext native-engine fallback settings"
        )
    if control.get("GRAPH_SERVICE_ENDPOINTS") != coordinator:
        raise ProductionAssetError(
            "control plane must use the TLS coordinator authority"
        )


def _validate_engine(documents: list[dict[str, Any]]) -> None:
    engines = [
        value
        for value in documents
        if _identity(value)[:2] == ("StatefulSet", "epistemic-graph-raft")
    ]
    if len(engines) != 1:
        raise ProductionAssetError(
            "exactly one authoritative engine StatefulSet is required"
        )
    engine = engines[0]
    if engine.get("spec", {}).get("replicas") != 3:
        raise ProductionAssetError(
            "the production Raft group requires exactly three members"
        )
    if engine.get("spec", {}).get("updateStrategy") != {"type": "OnDelete"}:
        raise ProductionAssetError(
            "engine releases require explicit atomic OnDelete activation"
        )
    template = engine["spec"]["template"]
    labels = template.get("metadata", {}).get("labels") or {}
    if labels.get("graphos.io/worker-role") != "projection-index-reasoning":
        raise ProductionAssetError(
            "correctness-critical projection/index/reasoning must stay with authority"
        )
    claims = engine["spec"].get("volumeClaimTemplates") or ()
    if not any(
        claim.get("metadata", {}).get("name") == "engine-data"
        and claim.get("spec", {}).get("storageClassName") == "graphos-retained-rwo"
        for claim in claims
    ):
        raise ProductionAssetError("engine has no mounted per-member PVC template")
    container = engine["spec"]["template"]["spec"]["containers"][0]
    mounts = {
        mount.get("name"): mount.get("mountPath")
        for mount in container.get("volumeMounts") or ()
    }
    if (
        mounts.get("engine-data") != "/var/lib/epistemic-graph/data"
        or mounts.get("object-archive") != "/archive"
        or mounts.get("engine-tls") != "/var/run/graphos/engine-tls"
    ):
        raise ProductionAssetError(
            "engine authoritative, archive or TLS identity storage is not mounted"
        )
    env = {
        entry.get("name"): entry.get("value")
        for entry in container.get("env") or ()
        if isinstance(entry, dict)
    }
    if (
        env.get("GRAPH_SERVICE_TLS_CERT") != "/var/run/graphos/engine-tls/tls.crt"
        or env.get("GRAPH_SERVICE_TLS_KEY") != "/var/run/graphos/engine-tls/tls.key"
    ):
        raise ProductionAssetError(
            "native engine server identity is not runtime-mounted"
        )
    if not str(env.get("GRAPH_SERVICE_TLS_SERVER_NAME") or "").strip():
        raise ProductionAssetError("native engine TLS server name is absent")
    command_text = " ".join(str(value) for value in container.get("args") or ())
    if "ALLOW_PLAINTEXT" in command_text or "allow-plaintext" in command_text:
        raise ProductionAssetError(
            "native engine plaintext acknowledgement is forbidden"
        )
    if "checkpoint-interval" in command_text:
        raise ProductionAssetError("retired engine checkpoint flags are forbidden")
    readiness = container.get("readinessProbe") or {}
    readiness_command = " ".join(
        str(value) for value in (readiness.get("exec") or {}).get("command") or ()
    )
    if (
        "ssl.create_default_context" not in readiness_command
        or "server_hostname" not in readiness_command
    ):
        raise ProductionAssetError(
            "engine readiness must validate native TLS trust and hostname"
        )
    volumes = {
        volume.get("name"): volume
        for volume in template.get("spec", {}).get("volumes") or ()
        if isinstance(volume, dict)
    }
    tls_secret = (volumes.get("engine-tls") or {}).get("secret") or {}
    if (
        tls_secret.get("secretName") != "graphos-engine-tls"
        or tls_secret.get("optional") is not False
    ):
        raise ProductionAssetError("required native engine TLS Secret is absent")
    peer_values = [
        entry.get("value", "")
        for entry in container.get("env") or ()
        if entry.get("name") == "EPISTEMIC_GRAPH_RAFT_PEERS"
    ]
    if len(peer_values) != 1 or any(
        f"{ordinal}@epistemic-graph-raft-{ordinal}." not in peer_values[0]
        for ordinal in range(3)
    ):
        raise ProductionAssetError("engine Raft peer set is incomplete")
    coordinator_services = [
        value
        for value in documents
        if _identity(value)[:2] == ("Service", "epistemic-graph-coordinator")
    ]
    selector = coordinator_services[0].get("spec", {}).get("selector", {})
    if selector != {"app.kubernetes.io/name": "epistemic-graph-raft"}:
        raise ProductionAssetError(
            "graph service must select every ready replicated member"
        )


def _validate_worker_autoscaling(documents: list[dict[str, Any]]) -> None:
    expected_commands = {
        "graphos-front": "graph-os",
        "graphos-dispatch-worker": "agent-dispatch-worker",
        "graphos-ingest-worker": "kg-ingest-worker",
        "graphos-analytics-worker": "graph-os-analytics-worker",
    }
    expected_metrics = {
        "graphos-front": "agent_utilities_gateway_request_p99_seconds",
        "graphos-dispatch-worker": "agent_utilities_dispatch_queue_depth",
        "graphos-ingest-worker": "agent_utilities_kg_ingest_consumer_lag",
        "graphos-analytics-worker": "epistemic_graph_analytics_jobs_ready",
    }
    by_identity = {_identity(value): value for value in documents}
    for name, command in expected_commands.items():
        namespace = "graphos-control" if name == "graphos-front" else "graphos-cell"
        deployment = by_identity.get(("Deployment", name, namespace)) or {}
        if deployment.get("spec", {}).get("strategy") != {"type": "Recreate"}:
            raise ProductionAssetError(
                f"{name} must use exact-release Recreate activation"
            )
        containers = _pod_spec(deployment).get("containers") or ()
        if (
            len(containers) != 1
            or (containers[0].get("command") or [None])[0] != command
        ):
            raise ProductionAssetError(
                f"Deployment/{name} does not run its native worker"
            )
    for name, metric_name in expected_metrics.items():
        namespace = "graphos-control" if name == "graphos-front" else "graphos-cell"
        hpa = by_identity.get(("HorizontalPodAutoscaler", name, namespace)) or {}
        metrics = (hpa.get("spec") or {}).get("metrics") or ()
        external_names = {
            ((metric.get("external") or {}).get("metric") or {}).get("name")
            for metric in metrics
            if metric.get("type") == "External"
        }
        if metric_name not in external_names:
            raise ProductionAssetError(
                f"HorizontalPodAutoscaler/{name} lacks its authority signal"
            )


def _validate_storage(documents: list[dict[str, Any]]) -> None:
    claims = {
        _identity(value)[1]: value.get("spec") or {}
        for value in documents
        if value.get("kind") == "PersistentVolumeClaim"
    }
    archive = claims.get("graphos-object-archive") or {}
    if archive.get(
        "storageClassName"
    ) != "graphos-cross-cell-object-rwx" or archive.get("accessModes") != [
        "ReadWriteMany"
    ]:
        raise ProductionAssetError(
            "backup archive is not bound to cross-cell RWX storage"
        )
    restore = claims.get("graphos-restore-validation") or {}
    if restore.get("storageClassName") != "graphos-retained-rwo" or restore.get(
        "accessModes"
    ) != ["ReadWriteOnce"]:
        raise ProductionAssetError(
            "restore scratch is not bound to retained RWO storage"
        )


def _validate_backup_restore(documents: list[dict[str, Any]]) -> None:
    cron_jobs = {
        _identity(value)[1]: value
        for value in documents
        if value.get("kind") == "CronJob"
    }
    for name in ("graphos-backup", "graphos-restore-validation"):
        job = cron_jobs.get(name) or {}
        spec = job.get("spec") or {}
        if spec.get("concurrencyPolicy") != "Forbid":
            raise ProductionAssetError(
                f"CronJob/{name} must serialize archive operations"
            )
        containers = _pod_spec(job).get("containers") or ()
        if len(containers) != 1:
            raise ProductionAssetError(f"CronJob/{name} has no single operation")
        archive_mounts = [
            mount
            for mount in containers[0].get("volumeMounts") or ()
            if mount.get("name") == "object-archive"
            and mount.get("mountPath") == "/archive"
        ]
        if len(archive_mounts) != 1 or archive_mounts[0].get("readOnly") is True:
            raise ProductionAssetError(
                f"CronJob/{name} cannot coordinate the shared archive lock"
            )


def _validate_mesh(documents: list[dict[str, Any]]) -> None:
    namespaces = {
        _identity(value)[1]: (value.get("metadata") or {}).get("labels") or {}
        for value in documents
        if value.get("kind") == "Namespace"
    }
    for name in ("graphos-control", "graphos-cell"):
        labels = namespaces.get(name) or {}
        if (
            labels.get("istio-injection") != "enabled"
            or labels.get("pod-security.kubernetes.io/enforce") != "restricted"
        ):
            raise ProductionAssetError(
                f"Namespace/{name} lacks strict mesh and pod-security admission"
            )
    peer_auth = [
        value
        for value in documents
        if value.get("kind") == "PeerAuthentication"
        and _identity(value)[1] == "default-strict"
    ]
    if {
        (_identity(value)[2], (value.get("spec") or {}).get("mtls", {}).get("mode"))
        for value in peer_auth
    } != {
        ("graphos-control", "STRICT"),
        ("graphos-cell", "STRICT"),
    }:
        raise ProductionAssetError(
            "both production namespaces require mesh-wide STRICT mTLS"
        )
    policies = [
        value
        for value in documents
        if _identity(value)[:2] == ("AuthorizationPolicy", "epistemic-graph-authority")
    ]
    if len(policies) != 1:
        raise ProductionAssetError(
            "engine service-account authorization policy is absent"
        )
    policy_text = json.dumps(policies[0], sort_keys=True)
    for service_account in (
        "graphos-front",
        "graphos-engine",
        "graphos-dispatch",
        "graphos-ingest",
        "graphos-analytics",
        "graphos-backup",
    ):
        if f"/sa/{service_account}" not in policy_text:
            raise ProductionAssetError(
                "engine authorization policy omits a workload identity"
            )


def _validate_network_boundaries(documents: list[dict[str, Any]]) -> None:
    policies = {
        (_identity(value)[1], _identity(value)[2]): value
        for value in documents
        if value.get("kind") == "NetworkPolicy"
    }
    engine_ingress = policies.get(("engine-ingress", "graphos-cell")) or {}
    backup_allowed = False
    raft_isolated = False
    for rule in (engine_ingress.get("spec") or {}).get("ingress") or ():
        ports = {int(port.get("port")) for port in rule.get("ports") or ()}
        sources = rule.get("from") or ()
        if 9100 in ports and any(
            (source.get("podSelector") or {})
            .get("matchLabels", {})
            .get("graphos.io/operation")
            == "backup"
            for source in sources
        ):
            backup_allowed = True
        if 9200 in ports:
            raft_isolated = len(sources) == 1 and (
                (sources[0].get("podSelector") or {})
                .get("matchLabels", {})
                .get("app.kubernetes.io/name")
                == "epistemic-graph-raft"
                and "namespaceSelector" not in sources[0]
            )
    if not backup_allowed:
        raise ProductionAssetError("backup identity cannot reach the engine authority")
    if not raft_isolated:
        raise ProductionAssetError("Raft ingress is not isolated to engine members")

    common_egress = policies.get(("cell-egress", "graphos-cell")) or {}
    if any(
        int(port.get("port")) == 9200
        for rule in (common_egress.get("spec") or {}).get("egress") or ()
        for port in rule.get("ports") or ()
    ):
        raise ProductionAssetError("common cell workloads may not reach the Raft port")
    raft_egress = policies.get(("engine-raft-egress", "graphos-cell")) or {}
    selector = (raft_egress.get("spec") or {}).get("podSelector", {})
    if selector.get("matchLabels", {}).get(
        "app.kubernetes.io/name"
    ) != "epistemic-graph-raft" or not any(
        int(port.get("port")) == 9200
        for rule in (raft_egress.get("spec") or {}).get("egress") or ()
        for port in rule.get("ports") or ()
    ):
        raise ProductionAssetError("Raft egress is not isolated to engine members")


def _validate_rendered_image_pins(directory: Path) -> None:
    value = yaml.safe_load(
        (directory / "kustomization.yaml").read_text(encoding="utf-8")
    )
    images = {
        str(image.get("name") or ""): image
        for image in value.get("images") or ()
        if isinstance(image, dict)
    }
    for name in ("graph-os-image", "epistemic-graph-image"):
        image = images.get(name) or {}
        digest = str(image.get("digest") or "")
        repository = str(image.get("newName") or "")
        if not re.fullmatch(r"sha256:[a-f0-9]{64}", digest):
            raise ProductionAssetError(f"rendered image {name} is not digest-pinned")
        if not repository or ".invalid" in repository or "newTag" in image:
            raise ProductionAssetError(
                f"rendered image {name} has no immutable repository pin"
            )


def _validate_certification_definition(repository_root: Path) -> None:
    path = repository_root / "deploy" / "release" / "certification-campaign.yml"
    campaign = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(campaign, dict):
        raise ProductionAssetError("certification campaign is not a mapping")
    if (
        campaign.get("apiVersion") != "graphos.io/v1"
        or campaign.get("kind") != "CertificationCampaign"
        or campaign.get("campaignVersion") != 1
        or float(campaign.get("scale") or 0) != 1.0
        or not 86400 <= int(campaign.get("durationSeconds") or 0) <= 259200
    ):
        raise ProductionAssetError(
            "certification campaign is not a scale=1 production run"
        )
    scenarios = campaign.get("scenarios")
    if not isinstance(scenarios, list):
        raise ProductionAssetError("certification campaign has no scenarios")
    scenario_ids = {str(item.get("id") or "") for item in scenarios}
    if scenario_ids != _CERTIFICATION_SCENARIOS or len(scenarios) != len(scenario_ids):
        raise ProductionAssetError("certification campaign scenario set is not exact")
    fractions = [float(item.get("atFraction", -1)) for item in scenarios]
    if fractions != sorted(fractions) or not all(
        0 <= value <= 0.95 for value in fractions
    ):
        raise ProductionAssetError(
            "certification faults are not ordered inside the campaign"
        )
    commit_scenario = next(
        item for item in scenarios if item["id"] == "kill-commit-phases"
    )
    if commit_scenario.get("phases") != [
        "before-proposal",
        "after-proposal-before-commit",
        "after-authoritative-commit",
        "before-projection-publication",
        "before-acknowledgement",
    ]:
        raise ProductionAssetError(
            "certification does not cover every commit kill point"
        )
    targets = campaign.get("targets") or {}
    if (
        float(targets.get("rpoSeconds") or float("inf")) > 60
        or float(targets.get("rtoSeconds") or float("inf")) > 300
        or int(targets.get("maximumQueueDepth") or 0) <= 0
    ):
        raise ProductionAssetError("certification recovery/SLO targets are incomplete")
    soak_test = (
        repository_root
        / "tests"
        / "scale"
        / "soak"
        / "test_production_certification.py"
    )
    campaign_source = (
        repository_root / "scripts" / "certification" / "campaign.py"
    ).read_text(encoding="utf-8")
    if not soak_test.is_file() or "pytest.skip" in soak_test.read_text(
        encoding="utf-8"
    ):
        raise ProductionAssetError(
            "production certification must be executable and fail closed"
        )
    if (
        "mock mode" in campaign_source.casefold()
        and "no mock mode" not in campaign_source.casefold()
    ):
        raise ProductionAssetError(
            "production certification contains a mock execution path"
        )


def check(directory: Path, *, rendered: bool, repository_root: Path) -> dict[str, Any]:
    if not directory.is_dir():
        raise ProductionAssetError("production manifest directory is absent")
    raw = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(directory.glob("*"))
        if path.is_file()
    )
    for pattern in _FORBIDDEN_TEXT:
        if pattern.search(raw):
            raise ProductionAssetError(
                "production assets contain a forbidden identifying or secret value"
            )
    if "workload-mtls" in raw:
        raise ProductionAssetError(
            "static workload certificates must not bypass mesh certificate rotation"
        )
    documents = _documents(directory)
    identities = {_identity(value) for value in documents}
    missing = sorted(_REQUIRED_OBJECTS - identities)
    if missing:
        raise ProductionAssetError(f"production resource set is incomplete: {missing}")
    if any(value.get("kind") == "Secret" for value in documents):
        raise ProductionAssetError("runtime Secret material must not be committed")
    for value in documents:
        if value.get("kind") in _WORKLOAD_KINDS:
            _validate_workload(value)
    _validate_config(documents)
    _validate_engine(documents)
    _validate_worker_autoscaling(documents)
    _validate_storage(documents)
    _validate_backup_restore(documents)
    _validate_mesh(documents)
    _validate_network_boundaries(documents)
    if rendered:
        _validate_rendered_image_pins(directory)
    if (repository_root / "deploy" / "k8s" / "graphos.yaml").exists():
        raise ProductionAssetError(
            "obsolete single-owner deployment manifest is still present"
        )
    required_release_assets = (
        "compatibility-matrix.yml",
        "compatibility-matrix.schema.json",
        "release-manifest.schema.json",
        "certification-campaign.yml",
        "certification-campaign.schema.json",
        "operational-evidence.schema.json",
        "connector-live-certification-ledger.schema.json",
        "index-migration-catalog.schema.json",
        "index-migrations.catalog.json",
    )
    absent = [
        name
        for name in required_release_assets
        if not (repository_root / "deploy" / "release" / name).is_file()
    ]
    if absent:
        raise ProductionAssetError(f"release/certification assets are absent: {absent}")
    if (
        repository_root / "tests" / "scale" / "soak" / "test_hardware_pending.py"
    ).exists():
        raise ProductionAssetError(
            "hardware-pending skip suite must be replaced by executable certification"
        )
    _validate_certification_definition(repository_root)
    return {"ok": True, "documents": len(documents), "rendered": rendered}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="check-graphos-production-assets")
    parser.add_argument(
        "--directory", type=Path, default=Path("deploy/k8s/production-cell")
    )
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--rendered", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = check(
            args.directory,
            rendered=args.rendered,
            repository_root=args.repository_root,
        )
    except Exception as exc:  # noqa: BLE001 - one privacy-safe CLI boundary
        print(json.dumps({"ok": False, "error": type(exc).__name__}, sort_keys=True))
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
