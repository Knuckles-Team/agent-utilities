from __future__ import annotations

import json
import logging
import math
import time
import typing
import uuid
from pathlib import Path

import yaml

from agent_utilities.core.config import setting
from agent_utilities.security.persistence_privacy import persistence_reference

if typing.TYPE_CHECKING:
    from .._engine_protocol import _EngineProtocol

    _Base = _EngineProtocol
else:
    _Base = object

from ...models.domains.infrastructure import (
    CrossTenantInsightNode,
    GPUAcceleratorNode,
    MCPServerPackageNode,
    PullRequestNode,
    StorageArrayNode,
)
from ...models.knowledge_graph import HostNode
from .engine_action_result import _persist_action_result

logger = logging.getLogger(__name__)

_MAX_INVENTORY_BYTES = 16 * 1024 * 1024
_MAX_INVENTORY_GROUPS = 10_000
_MAX_INVENTORY_HOSTS = 100_000
_INVENTORY_ROLES = {"compute", "compute_high", "gpu", "storage"}
_INVENTORY_STORAGE_TYPES = {"hdd", "hybrid", "nvme", "object", "sas", "ssd"}
_INVENTORY_OS_TYPES = {"darwin", "freebsd", "linux", "unix", "windows"}
_INVENTORY_ARCHES = {
    "aarch64",
    "amd64",
    "arm64",
    "ppc64le",
    "riscv64",
    "x86_64",
}
_INVENTORY_GPU_VENDORS = {"amd", "apple", "intel", "nvidia"}
_InventoryHost = tuple[str, dict[str, typing.Any], dict[str, typing.Any]]
_HostRecord = tuple[HostNode, str, dict[str, str], str]
_INVALID_INVENTORY = object()


def _read_inventory_file(path: Path) -> typing.Any:
    """Read one bounded YAML inventory, preserving its parsed value."""
    try:
        if not path.is_file() or path.stat().st_size > _MAX_INVENTORY_BYTES:
            logger.warning("Configured infrastructure inventory is not a bounded file")
            return _INVALID_INVENTORY
        with open(path) as stream:
            return yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        logger.warning(
            "Configured infrastructure inventory could not be read (%s)",
            type(exc).__name__,
        )
        return _INVALID_INVENTORY


def _load_inventory_root(
    inventory_path: str | None,
) -> dict[str, typing.Any] | None:
    """Resolve and parse the configured XDG-backed inventory file."""
    configured_path = inventory_path
    if configured_path is None:
        configured_path = str(setting("INFRA_INVENTORY_PATH", "") or "").strip()
    if not configured_path:
        logger.warning("Infrastructure inventory path is not configured")
        return None

    path = Path(configured_path).expanduser()
    if not path.exists():
        logger.warning("Configured infrastructure inventory is unavailable")
        return None
    data = _read_inventory_file(path)
    if data is _INVALID_INVENTORY:
        return None

    root = data.get("all", data) if isinstance(data, dict) else {}
    if not isinstance(root, dict):
        logger.error("Infrastructure inventory root must be a mapping")
        return None
    return root


def _append_group_hosts(
    group: dict[str, typing.Any],
    inherited: dict[str, typing.Any],
    discovered: list[_InventoryHost],
) -> None:
    """Append bounded host records from one inventory group."""
    hosts = group.get("hosts", {})
    if not isinstance(hosts, dict):
        return
    for alias, raw in hosts.items():
        if len(discovered) >= _MAX_INVENTORY_HOSTS:
            break
        info = raw if isinstance(raw, dict) else {}
        discovered.append((str(alias), info, inherited))


def _queue_group_children(
    group: dict[str, typing.Any],
    inherited: dict[str, typing.Any],
    pending: list[tuple[dict[str, typing.Any], dict[str, typing.Any]]],
) -> None:
    """Queue child groups while retaining their inherited variables."""
    children = group.get("children", {})
    if not isinstance(children, dict):
        return
    for child in children.values():
        if isinstance(child, dict):
            pending.append((child, inherited))


def _discover_inventory_hosts(
    root: dict[str, typing.Any],
) -> list[_InventoryHost]:
    """Walk Ansible groups in the same reverse-child order as the old loop."""
    discovered: list[_InventoryHost] = []
    seen_groups: set[int] = set()
    pending: list[tuple[dict[str, typing.Any], dict[str, typing.Any]]] = [(root, {})]
    while pending and len(seen_groups) < _MAX_INVENTORY_GROUPS:
        group, inherited = pending.pop()
        marker = id(group)
        if marker in seen_groups:
            continue
        seen_groups.add(marker)
        group_vars = group.get("vars", {})
        merged = {
            **inherited,
            **(group_vars if isinstance(group_vars, dict) else {}),
        }
        _append_group_hosts(group, merged, discovered)
        _queue_group_children(group, merged, pending)
    return discovered


def _bounded_inventory_numbers(
    info: dict[str, typing.Any],
) -> dict[str, str]:
    """Convert supported positive inventory metrics into bounded labels."""
    labels: dict[str, str] = {}
    for key, upper in (
        ("cores", 1_000_000),
        ("ram_gb", 1_000_000),
        ("capacity_tb", 1_000_000_000),
        ("vram_gb", 1_000_000),
    ):
        raw_value = info.get(key)
        if raw_value is None:
            continue
        try:
            number = float(raw_value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number) and 0 < number <= upper:
            labels[key] = f"{number:g}"
    return labels


def _normalise_inventory_choice(
    value: typing.Any,
    allowed: set[str],
    default: str,
) -> str:
    """Lower-case a constrained inventory value, falling back safely."""
    choice = str(value or default).lower()
    return choice if choice in allowed else default


def _inventory_labels(info: dict[str, typing.Any]) -> dict[str, str]:
    """Build the privacy-safe capability labels for one host."""
    role = _normalise_inventory_choice(
        str(info.get("role") or "compute").strip(),
        _INVENTORY_ROLES,
        "compute",
    )
    labels = {"role": role}
    labels.update(_bounded_inventory_numbers(info))
    storage_type = str(info.get("storage_type") or "").lower()
    if storage_type in _INVENTORY_STORAGE_TYPES:
        labels["storage_type"] = storage_type
    if info.get("gpu") not in (None, "", False):
        labels["gpu"] = "present"
    return labels


def _inventory_port(info: dict[str, typing.Any]) -> int:
    """Return a valid SSH port or the Ansible default."""
    try:
        port = int(info.get("ansible_port", 22))
    except (TypeError, ValueError):
        port = 22
    return port if 1 <= port <= 65535 else 22


def _inventory_docker_host(value: typing.Any) -> bool:
    """Interpret the inventory's boolean-ish Docker capability value."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _build_host_record(
    alias: str,
    host_info: dict[str, typing.Any],
    vars_dict: dict[str, typing.Any],
    timestamp: str,
) -> _HostRecord | None:
    """Create one pseudonymous host node and its asset metadata."""
    ansible_host = host_info.get("ansible_host") or vars_dict.get("ansible_host")
    if not ansible_host:
        return None

    host_ref = persistence_reference(
        "host", f"{alias}\0{ansible_host}", namespace="ansible-inventory"
    )
    merged_info = {**vars_dict, **host_info}
    labels = _inventory_labels(merged_info)
    account_ref = persistence_reference(
        "account", merged_info.get("ansible_user", ""), namespace=host_ref
    )
    os_type = _normalise_inventory_choice(
        merged_info.get("os_type"), _INVENTORY_OS_TYPES, "unknown"
    )
    arch = _normalise_inventory_choice(
        merged_info.get("arch"), _INVENTORY_ARCHES, "unknown"
    )
    gpu_vendor = _normalise_inventory_choice(
        merged_info.get("gpu_vendor"), _INVENTORY_GPU_VENDORS, "unknown"
    )
    node = HostNode(
        id=f"host:{host_ref}",
        name=host_ref,
        hostname=host_ref,
        alias=host_ref,
        port=_inventory_port(merged_info),
        user=account_ref,
        identity_file_ref="",
        os_type=os_type,
        arch=arch,
        labels=labels,
        docker_host=_inventory_docker_host(merged_info.get("docker_host", False)),
        timestamp=timestamp,
    )
    return node, host_ref, labels, gpu_vendor


def _persist_infrastructure_node(
    engine: typing.Any,
    node: typing.Any,
    label: str,
) -> None:
    """Write one node to the compute graph and optional persistent backend."""
    engine.graph.add_node(node.id, **engine._serialize_node(node))
    if engine.backend:
        serialized = engine._serialize_node(node, label=label)
        engine._upsert_node(label, node.id, serialized)


def _add_gpu_asset(
    engine: typing.Any,
    host_id: str,
    host_ref: str,
    labels: dict[str, str],
    gpu_vendor: str,
    timestamp: str,
) -> None:
    """Persist a host GPU asset and its typed relationship when present."""
    if "gpu" not in labels or "vram_gb" not in labels:
        return
    gpu_id = f"gpu:{host_ref}"
    gpu_node = GPUAcceleratorNode(
        id=gpu_id,
        name=f"{host_ref}-gpu",
        vram_gb=float(labels["vram_gb"]),
        vendor=gpu_vendor,
        timestamp=timestamp,
    )
    _persist_infrastructure_node(engine, gpu_node, "GPUAccelerator")
    engine.graph.add_edge(host_id, gpu_id, relationship="has_accelerator")
    if engine.backend:
        # The native engine's Cypher write subset cannot MERGE a relationship
        # pattern; link_nodes dispatches through the typed engine API.
        engine.link_nodes(host_id, gpu_id, "HAS_ACCELERATOR")


def _add_storage_asset(
    engine: typing.Any,
    host_id: str,
    host_ref: str,
    labels: dict[str, str],
    timestamp: str,
) -> None:
    """Persist a storage asset and its typed relationship when present."""
    if labels.get("role") != "storage" or "capacity_tb" not in labels:
        return
    storage_id = f"storage:{host_ref}"
    storage_node = StorageArrayNode(
        id=storage_id,
        name=f"{host_ref}-storage",
        capacity_tb=float(labels["capacity_tb"]),
        storage_type=str(labels.get("storage_type", "unknown")),
        timestamp=timestamp,
    )
    _persist_infrastructure_node(engine, storage_node, "StorageArray")
    engine.graph.add_edge(host_id, storage_id, relationship="attached_storage")
    if engine.backend:
        # See _add_gpu_asset for why this uses the typed edge API.
        engine.link_nodes(host_id, storage_id, "ATTACHED_STORAGE")


def _ingest_host_record(
    engine: typing.Any,
    record: _HostRecord,
    timestamp: str,
) -> str:
    """Persist a host record and any capability assets attached to it."""
    node, host_ref, labels, gpu_vendor = record
    _persist_infrastructure_node(engine, node, "Host")
    _add_gpu_asset(engine, node.id, host_ref, labels, gpu_vendor, timestamp)
    _add_storage_asset(engine, node.id, host_ref, labels, timestamp)
    return node.id


class InfrastructureEngineMixin(_Base):
    """Software Engineering & Infrastructure capabilities for the KG engine."""

    def register_mcp_package(
        self, name: str, protocol_version: str, transport: str
    ) -> str:
        """Register an MCP server package into the KG."""
        pkg_id = f"mcp:{uuid.uuid4().hex}"
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        node = MCPServerPackageNode(
            id=pkg_id,
            name=name,
            protocol_version=protocol_version,
            transport=transport,
            timestamp=ts,
        )
        self.graph.add_node(node.id, **self._serialize_node(node))

        if self.backend:
            data = self._serialize_node(node, label="MCPServerPackage")
            self._upsert_node("MCPServerPackage", pkg_id, data)
        return pkg_id

    def record_pull_request(
        self, pr_number: int, repo_id: str, status: str = "open"
    ) -> str:
        """Record a pull request associated with a software project."""
        return _persist_action_result(
            self,
            "pr",
            "PullRequest",
            PullRequestNode,
            lambda _node_id, _timestamp: {
                "name": f"PR #{pr_number}",
                "pr_number": pr_number,
                "status": status,
            },
            backend_links=((repo_id, None, "HAS_PR"),),
        )

    def share_cross_tenant_insight(self, source_tenant: str, insight_id: str) -> str:
        """Promote an anonymized insight across tenants."""
        cross_id = f"cross:{uuid.uuid4().hex}"
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        source_tenant_ref = persistence_reference(
            "tenant", source_tenant, namespace="cross-tenant-insight"
        )

        node = CrossTenantInsightNode(
            id=cross_id,
            name="Anonymized cross-tenant insight",
            source_tenant_id=source_tenant_ref,
            anonymized=True,
            timestamp=ts,
        )
        self.graph.add_node(node.id, **self._serialize_node(node))

        if self.backend:
            data = self._serialize_node(node, label="CrossTenantInsight")
            self._upsert_node("CrossTenantInsight", cross_id, data)
            # See register_mcp_package/record_pull_request above for why this
            # is a typed link, not a comma-pattern MATCH + edge MERGE.
            self.link_nodes(insight_id, cross_id, "MAPPED_TO_EXTERNAL")
        return cross_id

    def ingest_hosts_from_inventory(
        self, inventory_path: str | None = None
    ) -> list[str]:
        """Parse a generic Ansible inventory into pseudonymous host nodes.

        Inventory location is runtime configuration. Group names, account names,
        addresses, and key paths are never persisted; the topology stores stable
        HMAC-backed references and only a small OOTB hardware/capability field set.
        """
        root = _load_inventory_root(inventory_path)
        if root is None:
            return []

        discovered = _discover_inventory_hosts(root)
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        ingested_ids: list[str] = []
        for alias, host_info, vars_dict in discovered:
            record = _build_host_record(alias, host_info, vars_dict, timestamp)
            if record is not None:
                ingested_ids.append(_ingest_host_record(self, record, timestamp))

        logger.info("Ingested %d pseudonymous inventory hosts", len(ingested_ids))
        return ingested_ids

    def generate_matchmaking_recommendations(
        self, inventory_path: str | None = None
    ) -> list[typing.Any]:
        """Evaluate platform service requirements against host capabilities using OWL SPARQL queries."""
        # 1. Ensure hosts from inventory are ingested
        self.ingest_hosts_from_inventory(inventory_path)

        # Evaluate only PlatformService nodes supplied by runtime discovery or a
        # standard provider connector. Do not inject a site-specific service profile.

        # 2. Create the OWL bridge and run the cycle to populate OWL/RDF triples.
        from ..backends.owl import create_owl_backend
        from ..core.owl_bridge import OWLBridge

        default_ontology = str(
            Path(__file__).parent.parent / "ontology_infrastructure.ttl"
        )
        # The oxigraph OWL backend was removed; use the default (owlready2).
        owl_backend = create_owl_backend(
            ontology_path=default_ontology,
        )

        bridge = OWLBridge(
            graph=self.graph,
            owl_backend=owl_backend,
            backend=self.backend,
            importance_threshold=0.0,
        )

        # Execute the promotion cycle to populate ABox facts inside OWL backend
        bridge.run_cycle()

        # 4. Query Host capabilities via SPARQL. The engine's live projection
        # types every promoted node as au:CamelCase(node_type) — a "host" node
        # is au:Host, never au:BladeServer (that class belongs to an unrelated
        # vendor ontology under a completely different namespace/prefix in
        # ontology_infrastructure.ttl, not this engine-native projection).
        #
        # Edge predicates are NOT case-transformed on the engine-native path
        # (D-CYP-1): ``bridge.query_sparql`` tries the LIVE engine projection
        # FIRST (``OWLBridge.query_sparql`` strategy 1,
        # ``self.graph.sparql(...)``) and only falls back to the owlready2
        # backend's own SPARQL (strategy 2) when strategy 1 returns ZERO rows
        # -- and a `?host rdf:type au:Host` row always exists once hosts are
        # ingested, so strategy 1 always wins here. The engine's edge
        # projection (`eg-rdf`'s ``match_triple_pattern`` -> ``proj.pred_iri``)
        # emits an edge predicate from the LPG edge's raw ``relationship``
        # property value VERBATIM, with no case transform -- and that value is
        # the upper-snake-case ``link_nodes`` writes ("HAS_ACCELERATOR",
        # "ATTACHED_STORAGE" — see below). Only the owlready2 backend's
        # ``_promote_stable_edges()`` (strategy 2, effectively unreachable
        # here) resolves it through ``_EDGE_TYPE_TO_OWL_PROP`` into camelCase
        # ("hasAccelerator", "attachedStorage"). Querying camelCase-only
        # predicates silently starved every host of a GPU/storage match on
        # strategy 1 -- ``has_gpu`` was False for every host, so every
        # GPU-requiring service scored an identical tie and the "best" host
        # was whichever the sort's stability happened to keep first, NOT the
        # one actually carrying the accelerator. Matching BOTH spellings via
        # a property-path alternation keeps this correct under either
        # strategy.
        host_query = """
        PREFIX au: <http://agent-utilities.dev/ontology#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        SELECT ?host ?gpu ?storage WHERE {
            ?host rdf:type au:Host .
            OPTIONAL { ?host au:HAS_ACCELERATOR|au:hasAccelerator ?gpu . }
            OPTIONAL { ?host au:ATTACHED_STORAGE|au:attachedStorage ?storage . }
        }
        """

        # 5. Query Services via SPARQL
        service_query = """
        PREFIX au: <http://agent-utilities.dev/ontology#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        SELECT ?service WHERE {
            ?service rdf:type au:PlatformService .
        }
        """

        hosts_rdf = bridge.query_sparql(host_query)
        services_rdf = bridge.query_sparql(service_query)

        def _local_id(uri: str) -> str:
            """The bare node id from a SPARQL-bound URI.

            The engine-native SPARQL surface (and rdflib's SELECT bindings)
            return full URI terms wrapped in angle brackets
            (``<http://.../ontology#node:id>``), not bare strings — splitting
            on ``#`` alone left a trailing ``>`` glued onto the id, so it never
            matched a real node id and every row was silently dropped below.
            """
            return str(uri).strip().strip("<>").split("#")[-1]

        def _labels_of(node_data: dict) -> dict:
            """The node's "labels" dict property, decoded.

            dict-valued properties are JSON-encoded for storage
            (IntelligenceGraphEngine._serialize_node) and come back as a JSON
            string, not a dict.
            """
            raw = node_data.get("labels", {}) or {}
            if isinstance(raw, str):
                try:
                    decoded = json.loads(raw)
                except (TypeError, ValueError):
                    return {}
                return decoded if isinstance(decoded, dict) else {}
            return raw if isinstance(raw, dict) else {}

        # Parse SPARQL results and extract attributes from LPG
        hosts_by_id = {}
        for row in hosts_rdf:
            host_id = _local_id(row.get("host", ""))
            if not host_id or host_id not in self.graph:
                continue

            node_data = self.graph.nodes[host_id]
            labels = _labels_of(node_data)

            gpu_uri = row.get("gpu", "")
            gpu_id = _local_id(gpu_uri) if gpu_uri else None

            storage_uri = row.get("storage", "")
            storage_id = _local_id(storage_uri) if storage_uri else None

            hosts_by_id[host_id] = {
                "id": host_id,
                "name": node_data.get("name", host_id),
                "hostname": node_data.get("hostname", ""),
                "labels": labels,
                "has_gpu": gpu_id is not None,
                "gpu_details": self.graph.nodes.get(gpu_id, {}) if gpu_id else {},
                "has_storage": storage_id is not None,
                "storage_details": self.graph.nodes.get(storage_id, {})
                if storage_id
                else {},
            }

        recommendations = []

        for row in services_rdf:
            svc_id = _local_id(row.get("service", ""))
            if not svc_id or svc_id not in self.graph:
                continue

            svc_data = self.graph.nodes[svc_id]
            svc_labels = _labels_of(svc_data)
            svc_desc = svc_data.get("description", "")
            svc_name = svc_data.get("name", svc_id)

            # Evaluate suitability score for each host
            candidates = []
            for h_id, host in hosts_by_id.items():
                if not isinstance(host, dict):
                    continue
                h_labels = host.get("labels", {})
                if not isinstance(h_labels, dict):
                    h_labels = {}
                gpu_details = host.get("gpu_details", {})
                if not isinstance(gpu_details, dict):
                    gpu_details = {}
                storage_details = host.get("storage_details", {})
                if not isinstance(storage_details, dict):
                    storage_details = {}

                score = 50.0  # Base score
                reasons = []

                # Check GPU Match
                if svc_labels.get("requires_gpu") == "true":
                    if host.get("has_gpu"):
                        score += 40.0
                        gpu_name = gpu_details.get("name", "GPU")
                        reasons.append(f"Satisfies GPU requirement via {gpu_name}")
                    else:
                        score -= 40.0
                        reasons.append("Lacks required GPU accelerator")
                else:
                    if host.get("has_gpu"):
                        score -= 10.0  # Avoid placing non-GPU service on GPU host to save resources
                        reasons.append(
                            "Saves high-value GPU host for accelerator tasks"
                        )

                # Check Storage Match
                if svc_labels.get("requires_storage") == "true":
                    if host.get("has_storage"):
                        score += 30.0
                        cap = storage_details.get("capacity_tb", 0)
                        reasons.append(f"Provides attached storage array ({cap} TB)")
                    else:
                        score -= 20.0
                        reasons.append("Lacks required attached storage array")

                # Check Compute Match
                if svc_labels.get("requires_high_compute") == "true":
                    if h_labels.get("role") == "compute_high":
                        score += 40.0
                        cores = h_labels.get("cores", "unknown")
                        reasons.append(
                            f"Matches high-compute profile ({cores} CPU Cores)"
                        )
                    elif h_labels.get("role") == "compute":
                        score += 15.0
                        reasons.append("Standard compute capacity available")
                    else:
                        score -= 10.0
                        reasons.append("Sub-optimal core density for high compute")

                candidates.append(
                    {
                        "host_id": h_id,
                        "host_name": host.get("name", h_id),
                        "score": max(0.0, min(100.0, score)),
                        "reasons": reasons,
                    }
                )

            # Sort candidate hosts by suitability score descending
            candidates.sort(key=lambda x: x["score"], reverse=True)
            best_match = candidates[0]

            recommendations.append(
                {
                    "service_id": svc_id,
                    "service_name": svc_name,
                    "description": svc_desc,
                    # The node id (e.g. "host:pref_...", matching every other
                    # id this function and ingest_hosts_from_inventory() deal
                    # in), not the display name — a caller needs the id to
                    # look the host back up in the graph.
                    "best_host": best_match["host_id"],
                    "match_score": best_match["score"],
                    "rationale": best_match["reasons"],
                    "all_candidates": candidates,
                }
            )

        return recommendations
