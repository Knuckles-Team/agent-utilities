from __future__ import annotations

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

logger = logging.getLogger(__name__)

_MAX_INVENTORY_BYTES = 16 * 1024 * 1024
_MAX_INVENTORY_GROUPS = 10_000
_MAX_INVENTORY_HOSTS = 100_000


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
        self.graph.add_node(node.id, **node.model_dump())

        if self.backend:
            data = self._serialize_node(node, label="MCPServerPackage")
            self._upsert_node("MCPServerPackage", pkg_id, data)
        return pkg_id

    def record_pull_request(
        self, pr_number: int, repo_id: str, status: str = "open"
    ) -> str:
        """Record a pull request associated with a software project."""
        pr_id = f"pr:{uuid.uuid4().hex}"
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        node = PullRequestNode(
            id=pr_id,
            name=f"PR #{pr_number}",
            pr_number=pr_number,
            status=status,
            timestamp=ts,
        )
        self.graph.add_node(node.id, **node.model_dump())

        if self.backend:
            data = self._serialize_node(node, label="PullRequest")
            self._upsert_node("PullRequest", pr_id, data)
            self.backend.execute(
                "MATCH (r:SoftwareProject {id: $rid}), (p:PullRequest {id: $pid}) "
                "MERGE (r)-[:HAS_PR]->(p)",
                {"rid": repo_id, "pid": pr_id},
            )
        return pr_id

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
        self.graph.add_node(node.id, **node.model_dump())

        if self.backend:
            data = self._serialize_node(node, label="CrossTenantInsight")
            self._upsert_node("CrossTenantInsight", cross_id, data)
            self.backend.execute(
                "MATCH (i {id: $iid}), (c:CrossTenantInsight {id: $cid}) "
                "MERGE (i)-[:MAPPED_TO_EXTERNAL]->(c)",
                {"iid": insight_id, "cid": cross_id},
            )
        return cross_id

    def ingest_hosts_from_inventory(
        self, inventory_path: str | None = None
    ) -> list[str]:
        """Parse a generic Ansible inventory into pseudonymous host nodes.

        Inventory location is runtime configuration. Group names, account names,
        addresses, and key paths are never persisted; the topology stores stable
        HMAC-backed references and only a small OOTB hardware/capability field set.
        """
        if inventory_path is None:
            inventory_path = str(setting("INFRA_INVENTORY_PATH", "") or "").strip()
        if not inventory_path:
            logger.warning("Infrastructure inventory path is not configured")
            return []

        path = Path(inventory_path).expanduser()
        if not path.exists():
            logger.warning("Configured infrastructure inventory is unavailable")
            return []
        try:
            if not path.is_file() or path.stat().st_size > _MAX_INVENTORY_BYTES:
                logger.warning(
                    "Configured infrastructure inventory is not a bounded file"
                )
                return []
            with open(path) as f:
                data = yaml.safe_load(f)
        except (OSError, yaml.YAMLError) as exc:
            logger.warning(
                "Configured infrastructure inventory could not be read (%s)",
                type(exc).__name__,
            )
            return []

        root = data.get("all", data) if isinstance(data, dict) else {}
        if not isinstance(root, dict):
            logger.error("Infrastructure inventory root must be a mapping")
            return []

        discovered: list[tuple[str, dict[str, typing.Any], dict[str, typing.Any]]] = []
        seen_groups: set[int] = set()
        pending: list[tuple[dict[str, typing.Any], dict[str, typing.Any]]] = [
            (root, {})
        ]
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
            hosts = group.get("hosts", {})
            if isinstance(hosts, dict):
                for alias, raw in hosts.items():
                    if len(discovered) >= _MAX_INVENTORY_HOSTS:
                        break
                    info = raw if isinstance(raw, dict) else {}
                    discovered.append((str(alias), info, merged))
            children = group.get("children", {})
            if isinstance(children, dict):
                for child in children.values():
                    if isinstance(child, dict):
                        pending.append((child, merged))

        ingested_ids: list[str] = []
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        for alias, host_info, vars_dict in discovered:
            ansible_host = host_info.get("ansible_host") or vars_dict.get(
                "ansible_host"
            )
            if not ansible_host:
                continue

            namespace = "ansible-inventory"
            host_ref = persistence_reference(
                "host", f"{alias}\0{ansible_host}", namespace=namespace
            )
            host_id = f"host:{host_ref}"
            merged_info = {**vars_dict, **host_info}
            role = str(merged_info.get("role") or "compute").strip().lower()
            if role not in {"compute", "compute_high", "gpu", "storage"}:
                role = "compute"
            labels = {"role": role}
            for key, upper in (
                ("cores", 1_000_000),
                ("ram_gb", 1_000_000),
                ("capacity_tb", 1_000_000_000),
                ("vram_gb", 1_000_000),
            ):
                raw_value = merged_info.get(key)
                if raw_value is None:
                    continue
                try:
                    number = float(raw_value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(number) and 0 < number <= upper:
                    labels[key] = f"{number:g}"
            storage_type = str(merged_info.get("storage_type") or "").lower()
            if storage_type in {"hdd", "hybrid", "nvme", "object", "sas", "ssd"}:
                labels["storage_type"] = storage_type
            if merged_info.get("gpu") not in (None, "", False):
                labels["gpu"] = "present"
            account_ref = persistence_reference(
                "account", merged_info.get("ansible_user", ""), namespace=host_ref
            )
            try:
                port = int(merged_info.get("ansible_port", 22))
            except (TypeError, ValueError):
                port = 22
            if not 1 <= port <= 65535:
                port = 22
            os_type = str(merged_info.get("os_type") or "unknown").lower()
            if os_type not in {"darwin", "freebsd", "linux", "unix", "windows"}:
                os_type = "unknown"
            arch = str(merged_info.get("arch") or "unknown").lower()
            if arch not in {
                "aarch64",
                "amd64",
                "arm64",
                "ppc64le",
                "riscv64",
                "x86_64",
            }:
                arch = "unknown"
            gpu_vendor = str(merged_info.get("gpu_vendor") or "unknown").lower()
            if gpu_vendor not in {"amd", "apple", "intel", "nvidia"}:
                gpu_vendor = "unknown"
            docker_value = merged_info.get("docker_host", False)
            docker_host = (
                docker_value
                if isinstance(docker_value, bool)
                else str(docker_value).strip().lower() in {"1", "true", "yes", "on"}
            )

            node = HostNode(
                id=host_id,
                name=host_ref,
                hostname=host_ref,
                alias=host_ref,
                port=port,
                user=account_ref,
                identity_file_ref="",
                os_type=os_type,
                arch=arch,
                labels=labels,
                docker_host=docker_host,
                timestamp=ts,
            )

            # Save Host to the graph compute engine
            self.graph.add_node(node.id, **node.model_dump())

            # If backend is persistent, dual write
            if self.backend:
                serialized = self._serialize_node(node, label="Host")
                self._upsert_node("Host", host_id, serialized)

            ingested_ids.append(host_id)

            # Create sub-assets (GPU or Storage Array) and link them!
            if "gpu" in labels and "vram_gb" in labels:
                gpu_id = f"gpu:{host_ref}"
                gpu_node = GPUAcceleratorNode(
                    id=gpu_id,
                    name=f"{host_ref}-gpu",
                    vram_gb=float(labels["vram_gb"]),
                    vendor=gpu_vendor,
                    timestamp=ts,
                )
                self.graph.add_node(gpu_node.id, **gpu_node.model_dump())
                if self.backend:
                    s_gpu = self._serialize_node(gpu_node, label="GPUAccelerator")
                    self._upsert_node("GPUAccelerator", gpu_id, s_gpu)

                # Add edge has_accelerator
                self.graph.add_edge(host_id, gpu_id, relationship="has_accelerator")
                if self.backend:
                    self.backend.execute(
                        "MATCH (h:Host {id: $hid}), (g:GPUAccelerator {id: $gid}) "
                        "MERGE (h)-[:HAS_ACCELERATOR]->(g)",
                        {"hid": host_id, "gid": gpu_id},
                    )

            if labels["role"] == "storage" and "capacity_tb" in labels:
                storage_id = f"storage:{host_ref}"
                storage_node = StorageArrayNode(
                    id=storage_id,
                    name=f"{host_ref}-storage",
                    capacity_tb=float(labels["capacity_tb"]),
                    storage_type=str(labels.get("storage_type", "unknown")),
                    timestamp=ts,
                )
                self.graph.add_node(storage_node.id, **storage_node.model_dump())
                if self.backend:
                    s_storage = self._serialize_node(storage_node, label="StorageArray")
                    self._upsert_node("StorageArray", storage_id, s_storage)

                # Add edge attached_storage
                self.graph.add_edge(
                    host_id, storage_id, relationship="attached_storage"
                )
                if self.backend:
                    self.backend.execute(
                        "MATCH (h:Host {id: $hid}), (s:StorageArray {id: $sid}) "
                        "MERGE (h)-[:ATTACHED_STORAGE]->(s)",
                        {"hid": host_id, "sid": storage_id},
                    )

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

        # 4. Query Host capabilities via SPARQL
        host_query = """
        PREFIX au: <http://agent-utilities.dev/ontology#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        SELECT ?host ?gpu ?storage WHERE {
            ?host rdf:type au:BladeServer .
            OPTIONAL { ?host au:hasAccelerator ?gpu . }
            OPTIONAL { ?host au:attachedStorage ?storage . }
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

        # Parse SPARQL results and extract attributes from LPG
        hosts_by_id = {}
        for row in hosts_rdf:
            host_uri = row.get("host", "")
            host_id = host_uri.split("#")[-1] if "#" in host_uri else host_uri
            if not host_id or host_id not in self.graph:
                continue

            node_data = self.graph.nodes[host_id]
            labels = node_data.get("labels", {}) or {}

            gpu_uri = row.get("gpu", "")
            gpu_id = gpu_uri.split("#")[-1] if gpu_uri else None

            storage_uri = row.get("storage", "")
            storage_id = storage_uri.split("#")[-1] if storage_uri else None

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
            svc_uri = row.get("service", "")
            svc_id = svc_uri.split("#")[-1] if "#" in svc_uri else svc_uri
            if not svc_id or svc_id not in self.graph:
                continue

            svc_data = self.graph.nodes[svc_id]
            svc_labels = svc_data.get("labels", {}) or {}
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
                    "best_host": best_match["host_name"],
                    "match_score": best_match["score"],
                    "rationale": best_match["reasons"],
                    "all_candidates": candidates,
                }
            )

        return recommendations
