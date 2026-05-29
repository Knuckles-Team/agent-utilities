from __future__ import annotations

import logging
import os
import time
import typing
import uuid
from pathlib import Path

import yaml

if typing.TYPE_CHECKING:
    from .._engine_protocol import _EngineProtocol

    _Base = _EngineProtocol
else:
    _Base = object

from ...models.domains.infrastructure import (
    CrossTenantInsightNode,
    GPUAcceleratorNode,
    MCPServerPackageNode,
    PlatformServiceNode,
    PullRequestNode,
    StorageArrayNode,
)
from ...models.knowledge_graph import HostNode

logger = logging.getLogger(__name__)


class InfrastructureEngineMixin(_Base):
    """Software Engineering & Infrastructure capabilities for the KG engine."""

    def register_mcp_package(
        self, name: str, protocol_version: str, transport: str
    ) -> str:
        """Register an MCP server package into the KG."""
        pkg_id = f"mcp:{uuid.uuid4().hex[:8]}"
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
        pr_id = f"pr:{uuid.uuid4().hex[:8]}"
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
        cross_id = f"cross:{uuid.uuid4().hex[:8]}"
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        node = CrossTenantInsightNode(
            id=cross_id,
            name=f"Insight from {source_tenant}",
            source_tenant_id=source_tenant,
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
        """Parse hosts from Ansible-style inventory.yaml and ingest them into the KG."""
        if inventory_path is None:
            inventory_path = os.path.expanduser(
                "~/.config/agent-utilities/inventory.yaml"
            )

        path = Path(inventory_path)
        if not path.exists():
            logger.warning(f"Inventory file not found: {inventory_path}")
            return []

        with open(path) as f:
            data = yaml.safe_load(f)

        try:
            homelab = data["all"]["children"]["homelab"]
            hosts_dict = homelab.get("hosts", {})
            vars_dict = homelab.get("vars", {})
        except KeyError as e:
            logger.error(f"Failed to parse inventory.yaml: missing key {e}")
            return []

        ansible_user = vars_dict.get("ansible_user", "genius")
        key_file = vars_dict.get("ansible_ssh_private_key_file", "~/.ssh/id_rsa")

        ingested_ids = []
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        for alias, host_info in hosts_dict.items():
            ansible_host = host_info.get("ansible_host")
            if not ansible_host:
                continue

            host_id = f"host:{alias}"

            # Map hardware details based on host names/classes
            labels = {"role": "compute", "environment": "homelab"}
            if alias == "r510":
                labels["role"] = "storage"
                labels["capacity_tb"] = "24.0"
                labels["storage_type"] = "HDD-SAS"
            elif alias == "r710" or alias == "rw710":
                labels["role"] = "compute"
                labels["cores"] = "8"
                labels["ram_gb"] = "32"
            elif alias == "r820":
                labels["role"] = "compute_high"
                labels["cores"] = "32"
                labels["ram_gb"] = "128"
            elif alias == "gr1080":
                labels["role"] = "gpu"
                labels["gpu"] = "GTX1080"
                labels["vram_gb"] = "8"
            elif alias == "gb10":
                labels["role"] = "gpu"
                labels["gpu"] = "RTX4080"
                labels["vram_gb"] = "16"

            node = HostNode(
                id=host_id,
                name=alias,
                hostname=ansible_host,
                alias=alias,
                port=22,
                user=ansible_user,
                identity_file_ref=key_file,
                os_type="linux",
                arch="x86_64",
                labels=labels,
                docker_host=True,
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
            if "gpu" in labels:
                gpu_id = f"gpu:{alias}"
                gpu_node = GPUAcceleratorNode(
                    id=gpu_id,
                    name=f"{alias}-gpu",
                    vram_gb=float(labels["vram_gb"]),
                    vendor="Nvidia",
                    timestamp=ts,
                )
                self.graph.add_node(gpu_node.id, **gpu_node.model_dump())
                if self.backend:
                    s_gpu = self._serialize_node(gpu_node, label="GPUAccelerator")
                    self._upsert_node("GPUAccelerator", gpu_id, s_gpu)

                # Add edge has_accelerator
                self.graph.add_edge(host_id, gpu_id, type="has_accelerator")
                if self.backend:
                    self.backend.execute(
                        "MATCH (h:Host {id: $hid}), (g:GPUAccelerator {id: $gid}) "
                        "MERGE (h)-[:HAS_ACCELERATOR]->(g)",
                        {"hid": host_id, "gid": gpu_id},
                    )

            if labels["role"] == "storage":
                storage_id = f"storage:{alias}"
                storage_node = StorageArrayNode(
                    id=storage_id,
                    name=f"{alias}-storage",
                    capacity_tb=float(labels["capacity_tb"]),
                    storage_type="SAS",
                    timestamp=ts,
                )
                self.graph.add_node(storage_node.id, **storage_node.model_dump())
                if self.backend:
                    s_storage = self._serialize_node(storage_node, label="StorageArray")
                    self._upsert_node("StorageArray", storage_id, s_storage)

                # Add edge attached_storage
                self.graph.add_edge(host_id, storage_id, type="attached_storage")
                if self.backend:
                    self.backend.execute(
                        "MATCH (h:Host {id: $hid}), (s:StorageArray {id: $sid}) "
                        "MERGE (h)-[:ATTACHED_STORAGE]->(s)",
                        {"hid": host_id, "sid": storage_id},
                    )

        logger.info(f"Ingested {len(ingested_ids)} hosts from {inventory_path}.")
        return ingested_ids

    def generate_matchmaking_recommendations(
        self, inventory_path: str | None = None
    ) -> list[typing.Any]:
        """Evaluate platform service requirements against host capabilities using OWL SPARQL queries."""
        # 1. Ensure hosts from inventory are ingested
        self.ingest_hosts_from_inventory(inventory_path)

        # 2. Add some representative platform services to ensure we have a robust fleet to match
        # (if they don't already exist in the graph)
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        services_to_ensure = [
            (
                "ollama",
                "ollama-service",
                {
                    "requires_gpu": "true",
                    "description": "Local LLM inference server requiring GPU acceleration",
                },
            ),
            (
                "postgres-replica",
                "postgres",
                {
                    "requires_storage": "true",
                    "required_capacity_tb": "5.0",
                    "description": "Primary transactional database requiring large attached storage",
                },
            ),
            (
                "cognitive-reasoning-worker",
                "reasoner",
                {
                    "requires_high_compute": "true",
                    "description": "Heavy planning worker requiring high CPU core count",
                },
            ),
            (
                "gateway-proxy",
                "nginx",
                {
                    "description": "Lightweight ingress controller with basic CPU/RAM needs"
                },
            ),
        ]

        for s_id, s_name, labels in services_to_ensure:
            full_id = f"service:{s_id}"
            if full_id not in self.graph:
                node = PlatformServiceNode(
                    id=full_id,
                    name=s_name,
                    endpoint=f"http://{s_name}.local",
                    labels=labels,
                    description=labels.get("description", ""),
                    timestamp=ts,
                )
                self.graph.add_node(node.id, **node.model_dump())
                if self.backend:
                    serialized = self._serialize_node(node, label="PlatformService")
                    self._upsert_node("PlatformService", full_id, serialized)

        # 3. Create the OWL bridge and run the cycle to populate OWL/RDF triples!
        from ..backends.owl import create_owl_backend
        from ..core.owl_bridge import OWLBridge

        default_ontology = str(
            Path(__file__).parent.parent / "ontology_infrastructure.ttl"
        )
        owl_backend = create_owl_backend(
            backend_type="oxigraph",
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
