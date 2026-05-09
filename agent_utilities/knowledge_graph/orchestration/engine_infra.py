from __future__ import annotations

import logging
import time
import typing
import uuid

if typing.TYPE_CHECKING:
    from .._engine_protocol import _EngineProtocol

    _Base = _EngineProtocol
else:
    _Base = object

from ...models.domains.infrastructure import (
    CrossTenantInsightNode,
    MCPServerPackageNode,
    PullRequestNode,
)

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
