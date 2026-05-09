from pydantic import Field

from ...models.knowledge_graph import RegistryNode, RegistryNodeType


class FrontendPackageNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.FRONTEND_PACKAGE
    framework: str = "react"
    version: str = "latest"


class KernelPackageNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.KERNEL_PACKAGE
    language: str = "python"
    is_core: bool = True


class SkillPackageNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.SKILL_PACKAGE
    skill_domain: str = "general"
    supported_models: list[str] = Field(default_factory=list)


class MCPServerPackageNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.MCP_SERVER_PACKAGE
    protocol_version: str = "1.0"
    transport: str = "stdio"


class CommunicationMCPNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.COMMUNICATION_MCP
    channels_supported: list[str] = Field(default_factory=list)


class DataScienceMCPNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.DATA_SCIENCE_MCP
    supported_formats: list[str] = Field(default_factory=list)


class DevOpsMCPNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.DEV_OPS_MCP
    cloud_provider: str = "aws"


class InfrastructureMCPNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.INFRASTRUCTURE_MCP
    orchestrator: str = "kubernetes"


class ProductivityMCPNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.PRODUCTIVITY_MCP
    suite: str = "office365"


class MediaMCPNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.MEDIA_MCP
    media_types: list[str] = Field(default_factory=list)


class PullRequestNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.PULL_REQUEST
    pr_number: int = 0
    status: str = "open"
    base_branch: str = "main"


class MergeRequestNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.MERGE_REQUEST
    mr_number: int = 0
    status: str = "open"
    target_branch: str = "main"


class CrossTenantInsightNode(RegistryNode):
    type: RegistryNodeType = RegistryNodeType.CROSS_TENANT_INSIGHT
    source_tenant_id: str = ""
    anonymized: bool = True
    insight_type: str = "pattern"
