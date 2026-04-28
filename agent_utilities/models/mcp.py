import time
from typing import Any, Literal

from pydantic import BaseModel, Field

SpecialistTier = Literal["light", "medium", "heavy", "reasoning"]


class MCPConfigModel(BaseModel):
    mcpServers: dict[str, Any] = Field(default_factory=dict)


class MCPAgent(BaseModel):
    name: str = Field(description="Unique agent identifier / tag")
    agent_type: str = Field(default="prompt", description="Type: prompt, mcp, a2a")
    prompt_file: str | None = Field(
        default=None, description="Markdown prompt file path"
    )
    json_blueprint: dict[str, Any] | None = Field(
        default=None, description="JSON blueprint for structured prompting"
    )
    endpoint_url: str | None = Field(default=None, description="Connection URL / cmd")
    description: str = Field(default="", description="Specialized agent description")
    system_prompt: str = Field(default="", description="Synthesized system prompt")
    tools: list[str] = Field(default_factory=list, description="Tool names")
    mcp_server: str | None = Field(default=None, description="Source MCP server name")
    capabilities: list[str] = Field(
        default_factory=list, description="Skills/Capabilities"
    )
    mcp_tools: str | None = Field(default=None, description="MCP tool/tag patterns")
    extra_config: dict[str, Any] = Field(default_factory=dict, description="Metadata")
    is_custom: bool = Field(default=False, description="True if manually edited")
    tool_count: int = Field(default=0, description="Number of tools")
    avg_relevance_score: int = Field(default=0, description="Mean score (0-100)")
    default_tier: SpecialistTier = Field(
        default="medium",
        description=(
            "Routing tier hint passed to the model-registry specialist "
            "spawner. Use 'light' for cheap/fast researchers, 'heavy' for "
            "planners/synthesizers, 'reasoning' for deep-thinking nodes."
        ),
    )
    required_tags: list[str] = Field(
        default_factory=list,
        description=(
            "Capability tags every candidate model must carry (AND "
            "semantics) before the spawner considers it."
        ),
    )

    @property
    def tag(self) -> str:
        """Alias for name, used in routing and legacy test cases."""
        return self.name


class MCPToolInfo(BaseModel):
    name: str = Field(description="Full tool name")
    description: str = Field(description="Tool description")
    tag: str | None = Field(
        default=None, description="Primary tool tag for partitioning"
    )
    mcp_server: str = Field(description="Source MCP server")
    all_tags: list[str] = Field(
        default_factory=list, description="All tags associated with the tool"
    )
    relevance_score: int = Field(
        default=0, description="Deterministic quality score (0-100)"
    )
    requires_approval: bool = Field(
        default=False,
        description="Whether this tool requires human-in-the-loop approval",
    )


class MCPAgentRegistryModel(BaseModel):
    agents: list[MCPAgent] = Field(default_factory=list)
    tools: list[MCPToolInfo] = Field(default_factory=list)


class DiscoveredSpecialist(BaseModel):
    tag: str = Field(description="Routing key used by the dispatcher")
    name: str = Field(description="Human-readable display name")
    description: str = Field(default="", description="Specialist summary")
    source: str = Field(description="Origin: 'prompt', 'mcp', or 'a2a'")
    mcp_server: str = Field(default="", description="Source MCP server (MCP only)")
    tools: list[str] = Field(default_factory=list, description="Known tool names")
    url: str = Field(default="", description="Agent endpoint URL (A2A/MCP only)")
    capabilities: list[str] = Field(
        default_factory=list, description="Rich capabilities"
    )
    extra_config: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class MCPServerHealth(BaseModel):
    server_name: str = ""
    failures: int = 0
    last_failure: float = 0.0
    state: str = "closed"
    cooldown_seconds: float = 60.0
    max_failures: int = 3

    def record_failure(self) -> None:
        self.failures += 1
        self.last_failure = time.time()
        if self.failures >= self.max_failures:
            self.state = "open"

    def record_success(self) -> None:
        self.failures = 0
        self.state = "closed"

    def is_available(self) -> bool:
        if self.state == "closed":
            return True
        if self.state == "open":
            if time.time() - self.last_failure > self.cooldown_seconds:
                self.state = "half-open"
                return True
            return False
        return True
