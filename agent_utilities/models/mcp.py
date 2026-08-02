import time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SpecialistTier = Literal["light", "medium", "heavy", "reasoning"]


class MCPConfigModel(BaseModel):
    mcpServers: dict[str, Any] = Field(default_factory=dict)


class MCPAgent(BaseModel):
    name: str = Field(description="Unique agent identifier / tag")
    agent_type: Literal["specialist", "a2a"] = Field(
        default="specialist",
        description=(
            "Agent classification. 'specialist' for all local agents "
            "(regardless of origin: prompts, MCP partitioning, or skills). "
            "'a2a' for remote Agent-to-Agent peers."
        ),
    )
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
        """Routing tag for this specialist."""
        return self.name


class MCPToolInfo(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

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
        default=0,
        ge=0,
        le=100,
        strict=True,
        description="Deterministic quality score (0-100)",
    )
    requires_approval: bool = Field(
        default=False,
        description="Whether this tool requires human-in-the-loop approval",
    )

    @field_validator("relevance_score", mode="before")
    @classmethod
    def _normalize_legacy_relevance_score(cls, value: Any) -> Any:
        """Read legacy normalized graph scores without weakening the schema.

        Early Tool writers persisted floating-point scores in ``[0, 1]`` while
        every router and scorer uses integer points in ``[0, 100]``.  Floats in
        that legacy range are converted at this boundary; new writers must
        persist canonical integer points.  Other fractional or out-of-range
        values remain invalid so corrupt rows can be quarantined by callers.
        """
        if isinstance(value, float) and 0.0 <= value <= 1.0:
            return round(value * 100)
        return value


class MCPAgentRegistryModel(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    agents: list[MCPAgent] = Field(default_factory=list)
    # Registry snapshots are immutable at the collection boundary. Pydantic
    # still accepts list-shaped construction/assignment input and validates
    # every element, while tuple storage prevents append/setitem from bypassing
    # model validation after the registry has entered the process cache.
    tools: tuple[MCPToolInfo, ...] = Field(default_factory=tuple)


class DiscoveredSpecialist(BaseModel):
    tag: str = Field(description="Routing key used by the dispatcher")
    name: str = Field(description="Human-readable display name")
    description: str = Field(default="", description="Specialist summary")
    source: Literal["specialist", "a2a"] = Field(
        description=(
            "Origin: 'specialist' (unified local agent) or 'a2a' (remote peer)."
        ),
    )
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
