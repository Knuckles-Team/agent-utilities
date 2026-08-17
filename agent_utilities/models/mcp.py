import time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .tool_score import normalize_legacy_relevance_score

SpecialistTier = Literal["light", "medium", "heavy", "reasoning"]


class MCPConfigModel(BaseModel):
    mcpServers: dict[str, Any] = Field(default_factory=dict)


class MCPServerEntryModel(BaseModel):
    """One ``mcp_config.json`` ``mcpServers.<name>`` entry, typed for CRUD.

    Mirrors, field-for-field, the shape :meth:`agent_utilities.mcp.multiplexer.
    MCPMultiplexer._open_one_session` actually reads off the raw catalog dict
    (``command``/``url``/``transport``/``args``/``env``/``headers``/``timeout``/
    ``allowed_private_hosts``/``disabled``) -- this is the ONE typed model for
    that shape (none existed before; every call site re-derived it ad hoc from
    an untyped dict). Used for schema-derived add/edit forms (``.model_json_schema()``,
    matching the ``ChatModelConfig``/``EmbeddingModelConfig`` pattern) and to
    validate a submitted entry BEFORE it is written to the catalog file, so an
    invalid shape is rejected at CRUD time rather than silently failing every
    later spawn attempt.
    """

    model_config = ConfigDict(validate_assignment=True)

    command: str | None = Field(
        default=None,
        description="Executable to spawn over stdio. Mutually exclusive with 'url'.",
    )
    args: list[str] = Field(
        default_factory=list, description="Arguments passed to 'command'"
    )
    env: dict[str, str] = Field(
        default_factory=dict, description="Environment variables for the stdio child"
    )
    url: str | None = Field(
        default=None,
        description="Remote MCP endpoint URL. Mutually exclusive with 'command'.",
    )
    transport: Literal["", "streamable-http", "sse"] = Field(
        default="",
        description="Explicit remote transport; inferred from 'url' when blank",
    )
    headers: dict[str, str] = Field(
        default_factory=dict, description="Extra HTTP headers for a remote child"
    )
    disabled: bool = Field(
        default=False, description="Excluded from the mountable catalog when true"
    )
    timeout: float = Field(
        default=300.0, ge=0.001, le=3_600.0, description="Connect timeout, seconds"
    )
    allowed_private_hosts: list[str] = Field(
        default_factory=list,
        description="Extra plain-HTTP hostnames trusted for THIS server only",
    )

    @model_validator(mode="after")
    def _exactly_one_transport(self) -> "MCPServerEntryModel":
        """Match the multiplexer's own invariant: exactly one of command/url."""
        if bool(self.command) == bool(self.url):
            raise ValueError(
                "Exactly one of 'command' (stdio) or 'url' (remote) is required"
            )
        if self.transport and not self.url:
            raise ValueError("'transport' requires 'url'")
        return self


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

        Delegates to :func:`agent_utilities.models.tool_score.normalize_legacy_relevance_score`,
        the single source of truth shared with
        :class:`agent_utilities.models.knowledge_graph.ToolNode` (D-CDX-53/54)
        so the two models can never apply a different boundary.
        """
        return normalize_legacy_relevance_score(value)


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
