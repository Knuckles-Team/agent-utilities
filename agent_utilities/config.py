import os
from typing import Optional, Dict, List, Any
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from .base_utilities import (
    GET_DEFAULT_SSL_VERIFY,
    to_boolean,
    to_list,
    to_dict,
)

try:
    import logfire

    HAS_LOGFIRE = True
except ImportError:
    HAS_LOGFIRE = False

meta = {"name": "Agent", "description": "AI Agent"}


def get_env_file() -> Optional[str]:
    """Identify the caller package's .env file location."""
    from .base_utilities import retrieve_package_name
    from pathlib import Path

    pkg = retrieve_package_name()
    if pkg and pkg != "agent_utilities":

        candidates = [
            Path.cwd() / ".env",
            Path.cwd() / pkg / ".env",
            Path.cwd() / "agents" / pkg.replace("_", "-") / ".env",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
    return ".env"


class AgentConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=get_env_file(),
        env_ignore_empty=True,
        extra="ignore",
    )

    default_agent_name: str = Field(default=meta["name"], alias="DEFAULT_AGENT_NAME")
    agent_description: str = Field(
        default=meta["description"], alias="AGENT_DESCRIPTION"
    )
    agent_system_prompt: Optional[str] = Field(
        default=None, alias="AGENT_SYSTEM_PROMPT"
    )

    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=9000, alias="PORT")
    debug: bool = Field(default=False, alias="DEBUG")
    enable_web_ui: bool = Field(default=False, alias="ENABLE_WEB_UI")

    provider: Optional[str] = Field(default=None, alias="PROVIDER")
    model_id: Optional[str] = Field(default=None, alias="MODEL_ID")
    llm_base_url: Optional[str] = Field(default=None, alias="LLM_BASE_URL")
    llm_api_key: Optional[str] = Field(default=None, alias="LLM_API_KEY")

    mcp_url: Optional[str] = Field(default=None, alias="MCP_URL")
    mcp_config: Optional[str] = Field(default=None, alias="MCP_CONFIG")

    routing_strategy: str = Field(default="hybrid", alias="ROUTING_STRATEGY")
    graph_persistence_type: str = Field(default="file", alias="GRAPH_PERSISTENCE_TYPE")
    graph_persistence_path: str = Field(
        default="agent_data/graph_state", alias="GRAPH_PERSISTENCE_PATH"
    )
    enable_llm_validation: bool = Field(default=True, alias="ENABLE_LLM_VALIDATION")

    custom_skills_directory: Optional[str] = Field(
        default=None, alias="CUSTOM_SKILLS_DIRECTORY"
    )
    load_universal_skills: bool = Field(default=False, alias="LOAD_UNIVERSAL_SKILLS")
    load_skill_graphs: bool = Field(default=False, alias="LOAD_SKILL_GRAPHS")

    enable_otel: bool = Field(default=False, alias="ENABLE_OTEL")
    otel_exporter_otlp_endpoint: Optional[str] = Field(
        default=None, alias="OTEL_EXPORTER_OTLP_ENDPOINT"
    )
    otel_exporter_otlp_headers: Optional[str] = Field(
        default=None, alias="OTEL_EXPORTER_OTLP_HEADERS"
    )
    otel_exporter_otlp_public_key: Optional[str] = Field(
        default=None, alias="OTEL_EXPORTER_OTLP_PUBLIC_KEY"
    )
    otel_exporter_otlp_secret_key: Optional[str] = Field(
        default=None, alias="OTEL_EXPORTER_OTLP_SECRET_KEY"
    )
    otel_exporter_otlp_protocol: str = Field(
        default="http/protobuf", alias="OTEL_EXPORTER_OTLP_PROTOCOL"
    )

    a2a_broker: str = Field(default="in-memory", alias="A2A_BROKER")
    a2a_broker_url: Optional[str] = Field(default=None, alias="A2A_BROKER_URL")
    a2a_storage: str = Field(default="in-memory", alias="A2A_STORAGE")
    a2a_storage_url: Optional[str] = Field(default=None, alias="A2A_STORAGE_URL")

    max_tokens: int = Field(default=16384, alias="MAX_TOKENS")
    temperature: float = Field(default=0.7, alias="TEMPERATURE")
    top_p: float = Field(default=1.0, alias="TOP_P")
    timeout: float = Field(default=32400.0, alias="TIMEOUT")
    tool_timeout: float = Field(default=32400.0, alias="TOOL_TIMEOUT")
    parallel_tool_calls: bool = Field(default=True, alias="PARALLEL_TOOL_CALLS")
    seed: Optional[int] = Field(default=None, alias="SEED")
    presence_penalty: float = Field(default=0.0, alias="PRESENCE_PENALTY")
    frequency_penalty: float = Field(default=0.0, alias="FREQUENCY_PENALTY")

    logit_bias: Optional[Dict[str, float]] = Field(default=None, alias="LOGIT_BIAS")
    stop_sequences: Optional[List[str]] = Field(default=None, alias="STOP_SEQUENCES")
    extra_headers: Optional[Dict[str, str]] = Field(default=None, alias="EXTRA_HEADERS")
    extra_body: Optional[Dict[str, Any]] = Field(default=None, alias="EXTRA_BODY")

    min_confidence: float = Field(default=0.4, alias="MIN_CONFIDENCE")
    validation_mode: bool = Field(default=False, alias="VALIDATION_MODE")
    approval_timeout: float = Field(default=0.0, alias="APPROVAL_TIMEOUT")

    tool_guard_mode: str = Field(default="native", alias="TOOL_GUARD_MODE")
    sensitive_tool_patterns: List[str] = Field(
        default=[
            r".*delete.*",
            r".*remove.*",
            r".*rm_.*",
            r".*rmdir.*",
            r".*drop.*",
            r".*truncate.*",
            r".*prune.*",
            r".*kill.*",
            r".*terminate.*",
            r".*reboot.*",
            r".*shutdown.*",
            r".*install.*",
            r".*uninstall.*",
            r".*redeploy.*",
            r".*bump.*",
            r".*create.*",
            r".*add.*",
            r".*post.*",
            r".*put.*",
            r".*insert.*",
            r".*upload.*",
            r".*ingest.*",
            r".*write.*",
            r".*update.*",
            r".*patch.*",
            r".*set.*",
            r".*reset.*",
            r".*clear.*",
            r".*revert.*",
            r".*replace.*",
            r".*rename.*",
            r".*move.*",
            r".*rotate.*",
            r".*start.*",
            r".*stop.*",
            r".*restart.*",
            r".*pause.*",
            r".*unpause.*",
            r".*execute.*",
            r".*shell.*",
            r".*run_.*",
            r".*git_.*",
            r".*enable.*",
            r".*disable.*",
            r".*activate.*",
            r".*approve.*",
            r".*graphql.*",
            r".*mutation.*",
        ],
        alias="SENSITIVE_TOOL_PATTERNS",
    )


config = AgentConfig()


DEFAULT_AGENT_NAME = config.default_agent_name
DEFAULT_AGENT_DESCRIPTION = config.agent_description
DEFAULT_AGENT_SYSTEM_PROMPT = config.agent_system_prompt
DEFAULT_HOST = config.host
DEFAULT_PORT = config.port
DEFAULT_DEBUG = config.debug
DEFAULT_PROVIDER = config.provider
DEFAULT_MODEL_ID = config.model_id
DEFAULT_LLM_BASE_URL = config.llm_base_url
DEFAULT_LLM_API_KEY = config.llm_api_key
DEFAULT_MCP_URL = config.mcp_url


DEFAULT_MCP_CONFIG = config.mcp_config
DEFAULT_CUSTOM_SKILLS_DIRECTORY = config.custom_skills_directory
DEFAULT_LOAD_UNIVERSAL_SKILLS = config.load_universal_skills
DEFAULT_LOAD_SKILL_GRAPHS = config.load_skill_graphs
DEFAULT_ENABLE_WEB_UI = config.enable_web_ui
DEFAULT_ENABLE_OTEL = config.enable_otel

if not config.enable_otel:
    os.environ["OTEL_SDK_DISABLED"] = "true"

DEFAULT_OTEL_EXPORTER_OTLP_ENDPOINT = config.otel_exporter_otlp_endpoint
DEFAULT_OTEL_EXPORTER_OTLP_HEADERS = config.otel_exporter_otlp_headers
DEFAULT_OTEL_EXPORTER_OTLP_PUBLIC_KEY = config.otel_exporter_otlp_public_key
DEFAULT_OTEL_EXPORTER_OTLP_SECRET_KEY = config.otel_exporter_otlp_secret_key
DEFAULT_OTEL_EXPORTER_OTLP_PROTOCOL = config.otel_exporter_otlp_protocol

DEFAULT_A2A_BROKER = config.a2a_broker
DEFAULT_A2A_BROKER_URL = config.a2a_broker_url
DEFAULT_A2A_STORAGE = config.a2a_storage
DEFAULT_A2A_STORAGE_URL = config.a2a_storage_url

DEFAULT_MAX_TOKENS = config.max_tokens
DEFAULT_TEMPERATURE = config.temperature
DEFAULT_TOP_P = config.top_p
DEFAULT_TIMEOUT = config.timeout
DEFAULT_TOOL_TIMEOUT = config.tool_timeout
DEFAULT_PARALLEL_TOOL_CALLS = config.parallel_tool_calls
DEFAULT_SEED = config.seed
DEFAULT_PRESENCE_PENALTY = config.presence_penalty
DEFAULT_FREQUENCY_PENALTY = config.frequency_penalty


DEFAULT_LOGIT_BIAS = (
    config.logit_bias
    if config.logit_bias is not None
    else to_dict(os.getenv("LOGIT_BIAS", None))
)
DEFAULT_STOP_SEQUENCES = (
    config.stop_sequences
    if config.stop_sequences is not None
    else to_list(os.getenv("STOP_SEQUENCES", None))
)
DEFAULT_EXTRA_HEADERS = (
    config.extra_headers
    if config.extra_headers is not None
    else to_dict(os.getenv("EXTRA_HEADERS", None))
)
DEFAULT_EXTRA_BODY = (
    config.extra_body
    if config.extra_body is not None
    else to_dict(os.getenv("EXTRA_BODY", None))
)

DEFAULT_MIN_CONFIDENCE = config.min_confidence
DEFAULT_VALIDATION_MODE = config.validation_mode or to_boolean(
    os.getenv("VALIDATION_MODE", "False")
)
DEFAULT_SSL_VERIFY = GET_DEFAULT_SSL_VERIFY()
DEFAULT_APPROVAL_TIMEOUT = config.approval_timeout
DEFAULT_MAX_CRON_LOG_ENTRIES = 50


TOOL_GUARD_MODE = config.tool_guard_mode
SENSITIVE_TOOL_PATTERNS = config.sensitive_tool_patterns

DEFAULT_ROUTER_MODEL = os.getenv(
    "GRAPH_ROUTER_MODEL", os.getenv("MODEL_ID", config.model_id)
)
DEFAULT_GRAPH_AGENT_MODEL = os.getenv(
    "GRAPH_AGENT_MODEL", os.getenv("MODEL_ID", config.model_id)
)
DEFAULT_ROUTER_PROVIDER = os.getenv(
    "GRAPH_ROUTER_PROVIDER", os.getenv("PROVIDER", "openai")
)
DEFAULT_ROUTER_BASE_URL = os.getenv("GRAPH_ROUTER_BASE_URL", os.getenv("LLM_BASE_URL"))
DEFAULT_ROUTER_API_KEY = os.getenv("GRAPH_ROUTER_API_KEY", os.getenv("LLM_API_KEY"))

DEFAULT_GRAPH_PERSISTENCE_TYPE = config.graph_persistence_type
DEFAULT_GRAPH_PERSISTENCE_PATH = config.graph_persistence_path
DEFAULT_ENABLE_LLM_VALIDATION = config.enable_llm_validation
DEFAULT_ROUTING_STRATEGY = config.routing_strategy
