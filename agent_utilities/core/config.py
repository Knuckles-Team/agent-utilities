#!/usr/bin/python
"""Configuration Management Module.

This module handles the loading and validation of agent settings from environment
variables and .env files using Pydantic Settings. It defines a centralized
AgentConfig class and exports default configuration constants used throughout
the agent-utilities package.
"""

import os
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from agent_utilities.base_utilities import (
    GET_DEFAULT_SSL_VERIFY,
    to_boolean,
    to_dict,
    to_list,
)

try:
    import logfire  # noqa: F401

    HAS_LOGFIRE = True
except ImportError:
    HAS_LOGFIRE = False

os.environ.setdefault("OTEL_SDK_DISABLED", "false")
os.environ.setdefault("LOGFIRE_SEND_TO_LOGFIRE", "false")
_enable_otel = to_boolean(os.environ.get("ENABLE_OTEL", "False"))
if _enable_otel:
    os.environ.setdefault("OTEL_SDK_DISABLED", "true")

meta = {"name": "Agent", "description": "AI Agent"}


def get_env_file() -> str | None:
    """Identify the location of the .env file for the calling package.

    This function attempts to find a .env file by checking the current working
    directory and paths relative to the retrieved package name.

    Returns:
        The path to the found .env file as a string, or '.env' as a default fallback.

    """
    from pathlib import Path

    from agent_utilities.base_utilities import retrieve_package_name

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
    """Configuration schema for the AI Agent server.

    Uses Pydantic BaseSettings to automatically load values from environment
    variables (via aliases) or a .env file. Covers LLM settings, server networking,
    observability (OTEL), A2A communication, and safety guards.
    """

    model_config = SettingsConfigDict(
        env_file=get_env_file(),
        env_ignore_empty=True,
        extra="ignore",
    )

    default_agent_name: str = Field(default=meta["name"], alias="DEFAULT_AGENT_NAME")
    agent_description: str = Field(
        default=meta["description"], alias="AGENT_DESCRIPTION"
    )
    agent_system_prompt: str | None = Field(default=None, alias="AGENT_SYSTEM_PROMPT")

    host: str = Field(default="0.0.0.0", alias="HOST")  # nosec B104
    port: int = Field(default=9000, alias="PORT")
    debug: bool = Field(default=False, alias="DEBUG")
    enable_web_ui: bool = Field(default=False, alias="ENABLE_WEB_UI")
    enable_terminal_ui: bool = Field(default=False, alias="ENABLE_TERMINAL_UI")
    enable_web_logs: bool = Field(default=True, alias="ENABLE_WEB_LOGS")
    enable_acp: bool = Field(default=False, alias="ENABLE_ACP")
    acp_port: int = Field(default=8001, alias="ACP_PORT")
    acp_session_root: str = Field(default=".acp-sessions", alias="ACP_SESSION_ROOT")

    provider: str | None = Field(default=None, alias="PROVIDER")
    model_id: str | None = Field(default=None, alias="MODEL_ID")
    llm_base_url: str | None = Field(default=None, alias="LLM_BASE_URL")
    llm_api_key: str | None = Field(default=None, alias="LLM_API_KEY")

    mcp_url: str | None = Field(default=None, alias="MCP_URL")
    mcp_config: str | None = Field(default=None, alias="MCP_CONFIG")

    agent_api_key: str | None = Field(default=None, alias="AGENT_API_KEY")
    enable_api_auth: bool = Field(default=False, alias="ENABLE_API_AUTH")
    max_upload_size: int = Field(default=10 * 1024 * 1024, alias="MAX_UPLOAD_SIZE")

    auth_jwt_jwks_uri: str | None = Field(default=None, alias="AUTH_JWT_JWKS_URI")
    """JWKS URI for JWT Bearer token verification (e.g. Azure AD, Okta)."""

    auth_jwt_issuer: str | None = Field(default=None, alias="AUTH_JWT_ISSUER")
    """Expected JWT issuer claim for validation."""

    auth_jwt_audience: str | None = Field(default=None, alias="AUTH_JWT_AUDIENCE")
    """Expected JWT audience claim for validation."""

    allowed_origins: str | None = Field(default=None, alias="ALLOWED_ORIGINS")
    """Comma-separated list of allowed CORS origins. Defaults to ``*`` if not set."""

    allowed_hosts: str | None = Field(default=None, alias="ALLOWED_HOSTS")
    """Comma-separated list of allowed hosts for TrustedHostMiddleware."""

    routing_strategy: str = Field(default="hybrid", alias="ROUTING_STRATEGY")
    graph_persistence_type: str = Field(default="file", alias="GRAPH_PERSISTENCE_TYPE")
    graph_persistence_path: str = Field(
        default="graph_state", alias="GRAPH_PERSISTENCE_PATH"
    )
    enable_llm_validation: bool = Field(default=False, alias="ENABLE_LLM_VALIDATION")
    graph_router_timeout: float = Field(default=300.0, alias="GRAPH_ROUTER_TIMEOUT")
    graph_verifier_timeout: float = Field(default=300.0, alias="GRAPH_VERIFIER_TIMEOUT")
    enable_kg_embeddings: bool = Field(default=True, alias="ENABLE_KG_EMBEDDINGS")
    kg_backups: int = Field(default=3, alias="KG_BACKUPS")
    graph_direct_execution: bool = Field(default=True, alias="GRAPH_DIRECT_EXECUTION")
    """When True, AG-UI and ACP adapters bypass the LLM tool-call hop
    and invoke graph execution directly.  Set to False to restore the
    legacy agent -> run_graph_flow -> graph pipeline."""

    secrets_backend: str = Field(default="inmemory", alias="SECRETS_BACKEND")
    """Secrets storage backend: 'inmemory' (default), 'sqlite', or 'vault'."""

    secrets_sqlite_path: str | None = Field(default=None, alias="SECRETS_SQLITE_PATH")
    """Path to SQLite secrets database (e.g. ~/.agent-utilities/secrets.db)."""

    secrets_vault_url: str | None = Field(default=None, alias="SECRETS_VAULT_URL")
    """HashiCorp Vault URL for the vault backend."""

    secrets_vault_mount: str = Field(default="secret", alias="SECRETS_VAULT_MOUNT")
    """Vault KV v2 mount point."""

    custom_skills_directory: str | None = Field(
        default=None, alias="CUSTOM_SKILLS_DIRECTORY"
    )
    skill_types: list[str] | None = Field(default=None, alias="SKILL_TYPES")

    enable_otel: bool = Field(default=False, alias="ENABLE_OTEL")
    otel_exporter_otlp_endpoint: str | None = Field(
        default=None, alias="OTEL_EXPORTER_OTLP_ENDPOINT"
    )
    otel_exporter_otlp_headers: str | None = Field(
        default=None, alias="OTEL_EXPORTER_OTLP_HEADERS"
    )
    otel_exporter_otlp_public_key: str | None = Field(
        default=None, alias="OTEL_EXPORTER_OTLP_PUBLIC_KEY"
    )
    otel_exporter_otlp_secret_key: str | None = Field(
        default=None, alias="OTEL_EXPORTER_OTLP_SECRET_KEY"
    )
    otel_exporter_otlp_protocol: str = Field(
        default="http/protobuf", alias="OTEL_EXPORTER_OTLP_PROTOCOL"
    )

    a2a_broker: str = Field(default="in-memory", alias="A2A_BROKER")
    a2a_broker_url: str | None = Field(default=None, alias="A2A_BROKER_URL")
    a2a_storage: str = Field(default="in-memory", alias="A2A_STORAGE")
    a2a_storage_url: str | None = Field(default=None, alias="A2A_STORAGE_URL")

    max_tokens: int = Field(default=16384, alias="MAX_TOKENS")
    temperature: float = Field(default=0.7, alias="TEMPERATURE")
    top_p: float = Field(default=1.0, alias="TOP_P")
    timeout: float = Field(default=32400.0, alias="TIMEOUT")
    tool_timeout: float = Field(default=32400.0, alias="TOOL_TIMEOUT")
    parallel_tool_calls: bool = Field(default=True, alias="PARALLEL_TOOL_CALLS")
    seed: int | None = Field(default=None, alias="SEED")
    presence_penalty: float = Field(default=0.0, alias="PRESENCE_PENALTY")
    frequency_penalty: float = Field(default=0.0, alias="FREQUENCY_PENALTY")

    logit_bias: dict[str, float] | None = Field(default=None, alias="LOGIT_BIAS")
    stop_sequences: list[str] | None = Field(default=None, alias="STOP_SEQUENCES")
    extra_headers: dict[str, str] | None = Field(default=None, alias="EXTRA_HEADERS")
    extra_body: dict[str, Any] | None = Field(default=None, alias="EXTRA_BODY")

    min_confidence: float = Field(default=0.4, alias="MIN_CONFIDENCE")
    validation_mode: bool = Field(default=False, alias="VALIDATION_MODE")
    approval_timeout: float = Field(default=0.0, alias="APPROVAL_TIMEOUT")

    tool_guard_mode: str = Field(default="strict", alias="TOOL_GUARD_MODE")
    sensitive_tool_patterns: list[str] = Field(
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
            r".*run_shell.*",
            r".*run_command.*",
            r".*run_script.*",
            r".*run_code.*",
            r".*git_.*",
            r".*clone.*",
            r".*pull.*",
            r".*maintain.*",
            r".*setup.*",
            r".*build.*",
            r".*validate.*",
            r".*sync.*",
            r".*enable.*",
            r".*disable.*",
            r".*activate.*",
            r".*approve.*",
            r".*graphql.*",
            r".*mutation.*",
            r".*http.*",
            r".*eval.*",
            r".*exec.*",
            r".*compile.*",
            r".*socket.*",
            r".*connect.*",
            r".*os\..*",
            r".*subprocess\..*",
            r".*shutil\..*",
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
DEFAULT_SKILL_TYPES = config.skill_types
DEFAULT_ENABLE_WEB_UI = config.enable_web_ui
DEFAULT_ENABLE_TERMINAL_UI = config.enable_terminal_ui
DEFAULT_ENABLE_WEB_LOGS = config.enable_web_logs
DEFAULT_ENABLE_OTEL = config.enable_otel
DEFAULT_ENABLE_ACP = config.enable_acp
DEFAULT_ACP_PORT = config.acp_port
DEFAULT_ACP_SESSION_ROOT = config.acp_session_root

if not config.enable_otel:
    os.environ["OTEL_SDK_DISABLED"] = "true"
else:
    os.environ.pop("OTEL_SDK_DISABLED", None)

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
    else to_dict(os.getenv("LOGIT_BIAS"))
)
DEFAULT_STOP_SEQUENCES = (
    config.stop_sequences
    if config.stop_sequences is not None
    else to_list(os.getenv("STOP_SEQUENCES"))
)
DEFAULT_EXTRA_HEADERS = (
    config.extra_headers
    if config.extra_headers is not None
    else to_dict(os.getenv("EXTRA_HEADERS"))
)
DEFAULT_EXTRA_BODY = (
    config.extra_body
    if config.extra_body is not None
    else to_dict(os.getenv("EXTRA_BODY"))
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
DEFAULT_GRAPH_ROUTER_TIMEOUT = config.graph_router_timeout
DEFAULT_GRAPH_VERIFIER_TIMEOUT = config.graph_verifier_timeout
DEFAULT_ENABLE_KG_EMBEDDINGS = config.enable_kg_embeddings
DEFAULT_KG_BACKUPS = config.kg_backups
DEFAULT_GRAPH_DIRECT_EXECUTION = config.graph_direct_execution

AGENT_API_KEY = config.agent_api_key
ENABLE_API_AUTH = config.enable_api_auth
MAX_UPLOAD_SIZE = config.max_upload_size

SECRETS_BACKEND = config.secrets_backend
SECRETS_SQLITE_PATH = config.secrets_sqlite_path
SECRETS_VAULT_URL = config.secrets_vault_url
SECRETS_VAULT_MOUNT = config.secrets_vault_mount

AUTH_JWT_JWKS_URI = config.auth_jwt_jwks_uri
AUTH_JWT_ISSUER = config.auth_jwt_issuer
AUTH_JWT_AUDIENCE = config.auth_jwt_audience
ALLOWED_ORIGINS = config.allowed_origins
ALLOWED_HOSTS = config.allowed_hosts
