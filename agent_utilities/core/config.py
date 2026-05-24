#!/usr/bin/python
"""Configuration Management Module.

This module handles the loading and validation of agent settings from environment
variables and .env files using Pydantic Settings. It defines a centralized
AgentConfig class and exports default configuration constants used throughout
the agent-utilities package. Leverages XDG Standards for config file placement.
"""

import os
from typing import Any

import platformdirs
from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

DEFAULT_DB_PATH = str(
    platformdirs.user_data_path("agent-utilities", "knuckles-team") / "graph_state"
)

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


try:
    from dotenv import load_dotenv

    _env_file = get_env_file()
    if _env_file:
        load_dotenv(_env_file)
except ImportError:
    pass


def _load_xdg_json_config():
    import json
    from pathlib import Path

    import platformdirs

    APP_NAME = "agent-utilities"
    APP_AUTHOR = "knuckles-team"

    override = os.environ.get("AGENT_UTILITIES_CONFIG_DIR")
    if override:
        cfg_dir = Path(override).expanduser()
    else:
        cfg_dir = Path(platformdirs.user_config_path(APP_NAME, APP_AUTHOR))

    cfg_file = cfg_dir / "config.json"
    if cfg_file.exists():
        try:
            with open(cfg_file) as f:
                data = json.load(f)
                for k, v in data.items():
                    env_key = k.upper()
                    if env_key not in os.environ:
                        if isinstance(v, list | dict):
                            os.environ[env_key] = json.dumps(v)
                        elif v is None:
                            os.environ[env_key] = ""
                        else:
                            os.environ[env_key] = str(v)
        except Exception as e:
            import logging

            logging.getLogger(__name__).debug(f"Failed to load XDG JSON config: {e}")


from pydantic import BaseModel


class ChatModelConfig(BaseModel):
    id: str
    provider: str
    intelligence_level: str = "normal"
    base_url: str | None = None
    api_key: str | None = None
    supports_json: bool = False
    vision: bool = False
    reasoning: bool = False
    tools_enabled: bool = False
    parallel_instances: int = 1
    can_route: bool = False
    can_kg: bool = False


class EmbeddingModelConfig(BaseModel):
    id: str
    provider: str
    base_url: str | None = None
    api_key: str | None = None
    parallel_instances: int = 1
    chunk_size: int = 768


_load_xdg_json_config()


class NestedSecretsSettingsSource(PydanticBaseSettingsSource):
    """Custom settings source to load nested secrets from a centralized vault/file."""

    def __init__(self, settings_cls: type[BaseSettings], secrets_file: str = ""):
        super().__init__(settings_cls)
        self.secrets_file = secrets_file

    def get_field_value(self, field, field_name: str) -> tuple[Any, str, bool]:
        # Simple implementation that would integrate with actual Vault or nested JSON secrets
        return None, field_name, False

    def prepare_field_value(
        self, field_name: str, field, value: Any, _value_is_complex: bool
    ) -> Any:
        return value

    def __call__(self) -> dict[str, Any]:
        if not self.secrets_file:
            return {}
        try:
            import json

            with open(self.secrets_file) as f:
                return json.load(f)
        except Exception:
            return {}


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
        secrets_dir="/run/secrets" if os.path.exists("/run/secrets") else None,
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            NestedSecretsSettingsSource(
                settings_cls, os.getenv("AGENT_SECRETS_FILE", "")
            ),
            file_secret_settings,
        )

    chat_models: list[ChatModelConfig] = Field(
        default_factory=list, alias="CHAT_MODELS"
    )
    embedding_models: list[EmbeddingModelConfig] = Field(
        default_factory=list, alias="EMBEDDING_MODELS"
    )

    # --- Provider API Keys (global fallbacks for ad-hoc model creation) ---
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_base_url: str | None = Field(default=None, alias="OPENAI_BASE_URL")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")
    mistral_api_key: str | None = Field(default=None, alias="MISTRAL_API_KEY")
    hugging_face_api_key: str | None = Field(default=None, alias="HUGGING_FACE_API_KEY")
    deepseek_api_key: str | None = Field(default=None, alias="DEEPSEEK_API_KEY")
    deepseek_base_url: str | None = Field(default=None, alias="DEEPSEEK_BASE_URL")

    # --- Graph / KG tuning knobs ---
    graph_timeout: str | None = Field(default="1200000", alias="GRAPH_TIMEOUT")
    max_recursion_depth: str | None = Field(default="2", alias="MAX_RECURSION_DEPTH")
    routing_percentile: str | None = Field(default="50.0", alias="ROUTING_PERCENTILE")
    kg_embedding_dim: str | None = Field(default="768", alias="KG_EMBEDDING_DIM")

    # --- Model registry helpers (derive from chat_models / embedding_models) ---

    def _chat_model_by_level(self, level: str) -> ChatModelConfig | None:
        """Return the first chat model matching the given intelligence_level."""
        for m in self.chat_models:
            if m.intelligence_level == level:
                return m
        return None

    @property
    def default_chat_model(self) -> ChatModelConfig | None:
        """Primary chat model (intelligence_level='normal', fallback to first)."""
        return self._chat_model_by_level("normal") or (
            self.chat_models[0] if self.chat_models else None
        )

    @property
    def lite_chat_model(self) -> ChatModelConfig | None:
        """Lightweight chat model (intelligence_level='light')."""
        return self._chat_model_by_level("light") or self.default_chat_model

    @property
    def super_chat_model(self) -> ChatModelConfig | None:
        """Super/heavy chat model (intelligence_level='super')."""
        return self._chat_model_by_level("super") or self.default_chat_model

    @property
    def default_embedding_model(self) -> EmbeddingModelConfig | None:
        """Primary embedding model (first in list)."""
        return self.embedding_models[0] if self.embedding_models else None

    def reload(self):
        """Reload configuration from XDG config.json dynamically."""
        _load_xdg_json_config()
        # Reparse from environment
        new_config = self.__class__()
        for field in self.__class__.model_fields.keys():
            setattr(self, field, getattr(new_config, field))

    default_agent_name: str = Field(default=meta["name"], alias="DEFAULT_AGENT_NAME")
    agent_description: str = Field(
        default=meta["description"], alias="AGENT_DESCRIPTION"
    )
    agent_system_prompt: str | None = Field(default=None, alias="AGENT_SYSTEM_PROMPT")

    workspace_path: str | None = Field(default=None, alias="WORKSPACE_PATH")
    agent_utilities_config_dir: str | None = Field(
        default=None, alias="AGENT_UTILITIES_CONFIG_DIR"
    )

    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=9000, alias="PORT")
    debug: bool = Field(default=False, alias="DEBUG")
    enable_web_ui: bool = Field(default=False, alias="ENABLE_WEB_UI")
    enable_terminal_ui: bool = Field(default=False, alias="ENABLE_TERMINAL_UI")
    enable_web_logs: bool = Field(default=True, alias="ENABLE_WEB_LOGS")
    enable_acp: bool = Field(default=False, alias="ENABLE_ACP")
    acp_port: int = Field(default=8001, alias="ACP_PORT")
    acp_session_root: str = Field(default=".acp-sessions", alias="ACP_SESSION_ROOT")
    default_terminal_agent: str = Field(
        default="agent-terminal-ui", alias="DEFAULT_TERMINAL_AGENT"
    )

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

    # --- OIDC / OAuth 2.0 Delegation (CONCEPT:ECO-4.0) ---

    oidc_config_url: str | None = Field(default=None, alias="OIDC_CONFIG_URL")
    """OIDC discovery URL (e.g. https://idp.example.com/.well-known/openid-configuration).
    Works with any OIDC-compliant Identity Provider."""

    oidc_client_id: str | None = Field(default=None, alias="OIDC_CLIENT_ID")
    """OAuth 2.0 client ID registered with the Identity Provider."""

    oidc_client_secret: str | None = Field(default=None, alias="OIDC_CLIENT_SECRET")
    """OAuth 2.0 client secret registered with the Identity Provider."""

    enable_delegation: bool = Field(default=False, alias="ENABLE_DELEGATION")
    """Enable OIDC token delegation (RFC 8693 Token Exchange) for downstream API calls."""

    delegation_audience: str | None = Field(default=None, alias="AUDIENCE")
    """Target audience for delegated tokens (e.g. the downstream API base URL)."""

    delegated_scopes: str = Field(default="api", alias="DELEGATED_SCOPES")
    """Space-separated scopes requested during token delegation."""

    # --- Vault Secrets Backend (CONCEPT:OS-5.1) ---

    vault_url: str | None = Field(default=None, alias="SECRETS_VAULT_URL")
    """HashiCorp Vault URL for the secrets backend."""

    vault_mount: str = Field(default="secret", alias="SECRETS_VAULT_MOUNT")
    """Vault KV v2 mount point."""

    vault_auth_method: str = Field(default="auto", alias="VAULT_AUTH_METHOD")
    """Vault auth method: 'oidc', 'approle', 'token', 'kubernetes', 'auto'."""

    vault_auth_mount: str = Field(default="jwt", alias="VAULT_AUTH_MOUNT")
    """Vault auth method mount path.  Supports custom mounts
    (e.g. 'oidc', 'jwt', 'my-okta-auth')."""

    vault_role: str | None = Field(default=None, alias="VAULT_ROLE")
    """Vault role name for OIDC/JWT or Kubernetes login."""

    vault_path_prefix: str | None = Field(default=None, alias="VAULT_PATH_PREFIX")
    """Path prefix within the KV v2 mount (e.g. 'agents/mcp/')."""

    allowed_origins: str | None = Field(default=None, alias="ALLOWED_ORIGINS")
    """Comma-separated list of allowed CORS origins. Defaults to ``*`` if not set."""

    allowed_hosts: str | None = Field(default=None, alias="ALLOWED_HOSTS")
    """Comma-separated list of allowed hosts for TrustedHostMiddleware."""

    routing_strategy: str = Field(default="hybrid", alias="ROUTING_STRATEGY")
    graph_persistence_type: str = Field(default="file", alias="GRAPH_PERSISTENCE_TYPE")
    graph_persistence_path: str = Field(
        default=DEFAULT_DB_PATH, alias="GRAPH_PERSISTENCE_PATH"
    )
    enable_llm_validation: bool = Field(default=False, alias="ENABLE_LLM_VALIDATION")
    graph_router_timeout: float = Field(default=300.0, alias="GRAPH_ROUTER_TIMEOUT")
    graph_verifier_timeout: float = Field(default=300.0, alias="GRAPH_VERIFIER_TIMEOUT")
    enable_kg_embeddings: bool = Field(default=True, alias="ENABLE_KG_EMBEDDINGS")
    kg_backups: int = Field(default=3, alias="KG_BACKUPS")
    kg_ingestion_workers: int | None = Field(default=None, alias="KG_INGESTION_WORKERS")
    kg_llm_concurrency: int = Field(default=4, alias="KG_LLM_CONCURRENCY")
    """Max concurrent LLM calls for KG operations (Layer 2/3 analysis, embeddings).
    Set to match your inference endpoint's parallel capacity (e.g. LM Studio slots)."""

    kg_analysis_max_depth: int = Field(default=2, alias="KG_ANALYSIS_MAX_DEPTH")
    """Maximum recursive depth for background knowledge graph research daemons."""
    knowledge_graph_sync_background: bool = Field(
        default=True, alias="KNOWLEDGE_GRAPH_SYNC_BACKGROUND"
    )
    """Enable or disable background task workers for the Knowledge Graph pipeline."""
    model_registry_path: str | None = Field(default=None, alias="MODEL_REGISTRY_PATH")
    """Path to a YAML or JSON file defining the model registry."""
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

    langfuse_public_key: str | None = Field(default=None, alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str | None = Field(default=None, alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com", alias="LANGFUSE_BASE_URL"
    )
    langfuse_dataset_capture_threshold: float = Field(
        default=0.0, alias="LANGFUSE_DATASET_CAPTURE_THRESHOLD"
    )
    langfuse_latency_baseline_seconds: float = Field(
        default=60.0, alias="LANGFUSE_LATENCY_BASELINE_SECONDS"
    )
    langfuse_token_baseline: int = Field(default=20000, alias="LANGFUSE_TOKEN_BASELINE")
    langfuse_verifier_fallback_limit: int = Field(
        default=1, alias="LANGFUSE_VERIFIER_FALLBACK_LIMIT"
    )

    a2a_broker: str = Field(default="in-memory", alias="A2A_BROKER")
    a2a_broker_url: str | None = Field(default=None, alias="A2A_BROKER_URL")
    a2a_storage: str = Field(default="in-memory", alias="A2A_STORAGE")
    a2a_storage_url: str | None = Field(default=None, alias="A2A_STORAGE_URL")
    a2a_config: str | None = Field(default=None, alias="A2A_CONFIG")
    """Path to a2a_config.json for external A2A agent discovery (CONCEPT:ECO-4.0)."""
    a2a_refresh_interval: int = Field(default=300, alias="A2A_REFRESH_INTERVAL")
    """Interval in seconds for periodic A2A agent card re-fetch (CONCEPT:ECO-4.0)."""

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

    # --- Agent OS Architecture (CONCEPT:OS-5.2) ---

    cognitive_scheduler_enabled: bool = Field(
        default=True, alias="COGNITIVE_SCHEDULER_ENABLED"
    )
    """Enable the Cognitive Scheduler for priority-aware agent management (CONCEPT:OS-5.2)."""

    max_concurrent_agents: int = Field(default=5, alias="MAX_CONCURRENT_AGENTS")
    """Maximum number of concurrently running specialist agents (CONCEPT:OS-5.2)."""

    agent_token_quota: int = Field(default=100_000, alias="AGENT_TOKEN_QUOTA")
    """Default per-agent token budget before preemption (CONCEPT:OS-5.2)."""

    preemption_threshold_pct: float = Field(
        default=0.85, alias="PREEMPTION_THRESHOLD_PCT"
    )
    """Quota usage percentage that triggers preemption warning (CONCEPT:OS-5.2)."""

    agent_policies_path: str | None = Field(default=None, alias="AGENT_POLICIES_PATH")
    """Path to agent_policies.json for identity-based governance (CONCEPT:OS-5.2)."""

    permissions_signing_key: str | None = Field(
        default=None, alias="PERMISSIONS_SIGNING_KEY"
    )
    """HMAC signing key for agent identity tokens. Auto-generated if not set (CONCEPT:OS-5.2)."""

    specialist_registry_path: str | None = Field(
        default=None, alias="SPECIALIST_REGISTRY_PATH"
    )
    """Path to local specialist registry directory (CONCEPT:OS-5.2)."""

    # --- Native Messaging Backend (CONCEPT:ECO-4.5) ---

    messaging_enabled_backends: list[str] = Field(
        default_factory=list, alias="MESSAGING_ENABLED_BACKENDS"
    )
    """List of messaging backend IDs to auto-connect on startup (CONCEPT:ECO-4.5).
    Example: ["discord", "slack", "telegram"]."""

    messaging_kg_ingest: bool = Field(default=True, alias="MESSAGING_KG_INGEST")
    """Enable automatic Knowledge Graph ingestion for all inbound/outbound messages (CONCEPT:ECO-4.5)."""

    messaging_kg_memory_type: str = Field(
        default="episodic", alias="MESSAGING_KG_MEMORY_TYPE"
    )
    """Default KG memory tier for inbound messages: 'episodic', 'semantic', or 'procedural' (CONCEPT:ECO-4.5)."""

    messaging_route_to_planner: bool = Field(
        default=True, alias="MESSAGING_ROUTE_TO_PLANNER"
    )
    """Route inbound messaging events to the Planner Graph Agent for orchestration (CONCEPT:ECO-4.5)."""

    # Per-platform tokens (read from config.json or env vars)
    messaging_discord_token: str | None = Field(
        default=None, alias="MESSAGING_DISCORD_TOKEN"
    )
    """Discord bot token. Also reads from DISCORD_BOT_TOKEN (CONCEPT:ECO-4.5)."""

    messaging_slack_token: str | None = Field(
        default=None, alias="MESSAGING_SLACK_TOKEN"
    )
    """Slack bot token (xoxb-...). Also reads from SLACK_BOT_TOKEN (CONCEPT:ECO-4.5)."""

    messaging_slack_app_token: str | None = Field(
        default=None, alias="MESSAGING_SLACK_APP_TOKEN"
    )
    """Slack app-level token (xapp-...) for Socket Mode (CONCEPT:ECO-4.5)."""

    messaging_telegram_token: str | None = Field(
        default=None, alias="MESSAGING_TELEGRAM_TOKEN"
    )
    """Telegram bot token. Also reads from TELEGRAM_BOT_TOKEN (CONCEPT:ECO-4.5)."""

    messaging_whatsapp_token: str | None = Field(
        default=None, alias="MESSAGING_WHATSAPP_TOKEN"
    )
    """WhatsApp API token. Also reads from WHATSAPP_TOKEN (CONCEPT:ECO-4.5)."""

    messaging_whatsapp_phone_number_id: str | None = Field(
        default=None, alias="MESSAGING_WHATSAPP_PHONE_NUMBER_ID"
    )
    """WhatsApp Business API phone number ID (CONCEPT:ECO-4.5)."""

    messaging_whatsapp_use_business_api: bool = Field(
        default=False, alias="MESSAGING_WHATSAPP_USE_BUSINESS_API"
    )
    """Use official WhatsApp Business API (True) or neonize bridge (False) (CONCEPT:ECO-4.5)."""

    messaging_teams_app_id: str | None = Field(
        default=None, alias="MESSAGING_TEAMS_APP_ID"
    )
    """Microsoft Teams Bot Framework app ID (CONCEPT:ECO-4.5)."""

    messaging_teams_app_secret: str | None = Field(
        default=None, alias="MESSAGING_TEAMS_APP_SECRET"
    )
    """Microsoft Teams Bot Framework app password (CONCEPT:ECO-4.5)."""

    messaging_googlechat_service_account: str | None = Field(
        default=None, alias="MESSAGING_GOOGLECHAT_TOKEN"
    )
    """Path to Google Chat service account JSON file (CONCEPT:ECO-4.5)."""

    messaging_googlemeet_service_account: str | None = Field(
        default=None, alias="MESSAGING_GOOGLEMEET_TOKEN"
    )
    """Path to Google Meet service account JSON file (CONCEPT:ECO-4.5)."""

    messaging_mattermost_token: str | None = Field(
        default=None, alias="MESSAGING_MATTERMOST_TOKEN"
    )
    """Mattermost personal access token (CONCEPT:ECO-4.5)."""

    messaging_mattermost_url: str | None = Field(
        default=None, alias="MESSAGING_MATTERMOST_URL"
    )
    """Mattermost server URL (CONCEPT:ECO-4.5)."""

    messaging_matrix_token: str | None = Field(
        default=None, alias="MESSAGING_MATRIX_TOKEN"
    )
    """Matrix access token (CONCEPT:ECO-4.5)."""

    messaging_matrix_homeserver: str | None = Field(
        default=None, alias="MESSAGING_MATRIX_HOMESERVER"
    )
    """Matrix homeserver URL (CONCEPT:ECO-4.5)."""

    messaging_matrix_user_id: str | None = Field(
        default=None, alias="MESSAGING_MATRIX_USER_ID"
    )
    """Matrix user ID (e.g. @bot:matrix.org) (CONCEPT:ECO-4.5)."""

    messaging_irc_server: str | None = Field(default=None, alias="MESSAGING_IRC_SERVER")
    """IRC server hostname (CONCEPT:ECO-4.5)."""

    messaging_irc_port: int = Field(default=6667, alias="MESSAGING_IRC_PORT")
    """IRC server port (CONCEPT:ECO-4.5)."""

    messaging_irc_nickname: str = Field(
        default="agent_bot", alias="MESSAGING_IRC_NICKNAME"
    )
    """IRC nickname (CONCEPT:ECO-4.5)."""

    messaging_irc_channels: list[str] = Field(
        default_factory=list, alias="MESSAGING_IRC_CHANNELS"
    )
    """IRC channels to auto-join (CONCEPT:ECO-4.5)."""

    messaging_signal_phone: str | None = Field(
        default=None, alias="MESSAGING_SIGNAL_TOKEN"
    )
    """Signal phone number for semaphore-bot (CONCEPT:ECO-4.5)."""

    messaging_line_token: str | None = Field(default=None, alias="MESSAGING_LINE_TOKEN")
    """LINE channel access token (CONCEPT:ECO-4.5)."""

    messaging_twitch_token: str | None = Field(
        default=None, alias="MESSAGING_TWITCH_TOKEN"
    )
    """Twitch OAuth token (CONCEPT:ECO-4.5)."""

    messaging_twitch_channels: list[str] = Field(
        default_factory=list, alias="MESSAGING_TWITCH_CHANNELS"
    )
    """Twitch channels to join (CONCEPT:ECO-4.5)."""

    messaging_synology_webhook_url: str | None = Field(
        default=None, alias="MESSAGING_SYNOLOGY_WEBHOOK_URL"
    )
    """Synology Chat incoming webhook URL (CONCEPT:ECO-4.5)."""

    messaging_twilio_account_sid: str | None = Field(
        default=None, alias="MESSAGING_VOICECALL_APP_ID"
    )
    """Twilio account SID for voice/SMS (CONCEPT:ECO-4.5)."""

    messaging_twilio_auth_token: str | None = Field(
        default=None, alias="MESSAGING_VOICECALL_TOKEN"
    )
    """Twilio auth token for voice/SMS (CONCEPT:ECO-4.5)."""

    messaging_twilio_from_number: str | None = Field(
        default=None, alias="MESSAGING_VOICECALL_FROM_NUMBER"
    )
    """Twilio 'from' phone number (CONCEPT:ECO-4.5)."""

    messaging_nextcloud_url: str | None = Field(
        default=None, alias="MESSAGING_NEXTCLOUD_URL"
    )
    """Nextcloud server URL (CONCEPT:ECO-4.5)."""

    messaging_nextcloud_token: str | None = Field(
        default=None, alias="MESSAGING_NEXTCLOUD_TOKEN"
    )
    """Nextcloud app token (CONCEPT:ECO-4.5)."""

    messaging_nextcloud_user: str | None = Field(
        default=None, alias="MESSAGING_NEXTCLOUD_APP_ID"
    )
    """Nextcloud username (CONCEPT:ECO-4.5)."""

    # --- Parallel Engine (CONCEPT:ORCH-1.25) ---

    max_parallel_agents: int = Field(default=60, alias="MAX_PARALLEL_AGENTS")
    """Maximum concurrent agent executions across the engine (CONCEPT:ORCH-1.25).
    Acts as a global semaphore. Set higher for cloud deployments with high API limits."""

    parallel_batch_size: int = Field(default=25, alias="PARALLEL_BATCH_SIZE")
    """Number of agents per execution wave when batching is needed (CONCEPT:ORCH-1.25)."""

    synthesis_strategy: str = Field(default="auto", alias="SYNTHESIS_STRATEGY")
    """Default output synthesis strategy: 'auto', 'flat', 'hierarchical', 'progressive', 'rlm'.
    'auto' selects based on agent count and output size (CONCEPT:ORCH-1.26)."""

    synthesis_ratio: int = Field(default=10, alias="SYNTHESIS_RATIO")
    """In hierarchical synthesis, how many outputs per synthesis sub-node (CONCEPT:ORCH-1.26)."""

    agent_execution_timeout: float = Field(
        default=120.0, alias="AGENT_EXECUTION_TIMEOUT"
    )
    """Per-agent execution timeout in seconds (CONCEPT:ORCH-1.25)."""

    circuit_breaker_threshold: int = Field(default=3, alias="CIRCUIT_BREAKER_THRESHOLD")
    """Number of consecutive failures before disabling an agent type (CONCEPT:ORCH-1.25)."""

    enable_progressive_synthesis: bool = Field(
        default=True, alias="ENABLE_PROGRESSIVE_SYNTHESIS"
    )
    """Enable streaming synthesis as agents complete (CONCEPT:ORCH-1.26)."""

    # --- Innovation Framework (CONCEPT:OS-5.2 through CONCEPT:OS-5.2) ---

    homeostatic_downgrade_enabled: bool = Field(
        default=True, alias="HOMEOSTATIC_DOWNGRADE_ENABLED"
    )
    """Enable automatic model tier downgrade when budget is under pressure (CONCEPT:OS-5.2)."""

    adversarial_verification: bool = Field(
        default=False, alias="ADVERSARIAL_VERIFICATION"
    )
    """Enable adversarial verification pass (opt-in only, doubles verification cost) (CONCEPT:AHE-3.1)."""

    maintenance_token_budget: int = Field(default=0, alias="MAINTENANCE_TOKEN_BUDGET")
    """Token budget for autonomous maintenance cron (0 = unlimited) (CONCEPT:OS-5.2)."""

    maintenance_priority: str = Field(default="LOW", alias="MAINTENANCE_PRIORITY")
    """Priority level for autonomous maintenance tasks (LOW/MEDIUM/HIGH) (CONCEPT:OS-5.2)."""

    watchdog_patterns: list[str] = Field(
        default=[
            "pyproject.toml",
            "mcp_config.json",
            "requirements*.txt",
        ],
        alias="WATCHDOG_PATTERNS",
    )
    """File patterns to monitor for the file watcher trigger (CONCEPT:OS-5.0)."""

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
# --- Derive DEFAULT_LLM_* from chat_models / embedding_models registry ---
_default_chat = config.default_chat_model
_lite_chat = config.lite_chat_model
_super_chat = config.super_chat_model
_default_embed = config.default_embedding_model

DEFAULT_LLM_PROVIDER = (
    (_default_chat.provider if _default_chat else None)
    or os.getenv("PROVIDER")
    or "openai"
)
DEFAULT_LLM_MODEL_ID = (
    (_default_chat.id if _default_chat else None)
    or os.getenv("MODEL_ID")
    or "qwen/qwen3.5-9b"
)
DEFAULT_LLM_BASE_URL = _default_chat.base_url if _default_chat else None
DEFAULT_LLM_API_KEY = _default_chat.api_key if _default_chat else None

DEFAULT_LITE_LLM_PROVIDER = (
    _lite_chat.provider if _lite_chat else None
) or DEFAULT_LLM_PROVIDER
DEFAULT_LITE_LLM_MODEL_ID = (
    _lite_chat.id if _lite_chat else None
) or DEFAULT_LLM_MODEL_ID
DEFAULT_LITE_LLM_BASE_URL = (
    _lite_chat.base_url if _lite_chat else None
) or DEFAULT_LLM_BASE_URL
DEFAULT_LITE_LLM_API_KEY = (
    _lite_chat.api_key if _lite_chat else None
) or DEFAULT_LLM_API_KEY

DEFAULT_SUPER_LLM_PROVIDER = (
    _super_chat.provider if _super_chat else None
) or DEFAULT_LLM_PROVIDER
DEFAULT_SUPER_LLM_MODEL_ID = (
    _super_chat.id if _super_chat else None
) or DEFAULT_LLM_MODEL_ID
DEFAULT_SUPER_LLM_BASE_URL = (
    _super_chat.base_url if _super_chat else None
) or DEFAULT_LLM_BASE_URL
DEFAULT_SUPER_LLM_API_KEY = (
    _super_chat.api_key if _super_chat else None
) or DEFAULT_LLM_API_KEY

DEFAULT_EMBEDDING_PROVIDER = (
    _default_embed.provider if _default_embed else None
) or DEFAULT_LLM_PROVIDER
DEFAULT_EMBEDDING_MODEL_ID = (
    _default_embed.id if _default_embed else None
) or "text-embedding-nomic-embed-text-v2-moe"
DEFAULT_EMBEDDING_BASE_URL = (
    _default_embed.base_url if _default_embed else None
) or DEFAULT_LLM_BASE_URL
DEFAULT_EMBEDDING_API_KEY = (
    _default_embed.api_key if _default_embed else None
) or DEFAULT_LLM_API_KEY
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
DEFAULT_TERMINAL_AGENT = config.default_terminal_agent

if not config.enable_otel:
    os.environ["OTEL_SDK_DISABLED"] = "true"
else:
    os.environ.pop("OTEL_SDK_DISABLED", None)

DEFAULT_OTEL_EXPORTER_OTLP_ENDPOINT = config.otel_exporter_otlp_endpoint
DEFAULT_OTEL_EXPORTER_OTLP_HEADERS = config.otel_exporter_otlp_headers
DEFAULT_OTEL_EXPORTER_OTLP_PUBLIC_KEY = config.otel_exporter_otlp_public_key
DEFAULT_OTEL_EXPORTER_OTLP_SECRET_KEY = config.otel_exporter_otlp_secret_key
DEFAULT_OTEL_EXPORTER_OTLP_PROTOCOL = config.otel_exporter_otlp_protocol

DEFAULT_LANGFUSE_PUBLIC_KEY = config.langfuse_public_key
DEFAULT_LANGFUSE_SECRET_KEY = config.langfuse_secret_key
DEFAULT_LANGFUSE_HOST = config.langfuse_host
DEFAULT_LANGFUSE_DATASET_CAPTURE_THRESHOLD = config.langfuse_dataset_capture_threshold

DEFAULT_A2A_BROKER = config.a2a_broker
DEFAULT_A2A_BROKER_URL = config.a2a_broker_url
DEFAULT_A2A_STORAGE = config.a2a_storage
DEFAULT_A2A_STORAGE_URL = config.a2a_storage_url
DEFAULT_A2A_CONFIG = config.a2a_config
DEFAULT_A2A_REFRESH_INTERVAL = config.a2a_refresh_interval

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
DEFAULT_VALIDATION_MODE = (
    config.validation_mode
    or to_boolean(os.getenv("VALIDATION_MODE", "False"))
    or to_boolean(os.getenv("AGENT_UTILITIES_TESTING", "False"))
)
DEFAULT_SSL_VERIFY = GET_DEFAULT_SSL_VERIFY()
DEFAULT_APPROVAL_TIMEOUT = config.approval_timeout
DEFAULT_MAX_CRON_LOG_ENTRIES = 50


TOOL_GUARD_MODE = config.tool_guard_mode
SENSITIVE_TOOL_PATTERNS = config.sensitive_tool_patterns

# Router/KG models: find models with can_route/can_kg flags, else fallback to lite
_router_model = next((m for m in config.chat_models if m.can_route), _lite_chat)
_kg_model = next((m for m in config.chat_models if m.can_kg), _lite_chat)
DEFAULT_ROUTER_MODEL = (
    _router_model.id if _router_model else None
) or DEFAULT_LITE_LLM_MODEL_ID

DEFAULT_GRAPH_PERSISTENCE_TYPE = config.graph_persistence_type
DEFAULT_GRAPH_PERSISTENCE_PATH = config.graph_persistence_path
DEFAULT_ENABLE_LLM_VALIDATION = config.enable_llm_validation
DEFAULT_ROUTING_STRATEGY = config.routing_strategy
DEFAULT_GRAPH_ROUTER_TIMEOUT = config.graph_router_timeout
DEFAULT_GRAPH_VERIFIER_TIMEOUT = config.graph_verifier_timeout
DEFAULT_ENABLE_KG_EMBEDDINGS = config.enable_kg_embeddings
DEFAULT_KG_BACKUPS = config.kg_backups
DEFAULT_KG_INGESTION_WORKERS = config.kg_ingestion_workers
DEFAULT_KG_LLM_CONCURRENCY = config.kg_llm_concurrency
DEFAULT_KG_MODEL_ID = (_kg_model.id if _kg_model else None) or DEFAULT_LITE_LLM_MODEL_ID
DEFAULT_KG_ANALYSIS_MAX_DEPTH = config.kg_analysis_max_depth
DEFAULT_KNOWLEDGE_GRAPH_SYNC_BACKGROUND = config.knowledge_graph_sync_background
DEFAULT_GRAPH_DIRECT_EXECUTION = config.graph_direct_execution

# --- Parallel Engine Defaults (CONCEPT:ORCH-1.25) ---
DEFAULT_MAX_PARALLEL_AGENTS = config.max_parallel_agents
DEFAULT_PARALLEL_BATCH_SIZE = config.parallel_batch_size
DEFAULT_SYNTHESIS_STRATEGY = config.synthesis_strategy
DEFAULT_SYNTHESIS_RATIO = config.synthesis_ratio
DEFAULT_AGENT_EXECUTION_TIMEOUT = config.agent_execution_timeout
DEFAULT_CIRCUIT_BREAKER_THRESHOLD = config.circuit_breaker_threshold
DEFAULT_ENABLE_PROGRESSIVE_SYNTHESIS = config.enable_progressive_synthesis

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

# Agent OS Architecture defaults (CONCEPT:OS-5.2)
DEFAULT_COGNITIVE_SCHEDULER_ENABLED = config.cognitive_scheduler_enabled
DEFAULT_MAX_CONCURRENT_AGENTS = config.max_concurrent_agents
DEFAULT_AGENT_TOKEN_QUOTA = config.agent_token_quota
DEFAULT_PREEMPTION_THRESHOLD_PCT = config.preemption_threshold_pct
DEFAULT_AGENT_POLICIES_PATH = config.agent_policies_path
DEFAULT_PERMISSIONS_SIGNING_KEY = config.permissions_signing_key
DEFAULT_SPECIALIST_REGISTRY_PATH = config.specialist_registry_path

# Innovation Framework defaults (CONCEPT:OS-5.2 through CONCEPT:OS-5.2)
DEFAULT_HOMEOSTATIC_DOWNGRADE = config.homeostatic_downgrade_enabled
DEFAULT_ADVERSARIAL_VERIFICATION = config.adversarial_verification
DEFAULT_MAINTENANCE_TOKEN_BUDGET = config.maintenance_token_budget
DEFAULT_MAINTENANCE_PRIORITY = config.maintenance_priority
DEFAULT_WATCHDOG_PATTERNS = config.watchdog_patterns


def _ensure_config_template():
    import json
    from pathlib import Path

    import platformdirs

    APP_NAME = "agent-utilities"
    APP_AUTHOR = "knuckles-team"

    override = os.environ.get("AGENT_UTILITIES_CONFIG_DIR")
    if override:
        cfg_dir = Path(override).expanduser()
    else:
        cfg_dir = Path(platformdirs.user_config_path(APP_NAME, APP_AUTHOR))

    cfg_file = cfg_dir / "config.json"
    if not cfg_file.exists():
        try:
            cfg_dir.mkdir(parents=True, exist_ok=True)
            with open(cfg_file, "w") as f:
                json.dump(config.model_dump(by_alias=False), f, indent=4)
        except Exception as e:
            import logging

            logging.getLogger(__name__).warning(f"Failed to write config template: {e}")


_ensure_config_template()
