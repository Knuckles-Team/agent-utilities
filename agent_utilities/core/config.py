#!/usr/bin/python
"""Configuration Management Module.

This module handles the loading and validation of agent settings from environment
variables and .env files using Pydantic Settings. It defines a centralized
AgentConfig class and exports default configuration constants used throughout
the agent-utilities package. Leverages XDG Standards for config file placement.
"""

import os
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import platformdirs
from pydantic import Field, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

# Re-export the dependency-free env accessor (config discipline). Kept in its own
# module so it stays importable while this module is still initializing — see
# agent_utilities/core/_env.py. Modules use `from agent_utilities.core.config
# import setting`.
from agent_utilities.core._env import setting  # noqa: F401

DEFAULT_DB_PATH = str(
    platformdirs.user_data_path("agent-utilities", "knuckles-team") / "graph_state"
)

# CONCEPT:AU-ORCH.execution.reserved-inference-slots — local-inference slots always kept free for the interactive path
# (the messaging responder + graph-os-spawned pydantic-ai agents, which share the default
# model). Background KG work is bounded to (capacity − this). A constant, not a knob: 1 is
# the correct universal default (config discipline — no flag for a one-correct-value).
RESERVED_INTERACTIVE_INSTANCES = 1

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
    # Hermetic tests never read a deployment ``.env`` (gitignored, absent in CI)
    # — pydantic-settings' ``env_file`` would otherwise pull host-specific values
    # (state-store DSN, auth) into unit tests that assert genuine defaults.
    if to_boolean(os.environ.get("AGENT_UTILITIES_TESTING", "false")):
        return None

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


_env_loaded = False


def _ensure_env_loaded():
    global _env_loaded
    if _env_loaded:
        return
    _env_loaded = True

    # Hermetic tests: never inherit a deployment ``.env`` or a host's
    # ``config.json``. Those are operational artifacts (the ``.env`` is
    # gitignored; ``config.json`` is host-specific) and are both absent in CI —
    # loading them locally makes unit tests that assert genuine defaults
    # (state-store DSN, auth, fuseki) fail only on a configured host. Skipping
    # them under the test harness makes the local run match CI exactly. Tests
    # set whatever env they need explicitly via the conftest / monkeypatch.
    if to_boolean(os.environ.get("AGENT_UTILITIES_TESTING", "false")):
        return

    try:
        from dotenv import load_dotenv

        _env_file = get_env_file()
        if _env_file:
            load_dotenv(_env_file)
    except ImportError:
        pass

    _load_xdg_json_config()
    _load_docker_secrets()


def _load_docker_secrets(secrets_dir: str = "/run/secrets") -> None:
    """Bridge docker/swarm secrets (``/run/secrets/<NAME>``) into ``os.environ``.

    Swarm/compose secrets mount each secret as a file under ``/run/secrets``.
    pydantic reads them for typed ``AgentConfig`` fields (``secrets_dir``), but
    :func:`agent_utilities.core._env.setting` and bare reads see only
    ``os.environ`` — so a docker-secret-injected ``OIDC_CLIENT_SECRET`` would be
    invisible to the client-credentials minter. Mirror each secret file into
    ``os.environ`` (``setdefault`` — an explicit spec/config env always wins) so a
    secret is read everywhere WITHOUT ever appearing as a literal in the
    inspectable service spec. CONCEPT:AU-OS.config.fleet-xdg-standardization — fleet config single-source.
    """
    if not os.path.isdir(secrets_dir):
        return
    try:
        names = os.listdir(secrets_dir)
    except OSError:
        return
    for name in names:
        if name in os.environ:  # explicit spec env / config.json already set it → wins
            continue
        path = os.path.join(secrets_dir, name)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                value = fh.read().strip()
        except OSError:
            continue
        if value:
            os.environ[name] = value


def load_config(*, reload: bool = False) -> None:
    """Load the agent-utilities configuration into the process environment.

    Public, idempotent entry point for the shared dotenv + XDG ``config.json``
    injection. Calling it makes ``~/.config/agent-utilities/config.json`` (or
    ``$AGENT_UTILITIES_CONFIG_DIR``) the single source of truth for every agent
    package: each key is upper-cased and written into ``os.environ`` (a real env
    var always wins), after which any ``config.setting(...)`` or ``os.getenv(...)``
    read sees it.

    Agent MCP entry points call this in place of ``load_dotenv(find_dotenv())``
    so the whole fleet resolves settings through one shared config rather than a
    per-package ``.env`` + scattered bare reads. Idempotent and safe to call
    repeatedly; pass ``reload=True`` to re-read after the file changed. Under the
    test harness it is a deliberate no-op (hermetic tests set their own env).

    CONCEPT:AU-OS.config.fleet-xdg-standardization — fleet XDG config standardization
    """
    global _env_loaded
    if reload:
        _env_loaded = False
    _ensure_env_loaded()


def _under_pytest() -> bool:
    """True during a pytest session.

    Used to keep the developer's XDG-default deployment ``config.json`` out of
    the test environment even if a test flips ``AGENT_UTILITIES_TESTING`` off (a
    few tests do, to exercise real-validation branches). ``PYTEST_CURRENT_TEST``
    is set by pytest while a test runs — exactly when such a config reload would
    leak host-specific values into ``os.environ`` and pollute later tests.
    """
    import sys

    return "PYTEST_CURRENT_TEST" in os.environ or "pytest" in sys.modules


def _load_xdg_json_config():
    import json
    from pathlib import Path

    import platformdirs

    APP_NAME = "agent-utilities"
    APP_AUTHOR = "knuckles-team"

    override = os.environ.get("AGENT_UTILITIES_CONFIG_DIR")
    # Hermetic tests never read the developer's XDG-default deployment
    # ``config.json`` (e.g. a homelab ``GRAPH_DB_URI=postgresql://…pggraph.arpa``
    # with neo4j/falkor mirror targets, ``KG_AUTH_REQUIRED``, a brain-guard
    # backend). config-loading injects those into ``os.environ`` and they would
    # override the unit suite's defaults — making tests fail on a dev box while
    # staying green in CI (which has no such file). Mirrors the ``.env`` skip in
    # ``get_env_file``. An explicit ``AGENT_UTILITIES_CONFIG_DIR`` (integration
    # fixtures) is still honored.
    if not override and (
        _under_pytest()
        or to_boolean(os.environ.get("AGENT_UTILITIES_TESTING", "false"))
    ):
        return
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


def _xdg_config_file():
    """Path to the XDG ``config.json`` (honors ``AGENT_UTILITIES_CONFIG_DIR``)."""
    from pathlib import Path

    import platformdirs

    override = os.environ.get("AGENT_UTILITIES_CONFIG_DIR")
    cfg_dir = (
        Path(override).expanduser()
        if override
        else Path(platformdirs.user_config_path("agent-utilities", "knuckles-team"))
    )
    return cfg_dir / "config.json"


def save_config_item(key: str, value) -> str:
    """Persist one config item to ``config.json`` AND live ``os.environ``, then reload.

    CONCEPT:AU-KG.storage.config-writeback — the write-back companion to the read-only XDG loader, so a
    config change made via the MCP/REST surfaces survives restart and applies live
    for settings read at call time (``config.setting`` / re-parsed fields). Returns
    the resolved env key. Engine-rebuild settings (GRAPH_BACKEND/…) update the value
    but need a restart to take effect — see the restart classifier.
    """
    import json
    from pathlib import Path

    cfg_file = _xdg_config_file()
    Path(cfg_file).parent.mkdir(parents=True, exist_ok=True)
    data: dict = {}
    if cfg_file.exists():
        try:
            data = json.loads(cfg_file.read_text())
        except Exception:
            data = {}
    data[key.lower()] = value
    cfg_file.write_text(json.dumps(data, indent=2, default=str))

    env_key = key.upper()
    if isinstance(value, list | dict):
        os.environ[env_key] = json.dumps(value)
    elif value is None:
        os.environ[env_key] = ""
    else:
        os.environ[env_key] = str(value)
    # Reload the live singleton (defined later in this module) so typed fields and
    # config.setting() pick up the new value immediately.
    _singleton = globals().get("config")
    if _singleton is not None:
        try:
            _singleton.reload()
        except Exception:  # noqa: BLE001 — persistence already succeeded
            pass
    return env_key


from pydantic import BaseModel


def _total_model_capacity(parallel_instances: int, max_parallel_calls: int) -> int:
    """Resolve a model's total parallel-call capacity (CONCEPT:AU-KG.compute.concurrency-controller-sizing).

    ``total_capacity = parallel_instances * max_parallel_calls`` — the number of
    in-flight LLM/embedding calls the model can serve at once: ``N`` vLLM
    instances behind one endpoint, each serving ``max_parallel_calls`` concurrent
    requests. Always at least ``1`` (unknown/misconfigured collapses to safe
    sequential behaviour, never zero-capacity).
    """
    return max(1, int(parallel_instances or 1) * int(max_parallel_calls or 1))


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
    """Number of parallel vLLM instances behind this model's ``base_url``. The
    per-instance concurrency is ``max_parallel_calls``; total parallel-call
    capacity is the product (see :pyattr:`total_capacity`)."""
    max_parallel_calls: int = 1
    """How many concurrent requests ONE vLLM instance of this model can serve
    (its per-instance concurrency, e.g. vLLM ``--max-num-seqs``). Default ``1``
    keeps callers sequential and is always safe. CONCEPT:AU-KG.compute.concurrency-controller-sizing."""
    max_concurrent_requests: int | None = None
    """Hard ceiling on the **aggregate** in-flight requests this model's *server*
    can serve safely — its real vLLM ``--max-num-seqs`` / KV-cache + (for a
    unified-memory box like the GB10) the shared-memory headroom (CONCEPT:AU-KG.compute.same-semantics-as).

    This is the ONE number the model SERVER's capacity dictates, NOT the local
    host's CPU count. It is deployment-varying (a GB10 ≠ a Pi ≠ a cluster) and
    cannot be auto-derived from the local box, so it is a legitimate explicit
    config (Configuration-discipline). When set it caps the SUM of every demand
    source hitting this endpoint (embeds + enrichment + orchestration) via the
    shared priority gate, so the client can never pile hundreds of concurrent
    requests onto the server and OOM it. When unset it falls back to
    ``max(total_capacity, MODEL_MAX_CONCURRENT_REQUESTS)`` (a conservative
    default). Set it to your server's ``--max-num-seqs`` (or just below)."""
    gpu_group: str | None = None
    """Optional shared-GPU group tag (CONCEPT:AU-KG.compute.pure-config-enumeration-fail). Models that physically
    share one GPU are grouped under one concurrency budget so their fan-out cannot
    jointly oversubscribe the device. Explicit tag wins; when unset the group
    defaults to the ``base_url`` host (see :meth:`Config.gpu_group`)."""
    can_route: bool = False
    can_kg: bool = False

    @property
    def total_capacity(self) -> int:
        """Total in-flight calls this model can serve = instances × per-instance.

        CONCEPT:AU-KG.compute.concurrency-controller-sizing — used by the shared concurrency controller to size the
        fan-out gate for this model.
        """
        return _total_model_capacity(self.parallel_instances, self.max_parallel_calls)


class EmbeddingModelConfig(BaseModel):
    id: str
    provider: str
    base_url: str | None = None
    api_key: str | None = None
    parallel_instances: int = 1
    """Number of parallel vLLM instances behind this embedding model's
    ``base_url``. Total parallel-call capacity is ``parallel_instances *
    max_parallel_calls`` (see :pyattr:`total_capacity`)."""
    max_parallel_calls: int = 1
    """How many concurrent embedding requests ONE vLLM instance of this model can
    serve (its per-instance concurrency). Default ``1`` keeps batch embedding
    sequential and is always safe. CONCEPT:AU-KG.compute.concurrency-controller-sizing."""
    max_concurrent_requests: int | None = None
    """Hard ceiling on the aggregate in-flight embedding requests this model's
    *server* can serve safely (CONCEPT:AU-KG.compute.same-semantics-as). Same semantics as
    :pyattr:`ChatModelConfig.max_concurrent_requests`: the SERVER's real capacity,
    capping the embedding fan-out so bulk embedding can never oversubscribe the
    endpoint. On a unified-memory box the embedder shares the GB10's memory with
    the generator — keep this conservative. Unset → ``max(total_capacity,
    MODEL_MAX_CONCURRENT_REQUESTS)``."""
    gpu_group: str | None = None
    """Optional shared-GPU group tag (CONCEPT:AU-KG.compute.pure-config-enumeration-fail). Tag this with the same
    value as a chat model that shares the physical GPU (e.g. both ``"gb10"``) so
    bulk embedding yields its concurrency to latency-sensitive chat under
    contention. Explicit tag wins; else defaults to the ``base_url`` host."""
    chunk_size: int = 768
    fallback: "EmbeddingModelConfig | None" = None
    """Optional automatic-failover endpoint (CONCEPT:AU-KG.enrichment.each-call-resolves-active). When the PRIMARY
    embedder (this config) is unreachable — its circuit breaker (CONCEPT:AU-ORCH.routing.load-shedding-backoff)
    is OPEN — embedding traffic is transparently re-routed to this fallback
    endpoint, and routed back automatically once the primary recovers. The fallback
    is a full ``EmbeddingModelConfig`` with its OWN ``base_url``, ``gpu_group``, and
    ``max_concurrent_requests``, so the capacity guard applies the FALLBACK's GPU
    budget while failed-over: point the primary at a dedicated embedder (e.g.
    ``gr1080-embed.arpa`` ``gpu_group="gr1080"``) and the fallback at a shared box
    (e.g. ``vllm-embed.arpa`` ``gpu_group="gb10"``) so fallback embeds share the
    GB10's joint budget with the generator and can never OOM it. A nested
    ``fallback`` here is ignored (single-level failover)."""

    @property
    def total_capacity(self) -> int:
        """Total in-flight embedding calls this model can serve.

        CONCEPT:AU-KG.compute.concurrency-controller-sizing — ``parallel_instances × max_parallel_calls``; used by
        the concurrency controller to fan out embedding batches.
        """
        return _total_model_capacity(self.parallel_instances, self.max_parallel_calls)


# Self-referential ``fallback`` field (CONCEPT:AU-KG.enrichment.each-call-resolves-active): config.py does not use
# ``from __future__ import annotations``, so rebuild the model to resolve the
# forward reference to ``EmbeddingModelConfig`` after the class is defined.
EmbeddingModelConfig.model_rebuild()


# _load_xdg_json_config() is now called dynamically via _ensure_env_loaded()


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

    # --- Messaging reach + agent KG layer (CONCEPT:AU-ECO.messaging.messaging-reach-service-governed–4.61) ---
    # Outbound/inbound messaging (Telegram/Slack/Teams/Mattermost/…). Tokens per backend
    # (e.g. TELEGRAM_BOT_TOKEN) auto-enable that backend; these tune routing + the agent.
    telegram_bot_token: str | None = Field(default=None, alias="TELEGRAM_BOT_TOKEN")
    messaging_default_platform: str = Field(
        default="telegram", alias="MESSAGING_DEFAULT_PLATFORM"
    )
    messaging_default_channel: str = Field(
        default="", alias="MESSAGING_DEFAULT_CHANNEL"
    )
    # CONCEPT:AU-ECO.messaging.universal-graph-agent — the universal graph agent a chat turn routes to. Defaults to the
    # dedicated "messaging-assistant" identity in code; set to route a chat turn to a
    # different named agent. Unresolved names still go through the full orchestration graph.
    messaging_agent: str = Field(default="", alias="MESSAGING_AGENT")
    messaging_claude_trigger: str = Field(
        default="/claude", alias="MESSAGING_CLAUDE_TRIGGER"
    )
    messaging_claude_model: str = Field(
        default="claude-sonnet-4-6", alias="MESSAGING_CLAUDE_MODEL"
    )
    messaging_local_model: str = Field(default="", alias="MESSAGING_LOCAL_MODEL")
    messaging_reactions: str = Field(default="1", alias="MESSAGING_REACTIONS")
    # Burst coalescing (CONCEPT:AU-ECO.messaging.burst-mode-coalescing): collapse a rapid run of messages into ONE reply.
    messaging_burst_window_s: str = Field(
        default="2.5", alias="MESSAGING_BURST_WINDOW_S"
    )
    messaging_burst_max_s: str = Field(default="12", alias="MESSAGING_BURST_MAX_S")
    # Post-conversation enrichment (CONCEPT:AU-ECO.messaging.post-conversation-enrichment): mine chats → KG concepts (opt-out).
    messaging_enrich: str = Field(default="1", alias="MESSAGING_ENRICH")
    # Surface goals / SDD specs from chats (CONCEPT:AU-ECO.messaging.surfaced, opt-out).
    messaging_goals: str = Field(default="1", alias="MESSAGING_GOALS")
    # Webhook push (CONCEPT:AU-ECO.messaging.telegram-webhook-receiver-started): set the PUBLIC base URL (served via tunnel/edge to a
    # LOCAL port) to switch from polling to instant webhook delivery; empty = polling.
    messaging_webhook_base_url: str = Field(
        default="", alias="MESSAGING_WEBHOOK_BASE_URL"
    )
    messaging_webhook_port: str = Field(default="8443", alias="MESSAGING_WEBHOOK_PORT")
    messaging_webhook_secret: str = Field(default="", alias="MESSAGING_WEBHOOK_SECRET")
    # Voice input (CONCEPT:AU-ECO.messaging.telegram-voice-note): transcribe voice notes to text via Whisper (opt-out).
    messaging_voice: str = Field(default="1", alias="MESSAGING_VOICE")
    messaging_voice_model: str = Field(default="base", alias="MESSAGING_VOICE_MODEL")
    # KG as a first-class default tool layer for every agent (opt-out).
    agent_kg_tools: str = Field(default="True", alias="AGENT_KG_TOOLS")

    # --- Ingestion sources (CONCEPT:AU-KG.query.vendor-agnostic-traversal web-fetch) ---
    # When set, ArchiveBox (a deployed web-archiving instance reached via the
    # archivebox-api MCP server) is preferred over a live crawl: the unified
    # web-fetch resolver serves the preserved snapshot (fast, no re-crawl,
    # archive-on-miss). Unset → crawl4ai (if installed) → requests+markitdown.
    # The presence of a URL is the on-signal; the credential lives with the MCP
    # server, so only this toggle is needed here.
    archivebox_url: str | None = Field(default=None, alias="ARCHIVEBOX_URL")

    # --- Graph / KG tuning knobs ---
    # Whole-workflow orchestration budget (ms). Lowered 20min→10min: engine RPC
    # hangs are now caught in seconds by the client's per-RPC timeout, so this is a
    # backstop for a wedged multi-agent run, not the primary hang detector. Kept
    # generous enough for long legitimate multi-step workflows; override per deploy.
    graph_timeout: str | None = Field(default="600000", alias="GRAPH_TIMEOUT")
    max_recursion_depth: str | None = Field(default="2", alias="MAX_RECURSION_DEPTH")
    routing_percentile: str | None = Field(default="50.0", alias="ROUTING_PERCENTILE")
    # Must match the embedding model's output dimension (768). The schema vector
    # column size is derived from this, so a mismatch breaks node inserts.
    kg_embedding_dim: str | None = Field(default="768", alias="KG_EMBEDDING_DIM")

    # Single dev switch that disables ALL KG background daemons (maintenance
    # scheduler: enrichment/reconcile/file-watch/hygiene/task-reaper + the
    # embedding backfill). Production keeps them all on; this replaces the old
    # per-daemon KG_*_DAEMON env toggles (CONCEPT:EG-KG.storage.nonblocking-checkpoint, config discipline).
    kg_dev_mode: bool = Field(default=False, alias="KG_DEV_MODE")

    # Safety override: by default an unscoped Cypher query (no id/label, no
    # parseable WHERE) refuses to enumerate the whole graph and returns []
    # (ORCH-1.39). Set this for a rare, deliberate full enumeration. Off by
    # default so a buggy unscoped query can never silently scan the whole graph.
    kg_allow_full_scan: bool = Field(default=False, alias="KG_ALLOW_FULL_SCAN")

    # --- Observability / usage analytics (CONCEPT:AU-OS.observability.usage-analytics-store / ECO-4.40 / OS-5.31) ---
    # Backend for the usage/cost/session fact store. Zero-config default is a
    # per-host SQLite+FTS5 file (no external deps); "postgres" / "duckdb" promote
    # to enterprise-scale shared backends via the same UsageBackend interface.
    usage_db_backend: str = Field(default="sqlite", alias="USAGE_DB_BACKEND")
    # Optional explicit path/URI for the usage store. Empty = derive from the
    # state_store seam (per-host data dir for sqlite, STATE_DB_URI for postgres).
    usage_db_uri: str | None = Field(default=None, alias="USAGE_DB_URI")
    # Master switch for runtime usage instrumentation (plane B). Default-on but
    # best-effort: a recorder failure never breaks a graph run.
    usage_tracking_enabled: bool = Field(default=True, alias="USAGE_TRACKING_ENABLED")
    # LiteLLM pricing source. Empty keeps the bundled offline fallback only
    # (fully functional with no network); the daemon refreshes from this URL.
    pricing_litellm_url: str = Field(
        default=(
            "https://raw.githubusercontent.com/BerriAI/litellm/main/"
            "model_prices_and_context_window.json"
        ),
        alias="PRICING_LITELLM_URL",
    )

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

    # --- Parallel-call capacity resolution (CONCEPT:AU-KG.compute.concurrency-controller-sizing) ---

    def _resolve_model_config(
        self, model: str | None = None
    ) -> "ChatModelConfig | EmbeddingModelConfig | None":
        """Resolve a model id/role to its config object (CONCEPT:AU-KG.compute.concurrency-controller-sizing).

        ``model`` may be a model id (matched against both chat and embedding
        registries), a role (``"chat"``/``"default"``, ``"lite"``, ``"super"``,
        ``"embedding"``/``"embed"``), or ``None`` (→ default chat model). Returns
        ``None`` when nothing matches.
        """
        cfg: ChatModelConfig | EmbeddingModelConfig | None = None
        key = (model or "").strip().lower()
        if key in ("", "chat", "default"):
            cfg = self.default_chat_model
        elif key == "lite":
            cfg = self.lite_chat_model
        elif key == "super":
            cfg = self.super_chat_model
        elif key in ("embedding", "embed"):
            cfg = self.default_embedding_model
        elif key in ("embedding:fallback", "embed:fallback", "embedding-fallback"):
            # The automatic-failover endpoint (CONCEPT:AU-KG.enrichment.each-call-resolves-active): resolve it as a
            # first-class model key so the WHOLE capacity guard — server_ceiling,
            # adaptive capacity, gpu_group budget (CONCEPT:AU-KG.ingest.keys-off) — keys off the
            # FALLBACK endpoint's config (its own gpu_group / max_concurrent_requests)
            # while failed-over, so fallback embeds inherit the shared GPU's joint
            # budget and can't OOM it.
            primary = self.default_embedding_model
            cfg = primary.fallback if primary is not None else None
        else:
            for m in self.chat_models:
                if m.id == model:
                    cfg = m
                    break
            if cfg is None:
                for em in self.embedding_models:
                    if em.id == model:
                        cfg = em
                        break
        return cfg

    def model_capacity(self, model: str | None = None) -> int:
        """Resolve a model's total parallel-call capacity by id/role.

        CONCEPT:AU-KG.compute.concurrency-controller-sizing. ``model`` may be a model id (matched against both chat
        and embedding registries), one of the roles ``"chat"``/``"default"``,
        ``"lite"``, ``"super"``, ``"embedding"``/``"embed"``, or ``None`` (→
        default chat model). Unknown/unconfigured models resolve to ``1`` — safe
        sequential behaviour, never zero.
        """
        cfg = self._resolve_model_config(model)
        return cfg.total_capacity if cfg is not None else 1

    def model_max_concurrent_requests(self, model: str | None = None) -> int | None:
        """Resolve a model's explicit server-capacity ceiling, if configured.

        CONCEPT:AU-KG.compute.same-semantics-as. Returns the model's ``max_concurrent_requests`` (the
        server's real ``--max-num-seqs`` / safe in-flight budget) by id or role,
        or ``None`` when unset/unknown so the caller applies the conservative
        default. A non-positive/garbage value resolves to ``None`` (no hard cap
        from config — fall back to the default), never zero.
        """
        cfg = self._resolve_model_config(model)
        if cfg is None:
            return None
        val = getattr(cfg, "max_concurrent_requests", None)
        if val is None:
            return None
        try:
            v = int(val)
        except (TypeError, ValueError):
            return None
        return v if v > 0 else None

    def model_endpoint(self, model: str | None = None) -> tuple[str | None, str | None]:
        """Resolve a model id/role to its ``(model_id, base_url)`` (CONCEPT:AU-KG.compute.surfaces-universal-latency-signal).

        Used by the adaptive concurrency controller to derive a model's vLLM
        ``/metrics`` URL and the ``model_name`` label its Prometheus gauges carry.
        Returns ``(None, None)`` when the model is unknown/unconfigured.
        """
        cfg = self._resolve_model_config(model)
        if cfg is None:
            return (None, None)
        return (cfg.id, cfg.base_url)

    def gpu_group(self, model: str | None = None) -> str | None:
        """Resolve a model's shared-GPU group key (CONCEPT:AU-KG.compute.pure-config-enumeration-fail).

        Models that share one physical GPU are grouped so a per-GPU budget can cap
        their *joint* concurrency (e.g. embedding must leave headroom for chat on a
        shared unified-memory device). Resolution order:

        1. The model's explicit ``gpu_group`` tag, if set — this is the only way to
           group models served from **different** endpoints onto one GPU (our case:
           tag both ``bge-m3`` @ ``vllm-embed.arpa`` and ``qwen3.6-27b`` @
           ``vllm.arpa`` ``gpu_group="gb10"``).
        2. Otherwise the ``base_url`` host (netloc), so same-endpoint models group
           automatically with zero config.
        3. ``None`` when the model is unknown or has neither a tag nor a base_url —
           the caller then applies no budget (per-model behaviour, no regression).
        """
        cfg = self._resolve_model_config(model)
        if cfg is None:
            return None
        tag = getattr(cfg, "gpu_group", None)
        if tag:
            return str(tag).strip().lower() or None
        base_url = getattr(cfg, "base_url", None)
        if not base_url:
            return None
        from urllib.parse import urlsplit

        netloc = urlsplit(str(base_url)).netloc.strip().lower()
        return netloc or None

    def embedding_capacity(self) -> int:
        """Total parallel-call capacity of the default embedding model.

        CONCEPT:AU-KG.compute.concurrency-controller-sizing — convenience for the embedding fan-out path.
        """
        em = self.default_embedding_model
        return em.total_capacity if em is not None else 1

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

    # --- Knowledge Graph identity enforcement (CONCEPT:AU-OS.identity.authenticated-identity-enforcement) ---

    kg_auth_required: bool = Field(default=False, alias="KG_AUTH_REQUIRED")
    """Require a server-validated JWT identity for Knowledge Graph access.
    When True: HTTP requests without a valid Bearer token are rejected (401),
    caller-supplied ``_actor``/``_roles``/``_tenant`` tool kwargs are ignored
    (server-side identity only), and stdio MCP derives identity from a
    validated ``KG_AUTH_TOKEN`` or falls back to a read-only system actor.
    Default False preserves the legacy honor-system behaviour (with a one-time
    startup warning)."""

    kg_served_profile: bool = Field(default=True, alias="KG_SERVED_PROFILE")
    """Apply the fail-closed served-security profile when serving a network MCP
    transport (streamable-http/sse): refuse to start without a JWT validator and
    auto-enable auth + enforcement (CONCEPT:AU-OS.identity.authenticated-identity-enforcement). Set ``KG_SERVED_PROFILE=0``
    to serve a network transport WITHOUT enforced identity (local dev only)."""

    kg_auth_token: str | None = Field(default=None, alias="KG_AUTH_TOKEN")
    """Optional JWT used to mint the process identity for stdio MCP servers
    (no Authorization header exists on stdio). Validated against
    ``AUTH_JWT_JWKS_URI`` exactly like a gateway request. Only consulted when
    ``KG_AUTH_REQUIRED`` is on."""

    kg_acl_default_allow: bool = Field(default=False, alias="KG_ACL_DEFAULT_ALLOW")
    """Escape hatch for fail-closed permissioning. With ``KG_BRAIN_ENFORCE``
    on, nodes WITHOUT an ACL are denied by default (fail closed); setting this
    True restores allow-on-missing-ACL while keeping exception→deny. Ignored
    when enforcement is off (legacy default-allow)."""

    # --- Fleet events webhook ingress (CONCEPT:AU-OS.config.fleet-event-ingress) ---

    fleet_events_token: str | None = Field(default=None, alias="FLEET_EVENTS_TOKEN")
    """Optional shared secret for the ``POST /api/fleet/events`` webhook
    receiver. Monitoring senders (Prometheus Alertmanager, Uptime Kuma,
    Portainer) cannot mint JWTs, so when this is set every event POST must
    carry a matching ``X-Fleet-Events-Token`` header or it is rejected (401).
    Default ``None`` = no token required (the endpoint is then as open as the
    other fleet routes; the OS-5.14 identity middleware still applies when
    ``KG_AUTH_REQUIRED`` is on)."""

    # --- Gateway middle-tier hardening (CONCEPT:AU-OS.observability.no-op-without-metrics) ---

    gateway_metrics: bool = Field(default=True, alias="GATEWAY_METRICS")
    """Expose Python-tier Prometheus metrics on the gateway: a pure-ASGI
    middleware recording ``agent_utilities_gateway_*`` series (request totals,
    duration histogram, in-flight gauge, rate-limit/breaker counters) plus a
    ``GET /metrics`` endpoint (exempt from auth — scrapers cannot mint JWTs).
    Requires the optional ``metrics`` extra (``prometheus-client``); without it
    the middleware degrades to a no-op and ``/metrics`` returns a placeholder."""

    gateway_rate_limit: float = Field(default=0.0, alias="GATEWAY_RATE_LIMIT")
    """Per-tenant sustained request rate (requests/second) enforced by the
    gateway token-bucket middleware. ``0`` (default) disables rate limiting.
    Bucket key: ActorContext tenant → authenticated actor id → client IP.
    Buckets are in-memory and PER-PROCESS: with N workers/replicas the
    effective limit is N× this value (see docs/architecture/gateway_scaling.md)."""

    gateway_rate_burst: float = Field(default=0.0, alias="GATEWAY_RATE_BURST")
    """Token-bucket burst capacity (max requests served instantly from a full
    bucket). ``0`` (default) derives 2× ``GATEWAY_RATE_LIMIT``."""

    gateway_workers: int = Field(default=1, alias="GATEWAY_WORKERS")
    """Number of gateway worker processes. Default ``1`` preserves the
    single-process behaviour (in-process KG daemon, one event loop). With
    ``>1`` the server pre-forks workers sharing one listen socket; exactly ONE
    worker wins the KG host flock and runs the consolidated daemon/ticks while
    the rest self-heal to clients. Metrics and rate-limit buckets are
    per-worker. Ignored when the terminal UI is enabled or under pytest."""

    engine_breaker_threshold: int = Field(default=5, alias="ENGINE_BREAKER_THRESHOLD")
    """Consecutive engine connect/timeout failures before the epistemic-graph
    client circuit breaker opens (fast, typed ``EngineCircuitOpenError``
    instead of hammering a dead engine). ``0`` disables the breaker."""

    engine_breaker_cooldown: float = Field(
        default=15.0, alias="ENGINE_BREAKER_COOLDOWN"
    )
    """Seconds an open engine circuit breaker waits before allowing a single
    half-open probe; the probe's outcome closes or re-opens the circuit."""

    # --- MCP multiplexer child resilience (CONCEPT:AU-ECO.mcp.profile-differences-from-client) ---

    mcp_child_max_concurrency: int = Field(default=8, alias="MCP_CHILD_MAX_CONCURRENCY")
    """Maximum in-flight tool calls per multiplexer child server. Excess calls
    queue (bounded by ``MCP_CHILD_QUEUE_TIMEOUT``) instead of piling onto the
    child unbounded. Per-server override: the ``max_concurrency`` key on the
    server's ``mcp_config.json`` entry. ``0`` disables the limit."""

    mcp_child_queue_timeout: float = Field(
        default=30.0, alias="MCP_CHILD_QUEUE_TIMEOUT"
    )
    """Seconds a tool call may wait for a free per-child concurrency slot
    before failing with the typed ``MCPChildBusyError`` (no silent hangs).
    Per-server override: the ``queue_timeout`` key on the server entry."""

    mcp_child_pool_size: int = Field(default=1, alias="MCP_CHILD_POOL_SIZE")
    """Session-pool size for remote (streamable-http/SSE) multiplexer
    children: N independent connections per child, round-robin dispatched,
    enabling parallel in-flight calls. Default 1 preserves the historical
    one-connection resource profile. Stdio children are single-pipe and
    always keep exactly one session. Per-server override: the ``pool_size``
    key on the server entry."""

    mcp_child_max_restarts: int = Field(default=5, alias="MCP_CHILD_MAX_RESTARTS")
    """How many automatic restarts a crashed multiplexer child may consume
    within ``MCP_CHILD_RESTART_WINDOW`` before it is marked ``failed`` (calls
    then fail fast with the typed ``MCPChildUnavailableError`` instead of
    retry-looping forever). ``0`` disables auto-restart entirely."""

    mcp_child_restart_window: float = Field(
        default=300.0, alias="MCP_CHILD_RESTART_WINDOW"
    )
    """Sliding window (seconds) over which ``MCP_CHILD_MAX_RESTARTS`` is
    counted. Restarts older than the window are forgiven, so a child that
    crashes rarely keeps restarting indefinitely while a crash-looping child
    is parked as ``failed``."""

    mcp_child_breaker_threshold: int = Field(
        default=5, alias="MCP_CHILD_BREAKER_THRESHOLD"
    )
    """Consecutive transport failures/timeouts on one multiplexer child
    before its circuit breaker opens (fast, typed
    ``MCPChildCircuitOpenError`` instead of hammering a dead child). ``0``
    disables the breaker. Per-server override: ``breaker_threshold``."""

    mcp_child_breaker_cooldown: float = Field(
        default=15.0, alias="MCP_CHILD_BREAKER_COOLDOWN"
    )
    """Seconds an open per-child circuit breaker waits before allowing a
    single half-open probe call; the probe's outcome closes or re-opens the
    circuit. Per-server override: ``breaker_cooldown``."""

    # --- MCP multiplexer dynamic tool gateway (CONCEPT:AU-ECO.multiplexer.tool-gateway-catalog) ---

    mcp_multiplexer_mode: str = Field(default="eager", alias="MCP_MULTIPLEXER_MODE")
    """Tool-exposure strategy for the mcp-multiplexer. ``eager`` (default)
    spawns every configured child at boot and exposes all aggregated tools —
    the historical behaviour. ``dynamic`` boots exposing only the meta-tools
    (``find_tools``/``load_tools``/``unload_tools``/``multiplexer_status``)
    plus the always-on children; child servers are then spawned lazily and
    their tools mounted on demand via ``load_tools``, with a
    ``notifications/tools/list_changed`` sent to the client. This keeps a
    client's visible tool count tiny when the fleet has hundreds of tools."""

    mcp_dynamic_always_on: list[str] = Field(
        default_factory=lambda: ["graph-os"], alias="MCP_DYNAMIC_ALWAYS_ON"
    )
    """Child server names mounted at boot in ``dynamic`` mode (in addition to
    the meta-tools). Defaults to the knowledge-graph server (``graph-os``) so
    ``find_tools`` can rank candidate tools semantically out of the box. Any
    name here must also exist in the active ``mcp_config.json``."""

    mcp_dynamic_top_k: int = Field(default=8, alias="MCP_DYNAMIC_TOP_K")
    """Default number of ranked tool candidates ``find_tools`` returns when the
    caller does not specify ``top_k``. Kept small so the discovery result is
    itself cheap to read; callers can request more explicitly."""

    # --- OIDC / OAuth 2.0 Delegation (CONCEPT:AU-ECO.messaging.native-backend-abstraction) ---

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

    # --- Vault Secrets Backend (CONCEPT:AU-OS.config.secrets-authentication) ---

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
    # Knowledge-graph backend selection (mirrors create_backend resolution).
    # epistemic-graph is the ONE database — the authority that does compute, cache,
    # semantic and durable persistence in a single engine. The out-of-box default is
    # ``epistemic_graph`` (the self-contained engine, zero-infra). Set ``fanout`` to
    # add MIRRORS (Postgres/Neo4j/FalkorDB/Ladybug): the engine remains the authority
    # serving every read, and each write fans out losslessly to the mirrors.
    graph_backend: str = Field(default="epistemic_graph", alias="GRAPH_BACKEND")
    graph_db_uri: str | None = Field(default=None, alias="GRAPH_DB_URI")
    # Mirrors (CONCEPT:AU-KG.backend.mirror-health-repair). When GRAPH_BACKEND=fanout, the graph is served from
    # ONE authority (the engine) and every write is mirrored, losslessly, to the named
    # mirror connections. ``graph_authority`` names the source-of-truth connection (a
    # ``kg_connections`` name, or a bare backend type like ``epistemic_graph``/``age``);
    # ``graph_mirror_targets`` is the JSON list of mirror connection names. Both resolve
    # against ``kg_connections`` (CONCEPT:AU-KG.backend.multi-connection-registry), so the DSN/host/creds live in one
    # place. Zero-infra default is unchanged: mirrors are only built when configured.
    graph_authority: str = Field(default="epistemic_graph", alias="GRAPH_AUTHORITY")
    graph_mirror_targets: list[str] | None = Field(
        default=None, alias="GRAPH_MIRROR_TARGETS"
    )
    # Ingest task-queue selection (CONCEPT:AU-KG.backend.selectable-queue-backend): which durable queue carries
    # KG ingest tasks. Unset (default) = auto: ``postgres`` when ``state_db_uri``
    # is set, else the zero-infra per-host ``sqlite`` file — mirroring the
    # state-externalization convention. An EXPLICIT value is a hard contract:
    # ``kafka``/``postgres`` raise at startup when unreachable (never a silent
    # SQLite degrade). Values: sqlite | postgres | kafka.
    task_queue_backend: str | None = Field(default=None, alias="TASK_QUEUE_BACKEND")
    # Partition count ensured on the ``kg_tasks`` topic at startup when the
    # kafka task queue is selected (CONCEPT:AU-KG.backend.keyed-ingest-partitions). Grow-only: raising it adds
    # partitions idempotently; an existing topic is NEVER shrunk. Bounds the
    # max parallelism of the ``kg-ingest`` consumer group.
    kg_tasks_partitions: int = Field(default=6, alias="KG_TASKS_PARTITIONS")
    # Queue-driven agent dispatch (CONCEPT:AU-ORCH.dispatch.queue-agent-dispatch): how agent turns (goal
    # runs / orchestrator jobs) are dispatched. ``inline`` (default) keeps the
    # existing in-process execution exactly as-is; ``queue`` publishes a
    # session-keyed AgentTurnEnvelope onto the agent_turns queue (transport
    # follows TASK_QUEUE_BACKEND/auto) and returns a job handle — any host
    # running ``agent-dispatch-worker`` executes it, so the scheduler tier
    # scales horizontally and sessions are not pinned to their birth host.
    agent_dispatch_backend: str = Field(
        default="inline", alias="AGENT_DISPATCH_BACKEND"
    )
    # Partitions ensured on the ``agent_turns`` topic when the kafka transport
    # carries dispatched agent turns (CONCEPT:AU-ORCH.dispatch.queue-agent-dispatch). Grow-only, like
    # KG_TASKS_PARTITIONS. Bounds agent-dispatch consumer-group parallelism —
    # i.e. how many sessions can execute concurrently across the worker fleet.
    agent_turns_partitions: int = Field(default=6, alias="AGENT_TURNS_PARTITIONS")
    # Durable-state externalization (CONCEPT:AU-OS.state.unified-durable-state-externalization): ONE flag selects where the
    # platform's durable state lives — durable-execution checkpoints, sessions/
    # turns/goals, and the KG task queue. Unset (default) keeps the zero-infra
    # per-host SQLite files; a postgresql:// URI moves them all onto a shared
    # Postgres (one psycopg pool) so a second host can safely participate and
    # the gateway becomes stateless.
    state_db_uri: str | None = Field(default=None, alias="STATE_DB_URI")
    # Max connections in the shared state-store pool (min is always 1).
    state_db_pool_size: int = Field(default=8, alias="STATE_DB_POOL_SIZE")
    # Golden-loop breadth ingest roots (CONCEPT:AU-KG.query.vendor-agnostic-traversal): comma-separated paths the
    # one-shot ``loop`` cycle (and the 60-min daemon) auto-ingests — OSS
    # libraries and code repos — so evolution runs end-to-end with no manual
    # ingest. Deployment-specific (set in ``.env``); empty ⇒ breadth is a no-op.
    kg_breadth_library_roots: str = Field(default="", alias="KG_BREADTH_LIBRARY_ROOTS")
    kg_breadth_repo_roots: str = Field(default="", alias="KG_BREADTH_REPO_ROOTS")
    # Loop-engine (autonomous research) parameters. Typed config replaces the
    # scattered bare env reads (CONCEPT:AU-KG.query.vendor-agnostic-traversal). The loop-enable + stage flags are
    # KG_LOOP*; the separate governed auto-merge gate keeps KG_GOLDEN_AUTO_MERGE.
    kg_loop: bool = Field(default=False, alias="KG_LOOP")
    kg_loop_distill: bool = Field(default=False, alias="KG_LOOP_DISTILL")
    # Opt-in (external scholarx calls cost): the intake stage discovers + ingests
    # research papers (LLM concept/fact extraction) at the front of the unified
    # research-intelligence cycle, so the matcher then compares the fresh papers
    # against the ecosystem. Caller-supplied ``papers`` always run regardless.
    # (CONCEPT:AU-KG.research.research-intelligence-loop)
    kg_loop_discover: bool = Field(default=False, alias="KG_LOOP_DISCOVER")
    # On by default: the breadth stage auto-ingests the ecosystem so ``assimilate``
    # has the codebase capability map to compare research against. With no
    # KG_BREADTH_* roots set it self-configures from the XDG workspace.yml, so the
    # default is zero-config; content-addressed ingest makes re-runs cheap. Set
    # KG_LOOP_BREADTH=0 to opt out. (CONCEPT:AU-KG.query.vendor-agnostic-traversal)
    kg_loop_breadth: bool = Field(default=True, alias="KG_LOOP_BREADTH")
    kg_loop_standardize: bool = Field(default=False, alias="KG_LOOP_STANDARDIZE")
    # Discovery-flywheel mining pass (CONCEPT:AU-KG.evolution.mining-flywheel) — runs the
    # engine's graph_mine (associate/anomaly) + graph_learn (fit/predict) surfaces over
    # the KG's concept/capability/article nodes each cycle, writing back typed
    # :AssociationRule/:Anomaly/:PredictedEdge nodes for the evolution flywheel to
    # consume (propose-only — never auto-merges). Default ON: each sub-step is
    # independently best-effort and degrades to an empty/no-op result on a
    # no-mining engine build, so it's safe to leave on everywhere.
    kg_loop_mine_discovery: bool = Field(default=True, alias="KG_LOOP_MINE_DISCOVERY")
    # CONCEPT:AU-OS.config.autonomous-spec-develop-off — autonomous spec→develop. OFF by default = review-first: a
    # distilled spec is persisted as a :SpecProposal in ``pending_review`` and HOLDS
    # for Claude/human approval (graph_loops action=review) before any develop Loop
    # is created. Turning this ON lets the 24/7 loop auto-advance specs through the
    # ``spec_promotion`` ActionPolicy gate (still approval_required by default, so it
    # only auto-develops where an operator has explicitly relaxed that tier).
    kg_loop_auto_develop: bool = Field(default=False, alias="KG_LOOP_AUTO_DEVELOP")
    kg_golden_auto_merge: bool = Field(default=False, alias="KG_GOLDEN_AUTO_MERGE")
    kg_golden_merge_threshold: float | None = Field(
        default=None, alias="KG_GOLDEN_MERGE_THRESHOLD"
    )
    # Evolution→branch bridge (CONCEPT:AU-AHE.harness.evolution-branch-bridge): root directory the
    # LocalBranchPublisher creates fresh git worktrees under when publishing a
    # promoted proposal as a reviewable local branch. Empty (default) resolves
    # to ``data_dir()/evolution_worktrees`` — NEVER the canonical checkout's
    # working tree.
    evolution_worktree_root: str = Field(default="", alias="EVOLUTION_WORKTREE_ROOT")
    kg_loop_interval: float = Field(default=3600.0, alias="KG_LOOP_INTERVAL")
    kg_loop_topics: int = Field(default=5, alias="KG_LOOP_TOPICS")
    # CONCEPT:AU-KG.research.scholarx-rss-research-feed — ScholarX RSS research-feed loop that grades and fetches new papers.
    # A recurring schedule
    # that grades incoming RSS items (keyword taxonomy + ConceptMatcher novelty),
    # skips already-seen items, and enqueues a prioritized full-paper fetch+ingest
    # only for the high-graded ones. Default-ON (it no-ops safely without ScholarX
    # / network); set KG_RESEARCH_FEED=0 to disable the autonomous fetching.
    kg_research_feed: bool = Field(default=True, alias="KG_RESEARCH_FEED")
    kg_research_feed_interval: float = Field(
        default=1800.0, alias="KG_RESEARCH_FEED_INTERVAL"
    )
    # CONCEPT:AU-KG.ingest.rss-feed-connector — native RSS/Atom feed URLs (comma-separated) the zero-infra
    # `rss` connector ingests through the unified world-model gate. This is the SEED;
    # feeds added at runtime via graph_feeds live as :FeedSource nodes in the KG and
    # are swept too. Empty by default (a deployment opts in its feeds).
    kg_rss_feeds: str = Field(default="", alias="KG_RSS_FEEDS")
    # SAI factory self-specialization (CONCEPT:AU-AHE.harness.sai-controller). LLM-free, bounded, and
    # propose-only (it only persists a SaiFactoryCycle metrics node — nothing is
    # merged or deployed), and a *no-op when there is too little transition history*,
    # so it costs nothing on an idle system. Like the anomaly consumer, that makes it
    # safe to run natively ⇒ ON by default (set KG_SAI_FACTORY=0 to disable). The tick
    # grounds a learned world model in persisted WorldModelTransition history and
    # specializes its config; the same loop is reachable on demand via
    # graph_analyze(action='specialize') through the gateway.
    kg_sai_factory: bool = Field(default=True, alias="KG_SAI_FACTORY")
    kg_sai_factory_interval: float = Field(
        default=3600.0, alias="KG_SAI_FACTORY_INTERVAL"
    )
    # Failure-driven evolution — opt-in, off by default; pulls failures from
    # Langfuse into failure-gap topics the golden loop remediates (CONCEPT:AU-AHE.harness.failure-evolution).
    kg_failure_evolution: bool = Field(default=False, alias="KG_FAILURE_EVOLUTION")
    kg_failure_evolution_interval: float = Field(
        default=3600.0, alias="KG_FAILURE_EVOLUTION_INTERVAL"
    )
    kg_failure_evolution_window: float = Field(
        default=86400.0, alias="KG_FAILURE_EVOLUTION_WINDOW"
    )
    kg_failure_regression_dataset: bool = Field(
        default=False, alias="KG_FAILURE_REGRESSION_DATASET"
    )
    # DSPy optimization sweep (CONCEPT:AU-AHE.optimization.candidate-replaces-incumbent-only) — opt-in, off by default because each
    # pass runs an LLM-gated DSPy compile per target. The scheduled twin of the
    # `graph_orchestrate action=optimize_component` MCP action: a daemon tick periodically
    # optimizes the self-supervised targets (extraction / concept_match / routing) and
    # records propose-only optimization trajectories (auto-apply stays gated, like
    # KG_GOLDEN_AUTO_MERGE).
    kg_dspy_optimization: bool = Field(default=False, alias="KG_DSPY_OPTIMIZATION")
    kg_dspy_optimization_interval: float = Field(
        default=3600.0, alias="KG_DSPY_OPTIMIZATION_INTERVAL"
    )
    # Agent-facing auto-apply gate (CONCEPT:AU-AHE.harness.hardening-transparency-surface) — the high-impact half of the
    # hardening loop. A DSPy-optimized *system prompt* that beats baseline on its
    # agent's eval-corpus slice is only written to source (StructuredPrompt.save) when
    # this is True; otherwise the cycle is **propose-only / shadow** — it records a
    # queryable ``ProposedPromptChange`` for human/Claude review and leaves the live
    # prompt untouched. Default OFF (mirrors KG_GOLDEN_AUTO_MERGE): a prompt rewrite is
    # never silent. ``should_promote`` still gates even when this is enabled.
    kg_agent_auto_apply: bool = Field(default=False, alias="KG_AGENT_AUTO_APPLY")
    # PerformanceAnomaly consumer (CONCEPT:AU-AHE.optimization.performance-anomaly-consumer) — drains unconsumed
    # PerformanceAnomaly nodes into failure_gap topics for the golden loop.
    # Default ON: it is LLM-free, bounded, and propose-only (it writes topic
    # nodes; nothing merges without the AHE-3.20 governed auto-merge chain).
    kg_anomaly_consumer: bool = Field(default=True, alias="KG_ANOMALY_CONSUMER")
    # Interval (s) for the leaked-community-tenant GC tick (Phase A2).
    kg_tenant_gc_interval: float = Field(default=300.0, alias="KG_TENANT_GC_INTERVAL")

    kg_engine_pool_size: int = Field(default=8, alias="KG_ENGINE_POOL_SIZE")
    """Max warm per-tenant engine clients kept resident in one process
    (CONCEPT:AU-KG.sharding.elastic-over-kg-shard). The elastic layer over KG-2.58 shard routing: only the N
    most-recently-used tenant graphs stay warm; cold ones are evicted (the
    durable L3 mirror keeps them) and re-hydrated on the next access.

    Default 8 (was 0): per-use construction built a fresh background thread +
    event loop + socket + ``tenants.create`` round-trip on EVERY engine access — a
    connection-setup storm under load. A small warm set amortizes that; a
    single-tenant deployment simply keeps its one graph warm. Eviction is LRU and
    bounded; set ``KG_ENGINE_POOL_DROP_ON_EVICT=1`` to also unload the evicted
    tenant's graph from the engine to reclaim L1 memory. ``0`` restores the old
    per-use behavior."""

    kg_engine_pool_drop_on_evict: bool = Field(
        default=False, alias="KG_ENGINE_POOL_DROP_ON_EVICT"
    )
    """When a tenant is evicted from the engine pool (CONCEPT:AU-KG.sharding.elastic-over-kg-shard), also
    unload its named graph from the engine process to reclaim L1 memory
    (``GraphComputeEngine.drop_graph``). **Only safe when data is durably
    mirrored to L3** (the tiered backend), which re-hydrates on next access;
    otherwise the in-memory graph is lost. Default off (eviction only closes the
    client)."""
    # Fuseki ontology distribution (CONCEPT:AU-KG.ontology.authoritative-tbox) — opt-in daemon tick that
    # pushes the bundled ontology modules to an Apache Jena Fuseki triplestore
    # (KG-2.6 distribution, operationalized). Off by default because a Fuseki
    # deployment is optional infrastructure.
    kg_fuseki_publish: bool = Field(default=False, alias="KG_FUSEKI_PUBLISH")
    kg_fuseki_endpoint: str | None = Field(default=None, alias="KG_FUSEKI_ENDPOINT")
    """Fuseki server URL (e.g. ``http://jena_fuseki:3030``). ``None`` defers to
    the publisher's own resolution (``FUSEKI_ENDPOINT`` env, then localhost)."""
    kg_fuseki_publish_interval: float = Field(
        default=3600.0, alias="KG_FUSEKI_PUBLISH_INTERVAL"
    )
    # Execution-time workflow ontology gate (CONCEPT:AU-ORCH.execution.ontology-validation-execution-path) — SHACL-validate
    # a stored WorkflowDefinition before dispatch. Default ON: it is cheap,
    # LLM-free, and refuses malformed definitions before they burn agent runs.
    kg_workflow_shape_gate: bool = Field(default=True, alias="KG_WORKFLOW_SHAPE_GATE")

    # --- Autonomy control plane (CONCEPT:AU-OS.deployment.fleet-lifecycle-control — OS-5.27) ---

    action_policy_path: str = Field(default="", alias="ACTION_POLICY_PATH")
    """Path to the operational ActionPolicy YAML (CONCEPT:AU-OS.deployment.fleet-lifecycle-control). Empty
    (default) resolves to the shipped conservative policy
    (``deploy/action-policy.default.yml`` in a repo checkout, else the
    identical embedded default): every mutating action is approval_required,
    only no-op/diagnostic kinds run automatically. KG ``governance_rule``
    overrides (scope ``action_policy``) win over file rules either way."""

    fleet_reconciler: bool = Field(default=False, alias="FLEET_RECONCILER")
    """Opt-in desired-state fleet reconciler tick (CONCEPT:AU-OS.config.desired-state-fleet-reconciler). Diffs the
    fleet registry (+ optional desired-state override file) against the
    observed fleet and proposes convergence actions through the ActionPolicy
    decision point. Default False until a deployment wires real actuators —
    with the default dry-run actuator it only records intended actions."""

    fleet_reconciler_interval: float = Field(
        default=120.0, alias="FLEET_RECONCILER_INTERVAL"
    )
    """Seconds between fleet-reconciler ticks (leader-only)."""

    fleet_reconciler_max_actions: int = Field(
        default=5, alias="FLEET_RECONCILER_MAX_ACTIONS"
    )
    """Storm guard: max convergence actions processed per reconciler tick;
    further divergences are deferred to the next tick (CONCEPT:AU-OS.config.desired-state-fleet-reconciler)."""

    fleet_registry_path: str = Field(default="", alias="FLEET_REGISTRY_PATH")
    """Path to the fleet service registry YAML. Empty (default) resolves to
    ``deploy/mcp-fleet.registry.yml`` in a repo checkout."""

    fleet_desired_state_path: str = Field(default="", alias="FLEET_DESIRED_STATE_PATH")
    """Optional desired-state override YAML layered on the registry
    (per-service ``replicas`` / ``desired: running|stopped`` / ``version``)."""

    fleet_actuator: str = Field(default="dryrun", alias="FLEET_ACTUATOR")
    """Fleet actuator selection: ``dryrun`` (default — records intended
    actions as KG nodes + notifications, mutates nothing) or ``docker``
    (reference actuator via the docker CLI when available). Real
    Portainer/Swarm actuation is wired at deployment by registering a
    ``FleetActuator`` via ``orchestration.fleet_actuation.set_fleet_actuator``."""

    deploy_watch_window: float = Field(default=300.0, alias="DEPLOY_WATCH_WINDOW")
    """Default health-watch window (seconds) after a deploy/restart action
    (CONCEPT:AU-OS.config.health-gated-deploy-rollback): sustained green inside the window records success, an
    unhealthy observation triggers the policy-gated rollback/escalation."""

    deploy_watch_poll: float = Field(default=15.0, alias="DEPLOY_WATCH_POLL")
    """Seconds between health probes inside a deploy watch window."""

    fleet_autoscaler: bool = Field(default=False, alias="FLEET_AUTOSCALER")
    """Opt-in reactive replica autoscaler tick (CONCEPT:AU-OS.scaling.reactive-replica-autoscaling). For each
    service with a registry/override ``scaling:`` block: read its load signal,
    target-track a desired replica count inside the declared min/max bounds,
    and propose ``scale_service`` through the ActionPolicy gate + actuator
    seam (deploy-watched on scale-up). Default False; with the default
    dry-run actuator it records intent without mutating."""

    fleet_autoscaler_interval: float = Field(
        default=60.0, alias="FLEET_AUTOSCALER_INTERVAL"
    )
    """Seconds between autoscaler ticks (leader-only)."""

    scaling_prometheus_url: str | None = Field(
        default=None, alias="SCALING_PROMETHEUS_URL"
    )
    """Optional Prometheus base URL for autoscaling signals (CONCEPT:AU-OS.scaling.reactive-replica-autoscaling).
    Set → the autoscaler reads signals via instant HTTP queries
    (``PrometheusHttpProvider``); unset (default) → the zero-infra
    ``LocalMetricsProvider`` reads this process's own OS-5.23/KG-2.55 gauges.
    A custom provider injected via
    ``orchestration.scaling_signals.set_scaling_signal_provider`` wins over
    both."""

    @field_validator(
        "kg_failure_evolution",
        "kg_failure_regression_dataset",
        "kg_dspy_optimization",
        "kg_anomaly_consumer",
        "kg_fuseki_publish",
        "kg_workflow_shape_gate",
        "fleet_reconciler",
        "fleet_autoscaler",
        mode="before",
    )
    @classmethod
    def _coerce_failure_flags(cls, v: Any) -> bool:
        """Parse daemon/gate toggles via the canonical ``to_boolean``
        ({t,true,y,yes,1}) so ``"True"``/``"False"`` mcp_config strings behave
        consistently with the rest of the fleet's boolean flags."""
        return to_boolean(v)

    nats_url: str | None = Field(default=None, alias="NATS_URL")
    kafka_bootstrap_servers: str | None = Field(
        default=None, alias="KAFKA_BOOTSTRAP_SERVERS"
    )
    graph_compute_backend: str = Field(default="rust", alias="GRAPH_COMPUTE_BACKEND")
    graph_service_endpoints: list[str] | None = Field(
        default=None, alias="GRAPH_SERVICE_ENDPOINTS"
    )
    """Engine shard endpoints (comma-separated or JSON list; ``unix://``/
    ``tcp://`` schemes). One entry behaves exactly like the single
    socket/tcp_addr path; 2+ entries enable tenant-partitioned sharding —
    graphs are routed to shards by HRW rendezvous hashing (CONCEPT:AU-KG.sharding.tenant-partitioned-sharding-hrw).
    Overrides socket/tcp_addr when provided."""

    @field_validator("graph_service_endpoints", mode="before")
    @classmethod
    def _coerce_endpoint_list(cls, v: Any) -> Any:
        """Accept comma-separated or JSON-encoded GRAPH_SERVICE_ENDPOINTS
        (CONCEPT:AU-KG.sharding.tenant-partitioned-sharding-hrw) via the canonical ``to_list`` so env wiring matches
        the rest of the fleet's list flags."""
        if v is None or isinstance(v, list):
            return v
        items = [str(e).strip() for e in to_list(v) if str(e).strip()]
        return items or None

    kg_connections: list[dict[str, Any]] | None = Field(
        default=None, alias="KG_CONNECTIONS"
    )
    """Declarative named graph connections (CONCEPT:AU-KG.backend.multi-connection-registry). A JSON list of
    backend specs, each ``{"name": <str>, "backend": <type>, ...create_backend
    kwargs (uri/host/port/user/password/db_name)}``. These are registered into
    the multi-connection registry at first use so the SAME graph tools can target
    any one (``target=<name>``) or fan out to all (``target="all"``). The
    zero-infra default is fully preserved: unset → only the ambient ``default``
    connection exists. For Postgres, use ``"backend": "age"`` for native
    openCypher portability."""

    gitlab_instances: list[dict[str, Any]] | None = Field(
        default=None, alias="GITLAB_INSTANCES"
    )
    """GitLab instances to index into the KG (CONCEPT:AU-KG.backend.declared-columns-so-schema). A JSON list of
    ``{"name": <str>, "url": <str>, "token": <str>, "verify_ssl": <bool>}`` — the
    single source of truth shared by the agent-utilities GitLab indexer
    (``knowledge_graph/core/gitlab_indexer``) and the ``gitlab-api`` connector's
    instance registry, so one config drives multi-tenant indexing and API access.
    Unset → falls back to the single-host ``GITLAB_URL``/``GITLAB_TOKEN`` env."""

    jira_instances: list[dict[str, Any]] | None = Field(
        default=None, alias="JIRA_INSTANCES"
    )
    """Jira instances to ingest into the KG (CONCEPT:AU-KG.compute.jira-first-class-delta). A JSON list of
    ``{"name": <str>, "server": <atlassian-mcp server name>, "project_keys": [<str>],
    "jql": <optional extra JQL>}`` — each is drained via the ``jira`` mcp_tool preset
    through its named ``atlassian-mcp`` server (which holds the credentials), so two
    Atlassian sites are two server entries + two instances. Unset → one synthetic
    instance over ``atlassian-mcp`` filtered by ``JIRA_PROJECT_KEYS``."""

    confluence_instances: list[dict[str, Any]] | None = Field(
        default=None, alias="CONFLUENCE_INSTANCES"
    )
    """Confluence instances to mirror into the KG (CONCEPT:AU-KG.compute.confluence-first-class-delta). A JSON list of
    ``{"name": <str>, "server": <atlassian-mcp server name>, "spaces": [<space-id>]}``
    — each space is drained via the ``confluence`` mcp_tool preset and ingested as
    full-text ``:ConfluencePage`` Documents. Unset → one synthetic instance over
    ``atlassian-mcp`` across ``CONFLUENCE_SPACE_IDS``."""

    plane_instances: list[dict[str, Any]] | None = Field(
        default=None, alias="PLANE_INSTANCES"
    )
    """Plane instances to ingest into the KG (CONCEPT:AU-KG.compute.plane-first-class-delta). A JSON list of
    ``{"name": <str>, "server": <plane-mcp server name>, "projects": [<project-id>]}``
    — each is drained via the ``plane`` mcp_tool preset through its named ``plane-mcp``
    server, so a SECOND Plane workspace is just a second server entry + instance. Unset
    → one synthetic instance over ``plane-mcp`` across ``PLANE_PROJECT_IDS``."""

    @field_validator(
        "gitlab_instances",
        "jira_instances",
        "confluence_instances",
        "plane_instances",
        mode="before",
    )
    @classmethod
    def _coerce_instance_list(cls, v: Any) -> Any:
        """Accept a JSON-encoded string or an already-parsed list for the
        ``*_INSTANCES`` multi-instance connector configs (CONCEPT:AU-KG.backend.declared-columns-so-schema/2.123-2.125)."""
        if v is None or isinstance(v, list):
            return v
        if isinstance(v, str):
            import json

            s = v.strip()
            if not s:
                return None
            try:
                parsed = json.loads(s)
            except Exception:
                return None
            return parsed if isinstance(parsed, list) else None
        return None

    @field_validator("graph_mirror_targets", mode="before")
    @classmethod
    def _coerce_graph_mirror_targets(cls, v: Any) -> Any:
        """Accept a JSON list, a comma-separated string, or a parsed list for
        GRAPH_MIRROR_TARGETS (CONCEPT:AU-KG.backend.mirror-health-repair)."""
        if v is None or isinstance(v, list):
            return v
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return None
            if s.startswith("["):
                import json

                try:
                    parsed = json.loads(s)
                except Exception:
                    return None
                return parsed if isinstance(parsed, list) else None
            return [x.strip() for x in s.split(",") if x.strip()]
        return v

    @field_validator("kg_connections", mode="before")
    @classmethod
    def _coerce_kg_connections(cls, v: Any) -> Any:
        """Accept a JSON-encoded string or an already-parsed list for
        KG_CONNECTIONS (CONCEPT:AU-KG.backend.multi-connection-registry)."""
        if v is None or isinstance(v, list):
            return v
        if isinstance(v, str):
            import json

            s = v.strip()
            if not s:
                return None
            try:
                parsed = json.loads(s)
            except Exception:
                return None
            return parsed if isinstance(parsed, list) else None
        return None

    @model_validator(mode="after")
    def _auto_enable_from_dependencies(self) -> "AgentConfig":
        """Configure-by-default, opt-out: a capability auto-engages once its
        deployment *dependency* is configured, rather than requiring a second
        explicit flag. This keeps the zero-infra default fully intact — with no
        JWT / Fuseki configured, nothing turns on — while a real deployment that
        wires the dependency gets the capability without remembering to also flip
        the flag (the AGENTS.md "detect and self-configure over a knob" rule).

        - ``KG_AUTH_REQUIRED`` engages once a JWT issuer / JWKS is configured (you
          set up identity → you want it enforced). Opt out with ``KG_AUTH_REQUIRED=false``.
        - ``KG_FUSEKI_PUBLISH`` engages once a Fuseki endpoint is configured.

        An explicit value for either flag (env/config) always wins — it lands in
        ``model_fields_set`` and is left untouched.
        """
        explicit = self.model_fields_set
        if "kg_auth_required" not in explicit and (
            self.auth_jwt_issuer or self.auth_jwt_jwks_uri
        ):
            self.kg_auth_required = True
        if "kg_fuseki_publish" not in explicit and (
            self.kg_fuseki_endpoint or self.jena_fuseki_url
        ):
            self.kg_fuseki_publish = True
        return self

    kg_default_graph: str = Field(default="__commons__", alias="KG_DEFAULT_GRAPH")
    """Default named graph for engine clients that don't target an explicit
    graph. In sharded mode (2+ GRAPH_SERVICE_ENDPOINTS) an ambient
    ActorContext tenant maps this default onto its per-tenant graph
    (``tenant__<tenant>__<base>``) before HRW shard selection; single-endpoint
    deployments are unaffected (CONCEPT:AU-KG.sharding.tenant-partitioned-sharding-hrw)."""
    graph_service_socket: str | None = Field(default=None, alias="GRAPH_SERVICE_SOCKET")
    """Path to the epistemic-graph Tokio service UDS socket. Defaults to
    $XDG_RUNTIME_DIR/epistemic-graph.sock."""
    kg_ingest_graph_routing: bool = Field(
        default=False, alias="KG_INGEST_GRAPH_ROUTING"
    )
    """Spread ingestion across per-source/per-repo/per-tenant graphs instead of the
    single ``__commons__`` graph (CONCEPT:AU-KG.ingest.unified-query-routing). Each graph name hashes to one of
    the engine's K redb shard writers (EG-026), so routing content across names lets K
    cores commit in parallel instead of one. OFF (default) = byte-for-byte today's
    behaviour: ingestion writes the default/tenant graph and reads hit the single
    default graph. ON = routed writes + a unified read that fans across the active
    content-graph set so split content stays queryable as one KG. Only changes where
    NEW data lands; existing ``__commons__`` content is left in place."""
    kg_ingest_shard_fanout: bool = Field(default=False, alias="KG_INGEST_SHARD_FANOUT")
    """Within a single routed content source, spread writes across per-shard
    content-keyed sub-graphs (``src:freshrss#0`` … ``#K-1``) instead of one graph
    per source (CONCEPT:AU-KG.ingest.batched-cross-graph-writer). A high-volume source
    (e.g. a large FreshRSS backlog) otherwise pins its whole drain to ONE graph =
    ONE of the engine's K redb shard writers, so K-1 sit idle. Bucketing by a
    content key across ``#0..#K-1`` puts K distinct graph names in flight so the
    memory-gen write stage fans across all K shard writers. Requires
    ``KG_INGEST_GRAPH_ROUTING``; OFF (default) = one graph per source (unchanged).
    The ``#n`` sub-graphs keep their source prefix so unified read still unions
    them. Pairs with the engine's ``MultiGraphBatchUpdate`` op
    (CONCEPT:EG-KG.storage.multi-graph-batch-write) which commits the K sub-batches in
    one round-trip across the K writers."""
    kg_rerank_model: str | None = Field(default=None, alias="KG_RERANK_MODEL")
    """Remote reranker model served on vLLM (e.g. ``BAAI/bge-reranker-v2-m3``). When set,
    reranking scores (query, passage) on the remote ``/v1/rerank`` endpoint — no local model,
    consistent with embeddings/LLM on vLLM (CONCEPT:AU-KG.retrieval.unset-dependency-free). Unset → the dependency-free
    lexical scorer (or opt-in local neural via ``KG_RERANK_LOCAL_NEURAL``)."""
    kg_rerank_base_url: str | None = Field(default=None, alias="KG_RERANK_BASE_URL")
    """Base URL for the remote reranker endpoint; defaults to ``OPENAI_BASE_URL`` (the vLLM
    endpoint already serving the embedder/LLM)."""
    kg_ingest_engine_endpoint: str | None = Field(
        default=None, alias="KG_INGEST_ENGINE_ENDPOINT"
    )
    """Dedicated ingest-engine endpoint (CONCEPT:AU-KG.sharding.tenant-partitioned-sharding-hrw, Phase D). When set (e.g.
    ``unix:///tmp/epistemic-graph-ingest.sock``), the host daemon spawns a SECOND,
    EPHEMERAL engine there (no persist dir — it only handles stateless parse +
    throwaway community-detection tenants) and the codebase-ingest path routes its
    parse + community work to it, isolated from the query engine and the background
    daemons. Unset (default) = today's single-engine behavior, no change. The ingest
    path health-gates the endpoint and falls back to the query engine if it's down."""
    graph_service_tcp_addr: str | None = Field(
        default=None, alias="GRAPH_SERVICE_TCP_ADDR"
    )
    """TCP address for the epistemic-graph service (e.g., 0.0.0.0:9100)."""
    graph_service_auth_secret: str | None = Field(
        default=None, alias="GRAPH_SERVICE_AUTH_SECRET"
    )
    """HMAC-SHA256 shared secret for service authentication. When unset, a
    per-install secret is generated once and persisted under the XDG data dir
    (``data_dir()/engine_secret``, mode 0600) so every local process — and any
    engine this launcher spawns — agrees (CONCEPT:AU-OS.identity.authenticated-identity-enforcement)."""
    kg_engine_insecure: bool = Field(default=False, alias="KG_ENGINE_INSECURE")
    """Opt out of engine HMAC auth for dev (CONCEPT:AU-OS.identity.authenticated-identity-enforcement). When True no
    client auth token is sent and a spawned engine gets
    ``EPISTEMIC_GRAPH_ALLOW_INSECURE=1`` so binaries that refuse to start
    without a secret still come up. Default False = secure by default."""

    engine_mode: str = Field(default="auto", alias="ENGINE_MODE")
    """How this process reaches the ONE engine authority (CONCEPT:AU-OS.deployment.engine-resolver-auto-provision — the
    single resolver ``knowledge_graph/core/engine_resolver.resolve_engine``):

    * ``auto`` (default) — derive behaviour from the existing ``graph_service_*``
      fields: a configured remote (``GRAPH_SERVICE_ENDPOINTS`` / multi-shard /
      ``GRAPH_SERVICE_TCP_ADDR``) is used as-is and never autostarted; a local
      endpoint is shared if already running, else autostarted (shared,
      supervised). This is the auto-bundled-engine default.
    * ``remote`` — force the remote leg: connect to ``engine_endpoint`` (or the
      configured ``graph_service_*`` endpoint) and NEVER autostart a local
      stand-in (fail-loud if unreachable). "I deployed the engine in Docker on
      another host."
    * ``shared`` — prefer reusing an already-running LOCAL engine; autostart one
      (detached, shared) only if none is up.
    * ``embedded`` — always provision a LOCAL engine, autostarting it when absent.

    No env-sprawl: read via ``config.engine_mode`` (a typed field), set in
    ``config.json`` as ``ENGINE_MODE``."""

    engine_endpoint: str | None = Field(default=None, alias="ENGINE_ENDPOINT")
    """Explicit remote engine endpoint override for ``engine_mode=remote``
    (CONCEPT:AU-OS.deployment.engine-resolver-auto-provision), e.g. ``tcp://engine.internal:9100`` — the "engine deployed
    on another host" case. When set it is folded into the resolved endpoint list
    exactly like a single ``GRAPH_SERVICE_ENDPOINTS`` entry. Unset → the existing
    ``graph_service_*`` resolution applies."""

    engine_lifecycle: str = Field(default="refcounted", alias="ENGINE_LIFECYCLE")
    """Lifecycle of an AUTOSTARTED local engine (CONCEPT:AU-OS.deployment.engine-resolver-auto-provision):

    * ``refcounted`` (default) — reference-counted idle shutdown: the engine
      self-terminates ``engine_idle_shutdown_secs`` seconds after its LAST client
      disconnects (robust to client crashes). The shared-tiny default — auto-stops
      when idle.
    * ``persistent`` — LONG-LIVING: the engine NEVER self-stops, even when idle
      (it runs like a local service). Forces idle-shutdown off regardless of
      ``engine_idle_shutdown_secs``.

    A remote/cluster engine is inherently persistent — the resolver never passes
    an idle-shutdown flag in remote mode."""

    engine_idle_shutdown_secs: int = Field(
        default=60, alias="ENGINE_IDLE_SHUTDOWN_SECS"
    )
    """Idle-shutdown grace (seconds) for a ``refcounted`` autostarted engine
    (CONCEPT:AU-OS.deployment.engine-resolver-auto-provision). ``> 0`` → the autostart leg passes
    ``--idle-shutdown-secs <secs>`` so the engine stops that many seconds after
    its last client disconnects. ``<= 0`` (or ``engine_lifecycle=persistent``) →
    NO flag is passed and the engine is long-living (never auto-stops). Default
    60s. Gracefully omitted against an older engine binary that doesn't advertise
    the flag."""
    graph_service_checkpoint_secs: int = Field(
        default=300, alias="GRAPH_SERVICE_CHECKPOINT_SECS"
    )
    """Auto-checkpoint interval for the Tokio service (0 = disabled)."""
    graph_service_persist_on_shutdown: bool = Field(
        default=True, alias="GRAPH_SERVICE_PERSIST_ON_SHUTDOWN"
    )
    """Serialize all graphs to disk on service shutdown."""
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
    """Total parallel capacity of the local inference endpoint (e.g. vLLM/LM Studio slots).

    This is the ONE knob for local-model parallelism. CONCEPT:AU-ORCH.execution.reserved-inference-slots — the system always
    reserves ``RESERVED_INTERACTIVE_INSTANCES`` (1) of these slots for the **interactive**
    path (the Telegram/messaging responder and graph-os-spawned pydantic-ai agents, which
    share the default model); all background KG work (fact enrichment, Layer 2/3 analysis,
    embeddings) is bounded to ``background_llm_concurrency()`` = capacity − reserved. So a
    background sweep can never consume the slot you need to get an answer. Set this to your
    endpoint's real parallel capacity and the reservation scales automatically."""

    def background_llm_concurrency(self) -> int:
        """Concurrency ceiling for background KG LLM work — capacity minus the reserved
        interactive slot(s) (CONCEPT:AU-ORCH.execution.reserved-inference-slots). Floors at 1 so background never starves."""
        return max(1, self.kg_llm_concurrency - RESERVED_INTERACTIVE_INSTANCES)

    kg_analysis_max_depth: int = Field(default=2, alias="KG_ANALYSIS_MAX_DEPTH")
    """Maximum recursive depth for background knowledge graph research daemons."""
    knowledge_graph_sync_background: bool = Field(
        default=True, alias="KNOWLEDGE_GRAPH_SYNC_BACKGROUND"
    )
    """Enable or disable background task workers for the Knowledge Graph pipeline."""
    enable_sdd_watcher: bool = Field(default=True, alias="ENABLE_SDD_WATCHER")
    """Enable or disable the background plan/task watcher thread in the KG MCP server."""
    model_registry_path: str | None = Field(default=None, alias="MODEL_REGISTRY_PATH")
    """Path to a YAML or JSON file defining the model registry."""
    role_routing: dict[str, dict] = Field(
        default_factory=dict, alias="MODEL_ROLE_ROUTING"
    )
    """CONCEPT:AU-ORCH.routing.optional-role-override — optional role→{tier,tags} overrides for planner/generator/
    learner/judge model selection. Empty roles fall back to the built-in default map in
    `models/model_registry.py`. Merged into the active registry when no registry-level
    override is present."""
    graph_direct_execution: bool = Field(default=True, alias="GRAPH_DIRECT_EXECUTION")
    """When True, AG-UI and ACP adapters bypass the LLM tool-call hop
    and invoke graph execution directly.  Set to False to restore the
    legacy agent -> run_graph_flow -> graph pipeline."""

    sparql_endpoints: list[str] = Field(
        default=["https://query.wikidata.org/sparql"], alias="SPARQL_ENDPOINTS"
    )
    """List of external SPARQL endpoints to federate (CONCEPT:AU-KG.query.vendor-agnostic-traversal)."""

    jena_fuseki_url: str | None = Field(default=None, alias="JENA_FUSEKI_URL")
    """URL for local Apache Jena Fuseki instance (e.g. http://localhost:3030)."""

    pggraph_dsn: str | None = Field(default=None, alias="PGGRAPH_DSN")
    """DSN string for Postgres with ParadeDB, PGGraph, and PGVector."""

    vllm_base_url: str | None = Field(default=None, alias="VLLM_BASE_URL")
    """Dedicated base URL for vLLM inference server (e.g. http://vllm.arpa/v1)."""

    kafka_topic: str | None = Field(default=None, alias="KAFKA_TOPIC")
    """Default Kafka topic for messaging/event ingestion."""

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
        default="https://cloud.langfuse.com", alias="LANGFUSE_HOST"
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
    """Path to a2a_config.json for external A2A agent discovery (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""
    a2a_refresh_interval: int = Field(default=300, alias="A2A_REFRESH_INTERVAL")
    """Interval in seconds for periodic A2A agent card re-fetch (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

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

    # --- Agent OS Architecture (CONCEPT:AU-OS.state.cognitive-scheduler-preemption) ---

    cognitive_scheduler_enabled: bool = Field(
        default=True, alias="COGNITIVE_SCHEDULER_ENABLED"
    )
    """Enable the Cognitive Scheduler for priority-aware agent management (CONCEPT:AU-OS.state.cognitive-scheduler-preemption)."""

    max_concurrent_agents: int = Field(default=5, alias="MAX_CONCURRENT_AGENTS")
    """Maximum number of concurrently running specialist agents (CONCEPT:AU-OS.state.cognitive-scheduler-preemption)."""

    agent_token_quota: int = Field(default=100_000, alias="AGENT_TOKEN_QUOTA")
    """Default per-agent token budget before preemption (CONCEPT:AU-OS.state.cognitive-scheduler-preemption)."""

    preemption_threshold_pct: float = Field(
        default=0.85, alias="PREEMPTION_THRESHOLD_PCT"
    )
    """Quota usage percentage that triggers preemption warning (CONCEPT:AU-OS.state.cognitive-scheduler-preemption)."""

    agent_policies_path: str | None = Field(default=None, alias="AGENT_POLICIES_PATH")
    """Path to agent_policies.json for identity-based governance (CONCEPT:AU-OS.state.cognitive-scheduler-preemption)."""

    permissions_signing_key: str | None = Field(
        default=None, alias="PERMISSIONS_SIGNING_KEY"
    )
    """HMAC signing key for agent identity tokens. Auto-generated if not set (CONCEPT:AU-OS.state.cognitive-scheduler-preemption)."""

    specialist_registry_path: str | None = Field(
        default=None, alias="SPECIALIST_REGISTRY_PATH"
    )
    """Path to local specialist registry directory (CONCEPT:AU-OS.state.cognitive-scheduler-preemption)."""

    # --- Native Messaging Backend (CONCEPT:AU-ECO.messaging.native-backend-abstraction) ---

    messaging_enabled_backends: list[str] = Field(
        default_factory=list, alias="MESSAGING_ENABLED_BACKENDS"
    )
    """List of messaging backend IDs to auto-connect on startup (CONCEPT:AU-ECO.messaging.native-backend-abstraction).
    Example: ["discord", "slack", "telegram"]."""

    messaging_kg_ingest: bool = Field(default=True, alias="MESSAGING_KG_INGEST")
    """Enable automatic Knowledge Graph ingestion for all inbound/outbound messages (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    messaging_kg_memory_type: str = Field(
        default="episodic", alias="MESSAGING_KG_MEMORY_TYPE"
    )
    """Default KG memory tier for inbound messages: 'episodic', 'semantic', or 'procedural' (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    messaging_route_to_planner: bool = Field(
        default=True, alias="MESSAGING_ROUTE_TO_PLANNER"
    )
    """Route inbound messaging events to the Planner Graph Agent for orchestration (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    # Per-platform tokens (read from config.json or env vars)
    messaging_discord_token: str | None = Field(
        default=None, alias="MESSAGING_DISCORD_TOKEN"
    )
    """Discord bot token. Also reads from DISCORD_BOT_TOKEN (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    messaging_slack_token: str | None = Field(
        default=None, alias="MESSAGING_SLACK_TOKEN"
    )
    """Slack bot token (xoxb-...). Also reads from SLACK_BOT_TOKEN (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    messaging_slack_app_token: str | None = Field(
        default=None, alias="MESSAGING_SLACK_APP_TOKEN"
    )
    """Slack app-level token (xapp-...) for Socket Mode (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    messaging_telegram_token: str | None = Field(
        default=None, alias="MESSAGING_TELEGRAM_TOKEN"
    )
    """Telegram bot token. Also reads from TELEGRAM_BOT_TOKEN (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    messaging_whatsapp_token: str | None = Field(
        default=None, alias="MESSAGING_WHATSAPP_TOKEN"
    )
    """WhatsApp API token. Also reads from WHATSAPP_TOKEN (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    messaging_whatsapp_phone_number_id: str | None = Field(
        default=None, alias="MESSAGING_WHATSAPP_PHONE_NUMBER_ID"
    )
    """WhatsApp Business API phone number ID (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    messaging_whatsapp_use_business_api: bool = Field(
        default=False, alias="MESSAGING_WHATSAPP_USE_BUSINESS_API"
    )
    """Use official WhatsApp Business API (True) or neonize bridge (False) (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    messaging_teams_app_id: str | None = Field(
        default=None, alias="MESSAGING_TEAMS_APP_ID"
    )
    """Microsoft Teams Bot Framework app ID (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    messaging_teams_app_secret: str | None = Field(
        default=None, alias="MESSAGING_TEAMS_APP_SECRET"
    )
    """Microsoft Teams Bot Framework app password (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    messaging_googlechat_service_account: str | None = Field(
        default=None, alias="MESSAGING_GOOGLECHAT_TOKEN"
    )
    """Path to Google Chat service account JSON file (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    messaging_googlemeet_service_account: str | None = Field(
        default=None, alias="MESSAGING_GOOGLEMEET_TOKEN"
    )
    """Path to Google Meet service account JSON file (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    messaging_mattermost_token: str | None = Field(
        default=None, alias="MESSAGING_MATTERMOST_TOKEN"
    )
    """Mattermost personal access token (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    messaging_mattermost_url: str | None = Field(
        default=None, alias="MESSAGING_MATTERMOST_URL"
    )
    """Mattermost server URL (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    messaging_matrix_token: str | None = Field(
        default=None, alias="MESSAGING_MATRIX_TOKEN"
    )
    """Matrix access token (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    messaging_matrix_homeserver: str | None = Field(
        default=None, alias="MESSAGING_MATRIX_HOMESERVER"
    )
    """Matrix homeserver URL (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    messaging_matrix_user_id: str | None = Field(
        default=None, alias="MESSAGING_MATRIX_USER_ID"
    )
    """Matrix user ID (e.g. @bot:matrix.org) (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    messaging_irc_server: str | None = Field(default=None, alias="MESSAGING_IRC_SERVER")
    """IRC server hostname (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    messaging_irc_port: int = Field(default=6667, alias="MESSAGING_IRC_PORT")
    """IRC server port (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    messaging_irc_nickname: str = Field(
        default="agent_bot", alias="MESSAGING_IRC_NICKNAME"
    )
    """IRC nickname (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    messaging_irc_channels: list[str] = Field(
        default_factory=list, alias="MESSAGING_IRC_CHANNELS"
    )
    """IRC channels to auto-join (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    messaging_signal_phone: str | None = Field(
        default=None, alias="MESSAGING_SIGNAL_TOKEN"
    )
    """Signal phone number for semaphore-bot (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    messaging_line_token: str | None = Field(default=None, alias="MESSAGING_LINE_TOKEN")
    """LINE channel access token (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    messaging_twitch_token: str | None = Field(
        default=None, alias="MESSAGING_TWITCH_TOKEN"
    )
    """Twitch OAuth token (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    messaging_twitch_channels: list[str] = Field(
        default_factory=list, alias="MESSAGING_TWITCH_CHANNELS"
    )
    """Twitch channels to join (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    messaging_synology_webhook_url: str | None = Field(
        default=None, alias="MESSAGING_SYNOLOGY_WEBHOOK_URL"
    )
    """Synology Chat incoming webhook URL (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    messaging_twilio_account_sid: str | None = Field(
        default=None, alias="MESSAGING_VOICECALL_APP_ID"
    )
    """Twilio account SID for voice/SMS (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    messaging_twilio_auth_token: str | None = Field(
        default=None, alias="MESSAGING_VOICECALL_TOKEN"
    )
    """Twilio auth token for voice/SMS (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    messaging_twilio_from_number: str | None = Field(
        default=None, alias="MESSAGING_VOICECALL_FROM_NUMBER"
    )
    """Twilio 'from' phone number (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    messaging_nextcloud_url: str | None = Field(
        default=None, alias="MESSAGING_NEXTCLOUD_URL"
    )
    """Nextcloud server URL (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    messaging_nextcloud_token: str | None = Field(
        default=None, alias="MESSAGING_NEXTCLOUD_TOKEN"
    )
    """Nextcloud app token (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    messaging_nextcloud_user: str | None = Field(
        default=None, alias="MESSAGING_NEXTCLOUD_APP_ID"
    )
    """Nextcloud username (CONCEPT:AU-ECO.messaging.native-backend-abstraction)."""

    # --- Parallel Engine (CONCEPT:AU-ORCH.execution.parallel-engine-visualizer) ---

    max_parallel_agents: int = Field(default=60, alias="MAX_PARALLEL_AGENTS")
    """Maximum concurrent agent executions across the engine (CONCEPT:AU-ORCH.execution.parallel-engine-visualizer).
    Acts as a global semaphore. Set higher for cloud deployments with high API limits."""

    worker_pool_size: int = Field(default=8, alias="WORKER_POOL_SIZE")
    """Number of worker processes/threads provisioned per node for executing agent
    turns and graph mutations (CONCEPT:AU-ORCH.execution.parallel-engine-visualizer).

    Scale knob. Together with ``graph_service_endpoints`` (Postgres/L0 shard fan-out
    for the epistemic graph) and ``kafka_bootstrap_servers`` (event-throughput axis),
    this is one of the three horizontal-scale knobs modeled in
    ``docs/scaling/capacity_model.md``:

    * ``worker_pool_size`` x node count -> active-concurrency capacity.
    * ``graph_service_endpoints`` -> resident-population (shard) capacity.
    * ``kafka_bootstrap_servers`` partitions -> event-throughput capacity.
    """

    parallel_batch_size: int = Field(default=25, alias="PARALLEL_BATCH_SIZE")
    """Number of agents per execution wave when batching is needed (CONCEPT:AU-ORCH.execution.parallel-engine-visualizer)."""

    synthesis_strategy: str = Field(default="auto", alias="SYNTHESIS_STRATEGY")
    """Default output synthesis strategy: 'auto', 'flat', 'hierarchical', 'progressive', 'rlm'.
    'auto' selects based on agent count and output size (CONCEPT:AU-ORCH.execution.rlm-synthesis-failed-falling)."""

    synthesis_ratio: int = Field(default=10, alias="SYNTHESIS_RATIO")
    """In hierarchical synthesis, how many outputs per synthesis sub-node (CONCEPT:AU-ORCH.execution.rlm-synthesis-failed-falling)."""

    agent_execution_timeout: float = Field(
        default=120.0, alias="AGENT_EXECUTION_TIMEOUT"
    )
    """Per-agent execution timeout in seconds (CONCEPT:AU-ORCH.execution.parallel-engine-visualizer)."""

    circuit_breaker_threshold: int = Field(default=3, alias="CIRCUIT_BREAKER_THRESHOLD")
    """Number of consecutive failures before disabling an agent type (CONCEPT:AU-ORCH.execution.parallel-engine-visualizer)."""

    enable_progressive_synthesis: bool = Field(
        default=True, alias="ENABLE_PROGRESSIVE_SYNTHESIS"
    )
    """Enable streaming synthesis as agents complete (CONCEPT:AU-ORCH.execution.rlm-synthesis-failed-falling)."""

    # --- Innovation Framework (CONCEPT:AU-OS.state.cognitive-scheduler-preemption through CONCEPT:AU-OS.state.cognitive-scheduler-preemption) ---

    homeostatic_downgrade_enabled: bool = Field(
        default=True, alias="HOMEOSTATIC_DOWNGRADE_ENABLED"
    )
    """Enable automatic model tier downgrade when budget is under pressure (CONCEPT:AU-OS.state.cognitive-scheduler-preemption)."""

    adversarial_verification: bool = Field(
        default=False, alias="ADVERSARIAL_VERIFICATION"
    )
    """Enable adversarial verification pass (opt-in only, doubles verification cost) (CONCEPT:AU-AHE.evaluation.adaptive-reasoning-effort)."""

    maintenance_token_budget: int = Field(default=0, alias="MAINTENANCE_TOKEN_BUDGET")
    """Token budget for autonomous maintenance cron (0 = unlimited) (CONCEPT:AU-OS.state.cognitive-scheduler-preemption)."""

    maintenance_priority: str = Field(default="LOW", alias="MAINTENANCE_PRIORITY")
    """Priority level for autonomous maintenance tasks (LOW/MEDIUM/HIGH) (CONCEPT:AU-OS.state.cognitive-scheduler-preemption)."""

    watchdog_patterns: list[str] = Field(
        default=[
            "pyproject.toml",
            "mcp_config.json",
            "requirements*.txt",
        ],
        alias="WATCHDOG_PATTERNS",
    )
    """File patterns to monitor for the file watcher trigger (CONCEPT:AU-OS.safety.doom-loop-detection)."""

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

    def assert_production_safe(self, *, profile: str | None = None) -> None:
        """Raise if this config uses toy defaults under a production profile.

        Delegates to :func:`agent_utilities.core.profile_guard.assert_production_safe`.
        No-op outside a production profile (see ``APP_PROFILE``). See
        ``docs/scaling/capacity_model.md`` for the scale knobs.
        """
        from agent_utilities.core.profile_guard import assert_production_safe

        assert_production_safe(self, profile=profile)


# --- Lazy Configuration Management ---


class BoundedLRUCache:
    """A bounded, dict-like LRU cache.

    Behaves like a ``dict`` for the subset of operations used by the lazy
    configuration machinery (``__getitem__``, ``__setitem__``, ``__contains__``,
    ``get``), but never grows beyond ``max_size`` entries. When the cap is
    exceeded the least-recently-used entry is evicted.

    Recency is updated on both read (``__getitem__`` / ``get``) and write
    (``__setitem__``). This bounds memory for the process-wide configuration
    cache so it cannot grow without limit (e.g. under repeated reconfiguration
    or many derived keys).
    """

    def __init__(self, max_size: int = 4096) -> None:
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        self.max_size = max_size
        self._data: OrderedDict[str, Any] = OrderedDict()

    def __setitem__(self, key: str, value: Any) -> None:
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = value
        # Evict least-recently-used entries until within the cap.
        while len(self._data) > self.max_size:
            self._data.popitem(last=False)

    def __getitem__(self, key: str) -> Any:
        value = self._data[key]
        self._data.move_to_end(key)
        return value

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def get(self, key: str, default: Any = None) -> Any:
        if key in self._data:
            return self[key]
        return default

    def clear(self) -> None:
        self._data.clear()

    def keys(self):
        return self._data.keys()


# Maximum number of entries retained in the process-wide lazy config cache.
LAZY_CACHE_MAX_SIZE = 4096

_LAZY_CACHE: BoundedLRUCache = BoundedLRUCache(max_size=LAZY_CACHE_MAX_SIZE)


def _init_lazy_config():
    if "_config" in _LAZY_CACHE:
        return

    _ensure_env_loaded()

    cfg = AgentConfig()
    _LAZY_CACHE["_config"] = cfg
    _LAZY_CACHE["config"] = cfg

    _LAZY_CACHE["DEFAULT_AGENT_NAME"] = cfg.default_agent_name
    _LAZY_CACHE["DEFAULT_AGENT_DESCRIPTION"] = cfg.agent_description
    _LAZY_CACHE["DEFAULT_AGENT_SYSTEM_PROMPT"] = cfg.agent_system_prompt
    _LAZY_CACHE["DEFAULT_DEBUG"] = cfg.debug

    # --- Derive DEFAULT_LLM_* from chat_models / embedding_models registry ---
    _default_chat = cfg.default_chat_model
    _lite_chat = cfg.lite_chat_model
    _super_chat = cfg.super_chat_model
    _default_embed = cfg.default_embedding_model

    _LAZY_CACHE["DEFAULT_LLM_PROVIDER"] = (
        (_default_chat.provider if _default_chat else None)
        or os.getenv("PROVIDER")
        or "openai"
    )
    _LAZY_CACHE["DEFAULT_LLM_MODEL_ID"] = (
        (_default_chat.id if _default_chat else None)
        or os.getenv("MODEL_ID")
        or "qwen/qwen3.6-27b"
    )
    _LAZY_CACHE["DEFAULT_LLM_BASE_URL"] = (
        _default_chat.base_url if _default_chat else None
    )
    _LAZY_CACHE["DEFAULT_LLM_API_KEY"] = (
        _default_chat.api_key if _default_chat else None
    )

    _LAZY_CACHE["DEFAULT_LITE_LLM_PROVIDER"] = (
        _lite_chat.provider if _lite_chat else None
    ) or _LAZY_CACHE["DEFAULT_LLM_PROVIDER"]
    _LAZY_CACHE["DEFAULT_LITE_LLM_MODEL_ID"] = (
        _lite_chat.id if _lite_chat else None
    ) or _LAZY_CACHE["DEFAULT_LLM_MODEL_ID"]
    _LAZY_CACHE["DEFAULT_LITE_LLM_BASE_URL"] = (
        _lite_chat.base_url if _lite_chat else None
    ) or _LAZY_CACHE["DEFAULT_LLM_BASE_URL"]
    _LAZY_CACHE["DEFAULT_LITE_LLM_API_KEY"] = (
        _lite_chat.api_key if _lite_chat else None
    ) or _LAZY_CACHE["DEFAULT_LLM_API_KEY"]

    _LAZY_CACHE["DEFAULT_SUPER_LLM_PROVIDER"] = (
        _super_chat.provider if _super_chat else None
    ) or _LAZY_CACHE["DEFAULT_LLM_PROVIDER"]
    _LAZY_CACHE["DEFAULT_SUPER_LLM_MODEL_ID"] = (
        _super_chat.id if _super_chat else None
    ) or _LAZY_CACHE["DEFAULT_LLM_MODEL_ID"]
    _LAZY_CACHE["DEFAULT_SUPER_LLM_BASE_URL"] = (
        _super_chat.base_url if _super_chat else None
    ) or _LAZY_CACHE["DEFAULT_LLM_BASE_URL"]
    _LAZY_CACHE["DEFAULT_SUPER_LLM_API_KEY"] = (
        _super_chat.api_key if _super_chat else None
    ) or _LAZY_CACHE["DEFAULT_LLM_API_KEY"]

    _LAZY_CACHE["DEFAULT_EMBEDDING_PROVIDER"] = (
        _default_embed.provider if _default_embed else None
    ) or _LAZY_CACHE["DEFAULT_LLM_PROVIDER"]
    _LAZY_CACHE["DEFAULT_EMBEDDING_MODEL_ID"] = (
        _default_embed.id if _default_embed else None
    ) or "text-embedding-nomic-embed-text-v2-moe"
    _LAZY_CACHE["DEFAULT_EMBEDDING_BASE_URL"] = (
        _default_embed.base_url if _default_embed else None
    ) or _LAZY_CACHE["DEFAULT_LLM_BASE_URL"]
    _LAZY_CACHE["DEFAULT_EMBEDDING_API_KEY"] = (
        _default_embed.api_key if _default_embed else None
    ) or _LAZY_CACHE["DEFAULT_LLM_API_KEY"]
    _LAZY_CACHE["DEFAULT_MCP_URL"] = cfg.mcp_url

    _LAZY_CACHE["DEFAULT_MCP_CONFIG"] = cfg.mcp_config
    _LAZY_CACHE["DEFAULT_CUSTOM_SKILLS_DIRECTORY"] = cfg.custom_skills_directory
    _LAZY_CACHE["DEFAULT_SKILL_TYPES"] = cfg.skill_types
    _LAZY_CACHE["DEFAULT_ENABLE_WEB_UI"] = cfg.enable_web_ui
    _LAZY_CACHE["DEFAULT_ENABLE_TERMINAL_UI"] = cfg.enable_terminal_ui
    _LAZY_CACHE["DEFAULT_ENABLE_WEB_LOGS"] = cfg.enable_web_logs
    _LAZY_CACHE["DEFAULT_ENABLE_OTEL"] = cfg.enable_otel
    _LAZY_CACHE["DEFAULT_ENABLE_ACP"] = cfg.enable_acp
    _LAZY_CACHE["DEFAULT_ACP_PORT"] = cfg.acp_port
    _LAZY_CACHE["DEFAULT_ACP_SESSION_ROOT"] = cfg.acp_session_root
    _LAZY_CACHE["DEFAULT_TERMINAL_AGENT"] = cfg.default_terminal_agent

    if not cfg.enable_otel:
        os.environ["OTEL_SDK_DISABLED"] = "true"
    else:
        os.environ.pop("OTEL_SDK_DISABLED", None)

    _LAZY_CACHE["DEFAULT_OTEL_EXPORTER_OTLP_ENDPOINT"] = cfg.otel_exporter_otlp_endpoint
    _LAZY_CACHE["DEFAULT_OTEL_EXPORTER_OTLP_HEADERS"] = cfg.otel_exporter_otlp_headers
    _LAZY_CACHE[
        "DEFAULT_OTEL_EXPORTER_OTLP_PUBLIC_KEY"
    ] = cfg.otel_exporter_otlp_public_key
    _LAZY_CACHE[
        "DEFAULT_OTEL_EXPORTER_OTLP_SECRET_KEY"
    ] = cfg.otel_exporter_otlp_secret_key
    _LAZY_CACHE["DEFAULT_OTEL_EXPORTER_OTLP_PROTOCOL"] = cfg.otel_exporter_otlp_protocol

    _LAZY_CACHE["DEFAULT_LANGFUSE_PUBLIC_KEY"] = cfg.langfuse_public_key
    _LAZY_CACHE["DEFAULT_LANGFUSE_SECRET_KEY"] = cfg.langfuse_secret_key
    _LAZY_CACHE["DEFAULT_LANGFUSE_HOST"] = cfg.langfuse_host
    _LAZY_CACHE[
        "DEFAULT_LANGFUSE_DATASET_CAPTURE_THRESHOLD"
    ] = cfg.langfuse_dataset_capture_threshold

    _LAZY_CACHE["DEFAULT_A2A_BROKER"] = cfg.a2a_broker
    _LAZY_CACHE["DEFAULT_A2A_BROKER_URL"] = cfg.a2a_broker_url
    _LAZY_CACHE["DEFAULT_A2A_STORAGE"] = cfg.a2a_storage
    _LAZY_CACHE["DEFAULT_A2A_STORAGE_URL"] = cfg.a2a_storage_url
    _LAZY_CACHE["DEFAULT_A2A_CONFIG"] = cfg.a2a_config
    _LAZY_CACHE["DEFAULT_A2A_REFRESH_INTERVAL"] = cfg.a2a_refresh_interval

    _LAZY_CACHE["DEFAULT_MAX_TOKENS"] = cfg.max_tokens
    _LAZY_CACHE["DEFAULT_TEMPERATURE"] = cfg.temperature
    _LAZY_CACHE["DEFAULT_TOP_P"] = cfg.top_p
    _LAZY_CACHE["DEFAULT_TIMEOUT"] = cfg.timeout
    _LAZY_CACHE["DEFAULT_TOOL_TIMEOUT"] = cfg.tool_timeout
    _LAZY_CACHE["DEFAULT_PARALLEL_TOOL_CALLS"] = cfg.parallel_tool_calls
    _LAZY_CACHE["DEFAULT_SEED"] = cfg.seed
    _LAZY_CACHE["DEFAULT_PRESENCE_PENALTY"] = cfg.presence_penalty
    _LAZY_CACHE["DEFAULT_FREQUENCY_PENALTY"] = cfg.frequency_penalty

    _LAZY_CACHE["DEFAULT_LOGIT_BIAS"] = (
        cfg.logit_bias
        if cfg.logit_bias is not None
        else to_dict(os.getenv("LOGIT_BIAS"))
    )
    _LAZY_CACHE["DEFAULT_STOP_SEQUENCES"] = (
        cfg.stop_sequences
        if cfg.stop_sequences is not None
        else to_list(os.getenv("STOP_SEQUENCES"))
    )
    _LAZY_CACHE["DEFAULT_EXTRA_HEADERS"] = (
        cfg.extra_headers
        if cfg.extra_headers is not None
        else to_dict(os.getenv("EXTRA_HEADERS"))
    )
    _LAZY_CACHE["DEFAULT_EXTRA_BODY"] = (
        cfg.extra_body
        if cfg.extra_body is not None
        else to_dict(os.getenv("EXTRA_BODY"))
    )

    _LAZY_CACHE["DEFAULT_MIN_CONFIDENCE"] = cfg.min_confidence
    _LAZY_CACHE["DEFAULT_VALIDATION_MODE"] = (
        cfg.validation_mode
        or to_boolean(os.getenv("VALIDATION_MODE", "False"))
        or to_boolean(os.getenv("AGENT_UTILITIES_TESTING", "False"))
    )
    _LAZY_CACHE["DEFAULT_SSL_VERIFY"] = GET_DEFAULT_SSL_VERIFY()
    _LAZY_CACHE["DEFAULT_APPROVAL_TIMEOUT"] = cfg.approval_timeout
    _LAZY_CACHE["DEFAULT_MAX_CRON_LOG_ENTRIES"] = 50

    _LAZY_CACHE["TOOL_GUARD_MODE"] = cfg.tool_guard_mode
    _LAZY_CACHE["SENSITIVE_TOOL_PATTERNS"] = cfg.sensitive_tool_patterns

    # Router/KG models: find models with can_route/can_kg flags, else fallback to lite
    _router_model = next((m for m in cfg.chat_models if m.can_route), _lite_chat)
    _kg_model = next((m for m in cfg.chat_models if m.can_kg), _lite_chat)
    _LAZY_CACHE["DEFAULT_ROUTER_MODEL"] = (
        _router_model.id if _router_model else None
    ) or _LAZY_CACHE["DEFAULT_LITE_LLM_MODEL_ID"]

    _LAZY_CACHE["DEFAULT_GRAPH_PERSISTENCE_TYPE"] = cfg.graph_persistence_type
    _LAZY_CACHE["DEFAULT_GRAPH_PERSISTENCE_PATH"] = cfg.graph_persistence_path
    _LAZY_CACHE["DEFAULT_ENABLE_LLM_VALIDATION"] = cfg.enable_llm_validation
    _LAZY_CACHE["DEFAULT_ROUTING_STRATEGY"] = cfg.routing_strategy
    _LAZY_CACHE["DEFAULT_GRAPH_ROUTER_TIMEOUT"] = cfg.graph_router_timeout
    _LAZY_CACHE["DEFAULT_GRAPH_VERIFIER_TIMEOUT"] = cfg.graph_verifier_timeout
    _LAZY_CACHE["DEFAULT_ENABLE_KG_EMBEDDINGS"] = cfg.enable_kg_embeddings
    _LAZY_CACHE["DEFAULT_KG_BACKUPS"] = cfg.kg_backups
    _LAZY_CACHE["DEFAULT_KG_INGESTION_WORKERS"] = cfg.kg_ingestion_workers
    _LAZY_CACHE["DEFAULT_KG_LLM_CONCURRENCY"] = cfg.kg_llm_concurrency
    _LAZY_CACHE["DEFAULT_KG_MODEL_ID"] = (
        _kg_model.id if _kg_model else None
    ) or _LAZY_CACHE["DEFAULT_LITE_LLM_MODEL_ID"]
    _LAZY_CACHE["DEFAULT_KG_ANALYSIS_MAX_DEPTH"] = cfg.kg_analysis_max_depth
    _LAZY_CACHE[
        "DEFAULT_KNOWLEDGE_GRAPH_SYNC_BACKGROUND"
    ] = cfg.knowledge_graph_sync_background
    _LAZY_CACHE["DEFAULT_GRAPH_DIRECT_EXECUTION"] = cfg.graph_direct_execution

    # --- Parallel Engine Defaults ---
    _LAZY_CACHE["DEFAULT_MAX_PARALLEL_AGENTS"] = cfg.max_parallel_agents
    _LAZY_CACHE["DEFAULT_PARALLEL_BATCH_SIZE"] = cfg.parallel_batch_size
    _LAZY_CACHE["DEFAULT_SYNTHESIS_STRATEGY"] = cfg.synthesis_strategy
    _LAZY_CACHE["DEFAULT_SYNTHESIS_RATIO"] = cfg.synthesis_ratio
    _LAZY_CACHE["DEFAULT_AGENT_EXECUTION_TIMEOUT"] = cfg.agent_execution_timeout
    _LAZY_CACHE["DEFAULT_CIRCUIT_BREAKER_THRESHOLD"] = cfg.circuit_breaker_threshold
    _LAZY_CACHE[
        "DEFAULT_ENABLE_PROGRESSIVE_SYNTHESIS"
    ] = cfg.enable_progressive_synthesis

    _LAZY_CACHE["AGENT_API_KEY"] = cfg.agent_api_key
    _LAZY_CACHE["ENABLE_API_AUTH"] = cfg.enable_api_auth
    _LAZY_CACHE["MAX_UPLOAD_SIZE"] = cfg.max_upload_size

    _LAZY_CACHE["SECRETS_BACKEND"] = cfg.secrets_backend
    _LAZY_CACHE["SECRETS_SQLITE_PATH"] = cfg.secrets_sqlite_path
    _LAZY_CACHE["SECRETS_VAULT_URL"] = cfg.secrets_vault_url
    _LAZY_CACHE["SECRETS_VAULT_MOUNT"] = cfg.secrets_vault_mount

    _LAZY_CACHE["AUTH_JWT_JWKS_URI"] = cfg.auth_jwt_jwks_uri
    _LAZY_CACHE["AUTH_JWT_ISSUER"] = cfg.auth_jwt_issuer
    _LAZY_CACHE["AUTH_JWT_AUDIENCE"] = cfg.auth_jwt_audience
    _LAZY_CACHE["ALLOWED_ORIGINS"] = cfg.allowed_origins
    _LAZY_CACHE["ALLOWED_HOSTS"] = cfg.allowed_hosts

    # Agent OS Architecture defaults
    _LAZY_CACHE["DEFAULT_COGNITIVE_SCHEDULER_ENABLED"] = cfg.cognitive_scheduler_enabled
    _LAZY_CACHE["DEFAULT_MAX_CONCURRENT_AGENTS"] = cfg.max_concurrent_agents
    _LAZY_CACHE["DEFAULT_AGENT_TOKEN_QUOTA"] = cfg.agent_token_quota
    _LAZY_CACHE["DEFAULT_PREEMPTION_THRESHOLD_PCT"] = cfg.preemption_threshold_pct
    _LAZY_CACHE["DEFAULT_AGENT_POLICIES_PATH"] = cfg.agent_policies_path
    _LAZY_CACHE["DEFAULT_PERMISSIONS_SIGNING_KEY"] = cfg.permissions_signing_key
    _LAZY_CACHE["DEFAULT_SPECIALIST_REGISTRY_PATH"] = cfg.specialist_registry_path

    # Innovation Framework defaults
    _LAZY_CACHE["DEFAULT_HOMEOSTATIC_DOWNGRADE"] = cfg.homeostatic_downgrade_enabled
    _LAZY_CACHE["DEFAULT_ADVERSARIAL_VERIFICATION"] = cfg.adversarial_verification
    _LAZY_CACHE["DEFAULT_MAINTENANCE_TOKEN_BUDGET"] = cfg.maintenance_token_budget
    _LAZY_CACHE["DEFAULT_MAINTENANCE_PRIORITY"] = cfg.maintenance_priority
    _LAZY_CACHE["DEFAULT_WATCHDOG_PATTERNS"] = cfg.watchdog_patterns

    _ensure_config_template()


def _ensure_config_template():
    import json
    from pathlib import Path

    import platformdirs

    APP_NAME = "agent-utilities"
    APP_AUTHOR = "knuckles-team"

    override = os.environ.get("AGENT_UTILITIES_CONFIG_DIR")
    # Hermetic tests never write a template into the developer's XDG-default
    # config dir (companion to the read-skip in ``_load_xdg_json_config``).
    if not override and (
        _under_pytest()
        or to_boolean(os.environ.get("AGENT_UTILITIES_TESTING", "false"))
    ):
        return
    if override:
        cfg_dir = Path(override).expanduser()
    else:
        cfg_dir = Path(platformdirs.user_config_path(APP_NAME, APP_AUTHOR))

    cfg_file = cfg_dir / "config.json"
    if not cfg_file.exists():
        try:
            cfg_dir.mkdir(parents=True, exist_ok=True)
            with open(cfg_file, "w") as f:
                json.dump(_LAZY_CACHE["config"].model_dump(by_alias=False), f, indent=4)
        except Exception as e:
            import logging

            logging.getLogger(__name__).warning(f"Failed to write config template: {e}")


if TYPE_CHECKING:
    # These names are materialized at runtime via module ``__getattr__`` (PEP 562)
    # from the lazy cache. Declare their concrete types here so importers get
    # real typing instead of ``Any``.
    config: AgentConfig
    SENSITIVE_TOOL_PATTERNS: list[str]
    TOOL_GUARD_MODE: str
    DEFAULT_EMBEDDING_BASE_URL: str
    DEFAULT_EMBEDDING_MODEL_ID: str
    DEFAULT_KG_ANALYSIS_MAX_DEPTH: int


def __getattr__(name: str) -> Any:
    # Handle the decoupled HOST/PORT directly for instant resolution
    if name == "DEFAULT_HOST":
        return os.environ.get("HOST", "0.0.0.0")
    if name == "DEFAULT_PORT":
        try:
            return int(os.environ.get("PORT", "9000"))
        except ValueError:
            return 9000

    if name.startswith("__"):
        raise AttributeError(name)

    _init_lazy_config()

    if name in _LAZY_CACHE:
        return _LAZY_CACHE[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(
        list(globals().keys())
        + [
            "config",
            "DEFAULT_AGENT_NAME",
            "DEFAULT_AGENT_DESCRIPTION",
            "DEFAULT_AGENT_SYSTEM_PROMPT",
            "DEFAULT_HOST",
            "DEFAULT_PORT",
            "DEFAULT_DEBUG",
            "DEFAULT_LLM_PROVIDER",
            "DEFAULT_LLM_MODEL_ID",
            "DEFAULT_LLM_BASE_URL",
            "DEFAULT_LLM_API_KEY",
            "DEFAULT_LITE_LLM_PROVIDER",
            "DEFAULT_LITE_LLM_MODEL_ID",
            "DEFAULT_LITE_LLM_BASE_URL",
            "DEFAULT_LITE_LLM_API_KEY",
            "DEFAULT_SUPER_LLM_PROVIDER",
            "DEFAULT_SUPER_LLM_MODEL_ID",
            "DEFAULT_SUPER_LLM_BASE_URL",
            "DEFAULT_SUPER_LLM_API_KEY",
            "DEFAULT_EMBEDDING_PROVIDER",
            "DEFAULT_EMBEDDING_MODEL_ID",
            "DEFAULT_EMBEDDING_BASE_URL",
            "DEFAULT_EMBEDDING_API_KEY",
            "DEFAULT_MCP_URL",
            "DEFAULT_MCP_CONFIG",
            "DEFAULT_CUSTOM_SKILLS_DIRECTORY",
            "DEFAULT_SKILL_TYPES",
            "DEFAULT_ENABLE_WEB_UI",
            "DEFAULT_ENABLE_TERMINAL_UI",
            "DEFAULT_ENABLE_WEB_LOGS",
            "DEFAULT_ENABLE_OTEL",
            "DEFAULT_ENABLE_ACP",
            "DEFAULT_ACP_PORT",
            "DEFAULT_ACP_SESSION_ROOT",
            "DEFAULT_TERMINAL_AGENT",
            "DEFAULT_OTEL_EXPORTER_OTLP_ENDPOINT",
            "DEFAULT_OTEL_EXPORTER_OTLP_HEADERS",
            "DEFAULT_OTEL_EXPORTER_OTLP_PUBLIC_KEY",
            "DEFAULT_OTEL_EXPORTER_OTLP_SECRET_KEY",
            "DEFAULT_OTEL_EXPORTER_OTLP_PROTOCOL",
            "DEFAULT_LANGFUSE_PUBLIC_KEY",
            "DEFAULT_LANGFUSE_SECRET_KEY",
            "DEFAULT_LANGFUSE_HOST",
            "DEFAULT_LANGFUSE_DATASET_CAPTURE_THRESHOLD",
            "DEFAULT_A2A_BROKER",
            "DEFAULT_A2A_BROKER_URL",
            "DEFAULT_A2A_STORAGE",
            "DEFAULT_A2A_STORAGE_URL",
            "DEFAULT_A2A_CONFIG",
            "DEFAULT_A2A_REFRESH_INTERVAL",
            "DEFAULT_MAX_TOKENS",
            "DEFAULT_TEMPERATURE",
            "DEFAULT_TOP_P",
            "DEFAULT_TIMEOUT",
            "DEFAULT_TOOL_TIMEOUT",
            "DEFAULT_PARALLEL_TOOL_CALLS",
            "DEFAULT_SEED",
            "DEFAULT_PRESENCE_PENALTY",
            "DEFAULT_FREQUENCY_PENALTY",
            "DEFAULT_LOGIT_BIAS",
            "DEFAULT_STOP_SEQUENCES",
            "DEFAULT_EXTRA_HEADERS",
            "DEFAULT_EXTRA_BODY",
            "DEFAULT_MIN_CONFIDENCE",
            "DEFAULT_VALIDATION_MODE",
            "DEFAULT_SSL_VERIFY",
            "DEFAULT_APPROVAL_TIMEOUT",
            "DEFAULT_MAX_CRON_LOG_ENTRIES",
            "TOOL_GUARD_MODE",
            "SENSITIVE_TOOL_PATTERNS",
            "DEFAULT_ROUTER_MODEL",
            "DEFAULT_GRAPH_PERSISTENCE_TYPE",
            "DEFAULT_GRAPH_PERSISTENCE_PATH",
            "DEFAULT_ENABLE_LLM_VALIDATION",
            "DEFAULT_ROUTING_STRATEGY",
            "DEFAULT_GRAPH_ROUTER_TIMEOUT",
            "DEFAULT_GRAPH_VERIFIER_TIMEOUT",
            "DEFAULT_ENABLE_KG_EMBEDDINGS",
            "DEFAULT_KG_BACKUPS",
            "DEFAULT_KG_INGESTION_WORKERS",
            "DEFAULT_KG_LLM_CONCURRENCY",
            "DEFAULT_KG_MODEL_ID",
            "DEFAULT_KG_ANALYSIS_MAX_DEPTH",
            "DEFAULT_KNOWLEDGE_GRAPH_SYNC_BACKGROUND",
            "DEFAULT_GRAPH_DIRECT_EXECUTION",
            "DEFAULT_MAX_PARALLEL_AGENTS",
            "DEFAULT_PARALLEL_BATCH_SIZE",
            "DEFAULT_SYNTHESIS_STRATEGY",
            "DEFAULT_SYNTHESIS_RATIO",
            "DEFAULT_AGENT_EXECUTION_TIMEOUT",
            "DEFAULT_CIRCUIT_BREAKER_THRESHOLD",
            "DEFAULT_ENABLE_PROGRESSIVE_SYNTHESIS",
            "AGENT_API_KEY",
            "ENABLE_API_AUTH",
            "MAX_UPLOAD_SIZE",
            "SECRETS_BACKEND",
            "SECRETS_SQLITE_PATH",
            "SECRETS_VAULT_URL",
            "SECRETS_VAULT_MOUNT",
            "AUTH_JWT_JWKS_URI",
            "AUTH_JWT_ISSUER",
            "AUTH_JWT_AUDIENCE",
            "ALLOWED_ORIGINS",
            "ALLOWED_HOSTS",
            "DEFAULT_COGNITIVE_SCHEDULER_ENABLED",
            "DEFAULT_MAX_CONCURRENT_AGENTS",
            "DEFAULT_AGENT_TOKEN_QUOTA",
            "DEFAULT_PREEMPTION_THRESHOLD_PCT",
            "DEFAULT_AGENT_POLICIES_PATH",
            "DEFAULT_PERMISSIONS_SIGNING_KEY",
            "DEFAULT_SPECIALIST_REGISTRY_PATH",
            "DEFAULT_HOMEOSTATIC_DOWNGRADE",
            "DEFAULT_ADVERSARIAL_VERIFICATION",
            "DEFAULT_MAINTENANCE_TOKEN_BUDGET",
            "DEFAULT_MAINTENANCE_PRIORITY",
            "DEFAULT_WATCHDOG_PATTERNS",
        ]
    )


# --- Migrated from graph/config_helpers.py ---
import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from agent_utilities.base_utilities import to_integer
from agent_utilities.core.workspace import CORE_FILES, get_workspace_path
from agent_utilities.models import (
    MCPAgent,
    MCPAgentRegistryModel,
    MCPConfigModel,
    MCPToolInfo,
)

logger = logging.getLogger(__name__)

import os

# Whole-workflow orchestration execution budget (ms). 10min default (was 20min):
# the client's per-RPC timeout now catches engine hangs in seconds, so this only
# bounds a wedged multi-agent run. Override via GRAPH_TIMEOUT for unusually long
# workflows.
DEFAULT_GRAPH_TIMEOUT = to_integer(os.environ.get("GRAPH_TIMEOUT", "600000"))


# ---------------------------------------------------------------------------
# CONCEPT:AU-ORCH.adapter.hot-cache-invalidation — Session-Scoped Registry Cache
# ---------------------------------------------------------------------------


class _RegistryCache:
    """Session-scoped cache for KG registry data.

    CONCEPT:AU-ORCH.adapter.hot-cache-invalidation — Hot Cache Layer

    Populated on first access, invalidated by explicit event signals.
    No TTL — pure event-driven invalidation from four callsites:

    1. ``agent_manager.sync_mcp_agents()`` (MCP reload)
    2. Pipeline completion (``PipelineRunner.run()``)
    3. ``promote_coalition_to_template()`` (TeamConfig creation)
    4. ``MemoryRetriever.update_after_session()`` (proficiency update)
    """

    _registry: MCPAgentRegistryModel | None = None
    _prompts: dict[str, str] = {}
    _tool_agent_map: dict[str, list[str]] = {}

    @classmethod
    def invalidate(cls) -> None:
        """Clear all cached data.  Called by event-driven signals."""
        cls._registry = None
        cls._prompts.clear()
        cls._tool_agent_map.clear()
        logger.info(
            "[CACHE] Registry cache invalidated (CONCEPT:AU-ORCH.adapter.hot-cache-invalidation)."
        )

    @classmethod
    def get_registry(cls) -> MCPAgentRegistryModel:
        """Return the cached registry, populating on first access."""
        if cls._registry is None:
            cls._registry = _fetch_registry_from_kg()
            logger.info(
                "[CACHE] Registry cache populated: %d agents, %d tools.",
                len(cls._registry.agents),
                len(cls._registry.tools),
            )
        return cls._registry


def invalidate_registry_cache() -> None:
    """Public API to invalidate the hot cache.

    CONCEPT:AU-ORCH.adapter.hot-cache-invalidation — Hot Cache Layer

    Call this after any operation that changes the registry state:
    MCP reload, pipeline ingestion, TeamConfig promotion, or
    Self-Model update.
    """
    _RegistryCache.invalidate()


def _fetch_registry_from_kg() -> MCPAgentRegistryModel:
    """Fetch the full registry from the Knowledge Graph (uncached).

    This is the expensive operation that ``_RegistryCache`` wraps.
    Delegates to focused sub-functions for each data source.
    """
    if __import__("os").getenv("ENABLE_KG_REGISTRY_FETCH", "true").lower() in (
        "false",
        "0",
        "no",
    ):
        logger.info("Registry fetch bypassed via environment variable.")
        return MCPAgentRegistryModel()

    from ..knowledge_graph.core.engine import IntelligenceGraphEngine

    engine = IntelligenceGraphEngine.get_active()
    if not engine:
        from agent_utilities.core.paths import kg_db_path
        from agent_utilities.core.workspace import get_agent_workspace

        ws = get_agent_workspace()
        db_path = str(kg_db_path(ws))
        engine = IntelligenceGraphEngine(db_path=db_path)

    if not engine or not engine.backend:
        return MCPAgentRegistryModel()

    agents: list[MCPAgent] = []
    agents.extend(_fetch_prompt_agents(engine))
    agents.extend(_fetch_specialist_agents(engine))

    tools = _fetch_tools(engine)
    agents.extend(_synthesize_partition_agents(tools, {a.name for a in agents}))

    return MCPAgentRegistryModel(agents=agents, tools=tools)


def _fetch_prompt_agents(engine: Any) -> list[MCPAgent]:
    """Fetch Prompt-based agents from the KG."""
    agents: list[MCPAgent] = []
    try:
        prompt_rows = engine.backend.execute(
            "MATCH (p:Prompt) RETURN p.name AS name, p.description AS descriptionription, p.capabilities AS capabilities, p.system_prompt AS system_prompt, p.json_blueprint AS json_blueprint"
        )
        for row in prompt_rows:
            blueprint = row.get("json_blueprint")
            if isinstance(blueprint, str):
                try:
                    blueprint = json.loads(blueprint)
                except Exception:
                    try:
                        import ast

                        blueprint = ast.literal_eval(blueprint)
                    except Exception:
                        logger.debug(
                            f"Failed to parse json_blueprint as JSON or literal: {blueprint[:100]}..."
                        )

            if blueprint and not isinstance(blueprint, dict):
                logger.debug(
                    f"json_blueprint for {row.get('name')} is not a dict, type={type(blueprint)}"
                )

            parsed_blueprint: dict[str, Any] | None = (
                blueprint if isinstance(blueprint, dict) else None
            )
            agents.append(
                MCPAgent(
                    name=row.get("name", ""),
                    description=row.get("description", ""),
                    agent_type="specialist",
                    capabilities=row.get("capabilities", []),
                    system_prompt=row.get("system_prompt", ""),
                    json_blueprint=parsed_blueprint,
                )
            )
    except Exception as e:
        logger.debug(f"Failed to fetch Prompt nodes: {e}")
    return agents


def _fetch_specialist_agents(engine: Any) -> list[MCPAgent]:
    """Fetch Agent-type specialist nodes from the KG."""
    agents: list[MCPAgent] = []
    try:
        agent_rows = engine.backend.execute(
            "MATCH (a:Agent) RETURN a.name AS name, a.description AS descriptionription, a.agent_type AS agent_type, a.system_prompt AS system_prompt, a.tool_count AS tool_count, a.mcp_server AS mcp_server"
        )
        for row in agent_rows:
            # CONCEPT:AU-ORCH.adapter.hot-cache-invalidation: Normalize legacy prompt/mcp to unified specialist
            _raw_type = row.get("agent_type", "specialist")
            _agent_type = _raw_type if _raw_type == "a2a" else "specialist"
            agents.append(
                MCPAgent(
                    name=row.get("name", "unknown"),
                    description=row.get("description", ""),
                    agent_type=_agent_type,
                    system_prompt=row.get("system_prompt", ""),
                    tool_count=row.get("tool_count", 0),
                    mcp_server=row.get("mcp_server"),
                )
            )
    except Exception as e:
        logger.debug(f"Failed to fetch specialist agents from KG: {e}")
    return agents


def _fetch_tools(engine: Any) -> list[MCPToolInfo]:
    """Fetch Tool nodes from the KG."""
    tools: list[MCPToolInfo] = []
    try:
        tool_rows = engine.backend.execute(
            "MATCH (t:Tool) RETURN t.name, t.description, t.mcp_server, t.relevance_score, t.tags, t.requires_approval"
        )
        for row in tool_rows:
            tools.append(
                MCPToolInfo(
                    name=row.get("t.name", ""),
                    description=row.get("t.description", ""),
                    mcp_server=row.get("t.mcp_server", "unknown"),
                    relevance_score=row.get("t.relevance_score", 0),
                    all_tags=row.get("t.tags", []),
                    requires_approval=row.get("t.requires_approval", False),
                )
            )
    except Exception as e:
        logger.debug(f"Failed to fetch Tool nodes: {e}")
    return tools


def _synthesize_partition_agents(
    tools: list[MCPToolInfo],
    existing_agent_names: set[str],
) -> list[MCPAgent]:
    """Synthesize partition-based agents from tool tags.

    CONCEPT:AU-ORCH.adapter.hot-cache-invalidation — Re-derive Server Agents from Tools (Dynamic Partitioning at read-time)
    """
    partitions: dict[str, list[MCPToolInfo]] = {}
    for t in tools:
        tags = t.all_tags if t.all_tags else ([t.tag] if t.tag else [])
        server_tag = (
            t.mcp_server.lower()
            .replace("-mcp", "")
            .replace("_mcp", "")
            .replace("-manager", "")
            .replace("-agent", "")
            .replace("-server", "")
        )
        if not tags or tags == ["general"]:
            all_partition_tags = {f"{t.mcp_server}_general"}
        else:
            all_partition_tags = set(tags)
            all_partition_tags.add(server_tag)

        for tag in all_partition_tags:
            if tag not in partitions:
                partitions[tag] = []
            partitions[tag].append(t)

    agents: list[MCPAgent] = []
    for tag, partition_tools in partitions.items():
        if tag in existing_agent_names:
            continue

        mcp_servers = list(set(t.mcp_server for t in partition_tools))
        primary_server = mcp_servers[0] if mcp_servers else "unknown"

        agents.append(
            MCPAgent(
                name=tag,
                description=f"Dynamically synthesized agent for {tag} capabilities.",
                agent_type="specialist",
                system_prompt=f"You are the {tag} specialist.",
                tool_count=len(partition_tools),
                mcp_server=primary_server,
                tools=[t.name for t in partition_tools],
                capabilities=list(
                    set(
                        c_tag
                        for t in partition_tools
                        for c_tag in (
                            t.all_tags if t.all_tags else ([t.tag] if t.tag else [])
                        )
                    )
                ),
            )
        )

    return agents


def get_discovery_registry() -> MCPAgentRegistryModel:
    """Load the unified agent discovery registry (cached).

    CONCEPT:AU-ORCH.adapter.hot-cache-invalidation — Hot Cache Layer

    Returns the registry from the in-memory cache.  On first call,
    populates the cache from the Knowledge Graph.  Subsequent calls
    are O(1) until ``invalidate_registry_cache()`` is called.

    Returns:
        The populated MCPAgentRegistryModel.
    """
    return _RegistryCache.get_registry()


def get_relevant_specialists(
    query: str,
    engine: Any | None = None,
    top_n: int = 7,
) -> list[MCPAgent]:
    """Return the top-N adaptive_agent_router most relevant to a query.

    CONCEPT:AU-ORCH.adapter.hot-cache-invalidation — Hot Cache Layer

    Uses KG discovery results (hybrid search + tool matching) to filter
    the full specialist list down to the most relevant agents for a
    given query.  Falls back to the full list if KG discovery returns
    nothing or the engine is unavailable.

    Args:
        query: The user query to match against.
        engine: Optional ``IntelligenceGraphEngine`` for hybrid search.
        top_n: Maximum number of adaptive_agent_router to return.

    Returns:
        A list of the most relevant ``MCPAgent`` objects.
    """
    registry = get_discovery_registry()
    all_agents = registry.agents

    if not all_agents:
        return []

    if not engine or not query:
        return all_agents[:top_n]

    # Use hybrid search to find relevant nodes
    try:
        results = engine.search_hybrid(query, top_k=top_n * 3)
        matched_names: set[str] = set()
        for r in results:
            name = r.get("name", "")
            if name:
                matched_names.add(name.lower())
            # Also check the node type for agent/prompt matches
            node_type = str(r.get("type", "")).lower()
            if node_type in ("agent", "prompt"):
                matched_names.add(name.lower())

        # Score agents by whether they appear in search results
        relevant = [a for a in all_agents if a.name.lower() in matched_names]

        if relevant:
            return relevant[:top_n]
    except Exception as e:
        logger.debug(f"Hybrid search for adaptive_agent_router failed: {e}")

    # Fallback: return all agents (capped)
    return all_agents[:top_n]


def load_node_agents_registry() -> MCPAgentRegistryModel:
    """Legacy alias for get_discovery_registry."""
    return get_discovery_registry()


def load_mcp_config() -> MCPConfigModel:
    """Retrieve the global MCP server configuration from the workspace.

    Loads the mcp_config.json file which contains the definitions of
    external MCP servers (e.g., Docker, GitHub) and their connection
    parameters.

    Returns:
        An MCPConfigModel object containing server definitions and settings.

    """
    path = get_workspace_path(CORE_FILES["MCP_CONFIG"])
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return MCPConfigModel.model_validate(data)
        except Exception:
            return MCPConfigModel()
    return MCPConfigModel()


def save_mcp_config(config: MCPConfigModel):
    """Persist the MCP configuration model back to the workspace file.

    Args:
        config: The MCPConfigModel to be saved.

    """
    path = get_workspace_path(CORE_FILES["MCP_CONFIG"])
    path.write_text(config.model_dump_json(indent=2), encoding="utf-8")


def emit_graph_event(eq: asyncio.Queue[Any] | None, event_type: str, **kwargs):
    """Emit a standardized graph event for real-time UI visualization.

    Formats the event data as a sideband part compatible with the
    Agentic UI streaming protocol, allowing the frontend to visualize
    graph progression and tool activity.  Also emits a structured log
    line so the full execution trace is visible in server-side logs
    without requiring the UI.

    Args:
        eq: The asynchronous event queue to publish to.
        event_type: A string identifier for the event category.
        **kwargs: Additional metadata to include in the event payload.

    """
    ts = time.time()
    trace_kwargs = {k: v for k, v in kwargs.items() if k != "timestamp"}
    _log_graph_trace(event_type, ts, **trace_kwargs)

    if not eq:
        return

    try:
        eq.put_nowait(
            {
                "type": "data-graph-event",
                "data": {
                    "event": event_type,
                    "timestamp": ts,
                    **kwargs,
                },
            }
        )
    except Exception as e:
        logger.warning(f"Failed to emit graph event '{event_type}': {e}")


# ---------------------------------------------------------------------------
# Structured graph trace logging
# ---------------------------------------------------------------------------

_graph_trace_logger = logging.getLogger("agent_utilities.graph.trace")

_PHASE_MAP: dict[str, str] = {
    # ── Lifecycle ──────────────────────────────────────────────────────
    "graph_start": "LIFECYCLE",
    "graph_complete": "LIFECYCLE",
    "node_start": "LIFECYCLE",
    "node_complete": "LIFECYCLE",
    # ── Safety & Policy ───────────────────────────────────────────────
    "safety_warning": "SAFETY",
    # ── Routing & Planning ────────────────────────────────────────────
    "routing_started": "ROUTING",
    "routing_completed": "ROUTING",
    "plan_created": "PLANNING",
    "replanning_started": "REPLANNING",
    "replanning_completed": "REPLANNING",
    # ── Dispatch ──────────────────────────────────────────────────────
    "step_dispatched": "DISPATCH",
    "batch_dispatched": "DISPATCH",
    # ── Context Enrichment ────────────────────────────────────────────
    "context_gap_detected": "ENRICHMENT",
    # ── Specialist Execution ──────────────────────────────────────────
    "specialist_enter": "EXECUTION",
    "specialist_exit": "EXECUTION",
    "specialist_fallback": "FALLBACK",
    "expert_metadata": "EXECUTION",
    "expert_thinking": "EXECUTION",
    "expert_warning": "EXECUTION",
    "expert_text": "EXECUTION",
    "expert_complete": "EXECUTION",
    "tools_bound": "EXECUTION",
    "subagent_started": "EXECUTION",
    "subagent_completed": "EXECUTION",
    "subagent_thought": "EXECUTION",
    # ── Tool Calls ────────────────────────────────────────────────────
    "expert_tool_call": "TOOL_CALL",
    "subagent_tool_call": "TOOL_CALL",
    "tool_result": "TOOL_RESULT",
    # ── Parallel / Orthogonal Regions ─────────────────────────────────
    "orthogonal_regions_start": "PARALLEL",
    "orthogonal_regions_complete": "PARALLEL",
    "region_start": "PARALLEL",
    "region_complete": "PARALLEL",
    # ── Verification & Synthesis ──────────────────────────────────────
    "verification_result": "VERIFICATION",
    "agent_node_delta": "SYNTHESIS",
    "synthesis_fallback": "SYNTHESIS",
    # ── Human-in-the-Loop ─────────────────────────────────────────────
    "approval_required": "APPROVAL",
    "approval_resolved": "APPROVAL",
    "elicitation": "APPROVAL",
    # ── Recovery & Termination ────────────────────────────────────────
    "error_recovery_replan": "RECOVERY",
    "error_recovery_terminal": "RECOVERY",
    "graph_force_terminated": "TERMINATION",
    # ── Council Deliberation ──────────────────────────────────────────
    "council_started": "COUNCIL",
    "council_stage": "COUNCIL",
    "council_advisor_complete": "COUNCIL",
    "council_reviewer_complete": "COUNCIL",
    "council_completed": "COUNCIL",
    # ── KG-Driven Graph Materialization (CONCEPT:AU-ORCH.adapter.kg-graph-materialization) ─────────────
    "kg_query_start": "KG_BRIDGE",
    "kg_query_complete": "KG_BRIDGE",
    "kg_template_resolved": "KG_BRIDGE",
    "kg_prompt_injected": "KG_BRIDGE",
    "kg_topology_materialized": "KG_BRIDGE",
}


def _log_graph_trace(event_type: str, timestamp: float, **kwargs):
    """Emit a structured log line for a graph event."""
    phase = _PHASE_MAP.get(event_type, "GRAPH")
    detail_parts: list[str] = []

    for key in ("agent", "expert", "node_id", "id", "domain", "server"):
        if key in kwargs:
            detail_parts.append(f"{key}={kwargs[key]}")
    for key in ("count", "score", "batch_size", "attempt", "duration_ms"):
        if key in kwargs:
            detail_parts.append(f"{key}={kwargs[key]}")
    if "tool_name" in kwargs:
        detail_parts.append(f"tool={kwargs['tool_name']}")
    if "success" in kwargs:
        detail_parts.append(f"ok={kwargs['success']}")
    if "message" in kwargs and event_type in ("expert_warning", "safety_warning"):
        detail_parts.append(f"msg={kwargs['message'][:120]}")

    detail = " ".join(detail_parts) if detail_parts else ""
    _graph_trace_logger.info(f"[{phase}] {event_type} {detail}".rstrip())


def _render_prompt_payload(data: dict[str, Any]) -> str:
    """Render a prompt blueprint dict to the string the LLM should see.

    Prefers the modern JSON blueprint schema (with a ``content`` key) and
    falls back to :class:`StructuredPrompt` for legacy ``task``/``input``
    payloads. The returned string is always valid JSON so callers can
    forward it directly to ``system_prompt=`` kwargs.
    """
    content = data.get("content")
    if isinstance(content, str) and content.strip():
        return json.dumps(data, indent=2)

    try:
        from agent_utilities.prompting.structured import StructuredPrompt

        return StructuredPrompt.model_validate(data).render()
    except Exception as e:
        logger.debug(f"StructuredPrompt validation failed: {e}")
        return json.dumps(data, indent=2)


def load_specialized_prompts(prompt_name: str) -> str:
    """Load a specialized agent persona prompt from the registry defined path.

    The loader checks, in order:

    1. A matching agent in the Knowledge Graph registry with a
       ``json_blueprint`` payload.
    2. An agent whose ``prompt_file`` points at a local ``*.json`` file.
    3. A fallback ``agent_utilities/prompts/<prompt_name>.json`` file.

    Args:
        prompt_name: The slugified name/tag of the expert (e.g. ``router``).

    Returns:
        The specialized system prompt serialized as a JSON string.

    """
    registry = get_discovery_registry()
    agent = next((a for a in registry.agents if a.name == prompt_name), None)

    if agent:
        if agent.json_blueprint:
            return _render_prompt_payload(dict(agent.json_blueprint))

        if agent.prompt_file:
            # Check if it's a JSON file
            prompt_path = (Path(__file__).parent.parent / agent.prompt_file).resolve()
            if prompt_path.suffix == ".json" and prompt_path.exists():
                data = json.loads(prompt_path.read_text(encoding="utf-8"))
                return _render_prompt_payload(data)

    # Unified JSON loading from prompts/
    json_path = (
        Path(__file__).parent.parent / "prompts" / f"{prompt_name}.json"
    ).resolve()
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            return _render_prompt_payload(data)
        except Exception as e:
            logger.warning(
                f"Failed to load structured prompt JSON for '{prompt_name}': {e}"
            )

    logger.warning(
        f"Specialized prompt for '{prompt_name}' not found in registry "
        "or prompts/*.json."
    )
    return f"You are a helpful assistant specialized in {prompt_name}."


# --- Migrated from mcp/config_loader.py ---
import os
import shutil
import tempfile


def load_mcp_servers_from_config(config_path: str | Path) -> list[Any]:
    """Load and expand environment variables in an MCP config file.

    Reads the specified mcp_config.json, expands any environment variable
    placeholders (e.g., ${API_KEY}), performs robust pre-validation of
    executable commands in the PATH, and initializes the server objects.

    Args:
        config_path: Path to the mcp_config.json file.

    Returns:
        A list of initialized pydantic_ai.mcp.MCPServer objects (technically
        MCPToolSet in newer versions, but returned as list of servers here).

    """
    from pydantic_ai.mcp import load_mcp_toolsets

    from agent_utilities.base_utilities import expand_env_vars

    try:
        path = Path(config_path)
        if not path.exists():
            return []

        content = path.read_text()
        expanded_content = expand_env_vars(content)

        # Robust Validation: Check if commands exist before pydantic-ai tries to start them
        try:
            config_data = json.loads(expanded_content)
            mcp_servers = config_data.get("mcpServers", {})
            modified = False

            for name, cfg in mcp_servers.items():
                command = cfg.get("command")
                if command:
                    # Resolve command path with explicit ~/.local/bin support
                    search_path = os.environ.get("PATH", "")
                    local_bin = str(Path.home() / ".local" / "bin")
                    if local_bin not in search_path:
                        search_path = f"{local_bin}:{search_path}"

                    resolved = shutil.which(command, path=search_path)
                    if not resolved:
                        logger.warning(
                            f"MCP Config: Command '{command}' for server '{name}' NOT FOUND in PATH ({search_path}). Startup will likely fail."
                        )
                    else:
                        logger.debug(
                            f"MCP Config: Resolved command '{command}' to '{resolved}'"
                        )

                    # Ensure PATH and PYTHONPATH are preserved if not explicitly set
                    if "env" not in cfg:
                        cfg["env"] = {}

                    if "PATH" not in cfg["env"]:
                        cfg["env"]["PATH"] = search_path
                    if "PYTHONPATH" not in cfg["env"] and "PYTHONPATH" in os.environ:
                        cfg["env"]["PYTHONPATH"] = os.environ.get("PYTHONPATH", "")

                    # Suppress RequestsDependencyWarning in subprocesses
                    if "PYTHONWARNINGS" not in cfg["env"]:
                        cfg["env"][
                            "PYTHONWARNINGS"
                        ] = "ignore:urllib3 (2.3.0) or chardet"
                    else:
                        if "ignore:urllib3" not in cfg["env"]["PYTHONWARNINGS"]:
                            cfg["env"][
                                "PYTHONWARNINGS"
                            ] += ",ignore:urllib3 (2.3.0) or chardet"

                    # Token forwarding: propagate user session token to
                    # MCP subprocesses for delegated authentication.
                    # CONCEPT:AU-OS.config.secrets-authentication — Secrets & Authentication
                    if "AGENT_USER_TOKEN" not in cfg["env"]:
                        _user_token = os.environ.get("AGENT_USER_TOKEN")
                        if not _user_token:
                            try:
                                from agent_utilities.security.secrets_client import (
                                    create_secrets_client,
                                )

                                _sc = create_secrets_client()
                                _user_token = _sc.get("session_token")
                            except Exception:  # nosec B110
                                pass
                        if _user_token:
                            cfg["env"]["AGENT_USER_TOKEN"] = _user_token

                    modified = True

            # Centralized Knowledge Graph Coordination Protocol
            # CONCEPT:AU-KG.coordination.centralized-kg-coordination-protocol
            coordinated_kg_server = None
            if "agent-utilities-kg" in mcp_servers:
                from agent_utilities.core.config import DEFAULT_VALIDATION_MODE

                if not DEFAULT_VALIDATION_MODE:
                    from agent_utilities.mcp.kg_coordinator import KGCoordinator
                    from agent_utilities.mcp.toolset_factory import build_http_toolset

                    kg_host = os.getenv("KG_SERVER_HOST", "127.0.0.1")
                    kg_port = int(os.getenv("KG_SERVER_PORT", "8100"))

                    logger.info(
                        "Coordinated KG check: Intercepting agent-utilities-kg server config"
                    )
                    try:
                        KGCoordinator.get_kg_client(host=kg_host, port=kg_port)

                        # Remove from dict to prevent stdio spawning by pydantic-ai
                        mcp_servers.pop("agent-utilities-kg")
                        modified = True

                        # Create SSE client directly
                        coordinated_kg_server = build_http_toolset(
                            f"http://{kg_host}:{kg_port}/sse",
                            timeout=60,
                            toolset_id="agent-utilities-kg",
                        )
                    except Exception as e:
                        logger.error(f"Failed to coordinate centralized KG server: {e}")

            if modified:
                expanded_content = json.dumps(config_data)
        except Exception as e:
            logger.warning(f"MCP Config: Pre-validation failed: {e}")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            tmp.write(expanded_content)
            tmp_path = tmp.name

        try:
            servers = load_mcp_toolsets(tmp_path)
            # Re-attach IDs from config
            config_data = json.loads(expanded_content)
            mcp_servers_cfg = config_data.get("mcpServers", {})

            # Match by command and args as a heuristic if pydantic-ai doesn't preserve order or names
            for ts in servers:
                # pydantic-ai objects might not have a clean way to match back,
                # but they usually follow the order in the JSON.
                pass

            # Better: If we have a list, and the config had a dict, they MIGHT match by order
            # However, pydantic-ai load_mcp_servers is internal.
            # I'll just set the .id if they are list components.
            for i, (name, cfg) in enumerate(mcp_servers_cfg.items()):
                if i < len(servers):
                    servers[i].id = name
                    logger.debug(f"MCP Config: Loaded server '{name}'")

            # Add coordinated KG server back if present
            if coordinated_kg_server is not None:
                servers.append(coordinated_kg_server)
                logger.info(
                    "Coordinated KG check: successfully appended MCP toolset client for agent-utilities-kg"
                )

            return servers
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    except Exception as e:
        logger.error(f"Failed to load MCP config {config_path}: {e}")
        return []
