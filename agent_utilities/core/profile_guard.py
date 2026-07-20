#!/usr/bin/python
"""Production profile guard.

Provides a runtime validator that refuses to let "toy"/single-node defaults
through when the deployment declares itself a production profile.

A deployment is treated as production when the ``APP_PROFILE`` environment
variable (case-insensitive) is one of :data:`PROD_PROFILE_VALUES` (``prod`` /
``production``). In that mode several settings that are perfectly fine for local
development become unsafe because they pin the system to a single process or a
single machine and therefore cannot scale or survive a restart:

* ``graph_persistence_type`` of ``file`` / ``sqlite`` -> single-host, non-shardable.
* ``a2a_broker`` / ``a2a_storage`` other than ``epistemic_graph`` -> the
  native durable FastA2A delivery and CAS state planes are bypassed.
* ``USAGE_DB_BACKEND`` other than ``postgres`` -> per-process analytics state.
* usage tracking or the configured OTLP signal path disabled -> operational
  evidence and tenant-aware SLOs are incomplete.
* transcript/content capture or an absent HMAC identity key -> durable
  observability may retain source identities instead of opaque references.
* disabled identity/tenant/ACL enforcement or an explicitly insecure transport
  or development-only host capability -> the served boundary can be bypassed
  even though the deployment calls itself production.
* eager graph recovery, an unbounded resident graph cache, or an unpaged lazy
  open -> restart and memory pressure can make production availability unsafe.
* a packaged local durable engine without an external data-key reference ->
  transaction recovery and encryption-at-rest cannot start safely.

:func:`assert_production_safe` raises :class:`ProductionProfileError` listing every
offending setting so an operator can fix them all at once, instead of discovering
them one redeploy at a time.
"""

from __future__ import annotations

import ipaddress
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from agent_utilities.core.config import setting

if TYPE_CHECKING:  # pragma: no cover - typing only
    from agent_utilities.core.config import AgentConfig

#: Environment-variable name that selects the deployment profile.
PROFILE_ENV_VAR = "APP_PROFILE"

#: Values of ``APP_PROFILE`` that mean "this is production".
PROD_PROFILE_VALUES = frozenset({"prod", "production"})

#: ``graph_persistence_type`` values that are single-host / non-production.
UNSAFE_GRAPH_PERSISTENCE = frozenset({"file", "sqlite"})


def _is_loopback_host(value: str | None) -> bool:
    """Return whether a listener/URL host is explicitly loopback."""

    rendered = str(value or "").strip().strip("[]").lower()
    if rendered in {"localhost", "localhost."}:
        return True
    try:
        return ipaddress.ip_address(rendered).is_loopback
    except ValueError:
        return False


def _is_remote_plaintext_http(value: object) -> bool:
    """Return whether ``value`` is an HTTP URL whose peer is not loopback."""

    rendered = str(value or "").strip()
    if not rendered:
        return False
    try:
        parsed = urlsplit(rendered)
    except ValueError:
        return False
    return parsed.scheme.lower() == "http" and not _is_loopback_host(parsed.hostname)


def _comma_values(value: object) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(value or "").split(",") if part.strip())


class ProductionProfileError(RuntimeError):
    """Raised when a production profile is configured with unsafe toy settings.

    The :attr:`offending` attribute holds the list of human-readable problem
    descriptions, one per offending setting.
    """

    def __init__(self, offending: list[str], profile: str) -> None:
        self.offending = list(offending)
        self.profile = profile
        joined = "\n".join(f"  - {item}" for item in self.offending)
        super().__init__(
            f"Configuration is not production-safe under {PROFILE_ENV_VAR}={profile!r}.\n"
            f"The following settings use single-host/in-memory defaults that cannot "
            f"scale or survive a restart:\n{joined}\n"
            f"Set production-grade backends (e.g. graph_persistence_type='postgresql', "
            f"a2a_broker='epistemic_graph', a2a_storage='epistemic_graph', "
            f"USAGE_DB_BACKEND='postgres') and runtime-injected secret/OTLP references."
        )


def is_production_profile(profile: str | None = None) -> bool:
    """Return True when the active deployment profile is production.

    :param profile: Explicit profile value. When ``None`` the ``APP_PROFILE``
        environment variable is consulted.
    """
    if profile is None:
        profile = setting(PROFILE_ENV_VAR)
    if not profile:
        return False
    return profile.strip().lower() in PROD_PROFILE_VALUES


def collect_production_violations(config: AgentConfig) -> list[str]:
    """Return a list of production-safety violations for ``config``.

    The list is empty when the configuration is production-safe. This function
    does *not* consult ``APP_PROFILE``; it always evaluates the rules, so callers
    can report problems regardless of the active profile.
    """
    offending: list[str] = []

    gpt = (getattr(config, "graph_persistence_type", "") or "").strip().lower()
    if gpt in UNSAFE_GRAPH_PERSISTENCE:
        offending.append(
            f"graph_persistence_type={gpt!r} is single-host and non-shardable; "
            f"use a distributed backend (e.g. 'postgresql')."
        )

    # Production is a fail-secure posture, not merely a durability profile.
    # Keep custom CAs, mTLS, and secret-backed named profiles fully supported;
    # only explicit verification/authentication bypasses are prohibited.
    jwks_uri = str(getattr(config, "auth_jwt_jwks_uri", "") or "").strip()
    if not jwks_uri:
        offending.append("auth_jwt_jwks_uri is required for verified identity.")
    elif _is_remote_plaintext_http(jwks_uri):
        offending.append(
            "auth_jwt_jwks_uri uses plaintext HTTP to a remote identity provider; "
            "production identity discovery requires HTTPS."
        )
    if not str(getattr(config, "auth_jwt_issuer", "") or "").strip():
        offending.append("auth_jwt_issuer is required for issuer binding.")
    if not str(getattr(config, "auth_jwt_audience", "") or "").strip():
        offending.append("auth_jwt_audience is required for audience binding.")
    if not str(getattr(config, "kg_policy_version", "") or "").strip():
        offending.append("kg_policy_version is required for policy pinning.")
    insecure_development_options = ("kg_loop_allow_host_validation",)
    for field_name in insecure_development_options:
        if bool(getattr(config, field_name, False)):
            offending.append(
                f"{field_name} enables a development-only host capability and "
                "must be disabled in production."
            )

    if bool(getattr(config, "debug", False)):
        offending.append("debug is enabled; production must not expose debug behavior.")

    tool_guard_mode = (
        str(getattr(config, "tool_guard_mode", "strict") or "strict").strip().lower()
    )
    if tool_guard_mode not in {"on", "strict"}:
        offending.append(
            "tool_guard_mode must be 'on' or 'strict'; production cannot disable "
            "the governed tool boundary."
        )

    # The REST boundary is always JWT/session protected. A remotely reachable
    # listener additionally needs an exact Host allowlist and either direct TLS
    # identity or an explicitly declared TLS-terminating ingress.
    allowed_hosts = _comma_values(getattr(config, "allowed_hosts", None))
    if any("*" in value for value in allowed_hosts):
        offending.append(
            "allowed_hosts contains a wildcard; production requires exact authorities."
        )
    allowed_origins = _comma_values(getattr(config, "allowed_origins", None))
    if any("*" in value for value in allowed_origins):
        offending.append(
            "allowed_origins contains a wildcard; production requires exact origins."
        )
    if bool(getattr(config, "cors_allow_credentials", False)) and not allowed_origins:
        offending.append(
            "cors_allow_credentials requires an explicit production origin allowlist."
        )

    certfile = str(getattr(config, "server_tls_certfile", "") or "").strip()
    keyfile = str(getattr(config, "server_tls_keyfile", "") or "").strip()
    tls_terminated = bool(getattr(config, "server_tls_terminated", False))
    if bool(certfile) != bool(keyfile):
        offending.append(
            "server TLS identity is incomplete; configure both certificate and key."
        )
    listener = str(getattr(config, "host", "") or "").strip()
    if not _is_loopback_host(listener):
        if not allowed_hosts:
            offending.append(
                "a non-loopback REST listener requires an exact allowed_hosts list."
            )
        if not tls_terminated and not (certfile and keyfile):
            offending.append(
                "a non-loopback REST listener requires direct TLS or an explicitly "
                "declared TLS-terminating ingress."
            )

    mcp_allowed_hosts = _comma_values(getattr(config, "mcp_allowed_hosts", None))
    if any("*" in value for value in mcp_allowed_hosts):
        offending.append(
            "mcp_allowed_hosts contains a wildcard; production requires exact authorities."
        )
    mcp_certfile = str(getattr(config, "mcp_tls_certfile", "") or "").strip()
    mcp_keyfile = str(getattr(config, "mcp_tls_keyfile", "") or "").strip()
    if bool(mcp_certfile) != bool(mcp_keyfile):
        offending.append(
            "MCP TLS identity is incomplete; configure both certificate and key."
        )

    alert_port = getattr(config, "messaging_alert_intake_port", None)
    alert_remote = bool(getattr(config, "messaging_alert_intake_allow_remote", False))
    if alert_remote:
        offending.append(
            "messaging_alert_intake_allow_remote is a plaintext remote-serving escape "
            "hatch and must be disabled in production."
        )
    if alert_port is not None:
        if not str(
            getattr(config, "messaging_alert_intake_token_ref", "") or ""
        ).strip():
            offending.append(
                "messaging alert intake requires a runtime token reference."
            )
        if not _is_loopback_host(
            str(getattr(config, "messaging_alert_intake_host", "") or "")
        ):
            offending.append(
                "messaging alert intake must bind to loopback in production."
            )

    # Reject remote plaintext on every native HTTP endpoint that can be
    # expressed directly in AgentConfig. Connection-profile refs are validated
    # by their connector transport at materialization time.
    endpoint_fields: list[tuple[str, object]] = [
        ("mcp_url", getattr(config, "mcp_url", None)),
        ("fleet_mcp_url_template", getattr(config, "fleet_mcp_url_template", None)),
        ("openai_base_url", getattr(config, "openai_base_url", None)),
        ("deepseek_base_url", getattr(config, "deepseek_base_url", None)),
        ("vllm_base_url", getattr(config, "vllm_base_url", None)),
        ("kg_rerank_base_url", getattr(config, "kg_rerank_base_url", None)),
    ]
    if bool(getattr(config, "enable_otel", False)):
        endpoint_fields.append(
            (
                "otel_exporter_otlp_endpoint",
                getattr(config, "otel_exporter_otlp_endpoint", None),
            )
        )
    if str(getattr(config, "eunomia_type", "none") or "none").lower() == "remote":
        endpoint_fields.append(
            ("eunomia_remote_url", getattr(config, "eunomia_remote_url", None))
        )
    if bool(getattr(config, "kg_fuseki_publish", False)):
        endpoint_fields.append(
            ("kg_fuseki_endpoint", getattr(config, "kg_fuseki_endpoint", None))
        )
    endpoint_fields.extend(
        (f"sparql_endpoints[{index}]", endpoint)
        for index, endpoint in enumerate(getattr(config, "sparql_endpoints", ()) or ())
    )
    for index, model in enumerate(getattr(config, "chat_models", ()) or ()):
        endpoint_fields.append(
            (f"chat_models[{index}].base_url", getattr(model, "base_url", None))
        )
    for index, model in enumerate(getattr(config, "embedding_models", ()) or ()):
        current = model
        depth = 0
        while current is not None and depth < 2:
            label = f"embedding_models[{index}]"
            if depth:
                label += ".fallback"
            endpoint_fields.append(
                (f"{label}.base_url", getattr(current, "base_url", None))
            )
            current = getattr(current, "fallback", None)
            depth += 1
    for field_name, endpoint in endpoint_fields:
        if _is_remote_plaintext_http(endpoint):
            offending.append(
                f"{field_name} uses plaintext HTTP to a remote service; use HTTPS "
                "or a loopback transport in production."
            )

    broker = (getattr(config, "a2a_broker", "") or "").strip().lower()
    if broker != "epistemic_graph":
        offending.append(
            "a2a_broker must be 'epistemic_graph'; the native durable broker is "
            "the sole current FastA2A delivery plane."
        )

    # The Kafka AgentBus topology creates one consumer group per
    # recipient/subscription. That topology is useful at modest scale but is not
    # the million-agent production plane: it causes every subscriber to scan
    # shared-topic traffic. Production therefore uses the engine broker's
    # tenant-qualified durable inbox queues; Kafka remains available for the
    # bounded WorkItem executor pool.
    bus_backend = (
        str(getattr(config, "agent_bus_log_backend", "") or "").strip().lower()
    )
    if bus_backend != "engine":
        offending.append(
            "agent_bus_log_backend must be 'engine' in production; the Kafka "
            "AgentBus Kafka topology creates a consumer group per "
            "recipient, and graph fallback is not a scalable delivery plane."
        )

    storage = (getattr(config, "a2a_storage", "") or "").strip().lower()
    if storage != "epistemic_graph":
        offending.append(
            "a2a_storage must be 'epistemic_graph'; native CAS-fenced records are "
            "the sole current FastA2A state plane."
        )

    # Plan 08 Synergy 2: the single reactive ledger (telemetry, KG write-back,
    # UI fan-out, replay) requires a distributed Kafka/Redpanda broker. Without
    # kafka_bootstrap_servers the event backend silently falls back to the
    # single-process, non-durable in-memory EventBus.
    kafka = (getattr(config, "kafka_bootstrap_servers", "") or "").strip()
    if not kafka:
        offending.append(
            "kafka_bootstrap_servers is unset; the reactive event ledger falls "
            "back to the single-process in-memory EventBus (no durability, no "
            "cross-node fan-out). Set it to a Redpanda/Kafka cluster for prod."
        )

    resident_limit = int(getattr(config, "epistemic_graph_max_resident_graphs", 0) or 0)
    if resident_limit <= 0:
        offending.append(
            "epistemic_graph_max_resident_graphs must be positive in production; "
            "0 leaves the resident graph cache unbounded."
        )

    page_size = int(getattr(config, "epistemic_graph_lazy_open_page_size", 0) or 0)
    if page_size <= 0:
        offending.append(
            "epistemic_graph_lazy_open_page_size must be positive in production; "
            "0 performs an unbounded all-at-once lazy open."
        )

    node_limit = int(getattr(config, "epistemic_graph_max_nodes_per_graph", 0) or 0)
    if node_limit <= 0:
        offending.append(
            "epistemic_graph_max_nodes_per_graph must be positive in production; "
            "0 leaves one graph's resident node set unbounded."
        )

    # Configured external contacts are deployed and keyed independently. When
    # this process owns the packaged local engine lifecycle, production must
    # resolve its durable data key from an external bootstrap source; the
    # non-production tiny-only generated key is intentionally unavailable.
    local_engine_managed = not bool(
        getattr(config, "graph_service_endpoints", None) or []
    )
    if (
        local_engine_managed
        and not str(
            getattr(config, "epistemic_graph_encryption_key_ref", "") or ""
        ).strip()
    ):
        offending.append(
            "epistemic_graph_encryption_key_ref is required for a packaged local "
            "production engine."
        )

    usage_backend = (
        (getattr(config, "usage_db_backend", "sqlite") or "sqlite").strip().lower()
    )
    if usage_backend != "postgres":
        offending.append(
            f"usage_db_backend={usage_backend!r} is not a shared production "
            "analytics store; use 'postgres' for tenant-aware usage/SLO data."
        )

    if not bool(getattr(config, "usage_tracking_enabled", True)):
        offending.append(
            "usage_tracking_enabled is false; production requires tenant-aware "
            "usage and cost evidence."
        )

    content_retention = (
        str(getattr(config, "usage_content_retention", "metadata") or "metadata")
        .strip()
        .lower()
    )
    if content_retention != "metadata":
        offending.append(
            "usage_content_retention must be 'metadata' in production; the "
            "analytics store cannot retain prompts, thinking text, or tool inputs."
        )

    if bool(getattr(config, "langfuse_capture_content", False)):
        offending.append(
            "langfuse_capture_content is true; production traces must remain "
            "metadata-only."
        )

    langfuse_public = bool(getattr(config, "langfuse_public_key_ref", None))
    langfuse_secret = bool(getattr(config, "langfuse_secret_key_ref", None))
    langfuse_enabled = bool(
        getattr(config, "langfuse_mcp_enabled", False)
        or getattr(config, "kg_failure_evolution", False)
        or getattr(config, "trace_export_enabled", False)
    )
    if langfuse_public != langfuse_secret:
        offending.append(
            "Langfuse credential configuration is partial; configure a runtime "
            "secret reference for each key."
        )
    elif langfuse_enabled and not langfuse_public:
        offending.append(
            "a Langfuse integration is enabled without a complete runtime "
            "secret-reference pair."
        )
    if langfuse_enabled and not str(
        getattr(config, "langfuse_host", "") or ""
    ).startswith("https://"):
        offending.append("LANGFUSE_HOST must use HTTPS in production.")

    identity_key_ref = str(
        getattr(config, "persistence_identity_hmac_key_ref", "") or ""
    ).strip()
    if not identity_key_ref:
        offending.append(
            "persistence_identity_hmac_key_ref is unset; production durable "
            "identities require a secret-backed HMAC key reference."
        )

    permission_key_ref = str(
        getattr(config, "permissions_signing_key_ref", "") or ""
    ).strip()
    if not permission_key_ref:
        offending.append(
            "permissions_signing_key_ref is unset; production agent identities "
            "require one stable runtime secret reference."
        )

    otel_enabled = bool(getattr(config, "enable_otel", False))
    otel_endpoint = str(
        getattr(config, "otel_exporter_otlp_endpoint", "") or ""
    ).strip()
    if not otel_enabled or not otel_endpoint:
        offending.append(
            "the production OpenTelemetry signal path is incomplete; set "
            "ENABLE_OTEL=true and inject OTEL_EXPORTER_OTLP_ENDPOINT at runtime."
        )

    return offending


def assert_production_safe(config: AgentConfig, *, profile: str | None = None) -> None:
    """Validate ``config`` against production-safety rules.

    When the active profile is *not* production this is a no-op (dev/default
    profiles are unaffected). When it *is* production and any offending setting
    is found, raise :class:`ProductionProfileError` listing every offender.

    :param config: The :class:`AgentConfig` instance to validate.
    :param profile: Optional explicit profile override (otherwise ``APP_PROFILE``).
    :raises ProductionProfileError: If running as production with unsafe settings.
    """
    active_profile = profile if profile is not None else setting(PROFILE_ENV_VAR, "")
    if not is_production_profile(active_profile):
        return

    offending = collect_production_violations(config)
    if offending:
        raise ProductionProfileError(offending, active_profile)
