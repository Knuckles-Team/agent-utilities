#!/usr/bin/python
"""``agent-utilities doctor`` — one holistic health sweep of a deployment.

Like ``brew doctor`` / ``flutter doctor``: runs a battery of independent checks
across every subsystem, each reporting ok / warn / fail / skip with a concrete
**remediation** — and, where one exists, the **skill or command that fixes it** so
the operator (or Claude) can act or auto-fix. The doctor is a thin *aggregator*: it
composes the diagnostics that already exist (config_doctor, shard topology probe,
backend health_check, the hook doctor, the MCP-config validator, secrets resolution)
rather than re-implementing them.

Each check is defensive — a missing optional dependency or an unreachable service
yields ``skip``/``warn``/``fail`` with guidance, never a crash. ``run_doctor`` returns
a structured report; ``fix=True`` runs the conservative, idempotent auto-remediations
(only checks marked ``auto_fixable``).
"""

from __future__ import annotations

import hashlib
import importlib
import ipaddress
import logging
import stat
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

logger = logging.getLogger(__name__)

# Status precedence (worst wins for the overall verdict).
_RANK = {"ok": 0, "skip": 0, "warn": 1, "fail": 2, "error": 2}


def _result(
    name: str,
    status: str,
    detail: str,
    *,
    remediation: str | None = None,
    skill: str | None = None,
    auto_fixable: bool = False,
    data: Any = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "detail": detail,
        "remediation": remediation,
        "skill": skill,
        "auto_fixable": auto_fixable,
        "data": data,
    }


# ── individual checks (each returns one _result; never raises) ──────────────
def _check_python_env() -> dict[str, Any]:
    import platform
    import sys

    try:
        import agent_utilities

        ver = getattr(agent_utilities, "__version__", "unknown")
    except Exception as exc:  # noqa: BLE001
        return _result(
            "python_env",
            "fail",
            f"agent_utilities not importable ({type(exc).__name__})",
            remediation="pip install agent-utilities[all]",
        )
    optional = {}
    for mod, label in (
        ("rdflib", "owl/sparql"),
        ("psycopg", "postgres"),
        ("stardog", "stardog"),
    ):
        optional[label] = importlib.util.find_spec(mod) is not None
    py_ok = sys.version_info >= (3, 10)
    missing = [k for k, v in optional.items() if not v]
    status = "ok" if py_ok else "warn"
    detail = (
        f"Python {platform.python_version()}, agent-utilities {ver}; "
        f"optional extras present: {[k for k, v in optional.items() if v] or 'none'}"
    )
    if missing:
        detail += f"; absent (install if needed): {missing}"
    return _result(
        "python_env",
        status,
        detail,
        remediation=None if py_ok else "agent-utilities needs Python 3.10+",
        data=optional,
    )


def _check_config() -> dict[str, Any]:
    try:
        from agent_utilities.deployment.config_generator import config_doctor

        rep = config_doctor()
    except Exception as exc:  # noqa: BLE001
        return _result(
            "config", "error", f"config_doctor failed ({type(exc).__name__})"
        )
    healthy = rep.get("healthy")
    profile = rep.get("profile", "?")
    if healthy:
        return _result(
            "config", "ok", f"config healthy for profile {profile!r}", data=rep
        )
    # Tiny's durability findings are advisory.
    status = "warn" if profile == "tiny" else "fail"
    return _result(
        "config",
        status,
        f"config needs attention (profile {profile!r}) — see checks",
        remediation="`setup-config doctor` for detail; `setup-config generate --profile <p>` to (re)seed",
        skill="agent-utilities-deployment",
        data=rep,
    )


def _check_evolution_staging() -> dict[str, Any]:
    """Validate the evolution artifact boundary without exposing its path."""

    import os
    from pathlib import Path

    try:
        from agent_utilities.core.config import AgentConfig

        configured = AgentConfig().evolution_staging_root
    except Exception as exc:  # noqa: BLE001
        return _result(
            "evolution_staging",
            "error",
            f"evolution staging configuration unavailable ({type(exc).__name__})",
            data={"configured": False, "usable": False},
        )
    if not configured:
        return _result(
            "evolution_staging",
            "skip",
            "reviewable evolution artifact staging is not configured",
            remediation=(
                "Set EVOLUTION_STAGING_ROOT in AgentConfig before enabling artifact "
                "drafting or materialization."
            ),
            data={"configured": False, "usable": False},
        )
    try:
        raw = Path(configured).expanduser()
        if raw.is_symlink():
            raise PermissionError("symbolic-link root")
        root = raw.resolve(strict=True)
        if not root.is_dir() or not os.access(root, os.R_OK | os.W_OK | os.X_OK):
            raise PermissionError("unusable root")
        if os.name != "nt" and root.stat().st_mode & 0o077:
            return _result(
                "evolution_staging",
                "fail",
                "evolution staging is accessible outside the current account",
                remediation="Restrict the configured staging root to mode 0700.",
                data={"configured": True, "usable": False},
            )
    except Exception as exc:  # noqa: BLE001
        return _result(
            "evolution_staging",
            "fail",
            f"evolution staging is unusable ({type(exc).__name__})",
            remediation=(
                "Create a private non-symlink directory, set mode 0700, and point "
                "EVOLUTION_STAGING_ROOT to it."
            ),
            data={"configured": True, "usable": False},
        )
    return _result(
        "evolution_staging",
        "ok",
        "evolution staging is configured, private, and usable",
        data={"configured": True, "usable": True},
    )


def _check_execution_security() -> dict[str, Any]:
    """Surface dangerous host-execution escape hatches without executing them."""
    try:
        from agent_utilities.core.config import AgentConfig

        cfg = AgentConfig()
    except Exception as exc:  # noqa: BLE001
        return _result(
            "execution_security",
            "error",
            f"execution security configuration unavailable ({type(exc).__name__})",
        )
    hazards: list[str] = []
    if cfg.kg_loop_allow_host_validation:
        hazards.append("develop_loop_host_validation")
    if cfg.messaging_alert_intake_allow_remote:
        hazards.append("remote_alert_intake")
    if cfg.cors_allow_credentials and (
        not cfg.allowed_origins
        or "*" in {value.strip() for value in cfg.allowed_origins.split(",")}
    ):
        return _result(
            "execution_security",
            "fail",
            "credentialed CORS does not have an explicit origin allowlist",
            remediation=(
                "Set ALLOWED_ORIGINS to exact trusted origins or disable "
                "CORS_ALLOW_CREDENTIALS."
            ),
            data={"unsafe_execution_hazards": hazards},
        )
    if cfg.allowed_hosts and "*" in {
        value.strip() for value in cfg.allowed_hosts.split(",")
    }:
        return _result(
            "execution_security",
            "fail",
            "the REST Host-header allowlist contains a blocked wildcard",
            remediation=("Replace ALLOWED_HOSTS=* with exact authority names."),
            data={"unsafe_execution_hazards": hazards},
        )
    listener = cfg.host.strip().strip("[]").lower()
    try:
        loopback_listener = ipaddress.ip_address(listener).is_loopback
    except ValueError:
        loopback_listener = listener == "localhost"
    if not loopback_listener:
        authenticated = bool(cfg.auth_jwt_jwks_uri)
        if not authenticated:
            return _result(
                "execution_security",
                "fail",
                "a non-loopback REST listener has no authentication boundary",
                remediation=("Configure JWT authentication or bind HOST to loopback."),
                data={"unsafe_execution_hazards": hazards},
            )
        if not cfg.allowed_hosts:
            return _result(
                "execution_security",
                "fail",
                "a non-loopback REST listener has no Host-header allowlist",
                remediation="Set ALLOWED_HOSTS to the exact served authority names.",
                data={"unsafe_execution_hazards": hazards},
            )
    if cfg.messaging_alert_intake_port is not None:
        if not cfg.messaging_alert_intake_token_ref:
            return _result(
                "execution_security",
                "fail",
                "messaging alert intake is enabled without a token reference",
                remediation=(
                    "Set MESSAGING_ALERT_INTAKE_TOKEN_REF to a runtime secret-provider "
                    "reference or disable MESSAGING_ALERT_INTAKE_PORT."
                ),
                data={"unsafe_execution_hazards": hazards},
            )
        alert_listener = cfg.messaging_alert_intake_host.strip().strip("[]").lower()
        try:
            alert_loopback = ipaddress.ip_address(alert_listener).is_loopback
        except ValueError:
            alert_loopback = alert_listener in {"localhost", "localhost."}
        if not alert_loopback and not cfg.messaging_alert_intake_allow_remote:
            return _result(
                "execution_security",
                "fail",
                "messaging alert intake requests a non-loopback bind without approval",
                remediation=(
                    "Bind MESSAGING_ALERT_INTAKE_HOST to loopback or explicitly set "
                    "MESSAGING_ALERT_INTAKE_ALLOW_REMOTE=true behind a protected ingress."
                ),
                data={"unsafe_execution_hazards": hazards},
            )
    if hazards:
        return _result(
            "execution_security",
            "warn",
            "dangerous host execution escape hatches are enabled",
            remediation=(
                "Disable unsafe host execution flags and configure governed, "
                "isolated execution backends instead."
            ),
            data={"unsafe_execution_hazards": hazards},
        )
    return _result(
        "execution_security",
        "ok",
        "graph-carried validation and RLM model code cannot use unsafe host fallbacks",
        data={"unsafe_execution_hazards": []},
    )


def _check_permission_governance() -> dict[str, Any]:
    """Validate stable identity authority and policy readiness without disclosure."""

    data = {
        "signing_reference_configured": False,
        "signing_authority_ready": False,
        "custom_policy_configured": False,
        "policy_count": 0,
        "identity_verified": False,
        "redacted": True,
    }
    try:
        from agent_utilities.core.config import AgentConfig
        from agent_utilities.core.profile_guard import is_production_profile

        cfg = AgentConfig()
        data["signing_reference_configured"] = bool(cfg.permissions_signing_key_ref)
        data["custom_policy_configured"] = bool(cfg.agent_policies_path)
        if not cfg.permissions_signing_key_ref:
            production = is_production_profile(cfg.app_profile)
            return _result(
                "permission_governance",
                "fail" if production else "warn",
                "stable agent-identity signing authority is not configured",
                remediation=(
                    "Set PERMISSIONS_SIGNING_KEY_REF to an env://, vault://, or "
                    "secret:// runtime reference containing at least 32 bytes."
                ),
                data=data,
            )

        from agent_utilities.security.cli_secrets import (
            resolve_runtime_secret_reference,
        )
        from agent_utilities.security.permissions_kernel import (
            AgentRole,
            PermissionsKernel,
        )

        signing_key = resolve_runtime_secret_reference(cfg.permissions_signing_key_ref)
        kernel = PermissionsKernel(
            signing_key=signing_key,
            policies_path=cfg.agent_policies_path,
        )
        identity = kernel.issue_identity("doctor-probe", role=AgentRole.GUEST)
        data.update(
            signing_authority_ready=True,
            policy_count=len(kernel.get_policies()),
            identity_verified=kernel.verify_identity(identity),
        )
        if not data["identity_verified"]:
            raise RuntimeError("identity verification failed")
    except Exception as exc:  # noqa: BLE001 - details and paths stay redacted
        return _result(
            "permission_governance",
            "fail",
            f"permission governance is not ready ({type(exc).__name__})",
            remediation=(
                "Resolve the signing-key reference and repair or remove the "
                "configured policy document; configured policies fail closed."
            ),
            data=data,
        )
    return _result(
        "permission_governance",
        "ok",
        "stable signing authority, policy set, and identity verification are ready",
        data=data,
    )


def _check_ontology_release_signing() -> dict[str, Any]:
    """Validate the reference-only ontology release signer without disclosure."""

    data = {
        "signing_reference_configured": False,
        "signing_authority_ready": False,
        "trusted_public_key_count": 0,
        "signer_public_key_trusted": False,
        "redacted": True,
    }
    try:
        from agent_utilities.core.config import AgentConfig
        from agent_utilities.core.profile_guard import is_production_profile

        cfg = AgentConfig()
        reference = cfg.ontology_release_signing_private_key_ref
        data["signing_reference_configured"] = bool(reference)
        if not reference:
            production = is_production_profile(cfg.app_profile)
            return _result(
                "ontology_release_signing",
                "fail" if production else "warn",
                "ontology release signing authority is not configured",
                remediation=(
                    "Set ONTOLOGY_RELEASE_SIGNING_PRIVATE_KEY_REF to an env://, "
                    "vault://, or secret:// reference containing a 32-byte "
                    "base64url Ed25519 seed."
                ),
                data=data,
            )

        from agent_utilities.knowledge_graph.ontology import ontology_integrity

        signer = ontology_integrity.ReleaseSigner.from_runtime()
        trusted = ontology_integrity.release_trusted_public_keys()
        data.update(
            signing_authority_ready=True,
            trusted_public_key_count=len(trusted),
            signer_public_key_trusted=signer.public_key in trusted,
        )
    except Exception as exc:  # noqa: BLE001 - secret-provider details stay redacted
        return _result(
            "ontology_release_signing",
            "fail",
            f"ontology release signing is not ready ({type(exc).__name__})",
            remediation=(
                "Resolve the configured private-key reference and validate any "
                "ONTOLOGY_RELEASE_TRUSTED_PUBLIC_KEYS pins."
            ),
            data=data,
        )
    return _result(
        "ontology_release_signing",
        "ok",
        "stable ontology release signing authority is ready",
        data=data,
    )


def _check_transport_security() -> dict[str, Any]:
    """Validate TLS/auth handoff using only redacted readiness metadata."""
    try:
        import json

        def reject_constant(_value: str) -> None:
            raise ValueError("non-finite JSON constants are not supported")

        def reject_duplicate_keys(
            pairs: list[tuple[str, Any]],
        ) -> dict[str, Any]:
            value: dict[str, Any] = {}
            for key, item in pairs:
                if key in value:
                    raise ValueError("duplicate JSON keys are not supported")
                value[key] = item
            return value

        from agent_utilities.core.config import AgentConfig
        from agent_utilities.core.transport_security import (
            resolve_tls_profile,
            tls_environment_from_config,
        )

        cfg = AgentConfig()
        tls_refs = (
            cfg.tls_profile_ref,
            cfg.tls_profiles_ref,
            cfg.tls_ca_bundle_ref,
            cfg.tls_client_cert_ref,
            cfg.tls_client_key_ref,
            cfg.tls_client_key_password_ref,
            cfg.tls_proxy_url_ref,
            cfg.engine_tls_profile_ref,
        )
        needs_resolver = any(tls_refs) or bool(cfg.external_graph_connectors)
        resolver = None
        secrets_client = None
        if needs_resolver:
            from agent_utilities.security.cli_secrets import (
                resolve_runtime_secret_reference,
            )

            # Environment-backed profiles are already materialized in this
            # process and must not open (or autostart) the durable secret
            # backend. The canonical resolver initializes engine/Vault storage
            # only when the selected reference scheme actually requires it.
            resolver = resolve_runtime_secret_reference
            secrets_client = SimpleNamespace(resolve_ref=resolver)
        runtime_env = tls_environment_from_config(cfg)
        trust = resolve_tls_profile(
            "GLOBAL",
            environ=runtime_env,
            resolver=resolver,
        )
        tls_data = {
            "configured": trust.configured,
            "ready": True,
            "source": trust.source,
            "verify_enabled": trust.verify_enabled,
            "system_trust": trust.system_trust,
            "custom_ca": bool(trust.ca_bundle_path or trust.ca_directory),
            "mtls": bool(trust.client_cert_path),
            "proxy": bool(trust.proxy_url),
        }
        trust.cleanup()

        from agent_utilities.knowledge_graph.core.shard_topology import (
            resolve_endpoints,
        )

        engine_endpoints = resolve_endpoints(cfg)
        engine_tls_configured = any(
            endpoint.startswith("tls://") for endpoint in engine_endpoints
        ) or bool(cfg.engine_tls_profile or cfg.engine_tls_profile_ref)
        engine_data: dict[str, Any] = {
            "configured": engine_tls_configured,
            "ready": True,
            "endpoint_count": len(engine_endpoints),
            "verify_enabled": True,
            "custom_ca": False,
            "mtls": False,
        }
        from agent_utilities.knowledge_graph.core.engine_transport import (
            EngineTransportError,
            engine_client_transport_kwargs,
        )

        for endpoint in engine_endpoints:
            if not endpoint.startswith("tcp://"):
                continue
            try:
                engine_client_transport_kwargs(endpoint, config=cfg)
            except EngineTransportError:
                engine_data["ready"] = False
                break
        if engine_tls_configured:
            engine_trust = resolve_tls_profile(
                "ENGINE",
                profile_name=cfg.engine_tls_profile,
                profile_ref=cfg.engine_tls_profile_ref,
                resolver=resolver,
            )
            engine_data.update(
                verify_enabled=engine_trust.verify_enabled,
                custom_ca=bool(
                    engine_trust.ca_bundle_path or engine_trust.ca_directory
                ),
                mtls=bool(engine_trust.client_cert_path),
            )
            engine_trust.cleanup()

        connectors: list[dict[str, Any]] = []
        unresolved = 0
        property_bundle_ready: bool | None = None
        source_aliases = [
            connector.source_alias for connector in cfg.external_graph_connectors
        ]
        connection_names = [
            connector.name for connector in cfg.external_graph_connectors
        ]
        source_aliases_unique = (
            bool(
                all(source_aliases) and len(set(source_aliases)) == len(source_aliases)
            )
            if source_aliases
            else True
        )
        connection_names_unique = (
            bool(
                all(connection_names)
                and len(set(connection_names)) == len(connection_names)
            )
            if connection_names
            else True
        )
        for connector in cfg.external_graph_connectors:
            property_graph = connector.backend != "graphql"
            if property_graph and property_bundle_ready is None:
                try:
                    from agent_utilities.knowledge_graph.ontology.connector_manifest_gate import (
                        precheck_source,
                    )

                    bundle = precheck_source("external_graph")
                    property_bundle_ready = bool(
                        bundle.get("checked") and bundle.get("ok")
                    )
                except Exception:
                    property_bundle_ready = False
            sync_policy = (
                {
                    "allow_empty_snapshot": bool(
                        getattr(connector, "allow_empty_snapshot", False)
                    ),
                    "max_pages": int(getattr(connector, "ingest_max_pages", 100)),
                    "max_row_bytes": int(
                        getattr(connector, "ingest_max_row_bytes", 1_048_576)
                    ),
                    "max_total_bytes": int(
                        getattr(connector, "ingest_max_total_bytes", 16_777_216)
                    ),
                    "max_nesting_depth": int(
                        getattr(connector, "ingest_max_nesting_depth", 16)
                    ),
                    "max_collection_items": int(
                        getattr(connector, "ingest_max_collection_items", 10_000)
                    ),
                    "page_size": int(getattr(connector, "ingest_page_size", 500)),
                    "reconcile_deletions": bool(
                        getattr(connector, "reconcile_deletions", True)
                    ),
                    "sync_mode": str(getattr(connector, "sync_mode", "auto")),
                }
                if property_graph
                else None
            )
            refs = {
                "connection": connector.connection_profile_ref,
                "mapping": connector.mapping_policy_ref,
                "tls": connector.tls_profile_ref,
                "auth": connector.auth_profile_ref,
                "variables": getattr(connector, "variables_ref", None),
            }
            readiness: dict[str, bool | None] = {}
            resolved_mapping_policy: dict[str, Any] | None = None
            for label, ref in refs.items():
                if ref is None:
                    readiness[label] = None
                    continue
                try:
                    if label == "tls":
                        connector_trust = resolve_tls_profile(
                            "EXTERNAL_GRAPH",
                            profile_ref=str(ref),
                            resolver=resolver,
                        )
                        try:
                            ready = connector_trust.verify_enabled
                        finally:
                            connector_trust.cleanup()
                        resolved = None
                    else:
                        resolved = resolver(ref) if resolver is not None else None
                        ready = bool(resolved)
                    if (
                        label in {"auth", "connection", "mapping", "variables"}
                        and ready
                    ):
                        if len(str(resolved).encode("utf-8")) > 4 * 1024 * 1024:
                            raise ValueError("external profile exceeds its bound")
                        parsed = json.loads(
                            str(resolved),
                            parse_constant=reject_constant,
                            object_pairs_hook=reject_duplicate_keys,
                        )
                        ready = isinstance(parsed, dict)
                        if label == "mapping" and ready:
                            resolved_mapping_policy = parsed
                        if ready and connector.backend == "graphql":
                            from agent_utilities.knowledge_graph.ingestion.graphql_connection import (
                                GRAPHQL_AUTH_PROFILE_FORMAT,
                                GRAPHQL_CONNECTION_PROFILE_FORMAT,
                                GRAPHQL_MAPPING_POLICY_FORMAT,
                            )

                            if label == "connection":
                                ready = parsed.get(
                                    "profile_format"
                                ) == GRAPHQL_CONNECTION_PROFILE_FORMAT and isinstance(
                                    parsed.get("endpoint"), str
                                )
                            elif label == "mapping":
                                discovery = parsed.get("discovery") or {}
                                ready = (
                                    parsed.get("profile_format")
                                    == GRAPHQL_MAPPING_POLICY_FORMAT
                                    and isinstance(parsed.get("operations", {}), dict)
                                    and bool(
                                        parsed.get("operations")
                                        or (
                                            isinstance(discovery, dict)
                                            and discovery.get("enabled") is True
                                        )
                                    )
                                )
                            elif label == "auth":
                                ready = parsed.get(
                                    "profile_format"
                                ) == GRAPHQL_AUTH_PROFILE_FORMAT and isinstance(
                                    parsed.get("headers", {}), dict
                                )
                except Exception:
                    ready = False
                readiness[label] = ready
                unresolved += int(not ready)
            if connector.backend == "graphql":
                generated_bootstrap = (
                    connector.mapping_policy_ref is None
                    and connector.allow_introspection
                )
                readiness["generated_mapping"] = (
                    generated_bootstrap
                    if connector.mapping_policy_ref is None
                    else None
                )
                unresolved += int(
                    connector.mapping_policy_ref is None
                    and not connector.allow_introspection
                )
            lifecycle = "not_found"
            mapping_policy_drift: str | None = None
            if secrets_client is not None:
                try:
                    from agent_utilities.knowledge_graph.ingestion.external_graph_schema import (
                        external_mapping_policy_digest,
                        mapping_profile_status,
                    )

                    if connector.backend == "graphql":
                        from agent_utilities.knowledge_graph.ingestion.graphql_connection import (
                            GraphQLSourceAdapter,
                            graphql_mapping_profile_status,
                        )

                        source = GraphQLSourceAdapter(
                            connection=connector.name,
                            source_alias=connector.source_alias,
                            connection_profile_ref=connector.connection_profile_ref,
                            mapping_policy_ref=str(connector.mapping_policy_ref or ""),
                            auth_profile_ref=connector.auth_profile_ref,
                            tls_profile_ref=connector.tls_profile_ref,
                            variables_ref=getattr(connector, "variables_ref", None),
                            allow_introspection=connector.allow_introspection,
                            allow_empty_snapshot=bool(
                                getattr(connector, "allow_empty_snapshot", False)
                            ),
                            resolver=resolver,
                        )
                        try:
                            source.validate_runtime_profiles()
                        except Exception:
                            readiness["runtime_contract"] = False
                            unresolved += 1
                            raise
                        readiness["runtime_contract"] = True
                        mapping_status = graphql_mapping_profile_status(
                            source,
                            connection=connector.name,
                            secret_store=secrets_client,
                        )
                        mapping_policy_drift = str(
                            mapping_status.get("mapping_drift") or "unknown"
                        )
                    else:
                        if connector.mapping_policy_ref is None:
                            current_policy = {}
                        elif resolved_mapping_policy is not None:
                            current_policy = resolved_mapping_policy
                        else:
                            current_policy = None
                        current_policy_digest = (
                            external_mapping_policy_digest(
                                {**current_policy, "sync": sync_policy}
                            )
                            if current_policy is not None and sync_policy is not None
                            else None
                        )
                        mapping_status = mapping_profile_status(
                            connector.name,
                            secret_store=secrets_client,
                            runtime_policy_digest=current_policy_digest,
                        )
                        mapping_policy_drift = (
                            str(mapping_status.get("mapping_drift") or "unknown")
                            if current_policy_digest is not None
                            else "unknown"
                        )
                    lifecycle = str(mapping_status.get("status") or "not_found")
                except Exception:
                    lifecycle = "unavailable"
                    mapping_policy_drift = "unknown"
            connectors.append(
                {
                    "backend": connector.backend,
                    "refs_ready": readiness,
                    "mapping_lifecycle": lifecycle,
                    "mapping_policy_drift": mapping_policy_drift,
                    "capability_bundle_ready": (
                        property_bundle_ready if property_graph else None
                    ),
                    "sync_policy": sync_policy,
                    "semantic_mapping": connector.semantic_mapping,
                    "generated_mapping": bool(
                        connector.backend == "graphql"
                        and connector.mapping_policy_ref is None
                        and connector.allow_introspection
                    ),
                    "authoritative_empty_approval": bool(
                        connector.backend == "graphql"
                        and getattr(connector, "allow_empty_snapshot", False)
                    ),
                    "approval_required": connector.require_approval,
                    "drift_policy": connector.schema_drift_policy,
                }
            )
    except Exception as exc:  # noqa: BLE001 - doctor must remain defensive
        return _result(
            "transport_security",
            "fail",
            f"transport profile validation failed ({type(exc).__name__})",
            remediation=(
                "Configure a named TLS profile with secret refs; do not place "
                "endpoints, credentials, certificate material, or local paths in config."
            ),
            data={"ready": False},
        )

    lifecycle_unready = sum(
        1
        for connector in connectors
        if connector.get("mapping_lifecycle") != "approved"
        or connector.get("mapping_policy_drift") == "detected"
    )
    bundle_unready = sum(
        1
        for connector in connectors
        if connector.get("capability_bundle_ready") is False
    )
    verification_disabled = not trust.verify_enabled or not bool(
        engine_data["verify_enabled"]
    )
    status = (
        "fail"
        if (
            unresolved
            or bundle_unready
            or not source_aliases_unique
            or not connection_names_unique
            or not engine_data["ready"]
        )
        else ("warn" if verification_disabled or lifecycle_unready else "ok")
    )
    if not engine_data["ready"]:
        detail = "native engine transport policy is not ready"
    elif unresolved:
        detail = f"{unresolved} configured external profile reference(s) are unresolved"
    elif bundle_unready:
        detail = f"{bundle_unready} property-graph capability bundle(s) are unready"
    elif not source_aliases_unique:
        detail = "external graph source aliases are not unique"
    elif not connection_names_unique:
        detail = "external graph connection names are not unique"
    elif verification_disabled:
        detail = "one or more runtime transports have TLS verification disabled"
    elif lifecycle_unready:
        detail = f"{lifecycle_unready} external mapping lifecycle(s) require approval"
    else:
        detail = "runtime trust profile ready"
    return _result(
        "transport_security",
        status,
        detail,
        remediation=(
            None
            if status == "ok"
            else (
                "Repair unresolved secret refs or signed capability bundles, enable "
                "verified TLS, or complete the discover/propose/approve lifecycle."
            )
        ),
        data={
            "tls": tls_data,
            "native_engine_tls": engine_data,
            "external_graph_connectors": connectors,
            "external_graph_source_aliases_unique": source_aliases_unique,
            "external_graph_connection_names_unique": connection_names_unique,
        },
    )


def _check_google_workspace_oauth() -> dict[str, Any]:
    """Validate optional OAuth bootstrap without disclosing tenant configuration."""

    try:
        from agent_utilities.core.config import AgentConfig

        cfg = AgentConfig()
    except Exception as exc:  # noqa: BLE001
        return _result(
            "google_workspace_oauth",
            "error",
            f"OAuth configuration unavailable ({type(exc).__name__})",
        )
    client_ready = bool(cfg.google_workspace_oauth_client_id)
    broker_ready = bool(cfg.google_workspace_oauth_broker_url)
    data = {"client_id_configured": client_ready, "broker_configured": broker_ready}
    if not client_ready and not broker_ready:
        return _result(
            "google_workspace_oauth",
            "skip",
            "Google Workspace OAuth is not configured",
            data=data,
        )
    if not (client_ready and broker_ready):
        return _result(
            "google_workspace_oauth",
            "fail",
            "Google Workspace OAuth configuration is incomplete",
            remediation=(
                "Set both GOOGLE_WORKSPACE_OAUTH_CLIENT_ID and the HTTPS-only "
                "GOOGLE_WORKSPACE_OAUTH_BROKER_URL through AgentConfig/XDG runtime config."
            ),
            data=data,
        )
    return _result(
        "google_workspace_oauth",
        "ok",
        "Google Workspace OAuth runtime connection points are configured",
        data=data,
    )


def _check_source_egress() -> dict[str, Any]:
    """Report the shared SSRF/redirect/body boundary without exposing hosts."""
    try:
        from agent_utilities.core.config import AgentConfig
        from agent_utilities.core.transport_security import (
            resolve_tls_profile,
            tls_environment_from_config,
        )
        from agent_utilities.protocols.source_connectors.http_safety import (
            normalize_allowed_hosts,
        )

        cfg = AgentConfig()
        private_hosts = normalize_allowed_hosts(cfg.source_http_allowed_private_hosts)
        redirect_hosts = normalize_allowed_hosts(cfg.source_http_allowed_redirect_hosts)
        model_private_hosts = normalize_allowed_hosts(
            cfg.model_http_allowed_private_hosts
        )
        tls_environment = tls_environment_from_config(cfg)
        model_tls = resolve_tls_profile(
            "model",
            profile_name=cfg.model_tls_profile,
            profile_ref=cfg.model_tls_profile_ref,
            environ=tls_environment,
        )
        embedding_tls = resolve_tls_profile(
            "embedding",
            profile_name=cfg.embedding_tls_profile,
            profile_ref=cfg.embedding_tls_profile_ref,
            environ=tls_environment,
        )
        oauth2_token_tls = resolve_tls_profile(
            "oauth2-token",
            profile_name=cfg.oauth2_token_tls_profile,
            profile_ref=cfg.oauth2_token_tls_profile_ref,
            environ=tls_environment,
        )
        model_proxy_configured = bool(
            model_tls.proxy_url or embedding_tls.proxy_url or oauth2_token_tls.proxy_url
        )
        oauth2_model_count = sum(
            bool(getattr(model, "oauth2", None))
            for model in (*cfg.chat_models, *cfg.embedding_models)
        )
        model_tls_data = {
            "model_verify_enabled": model_tls.verify_enabled,
            "model_custom_ca": bool(model_tls.ca_bundle_path or model_tls.ca_directory),
            "model_mtls": bool(model_tls.client_cert_path),
            "embedding_verify_enabled": embedding_tls.verify_enabled,
            "embedding_custom_ca": bool(
                embedding_tls.ca_bundle_path or embedding_tls.ca_directory
            ),
            "embedding_mtls": bool(embedding_tls.client_cert_path),
            "oauth2_token_verify_enabled": oauth2_token_tls.verify_enabled,
            "oauth2_token_custom_ca": bool(
                oauth2_token_tls.ca_bundle_path or oauth2_token_tls.ca_directory
            ),
            "oauth2_token_mtls": bool(oauth2_token_tls.client_cert_path),
            "oauth2_model_count": oauth2_model_count,
            "model_proxy_configured": model_proxy_configured,
        }
        model_tls.cleanup()
        embedding_tls.cleanup()
        oauth2_token_tls.cleanup()
    except Exception as exc:  # noqa: BLE001 - doctor must remain defensive
        return _result(
            "source_egress",
            "fail",
            f"source egress policy is invalid ({type(exc).__name__})",
            remediation=(
                "Use exact hostnames (no URLs or wildcards), bounded response/redirect "
                "limits, and keep browser fetching disabled unless explicitly required."
            ),
            data={"ready": False},
        )

    browser_enabled = bool(cfg.source_http_allow_browser_fetch)
    if model_proxy_configured:
        return _result(
            "source_egress",
            "fail",
            "model transport profile is incompatible with DNS-pinned egress",
            remediation=(
                "Remove the model/embedder/OAuth2 token proxy and use direct TLS with "
                "a runtime CA or mTLS profile so destination DNS and peer identity "
                "can be pinned."
            ),
            data={"ready": False, **model_tls_data},
        )
    return _result(
        "source_egress",
        "warn" if browser_enabled else "ok",
        (
            "bounded source egress is active; browser-backed fetching is explicitly enabled"
            if browser_enabled
            else "bounded source egress is active; private hosts and cross-host redirects are denied by default"
        ),
        remediation=(
            "Disable SOURCE_HTTP_ALLOW_BROWSER_FETCH when rendered-page acquisition is not required."
            if browser_enabled
            else None
        ),
        data={
            "ready": True,
            "private_host_allowlist_count": len(private_hosts),
            "redirect_host_allowlist_count": len(redirect_hosts),
            "model_private_host_allowlist_count": len(model_private_hosts),
            "max_response_bytes": cfg.source_http_max_response_bytes,
            "max_redirects": cfg.source_http_max_redirects,
            "browser_fetch_enabled": browser_enabled,
            **model_tls_data,
        },
    )


def _check_eunomia() -> dict[str, Any]:
    """Validate the native policy-decision-point configuration without I/O."""
    try:
        from pathlib import Path
        from urllib.parse import urlsplit

        from agent_utilities.core.config import AgentConfig
        from agent_utilities.core.transport_security import (
            resolve_tls_profile,
            tls_environment_from_config,
        )
        from agent_utilities.protocols.source_connectors.http_safety import (
            normalize_allowed_hosts,
            require_safe_source_url,
        )

        cfg = AgentConfig()
        mode = cfg.eunomia_type
        private_hosts = normalize_allowed_hosts(cfg.eunomia_allowed_private_hosts)
        if mode == "none":
            return _result(
                "eunomia",
                "ok",
                "native MCP policy authorization is explicitly disabled",
                data={"mode": "none", "ready": True},
            )
        if mode == "embedded":
            policy = str(cfg.eunomia_policy_file or "mcp_policies.json")
            ready = Path(policy).expanduser().is_file()
            return _result(
                "eunomia",
                "ok" if ready else "fail",
                (
                    "embedded native MCP policy is configured"
                    if ready
                    else "embedded MCP policy file is unavailable"
                ),
                remediation=(
                    None
                    if ready
                    else "Set EUNOMIA_POLICY_FILE to a runtime-mounted policy document."
                ),
                data={"mode": "embedded", "ready": ready},
            )

        endpoint = str(cfg.eunomia_remote_url or "")
        if not endpoint:
            raise ValueError("remote endpoint is missing")
        host = require_safe_source_url(
            endpoint,
            allowed_private_hosts=private_hosts,
            resolve_dns=False,
        )
        parsed = urlsplit(endpoint)
        insecure_transport = parsed.scheme == "http" and host not in {
            "localhost",
            "127.0.0.1",
            "::1",
        }
        if insecure_transport:
            raise ValueError("remote endpoint requires HTTPS")
        trust = resolve_tls_profile(
            "eunomia",
            profile_name=cfg.eunomia_tls_profile,
            profile_ref=cfg.eunomia_tls_profile_ref,
            environ=tls_environment_from_config(cfg),
        )
        tls_data = {
            "verify_enabled": trust.verify_enabled,
            "custom_ca": bool(trust.ca_bundle_path or trust.ca_directory),
            "mtls": bool(trust.client_cert_path),
            "proxy_configured": bool(trust.proxy_url),
        }
        trust.cleanup()
        if tls_data["proxy_configured"]:
            raise ValueError("remote policy proxy is incompatible with DNS pinning")
        status = "ok"
        return _result(
            "eunomia",
            status,
            ("remote native MCP policy authorization is bounded and TLS-verified"),
            remediation=None,
            data={
                "mode": "remote",
                "ready": True,
                "private_host_allowlist_count": len(private_hosts),
                "api_key_ref_configured": bool(cfg.eunomia_api_key_ref),
                "timeout_seconds": cfg.eunomia_timeout_seconds,
                "max_response_bytes": cfg.eunomia_max_response_bytes,
                "bulk_check_max": cfg.eunomia_bulk_check_max,
                **tls_data,
            },
        )
    except Exception as exc:  # noqa: BLE001 - doctor is a defensive boundary
        return _result(
            "eunomia",
            "fail",
            f"native MCP policy configuration is invalid ({type(exc).__name__})",
            remediation=(
                "Configure EUNOMIA_TYPE plus either a runtime policy file or a "
                "bounded HTTPS endpoint, exact private-host allowlist, secret ref, "
                "and EUNOMIA TLS profile in AgentConfig/XDG."
            ),
            data={"ready": False, "redacted": True},
        )


def _check_runtime_integrations() -> dict[str, Any]:
    """Validate optional fleet, inventory, and media configuration offline.

    Endpoint identities and inventory paths are runtime-only material. This
    check reports aggregate configuration readiness and local file availability
    without returning any configured value or making a network request.
    """
    try:
        from pathlib import Path

        from agent_utilities.core.config import AgentConfig

        cfg = AgentConfig()
        inventory_configured = bool(cfg.infra_inventory_path)
        inventory_file_ready = False
        inventory_format_ready = False
        if inventory_configured:
            try:
                inventory_path = Path(str(cfg.infra_inventory_path)).expanduser()
                inventory_file_ready = inventory_path.is_file()
            except (OSError, ValueError):
                inventory_file_ready = False
            if inventory_file_ready:
                try:
                    import yaml

                    with inventory_path.open("rb") as stream:
                        raw_inventory = stream.read(8 * 1024 * 1024 + 1)
                    if len(raw_inventory) <= 8 * 1024 * 1024:
                        inventory = yaml.safe_load(raw_inventory.decode("utf-8"))
                        inventory_format_ready = isinstance(inventory, dict)
                except Exception:  # noqa: BLE001 - readiness is redacted
                    inventory_format_ready = False

        fleet_template_configured = bool(cfg.fleet_mcp_url_template)
        media_endpoint_count = sum(
            bool(value)
            for value in (
                cfg.comfyui_url,
                cfg.xtts_url,
                cfg.openai_tts_url,
                cfg.whisper_url,
                cfg.faster_whisper_url,
                cfg.flux_url,
                cfg.sd35_url,
                cfg.hunyuan_url,
                cfg.svd_url,
            )
        )
    except Exception as exc:  # noqa: BLE001 - doctor must remain defensive
        return _result(
            "runtime_integrations",
            "fail",
            f"runtime integration configuration is invalid ({type(exc).__name__})",
            remediation=(
                "Use bounded http(s) base URLs without inline credentials, a fleet "
                "template containing {server}, and a valid runtime inventory path."
            ),
            data={"ready": False, "redacted": True},
        )

    configured_category_count = sum(
        (
            inventory_configured,
            fleet_template_configured,
            media_endpoint_count > 0,
        )
    )
    ready_category_count = sum(
        (
            inventory_configured and inventory_file_ready and inventory_format_ready,
            fleet_template_configured,
            media_endpoint_count > 0,
        )
    )
    data = {
        "configured_category_count": configured_category_count,
        "ready_category_count": ready_category_count,
        "inventory_configured": inventory_configured,
        "inventory_file_ready": inventory_file_ready,
        "inventory_format_ready": inventory_format_ready,
        "fleet_template_configured": fleet_template_configured,
        "media_endpoint_configured_count": media_endpoint_count,
        "media_endpoint_slot_count": 9,
        "network_probed": False,
        "redacted": True,
    }
    if not configured_category_count:
        return _result(
            "runtime_integrations",
            "skip",
            "optional inventory, fleet-template, and media endpoints are not configured",
            data=data,
        )
    if inventory_configured and not (inventory_file_ready and inventory_format_ready):
        return _result(
            "runtime_integrations",
            "warn",
            "runtime integration configuration is valid but the inventory file is "
            "unavailable or malformed",
            remediation=(
                "Point INFRA_INVENTORY_PATH at a readable, bounded YAML mapping, or "
                "unset it when infrastructure inventory ingestion is not used."
            ),
            data=data,
        )
    return _result(
        "runtime_integrations",
        "ok",
        f"{ready_category_count}/{configured_category_count} optional integration "
        "category(s) are configuration-ready",
        data=data,
    )


def _check_engine() -> dict[str, Any]:
    try:
        from agent_utilities.core.config import AgentConfig
        from agent_utilities.knowledge_graph.core.engine_resolver import resolve_engine
        from agent_utilities.knowledge_graph.core.graph_compute import (
            engine_encryption_readiness,
        )
        from agent_utilities.knowledge_graph.core.shard_topology import (
            default_graph_name,
            shard_topology_status,
        )

        cfg = AgentConfig()
        st = shard_topology_status(cfg, probe=True, timeout=0.5)
        resolved = resolve_engine(cfg, default_graph_name(cfg))
        encryption = engine_encryption_readiness(cfg, remote=resolved.mode == "remote")
    except Exception as exc:  # noqa: BLE001
        return _result(
            "engine",
            "error",
            f"shard topology probe failed ({type(exc).__name__})",
        )
    st["resolved_mode"] = resolved.mode
    endpoints = st.get("endpoints", [])
    reachable = [e for e in endpoints if e.get("reachable")]
    # Endpoint strings can contain hostnames, usernames, local socket paths, or
    # customer-specific topology names. Doctor is frequently copied into issue
    # reports and traces, so expose readiness counts only.
    redacted_status = {
        "resolved_mode": resolved.mode,
        "topology_mode": st.get("mode", "unknown"),
        "configured_endpoint_count": len(endpoints),
        "reachable_endpoint_count": len(reachable),
        "placement_group_mapping_count": len(
            getattr(cfg, "graph_raft_group_endpoints", {}) or {}
        ),
        "autostart_allowed": bool(resolved.autostart_allowed),
        "idle_shutdown_configured": bool(resolved.idle_shutdown_secs > 0),
        "durable_encryption": encryption,
        "runtime_directory_ref_count": sum(
            bool(value)
            for value in (
                getattr(cfg, "epistemic_graph_sqlite_transfer_root_ref", None),
                getattr(cfg, "epistemic_graph_backup_root_ref", None),
            )
        ),
        "resource_limits": {
            "request_bytes": getattr(cfg, "epistemic_graph_max_request_bytes", 0),
            "response_bytes": getattr(cfg, "epistemic_graph_max_response_bytes", 0),
            "msgpack_items": getattr(cfg, "epistemic_graph_max_msgpack_items", 0),
            "ast_files": getattr(cfg, "epistemic_graph_ast_max_files", 0),
            "ast_source_bytes": getattr(cfg, "epistemic_graph_ast_max_source_bytes", 0),
            "ast_total_bytes": getattr(cfg, "epistemic_graph_ast_max_total_bytes", 0),
            "modality_bundle_bytes": getattr(
                cfg, "epistemic_graph_modality_max_bundle_bytes", 0
            ),
            "modality_source_bytes": getattr(
                cfg, "epistemic_graph_modality_max_source_bytes", 0
            ),
            "sqlite_bytes": getattr(cfg, "epistemic_graph_sqlite_max_bytes", 0),
            "sqlite_rows": getattr(cfg, "epistemic_graph_sqlite_max_rows", 0),
        },
        "redacted": True,
    }

    if not encryption["ready"]:
        return _result(
            "engine",
            "fail",
            "local durable-engine encryption is not configuration-ready",
            remediation=(
                "Configure EPISTEMIC_GRAPH_ENCRYPTION_KEY_REF with an external "
                "runtime secret reference that resolves to bounded key material."
            ),
            data=redacted_status,
        )

    if len(endpoints) > 1 and not getattr(cfg, "graph_raft_group_endpoints", {}):
        return _result(
            "engine",
            "fail",
            "multiple coordinator contacts have no Raft group endpoint mapping",
            remediation=(
                "Configure GRAPH_RAFT_GROUP_ENDPOINTS as a group-to-endpoint JSON map, "
                "or expose one stable coordinator contact. Clients never infer placement."
            ),
            data=redacted_status,
        )

    runtime_directory_refs = tuple(
        reference
        for reference in (
            getattr(cfg, "epistemic_graph_sqlite_transfer_root_ref", None),
            getattr(cfg, "epistemic_graph_backup_root_ref", None),
        )
        if reference
    )
    if resolved.mode != "remote" and runtime_directory_refs:
        try:
            import os
            import stat
            from pathlib import Path

            from agent_utilities.security.secrets_client import create_secrets_client

            resolver = create_secrets_client()
            for reference in runtime_directory_refs:
                raw = resolver.resolve_ref(reference)
                rendered = (
                    raw.decode("utf-8") if isinstance(raw, bytes) else str(raw or "")
                )
                if (
                    not rendered
                    or len(rendered.encode("utf-8")) > 4_096
                    or any(ord(character) < 32 for character in rendered)
                ):
                    raise ValueError("invalid runtime directory")
                candidate = Path(rendered)
                metadata = candidate.lstat()
                if candidate.is_symlink() or not candidate.is_dir():
                    raise ValueError("unsafe runtime directory")
                candidate.resolve(strict=True)
                if os.name == "posix" and stat.S_IMODE(metadata.st_mode) & 0o077:
                    raise ValueError("runtime directory is not private")
            redacted_status["runtime_directory_refs_ready"] = True
        except Exception:  # noqa: BLE001 - diagnostics must not reveal paths/providers
            redacted_status["runtime_directory_refs_ready"] = False
            return _result(
                "engine",
                "fail",
                "an enabled engine file capability has an unavailable or unsafe runtime directory",
                remediation=(
                    "Resolve each configured engine directory reference to an existing, "
                    "non-symlink private directory; do not place host paths in AgentConfig."
                ),
                data=redacted_status,
            )

    # CONCEPT:AU-OS.deployment.report-resolved-mode — report the RESOLVED mode (how this process reaches the
    # engine), not just transport reachability.
    if resolved.mode == "remote":
        if reachable:
            if runtime_directory_refs:
                return _result(
                    "engine",
                    "warn",
                    "remote engine reachable, but local runtime directory references are not applied remotely",
                    remediation=(
                        "Configure backup/SQLite roots in the remote engine deployment, "
                        "or remove the local-only references."
                    ),
                    data=redacted_status,
                )
            return _result(
                "engine",
                "ok",
                f"remote engine reachable ({len(reachable)}/{len(endpoints)} "
                "endpoint(s)) — resolved mode=remote (deployed elsewhere)",
                data=redacted_status,
            )
        return _result(
            "engine",
            "fail",
            "configured remote engine is unreachable — "
            "remote mode never autostarts a local stand-in (fail-loud)",
            remediation="start the external engine (Docker/host) or fix GRAPH_SERVICE_ENDPOINTS",
            skill="agent-utilities-deployment",
            data=redacted_status,
        )

    if reachable:
        return _result(
            "engine",
            "ok",
            "engine reachable — resolved mode=shared "
            "(reusing the already-running local engine)",
            data=redacted_status,
        )

    # Nothing up locally — describe the autostart behaviour the resolver WILL
    # take on first use, including the idle-shutdown lifecycle.
    if resolved.autostart_allowed:
        if resolved.idle_shutdown_secs > 0:
            life = (
                f"reference-counted (auto-stops {resolved.idle_shutdown_secs}s "
                "after the last client disconnects)"
            )
        else:
            life = "persistent (never auto-stops — runs like a local service)"
        return _result(
            "engine",
            "warn",
            "no engine running yet — resolved mode=autostart: "
            f"a detached, supervised engine will be spawned on first use, {life}",
            remediation="no action needed (auto-provisions on demand); start eagerly with `graph-os-daemon` if preferred",
            skill="agent-utilities-deployment",
            data=redacted_status,
        )
    return _result(
        "engine",
        "fail",
        f"no epistemic-graph engine endpoint reachable ({len(endpoints)} configured) and autostart disabled",
        remediation="remove GRAPH_SERVICE_ENDPOINTS for the packaged local lifecycle, or start the configured external engine",
        skill="agent-utilities-deployment",
        data=redacted_status,
    )


def _check_engine_request_context() -> dict[str, Any]:
    """Report the fail-secure native-engine request-context posture."""

    data = {
        "verified_context_required": True,
        "legacy_protocol_available": False,
        "unauthenticated_transport_available": False,
    }
    return _result(
        "engine_request_context",
        "ok",
        "engine request context is current-only and requires verified identity",
        data=data,
    )


def _check_graph_authority() -> dict[str, Any]:
    """Verify that the live read/write authority is EpistemicGraphBackend.

    The doctor only inspects an already-active backend. It never constructs a
    connector, opens a local store, or exposes connection material.
    """
    try:
        from agent_utilities.knowledge_graph.backends import get_active_backend
        from agent_utilities.knowledge_graph.backends.brain_guarded_backend import (
            BrainGuardedBackend,
        )
        from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
            EpistemicGraphBackend,
        )
        from agent_utilities.knowledge_graph.backends.fanout_backend import (
            FanOutBackend,
        )

        backend = get_active_backend()
        if backend is None:
            return _result(
                "graph_authority",
                "skip",
                "no graph authority active in this process (start GraphOS to evaluate)",
            )
        inner = backend.inner if isinstance(backend, BrainGuardedBackend) else backend
        authority = inner.authority if isinstance(inner, FanOutBackend) else inner
        if not isinstance(authority, EpistemicGraphBackend):
            return _result(
                "graph_authority",
                "fail",
                "active graph authority is not the required epistemic-graph engine",
                remediation=(
                    "restart GraphOS with current AgentConfig; declare external "
                    "databases only as source connectors or projection mirrors"
                ),
                skill="database-environment-setup",
                data={"authority_current": False},
            )
        hc = getattr(authority, "health_check", None)
        ok = hc() if callable(hc) else True
    except Exception as exc:  # noqa: BLE001
        return _result(
            "graph_authority",
            "warn",
            f"authority not evaluable ({type(exc).__name__})",
            remediation="restart GraphOS and validate the configured engine lifecycle",
            skill="database-environment-setup",
            data={"authority_current": False},
        )
    if ok:
        projection_count = len(getattr(inner, "_mirrors", {}))
        return _result(
            "graph_authority",
            "ok",
            "epistemic-graph authority reachable",
            data={
                "authority_current": True,
                "projection_count": projection_count,
            },
        )
    return _result(
        "graph_authority",
        "fail",
        "epistemic-graph authority health check failed",
        remediation="verify the managed epistemic-graph engine lifecycle",
        skill="database-environment-setup",
        data={"authority_current": True},
    )


def _check_secrets() -> dict[str, Any]:
    source_status: dict[str, Any] = {
        "state": "not_loaded",
        "present": False,
        "valid": True,
        "referenced_count": 0,
        "matched_count": 0,
        "projected_count": 0,
        "overridden_count": 0,
    }
    try:
        from agent_utilities.core.config import (
            AgentConfig,
            runtime_secret_source_status,
        )
        from agent_utilities.deployment.config_generator import _unresolved_secret_refs

        unresolved = _unresolved_secret_refs(AgentConfig())
        source_status = runtime_secret_source_status()
    except Exception as exc:  # noqa: BLE001
        return _result(
            "secrets",
            "fail",
            f"secrets backend not evaluated ({type(exc).__name__})",
            remediation=(
                "repair the private runtime source or configured secret backend"
            ),
            skill="agent-utilities-deployment",
            data={
                "runtime_source": source_status,
                "redacted": True,
            },
        )
    data = {
        "runtime_source": source_status,
        "unresolved_count": len(unresolved),
        "redacted": True,
    }
    if not unresolved:
        return _result(
            "secrets",
            "ok",
            "no unresolved secret references",
            data=data,
        )
    return _result(
        "secrets",
        "fail",
        f"{len(unresolved)} secret reference(s) are unresolved",
        remediation="seed the values in your secrets backend",
        skill="secret-vault-manager",
        data=data,
    )


def _check_auth() -> dict[str, Any]:
    from agent_utilities.core.config import config, setting
    from agent_utilities.security.request_identity import (
        local_process_authority_enabled,
    )

    if local_process_authority_enabled(config):
        return _result(
            "auth",
            "ok",
            "private ephemeral graph authority is ready for tiny stdio",
            data={
                "mode": "ephemeral_local",
                "network_transport_ready": False,
                "redacted": True,
            },
        )

    jwks = str(setting("AUTH_JWT_JWKS_URI", "") or "").strip()
    audience = str(setting("AUTH_JWT_AUDIENCE", "") or "").strip()
    policy_version = str(setting("KG_POLICY_VERSION", "") or "").strip()
    missing = [
        name
        for name, value in (
            ("AUTH_JWT_JWKS_URI", jwks),
            ("AUTH_JWT_AUDIENCE", audience),
            ("KG_POLICY_VERSION", policy_version),
        )
        if not value
    ]
    if not missing:
        # IdP-agnostic: any OIDC issuer's JWKS works. Name it for the report so
        # an operator on Okta isn't told they need Keycloak (CONCEPT:AU-OS.deployment.vault-first-routine-genesis genesis
        # IdP choice — keycloak deploy-if-absent OR an existing okta/other-oidc org).
        low = jwks.lower()
        idp = "Okta" if "okta" in low else ("Keycloak" if "keycloak" in low else "OIDC")
        return _result(
            "auth",
            "ok",
            f"verified graph authority configured ({idp}; audience + policy pinned)",
        )
    return _result(
        "auth",
        "fail",
        f"verified graph authority is incomplete ({len(missing)} setting(s) absent)",
        remediation=(
            "configure AUTH_JWT_JWKS_URI, AUTH_JWT_AUDIENCE, and KG_POLICY_VERSION; "
            "served graph operations fail closed until all three are present"
        ),
        skill="keycloak-client-onboarder",
        data={"missing_count": len(missing), "fail_closed": True},
    )


def _check_outbound_auth() -> dict[str, Any]:
    """Validate outbound MCP auth metadata without resolving credential material."""
    try:
        from agent_utilities.mcp.client_credentials import (
            outbound_auth_configuration_status,
        )

        status = outbound_auth_configuration_status()
    except Exception as exc:  # noqa: BLE001 - never report configured values
        return _result(
            "outbound_auth",
            "fail",
            f"outbound MCP authentication configuration is invalid ({type(exc).__name__})",
            remediation=(
                "Set MCP_CLIENT_AUTH and its canonical AgentConfig fields; keep "
                "credential material behind a runtime secret reference."
            ),
            data={"ready": False, "redacted": True},
        )
    mode = str(status["mode"])
    if mode == "none":
        return _result(
            "outbound_auth",
            "skip",
            "outbound MCP child authentication is disabled",
            data={"mode": mode, "ready": True, "redacted": True},
        )
    if bool(status["ready"]):
        return _result(
            "outbound_auth",
            "ok",
            f"outbound MCP child authentication is configured ({mode})",
            data={"mode": mode, "ready": True, "redacted": True},
        )
    missing = status.get("missing") or ()
    invalid = status.get("invalid") or ()
    return _result(
        "outbound_auth",
        "fail",
        "outbound MCP child authentication is incomplete",
        remediation=(
            "Configure OIDC_CLIENT_ID, OIDC_CLIENT_SECRET_REF, OIDC_AUDIENCE, "
            "and either OIDC_TOKEN_URL or OIDC_ISSUER in XDG AgentConfig."
        ),
        data={
            "mode": mode,
            "ready": False,
            "missing_count": len(missing),
            "invalid_count": len(invalid),
            "redacted": True,
        },
    )


def _check_skill_certification() -> dict[str, Any]:
    """Validate exact skill-certification inputs without exposing their values."""

    required_count = 8
    try:
        from pathlib import Path

        from agent_utilities.core.config import AgentConfig

        cfg = AgentConfig()
    except Exception as exc:  # noqa: BLE001 - report only the error category
        return _result(
            "skill_certification",
            "fail",
            f"skill certification configuration is invalid ({type(exc).__name__})",
            remediation=(
                "Configure the eight canonical SKILL_CERT and SKILL_VALIDATION "
                "fields through XDG AgentConfig."
            ),
            data={
                "configured_count": 0,
                "required_count": required_count,
                "ready": False,
                "redacted": True,
            },
        )

    path_values = (
        cfg.skill_cert_runtime_configuration,
        cfg.skill_cert_runtime_profile,
        cfg.skill_cert_release_spec,
        cfg.skill_cert_promotion_evidence,
    )
    command_values = (
        cfg.skill_cert_graphos_command,
        cfg.skill_validation_evidence_signer_command,
        cfg.skill_validation_evidence_verifier_command,
    )
    values = (*path_values, cfg.skill_cert_graphos_endpoint, *command_values)
    configured_count = sum(bool(value) for value in values)
    base_data = {
        "configured_count": configured_count,
        "required_count": required_count,
        "ready": False,
        "redacted": True,
    }
    if configured_count == 0:
        return _result(
            "skill_certification",
            "skip",
            "exact skill certification is not configured",
            data=base_data,
        )
    if configured_count != required_count:
        return _result(
            "skill_certification",
            "fail",
            "exact skill certification configuration is incomplete",
            remediation=(
                "Configure all eight canonical SKILL_CERT and SKILL_VALIDATION "
                "fields; partial certification authority is rejected."
            ),
            data=base_data,
        )

    try:
        from agent_utilities.core.paths import config_dir
        from agent_utilities.deployment.skill_validation_assets import (
            _configuration_proof,
            _identity_authority_configuration,
            _json_without_duplicates,
            _read_regular,
            _validate_profile,
        )
        from agent_utilities.skills.runtime_validation import (
            _validate_external_command_argv,
        )

        if any(value is None for value in path_values):
            raise RuntimeError("skill_certification_path_missing")
        configuration_path = Path(str(path_values[0]))
        profile_path = Path(str(path_values[1]))
        specification_path = Path(str(path_values[2]))
        promotion_path = Path(str(path_values[3]))
        configuration = _read_regular(
            configuration_path,
            limit=4 * 1024 * 1024,
            code="runtime_configuration_invalid",
        )
        profile = _read_regular(
            profile_path,
            limit=4 * 1024 * 1024,
            code="runtime_profile_invalid",
        )
        _read_regular(
            specification_path,
            limit=4 * 1024 * 1024,
            code="release_specification_invalid",
        )
        _read_regular(
            promotion_path,
            limit=8 * 1024 * 1024,
            code="promotion_evidence_invalid",
        )
        if not configuration_path.samefile(config_dir() / "config.json"):
            raise RuntimeError("runtime_configuration_not_active")
        proof = _configuration_proof(configuration)
        identity_authority = _identity_authority_configuration(
            _json_without_duplicates(
                configuration, code="runtime_configuration_invalid"
            )
        )
        _validate_profile(
            profile,
            configuration_digest=(
                "sha256:" + hashlib.sha256(configuration).hexdigest()
            ),
            model_registry_digest=str(proof["digest"]),
            identity_authority=identity_authority,
        )
        if (
            str(cfg.mcp_url or "").strip()
            != str(cfg.skill_cert_graphos_endpoint or "").strip()
        ):
            raise RuntimeError("graph_os_endpoint_not_active")
        graph_os = _validate_external_command_argv(command_values[0])
        _validate_external_command_argv(command_values[1])
        _validate_external_command_argv(command_values[2])
        if Path(graph_os[0]).name != "graph-os":
            raise RuntimeError("graph_os_executable_invalid")
    except Exception as exc:  # noqa: BLE001 - never report paths or values
        return _result(
            "skill_certification",
            "fail",
            f"exact skill certification material is unavailable ({type(exc).__name__})",
            remediation=(
                "Generate the runtime profile, provide bounded regular release "
                "inputs, select the active loopback GraphOS endpoint, and configure "
                "absolute non-shell signer and verifier argv arrays."
            ),
            data=base_data,
        )

    return _result(
        "skill_certification",
        "ok",
        "exact skill certification inputs and command boundaries are ready",
        data={
            "configured_count": required_count,
            "required_count": required_count,
            "regular_input_count": 4,
            "command_count": 3,
            "identity_authority_mode": cfg.skill_cert_identity_authority_mode,
            "identity_authority_lifecycle_owned": True,
            "identity_authority_tls_verification_required": True,
            "identity_authority_renewable_credentials_required": True,
            "ready": True,
            "redacted": True,
        },
    )


def _check_production_certification() -> dict[str, Any]:
    """Validate the complete production-campaign authority without disclosing it."""

    required_count = 13
    base_data = {
        "configured_count": 0,
        "required_count": required_count,
        "scenario_count": 0,
        "command_count": 0,
        "bearer_auth_configured": False,
        "ready": False,
        "redacted": True,
    }
    try:
        import os
        from pathlib import Path

        from agent_utilities.core.config import (
            PRODUCTION_CERTIFICATION_SCENARIOS,
            AgentConfig,
        )

        cfg = AgentConfig()
    except Exception as exc:  # noqa: BLE001 - never report paths or values
        return _result(
            "production_certification",
            "fail",
            f"production certification configuration is invalid ({type(exc).__name__})",
            remediation=(
                "Configure the current production-certification fields through "
                "XDG AgentConfig; retired direct hook variables and token files "
                "are rejected."
            ),
            data=base_data,
        )

    command_maps = (
        cfg.cert_hook_commands,
        cfg.cert_fault_action_commands,
        cfg.cert_fault_probe_commands,
    )
    tls_selector_configured = bool(
        cfg.cert_prometheus_tls_profile or cfg.cert_prometheus_tls_profile_ref
    )
    required_values = (
        cfg.certification_mode == "production",
        bool(cfg.cert_release_manifest),
        bool(cfg.cert_artifacts_dir),
        bool(cfg.cert_hardware_class),
        bool(cfg.cert_load_command),
        bool(cfg.cert_metrics_command),
        *(bool(value) for value in command_maps),
        bool(cfg.cert_evidence_signer_command),
        bool(cfg.cert_evidence_verifier_command),
        bool(cfg.cert_prometheus_url),
        tls_selector_configured,
    )
    configured_count = sum(required_values)
    base_data.update(
        {
            "configured_count": configured_count,
            "scenario_count": len(cfg.cert_hook_commands),
            "bearer_auth_configured": bool(cfg.cert_prometheus_bearer_token_ref),
        }
    )
    configured_material = any(required_values[1:]) or bool(
        cfg.cert_prometheus_bearer_token_ref
    )
    if cfg.certification_mode == "disabled" and not configured_material:
        return _result(
            "production_certification",
            "skip",
            "production certification is not configured",
            data=base_data,
        )
    if configured_count != required_count:
        return _result(
            "production_certification",
            "fail",
            "production certification configuration is incomplete",
            remediation=(
                "Set CERTIFICATION_MODE=production and configure the release, "
                "private artifacts directory, non-identifying hardware class, "
                "load/metrics commands, all three exact scenario command maps, "
                "evidence signer/verifier commands, HTTPS Prometheus endpoint, "
                "and its dedicated TLS profile selector through AgentConfig."
            ),
            data=base_data,
        )

    trust = None
    try:
        from importlib.resources import as_file, files

        import yaml

        from agent_utilities.core.transport_security import (
            resolve_configured_tls_profile,
        )
        from agent_utilities.deployment.skill_validation_assets import (
            _json_without_duplicates,
            _read_regular,
        )
        from agent_utilities.security.cli_secrets import (
            resolve_runtime_secret_reference,
        )
        from agent_utilities.skills.runtime_validation import (
            _validate_external_command_argv,
        )
        from scripts.release import check_compatibility as compatibility

        expected = set(PRODUCTION_CERTIFICATION_SCENARIOS)
        if any(set(command_map) != expected for command_map in command_maps):
            raise RuntimeError("production_certification_scenarios_not_exact")

        commands = [
            cfg.cert_load_command,
            cfg.cert_metrics_command,
            cfg.cert_evidence_signer_command,
            cfg.cert_evidence_verifier_command,
        ]
        for command_map in command_maps:
            commands.extend(
                command_map[scenario] for scenario in PRODUCTION_CERTIFICATION_SCENARIOS
            )
        if not any("{report_file}" in part for part in cfg.cert_load_command):
            raise RuntimeError("production_certification_load_report_missing")
        for command in commands:
            _validate_external_command_argv(command)

        release_path = Path(str(cfg.cert_release_manifest))
        release_payload = _read_regular(
            release_path,
            limit=64 * 1024 * 1024,
            code="production_release_manifest_invalid",
        )
        release = _json_without_duplicates(
            release_payload,
            code="production_release_manifest_invalid",
        )
        if not isinstance(release, dict):
            raise RuntimeError("production_release_manifest_invalid")
        matrix_resource = files("deploy.release").joinpath("compatibility-matrix.yml")
        with as_file(matrix_resource) as matrix_path:
            matrix_payload = _read_regular(
                matrix_path,
                limit=4 * 1024 * 1024,
                code="production_compatibility_matrix_invalid",
            )
            matrix = yaml.safe_load(matrix_payload)
            if not isinstance(matrix, dict):
                raise RuntimeError("production_compatibility_matrix_invalid")
            release_report = compatibility.verify_release_manifest(
                release,
                matrix,
                matrix_path=matrix_path,
                manifest_path=release_path,
                verify_signatures=True,
            )
        if (
            release_report.get("ok") is not True
            or release_report.get("signaturesVerified") is not True
        ):
            raise RuntimeError("production_release_signature_unverified")

        artifacts_path = Path(str(cfg.cert_artifacts_dir))
        artifacts_metadata = artifacts_path.lstat()
        if (
            artifacts_path.is_symlink()
            or not stat.S_ISDIR(artifacts_metadata.st_mode)
            or stat.S_IMODE(artifacts_metadata.st_mode) & 0o077
            or not os.access(artifacts_path, os.R_OK | os.W_OK | os.X_OK)
            or any(artifacts_path.iterdir())
        ):
            raise RuntimeError("production_artifacts_directory_invalid")

        if cfg.cert_prometheus_bearer_token_ref:
            token = resolve_runtime_secret_reference(
                cfg.cert_prometheus_bearer_token_ref
            )
            if (
                not token
                or len(token.encode("utf-8")) > 16_384
                or any(character in token for character in "\x00\r\n")
            ):
                raise RuntimeError("production_prometheus_bearer_token_invalid")

        trust = resolve_configured_tls_profile(
            "certification-prometheus",
            profile_name=cfg.cert_prometheus_tls_profile,
            profile_ref=cfg.cert_prometheus_tls_profile_ref,
            config=cfg,
        )
        if not trust.verify_enabled:
            raise RuntimeError("production_prometheus_tls_verification_disabled")
    except Exception as exc:  # noqa: BLE001 - never report paths or values
        return _result(
            "production_certification",
            "fail",
            f"production certification authority is unavailable ({type(exc).__name__})",
            remediation=(
                "Provide one signed regular release manifest, one empty private "
                "artifacts directory, 49 absolute non-shell executable argv "
                "commands, exact commands for all 15 scenarios, resolvable runtime "
                "secret references, and a verification-enforcing Prometheus TLS "
                "profile."
            ),
            data=base_data,
        )
    finally:
        if trust is not None:
            try:
                trust.cleanup()
            except Exception:  # noqa: BLE001 - cleanup cannot disclose material
                pass

    return _result(
        "production_certification",
        "ok",
        "production certification inputs and command boundaries are ready",
        data={
            "configured_count": required_count,
            "required_count": required_count,
            "scenario_count": len(PRODUCTION_CERTIFICATION_SCENARIOS),
            "command_count": len(commands),
            "bearer_auth_configured": bool(cfg.cert_prometheus_bearer_token_ref),
            "prometheus_tls_verification_required": True,
            "ready": True,
            "redacted": True,
        },
    )


def _check_graph_identity() -> dict[str, Any]:
    """Validate graph process identity without minting or exposing a token."""
    try:
        from agent_utilities.core.config import AgentConfig
        from agent_utilities.security.cli_secrets import (
            resolve_runtime_secret_reference,
        )
        from agent_utilities.security.request_identity import (
            local_process_authority_enabled,
        )

        cfg = AgentConfig()
        token_ref = str(cfg.kg_auth_token_ref or "").strip()
        oauth2 = cfg.kg_identity_oauth2
        if local_process_authority_enabled(cfg):
            return _result(
                "graph_identity",
                "ok",
                "private ephemeral graph process authority is ready",
                data={
                    "mode": "ephemeral_local",
                    "ready": True,
                    "redacted": True,
                },
            )
        if bool(token_ref) == bool(oauth2):
            return _result(
                "graph_identity",
                "fail",
                "graph process identity requires exactly one configured source",
                remediation=(
                    "Configure either KG_AUTH_TOKEN_REF or KG_IDENTITY_OAUTH2 "
                    "in XDG AgentConfig, never raw token material or both sources."
                ),
                data={"ready": False, "redacted": True},
            )
        if token_ref:
            ready = bool(resolve_runtime_secret_reference(token_ref))
            mode = "token_ref"
        else:
            assert oauth2 is not None
            secret_ref = str(oauth2.get("client_secret") or "")
            client_id = str(oauth2.get("client_id") or "")
            ready = bool(resolve_runtime_secret_reference(secret_ref))
            if client_id.startswith(("vault://", "env://", "secret://")):
                ready = ready and bool(resolve_runtime_secret_reference(client_id))
            mode = "oauth2_client_credentials"
        if not ready:
            return _result(
                "graph_identity",
                "fail",
                "graph process identity secret reference is unresolved",
                remediation="Seed the configured identity reference in the secrets backend.",
                data={"mode": mode, "ready": False, "redacted": True},
            )
        return _result(
            "graph_identity",
            "ok",
            f"graph process identity source is ready ({mode})",
            data={"mode": mode, "ready": True, "redacted": True},
        )
    except Exception as exc:  # noqa: BLE001 - doctor remains a redacted boundary
        return _result(
            "graph_identity",
            "fail",
            f"graph process identity configuration is invalid ({type(exc).__name__})",
            remediation=(
                "Validate AgentConfig and resolve its runtime secret references; "
                "no raw token or client secret is accepted."
            ),
            data={"ready": False, "redacted": True},
        )


def _check_mcp_fleet(live: bool = False) -> dict[str, Any]:
    try:
        import json
        from pathlib import Path

        from agent_utilities.core.workspace import get_mcp_config_path

        path = get_mcp_config_path()
        if not path or not Path(path).exists():
            return _result(
                "mcp_fleet",
                "skip",
                "no mcp_config.json found in workspace",
                remediation=(
                    "Create an optional fleet catalog in the AgentConfig XDG root, "
                    "or set MCP_CONFIG to an explicit external fleet catalog"
                ),
            )
        import importlib.util as _u

        spec = _u.find_spec("scripts.validate_mcp_config")
        if spec is None:
            return _result("mcp_fleet", "skip", "validate_mcp_config not importable")
        cfg = json.loads(Path(path).read_text())
        mod = importlib.import_module("scripts.validate_mcp_config")
        rep = mod.validate(
            cfg, mod.caddy_hosts() if hasattr(mod, "caddy_hosts") else set(), live=live
        )
    except Exception as exc:  # noqa: BLE001
        return _result(
            "mcp_fleet",
            "skip",
            f"fleet check skipped ({type(exc).__name__})",
        )
    safe_report = {
        "total": int(rep.get("total", 0)),
        "valid_count": len(rep.get("ok", [])),
        "invalid_count": len(rep.get("invalid", {})),
        "unreachable_count": len(rep.get("unreachable", {})),
        "missing_route_count": len(rep.get("missing_from_config", [])),
        "passed": bool(rep.get("passed", False)),
        "redacted": True,
    }
    if rep.get("passed"):
        return _result(
            "mcp_fleet",
            "ok",
            f"{safe_report['valid_count']} MCP server(s) valid",
            data=safe_report,
        )
    bad = {**rep.get("invalid", {}), **rep.get("unreachable", {})}
    return _result(
        "mcp_fleet",
        "warn",
        f"{len(bad)} MCP server(s) need attention",
        remediation="`python scripts/validate_mcp_config.py --live` for detail",
        data=safe_report,
    )


def _check_mcp_fleet_secrets() -> dict[str, Any]:
    """Validate neutral fleet alias resolution without disclosing alias metadata."""

    data = {
        "configured_alias_count": 0,
        "direct_alias_count": 0,
        "mapped_alias_count": 0,
        "unresolved_alias_count": 0,
        "redacted": True,
    }
    try:
        from agent_utilities.core.config import AgentConfig, setting
        from agent_utilities.security.cli_secrets import (
            resolve_runtime_secret_reference,
        )

        mappings = AgentConfig().mcp_fleet_secret_refs
        if not isinstance(mappings, dict) or len(mappings) > 512:
            raise ValueError("invalid fleet secret alias mapping")
        data["configured_alias_count"] = len(mappings)
        for alias, reference in mappings.items():
            try:
                direct = setting(alias)
                if direct not in (None, ""):
                    if any(character in str(direct) for character in "\x00\r\n"):
                        raise ValueError("invalid direct runtime material")
                    data["direct_alias_count"] += 1
                    continue
                resolved = resolve_runtime_secret_reference(reference)
                if resolved in (None, "") or any(
                    character in str(resolved) for character in "\x00\r\n"
                ):
                    raise ValueError("unavailable runtime reference")
                data["mapped_alias_count"] += 1
            except Exception:  # noqa: BLE001 - aliases and references stay redacted
                data["unresolved_alias_count"] += 1
    except Exception as exc:  # noqa: BLE001 - doctor remains a redacted boundary
        return _result(
            "mcp_fleet_secrets",
            "fail",
            f"MCP fleet secret alias configuration is invalid ({type(exc).__name__})",
            remediation=(
                "Configure MCP_FLEET_SECRET_REFS as neutral aliases mapped only "
                "to env://, vault://, or secret:// runtime references."
            ),
            data=data,
        )
    if data["unresolved_alias_count"]:
        return _result(
            "mcp_fleet_secrets",
            "fail",
            "one or more MCP fleet secret aliases cannot be resolved",
            remediation=(
                "Project the direct runtime alias or repair its configured "
                "runtime secret reference."
            ),
            data=data,
        )
    return _result(
        "mcp_fleet_secrets",
        "ok",
        "MCP fleet secret alias resolution is ready",
        data=data,
    )


def _check_provider_profiles() -> dict[str, Any]:
    """Resolve enabled provider profiles without exposing deployment metadata."""

    data = {
        "configured_count": 0,
        "enabled_count": 0,
        "disabled_count": 0,
        "ready_count": 0,
        "invalid_count": 0,
        "redacted": True,
    }
    try:
        from agent_utilities.core.config import AgentConfig
        from agent_utilities.core.provider_runtime import (
            prepare_provider_runtime_child_environment,
        )

        cfg = AgentConfig()
        profiles = cfg.provider_configs
        if not isinstance(profiles, dict) or len(profiles) > 256:
            raise ValueError("provider profile mapping is invalid")
        data["configured_count"] = len(profiles)
        for profile_name, profile in profiles.items():
            if not profile.enabled:
                data["disabled_count"] += 1
                continue
            data["enabled_count"] += 1
            try:
                prepared = prepare_provider_runtime_child_environment(
                    profile_name, config=cfg
                )
                prepared.close()
                data["ready_count"] += 1
            except Exception:  # noqa: BLE001 - deployment details stay redacted
                data["invalid_count"] += 1
    except Exception as exc:  # noqa: BLE001 - doctor is a redaction boundary
        return _result(
            "provider_profiles",
            "fail",
            f"provider runtime profile configuration is invalid ({type(exc).__name__})",
            remediation=(
                "Configure PROVIDER_CONFIGS with neutral profile names, runtime "
                "references, and explicit TLS profile selectors."
            ),
            data=data,
        )
    if not data["configured_count"]:
        return _result(
            "provider_profiles",
            "skip",
            "no external provider runtime profiles are configured",
            data=data,
        )
    if data["invalid_count"]:
        return _result(
            "provider_profiles",
            "fail",
            "one or more enabled provider runtime profiles are unavailable",
            remediation=(
                "Repair the referenced endpoint, credential, selector, or TLS "
                "profile in the deployment-owned configuration."
            ),
            data=data,
        )
    return _result(
        "provider_profiles",
        "ok",
        "enabled provider runtime profiles are ready",
        data=data,
    )


def _check_hooks() -> dict[str, Any]:
    try:
        from agent_utilities.ecosystem.hook_installer import HookInstaller

        rep = HookInstaller().doctor()
    except Exception as exc:  # noqa: BLE001
        return _result(
            "hooks", "skip", f"hook doctor unavailable ({type(exc).__name__})"
        )
    installed = [k for k, v in rep.items() if v.get("status") == "healthy"]
    stale = [k for k, v in rep.items() if v.get("status") == "stale"]
    if stale:
        return _result(
            "hooks",
            "warn",
            f"{len(stale)} stale hook(s): {stale}",
            remediation="re-install hooks (`graph_configure action=install_hooks`)",
            auto_fixable=True,
            data=rep,
        )
    return _result("hooks", "ok", f"{len(installed)} agent hook(s) healthy", data=rep)


def _check_observability() -> dict[str, Any]:
    from agent_utilities.core.config import AgentConfig, setting
    from agent_utilities.core.profile_guard import is_production_profile

    try:
        from agent_utilities.core.transport_security import (
            resolve_configured_tls_profile,
        )
        from agent_utilities.observability.custom_observability import _same_origin
        from agent_utilities.observability.langfuse_trust import (
            resolve_langfuse_credentials,
            resolve_langfuse_host,
        )
        from agent_utilities.security.cli_secrets import (
            resolve_runtime_secret_reference,
        )

        cfg = AgentConfig()
        production = is_production_profile()
        metrics = setting("GATEWAY_METRICS", False, cast=bool)
        langfuse_pair = bool(
            cfg.langfuse_public_key_ref and cfg.langfuse_secret_key_ref
        )
        endpoint = str(cfg.otel_exporter_otlp_endpoint or "").strip()
        endpoint_derived = not endpoint and langfuse_pair
        if endpoint_derived:
            endpoint = f"{resolve_langfuse_host('').rstrip('/')}/api/public/otel"
        langfuse_auth = bool(
            langfuse_pair
            and endpoint
            and _same_origin(endpoint, resolve_langfuse_host(""))
        )
        header_auth = bool(cfg.otel_exporter_otlp_headers_ref)
        key_auth = bool(
            cfg.otel_exporter_otlp_public_key_ref
            and cfg.otel_exporter_otlp_secret_key_ref
        )
        auth_ready = header_auth or key_auth or langfuse_auth
        data = {
            "enabled": bool(cfg.enable_otel),
            "endpoint_configured": bool(endpoint),
            "endpoint_derived_from_langfuse": endpoint_derived,
            "auth_reference_configured": auth_ready,
            "tls_profile_configured": bool(
                cfg.otel_tls_profile
                or cfg.otel_tls_profile_ref
                or (
                    langfuse_auth
                    and (cfg.langfuse_tls_profile or cfg.langfuse_tls_profile_ref)
                )
            ),
            "metrics_enabled": bool(metrics),
            "metadata_only": True,
            "redacted": True,
        }
        if not cfg.enable_otel and not production:
            return _result(
                "observability",
                "skip",
                "metadata-only OTLP export is disabled",
                data=data,
            )
        if not endpoint or not auth_ready:
            return _result(
                "observability",
                "fail" if cfg.enable_otel else "warn",
                "OTLP endpoint or authentication references are incomplete",
                remediation=(
                    "Configure OTLP reference-based authentication, or configure the "
                    "Langfuse reference pair for automatic same-origin OTLP wiring."
                ),
                skill="service-observability-provisioner",
                data=data,
            )
        if header_auth:
            resolve_runtime_secret_reference(cfg.otel_exporter_otlp_headers_ref)
        elif key_auth:
            resolve_runtime_secret_reference(cfg.otel_exporter_otlp_public_key_ref)
            resolve_runtime_secret_reference(cfg.otel_exporter_otlp_secret_key_ref)
        else:
            resolve_langfuse_credentials(agent_config=cfg)
        profile_name = cfg.otel_tls_profile
        profile_ref = cfg.otel_tls_profile_ref
        if langfuse_auth and not (profile_name or profile_ref):
            profile_name = cfg.langfuse_tls_profile
            profile_ref = cfg.langfuse_tls_profile_ref
        trust = resolve_configured_tls_profile(
            "OTEL",
            profile_name=profile_name,
            profile_ref=profile_ref,
            config=cfg,
        )
        data["tls_valid"] = trust.verify_enabled
        if production and not metrics:
            return _result(
                "observability",
                "warn",
                "metadata-only OTLP tracing is ready; gateway metrics are disabled",
                remediation="Enable GATEWAY_METRICS for the production profile.",
                skill="service-observability-provisioner",
                data=data,
            )
        return _result(
            "observability",
            "ok",
            "metadata-only OTLP authentication and TLS are ready",
            data=data,
        )
    except Exception as exc:  # noqa: BLE001 - doctor remains privacy-safe
        return _result(
            "observability",
            "error",
            f"OTLP readiness check failed ({type(exc).__name__})",
            data={"redacted": True, "metadata_only": True},
        )


def _run_async_doctor_probe(factory: Callable[[], Any]) -> Any:
    """Run an async probe from either a CLI or an already-running MCP loop."""
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(factory())

    # ``graph_configure(action=system_doctor)`` is itself async. Keep the sync
    # doctor API stable while giving the child transport its own event loop.
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="doctor-probe") as pool:
        return pool.submit(lambda: asyncio.run(factory())).result()


def _langfuse_rows(payload: Any) -> list[dict[str, Any]]:
    """Extract only bounded trace rows from a successful public API response."""
    if not isinstance(payload, dict):
        return []
    rows = payload.get("data")
    if not isinstance(rows, list):
        rows = payload.get("traces")
    if not isinstance(rows, list):
        return []
    return [row for row in rows[:100] if isinstance(row, dict)]


def _probe_langfuse_mcp_visibility(cfg: Any) -> bool:
    """Prove the mounted child can execute the current privacy-safe contract."""
    from pathlib import Path

    from agent_utilities.mcp.multiplexer import (
        MCPMultiplexer,
        _child_result_payload,
        attest_runtime_child_config,
    )
    from agent_utilities.observability.langfuse_trust import (
        native_langfuse_mcp_config,
    )

    child = native_langfuse_mcp_config(agent_config=cfg)
    if child is None:
        return False
    # Bound the diagnostic child startup independently of its operational
    # profile. The cached catalog avoids reading or persisting any local path.
    child = dict(child)
    child["timeout"] = min(float(child.get("timeout", 60.0)), 30.0)
    child = attest_runtime_child_config(child)

    async def probe() -> bool:
        mux = MCPMultiplexer(Path())
        mux._catalog = {"langfuse-mcp": child}
        try:
            await mux.mount_child("langfuse-mcp")
            matches = [
                prefixed
                for prefixed, (server, original) in mux.tool_to_server.items()
                if server == "langfuse-mcp" and original == "langfuse_observability"
            ]
            runtime = mux.children.get("langfuse-mcp")
            if len(matches) != 1 or runtime is None:
                return False

            posture_result = await runtime.call_tool(
                "langfuse_observability",
                {"action": "runtime_posture"},
            )
            if bool(getattr(posture_result, "isError", False)) or bool(
                getattr(posture_result, "is_error", False)
            ):
                return False
            posture = _child_result_payload(posture_result)
            if posture != {
                "content_capture_enabled": False,
                "metadata_only": True,
            }:
                return False

            # Execute the read through the mounted child itself. Direct API
            # reachability cannot prove that the child received the same host,
            # credential, and TLS contract. Keep the response bounded and
            # transient; no returned row enters doctor output.
            trace_result = await runtime.call_tool(
                "langfuse_observability",
                {
                    "action": "trace_list",
                    "page": 1,
                    "limit": 1,
                    "fields": "core",
                },
            )
            if bool(getattr(trace_result, "isError", False)) or bool(
                getattr(trace_result, "is_error", False)
            ):
                return False
            trace_payload = _child_result_payload(trace_result)
            rows = (
                trace_payload.get("data") if isinstance(trace_payload, dict) else None
            )
            return isinstance(rows, list) and len(rows) <= 1
        finally:
            await mux.aclose()

    return bool(_run_async_doctor_probe(probe))


def _probe_langfuse_live(cfg: Any) -> dict[str, Any]:
    """Prove API, optional MCP, and optional metadata-only trace round trip."""
    import time
    import uuid
    from datetime import UTC, datetime

    result: dict[str, Any] = {
        "live_probed": True,
        "api_reachable": False,
        "mcp_visible": None,
        "trace_round_trip": None,
        "redacted": True,
    }
    try:
        from langfuse_agent.api_client import LangfuseApi

        from agent_utilities.observability.langfuse_trust import (
            resolve_langfuse_credentials,
            resolve_langfuse_requests_transport,
        )

        public_key, secret_key = resolve_langfuse_credentials(agent_config=cfg)
        transport_kwargs = resolve_langfuse_requests_transport(agent_config=cfg)
        api = LangfuseApi(
            public_key=public_key,
            secret_key=secret_key,
            host=cfg.langfuse_host,
            timeout=10.0,
            transport_kwargs=transport_kwargs,
        )
        handshake = api.trace_list(page=1, limit=1, fields="core")
        if not isinstance(handshake, dict):
            result["error_code"] = "api_response_invalid"
            return result
        result["api_reachable"] = True
    except Exception:  # noqa: BLE001 - expose only a stable diagnostic code
        result["error_code"] = "api_handshake_failed"
        return result

    if cfg.langfuse_mcp_enabled:
        try:
            result["mcp_visible"] = _probe_langfuse_mcp_visibility(cfg)
        except Exception:  # noqa: BLE001 - child details may contain local material
            result["mcp_visible"] = False
        if not result["mcp_visible"]:
            result["error_code"] = "mcp_visibility_failed"
            return result

    if not cfg.trace_export_enabled:
        return result

    # The source token is random and never leaves this function. The exporter
    # turns it into a tenant-qualified opaque identifier before persistence;
    # input and caller metadata are intentionally empty.
    source_run_id = uuid.uuid4().hex
    try:
        from agent_utilities.observability.langfuse_exporter import LangfuseExporter
        from agent_utilities.usage.privacy import normalize_run_id

        try:
            from agent_utilities.security.brain_context import current_actor

            tenant_id = current_actor().tenant_id
        except Exception:  # noqa: BLE001 - empty tenant namespace is opaque too
            tenant_id = ""
        expected_name = (
            f"graph_run:{normalize_run_id(source_run_id, tenant_id=tenant_id)}"
        )
        started_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        exporter = LangfuseExporter(
            public_key=public_key,
            secret_key=secret_key,
            host=cfg.langfuse_host,
        )
        emitted = exporter.export_graph_run(
            run_id=source_run_id,
            query="",
            status="success",
            metadata={},
        )
        exporter.flush()
        if not emitted:
            result["error_code"] = "trace_export_failed"
            result["trace_round_trip"] = False
            return result
        for _ in range(10):
            traces = api.trace_list(
                page=1,
                limit=10,
                name=expected_name,
                from_timestamp=started_at,
                order_by="timestamp.desc",
                fields="core,basic",
            )
            if any(row.get("name") == expected_name for row in _langfuse_rows(traces)):
                result["trace_round_trip"] = True
                return result
            time.sleep(1.0)
    except Exception:  # noqa: BLE001 - never expose response, endpoint, or identity
        pass
    result["trace_round_trip"] = False
    result["error_code"] = "trace_round_trip_failed"
    return result


def _check_langfuse(live: bool = False) -> dict[str, Any]:
    """Validate Langfuse statically, or prove its live privacy-safe paths."""
    try:
        from agent_utilities.core.config import AgentConfig
        from agent_utilities.observability.langfuse_trust import (
            LangfuseTrustError,
            configure_langfuse_trust,
            langfuse_credentials_configured,
            langfuse_provider_contract_ready,
            resolve_langfuse_credentials,
            resolve_langfuse_persistence_hmac_key,
        )

        cfg = AgentConfig()
        public_input = bool(cfg.langfuse_public_key_ref)
        secret_input = bool(cfg.langfuse_secret_key_ref)
        enabled = bool(
            cfg.langfuse_mcp_enabled
            or cfg.kg_failure_evolution
            or cfg.trace_export_enabled
            or cfg.langfuse_kg_auto_ingest
        )
        executable_ready = langfuse_provider_contract_ready()
        data = {
            "enabled": enabled,
            "credential_pair_configured": public_input and secret_input,
            "credential_refs_configured": bool(
                cfg.langfuse_public_key_ref and cfg.langfuse_secret_key_ref
            ),
            "credential_material_ready": False,
            "tls_profile_configured": bool(
                cfg.langfuse_tls_profile or cfg.langfuse_tls_profile_ref
            ),
            "persistence_enabled": bool(cfg.langfuse_kg_auto_ingest),
            "persistence_key_ref_configured": bool(
                cfg.langfuse_persistence_hmac_key_ref
            ),
            "persistence_key_ready": False,
            "mcp_launcher_available": executable_ready,
            "mcp_launcher_required": bool(cfg.langfuse_mcp_enabled),
            "live_probed": False,
            "redacted": True,
        }
        if not enabled and not public_input and not secret_input:
            return _result(
                "langfuse",
                "skip",
                "Langfuse integration is not configured",
                data=data,
            )
        if public_input != secret_input or not langfuse_credentials_configured(
            agent_config=cfg
        ):
            return _result(
                "langfuse",
                "fail",
                "Langfuse credential configuration is incomplete",
                remediation=(
                    "Configure LANGFUSE_PUBLIC_KEY_REF and LANGFUSE_SECRET_KEY_REF; "
                    "resolve their material only at the runtime boundary."
                ),
                data=data,
            )
        try:
            resolve_langfuse_credentials(agent_config=cfg)
        except LangfuseTrustError as exc:
            data["error_code"] = exc.reason
            return _result(
                "langfuse",
                "fail",
                "Langfuse credential material is unavailable or invalid",
                remediation=(
                    "Verify both secret references resolve to real runtime key material; "
                    "redaction masks and unresolved templates are rejected locally."
                ),
                data=data,
            )
        except Exception:  # noqa: BLE001 - never expose provider details
            data["error_code"] = "langfuse_credentials_invalid"
            return _result(
                "langfuse",
                "fail",
                "Langfuse credential material is unavailable or invalid",
                remediation=(
                    "Verify both secret references resolve to real runtime key material; "
                    "redaction masks and unresolved templates are rejected locally."
                ),
                data=data,
            )
        data["credential_material_ready"] = True
        if cfg.langfuse_kg_auto_ingest and not cfg.langfuse_persistence_hmac_key_ref:
            return _result(
                "langfuse",
                "fail",
                "Langfuse graph persistence requires a dedicated identity key",
                remediation=(
                    "Configure LANGFUSE_PERSISTENCE_HMAC_KEY_REF; the project API "
                    "secret is never reused for identity derivation."
                ),
                data=data,
            )
        if cfg.langfuse_persistence_hmac_key_ref:
            try:
                resolve_langfuse_persistence_hmac_key(agent_config=cfg)
            except Exception:  # noqa: BLE001 - keep secret-provider details private
                return _result(
                    "langfuse",
                    "fail",
                    "Langfuse persistence identity key is unavailable",
                    remediation=(
                        "Verify LANGFUSE_PERSISTENCE_HMAC_KEY_REF resolves to at "
                        "least 32 bytes at the runtime boundary."
                    ),
                    data=data,
                )
            data["persistence_key_ready"] = True
        trust = configure_langfuse_trust(agent_config=cfg)
        data["tls_valid"] = trust.valid
        data["custom_trust_configured"] = trust.configured
        if not trust.valid:
            return _result(
                "langfuse",
                "fail",
                f"Langfuse TLS configuration is invalid ({trust.reason or 'invalid'})",
                remediation=(
                    "Configure a valid LANGFUSE_TLS_PROFILE_REF or runtime trust "
                    "environment; TLS verification cannot be disabled."
                ),
                data=data,
            )
        if cfg.langfuse_mcp_enabled and not executable_ready:
            data["error_code"] = "langfuse_mcp_provider_contract_unavailable"
            return _result(
                "langfuse",
                "fail",
                "Langfuse MCP is enabled but its current child contract is unavailable",
                remediation=(
                    "Install the current agent-utilities[serving] artifact in the "
                    "GraphOS runtime environment."
                ),
                data=data,
            )
        if not enabled:
            return _result(
                "langfuse",
                "warn",
                "Langfuse credentials are ready but all integrations are disabled",
                remediation=(
                    "Enable metadata-only trace export, MCP access, or governed "
                    "failure evolution as required."
                ),
                data=data,
            )
        if not live:
            return _result(
                "langfuse",
                "ok",
                "Langfuse credentials and TLS are statically valid; live proof was not requested",
                data=data,
            )
        live_data = _probe_langfuse_live(cfg)
        data.update(live_data)
        live_ok = bool(data.get("api_reachable"))
        if cfg.langfuse_mcp_enabled:
            live_ok = live_ok and data.get("mcp_visible") is True
        if cfg.trace_export_enabled:
            live_ok = live_ok and data.get("trace_round_trip") is True
        if not live_ok:
            return _result(
                "langfuse",
                "fail",
                "Langfuse live proof failed",
                remediation=(
                    "Verify the runtime secret references, TLS profile, API reachability, "
                    "and the Langfuse MCP child installation; diagnostic output is redacted."
                ),
                data=data,
            )
        return _result(
            "langfuse",
            "ok",
            "Langfuse API and enabled live paths are proven",
            data=data,
        )
    except Exception as exc:  # noqa: BLE001 - doctor output remains redacted
        return _result(
            "langfuse",
            "error",
            f"Langfuse readiness check failed ({type(exc).__name__})",
            data={"redacted": True},
        )


def _probe_native_optimizer_live() -> dict[str, Any]:
    """Submit one content-free ProgramOptimize job to the active authority."""
    from agent_utilities.harness.optimization_backend import (
        OptimizationRequest,
        try_native_optimization,
    )
    from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine

    engine = GraphComputeEngine.get_active()
    if engine is None:
        return {
            "live_probed": True,
            "operational": False,
            "error_code": "engine_authority_inactive",
            "privacy_safe_payload": True,
        }
    request = OptimizationRequest(
        target="diagnostic",
        objective="native-capability-probe",
        data={
            "examples": [
                {
                    "task": "synthetic-capability-probe",
                    "response": "synthetic-capability-result",
                    "success": True,
                }
            ]
        },
    )
    attempt = try_native_optimization(engine, request)
    out = {
        "live_probed": True,
        "operational": attempt.disposition == "completed",
        "privacy_safe_payload": True,
    }
    if attempt.disposition != "completed":
        out["error_code"] = attempt.error_code or f"native_{attempt.disposition}"
    return out


def _check_native_optimizer(live: bool = False) -> dict[str, Any]:
    """Report installed surface separately from a live ProgramOptimize proof."""
    try:
        from agent_utilities.core.config import AgentConfig
        from agent_utilities.knowledge_graph.core.graph_compute import (
            GraphComputeEngine,
        )

        cfg = AgentConfig()
        enabled = bool(cfg.kg_optimization_enabled)
        surface_available = callable(
            getattr(GraphComputeEngine, "optimize_program", None)
        )
        data = {
            "enabled": enabled,
            "native_surface_available": surface_available,
            "live_probed": False,
            "operational": None,
            "privacy_safe_payload": True,
        }
        if not enabled:
            return _result(
                "native_optimizer",
                "skip",
                "native program optimization is disabled",
                data=data,
            )
        if not surface_available:
            return _result(
                "native_optimizer",
                "fail",
                "the native ProgramOptimize surface is unavailable",
                remediation="Install the unified agent-utilities engine distribution.",
                data=data,
            )
        if not live:
            return _result(
                "native_optimizer",
                "ok",
                "the ProgramOptimize surface is installed; live proof was not requested",
                data=data,
            )
        live_data = _probe_native_optimizer_live()
        data.update(live_data)
        if not data["operational"]:
            return _result(
                "native_optimizer",
                "fail",
                "the active engine did not complete the ProgramOptimize capability probe",
                remediation=(
                    "Run the live doctor inside GraphOS after its engine authority is active, "
                    "then inspect the engine health check if ProgramOptimize still fails."
                ),
                data=data,
            )
        return _result(
            "native_optimizer",
            "ok",
            "the active engine completed a governed ProgramOptimize job",
            data=data,
        )
    except Exception as exc:  # noqa: BLE001 - never expose engine response material
        return _result(
            "native_optimizer",
            "error",
            f"native optimizer readiness check failed ({type(exc).__name__})",
            data={"redacted": True, "live_probed": live},
        )


def _check_graph_connections(live: bool = False) -> dict[str, Any]:
    """Validate graph declarations and optionally prove their native read paths.

    Every declaration, including sources declared only in ``KG_CONNECTIONS``, is
    checked on every run. Network probes run only for an explicit live doctor.
    Public output is aggregate metadata: aliases, endpoints, refs, identities,
    source rows, and exception details never cross the doctor boundary.
    """
    try:
        from agent_utilities.core.config import AgentConfig
        from agent_utilities.knowledge_graph.core.connection_registry import (
            validate_persistable_connection_spec,
        )
        from agent_utilities.mcp.kg_server import get_connection_registry

        cfg = AgentConfig()
        external_declarations: list[dict[str, Any]] = []
        for declared in cfg.external_graph_connectors or []:
            value = (
                declared.model_dump(exclude_none=True, exclude_defaults=True)
                if hasattr(declared, "model_dump")
                else dict(declared)
            )
            value["role"] = "read"
            external_declarations.append(value)
        kg_declarations = [dict(value) for value in (cfg.kg_connections or [])]
        declarations = [*external_declarations, *kg_declarations]

        invalid_declaration_count = 0
        for declaration in declarations:
            try:
                validate_persistable_connection_spec(declaration)
                if not str(declaration.get("name") or "").strip():
                    raise ValueError("connection declaration has no name")
            except Exception:  # noqa: BLE001 - expose only aggregate counts
                invalid_declaration_count += 1

        # KG_CONNECTIONS intentionally overrides an EXTERNAL_GRAPH_CONNECTORS
        # declaration with the same alias. Duplicates within either source are
        # invalid and cannot be hidden by that precedence rule.
        external_names = [
            str(value.get("name") or "").strip()
            for value in external_declarations
            if str(value.get("name") or "").strip()
        ]
        kg_names = [
            str(value.get("name") or "").strip()
            for value in kg_declarations
            if str(value.get("name") or "").strip()
        ]
        duplicate_declaration_count = (
            len(external_names)
            - len(set(external_names))
            + len(kg_names)
            - len(set(kg_names))
        )
        effective_names = set(external_names) | set(kg_names)
        registry = get_connection_registry()
        status = registry.status()
        conns = [
            connection
            for connection in status.get("connections", [])
            if connection.get("name") != "default"
        ]
        registered_names = {
            str(connection.get("name") or "").strip()
            for connection in conns
            if str(connection.get("name") or "").strip()
        }
        missing_declaration_count = len(effective_names - registered_names)
    except Exception:  # noqa: BLE001 - never expose deployment details
        return _result(
            "graph_connections",
            "fail",
            "graph connection registry is invalid",
            remediation=(
                "Repair KG_CONNECTIONS and external graph declarations; keep all "
                "transport, auth, and TLS material behind runtime references."
            ),
            data={"ready": False, "redacted": True, "live_probed": live},
        )

    probe_failed_count = 0
    ready_count = 0
    if live:
        for name in sorted(registered_names):
            try:
                if registry.probe(name):
                    ready_count += 1
                else:
                    probe_failed_count += 1
            except Exception:  # noqa: BLE001 - never expose connector details
                probe_failed_count += 1

    stalled_count = 0
    try:
        from agent_utilities.knowledge_graph.backends import get_active_backend
        from agent_utilities.knowledge_graph.backends.fanout_backend import (
            FanOutBackend,
        )

        backend = get_active_backend()
        cand = getattr(backend, "inner", backend)
        fan = cand if isinstance(cand, FanOutBackend) else None
        if isinstance(fan, FanOutBackend):
            mirrors = fan.durability_stats().get("mirrors") or {}
            stalled_count = sum(
                bool(state.get("stalled")) for state in mirrors.values()
            )
    except Exception:  # noqa: BLE001 — mirror stats are best-effort
        pass

    by_role: dict[str, int] = {}
    for connection in conns:
        role = str(connection.get("role") or "read")
        by_role[role] = by_role.get(role, 0) + 1
    data = {
        "configured_count": len(effective_names),
        "registered_count": len(registered_names),
        "ready_count": ready_count,
        "probe_failed_count": probe_failed_count,
        "invalid_declaration_count": invalid_declaration_count,
        "duplicate_declaration_count": duplicate_declaration_count,
        "missing_declaration_count": missing_declaration_count,
        "stalled_mirror_count": stalled_count,
        "roles": by_role,
        "redacted": True,
        "live_probed": live,
    }
    configuration_failures = (
        invalid_declaration_count
        + duplicate_declaration_count
        + missing_declaration_count
    )
    if configuration_failures or probe_failed_count:
        return _result(
            "graph_connections",
            "fail",
            (
                f"{len(registered_names)} external connection(s); "
                f"{configuration_failures} declaration failure(s), "
                f"{probe_failed_count} runtime probe failure(s)"
            ),
            remediation=(
                "Repair the referenced connection, authentication, and TLS profiles, "
                "then rerun the live doctor."
            ),
            skill="database-environment-setup",
            data=data,
        )
    if stalled_count:
        return _result(
            "graph_connections",
            "warn",
            f"{len(registered_names)} connection(s); {stalled_count} stalled mirror(s)",
            remediation="`graph_configure action=reconcile` and check the mirror backend",
            skill="database-environment-setup",
            data=data,
        )
    if not registered_names:
        detail = "no external connections registered"
    elif live:
        detail = f"{ready_count}/{len(registered_names)} external connection(s) ready"
    else:
        detail = (
            f"{len(registered_names)} external connection declaration(s) valid; "
            "live proof not requested"
        )
    return _result("graph_connections", "ok", detail, data=data)


def _check_ingestion_coverage() -> dict[str, Any]:
    """Assert the agent-packages repos are ingested + fresh (CONCEPT:AU-OS.deployment.flagging-repos).

    Native codebase-context-via-KG requires the index to be reliably populated:
    if a repo has no ``:Code`` symbols (or its last delta sync is stale) a KG code
    query returns nothing and the agent silently falls back to grep. This compares
    ``workspace.yml``'s agent-packages subtree against the live KG + DeltaManifest
    freshness, so coverage gaps are visible rather than silent (GAP 1). Repository
    identities remain internal; the doctor result contains aggregate counts only."""
    try:
        from agent_utilities.knowledge_graph.ingestion.coverage import (
            assess_coverage,
            enumerate_agent_packages_repos,
            find_workspace_manifest,
            repo_symbol_counts,
        )

        manifest = find_workspace_manifest()
        if manifest is None:
            return _result(
                "ingestion_coverage",
                "skip",
                "workspace.yml not found (not a workspace checkout)",
            )
        repos = enumerate_agent_packages_repos(manifest)
        if not repos:
            return _result(
                "ingestion_coverage", "skip", "no agent-packages repos in workspace.yml"
            )
        from agent_utilities.knowledge_graph.backends import get_active_backend

        backend = get_active_backend()
        counts = repo_symbol_counts(backend, repos)
    except Exception as exc:  # noqa: BLE001
        return _result(
            "ingestion_coverage",
            "skip",
            f"coverage probe unavailable ({type(exc).__name__})",
        )

    freshness: dict[str, str] = {}
    try:
        from agent_utilities.knowledge_graph.ingestion.manifest import DeltaManifest

        dm = DeltaManifest(backend=backend)
        for cat in ("codebase", "codebase_file"):
            freshness.update(dm.freshness("agent_graph", cat))
    except Exception:  # noqa: BLE001 — freshness is best-effort
        freshness = {}

    rep = assess_coverage(repos, counts, freshness)
    missing_count = len(rep["missing"])
    stale_count = len(rep["stale"])
    data = {
        "total": rep["total"],
        "covered": rep["covered"],
        "missing_count": missing_count,
        "stale_count": stale_count,
        "coverage_pct": rep["coverage_pct"],
        "total_symbols": rep["total_symbols"],
        "sla_days": rep["sla_days"],
        "redacted": True,
    }
    detail = (
        f"{rep['covered']}/{rep['total']} agent-packages repos ingested "
        f"({rep['coverage_pct']}%), {rep['total_symbols']} symbols"
    )
    if missing_count:
        detail += f", {missing_count} missing"
    if stale_count:
        detail += f", {stale_count} stale (>{rep['sla_days']}d)"
    if missing_count or stale_count:
        status = "fail" if rep["coverage_pct"] < 75 else "warn"
        return _result(
            "ingestion_coverage",
            status,
            detail,
            remediation="`source_sync source=all mode=delta` to ingest or refresh configured repositories",
            skill="graph-ingestion-and-integration",
            data=data,
        )
    return _result("ingestion_coverage", "ok", detail, data=data)


def _check_connector_coverage() -> dict[str, Any]:
    """Assert every configured connector is ingesting + fresh (CONCEPT:AU-OS.deployment.connector-coverage-check).

    The connector analogue of ``ingestion_coverage``: a dark or stale connector
    means the world-model for that domain (tickets, deploys, processes…) is silently
    wrong and the agent falls back to hitting the source system. Compares the
    expected connector set against their ``DeltaManifest`` watermarks. Connector
    identities remain internal; the doctor result contains aggregate counts only."""
    try:
        from agent_utilities.knowledge_graph.backends import get_active_backend
        from agent_utilities.knowledge_graph.ingestion.connector_coverage import (
            CONNECTOR_CATEGORY,
            assess_connector_coverage,
            enumerate_expected_connectors,
        )
        from agent_utilities.knowledge_graph.ingestion.manifest import DeltaManifest

        expected = enumerate_expected_connectors()
        if not expected:
            return _result("connector_coverage", "skip", "no connectors configured")
        backend = get_active_backend()
        dm = DeltaManifest(backend=backend)
        freshness: dict[str, str] = {}
        for graph in ("agent_graph", "__commons__"):
            freshness.update(dm.freshness(graph, CONNECTOR_CATEGORY))
    except Exception as exc:  # noqa: BLE001
        return _result(
            "connector_coverage",
            "skip",
            f"connector probe unavailable ({type(exc).__name__})",
        )

    rep = assess_connector_coverage(expected, freshness)
    missing_count = len(rep["missing"])
    stale_count = len(rep["stale"])
    data = {
        "total": rep["total"],
        "covered": rep["covered"],
        "missing_count": missing_count,
        "stale_count": stale_count,
        "coverage_pct": rep["coverage_pct"],
        "sla_days": rep["sla_days"],
        "redacted": True,
    }
    detail = (
        f"{rep['covered']}/{rep['total']} connectors ingesting ({rep['coverage_pct']}%)"
    )
    if missing_count:
        detail += f", {missing_count} dark"
    if stale_count:
        detail += f", {stale_count} stale (>{rep['sla_days']}d)"
    if missing_count or stale_count:
        return _result(
            "connector_coverage",
            "warn",
            detail,
            remediation=(
                "`source_sync source=all mode=delta` to refresh configured sources; "
                "verify their runtime credential references and presets"
            ),
            skill="graph-ingestion-and-integration",
            data=data,
        )
    return _result("connector_coverage", "ok", detail, data=data)


def _check_workspace_config() -> dict[str, Any]:
    """Validate the ``workspace.yml`` repository manifest (CONCEPT:AU-OS.deployment.os-4).

    ``workspace.yml`` is the canonical map of the ecosystem's repositories: the
    bootstrap (``clone_missing_projects``), the read-only project enumeration that
    self-configures KG ingestion breadth (``workspace_project_roots``, KG-2.7), and
    genesis all parse it. A malformed manifest, a repository entry with no ``url``,
    or an incoherent ``subdirectories`` shape silently shrinks what the platform
    clones/ingests — so we validate it through the SAME loader (no re-parse) and
    surface gaps as a doctor finding rather than a silent miss. The manifest path and
    entry-specific validation details never cross the doctor reporting boundary."""
    try:
        from agent_utilities.core.workspace_config import validate_workspace_yml

        rep = validate_workspace_yml()
    except Exception as exc:  # noqa: BLE001
        return _result(
            "workspace_config",
            "skip",
            f"workspace.yml validator unavailable ({type(exc).__name__})",
        )

    if not rep["found"]:
        return _result(
            "workspace_config",
            "skip",
            "no workspace.yml found (not a workspace checkout)",
            remediation=(
                "copy docs/examples/workspace.yml to the workspace root (or the "
                "agent-utilities XDG config dir) and edit it for your repos"
            ),
        )

    data = {
        "found": bool(rep["found"]),
        "parsed": bool(rep["parsed"]),
        "repo_count": int(rep["repo_count"]),
        "error_count": len(rep["errors"]),
        "warning_count": len(rep["warnings"]),
        "redacted": True,
    }
    if rep["errors"]:
        return _result(
            "workspace_config",
            "fail",
            f"workspace.yml has {len(rep['errors'])} validation error(s)",
            remediation=(
                "validate entries against docs/guides/workspace-config.md and the "
                "annotated template in docs/examples/workspace.yml"
            ),
            skill="agent-utilities-deployment",
            data=data,
        )
    detail = f"workspace.yml valid — {rep['repo_count']} repositories"
    if rep["warnings"]:
        nwarn = len(rep["warnings"])
        return _result(
            "workspace_config",
            "warn",
            detail + f", {nwarn} advisory warning(s)",
            remediation="see docs/guides/workspace-config.md for the full schema",
            data=data,
        )
    return _result("workspace_config", "ok", detail, data=data)


def _check_bus() -> dict[str, Any]:
    """Report bus presence, partition-log depth, and unpublished outbox work.

    CONCEPT:AU-ECO.bus.operator-view-agentbus — a growing log or pending send
    outbox means materializers or publishers are not making durable progress.
    """
    try:
        from agent_utilities.core.config import config
        from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
        from agent_utilities.messaging.bus import AgentBus

        engine = IntelligenceGraphEngine.get_active()
        if engine is None:
            return _result("bus", "skip", "no active engine")
        bus = AgentBus.instance(engine)
        st = bus.status()
        backend_stats = bus._log_backend().stats()
        log_depth = bus._depth_from_stats(backend_stats)
        pending_rows = bus._query(
            "MATCH (o:BusOutbox {status: 'pending'}) RETURN count(o) as n", {}
        )
        pending = int(pending_rows[0].get("n", 0)) if pending_rows else 0
        published_rows = bus._query(
            "MATCH (o:BusOutbox {status: 'published'}) RETURN count(o) as n", {}
        )
        published = int(published_rows[0].get("n", 0)) if published_rows else 0
        warning_depth = max(1, int(config.agent_bus_max_depth * 0.8))
    except Exception as exc:  # noqa: BLE001
        return _result("bus", "skip", f"bus probe unavailable ({type(exc).__name__})")

    detail = (
        f"{st['online']}/{st['agents']} participants online, "
        f"{len(st['topics'])} topics, log depth {log_depth}, "
        f"{pending} pending and {published} unmaterialized outbox record(s)"
    )
    data = {
        **st,
        "log_depth": log_depth,
        "pending_outbox": pending,
        "published_outbox": published,
        "log_backend": backend_stats.get("backend", "unknown"),
    }
    if (
        pending >= warning_depth
        or published >= warning_depth
        or log_depth >= warning_depth
    ):
        return _result(
            "bus",
            "warn",
            detail + " — delivery materializers or publishers are falling behind",
            remediation=(
                "check the configured AgentBus log backend and ensure graph_bus "
                "receivers are draining tenant partitions"
            ),
            data=data,
        )
    return _result("bus", "ok", detail, data=data)


def _check_skills() -> dict[str, Any]:
    """Report whether the agent-utilities skill toolkit is installed in the XDG dir.

    CONCEPT:AU-OS.deployment.agent-factory-autoload — the agent factory loads flat
    operator-owned skills plus valid managed subtrees for current providers under
    ``core.paths.skills_dir()``. The ten AU workflow skills unlock the platform. If
    they are absent, point at the one command that installs them. Local discovery
    paths never leave this probe.
    """
    try:
        from agent_utilities.core.providers import (
            _skill_identity,
            resolve_skill_provider_dirs,
        )
        from agent_utilities.skills import BUNDLED_SKILLS

        installed_names = {
            _skill_identity(root) for _provider, root in resolve_skill_provider_dirs()
        }
    except Exception as exc:  # noqa: BLE001
        return _result(
            "skills",
            "fail",
            f"current skill resolution failed ({type(exc).__name__})",
            remediation="reconcile provider registrations and run `agent-utilities install`",
            skill="agent-utilities-deployment",
            data={"ready": False, "redacted": True},
        )

    missing = sorted(set(BUNDLED_SKILLS) - installed_names)
    if missing:
        return _result(
            "skills",
            "warn",
            f"{len(missing)} of {len(BUNDLED_SKILLS)} pre-bundled workflow skills are missing",
            remediation="`agent-utilities install` (installs the ten-skill workflow toolkit)",
            skill="agent-utilities-deployment",
            data={"installed": len(installed_names), "missing": missing},
        )
    return _result(
        "skills",
        "ok",
        f"all {len(BUNDLED_SKILLS)} pre-bundled workflow skills are installed",
        data={"installed": len(installed_names), "required": len(BUNDLED_SKILLS)},
    )


def _check_unified_install() -> dict[str, Any]:
    """Assert the unified XDG tree exists and matches installed providers (CONCEPT:AU-OS.host.doctor-unified-install).

    ``agent-utilities install`` materializes every provider contribution (skills +
    prompts + ontologies, incl. the hub's OWN under ``agent-utilities``) into one XDG
    data tree the runtime reads from. This flags missing current providers, removed
    managed providers, and unmarked nested directories without reporting their local
    filesystem locations.
    """
    try:
        from agent_utilities.core.paths import ontology_dir, skills_dir
        from agent_utilities.core.provider_materialization import (
            ProviderAssetError,
            build_asset_manifest,
            marker_path_exists,
            read_managed_provider_marker,
            resolve_managed_generation,
        )
        from agent_utilities.core.providers import (
            ONTOLOGY_PROVIDER_GROUP,
            PROMPT_PROVIDER_GROUP,
            SKILL_PROVIDER_GROUP,
            provider_registrations,
        )
        from agent_utilities.core.unified_install import (
            OWN_PROVIDER,
            own_provider_asset,
            unified_prompts_dir,
        )
    except Exception as exc:  # noqa: BLE001
        return _result(
            "unified_install",
            "skip",
            f"unified-install probe unavailable ({type(exc).__name__})",
        )

    legs = {
        "skills": (SKILL_PROVIDER_GROUP, skills_dir()),
        "prompts": (PROMPT_PROVIDER_GROUP, unified_prompts_dir()),
        "ontologies": (ONTOLOGY_PROVIDER_GROUP, ontology_dir()),
    }
    expected_counts: dict[str, int] = {}
    missing = 0
    unresolved = 0
    materialized = 0
    stale_managed = 0
    unmanaged_nested = 0
    invalid_managed = 0
    for leg, (group, root) in legs.items():
        try:
            registrations = provider_registrations(group)
        except Exception as exc:  # noqa: BLE001
            return _result(
                "unified_install",
                "fail",
                f"provider registry invalid ({type(exc).__name__})",
                remediation="remove duplicate or invalid provider registrations",
                skill="agent-utilities-deployment",
                data={"ready": False, "redacted": True},
            )
        names = {item.name for item in registrations}
        names.add(OWN_PROVIDER)
        expected_counts[leg] = len(names)
        for registration in registrations:
            if registration.name == OWN_PROVIDER:
                continue
            if registration.source_root is None:
                unresolved += 1
                continue
            try:
                manifest = build_asset_manifest(
                    registration.source_root,
                    leg=leg,
                    allowed_relative_paths=registration.owned_paths,
                )
            except (OSError, ProviderAssetError, ValueError):
                unresolved += 1
                continue
            if (
                resolve_managed_generation(
                    root / registration.name,
                    provider=registration.name,
                    leg=leg,
                    registration=registration.digest,
                    source_manifest=manifest,
                )
                is not None
            ):
                materialized += 1
            else:
                missing += 1
        try:
            _source, own_digest, own_manifest = own_provider_asset(leg)
        except (OSError, ProviderAssetError, ValueError):
            unresolved += 1
        else:
            if (
                resolve_managed_generation(
                    root / OWN_PROVIDER,
                    provider=OWN_PROVIDER,
                    leg=leg,
                    registration=own_digest,
                    source_manifest=own_manifest,
                )
                is not None
            ):
                materialized += 1
            else:
                missing += 1
        if not root.is_dir():
            continue
        for child in root.iterdir():
            if child.name.startswith("."):
                continue
            try:
                child_info = child.lstat()
            except OSError:
                invalid_managed += 1
                continue
            is_junction = getattr(child, "is_junction", lambda: False)()
            if (
                child.is_symlink()
                or is_junction
                or not stat.S_ISDIR(child_info.st_mode)
            ):
                invalid_managed += 1
                continue
            has_marker = marker_path_exists(child)
            if leg == "skills" and not has_marker and (child / "SKILL.md").is_file():
                continue
            marker = read_managed_provider_marker(child, provider=child.name, leg=leg)
            if marker is None:
                if has_marker:
                    invalid_managed += 1
                else:
                    unmanaged_nested += 1
            elif child.name not in names:
                stale_managed += 1

    data = {
        # Readiness is reportable; machine-specific XDG locations are not.  A
        # doctor result can itself be exported as telemetry, so never place a
        # host filesystem reference in its structured payload.
        "roots_ready": {leg: root.is_dir() for leg, (_g, root) in legs.items()},
        "expected_counts": expected_counts,
        "missing": missing,
        "unresolved": unresolved,
        "materialized": materialized,
        "managed_ready": not any(
            (missing, unresolved, stale_managed, unmanaged_nested, invalid_managed)
        ),
        "stale_managed": stale_managed,
        "unmanaged_nested": unmanaged_nested,
        "invalid_managed": invalid_managed,
        "redacted": True,
    }
    if unresolved:
        return _result(
            "unified_install",
            "fail",
            f"current provider sources cannot be validated ({unresolved} issue(s))",
            remediation="repair provider distributions before materialization",
            skill="agent-utilities-deployment",
            data=data,
        )
    if missing or stale_managed or unmanaged_nested or invalid_managed:
        issues = missing + stale_managed + unmanaged_nested + invalid_managed
        return _result(
            "unified_install",
            "warn",
            f"unified provider materialization needs reconciliation ({issues} issue(s))",
            remediation=(
                "`agent-utilities install` (materializes current providers, marks "
                "ownership, and prunes removed managed providers)"
            ),
            skill="agent-utilities-deployment",
            data=data,
        )
    return _result(
        "unified_install",
        "ok",
        f"unified XDG tree complete — {materialized} provider contribution(s) materialized",
        data=data,
    )


def _check_warm_fork() -> dict[str, Any]:
    """Report available confined sandbox rungs and pooled VM parents."""
    rungs: dict[str, dict[str, Any]] = {}
    try:
        from agent_utilities.rlm.sandboxes.registry import default_sandboxes

        for b in default_sandboxes():
            caps = b.capabilities
            try:
                available = bool(b.is_available())
            except Exception:  # noqa: BLE001 - a probe must never crash the doctor
                available = False
            rungs[b.name] = {
                "available": available,
                "isolated": caps.isolated,
                "warm_fork": caps.warm_fork,
                "rank": caps.preference_rank,
            }
    except Exception as exc:  # noqa: BLE001
        return _result(
            "warm_fork",
            "error",
            f"could not enumerate sandbox rungs ({type(exc).__name__})",
        )

    try:
        from agent_utilities.runtime.warm_registry import WarmParentRegistry

        pool = WarmParentRegistry.get().stats()
    except Exception:  # noqa: BLE001
        pool = {}

    warm_rungs = sorted(
        n for n, r in rungs.items() if r["warm_fork"] and r["available"]
    )
    data = {"rungs": rungs, "warm_rungs": warm_rungs, "pool": pool}
    if warm_rungs:
        return _result(
            "warm_fork",
            "ok",
            f"native warm-fork available via: {', '.join(warm_rungs)}",
            data=data,
        )
    return _result(
        "warm_fork",
        "warn",
        "no warm-fork rung available — sandboxes will cold-start every run",
        remediation=(
            "Install the sandbox extra plus a WASI payload, configure an immutable "
            "container image, or connect a governed microVM controller."
        ),
        data=data,
    )


def _check_a2a_persistence() -> dict[str, Any]:
    """Validate the sole current FastA2A durability contract without network I/O."""

    try:
        from epistemic_graph.client import BrokerClient, NodeClient, TxnClient

        from agent_utilities.core.config import AgentConfig
        from agent_utilities.protocols.a2a_epistemic import (
            EpistemicGraphA2ABroker,
            EpistemicGraphA2AStorage,
        )

        cfg = AgentConfig()
    except Exception as exc:  # noqa: BLE001 - doctor reports no configuration values
        return _result(
            "a2a_persistence",
            "fail",
            f"native A2A persistence is unavailable ({type(exc).__name__})",
            remediation=(
                "Install the current agent-utilities and epistemic-graph[full] "
                "artifacts, then repair AgentConfig."
            ),
            data={"ready": False, "redacted": True},
        )

    required_broker_methods = {
        "declare_exchange",
        "declare_queue",
        "bind_queue",
        "publish_idempotent",
        "consume",
        "renew_tag",
        "ack_tag",
        "nack_tag",
    }
    required_node_methods = {
        "create_if_absent",
        "properties",
        "compare_and_set",
        "list_by_label",
    }
    required_txn_methods = {"begin", "cas", "commit", "rollback"}
    missing = sorted(
        f"broker.{name}"
        for name in required_broker_methods
        if not callable(getattr(BrokerClient, name, None))
    )
    missing.extend(
        sorted(
            f"nodes.{name}"
            for name in required_node_methods
            if not callable(getattr(NodeClient, name, None))
        )
    )
    missing.extend(
        sorted(
            f"txn.{name}"
            for name in required_txn_methods
            if not callable(getattr(TxnClient, name, None))
        )
    )
    selected = (
        cfg.a2a_broker == "epistemic_graph" and cfg.a2a_storage == "epistemic_graph"
    )
    bounded = all(
        value > 0
        for value in (
            cfg.a2a_broker_poll_interval_ms,
            cfg.a2a_broker_lease_ms,
            cfg.a2a_broker_prefetch,
            cfg.a2a_broker_message_ttl_ms,
            cfg.a2a_broker_max_delivery_count,
            cfg.a2a_max_payload_bytes,
            cfg.a2a_max_history,
            cfg.a2a_max_artifacts,
            cfg.a2a_max_context_messages,
            cfg.a2a_storage_update_retries,
            cfg.a2a_dispatch_reconcile_interval_ms,
            cfg.a2a_dispatch_reconcile_limit,
            cfg.a2a_cancellation_poll_interval_ms,
        )
    )
    adapters = all(
        value is not None
        for value in (EpistemicGraphA2ABroker, EpistemicGraphA2AStorage)
    )
    data = {
        "native_backend_selected": selected,
        "broker_contract_complete": not missing,
        "bounded_configuration": bounded,
        "adapter_count": 2 if adapters else 0,
        "redacted": True,
    }
    if not selected or missing or not bounded or not adapters:
        return _result(
            "a2a_persistence",
            "fail",
            "native A2A broker/storage contract is incomplete",
            remediation=(
                "Set A2A_BROKER=epistemic_graph and "
                "A2A_STORAGE=epistemic_graph, use positive bounded limits, and "
                "install epistemic-graph[full]."
            ),
            data=data,
        )
    return _result(
        "a2a_persistence",
        "ok",
        "native durable A2A broker/storage and bounded CAS policy are configured",
        data=data,
    )


# Registry: name -> callable. Order is the report order.
CHECKS: dict[str, Callable[..., dict[str, Any]]] = {
    "python_env": _check_python_env,
    "config": _check_config,
    "evolution_staging": _check_evolution_staging,
    "execution_security": _check_execution_security,
    "permission_governance": _check_permission_governance,
    "ontology_release_signing": _check_ontology_release_signing,
    "transport_security": _check_transport_security,
    "google_workspace_oauth": _check_google_workspace_oauth,
    "source_egress": _check_source_egress,
    "eunomia": _check_eunomia,
    "runtime_integrations": _check_runtime_integrations,
    "provider_profiles": _check_provider_profiles,
    "workspace_config": _check_workspace_config,
    "engine_request_context": _check_engine_request_context,
    "engine": _check_engine,
    "graph_authority": _check_graph_authority,
    "graph_connections": _check_graph_connections,
    "ingestion_coverage": _check_ingestion_coverage,
    "connector_coverage": _check_connector_coverage,
    "secrets": _check_secrets,
    "auth": _check_auth,
    "outbound_auth": _check_outbound_auth,
    "skill_certification": _check_skill_certification,
    "production_certification": _check_production_certification,
    "graph_identity": _check_graph_identity,
    "mcp_fleet_secrets": _check_mcp_fleet_secrets,
    "mcp_fleet": _check_mcp_fleet,
    "hooks": _check_hooks,
    "observability": _check_observability,
    "langfuse": _check_langfuse,
    "native_optimizer": _check_native_optimizer,
    "a2a_persistence": _check_a2a_persistence,
    "bus": _check_bus,
    "skills": _check_skills,
    "unified_install": _check_unified_install,
    "warm_fork": _check_warm_fork,
}


def _auto_fix(name: str) -> dict[str, Any]:
    """Run a conservative, idempotent remediation for an auto-fixable check."""
    if name == "hooks":
        try:
            from agent_utilities.ecosystem.hook_installer import HookInstaller

            inst = HookInstaller()
            inst.install()
            return {
                "fixed": name,
                "result": "re-installed hooks",
                "errors": inst.errors,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "fixed": name,
                "error": f"auto-fix failed ({type(exc).__name__})",
            }
    return {"fixed": name, "result": "no auto-fix available"}


def run_doctor(
    only: list[str] | None = None, *, fix: bool = False, live: bool = False
) -> dict[str, Any]:
    """Run the health sweep and return a structured report.

    Args:
        only: restrict to these check names (default: all).
        fix: run conservative auto-remediations for ``auto_fixable`` checks, then
            re-run those checks.
        live: prove network and engine capabilities for live-aware checks.
    """
    if only is None:
        names = list(CHECKS)
    elif (
        not isinstance(only, list)
        or not only
        or len(only) > len(CHECKS)
        or any(not isinstance(name, str) or name not in CHECKS for name in only)
        or len(set(only)) != len(only)
    ):
        result = _result(
            "selection",
            "error",
            "doctor check selection is invalid",
            remediation="select one or more registered doctor checks",
            data={"redacted": True},
        )
        return {
            "status": "unhealthy",
            "counts": {"error": 1},
            "checks": [result],
            "fixes": [],
            "summary": _summarize("unhealthy", [result]),
        }
    else:
        names = only
    # Every runtime entry point consumes the same XDG AgentConfig document.  A
    # doctor launched directly from its console script must do that too; without
    # this load, checks that instantiate ``AgentConfig`` would silently inspect
    # package defaults instead of the deployment GraphOS actually uses.
    try:
        from agent_utilities.core.config import load_config

        load_config()
    except Exception as exc:  # noqa: BLE001 - source details may be sensitive
        load_results = [
            _result(
                name,
                "error",
                f"configuration load failed ({type(exc).__name__})",
                remediation=(
                    "Repair the private AgentConfig source, then rerun the doctor; "
                    "configuration values are intentionally not reported."
                ),
                data={"redacted": True},
            )
            for name in names
        ]
        return {
            "status": "unhealthy",
            "counts": {"error": len(load_results)},
            "checks": load_results,
            "fixes": [],
            "summary": _summarize("unhealthy", load_results),
        }
    results: list[dict[str, Any]] = []
    for name in names:
        fn = CHECKS.get(name)
        if fn is None:
            continue
        try:
            res = (
                fn(live=live)
                if name
                in {"graph_connections", "mcp_fleet", "langfuse", "native_optimizer"}
                else fn()
            )
        except Exception as exc:  # noqa: BLE001 — a check must never crash the doctor
            res = _result(name, "error", f"check raised ({type(exc).__name__})")
        results.append(res)

    fixes: list[dict[str, Any]] = []
    if fix:
        for res in results:
            if res["status"] in ("warn", "fail") and res.get("auto_fixable"):
                fixes.append(_auto_fix(res["name"]))
                try:
                    res.update(CHECKS[res["name"]]())  # re-run after fix
                except Exception:  # noqa: BLE001
                    pass

    worst = max((_RANK[r["status"]] for r in results), default=0)
    overall = {0: "healthy", 1: "warnings", 2: "unhealthy"}[worst]
    counts: dict[str, int] = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    return {
        "status": overall,
        "counts": counts,
        "checks": results,
        "fixes": fixes,
        "summary": _summarize(overall, results),
    }


def _summarize(overall: str, results: list[dict[str, Any]]) -> str:
    bad = [r["name"] for r in results if r["status"] in ("warn", "fail", "error")]
    if overall == "healthy":
        return "All checks passed."
    return f"{overall}: attend to {bad}. Each failing check lists a remediation/skill."


def main(argv: list[str] | None = None) -> int:
    """``agent-utilities-doctor`` console entry."""
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="agent-utilities-doctor",
        description="Holistic health sweep of an agent-utilities deployment.",
    )
    parser.add_argument("--only", nargs="*", choices=list(CHECKS), default=None)
    parser.add_argument(
        "--fix", action="store_true", help="Run safe auto-remediations."
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help=(
            "Prove live graph connections, MCP, Langfuse, and native "
            "ProgramOptimize capabilities."
        ),
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit JSON instead of text."
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Run the host DEPENDENCY preflight (runtimes/tools) instead of the deployment sweep.",
    )
    parser.add_argument(
        "--profile",
        default="tiny",
        help="Deployment profile for --preflight (tiny | single-node-prod | enterprise).",
    )
    parser.add_argument(
        "--component",
        dest="components",
        action="append",
        default=None,
        help="UI component to preflight (repeatable): agent-webui | geniusbot | agent-terminal-ui.",
    )
    args = parser.parse_args(argv)

    if args.preflight:
        from .preflight import run_preflight

        report = run_preflight(args.profile, args.components)
        if args.json:
            print(json.dumps(report, indent=2, default=str))
        else:
            _print_human(report, title="agent-utilities preflight")
        return 0 if report["status"] != "blocked" else 1

    report = run_doctor(args.only, fix=args.fix, live=args.live)
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        _print_human(report)
    return 0 if report["status"] != "unhealthy" else 1


def _print_human(report: dict[str, Any], title: str = "agent-utilities doctor") -> None:
    glyph = {"ok": "✓", "warn": "!", "fail": "✗", "error": "✗", "skip": "·"}
    print(f"{title} — {report['status'].upper()}\n")
    for r in report["checks"]:
        line = f"  {glyph.get(r['status'], '?')} {r['name']:<14} {r['detail']}"
        print(line)
        if r["status"] in ("warn", "fail", "error"):
            if r.get("remediation"):
                print(f"      → fix: {r['remediation']}")
            if r.get("skill"):
                print(f"      → skill: {r['skill']}")
    print(f"\n{report['summary']}")


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(main())
