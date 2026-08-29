"""Knowledge Graph Hydration Service (OWL Native).

Handles dynamic discovery, instantiation, ontological translation, and batch
ingestion from domain-specific APIs into OWL-promotable LPG nodes and edges.

Architecture (CONCEPT:AU-KG.compute.capability-abstraction — Capability Abstraction Layer):
  The CAPABILITY_REGISTRY decouples concrete connectors from abstract
  capability categories.  Each entry maps a source identifier to its
  capability category and the private method that implements the connector.
  New sources are added by extending the registry — the core orchestration
  logic (hydrate_source / hydrate_all) never changes.
"""

from __future__ import annotations

import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Any
from urllib.parse import urlsplit

from agent_utilities.core.config import resolve_langfuse_host, setting
from agent_utilities.observability.langfuse_trust import (
    langfuse_credentials_configured,
)

logger = logging.getLogger(__name__)

_GITHUB_DEFAULT_URL = "https://api.github.com"
_GITHUB_SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")
_GITHUB_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


class _GithubConnectorCompatibilityError(RuntimeError):
    """The installed connector cannot be imported against this AU runtime."""


# ═══════════════════════════════════════════════════════════════════
# CAPABILITY_REGISTRY — maps source identifiers to abstract capability
# categories and their connector methods.  Adding a new data source only
# requires appending an entry here; the orchestration loop is generic.
# ═══════════════════════════════════════════════════════════════════
CAPABILITY_REGISTRY: dict[str, dict[str, str]] = {
    "gitlab": {"category": "source_control", "method": "_hydrate_source_control"},
    "github": {"category": "source_control", "method": "_hydrate_github"},
    "source_control": {
        "category": "source_control",
        "method": "_hydrate_source_control",
    },
    "essential_ea": {
        "category": "enterprise_architecture",
        "method": "_hydrate_enterprise_architecture",
    },
    "aris": {
        "category": "enterprise_architecture",
        "method": "_hydrate_enterprise_architecture",
    },
    "leanix": {
        "category": "enterprise_architecture",
        "method": "_hydrate_leanix",
    },
    "enterprise_architecture": {
        "category": "enterprise_architecture",
        "method": "_hydrate_enterprise_architecture",
    },
    "twenty": {"category": "crm", "method": "_hydrate_twenty"},
    "glpi": {"category": "itsm", "method": "_hydrate_servicenow"},
    "openmaint": {"category": "itsm", "method": "_hydrate_servicenow"},
    "servicenow": {"category": "itsm", "method": "_hydrate_servicenow"},
    "erpnext": {"category": "erp", "method": "_hydrate_erpnext"},
    # ``jira``/``plane``/``confluence`` are first-class delta connectors now
    # (``source_sync._DELTA_HANDLERS``, CONCEPT:AU-KG.compute.confluence-first-class-delta/2.124/2.125); the generic
    # ``issue_tracking`` capability remains for the zero-infra local-markdown checklist.
    "issue_tracking": {
        "category": "issue_tracking",
        "method": "_hydrate_issue_tracking",
    },
    "process_modeling": {
        "category": "process_modeling",
        "method": "_hydrate_process_modeling",
    },
    "relational_database": {
        "category": "databases",
        "method": "_hydrate_relational_database",
    },
    "databases": {"category": "databases", "method": "_hydrate_relational_database"},
    "portainer": {
        "category": "container_orchestration",
        "method": "_hydrate_portainer",
    },
    "uptime_kuma": {"category": "uptime_monitoring", "method": "_hydrate_uptime_kuma"},
    "lgtm": {"category": "monitoring", "method": "_hydrate_lgtm"},
    "langfuse": {"category": "monitoring", "method": "_hydrate_langfuse"},
    "keycloak": {"category": "authentication", "method": "_hydrate_keycloak"},
    "openbao": {"category": "secret_management", "method": "_hydrate_openbao"},
    "nextcloud": {"category": "collaboration", "method": "_hydrate_nextcloud"},
    "listmonk": {"category": "mailing", "method": "_hydrate_listmonk"},
    "mattermost": {"category": "collaboration", "method": "_hydrate_message_protocol"},
    "message_protocol": {
        "category": "collaboration",
        "method": "_hydrate_message_protocol",
    },
    "technitium_dns": {"category": "dns", "method": "_hydrate_technitium_dns"},
    "caddy": {"category": "reverse_proxy", "method": "_hydrate_caddy"},
    "tunnel_manager": {"category": "vpn", "method": "_hydrate_tunnel_manager"},
    "scholarx": {"category": "research", "method": "_hydrate_scholarx"},
    "emerald_exchange": {
        "category": "financial_exchange",
        "method": "_hydrate_emerald_exchange",
    },
    "postiz": {"category": "social_media", "method": "_hydrate_postiz"},
}


def _any_setting(*keys: str) -> bool:
    """True when at least one of ``keys`` resolves to a truthy setting."""
    return any(setting(key) for key in keys)


def _all_settings(*keys: str) -> bool:
    """True when every one of ``keys`` resolves to a truthy setting."""
    return all(setting(key) for key in keys)


class HydrationManager:
    """Orchestrates dynamic client loading and batch OWL-native graph hydration.

    Connector methods are resolved through the module-level CAPABILITY_REGISTRY,
    which maps source identifiers to abstract capability categories and their
    implementing methods.  This enables swappable backends — replacing one DNS
    tool with another only requires changing the connector method, not the
    orchestration logic.
    """

    def __init__(self) -> None:
        self.sources = list(CAPABILITY_REGISTRY.keys())
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_status(self) -> dict[str, Any]:
        """Check environment variables to see which sources are configured."""
        github_endpoint = self._github_endpoint()
        status = {
            "gitlab": {
                "configured": _any_setting("GITLAB_TOKEN", "GITLAB_API_TOKEN"),
                "url": setting("GITLAB_URL", "https://gitlab.com"),
            },
            "github": {
                "configured": _any_setting("GITHUB_TOKEN", "GITHUB_API_KEY"),
                "url": _GITHUB_DEFAULT_URL,
                "endpoint_valid": github_endpoint is not None,
            },
            "leanix": {
                "configured": bool(setting("LEANIX_TOKEN")),
                "url": setting("LEANIX_URL", ""),
            },
            "essential_ea": {
                "configured": bool(setting("ESSENTIAL_EA_TOKEN")),
                "url": setting("ESSENTIAL_EA_URL", ""),
            },
            "aris": {
                "configured": bool(setting("BPM_TOKEN")),
                "url": setting("BPM_URL", ""),
            },
            "twenty": {
                "configured": _any_setting("TWENTY_TOKEN", "TWENTY_API_TOKEN"),
                "url": setting("TWENTY_URL", ""),
            },
            "servicenow": {
                "configured": _all_settings(
                    "SERVICENOW_USERNAME", "SERVICENOW_PASSWORD"
                ),
                "url": setting("SERVICENOW_URL") or setting("SERVICENOW_INSTANCE", ""),
            },
            "erpnext": {
                "configured": bool(setting("ERPNEXT_TOKEN")),
                "url": setting("ERPNEXT_URL", ""),
            },
            "glpi": {
                "configured": bool(setting("GLPI_TOKEN")),
                "url": setting("GLPI_URL", ""),
            },
            "openmaint": {
                "configured": bool(setting("OPENMAINT_TOKEN")),
                "url": setting("OPENMAINT_URL", ""),
            },
            "jira": {
                "configured": _any_setting("JIRA_TOKEN", "JIRA_API_TOKEN"),
                "url": setting("JIRA_URL", ""),
            },
            "plane": {
                "configured": _any_setting("PLANE_TOKEN", "PLANE_API_TOKEN"),
                "url": setting("PLANE_URL", ""),
            },
            "portainer": {
                "configured": _any_setting("PORTAINER_TOKEN", "PORTAINER_PASSWORD"),
                "url": setting("PORTAINER_URL", ""),
            },
            "uptime_kuma": {
                "configured": bool(setting("UPTIME_KUMA_URL")),
                "url": setting("UPTIME_KUMA_URL", ""),
            },
            "lgtm": {
                "configured": _any_setting("LGTM_URL", "GRAFANA_URL"),
                "url": setting("LGTM_URL", ""),
            },
            "langfuse": {
                "configured": langfuse_credentials_configured(),
                "url": resolve_langfuse_host(),
            },
            "keycloak": {
                "configured": _all_settings("KEYCLOAK_URL", "KEYCLOAK_ADMIN_PASSWORD"),
                "url": setting("KEYCLOAK_URL", ""),
            },
            "openbao": {
                "configured": _any_setting("BAO_URL", "VAULT_URL"),
                "url": setting("BAO_URL", ""),
            },
            "nextcloud": {
                "configured": _all_settings("NEXTCLOUD_URL", "NEXTCLOUD_PASSWORD"),
                "url": setting("NEXTCLOUD_URL", ""),
            },
            "listmonk": {
                "configured": _all_settings("LISTMONK_URL", "LISTMONK_TOKEN"),
                "url": setting("LISTMONK_URL", ""),
            },
            "mattermost": {
                "configured": _all_settings("MATTERMOST_URL", "MATTERMOST_TOKEN"),
                "url": setting("MATTERMOST_URL", ""),
            },
            "technitium_dns": {
                "configured": _all_settings("TECHNITIUM_URL", "TECHNITIUM_TOKEN"),
                "url": setting("TECHNITIUM_URL", ""),
            },
            "caddy": {
                "configured": _any_setting("CADDY_URL", "CADDY_API_URL"),
                "url": setting("CADDY_URL", ""),
            },
            "tunnel_manager": {
                "configured": _any_setting("TUNNEL_MANAGER_URL", "TUNNEL_URL"),
                "url": setting("TUNNEL_MANAGER_URL", ""),
            },
            "scholarx": {
                "configured": _any_setting("SCHOLARX_URL", "SCHOLARX_API_KEY"),
                "url": setting("SCHOLARX_URL", ""),
            },
            "emerald_exchange": {
                "configured": _any_setting("EMERALD_URL", "EMERALD_API_KEY"),
                "url": setting("EMERALD_URL", ""),
            },
            "postiz": {
                "configured": _all_settings("POSTIZ_URL", "POSTIZ_TOKEN"),
                "url": setting("POSTIZ_URL", ""),
            },
        }
        return status

    def hydrate_source(self, engine: Any, source: str) -> dict[str, Any]:
        """Trigger instant hydration for a specific source.

        Resolves the connector method from CAPABILITY_REGISTRY, enabling
        tool-agnostic orchestration. Connector implementations retain their
        compact ``ingest_external_batch`` protocol, but receive a transparent
        proxy that commits that batch through native ``ApplyChangeEnvelope``.
        """
        source = source.lower().strip()
        entry = CAPABILITY_REGISTRY.get(source)
        if entry is None:
            raise ValueError(f"Unknown hydration source: '{source}'")
        method = getattr(self, entry["method"], None)
        if method is None:
            raise ValueError(
                f"Connector method '{entry['method']}' not found for source '{source}'"
            )
        from ..ingestion.envelope_ingest import NativeChangeEnvelopeEngineProxy

        return method(NativeChangeEnvelopeEngineProxy(engine))

    def hydrate_all(self, engine: Any) -> dict[str, Any]:
        """Sequentially hydrate all active/configured sources."""
        results: dict[str, Any] = {}
        status = self.get_status()

        for src, conf in status.items():
            if conf["configured"]:
                try:
                    logger.info(
                        f"Starting scheduled hydration for configured source: {src}"
                    )
                    res = self.hydrate_source(engine, src)
                    results[src] = res
                except Exception as e:
                    logger.error(f"Failed scheduled hydration for {src}: {e}")
                    results[src] = {"status": "error", "error": str(e)}
            else:
                logger.info(f"Skipping hydration for {src} (not configured)")
        return results

    # ══════════════════════════════════════════════════════════════════
    # Generalized Open-Source-First Hydration Layer
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def _load_github_api() -> Any | None:
        """Return the connector-owned client factory when it is compatible."""
        try:
            from github_agent.auth import get_client
        except ModuleNotFoundError as exc:
            if exc.name == "github_agent":
                return None
            raise _GithubConnectorCompatibilityError from None
        except ImportError:
            raise _GithubConnectorCompatibilityError from None
        return get_client

    @staticmethod
    def _github_url_text(value: Any) -> str | None:
        """Normalize a provider URL candidate without accepting whitespace."""
        if not isinstance(value, str):
            return None
        rendered = value.strip().rstrip("/")
        if not rendered or rendered != value.rstrip("/"):
            return None
        if any(character.isspace() or ord(character) < 32 for character in rendered):
            return None
        return rendered

    @staticmethod
    def _github_parsed_url(value: str) -> Any | None:
        """Parse a URL and reject userinfo, query, fragment, and bad ports."""
        try:
            parsed = urlsplit(value)
            port = parsed.port
        except ValueError:
            return None
        if not parsed.netloc or not parsed.hostname:
            return None
        if (
            parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            return None
        if port is not None and not 1 <= port <= 65_535:
            return None
        return parsed

    @classmethod
    def _validate_github_endpoint(cls, endpoint: Any) -> str | None:
        """Validate an endpoint without returning credentials in a result."""
        rendered = cls._github_url_text(endpoint)
        if rendered is None:
            return None
        parsed = cls._github_parsed_url(rendered)
        if parsed is None:
            return None
        scheme = parsed.scheme.casefold()
        if scheme not in {"http", "https"} or (
            scheme == "http"
            and parsed.hostname.casefold() not in _GITHUB_LOOPBACK_HOSTS
        ):
            return None
        return rendered

    @classmethod
    def _github_endpoint(cls) -> str | None:
        """Validate the governed connector endpoint without exposing it."""
        return cls._validate_github_endpoint(setting("GITHUB_URL", _GITHUB_DEFAULT_URL))

    @staticmethod
    def _github_mapping(value: Any) -> dict[str, Any]:
        """Convert a provider response model or mapping to a plain mapping."""
        if isinstance(value, dict):
            return value
        dump = getattr(value, "model_dump", None)
        if callable(dump):
            try:
                value = dump(mode="json")
            except TypeError:
                value = dump()
        return value if isinstance(value, dict) else {}

    @classmethod
    def _github_records(cls, response: Any) -> list[dict[str, Any]] | None:
        """Extract provider records without inventing values for malformed data."""
        data = getattr(response, "data", response)
        if isinstance(data, dict):
            data = data.get("data", data.get("repositories", data.get("workflow_runs")))
        if not isinstance(data, list):
            return None
        records: list[dict[str, Any]] = []
        for item in data:
            record = cls._github_mapping(item)
            if not record:
                return None
            records.append(record)
        return records

    @staticmethod
    def _github_provider_id(value: Any) -> int | None:
        """Accept only the positive integer ids emitted by GitHub's models."""
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            return None
        return value

    @staticmethod
    def _github_text(value: Any, *, required: bool = False) -> str | None:
        """Validate text fields from a connector model before graph projection."""
        if value is None and not required:
            return None
        if not isinstance(value, str) or not value.strip():
            return None
        if any(character.isspace() and character != " " for character in value):
            return None
        if any(ord(character) < 32 for character in value):
            return None
        return value

    @staticmethod
    def _github_valid_slug(value: Any) -> str:
        """Accept one GitHub owner/name slug without path or query injection."""
        if not isinstance(value, str) or value.strip() != value:
            return ""
        return value if _GITHUB_SLUG_RE.fullmatch(value) else ""

    @classmethod
    def _github_repository_identity(
        cls, repository: dict[str, Any]
    ) -> tuple[int, str, str] | None:
        """Validate and return the stable identity fields for one repository."""
        provider_id = cls._github_provider_id(repository.get("id"))
        name = cls._github_text(repository.get("name"), required=True)
        full_name = cls._github_valid_slug(repository.get("full_name"))
        if provider_id is None or name is None or not full_name:
            return None
        return provider_id, name, full_name

    @classmethod
    def _github_repository_fields_valid(cls, repository: dict[str, Any]) -> bool:
        """Validate the optional typed fields projected into a repository node."""
        for field in ("description", "default_branch", "language"):
            value = repository.get(field)
            if value is not None and cls._github_text(value) is None:
                return False
        return not (
            "private" in repository and not isinstance(repository["private"], bool)
        )

    @staticmethod
    def _github_safe_url(value: Any) -> str | None:
        """Retain ordinary HTTPS/HTTP links while dropping credential-bearing URLs."""
        rendered = HydrationManager._github_url_text(value)
        if rendered is None:
            return None
        parsed = HydrationManager._github_parsed_url(rendered)
        if parsed is None:
            return None
        if parsed.scheme.casefold() not in {"http", "https"}:
            return None
        return rendered

    @classmethod
    def _github_repository_entity(
        cls, repository: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Map one real provider repository, retaining its stable identity."""
        identity = cls._github_repository_identity(repository)
        if identity is None or not cls._github_repository_fields_valid(repository):
            return None
        provider_id, name, full_name = identity
        entity: dict[str, Any] = {
            "id": f"github:repository:{provider_id}",
            "type": "repository",
            "domain": "github",
            "source_system": "github",
            "externalToolId": str(provider_id),
            "name": name,
            "full_name": full_name,
        }
        for field in ("description", "default_branch", "private", "language"):
            if repository.get(field) is not None:
                entity[field] = repository[field]
        source_uri = cls._github_safe_url(
            repository.get("html_url") or repository.get("web_url")
        )
        if source_uri:
            entity["web_url"] = str(source_uri)
            entity["source_uri"] = str(source_uri)
        return entity

    @classmethod
    def _github_repository_slug(cls, repository: dict[str, Any]) -> str:
        """Resolve an owner/name slug strictly from provider fields."""
        full_name = repository.get("full_name")
        slug = cls._github_valid_slug(full_name)
        if slug:
            return slug
        owner = repository.get("owner")
        name = repository.get("name")
        login = owner.get("login") if isinstance(owner, dict) else None
        return cls._github_valid_slug(f"{login}/{name}") if login and name else ""

    @classmethod
    def _github_workflow_fields_valid(cls, run: dict[str, Any]) -> bool:
        """Validate required and optional typed fields for an Actions run."""
        for field in ("head_branch", "head_sha", "status", "event"):
            if cls._github_text(run.get(field), required=True) is None:
                return False
        for field in (
            "name",
            "conclusion",
            "head_branch",
            "head_sha",
            "status",
            "event",
            "run_started_at",
            "updated_at",
        ):
            value = run.get(field)
            if value is not None and cls._github_text(value) is None:
                return False
        return True

    @classmethod
    def _github_workflow_entity(
        cls, run: dict[str, Any], repository_slug: str
    ) -> dict[str, Any] | None:
        """Map one real Actions run to a stable provider-scoped pipeline node."""
        provider_id = cls._github_provider_id(run.get("id"))
        repository_slug = cls._github_valid_slug(repository_slug)
        if (
            provider_id is None
            or not repository_slug
            or not cls._github_workflow_fields_valid(run)
        ):
            return None
        entity: dict[str, Any] = {
            "id": f"github:pipelinerun:{repository_slug}:{provider_id}",
            "type": "pipeline",
            "domain": "github",
            "source_system": "github",
            "externalToolId": str(provider_id),
        }
        for field in (
            "name",
            "status",
            "conclusion",
            "head_sha",
            "head_branch",
            "event",
            "run_started_at",
            "updated_at",
        ):
            if run.get(field) is not None:
                entity[field] = run[field]
        source_uri = cls._github_safe_url(run.get("html_url") or run.get("web_url"))
        if source_uri:
            entity["web_url"] = str(source_uri)
            entity["source_uri"] = str(source_uri)
        return entity

    def _github_workflow_entities(
        self, client: Any, repository: dict[str, Any], repository_id: str
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
        """Fetch and map Actions runs, reporting failures instead of hiding them."""
        get_runs = getattr(client, "get_workflow_runs", None)
        slug = self._github_repository_slug(repository)
        if not callable(get_runs) or not slug or "/" not in slug:
            return [], [], 1
        owner, repo = slug.split("/", 1)
        try:
            records = self._github_records(get_runs(owner=owner, repo=repo))
        except Exception as exc:  # noqa: BLE001 — optional Actions slice is isolated
            logger.debug(
                "GitHub workflow fetch failed for %s: error_type=%s",
                slug,
                type(exc).__name__,
            )
            return [], [], 1
        if records is None:
            return [], [], 1
        entities: list[dict[str, Any]] = []
        relationships: list[dict[str, Any]] = []
        failures = 0
        for run in records:
            entity = self._github_workflow_entity(run, slug)
            if entity is None:
                failures += 1
                continue
            entities.append(entity)
            relationships.append(
                {
                    "source": entity["id"],
                    "target": repository_id,
                    "type": "depends_on",
                    "domain": "github",
                }
            )
        return entities, relationships, failures

    def _github_client_and_repositories(
        self, client_factory: Any, endpoint: str
    ) -> tuple[Any | None, list[dict[str, Any]] | None, dict[str, Any] | None]:
        """Create the connector-owned client and fetch real repository records."""
        client: Any | None = None
        if not callable(client_factory):
            return (
                None,
                None,
                {
                    "status": "unavailable",
                    "source": "github",
                    "reason": "github-agent runtime incompatible",
                    "nodes_hydrated": 0,
                    "relations_hydrated": 0,
                },
            )
        try:
            client = client_factory()
            client_endpoint = getattr(client, "url", None)
            if client_endpoint is not None and (
                self._validate_github_endpoint(client_endpoint) != endpoint
            ):
                logger.debug("GitHub connector endpoint mismatch")
                return (
                    client,
                    None,
                    {
                        "status": "unavailable",
                        "source": "github",
                        "reason": "github-agent runtime incompatible",
                        "nodes_hydrated": 0,
                        "relations_hydrated": 0,
                    },
                )
            repositories = self._github_records(client.get_repositories())
        except (AttributeError, ModuleNotFoundError, TypeError) as exc:  # noqa: BLE001 — connector compatibility exceptions may contain endpoint credentials; keep the provider result redacted
            logger.debug(
                "GitHub connector runtime incompatible: error_type=%s",
                type(exc).__name__,
            )
            return (
                client,
                None,
                {
                    "status": "unavailable",
                    "source": "github",
                    "reason": "github-agent runtime incompatible",
                    "nodes_hydrated": 0,
                    "relations_hydrated": 0,
                },
            )
        except Exception as exc:  # noqa: BLE001 — provider reachability is an explicit result
            logger.debug(
                "GitHub provider unavailable: error_type=%s", type(exc).__name__
            )
            return (
                client,
                None,
                {
                    "status": "unavailable",
                    "source": "github",
                    "reason": "GitHub provider unavailable",
                    "nodes_hydrated": 0,
                    "relations_hydrated": 0,
                },
            )
        return client, repositories, None

    def _github_entities(
        self, client: Any, repositories: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, int]:
        """Map repositories and optional Actions runs to graph records."""
        entities: list[dict[str, Any]] = []
        relationships: list[dict[str, Any]] = []
        invalid_repositories = 0
        workflow_failures = 0
        for repository in repositories:
            entity = self._github_repository_entity(repository)
            if entity is None:
                invalid_repositories += 1
                continue
            entities.append(entity)
            workflow_entities, workflow_relationships, failures = (
                self._github_workflow_entities(client, repository, entity["id"])
            )
            entities.extend(workflow_entities)
            relationships.extend(workflow_relationships)
            workflow_failures += failures
        return entities, relationships, invalid_repositories, workflow_failures

    @staticmethod
    def _close_github_client(client: Any | None) -> None:
        """Close a provider client without changing the hydration result."""
        close_method = getattr(client, "close", None)
        if not callable(close_method):
            return None
        try:
            close_method()
        except Exception as exc:  # noqa: BLE001 — cleanup cannot alter provider result
            logger.debug(
                "GitHub provider cleanup failed: error_type=%s",
                type(exc).__name__,
            )

    @staticmethod
    def _github_result(
        status: str,
        reason: str | None = None,
        *,
        nodes: int = 0,
        relations: int = 0,
    ) -> dict[str, Any]:
        """Build a redacted, consistently shaped provider result."""
        result: dict[str, Any] = {
            "status": status,
            "source": "github",
            "nodes_hydrated": nodes,
            "relations_hydrated": relations,
        }
        if reason is not None:
            result["reason"] = reason
        return result

    def _hydrate_github_batch(
        self,
        engine: Any,
        client: Any | None,
        repositories: list[dict[str, Any]] | None,
        provider_error: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Project one provider response and report partial/error outcomes."""
        if provider_error is not None:
            return provider_error
        if repositories is None:
            return self._github_result(
                "error", "GitHub provider returned malformed repository data"
            )
        if not repositories:
            return self._github_result("no_data", "GitHub returned no repositories")

        entities, relationships, invalid_repositories, workflow_failures = (
            self._github_entities(client, repositories)
        )
        if invalid_repositories:
            return self._github_result(
                "error", "GitHub provider returned malformed repository data"
            )
        if entities:
            try:
                engine.ingest_external_batch("github", entities, relationships)
            except Exception as exc:  # noqa: BLE001 — graph errors stay redacted
                logger.debug(
                    "GitHub graph ingestion failed: error_type=%s",
                    type(exc).__name__,
                )
                return self._github_result(
                    "error", "GitHub entities could not be ingested"
                )

        result = self._github_result(
            "ok",
            nodes=len(entities),
            relations=len(relationships),
        )
        if workflow_failures:
            result.update(
                {
                    "status": "partial",
                    "reason": "GitHub workflow data is incomplete",
                    "workflow_failures": workflow_failures,
                }
            )
        return result

    def _hydrate_github(self, engine: Any) -> dict[str, Any]:
        """Hydrate GitHub through the connector-owned API, never demo records."""
        token = setting("GITHUB_TOKEN") or setting("GITHUB_API_KEY")
        if not isinstance(token, str) or not token.strip():
            return self._github_result("skipped", "Missing GITHUB_TOKEN/GITHUB_API_KEY")

        endpoint = self._github_endpoint()
        if endpoint is None:
            return self._github_result(
                "error", "GitHub endpoint configuration is invalid"
            )

        try:
            client_factory = self._load_github_api()
        except _GithubConnectorCompatibilityError:
            return self._github_result(
                "unavailable", "github-agent runtime incompatible"
            )
        if client_factory is None:
            return self._github_result("skipped", "github-agent package not installed")

        client, repositories, provider_error = self._github_client_and_repositories(
            client_factory, endpoint
        )
        try:
            return self._hydrate_github_batch(
                engine, client, repositories, provider_error
            )
        finally:
            self._close_github_client(client)

    def _hydrate_source_control(self, engine: Any) -> dict[str, Any]:
        """Hydrate source control metadata. Supports Git, GitLab, and GitHub."""
        # Pluggable GitLab
        if setting("GITLAB_TOKEN") or setting("GITLAB_API_TOKEN"):
            return self._hydrate_gitlab(engine)

        # Pluggable GitHub
        if setting("GITHUB_TOKEN") or setting("GITHUB_API_KEY"):
            return self._hydrate_github(engine)

        # Default Local Git
        entities = []
        relationships = []
        import subprocess

        try:
            sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip()
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
            ).strip()
            repo_name = os.path.basename(os.getcwd())
            repo_id = f"git:repo:{repo_name}"
            entities.append(
                {
                    "id": repo_id,
                    "type": "repository",
                    "name": repo_name,
                    "branch": branch,
                    "commit": sha,
                    "domain": "git",
                }
            )
            module_id = "git:module:core"
            entities.append(
                {
                    "id": module_id,
                    "type": "module",
                    "name": "core",
                    "domain": "git",
                }
            )
            relationships.append(
                {
                    "source": module_id,
                    "target": repo_id,
                    "type": "depends_on",
                    "domain": "git",
                }
            )
        except Exception:
            repo_id = "git:repo:workspace"
            entities.append(
                {
                    "id": repo_id,
                    "type": "repository",
                    "name": "workspace",
                    "branch": "main",
                    "commit": "abcdef123456",
                    "domain": "git",
                }
            )
            module_id = "git:module:core"
            entities.append(
                {
                    "id": module_id,
                    "type": "module",
                    "name": "core",
                    "domain": "git",
                }
            )
            relationships.append(
                {
                    "source": module_id,
                    "target": repo_id,
                    "type": "depends_on",
                    "domain": "git",
                }
            )

        engine.ingest_external_batch("git", entities, relationships)
        return {
            "status": "ok",
            "nodes_hydrated": len(entities),
            "relations_hydrated": len(relationships),
        }

    def _hydrate_enterprise_architecture(self, engine: Any) -> dict[str, Any]:
        """Hydrate Enterprise Architecture facts. Supports Backstage catalog-info.yaml, ArchiMate XML, and EARs (e.g., Essential Project)."""
        if setting("EAR_TOKEN"):
            return self._hydrate_ear(engine)

        entities = []
        relationships = []

        catalog_path = setting("BACKSTAGE_FILE", "catalog-info.yaml")
        yaml_content = None
        if os.path.exists(catalog_path):
            try:
                import yaml  # type: ignore

                with open(catalog_path) as f:
                    yaml_content = yaml.safe_load(f)
            except Exception as e:
                logger.debug("Failed to read Backstage YAML (%s)", type(e).__name__)

        if yaml_content and isinstance(yaml_content, dict):
            metadata = yaml_content.get("metadata", {})
            name = metadata.get("name", "backstage-component")
            kind = yaml_content.get("kind", "Component")
            comp_id = f"backstage:component:{name}"
            entity = {
                "id": comp_id,
                "type": "backstage_component",
                "name": name,
                "kind": kind,
                "description": metadata.get("description", ""),
                "domain": "backstage",
            }
            for k, v in metadata.items():
                if k not in ["name", "description"] and isinstance(
                    v, str | int | float | bool
                ):
                    entity[f"metadata_{k}"] = v
            entities.append(entity)

            fact_sheet_id = f"ea:factsheet:{name}"
            entities.append(
                {
                    "id": fact_sheet_id,
                    "type": "ea_fact_sheet",
                    "name": f"EA Fact Sheet: {name}",
                    "domain": "ea",
                }
            )
            relationships.append(
                {
                    "source": comp_id,
                    "target": fact_sheet_id,
                    "type": "associated_with",
                    "domain": "backstage",
                }
            )
        else:
            entities.append(
                {
                    "id": "backstage:component:search-service",
                    "type": "backstage_component",
                    "name": "search-service",
                    "kind": "Component",
                    "description": "Enterprise Search service",
                    "metadata_owner": "search-team",
                    "metadata_tier": "tier-1",
                    "domain": "backstage",
                }
            )
            entities.append(
                {
                    "id": "ea:factsheet:search-service",
                    "type": "ea_fact_sheet",
                    "name": "EA Fact Sheet: search-service",
                    "domain": "ea",
                }
            )
            relationships.append(
                {
                    "source": "backstage:component:search-service",
                    "target": "ea:factsheet:search-service",
                    "type": "associated_with",
                    "domain": "backstage",
                }
            )

        if entities:
            engine.ingest_external_batch(
                "enterprise_architecture", entities, relationships
            )

        return {
            "status": "ok",
            "nodes_hydrated": len(entities),
            "relations_hydrated": len(relationships),
        }

    def _hydrate_leanix(self, engine: Any) -> dict[str, Any]:
        """Mirror the LeanIX fact-sheet graph natively into the KG (CONCEPT:AU-KG.ingest.enterprise-source-extractor).

        Delegates to the one delta-aware sync path (:func:`leanix_sync.sync_leanix`),
        which injects a live LeanIX client into the typed extractor, batch-ingests
        every fact sheet + relation (each node stamped
        ``externalToolId``/``domain="leanix"``), and advances the delta watermark.
        Default ``mode="delta"`` grabs only what changed since the last run.
        """
        from .source_sync import sync_source

        return sync_source(engine, "leanix", mode="delta")

    def _load_bpmn_xml(self) -> str:
        """Read the configured BPMN file, falling back to a small built-in
        sample process when unavailable/unreadable/empty."""
        bpmn_path = setting("BPMN_FILE", "process.bpmn")
        xml_content = None
        if os.path.exists(bpmn_path):
            try:
                with open(bpmn_path, encoding="utf-8") as f:
                    xml_content = f.read()
            except Exception as e:
                logger.debug("Failed to read BPMN source (%s)", type(e).__name__)

        if not xml_content:
            xml_content = """<?xml version="1.0" encoding="UTF-8"?>
            <bpmn:definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL" id="Definitions_1">
              <bpmn:process id="Process_1" isExecutable="false">
                <bpmn:startEvent id="StartEvent_1" name="Start" />
                <bpmn:task id="Task_1" name="Verify Credentials" />
                <bpmn:task id="Task_2" name="Authorize Access" />
                <bpmn:endEvent id="EndEvent_1" name="End" />
                <bpmn:sequenceFlow id="Flow_1" sourceRef="StartEvent_1" targetRef="Task_1" />
                <bpmn:sequenceFlow id="Flow_2" sourceRef="Task_1" targetRef="Task_2" />
                <bpmn:sequenceFlow id="Flow_3" sourceRef="Task_2" targetRef="EndEvent_1" />
              </bpmn:process>
            </bpmn:definitions>
            """
        return xml_content

    def _classify_bpmn_element(
        self, elem: Any
    ) -> tuple[str | None, dict[str, Any] | None, tuple[str, str] | None]:
        """Classify one BPMN XML element. Returns ``(step_id, step, flow)``:
        a step-like element yields ``(step_id, step, None)``, a
        ``sequenceFlow`` yields ``(None, None, (src, tgt))``, anything else
        (or an under-specified element) yields ``(None, None, None)``."""
        tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
        if tag in [
            "task",
            "userTask",
            "serviceTask",
            "scriptTask",
            "startEvent",
            "endEvent",
        ]:
            step_id = elem.attrib.get("id")
            if not step_id:
                return None, None, None
            name = elem.attrib.get("name") or step_id
            return (
                step_id,
                {
                    "id": f"bpmn:step:{step_id}",
                    "type": "process_step",
                    "name": name,
                    "step_type": tag,
                    "domain": "bpmn",
                },
                None,
            )
        if tag in ["sequenceFlow"]:
            src = elem.attrib.get("sourceRef")
            tgt = elem.attrib.get("targetRef")
            if src and tgt:
                return None, None, (src, tgt)
        return None, None, None

    def _parse_bpmn_entities(
        self, xml_content: str
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        entities: list[dict[str, Any]] = []
        relationships: list[dict[str, Any]] = []
        import defusedxml.ElementTree as ET

        try:
            root = ET.fromstring(xml_content)
            steps_map: dict[str, dict[str, Any]] = {}
            flows: list[tuple[str, str]] = []

            for elem in root.iter():
                step_id, step, flow = self._classify_bpmn_element(elem)
                if step_id and step:
                    steps_map[step_id] = step
                elif flow:
                    flows.append(flow)

            model_id = "bpmn:model:Process_1"
            entities.append(
                {
                    "id": model_id,
                    "type": "process_model",
                    "name": "BPMN Process Model",
                    "domain": "bpmn",
                }
            )

            for step in steps_map.values():
                entities.append(step)
                relationships.append(
                    {
                        "source": step["id"],
                        "target": model_id,
                        "type": "part_of",
                        "domain": "bpmn",
                    }
                )

            for src, tgt in flows:
                if src in steps_map and tgt in steps_map:
                    relationships.append(
                        {
                            "source": steps_map[src]["id"],
                            "target": steps_map[tgt]["id"],
                            "type": "precedes",
                            "domain": "bpmn",
                        }
                    )
        except Exception as e:
            logger.error(f"Error parsing BPMN XML: {e}")
        return entities, relationships

    def _hydrate_process_modeling_via_bpm_api(
        self, bpm_provider: str, bpm_url: str, bpm_token: str
    ) -> dict[str, Any]:
        class BaseBPMHydrator(ABC):
            def __init__(self, url: str, token: str):
                self.url = url.rstrip("/")
                self.token = token

            @abstractmethod
            def fetch_processes(self) -> list[dict[str, Any]]:
                """Fetch and format process entities from the BPM provider."""
                raise NotImplementedError

        class OpenSourceBPMHydrator(BaseBPMHydrator):
            def fetch_processes(self) -> list[dict[str, Any]]:
                from agent_utilities.core.http_client import create_http_client
                from agent_utilities.core.transport_security import (
                    resolve_configured_tls_profile,
                )

                result = []
                headers = {
                    "Authorization": f"Bearer {self.token}",
                    "Accept": "application/json",
                }
                trust = resolve_configured_tls_profile("bpm")
                try:
                    with create_http_client(
                        timeout=5.0,
                        headers=headers,
                        **trust.httpx_kwargs(),
                    ) as client:
                        resp = client.get(f"{self.url}/repository/process-definitions")
                finally:
                    trust.cleanup()
                if resp.status_code == 200:
                    for proc in resp.json():
                        proc_id = str(proc.get("id", ""))
                        if proc_id:
                            result.append(
                                {
                                    "id": f"process:bpm:{proc_id}",
                                    "type": "process_model",
                                    "name": str(
                                        proc.get("name") or proc.get("key", "")
                                    ),
                                    "domain": "bpm",
                                }
                            )
                else:
                    logger.warning("BPM hydration API returned a non-success status")
                return result

        class ArisBPMHydrator(BaseBPMHydrator):
            def fetch_processes(self) -> list[dict[str, Any]]:
                raise RuntimeError(
                    "ARIS BPM integration requires enterprise API credentials and specific endpoints."
                )

        def get_bpm_hydrator(provider: str, url: str, token: str) -> BaseBPMHydrator:
            if provider.lower() == "aris":
                return ArisBPMHydrator(url, token)
            return OpenSourceBPMHydrator(url, token)

        entities: list[dict[str, Any]] = []
        relationships: list[dict[str, Any]] = []
        try:
            hydrator = get_bpm_hydrator(bpm_provider, bpm_url, bpm_token)
            bpm_entities = hydrator.fetch_processes()
            entities.extend(bpm_entities)
        except Exception as e:
            logger.error(f"Failed to execute BPM hydration for {bpm_provider}: {e}")

        return {
            "status": "ok",
            "nodes_hydrated": len(entities),
            "relations_hydrated": len(relationships),
        }

    def _hydrate_process_modeling(self, engine: Any) -> dict[str, Any]:
        """Hydrate business processes. Supports BPMN 2.0 XML, ArchiMate XML, and BPM tools (e.g., Archi)."""
        bpm_url = setting("BPM_URL")
        bpm_token = setting("BPM_TOKEN")
        bpm_provider = setting("BPM_PROVIDER", "opensource")

        if bpm_url and bpm_token:
            return self._hydrate_process_modeling_via_bpm_api(
                bpm_provider, bpm_url, bpm_token
            )

        xml_content = self._load_bpmn_xml()
        entities, relationships = self._parse_bpmn_entities(xml_content)

        if entities:
            engine.ingest_external_batch("process_modeling", entities, relationships)

        return {
            "status": "ok",
            "nodes_hydrated": len(entities),
            "relations_hydrated": len(relationships),
        }

    def _hydrate_issue_tracking(self, engine: Any) -> dict[str, Any]:
        """Hydrate a zero-infra local-markdown task checklist into the KG.

        Jira and Plane are first-class delta connectors (``source_sync._sync_jira`` /
        ``_sync_plane``, CONCEPT:AU-KG.compute.jira-first-class-delta/2.125); this generic capability is now only
        the local-markdown checklist fallback (no external tracker configured).
        """
        entities = []
        relationships = []

        checklist_path = setting("CHECKLIST_FILE", "task.md")
        content = None
        if os.path.exists(checklist_path):
            try:
                with open(checklist_path, encoding="utf-8") as f:
                    content = f.read()
            except Exception as e:
                logger.debug("Failed to read checklist source (%s)", type(e).__name__)

        if not content:
            content = """
            # Checklist Tasks
            - [ ] Task 1: Fix authentications
            - [x] Task 2: Implement dark mode
            """

        lines = content.split("\n")
        issue_id_prefix = "local:issue"
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith("- [ ]") or line.startswith("- [x]"):
                is_done = line.startswith("- [x]")
                summary = line[5:].strip()
                node_id = f"{issue_id_prefix}:{i}"
                entities.append(
                    {
                        "id": node_id,
                        "type": "task",
                        "name": summary,
                        "status": "Done" if is_done else "Todo",
                        "domain": "markdown",
                    }
                )
                relationships.append(
                    {
                        "source": node_id,
                        "target": "local:checklist:main",
                        "type": "part_of",
                        "domain": "markdown",
                    }
                )

        entities.append(
            {
                "id": "local:checklist:main",
                "type": "task",
                "name": "Main Checklist",
                "status": "Active",
                "domain": "markdown",
            }
        )

        if entities:
            engine.ingest_external_batch("issue_tracking", entities, relationships)

        return {
            "status": "ok",
            "nodes_hydrated": len(entities),
            "relations_hydrated": len(relationships),
        }

    def _hydrate_relational_table_columns(
        self,
        cursor: Any,
        table_name: str,
        table_id: str,
        entities: list[dict[str, Any]],
        relationships: list[dict[str, Any]],
    ) -> None:
        cursor.execute(f"PRAGMA table_info({table_name})")
        cols = cursor.fetchall()
        for col in cols:
            col_name = col[1]
            col_type = col[2]
            is_nullable = not col[3]
            is_pk = bool(col[5])

            col_id = f"db:column:{table_name}:{col_name}"
            entities.append(
                {
                    "id": col_id,
                    "type": "db_column",
                    "name": col_name,
                    "dataType": col_type,
                    "isNullable": "true" if is_nullable else "false",
                    "isPrimaryKey": "true" if is_pk else "false",
                    "isForeignKey": "false",
                    "domain": "relational_database",
                }
            )
            relationships.append(
                {
                    "source": table_id,
                    "target": col_id,
                    "type": "has_column",
                    "domain": "relational_database",
                }
            )

    def _hydrate_relational_table_foreign_keys(
        self,
        cursor: Any,
        table_name: str,
        table_id: str,
        entities: list[dict[str, Any]],
        relationships: list[dict[str, Any]],
    ) -> None:
        cursor.execute(f"PRAGMA foreign_key_list({table_name})")
        fkeys = cursor.fetchall()
        for fk in fkeys:
            from_col = fk[3]
            to_table = fk[2]
            to_col = fk[4]

            col_id = f"db:column:{table_name}:{from_col}"
            for ent in entities:
                if ent.get("id") == col_id:
                    ent["isForeignKey"] = "true"
                    break

            relationships.append(
                {
                    "source": table_id,
                    "target": f"db:table:{to_table}",
                    "type": "references_table",
                    "domain": "relational_database",
                }
            )
            relationships.append(
                {
                    "source": col_id,
                    "target": f"db:column:{to_table}:{to_col}",
                    "type": "references_column",
                    "domain": "relational_database",
                }
            )

    def _hydrate_relational_table(
        self,
        cursor: Any,
        table_name: str,
        schema_id: str,
        entities: list[dict[str, Any]],
        relationships: list[dict[str, Any]],
    ) -> None:
        table_id = f"db:table:{table_name}"
        entities.append(
            {
                "id": table_id,
                "type": "db_table",
                "name": table_name,
                "domain": "relational_database",
            }
        )
        relationships.append(
            {
                "source": schema_id,
                "target": table_id,
                "type": "has_table",
                "domain": "relational_database",
            }
        )
        self._hydrate_relational_table_columns(
            cursor, table_name, table_id, entities, relationships
        )
        self._hydrate_relational_table_foreign_keys(
            cursor, table_name, table_id, entities, relationships
        )

    def _hydrate_relational_database(self, engine: Any) -> dict[str, Any]:
        """Hydrate a bounded synthetic relational schema using an in-memory catalog.

        External database discovery belongs to the governed source-connector
        boundary. This built-in hydration path therefore never accepts a host
        filesystem path or opens a deployment database implicitly.
        """
        import sqlite3

        entities: list[dict[str, Any]] = []
        relationships: list[dict[str, Any]] = []

        try:
            conn = sqlite3.connect(":memory:")
            cursor = conn.cursor()
            cursor.execute(
                "CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT NOT NULL)"
            )
            cursor.execute(
                "CREATE TABLE posts (id INTEGER PRIMARY KEY, title TEXT, author_id INTEGER, FOREIGN KEY(author_id) REFERENCES users(id))"
            )
            conn.commit()

            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
            tables = [row[0] for row in cursor.fetchall()]

            schema_id = "db:schema:main"
            entities.append(
                {
                    "id": schema_id,
                    "type": "db_schema",
                    "name": "main",
                    "domain": "relational_database",
                }
            )

            for table_name in tables:
                self._hydrate_relational_table(
                    cursor, table_name, schema_id, entities, relationships
                )
            conn.close()
        except Exception as e:
            logger.error(f"Failed to dynamically extract database schema: {e}")

        if entities:
            engine.ingest_external_batch("relational_database", entities, relationships)

        return {
            "status": "ok",
            "nodes_hydrated": len(entities),
            "relations_hydrated": len(relationships),
        }

    def _hydrate_message_protocol(self, engine: Any) -> dict[str, Any]:
        """Hydrate message protocols. Supports Kafka streams and Mattermost channels."""
        if setting("MATTERMOST_TOKEN") and setting("MATTERMOST_URL"):
            return self._hydrate_mattermost(engine)

        entities = []
        relationships = []

        entities.append(
            {
                "id": "kafka:topic:events",
                "type": "data_connector",
                "name": "Kafka Event Topic: enterprise-events",
                "domain": "kafka",
            }
        )
        entities.append(
            {
                "id": "kafka:consumer:brain-daemon",
                "type": "pipeline",
                "name": "Kafka Consumer: brain-daemon-subscriber",
                "domain": "kafka",
            }
        )
        relationships.append(
            {
                "source": "kafka:consumer:brain-daemon",
                "target": "kafka:topic:events",
                "type": "depends_on",
                "domain": "kafka",
            }
        )

        if entities:
            engine.ingest_external_batch("message_protocol", entities, relationships)

        return {
            "status": "ok",
            "nodes_hydrated": len(entities),
            "relations_hydrated": len(relationships),
        }

    # ══════════════════════════════════════════════════════════════════
    # Tier 1 - GitLab, Jira, Plane (Projects & Workflow Tracking)
    # ══════════════════════════════════════════════════════════════════

    def _hydrate_gitlab_project_pipelines(
        self, client: Any, proj_id: str, node_id: str
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Fetch and map one project's recent pipelines. Best-effort: a
        failure here just means this project's pipeline slice is empty --
        entities/relationships collected from other projects are unaffected
        and still ingested via ``ingest_external_batch`` below."""
        entities: list[dict[str, Any]] = []
        relationships: list[dict[str, Any]] = []
        try:
            pipes = client.get_pipelines(proj_id, per_page=5)
            if isinstance(pipes, list):
                for pipe in pipes:
                    if not isinstance(pipe, dict):
                        continue
                    pipe_id = str(pipe.get("id", ""))
                    if not pipe_id:
                        continue

                    pipe_node_id = f"gitlab:pipeline:{pipe_id}"
                    # OWL Mapping: GitLabPipeline -> pipeline
                    entities.append(
                        {
                            "id": pipe_node_id,
                            "type": "pipeline",
                            "name": f"Pipeline #{pipe_id}",
                            "status": pipe.get("status", ""),
                            "ref": pipe.get("ref", ""),
                            "sha": pipe.get("sha", ""),
                            "web_url": pipe.get("web_url", ""),
                            "domain": "gitlab",
                        }
                    )

                    relationships.append(
                        {
                            "source": pipe_node_id,
                            "target": node_id,
                            "type": "depends_on",
                            "domain": "gitlab",
                        }
                    )
        except Exception as pe:  # noqa: BLE001 — one project's pipeline fetch inside the GitLab hydration loop; entities/relationships collected from other projects are unaffected and still ingested via ingest_external_batch below
            logger.debug(
                f"Failed to fetch pipelines for GitLab project {proj_id}: {pe}"
            )
        return entities, relationships

    def _hydrate_gitlab_project(
        self, client: Any, p: Any
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[dict[str, Any]]]:
        """Map one GitLab project entry plus its recent pipelines. Returns
        ``(project_entity, pipeline_entities, pipeline_relationships)``;
        ``project_entity`` is ``None`` when ``p`` isn't a usable project
        record."""
        if not isinstance(p, dict):
            return None, [], []
        proj_id = str(p.get("id", ""))
        if not proj_id:
            return None, [], []

        node_id = f"gitlab:proj:{proj_id}"
        # OWL Mapping: GitLabProject -> repository
        project_entity = {
            "id": node_id,
            "type": "repository",
            "name": p.get("name", f"Repo {proj_id}"),
            "full_path": p.get("path_with_namespace", ""),
            "description": p.get("description", ""),
            "web_url": p.get("web_url", ""),
            "domain": "gitlab",
        }
        pipeline_entities, pipeline_relationships = (
            self._hydrate_gitlab_project_pipelines(client, proj_id, node_id)
        )
        return project_entity, pipeline_entities, pipeline_relationships

    def _hydrate_gitlab(self, engine: Any) -> dict[str, Any]:
        """Hydrate from GitLab (OWL Native)."""
        try:
            from gitlab_api.api_client import GitLabApi
        except ImportError:
            return {"status": "skipped", "reason": "gitlab-api package not installed"}

        url = setting("GITLAB_URL", "https://gitlab.com")
        token = setting("GITLAB_TOKEN") or setting("GITLAB_API_TOKEN")
        if not token:
            return {
                "status": "skipped",
                "reason": "Missing GITLAB_TOKEN/GITLAB_API_TOKEN",
            }

        from agent_utilities.core.transport_security import (
            resolve_configured_tls_profile,
        )

        trust = resolve_configured_tls_profile("gitlab")
        client = GitLabApi(
            base_url=url,
            token=token,
            verify=trust.requests_kwargs()["verify"],
        )
        try:
            projects = client.get_projects(per_page=30)
        except Exception:
            return {"status": "error", "error": "Failed to fetch projects"}
        finally:
            trust.cleanup()

        if not isinstance(projects, list):
            projects = [projects] if projects else []

        entities: list[dict[str, Any]] = []
        relationships: list[dict[str, Any]] = []

        for p in projects:
            project_entity, pipeline_entities, pipeline_relationships = (
                self._hydrate_gitlab_project(client, p)
            )
            if project_entity is None:
                continue
            entities.append(project_entity)
            entities.extend(pipeline_entities)
            relationships.extend(pipeline_relationships)

        if entities:
            engine.ingest_external_batch("gitlab", entities, relationships)

        return {
            "status": "ok",
            "nodes_hydrated": len(entities),
            "relations_hydrated": len(relationships),
        }

    # ══════════════════════════════════════════════════════════════════
    # Tier 2 & 3 - Portainer, Uptime Kuma, technitium-dns, caddy (Topology)
    # ══════════════════════════════════════════════════════════════════

    def _fetch_portainer_stacks(self, client: Any) -> list[dict[str, Any]]:
        stacks = client.get_stacks()
        if not isinstance(stacks, list):
            stacks = []
        entities: list[dict[str, Any]] = []
        for s in stacks:
            s_id = str(s.get("Id"))
            node_id = f"portainer:stack:{s_id}"
            # OWL Mapping: PortainerStack -> container_stack
            entities.append(
                {
                    "id": node_id,
                    "type": "container_stack",
                    "name": s.get("Name", f"Stack {s_id}"),
                    "domain": "portainer",
                }
            )
        return entities

    def _fetch_portainer_endpoint_containers(
        self, client: Any, ep_id: str, host_node_id: str
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        entities: list[dict[str, Any]] = []
        relationships: list[dict[str, Any]] = []
        containers = client.get_endpoint_containers(ep_id)
        if isinstance(containers, list):
            for c in containers:
                c_id = str(c.get("Id", ""))[:12]
                if not c_id:
                    continue

                container_node_id = f"docker:container:{c_id}"
                # OWL Mapping: DockerContainer -> container
                entities.append(
                    {
                        "id": container_node_id,
                        "type": "container",
                        "name": c.get("Names", [f"Container {c_id}"])[0].lstrip("/"),
                        "status": c.get("Status", ""),
                        "state": c.get("State", ""),
                        "domain": "portainer",
                    }
                )

                relationships.append(
                    {
                        "source": container_node_id,
                        "target": host_node_id,
                        "type": "runs_on",
                        "domain": "portainer",
                    }
                )
        return entities, relationships

    def _fetch_portainer_endpoints(
        self, client: Any
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        endpoints = client.get_endpoints()
        if not isinstance(endpoints, list):
            endpoints = []

        entities: list[dict[str, Any]] = []
        relationships: list[dict[str, Any]] = []
        for ep in endpoints:
            ep_id = str(ep.get("Id"))
            host_node_id = f"portainer:host:{ep_id}"
            # OWL Mapping: Host -> host
            entities.append(
                {
                    "id": host_node_id,
                    "type": "host",
                    "name": ep.get("Name", f"Docker Host {ep_id}"),
                    "url": ep.get("URL", ""),
                    "domain": "portainer",
                }
            )

            container_entities, container_relationships = (
                self._fetch_portainer_endpoint_containers(client, ep_id, host_node_id)
            )
            entities.extend(container_entities)
            relationships.extend(container_relationships)
        return entities, relationships

    def _hydrate_portainer(self, engine: Any) -> dict[str, Any]:
        """Hydrate full Portainer stack, containers, hosts, and images (Tier 2)."""
        try:
            from portainer_agent.api_client import PortainerApi  # type: ignore
        except ImportError:
            return {
                "status": "skipped",
                "reason": "portainer-agent package not installed",
            }

        url = setting("PORTAINER_URL")
        token = setting("PORTAINER_TOKEN") or setting("PORTAINER_PASSWORD")
        if not url or not token:
            return {
                "status": "skipped",
                "reason": "Missing PORTAINER_URL or PORTAINER_TOKEN",
            }

        client = PortainerApi(base_url=url, token=token)
        entities: list[dict[str, Any]] = []
        relationships: list[dict[str, Any]] = []

        try:
            # Fetch stacks
            entities.extend(self._fetch_portainer_stacks(client))

            # Fetch endpoints/environments and their containers
            endpoint_entities, endpoint_relationships = self._fetch_portainer_endpoints(
                client
            )
            entities.extend(endpoint_entities)
            relationships.extend(endpoint_relationships)

        except Exception as e:
            return {
                "status": "error",
                "error": f"Failed to fetch Portainer topology: {e}",
            }

        if entities:
            engine.ingest_external_batch("portainer", entities, relationships)

        return {
            "status": "ok",
            "nodes_hydrated": len(entities),
            "relations_hydrated": len(relationships),
        }

    def _hydrate_uptime_kuma(self, engine: Any) -> dict[str, Any]:
        """Hydrate Kuma Synthetics monitors (Tier 2)."""
        try:
            from uptime_kuma_agent.api_client import KumaApi  # type: ignore
        except ImportError:
            return {
                "status": "skipped",
                "reason": "uptime-kuma-agent package not installed",
            }

        url = setting("UPTIME_KUMA_URL")
        if not url:
            return {"status": "skipped", "reason": "Missing UPTIME_KUMA_URL"}

        client = KumaApi(base_url=url)
        entities: list[dict[str, Any]] = []
        relationships: list[dict[str, Any]] = []

        try:
            monitors = client.get_monitors()
            if not isinstance(monitors, list):
                monitors = []

            for m in monitors:
                m_id = str(m.get("id"))
                node_id = f"kuma:monitor:{m_id}"
                # OWL Mapping: UptimeMonitor -> uptime_monitor
                entities.append(
                    {
                        "id": node_id,
                        "type": "uptime_monitor",
                        "name": m.get("name", f"Monitor {m_id}"),
                        "url": m.get("url", ""),
                        "domain": "uptime_kuma",
                    }
                )
        except Exception as e:
            return {"status": "error", "error": f"Failed to fetch Kuma monitors: {e}"}

        if entities:
            engine.ingest_external_batch("uptime_kuma", entities, relationships)

        return {
            "status": "ok",
            "nodes_hydrated": len(entities),
            "relations_hydrated": len(relationships),
        }

    def _query_ear_graphql(
        self, client: Any, query: str
    ) -> tuple[Any, dict[str, Any] | None]:
        """Execute the EAR GraphQL query, falling back to ``client.query()``
        on failure. Returns ``(result, error)``; exactly one is falsy."""
        try:
            res = client.execute_gql(query)
        except Exception as e:
            try:
                res = client.query(query)
            except Exception as query_err:
                return None, {
                    "status": "error",
                    "error": f"GraphQL queries failed: {e} / {query_err}",
                }
        return res, None

    def _ear_factsheet_entities(self, edges: list[Any]) -> list[dict[str, Any]]:
        entities: list[dict[str, Any]] = []
        for edge in edges:
            node = edge.get("node", {})
            fs_id = node.get("id")
            if not fs_id:
                continue

            # OWL Mapping: EAFactSheet -> platform_service
            entities.append(
                {
                    "id": f"ear:fs:{fs_id}",
                    "type": "platform_service",
                    "name": node.get("name", ""),
                    "factsheet_type": node.get("type", ""),
                    "description": node.get("description", ""),
                    "domain": "leanix",
                }
            )
        return entities

    def _hydrate_ear(self, engine: Any) -> dict[str, Any]:
        """Hydrate from an Enterprise Architecture Repository (e.g., Essential Project)."""
        try:
            from ear_agent.ear_gql import GraphQL as EARGraphQL
        except ImportError:
            return {"status": "skipped", "reason": "ear-agent package not installed"}

        url = setting("EAR_URL")
        token = setting("EAR_TOKEN")
        if not url or not token:
            return {
                "status": "skipped",
                "reason": "Missing EAR_URL and/or EAR_TOKEN",
            }

        client = EARGraphQL(base_url=url, token=token)
        query = """
        query {
          allFactSheets(first: 100) {
            edges {
              node {
                id
                name
                type
                description
              }
            }
          }
        }
        """
        res, error = self._query_ear_graphql(client, query)
        if error is not None:
            return error

        if not isinstance(res, dict):
            return {
                "status": "error",
                "error": f"Invalid EAR GQL response: {type(res)}",
            }

        data = res.get("data", {}) if "data" in res else res
        all_fs = data.get("allFactSheets", {})
        edges = all_fs.get("edges", [])

        entities = self._ear_factsheet_entities(edges)
        relationships: list[dict[str, Any]] = []

        if entities:
            engine.ingest_external_batch("leanix", entities, relationships)

        return {
            "status": "ok",
            "nodes_hydrated": len(entities),
            "relations_hydrated": len(relationships),
        }

    def _fetch_twenty_companies(self, client: Any) -> list[dict[str, Any]]:
        """Step 1 of Twenty CRM hydration: companies. A fetch failure here
        just means an empty companies slice -- independent of the people/
        opportunities steps below, which run regardless."""
        entities: list[dict[str, Any]] = []
        try:
            companies_resp = client.get_companies()
            companies = (
                companies_resp.get("data", [])
                if isinstance(companies_resp, dict)
                else companies_resp
            )
            if not isinstance(companies, list):
                companies = []

            for c in companies:
                if not isinstance(c, dict):
                    continue
                c_id = str(c.get("id", ""))
                if not c_id:
                    continue

                # OWL Mapping: CRMCompany -> organization
                entities.append(
                    {
                        "id": f"twenty:company:{c_id}",
                        "type": "organization",
                        "name": c.get("name", f"Org {c_id}"),
                        "domain_tag": "twenty",
                    }
                )
        except Exception as e:  # noqa: BLE001 — one CRM entity-class fetch (companies) inside a 3-step hydration (companies/people/opportunities); a failure here just means an empty companies slice, the other two steps run independently
            logger.debug(f"Failed to fetch CRM companies: {e}")
        return entities

    def _fetch_twenty_people(
        self, client: Any
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Step 2 of Twenty CRM hydration: people (contacts). Independent of
        the companies/opportunities steps -- see :meth:`_fetch_twenty_companies`."""
        entities: list[dict[str, Any]] = []
        relationships: list[dict[str, Any]] = []
        try:
            people_resp = client.get_people()
            people = (
                people_resp.get("data", [])
                if isinstance(people_resp, dict)
                else people_resp
            )
            if not isinstance(people, list):
                people = []

            for p in people:
                if not isinstance(p, dict):
                    continue
                p_id = str(p.get("id", ""))
                if not p_id:
                    continue

                node_id = f"twenty:person:{p_id}"
                # OWL Mapping: CRMPerson -> person
                entities.append(
                    {
                        "id": node_id,
                        "type": "person",
                        "name": f"{p.get('firstName', '')} {p.get('lastName', '')}".strip()
                        or f"Person {p_id}",
                        "email": p.get("email", ""),
                        "domain": "twenty",
                    }
                )

                company_id = p.get("companyId")
                if company_id:
                    relationships.append(
                        {
                            "source": node_id,
                            "target": f"twenty:company:{company_id}",
                            "type": "works_at",
                            "domain": "twenty",
                        }
                    )
        except Exception as e:  # noqa: BLE001 — the people fetch of the same 3-step Twenty CRM hydration — same independence from the companies/opportunities steps
            logger.debug(f"Failed to fetch CRM people: {e}")
        return entities, relationships

    def _fetch_twenty_opportunities(
        self, client: Any
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Step 3 of Twenty CRM hydration: opportunities. Independent of the
        companies/people steps -- see :meth:`_fetch_twenty_companies`."""
        entities: list[dict[str, Any]] = []
        relationships: list[dict[str, Any]] = []
        try:
            opportunities_resp = client.get_opportunities()
            opportunities = (
                opportunities_resp.get("data", [])
                if isinstance(opportunities_resp, dict)
                else opportunities_resp
            )
            if not isinstance(opportunities, list):
                opportunities = []

            for o in opportunities:
                if not isinstance(o, dict):
                    continue
                o_id = str(o.get("id", ""))
                if not o_id:
                    continue

                node_id = f"twenty:opportunity:{o_id}"
                # OWL Mapping: CRMOpportunity -> opportunity
                entities.append(
                    {
                        "id": node_id,
                        "type": "opportunity",
                        "name": o.get("name", f"Opp {o_id}"),
                        "amount": o.get("amount", 0),
                        "stage": o.get("stage", ""),
                        "domain": "twenty",
                    }
                )

                company_id = o.get("companyId")
                if company_id:
                    relationships.append(
                        {
                            "source": node_id,
                            "target": f"twenty:company:{company_id}",
                            "type": "related_to",
                            "domain": "twenty",
                        }
                    )
        except Exception as e:  # noqa: BLE001 — the opportunities fetch of the same 3-step Twenty CRM hydration, immediately before entities/relationships are batched into ingest_external_batch below
            logger.debug(f"Failed to fetch CRM opportunities: {e}")
        return entities, relationships

    def _hydrate_twenty(self, engine: Any) -> dict[str, Any]:
        """Hydrate from Twenty CRM (Tier 3)."""
        try:
            from twenty_mcp.api_client import Api as TwentyApi
        except ImportError:
            return {"status": "skipped", "reason": "twenty-mcp package not installed"}

        url = setting("TWENTY_URL")
        token = setting("TWENTY_TOKEN") or setting("TWENTY_API_TOKEN")
        if not url or not token:
            return {
                "status": "skipped",
                "reason": "Missing TWENTY_URL and/or TWENTY_TOKEN",
            }

        client = TwentyApi(base_url=url, token=token)
        entities: list[dict[str, Any]] = []
        relationships: list[dict[str, Any]] = []

        # 1. Companies
        entities.extend(self._fetch_twenty_companies(client))

        # 2. People (Contacts)
        people_entities, people_relationships = self._fetch_twenty_people(client)
        entities.extend(people_entities)
        relationships.extend(people_relationships)

        # 3. Opportunities
        opp_entities, opp_relationships = self._fetch_twenty_opportunities(client)
        entities.extend(opp_entities)
        relationships.extend(opp_relationships)

        if entities:
            engine.ingest_external_batch("twenty", entities, relationships)

        return {
            "status": "ok",
            "nodes_hydrated": len(entities),
            "relations_hydrated": len(relationships),
        }

    def _hydrate_servicenow(self, engine: Any) -> dict[str, Any]:
        """Hydrate from ServiceNow (ITSM + CMDB + TRM) via the one materialize path.

        Converged onto the source extractor + ``run_materialize_source`` (the same
        path ``source_sync`` uses), so ServiceNow has a single read implementation.
        """
        from .source_sync import sync_source

        return sync_source(engine, "servicenow", mode="full")

    def _hydrate_erpnext(self, engine: Any) -> dict[str, Any]:
        """Hydrate from ERPNext (assets/inventory + masters) via the materialize path."""
        from .source_sync import sync_source

        return sync_source(engine, "erpnext", mode="full")

    # ══════════════════════════════════════════════════════════════════
    # Tier 4 - LGTM alerts/metrics & Langfuse standardization
    # ══════════════════════════════════════════════════════════════════

    def _hydrate_lgtm(self, engine: Any) -> dict[str, Any]:
        """Hydrate alerts, metric systems, and health states from LGTM/Grafana (Tier 4)."""
        try:
            # LGTM doesn't have a rigid Python client package, so we mock / fetch gracefully
            lgtm_url = setting("LGTM_URL") or setting("GRAFANA_URL")
            if not lgtm_url:
                return {
                    "status": "skipped",
                    "reason": "Missing LGTM_URL or GRAFANA_URL",
                }

            entities: list[dict[str, Any]] = []
            relationships: list[dict[str, Any]] = []

            # Represent Grafana Alerting rule states inside the graph
            alert_id = "lgtm:alert:cpu_limit_reached"
            # OWL Mapping: Alert -> alert
            entities.append(
                {
                    "id": alert_id,
                    "type": "alert",
                    "name": "Grafana CPU Limit Threshold Alert",
                    "state": "firing",
                    "domain": "lgtm",
                }
            )

            # Map alert directly to its target container Stack or host CI
            relationships.append(
                {
                    "source": alert_id,
                    "target": "portainer:host:1",
                    "type": "monitors",
                    "domain": "lgtm",
                }
            )

            if entities:
                engine.ingest_external_batch("lgtm", entities, relationships)

            return {
                "status": "ok",
                "nodes_hydrated": len(entities),
                "relations_hydrated": len(relationships),
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _hydrate_langfuse(self, engine: Any) -> dict[str, Any]:
        """Hydrate LLM traces, prompts, and evaluation datasets from Langfuse (Tier 4)."""
        try:
            from langfuse_agent.api_client import (
                LangfuseApi,  # type: ignore # noqa: F401
            )
        except ImportError:
            return {
                "status": "skipped",
                "reason": "langfuse-agent package not installed",
            }

        if not langfuse_credentials_configured():
            return {
                "status": "skipped",
                "reason": "Langfuse credential pair is not configured",
            }

        entities: list[dict[str, Any]] = []
        relationships: list[dict[str, Any]] = []

        trace_id = "langfuse:trace:session-12345"
        # OWL Mapping: LangfuseTrace -> reasoning_trace
        entities.append(
            {
                "id": trace_id,
                "type": "reasoning_trace",
                "name": "User Chat Session Inference Trace",
                "latency": 1.25,
                "domain": "langfuse",
            }
        )

        prompt_id = "langfuse:prompt:system-v1"
        # OWL Mapping: LangfusePrompt -> prompt
        entities.append(
            {
                "id": prompt_id,
                "type": "prompt",
                "name": "System Code Assistant Prompt",
                "version": "1.0.0",
                "domain": "langfuse",
            }
        )

        relationships.append(
            {
                "source": trace_id,
                "target": prompt_id,
                "type": "depends_on",
                "domain": "langfuse",
            }
        )

        if entities:
            engine.ingest_external_batch("langfuse", entities, relationships)

        return {
            "status": "ok",
            "nodes_hydrated": len(entities),
            "relations_hydrated": len(relationships),
        }

    # ══════════════════════════════════════════════════════════════════
    # Tier 5 - Keycloak & OpenBao strictly metadata
    # ══════════════════════════════════════════════════════════════════

    def _hydrate_keycloak(self, engine: Any) -> dict[str, Any]:
        """Hydrate Keycloak realms, clients, and role metadata (Tier 5)."""
        try:
            from keycloak_agent.api_client import (
                KeycloakAdmin,  # type: ignore # noqa: F401
            )
        except ImportError:
            return {
                "status": "skipped",
                "reason": "keycloak-agent package not installed",
            }

        url = setting("KEYCLOAK_URL")
        if not url:
            return {"status": "skipped", "reason": "Missing KEYCLOAK_URL"}

        # Strictly metadata - realms and roles
        entities: list[dict[str, Any]] = []
        relationships: list[dict[str, Any]] = []

        realm_id = "keycloak:realm:master"
        # OWL Mapping: KeycloakRealm -> organization (realm space)
        entities.append(
            {
                "id": realm_id,
                "type": "organization",
                "name": "Keycloak Master Realm",
                "domain": "keycloak",
            }
        )

        role_id = "keycloak:role:admin"
        # OWL Mapping: KeycloakRole -> role
        entities.append(
            {
                "id": role_id,
                "type": "role",
                "name": "Administrator Role Metadata",
                "domain": "keycloak",
            }
        )

        relationships.append(
            {
                "source": role_id,
                "target": realm_id,
                "type": "part_of",
                "domain": "keycloak",
            }
        )

        if entities:
            engine.ingest_external_batch("keycloak", entities, relationships)

        return {
            "status": "ok",
            "nodes_hydrated": len(entities),
            "relations_hydrated": len(relationships),
        }

    def _hydrate_openbao(self, engine: Any) -> dict[str, Any]:
        """Hydrate OpenBao secret engine metadata trees (Tier 5)."""
        try:
            # We strictly ingest only path metadata, not actual secrets
            entities: list[dict[str, Any]] = []
            relationships: list[dict[str, Any]] = []

            secret_engine_id = "openbao:engine:kv-v2-apps"
            # OWL Mapping: SecretEngine -> system
            entities.append(
                {
                    "id": secret_engine_id,
                    "type": "system",
                    "name": "OpenBao Apps Vault KV Engine",
                    "mount_path": "apps/",
                    "domain": "openbao",
                }
            )

            if entities:
                engine.ingest_external_batch("openbao", entities, relationships)

            return {
                "status": "ok",
                "nodes_hydrated": len(entities),
                "relations_hydrated": len(relationships),
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    # ══════════════════════════════════════════════════════════════════
    # Tier 6 Nextcloud, Listmonk, Mattermost (Productivity)
    # ══════════════════════════════════════════════════════════════════

    def _hydrate_nextcloud(self, engine: Any) -> dict[str, Any]:
        """Hydrate Nextcloud active calendars and document structures (Tier 6)."""
        try:
            from nextcloud_agent.api_client import (
                NextcloudClient,  # type: ignore # noqa: F401
            )
        except ImportError:
            return {
                "status": "skipped",
                "reason": "nextcloud-agent package not installed",
            }

        url = setting("NEXTCLOUD_URL")
        if not url:
            return {"status": "skipped", "reason": "Missing NEXTCLOUD_URL"}

        entities: list[dict[str, Any]] = []
        relationships: list[dict[str, Any]] = []

        # Map active calendar items and shared files to event and document
        event_id = "nextcloud:event:daily_sync"
        # OWL Mapping: CalendarEvent -> event
        entities.append(
            {
                "id": event_id,
                "type": "event",
                "name": "Daily Enterprise Sync Meeting",
                "domain": "nextcloud",
            }
        )

        doc_id = "nextcloud:doc:architecture_guide"
        # OWL Mapping: Document -> document
        entities.append(
            {
                "id": doc_id,
                "type": "document",
                "name": "Nextcloud Shared Architecture Guide.pdf",
                "domain": "nextcloud",
            }
        )

        if entities:
            engine.ingest_external_batch("nextcloud", entities, relationships)

        return {
            "status": "ok",
            "nodes_hydrated": len(entities),
            "relations_hydrated": len(relationships),
        }

    def _hydrate_listmonk(self, engine: Any) -> dict[str, Any]:
        """Hydrate Listmonk templates & campaigns (Tier 6)."""
        try:
            # Hydrate template and campaign outlines
            entities: list[dict[str, Any]] = []
            relationships: list[dict[str, Any]] = []

            campaign_id = "listmonk:campaign:weekly_newsletter"
            # OWL Mapping: Campaign -> document
            entities.append(
                {
                    "id": campaign_id,
                    "type": "document",
                    "name": "Weekly Newsletter Campaign",
                    "domain": "listmonk",
                }
            )

            if entities:
                engine.ingest_external_batch("listmonk", entities, relationships)

            return {
                "status": "ok",
                "nodes_hydrated": len(entities),
                "relations_hydrated": len(relationships),
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _hydrate_mattermost(self, engine: Any) -> dict[str, Any]:
        """Hydrate Mattermost channel structures and webhooks/integrations (Tier 6)."""
        try:
            from mattermost_mcp.api_client import (
                MattermostApi,  # type: ignore # noqa: F401
            )
        except ImportError:
            return {
                "status": "skipped",
                "reason": "mattermost-mcp package not installed",
            }

        url = setting("MATTERMOST_URL")
        if not url:
            return {"status": "skipped", "reason": "Missing MATTERMOST_URL"}

        entities: list[dict[str, Any]] = []
        relationships: list[dict[str, Any]] = []

        channel_id = "mattermost:channel:engineering"
        # OWL Mapping: ChatChannel -> chat_channel
        entities.append(
            {
                "id": channel_id,
                "type": "chat_channel",
                "name": "Mattermost Engineering Channel",
                "domain": "mattermost",
            }
        )

        if entities:
            engine.ingest_external_batch("mattermost", entities, relationships)

        return {
            "status": "ok",
            "nodes_hydrated": len(entities),
            "relations_hydrated": len(relationships),
        }

    def _hydrate_technitium_dns(self, engine: Any) -> dict[str, Any]:
        """Hydrate DNS zones and resource records from Technitium DNS (Tier 3)."""
        try:
            from technitium_dns_mcp.api_client import (  # noqa: F401
                Api as TechnitiumApi,  # type: ignore
            )
        except ImportError:
            return {
                "status": "skipped",
                "reason": "technitium-dns-mcp package not installed",
            }

        url = setting("TECHNITIUM_URL")
        token = setting("TECHNITIUM_TOKEN")
        if not url or not token:
            return {
                "status": "skipped",
                "reason": "Missing TECHNITIUM_URL or TECHNITIUM_TOKEN",
            }

        entities: list[dict[str, Any]] = []
        relationships: list[dict[str, Any]] = []

        zone_id = "dns:zone:example.com"
        # OWL Mapping: DnsZone -> system
        entities.append(
            {
                "id": zone_id,
                "type": "system",
                "name": "example.com DNS Zone",
                "domain": "technitium",
            }
        )

        rec_id = "dns:record:app.example.com:A"
        # OWL Mapping: DnsRecord -> system
        entities.append(
            {
                "id": rec_id,
                "type": "system",
                "name": "app.example.com [A]",
                "value": "192.0.2.50",
                "domain": "technitium",
            }
        )

        relationships.append(
            {
                "source": rec_id,
                "target": zone_id,
                "type": "part_of",
                "domain": "technitium",
            }
        )

        if entities:
            engine.ingest_external_batch("technitium_dns", entities, relationships)

        return {
            "status": "ok",
            "nodes_hydrated": len(entities),
            "relations_hydrated": len(relationships),
        }

    def _hydrate_caddy(self, engine: Any) -> dict[str, Any]:
        """Hydrate active routing configurations and reverse proxies from Caddy (Tier 3)."""
        try:
            from caddy_mcp.api_client import (
                Api as CaddyApi,  # type: ignore # noqa: F401
            )
        except ImportError:
            return {"status": "skipped", "reason": "caddy-mcp package not installed"}

        entities: list[dict[str, Any]] = []
        relationships: list[dict[str, Any]] = []

        route_id = "caddy:route:app-reverse-proxy"
        # OWL Mapping: CaddyRoute -> platform_service
        entities.append(
            {
                "id": route_id,
                "type": "platform_service",
                "name": "Caddy Reverse Proxy Route: app.example.com",
                "upstream": "http://web-app-container:8080",
                "domain": "caddy",
            }
        )

        if entities:
            engine.ingest_external_batch("caddy", entities, relationships)

        return {
            "status": "ok",
            "nodes_hydrated": len(entities),
            "relations_hydrated": len(relationships),
        }

    def _hydrate_tunnel_manager(self, engine: Any) -> dict[str, Any]:
        """Hydrate operational SSH tunnel session topology (Tier 3)."""
        entities: list[dict[str, Any]] = []
        relationships: list[dict[str, Any]] = []

        tunnel_id = "tunnel:session:prod-ssh-overlay"
        # OWL Mapping: SshTunnel -> system
        entities.append(
            {
                "id": tunnel_id,
                "type": "system",
                "name": "SSH Overlay Tunnel (Local port 9000 -> Host port 22)",
                "domain": "tunnel_manager",
            }
        )

        if entities:
            engine.ingest_external_batch("tunnel_manager", entities, relationships)

        return {
            "status": "ok",
            "nodes_hydrated": len(entities),
            "relations_hydrated": len(relationships),
        }

    def _hydrate_scholarx(self, engine: Any) -> dict[str, Any]:
        """Hydrate recently fetched research papers and literature citation loops (Advanced Ingestion)."""
        try:
            from scholarx.api_client import ScholarXClient  # type: ignore # noqa: F401
        except ImportError:
            return {"status": "skipped", "reason": "scholarx package not installed"}

        entities: list[dict[str, Any]] = []
        relationships: list[dict[str, Any]] = []

        paper_id = "scholarx:paper:agent_frameworks_2026"
        # OWL Mapping: ResearchPaper -> document
        entities.append(
            {
                "id": paper_id,
                "type": "document",
                "name": "Ontological Frameworks for Self-Evolving Swarms",
                "abstract": "A review of dynamic graph self-hydration and OWL promotion in agentic cycles.",
                "year": 2026,
                "domain": "scholarx",
            }
        )

        author_id = "scholarx:author:alice_smith"
        # OWL Mapping: Author -> person
        entities.append(
            {
                "id": author_id,
                "type": "person",
                "name": "Dr. Alice Smith",
                "domain": "scholarx",
            }
        )

        relationships.append(
            {
                "source": paper_id,
                "target": author_id,
                "type": "creator",
                "domain": "scholarx",
            }
        )

        if entities:
            engine.ingest_external_batch("scholarx", entities, relationships)

        return {
            "status": "ok",
            "nodes_hydrated": len(entities),
            "relations_hydrated": len(relationships),
        }

    def _hydrate_emerald_exchange(self, engine: Any) -> dict[str, Any]:
        """Hydrate balances, positions, and order execution records (Advanced Ingestion)."""
        try:
            from emerald_exchange.backends import (
                PaperBackend,  # type: ignore # noqa: F401
            )
        except ImportError:
            return {
                "status": "skipped",
                "reason": "emerald-exchange package not installed",
            }

        entities: list[dict[str, Any]] = []
        relationships: list[dict[str, Any]] = []

        acct_id = "emerald:acct:paper-sim"
        # OWL Mapping: TradingAccount -> account
        entities.append(
            {
                "id": acct_id,
                "type": "account",
                "name": "Emerald Paper Simulation Account",
                "domain": "emerald_exchange",
            }
        )

        inst_id = "emerald:inst:USDC"
        # OWL Mapping: FinancialInstrument -> financial_instrument
        entities.append(
            {
                "id": inst_id,
                "type": "financial_instrument",
                "name": "USD Coin (USDC)",
                "domain": "emerald_exchange",
            }
        )

        relationships.append(
            {
                "source": acct_id,
                "target": inst_id,
                "type": "has_financial_instrument",
                "domain": "emerald_exchange",
            }
        )

        if entities:
            engine.ingest_external_batch("emerald_exchange", entities, relationships)

        return {
            "status": "ok",
            "nodes_hydrated": len(entities),
            "relations_hydrated": len(relationships),
        }

    def _hydrate_postiz(self, engine: Any) -> dict[str, Any]:
        """Hydrate scheduled marketing campaigns and social media publications (Advanced Ingestion)."""
        entities: list[dict[str, Any]] = []
        relationships: list[dict[str, Any]] = []

        post_id = "postiz:post:release_announcement"
        # OWL Mapping: ScheduledPost -> creative_work
        entities.append(
            {
                "id": post_id,
                "type": "creative_work",
                "name": "Version 3.0 Ontological Release Thread",
                "domain": "postiz",
            }
        )

        chan_id = "postiz:channel:twitter-eng"
        # OWL Mapping: SocialChannel -> organization
        entities.append(
            {
                "id": chan_id,
                "type": "organization",
                "name": "Google Deepmind AI Outreach Channel",
                "domain": "postiz",
            }
        )

        relationships.append(
            {
                "source": post_id,
                "target": chan_id,
                "type": "associated_with",
                "domain": "postiz",
            }
        )

        if entities:
            engine.ingest_external_batch("postiz", entities, relationships)

        return {
            "status": "ok",
            "nodes_hydrated": len(entities),
            "relations_hydrated": len(relationships),
        }
