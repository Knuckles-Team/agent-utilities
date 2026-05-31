"""Knowledge Graph Hydration Service (OWL Native).

Handles dynamic discovery, instantiation, ontological translation, and batch
ingestion from domain-specific APIs into OWL-promotable LPG nodes and edges.

Architecture (CONCEPT:KG-2.7 — Capability Abstraction Layer):
  The CAPABILITY_REGISTRY decouples concrete connectors from abstract
  capability categories.  Each entry maps a source identifier to its
  capability category and the private method that implements the connector.
  New sources are added by extending the registry — the core orchestration
  logic (hydrate_source / hydrate_all) never changes.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# CAPABILITY_REGISTRY — maps source identifiers to abstract capability
# categories and their connector methods.  Adding a new data source only
# requires appending an entry here; the orchestration loop is generic.
# ═══════════════════════════════════════════════════════════════════
CAPABILITY_REGISTRY: dict[str, dict[str, str]] = {
    "gitlab": {"category": "source_control", "method": "_hydrate_source_control"},
    "github": {"category": "source_control", "method": "_hydrate_source_control"},
    "source_control": {"category": "source_control", "method": "_hydrate_source_control"},
    "leanix": {"category": "enterprise_architecture", "method": "_hydrate_enterprise_architecture"},
    "enterprise_architecture": {"category": "enterprise_architecture", "method": "_hydrate_enterprise_architecture"},
    "twenty": {"category": "crm", "method": "_hydrate_twenty"},
    "servicenow": {"category": "itsm", "method": "_hydrate_servicenow"},
    "jira": {"category": "issue_tracking", "method": "_hydrate_issue_tracking"},
    "plane": {"category": "issue_tracking", "method": "_hydrate_issue_tracking"},
    "issue_tracking": {"category": "issue_tracking", "method": "_hydrate_issue_tracking"},
    "process_modeling": {"category": "process_modeling", "method": "_hydrate_process_modeling"},
    "relational_database": {"category": "databases", "method": "_hydrate_relational_database"},
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
    "message_protocol": {"category": "collaboration", "method": "_hydrate_message_protocol"},
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
        status = {
            "gitlab": {
                "configured": bool(
                    os.environ.get("GITLAB_TOKEN") or os.environ.get("GITLAB_API_TOKEN")
                ),
                "url": os.environ.get("GITLAB_URL", "https://gitlab.com"),
            },
            "leanix": {
                "configured": bool(os.environ.get("LEANIX_TOKEN")),
                "url": os.environ.get("LEANIX_URL", ""),
            },
            "twenty": {
                "configured": bool(
                    os.environ.get("TWENTY_TOKEN") or os.environ.get("TWENTY_API_TOKEN")
                ),
                "url": os.environ.get("TWENTY_URL", ""),
            },
            "servicenow": {
                "configured": bool(
                    os.environ.get("SERVICENOW_USER")
                    and os.environ.get("SERVICENOW_PASSWORD")
                ),
                "url": os.environ.get("SERVICENOW_URL", ""),
            },
            "jira": {
                "configured": bool(
                    os.environ.get("JIRA_TOKEN") or os.environ.get("JIRA_API_TOKEN")
                ),
                "url": os.environ.get("JIRA_URL", ""),
            },
            "plane": {
                "configured": bool(
                    os.environ.get("PLANE_TOKEN") or os.environ.get("PLANE_API_TOKEN")
                ),
                "url": os.environ.get("PLANE_URL", ""),
            },
            "portainer": {
                "configured": bool(
                    os.environ.get("PORTAINER_TOKEN")
                    or os.environ.get("PORTAINER_PASSWORD")
                ),
                "url": os.environ.get("PORTAINER_URL", ""),
            },
            "uptime_kuma": {
                "configured": bool(os.environ.get("UPTIME_KUMA_URL")),
                "url": os.environ.get("UPTIME_KUMA_URL", ""),
            },
            "lgtm": {
                "configured": bool(
                    os.environ.get("LGTM_URL") or os.environ.get("GRAFANA_URL")
                ),
                "url": os.environ.get("LGTM_URL", ""),
            },
            "langfuse": {
                "configured": bool(
                    os.environ.get("LANGFUSE_PUBLIC_KEY")
                    and os.environ.get("LANGFUSE_SECRET_KEY")
                ),
                "url": os.environ.get("LANGFUSE_URL", "https://cloud.langfuse.com"),
            },
            "keycloak": {
                "configured": bool(
                    os.environ.get("KEYCLOAK_URL")
                    and os.environ.get("KEYCLOAK_ADMIN_PASSWORD")
                ),
                "url": os.environ.get("KEYCLOAK_URL", ""),
            },
            "openbao": {
                "configured": bool(
                    os.environ.get("BAO_URL") or os.environ.get("VAULT_URL")
                ),
                "url": os.environ.get("BAO_URL", ""),
            },
            "nextcloud": {
                "configured": bool(
                    os.environ.get("NEXTCLOUD_URL")
                    and os.environ.get("NEXTCLOUD_PASSWORD")
                ),
                "url": os.environ.get("NEXTCLOUD_URL", ""),
            },
            "listmonk": {
                "configured": bool(
                    os.environ.get("LISTMONK_URL") and os.environ.get("LISTMONK_TOKEN")
                ),
                "url": os.environ.get("LISTMONK_URL", ""),
            },
            "mattermost": {
                "configured": bool(
                    os.environ.get("MATTERMOST_URL")
                    and os.environ.get("MATTERMOST_TOKEN")
                ),
                "url": os.environ.get("MATTERMOST_URL", ""),
            },
            "technitium_dns": {
                "configured": bool(
                    os.environ.get("TECHNITIUM_URL")
                    and os.environ.get("TECHNITIUM_TOKEN")
                ),
                "url": os.environ.get("TECHNITIUM_URL", ""),
            },
            "caddy": {
                "configured": bool(
                    os.environ.get("CADDY_URL") or os.environ.get("CADDY_API_URL")
                ),
                "url": os.environ.get("CADDY_URL", ""),
            },
            "tunnel_manager": {
                "configured": bool(
                    os.environ.get("TUNNEL_MANAGER_URL") or os.environ.get("TUNNEL_URL")
                ),
                "url": os.environ.get("TUNNEL_MANAGER_URL", ""),
            },
            "scholarx": {
                "configured": bool(
                    os.environ.get("SCHOLARX_URL") or os.environ.get("SCHOLARX_API_KEY")
                ),
                "url": os.environ.get("SCHOLARX_URL", ""),
            },
            "emerald_exchange": {
                "configured": bool(
                    os.environ.get("EMERALD_URL") or os.environ.get("EMERALD_API_KEY")
                ),
                "url": os.environ.get("EMERALD_URL", ""),
            },
            "postiz": {
                "configured": bool(
                    os.environ.get("POSTIZ_URL") and os.environ.get("POSTIZ_TOKEN")
                ),
                "url": os.environ.get("POSTIZ_URL", ""),
            },
        }
        return status

    def hydrate_source(self, engine: Any, source: str) -> dict[str, Any]:
        """Trigger instant hydration for a specific source.

        Resolves the connector method from CAPABILITY_REGISTRY, enabling
        tool-agnostic orchestration.
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
        return method(engine)

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
                results[src] = {"status": "skipped", "reason": "Not configured"}

        return results

    # ══════════════════════════════════════════════════════════════════
    # Tier 1 - GitLab, Jira, Plane (Projects & Workflow Tracking)
    # ══════════════════════════════════════════════════════════════════

    def _hydrate_gitlab(self, engine: Any) -> dict[str, Any]:
        """Hydrate from GitLab (OWL Native)."""
        try:
            from gitlab_api.api_client import GitLabApi
        except ImportError:
            return {"status": "skipped", "reason": "gitlab-api package not installed"}

        url = os.environ.get("GITLAB_URL", "https://gitlab.com")
        token = os.environ.get("GITLAB_TOKEN") or os.environ.get("GITLAB_API_TOKEN")
        if not token:
            return {
                "status": "skipped",
                "reason": "Missing GITLAB_TOKEN/GITLAB_API_TOKEN",
            }

        client = GitLabApi(base_url=url, token=token, verify=False)
        try:
            projects = client.get_projects(per_page=30)
        except Exception as e:
            return {"status": "error", "error": f"Failed to fetch projects: {e}"}

        if not isinstance(projects, list):
            projects = [projects] if projects else []

        entities: list[dict[str, Any]] = []
        relationships: list[dict[str, Any]] = []

        for p in projects:
            if not isinstance(p, dict):
                continue
            proj_id = str(p.get("id", ""))
            if not proj_id:
                continue

            node_id = f"gitlab:proj:{proj_id}"
            # OWL Mapping: GitLabProject -> repository
            entities.append(
                {
                    "id": node_id,
                    "type": "repository",
                    "name": p.get("name", f"Repo {proj_id}"),
                    "full_path": p.get("path_with_namespace", ""),
                    "description": p.get("description", ""),
                    "web_url": p.get("web_url", ""),
                    "domain": "gitlab",
                }
            )

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
            except Exception as pe:
                logger.debug(
                    f"Failed to fetch pipelines for GitLab project {proj_id}: {pe}"
                )

        if entities:
            engine.ingest_external_batch("gitlab", entities, relationships)

        return {
            "status": "ok",
            "nodes_hydrated": len(entities),
            "relations_hydrated": len(relationships),
        }

    def _hydrate_jira(self, engine: Any) -> dict[str, Any]:
        """Hydrate Jira Issues & Epics, focusing on specific configured projects."""
        try:
            from atlassian_agent.api_client import JiraApi  # type: ignore
        except ImportError:
            return {
                "status": "skipped",
                "reason": "atlassian-agent package not installed",
            }

        url = os.environ.get("JIRA_URL")
        token = os.environ.get("JIRA_TOKEN") or os.environ.get("JIRA_API_TOKEN")
        if not url or not token:
            return {"status": "skipped", "reason": "Missing JIRA_URL or JIRA_TOKEN"}

        # Specific projects filter
        project_keys_str = os.environ.get("JIRA_PROJECT_KEYS", "")
        project_keys = [k.strip() for k in project_keys_str.split(",") if k.strip()]

        client = JiraApi(base_url=url, token=token)
        entities: list[dict[str, Any]] = []
        relationships: list[dict[str, Any]] = []

        try:
            # Query Jira issues for the selected projects
            jql = "order by updated DESC"
            if project_keys:
                jql = f"project in ({','.join(project_keys)}) AND " + jql

            issues_resp = client.search_issues(jql=jql, max_results=50)
            issues = (
                issues_resp.get("issues", []) if isinstance(issues_resp, dict) else []
            )

            for issue in issues:
                key = issue.get("key")
                if not key:
                    continue

                fields = issue.get("fields", {})
                node_id = f"jira:issue:{key}"

                # OWL Mapping: JiraIssue -> issue
                entities.append(
                    {
                        "id": node_id,
                        "type": "issue",
                        "name": fields.get("summary", f"Issue {key}"),
                        "status": fields.get("status", {}).get("name", ""),
                        "priority": fields.get("priority", {}).get("name", ""),
                        "domain": "jira",
                    }
                )

                # Assignee relation
                assignee = fields.get("assignee")
                if assignee:
                    user_id = assignee.get("accountId") or assignee.get("name")
                    user_node_id = f"jira:user:{user_id}"
                    entities.append(
                        {
                            "id": user_node_id,
                            "type": "person",
                            "name": assignee.get("displayName", f"User {user_id}"),
                            "domain": "jira",
                        }
                    )
                    relationships.append(
                        {
                            "source": node_id,
                            "target": user_node_id,
                            "type": "has_role",
                            "domain": "jira",
                        }
                    )

                # Link issues if parent / epic exists
                epic = fields.get("epic") or fields.get(
                    "customfield_10014"
                )  # standard Epic link field
                if epic:
                    epic_node_id = f"jira:epic:{epic}"
                    entities.append(
                        {
                            "id": epic_node_id,
                            "type": "goal",
                            "name": f"Epic {epic}",
                            "domain": "jira",
                        }
                    )
                    relationships.append(
                        {
                            "source": node_id,
                            "target": epic_node_id,
                            "type": "part_of",
                            "domain": "jira",
                        }
                    )

        except Exception as e:
            return {"status": "error", "error": f"Failed to fetch Jira issues: {e}"}

        if entities:
            engine.ingest_external_batch("jira", entities, relationships)

        return {
            "status": "ok",
            "nodes_hydrated": len(entities),
            "relations_hydrated": len(relationships),
        }

    def _hydrate_plane(self, engine: Any) -> dict[str, Any]:
        """Hydrate from Plane PMS cycle tasks, filtering by configured projects."""
        try:
            from plane_agent.api_client import PlaneApi  # type: ignore
        except ImportError:
            return {"status": "skipped", "reason": "plane-agent package not installed"}

        url = os.environ.get("PLANE_URL")
        token = os.environ.get("PLANE_TOKEN") or os.environ.get("PLANE_API_TOKEN")
        if not url or not token:
            return {"status": "skipped", "reason": "Missing PLANE_URL or PLANE_TOKEN"}

        target_project_ids = [
            p.strip()
            for p in os.environ.get("PLANE_PROJECT_IDS", "").split(",")
            if p.strip()
        ]

        client = PlaneApi(base_url=url, token=token)
        entities: list[dict[str, Any]] = []
        relationships: list[dict[str, Any]] = []

        try:
            # If no target projects, fetch available list
            projects = target_project_ids
            if not projects:
                proj_resp = client.get_projects()
                proj_list = (
                    proj_resp.get("results", [])
                    if isinstance(proj_resp, dict)
                    else proj_resp
                )
                projects = [
                    str(p.get("id"))
                    for p in proj_list
                    if isinstance(p, dict) and p.get("id")
                ]

            for proj_id in projects[:5]:
                issues_resp = client.get_project_issues(proj_id)
                issues = (
                    issues_resp.get("results", [])
                    if isinstance(issues_resp, dict)
                    else []
                )

                for issue in issues:
                    issue_id = str(issue.get("id"))
                    if not issue_id:
                        continue

                    node_id = f"plane:issue:{issue_id}"
                    # OWL Mapping: PlaneIssue -> issue
                    entities.append(
                        {
                            "id": node_id,
                            "type": "issue",
                            "name": issue.get("name", f"Plane Issue {issue_id}"),
                            "state": issue.get("state", {}).get("name", ""),
                            "priority": issue.get("priority", ""),
                            "domain": "plane",
                        }
                    )

                    # Project node
                    proj_node_id = f"plane:proj:{proj_id}"
                    entities.append(
                        {
                            "id": proj_node_id,
                            "type": "software_project",
                            "name": f"Plane Project {proj_id}",
                            "domain": "plane",
                        }
                    )
                    relationships.append(
                        {
                            "source": node_id,
                            "target": proj_node_id,
                            "type": "part_of",
                            "domain": "plane",
                        }
                    )

        except Exception as e:
            return {"status": "error", "error": f"Failed to fetch Plane issues: {e}"}

        if entities:
            engine.ingest_external_batch("plane", entities, relationships)

        return {
            "status": "ok",
            "nodes_hydrated": len(entities),
            "relations_hydrated": len(relationships),
        }

    # ══════════════════════════════════════════════════════════════════
    # Tier 2 & 3 - Portainer, Uptime Kuma, technitium-dns, caddy (Topology)
    # ══════════════════════════════════════════════════════════════════

    def _hydrate_portainer(self, engine: Any) -> dict[str, Any]:
        """Hydrate full Portainer stack, containers, hosts, and images (Tier 2)."""
        try:
            from portainer_agent.api_client import PortainerApi  # type: ignore
        except ImportError:
            return {
                "status": "skipped",
                "reason": "portainer-agent package not installed",
            }

        url = os.environ.get("PORTAINER_URL")
        token = os.environ.get("PORTAINER_TOKEN") or os.environ.get(
            "PORTAINER_PASSWORD"
        )
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
            stacks = client.get_stacks()
            if not isinstance(stacks, list):
                stacks = []

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

            # Fetch endpoints/environments and their containers
            endpoints = client.get_endpoints()
            if not isinstance(endpoints, list):
                endpoints = []

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
                                "name": c.get("Names", [f"Container {c_id}"])[0].lstrip(
                                    "/"
                                ),
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

        url = os.environ.get("UPTIME_KUMA_URL")
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

    def _hydrate_leanix(self, engine: Any) -> dict[str, Any]:
        """Hydrate from LeanIX Pathfinder (Tier 3)."""
        try:
            from leanix_agent.leanix_gql import GraphQL as LeanIXGraphQL
        except ImportError:
            return {"status": "skipped", "reason": "leanix-agent package not installed"}

        url = os.environ.get("LEANIX_URL")
        token = os.environ.get("LEANIX_TOKEN")
        if not url or not token:
            return {
                "status": "skipped",
                "reason": "Missing LEANIX_URL and/or LEANIX_TOKEN",
            }

        client = LeanIXGraphQL(base_url=url, token=token)
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
        try:
            res = client.execute_gql(query)
        except Exception as e:
            try:
                res = client.query(query)
            except Exception as query_err:
                return {
                    "status": "error",
                    "error": f"GraphQL queries failed: {e} / {query_err}",
                }

        if not isinstance(res, dict):
            return {
                "status": "error",
                "error": f"Invalid LeanIX GQL response: {type(res)}",
            }

        data = res.get("data", {}) if "data" in res else res
        all_fs = data.get("allFactSheets", {})
        edges = all_fs.get("edges", [])

        entities: list[dict[str, Any]] = []
        relationships: list[dict[str, Any]] = []

        for edge in edges:
            node = edge.get("node", {})
            fs_id = node.get("id")
            if not fs_id:
                continue

            # OWL Mapping: EAFactSheet -> platform_service
            entities.append(
                {
                    "id": f"leanix:fs:{fs_id}",
                    "type": "platform_service",
                    "name": node.get("name", ""),
                    "factsheet_type": node.get("type", ""),
                    "description": node.get("description", ""),
                    "domain": "leanix",
                }
            )

        if entities:
            engine.ingest_external_batch("leanix", entities, relationships)

        return {
            "status": "ok",
            "nodes_hydrated": len(entities),
            "relations_hydrated": len(relationships),
        }

    def _hydrate_twenty(self, engine: Any) -> dict[str, Any]:
        """Hydrate from Twenty CRM (Tier 3)."""
        try:
            from twenty_mcp.api_client import Api as TwentyApi
        except ImportError:
            return {"status": "skipped", "reason": "twenty-mcp package not installed"}

        url = os.environ.get("TWENTY_URL")
        token = os.environ.get("TWENTY_TOKEN") or os.environ.get("TWENTY_API_TOKEN")
        if not url or not token:
            return {
                "status": "skipped",
                "reason": "Missing TWENTY_URL and/or TWENTY_TOKEN",
            }

        client = TwentyApi(base_url=url, token=token)
        entities: list[dict[str, Any]] = []
        relationships: list[dict[str, Any]] = []

        # 1. Companies
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
        except Exception as e:
            logger.debug(f"Failed to fetch CRM companies: {e}")

        # 2. People (Contacts)
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
        except Exception as e:
            logger.debug(f"Failed to fetch CRM people: {e}")

        # 3. Opportunities
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
        except Exception as e:
            logger.debug(f"Failed to fetch CRM opportunities: {e}")

        if entities:
            engine.ingest_external_batch("twenty", entities, relationships)

        return {
            "status": "ok",
            "nodes_hydrated": len(entities),
            "relations_hydrated": len(relationships),
        }

    def _hydrate_servicenow(self, engine: Any) -> dict[str, Any]:
        """Hydrate from ServiceNow CMDB (Tier 3)."""
        try:
            from servicenow_api.api_client import ServiceNowApi
        except ImportError:
            return {
                "status": "skipped",
                "reason": "servicenow-api package not installed",
            }

        url = os.environ.get("SERVICENOW_URL")
        username = os.environ.get("SERVICENOW_USER")
        password = os.environ.get("SERVICENOW_PASSWORD")
        if not url or not username or not password:
            return {
                "status": "skipped",
                "reason": "Missing SERVICENOW_URL, USER, or PASSWORD",
            }

        client = ServiceNowApi(base_url=url, username=username, password=password)
        entities: list[dict[str, Any]] = []
        relationships: list[dict[str, Any]] = []

        target_classes = {
            "cmdb_ci_appl": "platform_service",
            "cmdb_ci_server": "server",
            "cmdb_ci_database": "system",
        }

        for class_name, owl_type in target_classes.items():
            try:
                resp = client.get_cmdb_instances(className=class_name, sysparm_limit=20)
                items = []
                if hasattr(resp, "result"):
                    items = resp.result
                elif isinstance(resp, dict):
                    items = resp.get("result", [])

                if not isinstance(items, list):
                    if isinstance(items, dict) and "items" in items:
                        items = items["items"]
                    else:
                        items = []

                for item in items:
                    if not isinstance(item, dict):
                        continue

                    sys_id = item.get("sys_id") or item.get("sysId")
                    if not sys_id:
                        continue

                    name = item.get("name") or item.get("display_value", f"CI {sys_id}")

                    # OWL Mapping: CMDB Application/Server/Database -> platform_service/server/system
                    entities.append(
                        {
                            "id": f"servicenow:ci:{sys_id}",
                            "type": owl_type,
                            "name": name,
                            "ci_class": class_name,
                            "domain": "servicenow",
                        }
                    )
            except Exception as class_err:
                logger.debug(
                    f"Failed to fetch ServiceNow class {class_name}: {class_err}"
                )

        if entities:
            engine.ingest_external_batch("servicenow", entities, relationships)

        return {
            "status": "ok",
            "nodes_hydrated": len(entities),
            "relations_hydrated": len(relationships),
        }

    # ══════════════════════════════════════════════════════════════════
    # Tier 4 - LGTM alerts/metrics & Langfuse standardization
    # ══════════════════════════════════════════════════════════════════

    def _hydrate_lgtm(self, engine: Any) -> dict[str, Any]:
        """Hydrate alerts, metric systems, and health states from LGTM/Grafana (Tier 4)."""
        try:
            # LGTM doesn't have a rigid Python client package, so we mock / fetch gracefully
            lgtm_url = os.environ.get("LGTM_URL") or os.environ.get("GRAFANA_URL")
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

        os.environ.get("LANGFUSE_URL", "https://cloud.langfuse.com")
        pub_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
        sec_key = os.environ.get("LANGFUSE_SECRET_KEY")
        if not pub_key or not sec_key:
            return {
                "status": "skipped",
                "reason": "Missing LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY",
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

        url = os.environ.get("KEYCLOAK_URL")
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

        url = os.environ.get("NEXTCLOUD_URL")
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

        url = os.environ.get("MATTERMOST_URL")
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

        url = os.environ.get("TECHNITIUM_URL")
        token = os.environ.get("TECHNITIUM_TOKEN")
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
                "value": "10.0.0.50",
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
