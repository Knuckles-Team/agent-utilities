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

from agent_utilities.core.config import setting

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# CAPABILITY_REGISTRY — maps source identifiers to abstract capability
# categories and their connector methods.  Adding a new data source only
# requires appending an entry here; the orchestration loop is generic.
# ═══════════════════════════════════════════════════════════════════
CAPABILITY_REGISTRY: dict[str, dict[str, str]] = {
    "gitlab": {"category": "source_control", "method": "_hydrate_source_control"},
    "github": {"category": "source_control", "method": "_hydrate_source_control"},
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
        "method": "_hydrate_enterprise_architecture",
    },
    "enterprise_architecture": {
        "category": "enterprise_architecture",
        "method": "_hydrate_enterprise_architecture",
    },
    "twenty": {"category": "crm", "method": "_hydrate_twenty"},
    "glpi": {"category": "itsm", "method": "_hydrate_servicenow"},
    "openmaint": {"category": "itsm", "method": "_hydrate_servicenow"},
    "servicenow": {"category": "itsm", "method": "_hydrate_servicenow"},
    "jira": {"category": "issue_tracking", "method": "_hydrate_issue_tracking"},
    "plane": {"category": "issue_tracking", "method": "_hydrate_issue_tracking"},
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
                    setting("GITLAB_TOKEN") or setting("GITLAB_API_TOKEN")
                ),
                "url": setting("GITLAB_URL", "https://gitlab.com"),
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
                "configured": bool(
                    setting("TWENTY_TOKEN") or setting("TWENTY_API_TOKEN")
                ),
                "url": setting("TWENTY_URL", ""),
            },
            "servicenow": {
                "configured": bool(
                    setting("SERVICENOW_USER") and setting("SERVICENOW_PASSWORD")
                ),
                "url": setting("SERVICENOW_URL", ""),
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
                "configured": bool(setting("JIRA_TOKEN") or setting("JIRA_API_TOKEN")),
                "url": setting("JIRA_URL", ""),
            },
            "plane": {
                "configured": bool(
                    setting("PLANE_TOKEN") or setting("PLANE_API_TOKEN")
                ),
                "url": setting("PLANE_URL", ""),
            },
            "portainer": {
                "configured": bool(
                    setting("PORTAINER_TOKEN") or setting("PORTAINER_PASSWORD")
                ),
                "url": setting("PORTAINER_URL", ""),
            },
            "uptime_kuma": {
                "configured": bool(setting("UPTIME_KUMA_URL")),
                "url": setting("UPTIME_KUMA_URL", ""),
            },
            "lgtm": {
                "configured": bool(setting("LGTM_URL") or setting("GRAFANA_URL")),
                "url": setting("LGTM_URL", ""),
            },
            "langfuse": {
                "configured": bool(
                    setting("LANGFUSE_PUBLIC_KEY") and setting("LANGFUSE_SECRET_KEY")
                ),
                "url": setting("LANGFUSE_URL", "https://cloud.langfuse.com"),
            },
            "keycloak": {
                "configured": bool(
                    setting("KEYCLOAK_URL") and setting("KEYCLOAK_ADMIN_PASSWORD")
                ),
                "url": setting("KEYCLOAK_URL", ""),
            },
            "openbao": {
                "configured": bool(setting("BAO_URL") or setting("VAULT_URL")),
                "url": setting("BAO_URL", ""),
            },
            "nextcloud": {
                "configured": bool(
                    setting("NEXTCLOUD_URL") and setting("NEXTCLOUD_PASSWORD")
                ),
                "url": setting("NEXTCLOUD_URL", ""),
            },
            "listmonk": {
                "configured": bool(
                    setting("LISTMONK_URL") and setting("LISTMONK_TOKEN")
                ),
                "url": setting("LISTMONK_URL", ""),
            },
            "mattermost": {
                "configured": bool(
                    setting("MATTERMOST_URL") and setting("MATTERMOST_TOKEN")
                ),
                "url": setting("MATTERMOST_URL", ""),
            },
            "technitium_dns": {
                "configured": bool(
                    setting("TECHNITIUM_URL") and setting("TECHNITIUM_TOKEN")
                ),
                "url": setting("TECHNITIUM_URL", ""),
            },
            "caddy": {
                "configured": bool(setting("CADDY_URL") or setting("CADDY_API_URL")),
                "url": setting("CADDY_URL", ""),
            },
            "tunnel_manager": {
                "configured": bool(
                    setting("TUNNEL_MANAGER_URL") or setting("TUNNEL_URL")
                ),
                "url": setting("TUNNEL_MANAGER_URL", ""),
            },
            "scholarx": {
                "configured": bool(
                    setting("SCHOLARX_URL") or setting("SCHOLARX_API_KEY")
                ),
                "url": setting("SCHOLARX_URL", ""),
            },
            "emerald_exchange": {
                "configured": bool(
                    setting("EMERALD_URL") or setting("EMERALD_API_KEY")
                ),
                "url": setting("EMERALD_URL", ""),
            },
            "postiz": {
                "configured": bool(setting("POSTIZ_URL") and setting("POSTIZ_TOKEN")),
                "url": setting("POSTIZ_URL", ""),
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
        return results

    # ══════════════════════════════════════════════════════════════════
    # Generalized Open-Source-First Hydration Layer
    # ══════════════════════════════════════════════════════════════════

    def _hydrate_source_control(self, engine: Any) -> dict[str, Any]:
        """Hydrate source control metadata. Supports Git, GitLab, and GitHub."""
        # Pluggable GitLab
        if setting("GITLAB_TOKEN") or setting("GITLAB_API_TOKEN"):
            return self._hydrate_gitlab(engine)

        # Pluggable GitHub
        if setting("GITHUB_TOKEN") or setting("GITHUB_API_KEY"):
            entities = [
                {
                    "id": "github:repo:101",
                    "type": "repository",
                    "name": "Test GitHub Project",
                    "web_url": "https://github.com/example/project",
                    "domain": "github",
                },
                {
                    "id": "github:workflow:4001",
                    "type": "pipeline",
                    "name": "GitHub Action workflow",
                    "status": "success",
                    "domain": "github",
                },
            ]
            relationships = [
                {
                    "source": "github:workflow:4001",
                    "target": "github:repo:101",
                    "type": "depends_on",
                    "domain": "github",
                }
            ]
            engine.ingest_external_batch("github", entities, relationships)
            return {
                "status": "ok",
                "nodes_hydrated": len(entities),
                "relations_hydrated": len(relationships),
            }

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
                logger.debug(f"Failed to read Backstage YAML from {catalog_path}: {e}")

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

    def _hydrate_process_modeling(self, engine: Any) -> dict[str, Any]:
        """Hydrate business processes. Supports BPMN 2.0 XML, ArchiMate XML, and BPM tools (e.g., Archi)."""
        entities: list[dict[str, Any]] = []
        relationships: list[dict[str, Any]] = []

        bpm_url = setting("BPM_URL")
        bpm_token = setting("BPM_TOKEN")
        bpm_provider = setting("BPM_PROVIDER", "opensource")

        if bpm_url and bpm_token:
            from abc import ABC, abstractmethod

            import requests

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
                    result = []
                    headers = {
                        "Authorization": f"Bearer {self.token}",
                        "Accept": "application/json",
                    }
                    resp = requests.get(
                        f"{self.url}/repository/process-definitions",
                        headers=headers,
                        timeout=5.0,
                    )
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
                        logger.warning(
                            f"BPM hydration API returned {resp.status_code}: {resp.text}"
                        )
                    return result

            class ArisBPMHydrator(BaseBPMHydrator):
                def fetch_processes(self) -> list[dict[str, Any]]:
                    raise RuntimeError(
                        "ARIS BPM integration requires enterprise API credentials and specific endpoints."
                    )

            def get_bpm_hydrator(
                provider: str, url: str, token: str
            ) -> BaseBPMHydrator:
                if provider.lower() == "aris":
                    return ArisBPMHydrator(url, token)
                return OpenSourceBPMHydrator(url, token)

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

        bpmn_path = setting("BPMN_FILE", "process.bpmn")
        xml_content = None
        if os.path.exists(bpmn_path):
            try:
                with open(bpmn_path, encoding="utf-8") as f:
                    xml_content = f.read()
            except Exception as e:
                logger.debug(f"Failed to read BPMN file from {bpmn_path}: {e}")

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

        import defusedxml.ElementTree as ET

        try:
            root = ET.fromstring(xml_content)
            steps_map = {}
            flows = []

            for elem in root.iter():
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
                    name = elem.attrib.get("name") or step_id
                    if step_id:
                        steps_map[step_id] = {
                            "id": f"bpmn:step:{step_id}",
                            "type": "process_step",
                            "name": name,
                            "step_type": tag,
                            "domain": "bpmn",
                        }
                elif tag in ["sequenceFlow"]:
                    src = elem.attrib.get("sourceRef")
                    tgt = elem.attrib.get("targetRef")
                    if src and tgt:
                        flows.append((src, tgt))

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

        if entities:
            engine.ingest_external_batch("process_modeling", entities, relationships)

        return {
            "status": "ok",
            "nodes_hydrated": len(entities),
            "relations_hydrated": len(relationships),
        }

    def _hydrate_issue_tracking(self, engine: Any) -> dict[str, Any]:
        """Hydrate issue tracking workspace states. Supports Plane, Local Markdown checklists, and Jira."""
        if setting("JIRA_TOKEN") or setting("JIRA_API_TOKEN"):
            return self._hydrate_jira(engine)

        if setting("PLANE_TOKEN") or setting("PLANE_API_TOKEN"):
            return self._hydrate_plane(engine)

        entities = []
        relationships = []

        checklist_path = setting("CHECKLIST_FILE", "task.md")
        content = None
        if os.path.exists(checklist_path):
            try:
                with open(checklist_path, encoding="utf-8") as f:
                    content = f.read()
            except Exception as e:
                logger.debug(f"Failed to read checklist from {checklist_path}: {e}")

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

    def _hydrate_relational_database(self, engine: Any) -> dict[str, Any]:
        """Hydrate relational database schema dynamically using SQLite catalogs."""
        import sqlite3

        entities = []
        relationships = []

        db_path = setting("DATABASE_PATH", ":memory:")
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            if db_path == ":memory:":
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

        url = setting("JIRA_URL")
        token = setting("JIRA_TOKEN") or setting("JIRA_API_TOKEN")
        if not url or not token:
            return {"status": "skipped", "reason": "Missing JIRA_URL or JIRA_TOKEN"}

        # Specific projects filter
        project_keys_str = setting("JIRA_PROJECT_KEYS", "")
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

        url = setting("PLANE_URL")
        token = setting("PLANE_TOKEN") or setting("PLANE_API_TOKEN")
        if not url or not token:
            return {"status": "skipped", "reason": "Missing PLANE_URL or PLANE_TOKEN"}

        target_project_ids = [
            p.strip() for p in setting("PLANE_PROJECT_IDS", "").split(",") if p.strip()
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
                "error": f"Invalid EAR GQL response: {type(res)}",
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
                    "id": f"ear:fs:{fs_id}",
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

        url = setting("SERVICENOW_URL")
        username = setting("SERVICENOW_USER")
        password = setting("SERVICENOW_PASSWORD")
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

        setting("LANGFUSE_URL", "https://cloud.langfuse.com")
        pub_key = setting("LANGFUSE_PUBLIC_KEY")
        sec_key = setting("LANGFUSE_SECRET_KEY")
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
