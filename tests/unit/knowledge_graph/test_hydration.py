"""Unit tests for Knowledge Graph Hydration service."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from agent_utilities.knowledge_graph.core.hydration import HydrationManager


@pytest.fixture
def mock_engine():
    engine = MagicMock()
    return engine


@patch.dict(
    os.environ,
    {"GITLAB_TOKEN": "test-gitlab-token", "GITLAB_URL": "https://gitlab.example.com"},
)
def test_hydrate_gitlab(mock_engine):
    mock_gitlab_api_class = MagicMock()
    mock_client = MagicMock()
    mock_gitlab_api_class.return_value = mock_client

    mock_client.get_projects.return_value = [
        {
            "id": 101,
            "name": "Test Project 1",
            "path_with_namespace": "group/test-project-1",
            "description": "A description of project 1",
            "web_url": "https://gitlab.example.com/group/test-project-1",
        }
    ]

    mock_client.get_pipelines.return_value = [
        {
            "id": 4001,
            "status": "success",
            "ref": "main",
            "sha": "abcdef123456",
            "web_url": "https://gitlab.example.com/group/test-project-1/pipelines/4001",
        }
    ]

    modules = {
        "gitlab_api": MagicMock(),
        "gitlab_api.api_client": MagicMock(),
    }
    with patch.dict("sys.modules", modules):
        modules["gitlab_api.api_client"].GitLabApi = mock_gitlab_api_class

        manager = HydrationManager()
        res = manager.hydrate_source(mock_engine, "gitlab")

        assert res["status"] == "ok"
        assert res["nodes_hydrated"] == 2  # 1 project + 1 pipeline
        assert res["relations_hydrated"] == 1

        mock_engine.ingest_external_batch.assert_called_once()
        args, kwargs = mock_engine.ingest_external_batch.call_args
        assert args[0] == "gitlab"
        entities = args[1]
        relationships = args[2]

        assert entities[0]["id"] == "gitlab:proj:101"
        assert entities[0]["type"] == "repository"  # OWL Native
        assert entities[0]["name"] == "Test Project 1"

        assert entities[1]["id"] == "gitlab:pipeline:4001"
        assert entities[1]["type"] == "pipeline"  # OWL Native

        assert relationships[0]["source"] == "gitlab:pipeline:4001"
        assert relationships[0]["target"] == "gitlab:proj:101"
        assert relationships[0]["type"] == "depends_on"  # OWL Native


@patch.dict(
    os.environ,
    {"LEANIX_URL": "https://leanix.example.com", "LEANIX_TOKEN": "test-leanix-token"},
)
def test_hydrate_leanix(mock_engine):
    mock_leanix_gql_class = MagicMock()
    mock_client = MagicMock()
    mock_leanix_gql_class.return_value = mock_client

    mock_client.execute_gql.return_value = {
        "data": {
            "allFactSheets": {
                "edges": [
                    {
                        "node": {
                            "id": "fs-id-99",
                            "name": "Enterprise Core ERP",
                            "type": "Application",
                            "description": "Core ERP Application",
                        }
                    }
                ]
            }
        }
    }

    modules = {
        "leanix_agent": MagicMock(),
        "leanix_agent.leanix_gql": MagicMock(),
    }
    with patch.dict("sys.modules", modules):
        modules["leanix_agent.leanix_gql"].GraphQL = mock_leanix_gql_class

        manager = HydrationManager()
        res = manager.hydrate_source(mock_engine, "leanix")

        assert res["status"] == "ok"
        assert res["nodes_hydrated"] == 1

        mock_engine.ingest_external_batch.assert_called_once()
        args, _ = mock_engine.ingest_external_batch.call_args
        assert args[0] == "leanix"
        assert args[1][0]["id"] == "leanix:fs:fs-id-99"
        assert args[1][0]["type"] == "platform_service"  # OWL Native
        assert args[1][0]["name"] == "Enterprise Core ERP"


@patch.dict(
    os.environ,
    {"TWENTY_URL": "https://twenty.example.com", "TWENTY_TOKEN": "test-twenty-token"},
)
def test_hydrate_twenty(mock_engine):
    mock_twenty_api_class = MagicMock()
    mock_client = MagicMock()
    mock_twenty_api_class.return_value = mock_client

    mock_client.get_companies.return_value = [
        {"id": "comp-1", "name": "Google", "domain": "google.com", "employees": 150000}
    ]
    mock_client.get_people.return_value = [
        {
            "id": "person-1",
            "firstName": "Sundar",
            "lastName": "Pichai",
            "email": "sundar@google.com",
            "companyId": "comp-1",
        }
    ]
    mock_client.get_opportunities.return_value = [
        {
            "id": "opp-1",
            "name": "Cloud Deal",
            "amount": 10000000,
            "stage": "Negotiation",
            "companyId": "comp-1",
        }
    ]

    modules = {
        "twenty_mcp": MagicMock(),
        "twenty_mcp.api_client": MagicMock(),
    }
    with patch.dict("sys.modules", modules):
        modules["twenty_mcp.api_client"].Api = mock_twenty_api_class

        manager = HydrationManager()
        res = manager.hydrate_source(mock_engine, "twenty")

        assert res["status"] == "ok"
        assert res["nodes_hydrated"] == 3
        assert res["relations_hydrated"] == 2

        mock_engine.ingest_external_batch.assert_called_once()
        args, _ = mock_engine.ingest_external_batch.call_args
        assert args[0] == "twenty"
        entities = args[1]
        relationships = args[2]

        assert entities[0]["type"] == "organization"  # OWL Native
        assert entities[1]["type"] == "person"  # OWL Native
        assert entities[2]["type"] == "opportunity"  # OWL Native

        assert relationships[0]["type"] == "works_at"  # OWL Native
        assert relationships[1]["type"] == "related_to"  # OWL Native


@patch.dict(
    os.environ,
    {
        "SERVICENOW_URL": "https://servicenow.example.com",
        "SERVICENOW_USER": "admin",
        "SERVICENOW_PASSWORD": "password",
    },
)
def test_hydrate_servicenow(mock_engine):
    mock_servicenow_api_class = MagicMock()
    mock_client = MagicMock()
    mock_servicenow_api_class.return_value = mock_client

    mock_response = MagicMock()
    mock_response.result = [
        {
            "sys_id": "sys-id-appl-99",
            "name": "GeniusBot Portal",
            "display_value": "GeniusBot Portal",
        }
    ]
    mock_client.get_cmdb_instances.return_value = mock_response

    modules = {
        "servicenow_api": MagicMock(),
        "servicenow_api.api_client": MagicMock(),
    }
    with patch.dict("sys.modules", modules):
        modules["servicenow_api.api_client"].ServiceNowApi = mock_servicenow_api_class

        manager = HydrationManager()
        res = manager.hydrate_source(mock_engine, "servicenow")

        assert res["status"] == "ok"
        assert res["nodes_hydrated"] == 3

        mock_engine.ingest_external_batch.assert_called_once()
        args, _ = mock_engine.ingest_external_batch.call_args
        assert args[0] == "servicenow"
        assert args[1][0]["id"] == "servicenow:ci:sys-id-appl-99"
        assert args[1][0]["type"] == "platform_service"  # OWL Native


@patch.dict(
    os.environ,
    {
        "JIRA_URL": "https://jira.example.com",
        "JIRA_TOKEN": "jira-tok",
        "JIRA_PROJECT_KEYS": "PROJ",
    },
)
def test_hydrate_jira(mock_engine):
    mock_jira_class = MagicMock()
    mock_client = MagicMock()
    mock_jira_class.return_value = mock_client

    mock_client.search_issues.return_value = {
        "issues": [
            {
                "key": "PROJ-123",
                "fields": {
                    "summary": "Fix broken auth loop",
                    "status": {"name": "In Progress"},
                    "priority": {"name": "High"},
                    "assignee": {"accountId": "user-abc", "displayName": "Alice Smith"},
                },
            }
        ]
    }

    modules = {
        "atlassian_agent": MagicMock(),
        "atlassian_agent.api_client": MagicMock(),
    }
    with patch.dict("sys.modules", modules):
        modules["atlassian_agent.api_client"].JiraApi = mock_jira_class

        manager = HydrationManager()
        res = manager.hydrate_source(mock_engine, "jira")

        assert res["status"] == "ok"
        assert res["nodes_hydrated"] == 2
        assert res["relations_hydrated"] == 1


@patch.dict(
    os.environ,
    {
        "PLANE_URL": "https://plane.example.com",
        "PLANE_TOKEN": "plane-tok",
        "PLANE_PROJECT_IDS": "proj-1",
    },
)
def test_hydrate_plane(mock_engine):
    mock_plane_class = MagicMock()
    mock_client = MagicMock()
    mock_plane_class.return_value = mock_client

    mock_client.get_project_issues.return_value = {
        "results": [
            {
                "id": "issue-xyz",
                "name": "Implement dark mode support",
                "state": {"name": "Backlog"},
                "priority": "medium",
            }
        ]
    }

    modules = {
        "plane_agent": MagicMock(),
        "plane_agent.api_client": MagicMock(),
    }
    with patch.dict("sys.modules", modules):
        modules["plane_agent.api_client"].PlaneApi = mock_plane_class

        manager = HydrationManager()
        res = manager.hydrate_source(mock_engine, "plane")

        assert res["status"] == "ok"
        assert res["nodes_hydrated"] == 2
        assert res["relations_hydrated"] == 1


@patch.dict(
    os.environ,
    {"PORTAINER_URL": "https://portainer.example.com", "PORTAINER_TOKEN": "port-tok"},
)
def test_hydrate_portainer(mock_engine):
    mock_portainer_class = MagicMock()
    mock_client = MagicMock()
    mock_portainer_class.return_value = mock_client

    mock_client.get_stacks.return_value = [{"Id": 1, "Name": "web-stack"}]
    mock_client.get_endpoints.return_value = [
        {"Id": 2, "Name": "host-prod", "URL": "tcp://1.2.3.4"}
    ]
    mock_client.get_endpoint_containers.return_value = [
        {"Id": "abcdef123456", "Names": ["/web-app-container"], "State": "running"}
    ]

    modules = {
        "portainer_agent": MagicMock(),
        "portainer_agent.api_client": MagicMock(),
    }
    with patch.dict("sys.modules", modules):
        modules["portainer_agent.api_client"].PortainerApi = mock_portainer_class

        manager = HydrationManager()
        res = manager.hydrate_source(mock_engine, "portainer")

        assert res["status"] == "ok"
        assert res["nodes_hydrated"] == 3
        assert res["relations_hydrated"] == 1


@patch.dict(os.environ, {"UPTIME_KUMA_URL": "https://kuma.example.com"})
def test_hydrate_uptime_kuma(mock_engine):
    mock_kuma_class = MagicMock()
    mock_client = MagicMock()
    mock_kuma_class.return_value = mock_client

    mock_client.get_monitors.return_value = [
        {"id": 12, "name": "Google DNS", "url": "https://8.8.8.8"}
    ]

    modules = {
        "uptime_kuma_agent": MagicMock(),
        "uptime_kuma_agent.api_client": MagicMock(),
    }
    with patch.dict("sys.modules", modules):
        modules["uptime_kuma_agent.api_client"].KumaApi = mock_kuma_class

        manager = HydrationManager()
        res = manager.hydrate_source(mock_engine, "uptime_kuma")

        assert res["status"] == "ok"
        assert res["nodes_hydrated"] == 1


@patch.dict(os.environ, {"LGTM_URL": "https://lgtm.example.com"})
def test_hydrate_lgtm(mock_engine):
    manager = HydrationManager()
    res = manager.hydrate_source(mock_engine, "lgtm")

    assert res["status"] == "ok"
    assert res["nodes_hydrated"] == 1
    assert res["relations_hydrated"] == 1


@patch.dict(
    os.environ,
    {
        "KEYCLOAK_URL": "https://keycloak.example.com",
        "KEYCLOAK_ADMIN_PASSWORD": "admin",
    },
)
def test_hydrate_keycloak(mock_engine):
    mock_kc_class = MagicMock()
    modules = {
        "keycloak_agent": MagicMock(),
        "keycloak_agent.api_client": MagicMock(),
    }
    with patch.dict("sys.modules", modules):
        modules["keycloak_agent.api_client"].KeycloakAdmin = mock_kc_class

        manager = HydrationManager()
        res = manager.hydrate_source(mock_engine, "keycloak")

        assert res["status"] == "ok"
        assert res["nodes_hydrated"] == 2
        assert res["relations_hydrated"] == 1


@patch.dict(os.environ, {"BAO_URL": "https://bao.example.com"})
def test_hydrate_openbao(mock_engine):
    manager = HydrationManager()
    res = manager.hydrate_source(mock_engine, "openbao")

    assert res["status"] == "ok"
    assert res["nodes_hydrated"] == 1


@patch.dict(
    os.environ,
    {"NEXTCLOUD_URL": "https://nc.example.com", "NEXTCLOUD_PASSWORD": "pass"},
)
def test_hydrate_nextcloud(mock_engine):
    mock_nc_class = MagicMock()
    modules = {
        "nextcloud_agent": MagicMock(),
        "nextcloud_agent.api_client": MagicMock(),
    }
    with patch.dict("sys.modules", modules):
        modules["nextcloud_agent.api_client"].NextcloudClient = mock_nc_class

        manager = HydrationManager()
        res = manager.hydrate_source(mock_engine, "nextcloud")

        assert res["status"] == "ok"
        assert res["nodes_hydrated"] == 2


@patch.dict(
    os.environ, {"LISTMONK_URL": "https://list.example.com", "LISTMONK_TOKEN": "tok"}
)
def test_hydrate_listmonk(mock_engine):
    manager = HydrationManager()
    res = manager.hydrate_source(mock_engine, "listmonk")

    assert res["status"] == "ok"
    assert res["nodes_hydrated"] == 1


@pytest.fixture
def mock_mm_client():
    return MagicMock()


@patch.dict(
    os.environ, {"MATTERMOST_URL": "https://mm.example.com", "MATTERMOST_TOKEN": "tok"}
)
def test_hydrate_mattermost(mock_engine):
    mock_mm_class = MagicMock()
    modules = {
        "mattermost_mcp": MagicMock(),
        "mattermost_mcp.api_client": MagicMock(),
    }
    with patch.dict("sys.modules", modules):
        modules["mattermost_mcp.api_client"].MattermostApi = mock_mm_class

        manager = HydrationManager()
        res = manager.hydrate_source(mock_engine, "mattermost")

        assert res["status"] == "ok"
        assert res["nodes_hydrated"] == 1


@patch.dict(
    os.environ,
    {"TECHNITIUM_URL": "https://dns.example.com", "TECHNITIUM_TOKEN": "dns-tok"},
)
def test_hydrate_technitium_dns(mock_engine):
    mock_dns_class = MagicMock()
    modules = {
        "technitium_dns_mcp": MagicMock(),
        "technitium_dns_mcp.api_client": MagicMock(),
    }
    with patch.dict("sys.modules", modules):
        modules["technitium_dns_mcp.api_client"].Api = mock_dns_class

        manager = HydrationManager()
        res = manager.hydrate_source(mock_engine, "technitium_dns")

        assert res["status"] == "ok"
        assert res["nodes_hydrated"] == 2
        assert res["relations_hydrated"] == 1


@patch.dict(os.environ, {"CADDY_URL": "https://caddy.example.com"})
def test_hydrate_caddy(mock_engine):
    mock_caddy_class = MagicMock()
    modules = {
        "caddy_mcp": MagicMock(),
        "caddy_mcp.api_client": MagicMock(),
    }
    with patch.dict("sys.modules", modules):
        modules["caddy_mcp.api_client"].Api = mock_caddy_class

        manager = HydrationManager()
        res = manager.hydrate_source(mock_engine, "caddy")

        assert res["status"] == "ok"
        assert res["nodes_hydrated"] == 1


def test_hydrate_tunnel_manager(mock_engine):
    manager = HydrationManager()
    res = manager.hydrate_source(mock_engine, "tunnel_manager")

    assert res["status"] == "ok"
    assert res["nodes_hydrated"] == 1


def test_hydrate_scholarx(mock_engine):
    mock_scholarx_class = MagicMock()
    modules = {
        "scholarx": MagicMock(),
        "scholarx.api_client": MagicMock(),
    }
    with patch.dict("sys.modules", modules):
        modules["scholarx.api_client"].ScholarXClient = mock_scholarx_class

        manager = HydrationManager()
        res = manager.hydrate_source(mock_engine, "scholarx")

        assert res["status"] == "ok"
        assert res["nodes_hydrated"] == 2
        assert res["relations_hydrated"] == 1


def test_hydrate_emerald_exchange(mock_engine):
    mock_exchange_class = MagicMock()
    modules = {
        "emerald_exchange": MagicMock(),
        "emerald_exchange.backends": MagicMock(),
    }
    with patch.dict("sys.modules", modules):
        modules["emerald_exchange.backends"].PaperBackend = mock_exchange_class

        manager = HydrationManager()
        res = manager.hydrate_source(mock_engine, "emerald_exchange")

        assert res["status"] == "ok"
        assert res["nodes_hydrated"] == 2
        assert res["relations_hydrated"] == 1


def test_hydrate_postiz(mock_engine):
    manager = HydrationManager()
    res = manager.hydrate_source(mock_engine, "postiz")

    assert res["status"] == "ok"
    assert res["nodes_hydrated"] == 2
    assert res["relations_hydrated"] == 1


@patch.dict(
    os.environ, {"LANGFUSE_PUBLIC_KEY": "pk-123", "LANGFUSE_SECRET_KEY": "sk-123"}
)
def test_hydrate_langfuse(mock_engine):
    mock_lf_class = MagicMock()
    modules = {
        "langfuse_agent": MagicMock(),
        "langfuse_agent.api_client": MagicMock(),
    }
    with patch.dict("sys.modules", modules):
        modules["langfuse_agent.api_client"].LangfuseApi = mock_lf_class

        manager = HydrationManager()
        res = manager.hydrate_source(mock_engine, "langfuse")

        assert res["status"] == "ok"
        assert res["nodes_hydrated"] == 2
        assert res["relations_hydrated"] == 1
