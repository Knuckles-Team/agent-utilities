"""Unit tests for Knowledge Graph Hydration service."""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent_utilities.knowledge_graph.core.hydration import HydrationManager


@pytest.fixture
def mock_engine():
    engine = MagicMock()
    return engine


@pytest.fixture(autouse=True)
def _capture_native_proxy_submission(monkeypatch):
    """Keep these mapping tests isolated from the native engine contract tests."""

    def capture(proxy, domain, entities, relationships=None):
        return proxy.authority.ingest_external_batch(domain, entities, relationships)

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ingestion.envelope_ingest.NativeChangeEnvelopeEngineProxy.ingest_external_batch",
        capture,
    )


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
        "SERVICENOW_INSTANCE": "https://servicenow.example.com",
        "SERVICENOW_USERNAME": "admin",
        "SERVICENOW_PASSWORD": "password",
    },
)
def test_hydrate_servicenow(mock_engine):
    """ServiceNow hydration delegates to the manifest-gated source-sync path."""
    import agent_utilities.knowledge_graph.core.source_sync as source_sync

    with patch.object(
        source_sync,
        "sync_source",
        return_value={"status": "materialized", "source": "servicenow", "nodes": 3},
    ) as mock_run:
        res = HydrationManager().hydrate_source(mock_engine, "servicenow")

    assert res["status"] == "materialized"
    assert res["nodes"] == 3
    assert mock_run.call_args[0][1] == "servicenow"


def test_servicenow_status_uses_provider_runtime_aliases(monkeypatch):
    """Readiness must reflect the auth aliases accepted by servicenow-api."""
    monkeypatch.delenv("SERVICENOW_URL", raising=False)
    monkeypatch.setenv("SERVICENOW_INSTANCE", "https://servicenow.example.com")
    monkeypatch.setenv("SERVICENOW_USERNAME", "service-account")
    monkeypatch.setenv("SERVICENOW_PASSWORD", "test-password")

    status = HydrationManager().get_status()["servicenow"]

    assert status == {
        "configured": True,
        "url": "https://servicenow.example.com",
    }


# ``jira``/``plane`` are no longer generic ``HydrationManager`` sources: they
# were re-homed to first-class delta connectors (``source_sync._sync_jira`` /
# ``_sync_plane``, CONCEPT:AU-KG.compute.confluence-first-class-delta/2.124/2.125), so ``CAPABILITY_REGISTRY`` no
# longer carries them and ``hydrate_source(..., "jira"|"plane")`` correctly
# raises ``Unknown hydration source``. The stale ``test_hydrate_jira`` /
# ``test_hydrate_plane`` cases that exercised the removed generic path were
# deleted (No-Legacy: delete the test of a deleted path).


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


def test_hydrate_source_control_default(mock_engine):
    manager = HydrationManager()
    res = manager.hydrate_source(mock_engine, "source_control")
    assert res["status"] == "ok"
    assert res["nodes_hydrated"] >= 1


@patch.dict(os.environ, {"GITHUB_TOKEN": "gh-tok"}, clear=True)
def test_hydrate_source_control_github_unavailable(mock_engine, monkeypatch):
    """Credential presence must not create a provider-shaped demo batch."""

    class UnavailableGitHubApi:
        def get_repositories(self):
            raise ConnectionError("GitHub API is unreachable")

    manager = HydrationManager()
    monkeypatch.setattr(
        manager,
        "_load_github_api",
        lambda: lambda: UnavailableGitHubApi(),
    )

    res = manager.hydrate_source(mock_engine, "source_control")

    assert res["status"] == "unavailable"
    assert res["nodes_hydrated"] == 0
    assert res["relations_hydrated"] == 0
    mock_engine.ingest_external_batch.assert_not_called()
    assert "Test GitHub Project" not in str(res)
    assert "github:repo:101" not in str(res)
    assert "github:workflow:4001" not in str(res)


@patch.dict(os.environ, {"GITHUB_TOKEN": "gh-tok"}, clear=True)
def test_hydrate_source_control_github_preserves_provider_identity(
    mock_engine, monkeypatch
):
    """Injected provider records retain real IDs and source provenance."""

    class InjectedGitHubApi:
        def get_repositories(self):
            return SimpleNamespace(
                data=[
                    {
                        "id": 31415,
                        "name": "ledger",
                        "full_name": "acme/ledger",
                        "description": "Authoritative ledger repository",
                        "default_branch": "main",
                        "html_url": "https://github.com/acme/ledger",
                    }
                ]
            )

        def get_workflow_runs(self, *, owner, repo):
            assert (owner, repo) == ("acme", "ledger")
            return SimpleNamespace(
                data=[
                    {
                        "id": 2718,
                        "name": "CI",
                        "head_branch": "main",
                        "head_sha": "abcdef123456",
                        "status": "completed",
                        "conclusion": "success",
                        "event": "push",
                        "html_url": (
                            "https://github.com/acme/ledger/actions/runs/2718"
                        ),
                    }
                ]
            )

        def close(self):
            return None

    manager = HydrationManager()
    monkeypatch.setattr(
        manager,
        "_load_github_api",
        lambda: lambda: InjectedGitHubApi(),
    )

    res = manager.hydrate_source(mock_engine, "source_control")

    assert res == {
        "status": "ok",
        "source": "github",
        "nodes_hydrated": 2,
        "relations_hydrated": 1,
    }
    mock_engine.ingest_external_batch.assert_called_once()
    domain, entities, relationships = mock_engine.ingest_external_batch.call_args.args
    assert domain == "github"
    assert entities == [
        {
            "id": "github:repository:31415",
            "type": "repository",
            "domain": "github",
            "source_system": "github",
            "externalToolId": "31415",
            "name": "ledger",
            "full_name": "acme/ledger",
            "description": "Authoritative ledger repository",
            "default_branch": "main",
            "web_url": "https://github.com/acme/ledger",
            "source_uri": "https://github.com/acme/ledger",
        },
        {
            "id": "github:pipelinerun:acme/ledger:2718",
            "type": "pipeline",
            "domain": "github",
            "source_system": "github",
            "externalToolId": "2718",
            "name": "CI",
            "status": "completed",
            "conclusion": "success",
            "head_sha": "abcdef123456",
            "head_branch": "main",
            "event": "push",
            "web_url": "https://github.com/acme/ledger/actions/runs/2718",
            "source_uri": "https://github.com/acme/ledger/actions/runs/2718",
        },
    ]
    assert relationships == [
        {
            "source": "github:pipelinerun:acme/ledger:2718",
            "target": "github:repository:31415",
            "type": "depends_on",
            "domain": "github",
        }
    ]


def _github_repository_model():
    """Build the real connector Repository model used by the hydration path."""
    models = pytest.importorskip("github_agent.github_response_models")
    Repository = models.Repository

    user = {
        "login": "acme",
        "id": 7,
        "node_id": "org-7",
        "avatar_url": "https://github.example.test/acme.png",
        "url": "https://github.example.test/users/acme",
        "html_url": "https://github.example.test/acme",
        "type": "Organization",
        "site_admin": False,
    }
    return Repository.model_validate(
        {
            "id": 31415,
            "node_id": "repo-31415",
            "name": "ledger",
            "full_name": "acme/ledger",
            "private": True,
            "owner": user,
            "html_url": "https://github.example.test/acme/ledger",
            "description": "Authoritative ledger repository",
            "fork": False,
            "url": "https://github.example.test/api/v3/repos/acme/ledger",
            "created_at": "2026-08-01T00:00:00Z",
            "updated_at": "2026-08-02T00:00:00Z",
            "pushed_at": "2026-08-02T00:00:00Z",
            "git_url": "git://github.example.test/acme/ledger.git",
            "ssh_url": "git@github.example.test:acme/ledger.git",
            "clone_url": "https://github.example.test/acme/ledger.git",
            "svn_url": "https://github.example.test/acme/ledger",
            "homepage": None,
            "size": 10,
            "stargazers_count": 0,
            "watchers_count": 0,
            "language": "Python",
            "has_issues": True,
            "has_projects": True,
            "has_downloads": True,
            "has_wiki": False,
            "has_pages": False,
            "forks_count": 0,
            "mirror_url": None,
            "archived": False,
            "disabled": False,
            "open_issues_count": 0,
            "license": None,
            "allow_forking": False,
            "is_template": False,
            "topics": [],
            "visibility": "private",
            "forks": 0,
            "open_issues": 0,
            "watchers": 0,
            "default_branch": "main",
        }
    )


@patch.dict(os.environ, {"GITHUB_TOKEN": "gh-tok"}, clear=True)
def test_hydrate_github_accepts_actual_connector_models(mock_engine, monkeypatch):
    """The adapter consumes github-agent's typed Repository/WorkflowRun models."""
    models = pytest.importorskip("github_agent.github_response_models")

    repository = _github_repository_model()
    workflow = models.WorkflowRun.model_validate(
        {
            "id": 2718,
            "name": "CI",
            "head_branch": "main",
            "head_sha": "abcdef123456",
            "status": "completed",
            "conclusion": "success",
            "event": "push",
            "html_url": "https://github.example.test/acme/ledger/actions/2718",
        }
    )

    class TypedGitHubClient:
        def get_repositories(self):
            return SimpleNamespace(data=[repository])

        def get_workflow_runs(self, *, owner, repo):
            assert (owner, repo) == ("acme", "ledger")
            return SimpleNamespace(data=[workflow])

    manager = HydrationManager()
    monkeypatch.setattr(
        manager,
        "_load_github_api",
        lambda: lambda: TypedGitHubClient(),
    )
    result = manager.hydrate_source(mock_engine, "github")

    assert result["status"] == "ok"
    assert result["nodes_hydrated"] == 2
    assert result["relations_hydrated"] == 1
    entities = mock_engine.ingest_external_batch.call_args.args[1]
    assert entities[0]["id"] == "github:repository:31415"
    assert entities[1]["id"] == "github:pipelinerun:acme/ledger:2718"


@patch.dict(os.environ, {"GITHUB_TOKEN": "gh-tok"}, clear=True)
def test_hydrate_github_missing_connector_is_skipped(mock_engine, monkeypatch):
    manager = HydrationManager()
    monkeypatch.setattr(manager, "_load_github_api", lambda: None)

    result = manager.hydrate_source(mock_engine, "github")

    assert result == {
        "status": "skipped",
        "source": "github",
        "reason": "github-agent package not installed",
        "nodes_hydrated": 0,
        "relations_hydrated": 0,
    }
    mock_engine.ingest_external_batch.assert_not_called()


@patch.dict(os.environ, {"GITHUB_TOKEN": "gh-tok"}, clear=True)
def test_hydrate_github_reports_connector_runtime_incompatibility(
    mock_engine, monkeypatch
):
    from agent_utilities.knowledge_graph.core.hydration import (
        _GithubConnectorCompatibilityError,
    )

    def incompatible_connector():
        raise _GithubConnectorCompatibilityError

    manager = HydrationManager()
    monkeypatch.setattr(manager, "_load_github_api", incompatible_connector)

    result = manager.hydrate_source(mock_engine, "github")

    assert result == {
        "status": "unavailable",
        "source": "github",
        "reason": "github-agent runtime incompatible",
        "nodes_hydrated": 0,
        "relations_hydrated": 0,
    }
    mock_engine.ingest_external_batch.assert_not_called()


def test_hydrate_github_without_credentials_is_skipped(mock_engine, monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_API_KEY", raising=False)
    manager = HydrationManager()
    monkeypatch.setattr(
        manager,
        "_load_github_api",
        lambda: pytest.fail("connector must not load without credentials"),
    )

    result = manager.hydrate_source(mock_engine, "github")

    assert result["status"] == "skipped"
    assert result["reason"] == "Missing GITHUB_TOKEN/GITHUB_API_KEY"
    mock_engine.ingest_external_batch.assert_not_called()


@patch.dict(os.environ, {"GITHUB_TOKEN": "gh-tok"}, clear=True)
def test_hydrate_github_empty_response_is_no_data(mock_engine, monkeypatch):
    class EmptyGitHubClient:
        def get_repositories(self):
            return SimpleNamespace(data=[])

    manager = HydrationManager()
    monkeypatch.setattr(
        manager, "_load_github_api", lambda: lambda: EmptyGitHubClient()
    )

    result = manager.hydrate_source(mock_engine, "github")

    assert result["status"] == "no_data"
    assert result["nodes_hydrated"] == 0
    assert result["relations_hydrated"] == 0
    mock_engine.ingest_external_batch.assert_not_called()


@patch.dict(os.environ, {"GITHUB_TOKEN": "gh-tok"}, clear=True)
def test_hydrate_github_malformed_response_is_error(mock_engine, monkeypatch):
    class MalformedGitHubClient:
        def get_repositories(self):
            return SimpleNamespace(data={"repositories": "not-a-list"})

    manager = HydrationManager()
    monkeypatch.setattr(
        manager,
        "_load_github_api",
        lambda: lambda: MalformedGitHubClient(),
    )

    result = manager.hydrate_source(mock_engine, "github")

    assert result["status"] == "error"
    assert result["reason"] == "GitHub provider returned malformed repository data"
    mock_engine.ingest_external_batch.assert_not_called()


@patch.dict(os.environ, {"GITHUB_TOKEN": "gh-tok"}, clear=True)
def test_hydrate_github_invalid_provider_id_is_rejected(mock_engine, monkeypatch):
    class InvalidIdGitHubClient:
        def get_repositories(self):
            return SimpleNamespace(
                data=[
                    {
                        "id": "not-an-integer",
                        "name": "ledger",
                        "full_name": "acme/ledger",
                    }
                ]
            )

    manager = HydrationManager()
    monkeypatch.setattr(
        manager,
        "_load_github_api",
        lambda: lambda: InvalidIdGitHubClient(),
    )

    result = manager.hydrate_source(mock_engine, "github")

    assert result["status"] == "error"
    assert result["reason"] == "GitHub provider returned malformed repository data"
    mock_engine.ingest_external_batch.assert_not_called()


@patch.dict(os.environ, {"GITHUB_TOKEN": "gh-tok"}, clear=True)
def test_hydrate_github_invalid_provider_type_is_rejected(mock_engine, monkeypatch):
    class InvalidTypeGitHubClient:
        def get_repositories(self):
            return SimpleNamespace(
                data=[
                    {
                        "id": 31415,
                        "name": "ledger",
                        "full_name": "acme/ledger",
                        "private": "false",
                    }
                ]
            )

    manager = HydrationManager()
    monkeypatch.setattr(
        manager,
        "_load_github_api",
        lambda: lambda: InvalidTypeGitHubClient(),
    )

    result = manager.hydrate_source(mock_engine, "github")

    assert result["status"] == "error"
    assert result["reason"] == "GitHub provider returned malformed repository data"
    mock_engine.ingest_external_batch.assert_not_called()


@patch.dict(os.environ, {"GITHUB_TOKEN": "gh-tok"}, clear=True)
def test_hydrate_github_workflow_failure_is_partial_and_redacted(
    mock_engine, monkeypatch
):
    repository = {
        "id": 31415,
        "name": "ledger",
        "full_name": "acme/ledger",
    }

    class WorkflowFailureGitHubClient:
        def get_repositories(self):
            return SimpleNamespace(data=[repository])

        def get_workflow_runs(self, *, owner, repo):
            assert (owner, repo) == ("acme", "ledger")
            raise RuntimeError("credential=query-fixture")

    manager = HydrationManager()
    monkeypatch.setattr(
        manager,
        "_load_github_api",
        lambda: lambda: WorkflowFailureGitHubClient(),
    )

    result = manager.hydrate_source(mock_engine, "github")

    assert result["status"] == "partial"
    assert result["workflow_failures"] == 1
    assert result["nodes_hydrated"] == 1
    assert result["relations_hydrated"] == 0
    assert "credential=query-fixture" not in str(result)
    mock_engine.ingest_external_batch.assert_called_once()


@patch.dict(
    os.environ,
    {
        "GITHUB_TOKEN": "gh-tok",
        "GITHUB_URL": "https://github.example.test/api/v3",
    },
    clear=True,
)
def test_hydrate_github_uses_governed_enterprise_endpoint(mock_engine, monkeypatch):
    endpoint = os.environ["GITHUB_URL"]
    factory_calls = 0

    class EndpointGitHubClient:
        def __init__(self):
            self.url = endpoint

        def get_repositories(self):
            return SimpleNamespace(data=[])

    def factory():
        nonlocal factory_calls
        factory_calls += 1
        return EndpointGitHubClient()

    manager = HydrationManager()
    monkeypatch.setattr(manager, "_load_github_api", lambda: factory)

    result = manager.hydrate_source(mock_engine, "github")

    assert factory_calls == 1
    assert result["status"] == "no_data"
    assert endpoint not in str(result)
    status = manager.get_status()["github"]
    assert status["endpoint_valid"] is True
    assert status["url"] == "https://api.github.com"
    assert endpoint not in str(status)


@patch.dict(os.environ, {"GITHUB_TOKEN": "gh-tok"}, clear=True)
def test_hydrate_github_rejects_credential_bearing_endpoint(mock_engine, monkeypatch):
    query_marker = "query-fixture-token"
    hostile_endpoint = "https://github.example.test/api/v3?token=" + query_marker
    monkeypatch.setenv("GITHUB_URL", hostile_endpoint)
    factory_calls = 0

    def factory():
        nonlocal factory_calls
        factory_calls += 1
        pytest.fail("hostile endpoint must be rejected before connector construction")

    manager = HydrationManager()
    monkeypatch.setattr(manager, "_load_github_api", lambda: factory)

    result = manager.hydrate_source(mock_engine, "github")

    assert result["status"] == "error"
    assert result["reason"] == "GitHub endpoint configuration is invalid"
    assert factory_calls == 0
    assert hostile_endpoint not in str(result)
    assert query_marker not in str(result)
    mock_engine.ingest_external_batch.assert_not_called()


@patch.dict(os.environ, {"GITHUB_TOKEN": "gh-tok"}, clear=True)
def test_hydrate_github_redacts_credential_bearing_provider_urls(
    mock_engine, monkeypatch
):
    query_marker = "provider-fixture-token"
    repository = {
        "id": 31415,
        "name": "ledger",
        "full_name": "acme/ledger",
        "html_url": "https://github.example.test/acme/ledger?token=" + query_marker,
    }

    class HostileUrlGitHubClient:
        def get_repositories(self):
            return SimpleNamespace(data=[repository])

        def get_workflow_runs(self, *, owner, repo):
            return SimpleNamespace(data=[])

    manager = HydrationManager()
    monkeypatch.setattr(
        manager,
        "_load_github_api",
        lambda: lambda: HostileUrlGitHubClient(),
    )

    result = manager.hydrate_source(mock_engine, "github")

    assert result["status"] == "ok"
    entities = mock_engine.ingest_external_batch.call_args.args[1]
    assert "web_url" not in entities[0]
    assert query_marker not in str(entities)


def test_hydrate_enterprise_architecture_default(mock_engine):
    manager = HydrationManager()
    res = manager.hydrate_source(mock_engine, "enterprise_architecture")
    assert res["status"] == "ok"
    assert res["nodes_hydrated"] == 2
    assert res["relations_hydrated"] == 1


def test_hydrate_process_modeling_default(mock_engine):
    manager = HydrationManager()
    res = manager.hydrate_source(mock_engine, "process_modeling")
    assert res["status"] == "ok"
    assert res["nodes_hydrated"] >= 1
    assert res["relations_hydrated"] >= 1


def test_hydrate_issue_tracking_default(mock_engine):
    manager = HydrationManager()
    res = manager.hydrate_source(mock_engine, "issue_tracking")
    assert res["status"] == "ok"
    assert res["nodes_hydrated"] >= 1


def test_hydrate_relational_database(mock_engine):
    manager = HydrationManager()
    res = manager.hydrate_source(mock_engine, "relational_database")
    assert res["status"] == "ok"
    assert res["nodes_hydrated"] >= 1


def test_hydrate_message_protocol(mock_engine):
    manager = HydrationManager()
    res = manager.hydrate_source(mock_engine, "message_protocol")
    assert res["status"] == "ok"
    assert res["nodes_hydrated"] == 2
    assert res["relations_hydrated"] == 1


def test_hydrate_leanix_mirrors_factsheets(mock_engine):
    """_hydrate_leanix injects a live client into the extractor and, via the
    AU-P1-5 envelope-native sync path, commits one native ChangeEnvelope per
    fact sheet plus a relationship-projection envelope.

    Stale-test note (category 2): this test previously asserted the OLDER
    ``mock_engine.ingest_external_batch(domain, entities, relationships)``
    call shape shared by the other ``_hydrate_X`` sources in this file.
    ``_hydrate_leanix`` -> ``source_sync._sync_leanix`` has since migrated to
    the envelope-native path (its own docstring: "AU-P1-5 envelope-native
    ... Migrated first as the flagship delta connector"), which commits
    through ``envelope_ingest.ingest_envelope`` instead. A bare
    ``MagicMock()`` cannot satisfy that path's
    ``_resolve_native_authority``, which relies on ``getattr(x, attr, None)``
    sentinel checks — a MagicMock auto-vivifies every attribute access, so it
    can never look like "capability genuinely absent" and always raises
    ``NativeChangeEnvelopeUnavailable``. Patch the native commit boundary
    (``ingest_envelope``) instead, matching how other boundary-mocked tests
    in this session (SSRF-safe HTTP wrappers, A2A discovery) stub the actual
    transport instead of fighting deep mock plumbing.
    """
    from types import SimpleNamespace

    fake_client = SimpleNamespace(
        meta_model=lambda: {
            "factSheets": {
                "Application": {
                    "fields": {},
                    "relations": {
                        "relApplicationToITComponent": {
                            "targetFactSheetType": "ITComponent"
                        }
                    },
                },
                "ITComponent": {"fields": {}, "relations": {}},
            }
        },
        factsheets=lambda type=None, since=None, ids=None: {
            "Application": [
                {
                    "id": "a1",
                    "name": "Billing",
                    "type": "Application",
                    "relApplicationToITComponent": [{"factSheetId": "ic1"}],
                }
            ],
            "ITComponent": [{"id": "ic1", "name": "PG", "type": "ITComponent"}],
        }.get(type, []),
    )

    with (
        patch(
            "agent_utilities.ecosystem.ea_clients.get_leanix_client",
            return_value=fake_client,
        ),
        patch(
            "agent_utilities.knowledge_graph.ingestion.envelope_ingest.ingest_envelope",
            return_value={"status": "success"},
        ) as mock_ingest,
    ):
        res = HydrationManager().hydrate_source(mock_engine, "leanix")

    assert res["status"] == "ok"
    # nodes_hydrated/relations_hydrated aren't canonical EtlResult fields
    # (sync_source's model_validate projection, agent_utilities/knowledge_graph
    # /core/source_sync.py) — they land under "details" alongside every other
    # connector-specific diagnostic.
    assert res["details"]["nodes_hydrated"] == 2
    assert res["details"]["relations_hydrated"] == 1
    # 2 fact-sheet (entity) envelopes + 1 relationship-projection envelope.
    assert mock_ingest.call_count == 3
    envelopes = [call.args[1] for call in mock_ingest.call_args_list]
    assert all(env.connector == "leanix" for env in envelopes)
    entity_envelopes = [
        env
        for env in envelopes
        if env.typed_payload.get("type") != "SourceRelationshipProjection"
    ]
    assert {env.source_object_id for env in entity_envelopes} == {
        "app:a1",
        "itcomponent:ic1",
    }
    relation_envelope = next(
        env
        for env in envelopes
        if env.typed_payload.get("type") == "SourceRelationshipProjection"
    )
    assert relation_envelope.typed_payload["relationship_count"] == 1
    assert (
        relation_envelope.typed_payload["_links"][0]["type"]
        == "REL_APPLICATION_TO_IT_COMPONENT"
    )


def test_hydrate_leanix_no_client_skips(mock_engine):
    with patch(
        "agent_utilities.ecosystem.ea_clients.get_leanix_client", return_value=None
    ):
        res = HydrationManager().hydrate_source(mock_engine, "leanix")
    assert res["status"] == "skipped"
    mock_engine.ingest_external_batch.assert_not_called()
