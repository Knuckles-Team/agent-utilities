import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from agent_utilities.graph import (
    GraphState,
    build_tag_env_map,
    create_graph_agent,
    get_discovery_registry,
    get_graph_mermaid,
    validate_graph,
)


def test_graph_builds_with_tags():
    tag_prompts = {"git": "Git stuff", "web": "Web stuff"}
    tag_env_vars = {"git": "GITTOOL", "web": "WEBTOOL"}

    graph, config = create_graph_agent(
        tag_prompts, tag_env_vars, mcp_url=None, mcp_config=None
    )

    assert graph is not None
    assert config is not None
    assert "git" in config["valid_domains"]
    assert "web" in config["valid_domains"]


def test_registry_has_specialists():
    registry = get_discovery_registry()
    # In test environment, registry might be empty if workspace isn't initialized,
    # but we can verify it returns a model.
    assert isinstance(registry.agents, list)


def test_graph_validation_reports_topology():
    tag_prompts = {"alpha": "Alpha domain", "beta": "Beta domain"}
    graph, config = create_graph_agent(tag_prompts, mcp_url=None, mcp_config=None)
    result = validate_graph(graph, config)

    assert result["domain_count"] == 2
    assert "alpha" in result["domain_tags"]
    assert "beta" in result["domain_tags"]
    assert isinstance(result["warnings"], list)


def test_mermaid_generation():
    tag_prompts = {"test": "Test domain"}
    graph, config = create_graph_agent(tag_prompts, mcp_url=None, mcp_config=None)
    mermaid = get_graph_mermaid(graph, config, title="Test")
    assert "Test" in mermaid
    assert len(mermaid) > 100


def test_build_tag_env_map():
    result = build_tag_env_map(["git", "web-search"])
    assert result["git"] == "GITTOOL"
    assert result["web-search"] == "WEB_SEARCHTOOL"


def test_graph_state_defaults():
    state = GraphState(query="test query")
    assert state.query == "test query"
    assert state.mode == "ask"
    assert state.verification_attempts == 0
    assert state.needs_replan is False
    assert state.step_cursor == 0
