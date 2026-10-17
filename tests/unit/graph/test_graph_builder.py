"""CONCEPT:AU-ORCH.execution.inject-signal-board-observations"""


from agent_utilities.graph import builder


def test_build_tag_env_map():
    tags = ["incidents", "git-ops"]
    env_map = builder.build_tag_env_map(tags)
    assert env_map["incidents"] == "INCIDENTSTOOL"
    assert env_map["git-ops"] == "GIT_OPSTOOL"


def test_pydantic_graph_is_a_required_runtime_dependency():
    assert builder.GraphBuilder is not None
