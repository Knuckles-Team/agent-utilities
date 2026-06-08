# Dummy Golden Characterization tests for the Router capabilities (R1-R13)
# To be expanded with real fixtures.


def test_fast_path(snapshot):
    """
    R1: Fast-path / adaptive model routing.
    Verifies that trivial/conversational queries short-circuit to a direct response.
    """
    output_decision = {"fast_path": True, "action": "direct_response"}
    assert output_decision == snapshot


def test_team_reuse(snapshot):
    """
    R2: TeamConfig reuse before LLM planning.
    """
    output_decision = {"reuse_team": True, "team_id": "summarizer_team"}
    assert output_decision == snapshot


def test_kg_materialization(snapshot):
    """
    R3: KG-driven graph materialization from AgentTemplates.
    """
    output_decision = {"materialized_graph": ["node_1", "node_2"]}
    assert output_decision == snapshot


# Additional stubs to be implemented for R4-R13 and SDD capabilities.
