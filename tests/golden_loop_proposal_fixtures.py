"""Shared golden-loop ``TeamSpec`` proposal builders (CONCEPT:AU-AHE.assimilation.research-auto-merge).

``strong_team``/``weak_team`` were copy-pasted identically across
tests/unit/test_promotion_governance.py,
tests/characterization/knowledge_graph/research/test_auto_merge_consider_characterization.py,
and tests/unit/knowledge_graph/test_loop_auto_merge.py. One shared builder pair
instead of three copies.
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.enrichment.orchestration import TeamSpec


def strong_team() -> TeamSpec:
    return TeamSpec(
        name="Resolver Team",
        goal="Address open KG topics about retrieval quality",
        lead="Lead",
        members=["Researcher", "Validator"],
        description="A complete, well-formed team proposal.",
    )


def weak_team() -> TeamSpec:
    return TeamSpec(name="bare", goal="", lead="", members=[])
