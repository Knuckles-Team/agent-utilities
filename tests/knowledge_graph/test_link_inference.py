from __future__ import annotations

"""Tests for zero-LLM, pack-driven typed-edge extraction.

CONCEPT:KG-2.33 — Zero-LLM Pack-Driven Link Inference
"""


import time
from unittest.mock import MagicMock

from agent_utilities.knowledge_graph.kb.link_inference import (
    MAX_INPUT_CHARS,
    infer_links,
)
from agent_utilities.models.knowledge_graph import RegistryEdgeType
from agent_utilities.models.schema_pack import LinkInferenceRule
from agent_utilities.models.schema_packs import get_schema_pack


def test_infers_supports_edge_to_wikilink_target():
    rules = [
        LinkInferenceRule(
            pattern=r"\bsupports?\s+\[\[([^\]|]+)",
            edge_type="supports_belief",
            source="doc",
            target="group:1",
        )
    ]
    rels = infer_links("Our method supports [[Smith2020]].", "doc:1", rules)
    assert len(rels) == 1
    assert rels[0].source_name == "doc:1"
    assert rels[0].target_name == "Smith2020"
    assert rels[0].relationship_type == "supports_belief"


def test_multiple_matches_and_self_loop_dropped():
    rules = [
        LinkInferenceRule(pattern=r"links \[\[([^\]|]+)", edge_type="supports_belief")
    ]
    rels = infer_links("links [[A]] and links [[B]]", "doc:1", rules)
    assert {r.target_name for r in rels} == {"A", "B"}


def test_redos_pattern_is_bounded():
    # Catastrophic-backtracking pattern against a long input must return quickly.
    rules = [LinkInferenceRule(pattern=r"(a+)+$", edge_type="supports_belief")]
    start = time.monotonic()
    rels = infer_links("a" * 100_000 + "!", "doc:1", rules)
    elapsed = time.monotonic() - start
    assert elapsed < 1.0
    assert isinstance(rels, list)


def test_input_is_truncated():
    # A match placed beyond MAX_INPUT_CHARS must not be seen.
    rules = [
        LinkInferenceRule(pattern=r"cites \[\[([^\]|]+)", edge_type="cites_source")
    ]
    content = ("x" * (MAX_INPUT_CHARS + 50)) + " cites [[Hidden]]"
    rels = infer_links(content, "doc:1", rules)
    assert rels == []


def test_empty_rules_or_content_noop():
    assert infer_links("supports [[A]]", "doc:1", []) == []
    assert (
        infer_links("", "doc:1", get_schema_pack("research-state").link_inference) == []
    )


def test_live_path_extract_and_persist_creates_edge():
    """LIVE-PATH: EntityClaimExtractor.extract_and_persist persists a pack edge.

    Exercises the *existing* extractor entry point (not infer_links in isolation)
    and asserts the SUPPORTS_BELIEF edge is written via engine.link_nodes — proving
    the pack link-inference is actually invoked on the write path.
    """
    from agent_utilities.knowledge_graph.kb.entity_claim_extractor import (
        EntityClaimExtractor,
    )

    engine = MagicMock()
    engine.graph.__contains__ = lambda _self, _k: True  # all ids "present"
    extractor = EntityClaimExtractor(
        engine, schema_pack=get_schema_pack("research-state")
    )

    extractor.extract_and_persist(
        "The result supports [[Smith2020]] strongly.", source_id="paper:1"
    )

    edge_types = [
        call.args[2] for call in engine.link_nodes.call_args_list if len(call.args) >= 3
    ]
    assert RegistryEdgeType.SUPPORTS_BELIEF in edge_types
