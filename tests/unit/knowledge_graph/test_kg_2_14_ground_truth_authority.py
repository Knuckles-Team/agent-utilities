"""CONCEPT:AU-KG.memory.ground-truth-preamble-declaring — Ground-Truth Context Authority.

Verifies the authority tier classification, the priority boost for authoritative (durable,
injected) memory, and the Ground-Truth Hierarchy preamble that tells the agent injected memory is
authoritative (assimilated from memory-os Layer 7).
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.memory.memory_engine import (
    AUTHORITY_ADVISORY,
    AUTHORITY_AUTHORITATIVE,
    AUTHORITY_BOOST,
    AUTHORITY_STANDARD,
    StartupChunk,
    StartupContextBuilder,
    _authority_for,
)


@pytest.mark.concept(id="AU-KG.memory.ground-truth-preamble-declaring")
def test_authority_classification():
    assert _authority_for("profile", "Preferences") == AUTHORITY_AUTHORITATIVE
    assert _authority_for("team", "Team: core") == AUTHORITY_AUTHORITATIVE
    assert _authority_for("agents_md", "Project Rules") == AUTHORITY_AUTHORITATIVE
    assert _authority_for("active", "Core Identity") == AUTHORITY_AUTHORITATIVE
    assert _authority_for("recall", "hint") == AUTHORITY_ADVISORY
    assert _authority_for("active", "Misc Notes") == AUTHORITY_STANDARD


@pytest.mark.concept(id="AU-KG.memory.ground-truth-preamble-declaring")
def test_startup_chunk_has_authority_field_default():
    c = StartupChunk(source="x", heading="h", body="b", handle="hd", priority=4)
    assert c.source_authority == AUTHORITY_STANDARD
    c2 = StartupChunk(
        source="profile",
        heading="Preferences",
        body="b",
        handle="hd",
        priority=10,
        source_authority=AUTHORITY_AUTHORITATIVE,
    )
    assert c2.source_authority == AUTHORITY_AUTHORITATIVE


@pytest.mark.concept(id="AU-KG.memory.ground-truth-preamble-declaring")
def test_chunk_priority_boosts_authoritative():
    # Bypass __init__ (no engine needed for _chunk_priority).
    b = StartupContextBuilder.__new__(StartupContextBuilder)
    auth = b._chunk_priority(
        "profile", "Preferences", "body", cwd=None, task=None, agent=None
    )
    plain = b._chunk_priority(
        "active", "Random Notes", "body", cwd=None, task=None, agent=None
    )
    # Preferences is authoritative (base 10 + boost); random notes is standard (base 4).
    assert auth >= 10 + AUTHORITY_BOOST
    assert plain == 4
    assert auth - plain >= AUTHORITY_BOOST


@pytest.mark.concept(id="AU-KG.memory.ground-truth-preamble-declaring")
def test_authority_preamble_present_and_names_sources():
    b = StartupContextBuilder.__new__(StartupContextBuilder)
    text = b._build_authority_preamble(["profile", "agents_md"])
    assert "Ground Truth Hierarchy" in text
    assert "authoritative" in text.lower()
    assert "do not re-fetch" in text.lower()
    assert "user profile" in text and "project rules" in text.lower()


@pytest.mark.concept(id="AU-KG.memory.ground-truth-preamble-declaring")
def test_authority_preamble_empty_when_no_authoritative_sources():
    b = StartupContextBuilder.__new__(StartupContextBuilder)
    assert b._build_authority_preamble([]) == ""
