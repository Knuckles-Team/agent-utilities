"""Tests for the concept-hierarchy grammar (CONCEPT:OS-5.76 / B5).

Covers the flat↔dotted mapping rules, the alias/resolution strategy, the
allocator's new 3-level awareness, and — critically — that the EXISTING flat
scheme stays valid (non-breaking).
"""

from __future__ import annotations

import pytest

from agent_utilities.governance import concept_allocator as ca
from agent_utilities.governance import concept_hierarchy as ch


# ── Grammar / marker regex ──────────────────────────────────────────────────
@pytest.mark.parametrize(
    "text,expected",
    [
        ("CONCEPT:EG-321", ["EG-321"]),  # flat
        ("CONCEPT:KG-2.312", ["KG-2.312"]),  # 2-level
        ("CONCEPT:EG-3.31.20", ["EG-3.31.20"]),  # 3-level (new)
        ("CONCEPT:KG-2.20g", ["KG-2.20g"]),  # placeholder letter
        ("CONCEPT:KEY-001", ["KEY-001"]),  # package-scoped
    ],
)
def test_marker_re_accepts_all_levels(text, expected):
    assert ch.HIERARCHY_MARKER_RE.findall(text) == expected
    # The shared allocator regex must also accept 3-level (the ONE grammar).
    assert ca.MARKER_RE.findall(text) == expected


def test_allocator_marker_re_unchanged_for_legacy():
    # Non-breaking: every previously-matching id still matches identically.
    for cid in ("AHE-3.49", "ORCH-1.105", "OS-5.72", "ECO-4.99", "KG-2.20g"):
        assert ca.MARKER_RE.findall(f"CONCEPT:{cid}") == [cid]
    # A bare letter suffix with NO dot was never a separate segment (canonical
    # regex truncates at the digit run) — unchanged by the `?`→`*` edit.
    assert ca.MARKER_RE.findall("CONCEPT:EG-028b") == ["EG-028"]


# ── Canonicalization / mapping rules ────────────────────────────────────────
def test_flat_project_id_maps_to_legacy_pillar():
    p = ch.parse_concept_id("EG-321")
    assert p.is_project
    assert p.pillar == ch.LEGACY_PILLAR == "0"
    assert p.concept == "321"
    assert p.canonical == "EG-0.321"
    assert p.needs_curation
    assert "EG-321" in p.aliases and "EG-0.321" in p.aliases


def test_two_level_project_id_already_compliant():
    p = ch.parse_concept_id("KG-2.312")
    assert p.is_project
    assert (p.pillar, p.concept, p.segment) == ("2", "312", None)
    assert p.canonical == "KG-2.312"  # self — no rewrite
    assert not p.needs_curation


def test_three_level_id_roundtrips():
    p = ch.parse_concept_id("EG-3.31.20")
    assert (p.pillar, p.concept, p.segment) == ("3", "31", "20")
    assert p.canonical == "EG-3.31.20"


def test_package_namespace_passthrough():
    p = ch.parse_concept_id("KEY-001")
    assert not p.is_project
    assert p.canonical == "KEY-001"  # untouched
    assert "package-scoped" in p.flags


def test_observed_project_namespace_promotes_dotted_ns():
    # A namespace seen with a 2-seg id is treated as a project namespace even if
    # not in the curated set.
    observed = ch.observed_project_namespaces(["ZZZ-1.5", "ZZZ-9"])
    assert "ZZZ" in observed
    p = ch.parse_concept_id("ZZZ-9", observed_project_ns=observed)
    assert p.is_project
    assert p.canonical == "ZZZ-0.9"


def test_pillar_map_override():
    ch.PILLAR_MAP[("EG", "321")] = "3"
    try:
        p = ch.parse_concept_id("EG-321")
        assert p.canonical == "EG-3.321"
        assert not p.needs_curation
    finally:
        del ch.PILLAR_MAP[("EG", "321")]


def test_unparseable_raises():
    with pytest.raises(ValueError):
        ch.parse_concept_id("not-an-id")


# ── Alias index / resolution ────────────────────────────────────────────────
def test_build_alias_index_resolves_both_forms():
    idx = ch.build_alias_index(["EG-321", "KG-2.312"])
    assert idx["EG-321"] == "EG-0.321"
    assert idx["EG-0.321"] == "EG-0.321"  # canonical resolves to itself
    assert idx["KG-2.312"] == "KG-2.312"


def test_canonicalize_matches_parse():
    assert ch.canonicalize("EG-321") == "EG-0.321"
    assert ch.canonicalize("KG-2.312") == "KG-2.312"


# ── partOf edge derivation ──────────────────────────────────────────────────
def test_derive_part_of_edges():
    parsed = [ch.parse_concept_id(c) for c in ("KG-2.312", "EG-321", "KEY-001")]
    edges = set(ch.derive_part_of_edges(parsed))
    assert ("KG-2.312", "KG-2") in edges  # concept → pillar
    assert ("KG-2", "KG") in edges  # pillar → namespace
    assert ("EG-0.321", "EG-0") in edges
    assert ("KEY-001", "KEY") in edges  # package → namespace directly


# ── Allocator: 3-level segment minting ──────────────────────────────────────
def test_allocator_mints_third_level_segment():
    taken = {"KG-2.312", "KG-2.312.1", "KG-2.312.2"}
    assert ca.next_id("KG-2.312", taken) == "KG-2.312.3"


def test_allocator_pillar_and_package_unchanged():
    assert ca.next_id("KG-2", {"KG-2.312"}) == "KG-2.313"
    assert ca.next_id("KEY", {"KEY-001"}) == "KEY-002"


def test_allocator_format_id_concept_scope():
    assert ca.format_id("KG-2.312", 5) == "KG-2.312.5"
