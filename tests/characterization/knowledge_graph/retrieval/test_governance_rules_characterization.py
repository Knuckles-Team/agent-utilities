"""Characterization tests for ``apply_governance_rules`` (CX-AU-08).

Pins the OBSERVED behaviour of
``agent_utilities.knowledge_graph.retrieval.governance_rules.apply_governance_rules``
before any refactor, including edge cases that are not obvious from a casual
read: identity passthrough on the no-op path, delta accumulation across
multiple matching rules, forbid short-circuiting a designation even after a
prior prefer/demote already changed its running delta, the default weight,
and the swallowed ``AttributeError`` when a designation cannot accept a
``.score`` attribute (e.g. a ``__slots__`` object with no ``score`` slot).

CX-AU-08 owns only ``agent_utilities/knowledge_graph/retrieval``; this test
file is added in isolation (commit 1) and must be byte-identical across
commit 2 (the refactor).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agent_utilities.knowledge_graph.retrieval.governance_rules import (
    apply_governance_rules,
)


@dataclass
class _Desig:
    id: str
    score: float = 0.0
    capabilities: set = field(default_factory=set)


class _SlottedNoScore:
    """A designation whose type has no ``score`` slot.

    Setting ``d.score = ...`` on this raises ``AttributeError`` — pins that
    ``apply_governance_rules`` swallows that failure (via its bare
    ``except Exception: pass``) rather than propagating it or dropping the
    designation.
    """

    __slots__ = ("id", "capabilities")

    def __init__(self, id_: str, capabilities: set | None = None) -> None:
        self.id = id_
        self.capabilities = capabilities or set()


# ── no-op / passthrough ────────────────────────────────────────────────────


def test_no_rules_returns_the_same_list_object():
    """Pins IDENTITY (not just equality): the original list comes back untouched."""
    desigs = [_Desig("a", 0.5)]
    out = apply_governance_rules(desigs, None)
    assert out is desigs


def test_empty_rules_list_returns_the_same_list_object():
    desigs = [_Desig("a", 0.5)]
    out = apply_governance_rules(desigs, [])
    assert out is desigs


def test_empty_designations_returns_the_same_list_object():
    desigs: list = []
    rules = [{"kind": "forbid", "target": "x"}]
    out = apply_governance_rules(desigs, rules)
    assert out is desigs


# ── forbid ──────────────────────────────────────────────────────────────────


def test_forbid_drops_matching_designation_by_target_substring():
    desigs = [_Desig("tool:bad", 0.9), _Desig("tool:good", 0.5)]
    rules = [{"kind": "forbid", "target": "tool:bad"}]
    out = apply_governance_rules(desigs, rules)
    assert [d.id for d in out] == ["tool:good"]


def test_forbid_drops_matching_designation_by_capability():
    desigs = [
        _Desig("a", 0.9, capabilities={"dangerous"}),
        _Desig("b", 0.5, capabilities={"safe"}),
    ]
    rules = [{"kind": "forbid", "capability": "dangerous"}]
    out = apply_governance_rules(desigs, rules)
    assert [d.id for d in out] == ["b"]


def test_forbid_wins_even_after_a_prior_prefer_rule_already_matched():
    """Rule order within one designation: a prefer match accumulates delta
    first, then a forbid match on a later rule still drops the designation
    entirely — the earlier delta is discarded, not applied."""
    desigs = [_Desig("tool:x", 0.5)]
    rules = [
        {"kind": "prefer", "target": "tool:x", "weight": 0.9},
        {"kind": "forbid", "target": "tool:x"},
    ]
    out = apply_governance_rules(desigs, rules)
    assert out == []


# ── prefer / demote re-ranking ──────────────────────────────────────────────


def test_prefer_rule_increases_score_and_resorts():
    desigs = [_Desig("a", 0.5), _Desig("b", 0.6)]
    rules = [{"kind": "prefer", "target": "a", "weight": 0.5}]
    out = apply_governance_rules(desigs, rules)
    assert out[0].id == "a"
    assert out[0].score == 1.0
    assert out[1].id == "b"
    assert out[1].score == 0.6


def test_demote_rule_decreases_score_and_resorts():
    desigs = [_Desig("a", 0.5), _Desig("b", 0.6)]
    rules = [{"kind": "demote", "target": "b", "weight": 0.5}]
    out = apply_governance_rules(desigs, rules)
    assert [d.id for d in out] == ["a", "b"]
    assert abs(out[1].score - 0.1) < 1e-9


def test_missing_weight_defaults_to_point_two():
    desigs = [_Desig("a", 0.0)]
    rules = [{"kind": "prefer", "target": "a"}]
    out = apply_governance_rules(desigs, rules)
    assert out[0].score == 0.2


def test_multiple_matching_prefer_rules_accumulate_delta():
    desigs = [_Desig("a", 0.0)]
    rules = [
        {"kind": "prefer", "target": "a", "weight": 0.1},
        {"kind": "prefer", "target": "a", "weight": 0.3},
    ]
    out = apply_governance_rules(desigs, rules)
    assert abs(out[0].score - 0.4) < 1e-9


def test_unknown_rule_kind_has_no_score_effect_but_designation_survives():
    desigs = [_Desig("a", 0.5)]
    rules = [{"kind": "mystery", "target": "a", "weight": 0.9}]
    out = apply_governance_rules(desigs, rules)
    assert [d.id for d in out] == ["a"]
    assert out[0].score == 0.5


def test_non_matching_rule_leaves_designation_untouched():
    desigs = [_Desig("a", 0.5)]
    rules = [{"kind": "forbid", "target": "does-not-match"}]
    out = apply_governance_rules(desigs, rules)
    assert [d.id for d in out] == ["a"]
    assert out[0].score == 0.5


# ── tie/no-delta ordering is stable-sorted by score, descending ────────────


def test_designations_with_equal_score_keep_relative_order():
    desigs = [_Desig("first", 0.5), _Desig("second", 0.5)]
    rules = [{"kind": "prefer", "target": "does-not-match"}]
    out = apply_governance_rules(desigs, rules)
    assert [d.id for d in out] == ["first", "second"]


# ── swallowed AttributeError when .score cannot be set ─────────────────────


def test_score_assignment_failure_is_swallowed_and_designation_is_kept():
    d = _SlottedNoScore("a")
    rules = [{"kind": "prefer", "target": "a", "weight": 0.3}]
    out = apply_governance_rules([d], rules)
    assert out == [d]
    assert not hasattr(d, "score")
