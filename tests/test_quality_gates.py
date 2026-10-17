"""Pre-emit quality gate: antipatterns, preflight checklist, critique, composed gate."""

from __future__ import annotations

import pytest

from agent_utilities.harness.quality_gates import (
    AntipatternRegistry,
    MultiDimensionalCritique,
    PreEmitGate,
    PreflightGate,
    Priority,
    parse_checklist,
)

STRONG = (
    "This is a thorough, specific answer. " * 20 + "\nSee source: https://example.com."
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Our system is 10x faster than the rest.", {"invented-metric"}),
        ("Feature One and lorem ipsum", {"filler-copy"}),
        ("A concise, specific paragraph with a source: example.com", set()),
    ],
)
def test_antipattern_detection(text, expected):
    found = {a.name for a in AntipatternRegistry().detect(text)}
    assert expected <= found if expected else found == set()


def test_checklist_parse_and_preflight():
    rules = parse_checklist(
        "- [P0] must have title\n- [P1] nice spacing\n* [P2] footer\nnot a rule"
    )
    assert [r.priority for r in rules] == [Priority.P0, Priority.P1, Priority.P2]

    gate = PreflightGate(
        parse_checklist("- [P0] contains TITLE\n- [P1] contains FOOTER"),
        predicate=lambda rule, out: rule.text.split()[-1] in out,
    )
    failing = gate.check("has FOOTER but no header")
    assert failing.passed is False and failing.failed_p0
    assert gate.check("TITLE and FOOTER").passed is True


@pytest.mark.parametrize(
    ("text", "passed", "check"),
    [
        ("", False, lambda r: r.weakest in r.scores),
        (STRONG, True, lambda r: all(v >= 3 for v in r.scores.values())),
        (
            "api_key = sk-12345 " + "padding text " * 30,
            False,
            lambda r: r.scores["safety"] < 3,
        ),
        (
            "A detailed and specific writeup. " * 20 + " It is 99.9% uptime.",
            False,
            lambda r: "invented-metric" in r.antipatterns,
        ),
    ],
)
def test_critique(text, passed, check):
    result = MultiDimensionalCritique().critique(text)
    assert result.passed is passed
    assert check(result)


@pytest.mark.parametrize(("mode", "blocked"), [("warn", False), ("block", True)])
def test_pre_emit_gate_mode(mode, blocked):
    result = PreEmitGate(mode=mode).evaluate("")
    assert result.ok is False and result.blocked is blocked
    assert PreEmitGate(mode=mode).evaluate(STRONG).ok is True
