"""CONCEPT:AHE-3.13 — Layered Pre-Emit Gate Pipeline tests.

Covers the anti-slop registry, P0/P1/P2 checklist parsing + blocking, the multi-dimensional critique
(score 1-5, weakest dimension, pass threshold), the composed pre-emit gate (warn vs block), and the
live engine path that runs the gate on adapter output.
"""

from __future__ import annotations

import shutil

import pytest

from agent_utilities.harness.quality_gates import (
    AntipatternRegistry,
    MultiDimensionalCritique,
    PreEmitGate,
    PreflightGate,
    Priority,
    parse_checklist,
)

pytestmark = pytest.mark.concept(id="AHE-3.13")


# ── anti-slop registry ──────────────────────────────────────────────


def test_antipattern_detects_invented_metric():
    reg = AntipatternRegistry()
    found = {a.name for a in reg.detect("Our system is 10x faster than the rest.")}
    assert "invented-metric" in found


def test_antipattern_detects_filler():
    reg = AntipatternRegistry()
    assert any(
        a.name == "filler-copy" for a in reg.detect("Feature One and lorem ipsum")
    )


def test_antipattern_clean_text():
    reg = AntipatternRegistry()
    assert reg.detect("A concise, specific paragraph with a source: example.com") == []


# ── preflight checklist ─────────────────────────────────────────────


def test_parse_checklist():
    md = "- [P0] must have title\n- [P1] nice spacing\n* [P2] optional footer\nnot a rule"
    rules = parse_checklist(md)
    assert len(rules) == 3
    assert rules[0].priority is Priority.P0


def test_preflight_blocks_on_failing_p0():
    rules = parse_checklist("- [P0] contains TITLE\n- [P1] contains FOOTER")

    # predicate: rule passes only if its trailing keyword (upper word) is present in output
    def pred(rule, out):
        kw = rule.text.split()[-1]
        return kw in out

    gate = PreflightGate(rules, predicate=pred)
    res = gate.check("has FOOTER but no header")
    assert res.passed is False
    assert res.failed_p0  # the P0 (TITLE) failed


def test_preflight_passes_when_p0_met():
    rules = parse_checklist("- [P0] contains TITLE")
    gate = PreflightGate(rules, predicate=lambda r, o: "TITLE" in o)
    assert gate.check("has TITLE").passed is True


# ── multi-dimensional critique ──────────────────────────────────────


def test_critique_flags_weak_output():
    crit = MultiDimensionalCritique()
    res = crit.critique("")  # empty → low coverage
    assert res.passed is False
    assert res.weakest in res.scores


def test_critique_passes_strong_output():
    crit = MultiDimensionalCritique()
    strong = (
        "This is a thorough, specific answer. " * 20
        + "\nSee source: https://example.com for detail."
    )
    res = crit.critique(strong)
    assert res.passed is True
    assert all(v >= 3 for v in res.scores.values())


def test_critique_flags_secret_leak_safety():
    crit = MultiDimensionalCritique()
    res = crit.critique("api_key = sk-12345 " + "padding text " * 30)
    assert res.scores["safety"] < 3
    assert res.passed is False


def test_critique_flags_antipattern_even_if_scores_ok():
    crit = MultiDimensionalCritique()
    text = "A detailed and specific writeup. " * 20 + " It is 99.9% uptime."
    res = crit.critique(text)
    assert "invented-metric" in res.antipatterns
    assert res.passed is False


# ── composed pre-emit gate ──────────────────────────────────────────


def test_pre_emit_gate_warn_does_not_block():
    gate = PreEmitGate(mode="warn")
    res = gate.evaluate("")  # weak
    assert res.ok is False
    assert res.blocked is False  # warn never blocks


def test_pre_emit_gate_block_blocks_weak():
    gate = PreEmitGate(mode="block")
    res = gate.evaluate("")
    assert res.ok is False
    assert res.blocked is True


# ── live engine path ────────────────────────────────────────────────


async def test_engine_runs_gate_on_adapter_output():
    if not shutil.which("echo"):
        pytest.skip("no echo on PATH")
    from agent_utilities.core.execution.adapters import (
        AdapterDefinition,
        AdapterRegistry,
        StreamFormat,
    )
    from agent_utilities.core.execution.engine import UnifiedExecutionEngine
    from agent_utilities.models.execution_manifest import AgentSpec, ExecutionManifest

    reg = AdapterRegistry(load_builtins=False)
    reg.register(
        AdapterDefinition(
            id="echo",
            bin="echo",
            build_args=lambda m, p: [p],
            stream_format=StreamFormat.PLAIN,
        )
    )
    eng = UnifiedExecutionEngine(registry=reg)
    m = ExecutionManifest(
        agents=[AgentSpec(agent_id="a", task_template="hi")],
        metadata={"runtime": "echo", "quality_gate": "warn"},
    )
    res = await eng.run(m)
    assert "quality_gate" in res.telemetry
    assert "scores" in res.telemetry["quality_gate"]
