"""Plan 03: characterization tests for the R2–R9 routing capabilities extracted
from the router monolith into the strategy package.

Each test pins the behaviour of an extracted named symbol so the No-Capability-
Lost gate can mark its ledger row `migrated`.
"""

from __future__ import annotations

import types

from agent_utilities.graph.routing.enrichers.self_model import (
    format_pheromone_affinities,
    format_proficiency_context,
    self_model_context,
)
from agent_utilities.graph.routing.strategies.fallback import (
    match_specialists_in_text,
    unstructured_fallback_prompt,
)
from agent_utilities.graph.routing.strategies.llm_planner import (
    is_complex_query,
    parse_rlm_plan,
    rlm_plan_instruction,
    subtask_and_widesearch_instructions,
)
from agent_utilities.graph.routing.strategies.optimization import (
    filter_by_pheromone,
    format_specialist_step_info,
    prune_by_telemetry,
)
from agent_utilities.graph.routing.strategies.team_reuse import select_reusable_team


def _spec(name, description="d"):
    return types.SimpleNamespace(name=name, description=description)


# ---- R2: TeamConfig reuse -----------------------------------------------------


def test_r2_select_reusable_team():
    team = types.SimpleNamespace(success_rate=0.9, reuse_threshold=0.7)
    assert select_reusable_team([team]) is team
    weak = types.SimpleNamespace(success_rate=0.5, reuse_threshold=0.7)
    assert select_reusable_team([weak]) is None
    assert select_reusable_team([]) is None
    assert select_reusable_team(None) is None


# ---- R4 / R5: self-model context ---------------------------------------------


def test_r4_proficiency_context_top5_sorted():
    rates = {"py": 0.9, "rust": 0.5, "go": 0.7, "js": 0.1, "c": 0.6, "java": 0.95}
    out = format_proficiency_context(rates)
    assert "YOUR DOMAIN PROFICIENCY" in out
    # Highest first, capped at 5 -> "js" (0.1) is dropped.
    assert out.index("java") < out.index("py") < out.index("go")
    assert "js" not in out
    assert format_proficiency_context({}) == ""


def test_r5_pheromone_affinities_top_domain():
    trails = {"agentA": {"web": 0.8, "math": 0.2}, "agentB": {"db": 0.9}}
    out = format_pheromone_affinities(trails)
    assert "SPECIALIST AFFINITIES" in out
    assert "agentB → db (affinity: 90%)" in out
    assert "agentA → web (affinity: 80%)" in out
    assert format_pheromone_affinities(None) == ""


def test_self_model_context_combines_r4_then_r5():
    current = types.SimpleNamespace(
        domain_success_rates={"py": 0.9},
        pheromone_trails={"a": {"web": 0.8}},
    )
    out = self_model_context(current)
    assert out.index("DOMAIN PROFICIENCY") < out.index("SPECIALIST AFFINITIES")
    # R5 only injected when R4 present (monolith ordering).
    only_trails = types.SimpleNamespace(
        domain_success_rates={}, pheromone_trails={"a": {"x": 0.5}}
    )
    assert self_model_context(only_trails) == ""
    assert self_model_context(None) == ""


# ---- R6: filtered specialist injection ---------------------------------------


def test_r6_step_info_lists_relevant_and_other_names():
    relevant = [_spec("python", "py dev"), _spec("rust", "rust dev")]
    tags = {"python": "py", "rust": "rs", "docs": "writer"}
    out = format_specialist_step_info(relevant, tags)
    assert "- python: py dev" in out
    assert "Other available specialists (request if needed): docs" in out
    # Empty relevant -> full tag list.
    assert format_specialist_step_info([], tags).count("- ") == 3


# ---- R7: reward-driven (pheromone) optimization ------------------------------


def test_r7_filter_by_pheromone_drops_low_affinity():
    relevant = [_spec("good"), _spec("bad")]
    trails = {"good": {"x": 0.9}, "bad": {"x": 0.05}}  # bad avg < 0.1
    out = [a.name for a in filter_by_pheromone(relevant, trails)]
    assert out == ["good"]
    # No trails -> untouched.
    assert len(filter_by_pheromone(relevant, None)) == 2


# ---- R8: telemetry-driven optimization ---------------------------------------


def test_r8_prune_by_telemetry_drops_anomalous():
    relevant = [_spec("healthy"), _spec("flaky")]
    anomalies = {"flaky": 9, "healthy": 1}
    out = [a.name for a in prune_by_telemetry(relevant, anomalies)]
    assert out == ["healthy"]
    assert len(prune_by_telemetry(relevant, {})) == 2


# ---- R9: subtask + wide-search instructions ----------------------------------


def test_r9_instructions_contain_both_sections():
    txt = subtask_and_widesearch_instructions()
    assert "SUBTASK SPECIFICATION (CONCEPT:AU-ORCH.planning.recursion-nesting-depth)" in txt
    assert "WIDE-SEARCH ORCHESTRATION (CONCEPT:AU-ORCH.planning.recursion-nesting-depth)" in txt
    assert "refined_subtask" in txt


# ---- R11: complexity detection (text heuristic) ------------------------------


def test_r11_is_complex_query():
    assert is_complex_query("architect a distributed system") is True
    assert is_complex_query("this is a complex task") is True
    assert is_complex_query("hi") is False
    # Many candidate specialists also signals complexity.
    assert is_complex_query("simple", num_specialists=4) is True
    assert is_complex_query("simple", num_specialists=2) is False


# ---- R10: RLM planning + fallback parser -------------------------------------


class _FakePlan:
    def __init__(self, **data):
        self.data = data

    @classmethod
    def model_validate(cls, data):
        if not isinstance(data, dict) or "steps" not in data:
            raise ValueError("invalid plan")
        return cls(**data)


def test_r10_rlm_instruction_and_parser():
    instr = rlm_plan_instruction("build an API")
    assert "build an API" in instr and "GraphPlan" in instr
    # Valid JSON -> a plan; invalid -> None (triggers the router's fallback parser).
    ok = parse_rlm_plan('{"steps": [], "metadata": {}}', _FakePlan)
    assert isinstance(ok, _FakePlan)
    assert parse_rlm_plan("not json", _FakePlan) is None
    assert parse_rlm_plan('{"nope": 1}', _FakePlan) is None


# ---- R13: multi-level fallback chain -----------------------------------------


def test_r13_fallback_prompt_and_name_matching():
    prompt = unstructured_fallback_prompt("BASE PROMPT")
    assert prompt.startswith("BASE PROMPT")
    assert "comma-separated" in prompt
    available = ["python_programmer", "rust_dev", "writer"]
    raw = "I would use the Python_Programmer and the writer."
    assert match_specialists_in_text(raw, available) == ["python_programmer", "writer"]
    assert match_specialists_in_text("", available) == []
