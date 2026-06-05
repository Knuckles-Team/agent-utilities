"""CONCEPT:ORCH-1.28 — Composable Skills + Generic Environment Adapter."""

from __future__ import annotations

import pytest

from agent_utilities.rlm.skills import (
    RegistryEnvironmentAdapter,
    Skill,
    merge_skills,
)


@pytest.mark.concept(id="ORCH-1.28")
def test_merge_skills_dedups_packages_and_headers_instructions():
    a = Skill(name="a", instructions="do A", packages=["httpx", "numpy"])
    b = Skill(name="b", instructions="do B", packages=["numpy", "pandas"])
    m = merge_skills([a, b])
    assert m.packages == ["httpx", "numpy", "pandas"]  # order-preserving dedup
    assert "## Skill: a" in m.instructions and "## Skill: b" in m.instructions
    assert "do A" in m.instructions and "do B" in m.instructions


@pytest.mark.concept(id="ORCH-1.28")
def test_merge_skills_raises_on_module_and_tool_conflict():
    a = Skill(name="a", modules={"util": "x=1"})
    b = Skill(name="b", modules={"util": "y=2"})
    with pytest.raises(ValueError, match="Module name conflict"):
        merge_skills([a, b])
    c = Skill(name="c", tools={"search": "def f(): ..."})
    d = Skill(name="d", tools={"search": "def g(): ..."})
    with pytest.raises(ValueError, match="Tool name conflict"):
        merge_skills([c, d])


@pytest.mark.concept(id="ORCH-1.28")
def test_generic_adapter_small_surface_and_preserved_evaluator():
    calls = {"login": lambda **k: "ok", "search": lambda q: [q.upper()]}
    seen = {}

    def evaluator(answer, log):
        seen["answer"] = answer
        return {"score": 1.0 if answer == "DONE" else 0.0, "calls": len(log)}

    env = RegistryEnvironmentAdapter(
        calls, descriptions={"login": "auth"}, evaluator=evaluator
    )
    assert env.list_items() == ["login", "search"]
    assert "auth" in env.describe("login")
    assert env.call("search", q="hi") == ["HI"]
    assert len(env.call_log) == 1
    result = env.submit("DONE")
    assert result["score"] == 1.0 and result["calls"] == 1
    assert seen["answer"] == "DONE"  # host evaluator received the answer


@pytest.mark.concept(id="ORCH-1.28")
def test_generic_adapter_unknown_item_raises():
    env = RegistryEnvironmentAdapter({"a": lambda: 1})
    with pytest.raises(KeyError):
        env.call("missing")


@pytest.mark.concept(id="ORCH-1.28")
def test_predict_rlm_mount_skill_unit():
    import os

    os.environ.setdefault("AGENT_UTILITIES_TESTING", "true")
    from pydantic import BaseModel

    from agent_utilities.rlm.predict_rlm import OutputField, PredictRLM

    class Sig(BaseModel):
        """t"""

        result: str = OutputField(description="r")

    rlm = PredictRLM(Sig)
    rlm.mount_skill_unit(Skill(name="s", instructions="SOP here", tools={"foo": "def foo(): ..."}))
    assert "foo" in rlm.skills
    assert "SOP here" in rlm._extra_instructions
    # Conflict on re-mount.
    with pytest.raises(ValueError):
        rlm.mount_skill_unit(Skill(name="s2", tools={"foo": "def foo(): ..."}))
