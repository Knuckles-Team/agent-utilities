"""Task-aware / evolvable / ontology-mapped sampling profiles.

Covers all five layers of CONCEPT:ORCH-1.57 + AHE-3.38 + KG-2.93/2.94/2.95:
A per-call threading, B task-aware selection, C evolution, D ontology, E surfaces.
"""

from __future__ import annotations

import json

import pytest

from agent_utilities.agent.sampling_profile import (
    DEFAULT_PROFILE,
    SamplingProfile,
    attach_profile_resolver,
    classify_task,
    resolve_sampling_profile,
)
from agent_utilities.models.model_registry import (
    ModelRegistry,
    inference_owl_ttl,
    load_active_registry,
    reset_active_registry,
)


@pytest.fixture(autouse=True)
def _fresh_registry():
    """Each test starts from the curated built-ins, never a leaked learned profile."""
    reset_active_registry()
    yield
    reset_active_registry()


# ── Layer A — per-call threading ─────────────────────────────────────────────


def test_to_model_settings_merges_extra_body_without_clobber():
    base = {
        "temperature": 0.7,
        "top_p": 1.0,
        "max_tokens": 16384,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    prof = SamplingProfile(task_class="extraction", temperature=0.0, top_k=20, min_p=0.0)
    ms = prof.to_model_settings(base)
    assert ms["temperature"] == 0.0
    # vLLM knobs land in extra_body, and the pre-existing key survives.
    assert ms["extra_body"]["top_k"] == 20
    assert ms["extra_body"]["min_p"] == 0.0
    assert ms["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}
    # unset knob inherits the base value (pydantic-ai replaces, so we build from base)
    assert ms["max_tokens"] == 16384


def test_default_profile_is_a_noop_over_base():
    base = {"temperature": 0.7, "top_p": 1.0}
    assert dict(DEFAULT_PROFILE.to_model_settings(base)) == base


def test_attach_profile_resolver_threads_per_call():
    captured = {}

    class FakeAgent:
        def run(self, user_prompt=None, **kwargs):
            captured["ms"] = kwargs.get("model_settings")
            return "ok"

    base = {"temperature": 0.7, "top_p": 1.0, "max_tokens": 16384}
    agent = FakeAgent()
    attach_profile_resolver(agent, base)

    agent.run("implement a python function and fix the bug")
    assert captured["ms"]["temperature"] <= 0.2  # code → deterministic
    assert "top_k" in captured["ms"].get("extra_body", {})

    agent.run("brainstorm creative product names")
    assert captured["ms"]["temperature"] >= 0.9  # brainstorm → exploratory


def test_attach_respects_explicit_model_settings():
    captured = {}

    class FakeAgent:
        def run(self, user_prompt=None, **kwargs):
            captured["ms"] = kwargs.get("model_settings")

    agent = FakeAgent()
    attach_profile_resolver(agent, {"temperature": 0.7})
    agent.run("implement code", model_settings={"temperature": 0.42})
    assert captured["ms"]["temperature"] == 0.42  # caller wins, no override


def test_attach_is_idempotent():
    class FakeAgent:
        def run(self, user_prompt=None, **kwargs):
            return kwargs.get("model_settings")

    agent = FakeAgent()
    attach_profile_resolver(agent, {"temperature": 0.7})
    first = agent.run
    attach_profile_resolver(agent, {"temperature": 0.7})
    assert agent.run is first  # not double-wrapped


# ── Layer B — task-aware selection ───────────────────────────────────────────


@pytest.mark.parametrize(
    "text,expected",
    [
        ("extract entities from this document", "extraction"),
        ("brainstorm some ideas", "brainstorm"),
        ("implement a rust function", "code"),
        ("plan the roadmap", "plan"),
        ("hello there", "default"),
    ],
)
def test_classify_task(text, expected):
    assert classify_task(text) == expected


def test_registry_profiles_by_task_and_role():
    reg = ModelRegistry()
    assert reg.pick_profile_for_task("code").temperature == pytest.approx(0.1)
    assert reg.pick_profile_for_task("brainstorm").temperature == pytest.approx(1.0)
    # role → task-class mapping
    assert reg.pick_profile_for_role("learner").task_class == "extraction"
    assert reg.pick_profile_for_role("rlm-proposer").task_class == "reasoning"
    # unknown → inherit-everything default
    assert reg.pick_profile_for_task("nonexistent") is DEFAULT_PROFILE


def test_router_populates_sampling_profile():
    from agent_utilities.graph.adaptive_agent_router import (
        RoutingCandidate,
        RoutingPrimitive,
        RuleBasedPolicy,
    )

    cands = [RoutingCandidate(model_id="m1", primitive=RoutingPrimitive.DIRECT, confidence=0.9)]
    policy = RuleBasedPolicy()
    d_code = policy.route("extract entities from this invoice", cands)
    assert d_code.sampling_profile.temperature <= 0.2
    d_brain = policy.route("brainstorm creative ideas", cands)
    assert d_brain.sampling_profile.temperature >= 0.9


def test_rlm_role_profiles():
    # repl.py threads these two roles by depth.
    assert resolve_sampling_profile(role="rlm-proposer").task_class == "reasoning"
    assert resolve_sampling_profile(role="rlm-executor").task_class == "code"


# ── Layer C — evolution ──────────────────────────────────────────────────────


def test_evolve_profile_promotes_best_and_registry_serves_it():
    from agent_utilities.harness.variant_pool import VariantPool
    from agent_utilities.knowledge_graph.retrieval.capability_index import CapabilityIndex
    import random

    reg = ModelRegistry()
    ci = CapabilityIndex(dim=8)
    vp = VariantPool.__new__(VariantPool)

    # Reward lower temperature for 'code'.
    def evaluator(p):
        t = p.temperature if p.temperature is not None else 1.0
        return max(0.0, 1.0 - t)

    rng = random.Random(7)
    promoted = None
    for _ in range(8):
        promoted = vp.evolve_profile(reg, "code", ci, evaluator, count=5, rng=rng)

    assert promoted.source == "learned"
    assert promoted.temperature < 0.1  # drifted below the 0.1 seed
    # the registry now serves the learned profile (Layer B consumes it)
    assert reg.pick_profile_for_task("code").temperature == promoted.temperature


def test_mutate_profile_stays_within_bounds():
    from agent_utilities.harness.variant_pool import VariantPool
    from agent_utilities.knowledge_graph.ontology.value_types import (
        sampling_profile_violations,
    )
    import random

    vp = VariantPool.__new__(VariantPool)
    base = SamplingProfile(task_class="code", temperature=0.1, top_p=0.9, top_k=20, min_p=0.0)
    for child in vp.mutate_profile(base, count=20, jitter=0.9, rng=random.Random(1)):
        assert sampling_profile_violations(child.model_dump()) == []


# ── Layer D — ontology mapping ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "vt_name,value,valid",
    [
        ("Temperature", 0.7, True),
        ("Temperature", 2.5, False),
        ("TopP", 1.5, False),
        ("TopK", 0, False),
        ("TopK", 20, True),
        ("RepetitionPenalty", 0, False),  # exclusive min
        ("RepetitionPenalty", 1.1, True),
    ],
)
def test_inference_value_type_bounds(vt_name, value, valid):
    from agent_utilities.knowledge_graph.ontology.value_types import validate_value_type

    assert validate_value_type(vt_name, value) is valid


def test_inference_profile_interface_and_model_implementer():
    from agent_utilities.knowledge_graph.ontology.interfaces import (
        DEFAULT_INTERFACE_REGISTRY as R,
    )

    assert R.get("InferenceProfile") is not None
    assert "model" in R.find_implementers("SamplingConfigurable")


def test_inference_links_registered():
    from agent_utilities.knowledge_graph.ontology.links import DEFAULT_LINK_REGISTRY as L

    for name in ("model_has_profile", "profile_tuned_for", "agent_uses_profile"):
        assert L.get(name) is not None


def test_owl_projection_contains_profiles():
    ttl = inference_owl_ttl(load_active_registry())
    assert "kg:InferenceProfile" in ttl
    assert "kg:tunedFor" in ttl
    assert "kg:profile_code" in ttl


def test_sampling_profile_violations_gate():
    from agent_utilities.knowledge_graph.ontology.value_types import (
        sampling_profile_violations,
    )

    assert sampling_profile_violations({"temperature": 2.5}) != []
    assert sampling_profile_violations({"temperature": 0.1, "top_k": 20}) == []


# ── Layer E — two surfaces ───────────────────────────────────────────────────


@pytest.fixture
def sampling_tool():
    # Use the canonical full registration so we never leave REGISTERED_TOOLS in a
    # partial state (which would starve ensure_tools_registered's short-circuit and
    # break the gateway/MCP parity test that runs in the same session).
    from agent_utilities.mcp import kg_server

    kg_server.ensure_tools_registered()
    return kg_server.REGISTERED_TOOLS["ontology_sampling_profile"]


def _call_tool(tool, **kwargs):
    # Provide every arg explicitly (FastMCP substitutes Field defaults; a raw call
    # would otherwise leave FieldInfo objects in place).
    args = {"action": "list", "task_class": "", "task_text": "", "role": "", "profile_json": "{}"}
    args.update(kwargs)
    return json.loads(tool(**args))


def test_tool_list_describe_resolve(sampling_tool):
    listed = _call_tool(sampling_tool, action="list")["profiles"]
    assert "code" in listed and "brainstorm" in listed
    assert _call_tool(sampling_tool, action="describe", task_class="code")["temperature"] == pytest.approx(0.1)
    assert (
        _call_tool(sampling_tool, action="resolve", task_text="extract fields from invoice")["task_class"]
        == "extraction"
    )


def test_tool_set_validates_and_persists(sampling_tool):
    bad = _call_tool(sampling_tool, action="set", task_class="code", profile_json='{"temperature": 1.99, "top_p": 5.0}')
    assert "error" in bad  # top_p out of bounds
    good = _call_tool(sampling_tool, action="set", task_class="code", profile_json='{"temperature": 0.05, "top_k": 10}')
    assert good["set"]["temperature"] == pytest.approx(0.05)
    # persists in the process-global registry
    assert _call_tool(sampling_tool, action="describe", task_class="code")["temperature"] == pytest.approx(0.05)


def test_tool_owl(sampling_tool):
    assert "kg:InferenceProfile" in _call_tool(sampling_tool, action="owl")["owl"]


def test_route_map_registered():
    from agent_utilities.mcp import kg_server

    assert kg_server.ACTION_TOOL_ROUTES["ontology_sampling_profile"] == "/ontology/sampling-profiles"
