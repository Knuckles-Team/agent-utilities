"""Tests for the shared-GPU concurrency budget (CONCEPT:KG-2.146).

No live GPU: a fake config (chat + embedding models sharing one ``gpu_group``) and
the env-driven ``GPU_CONCURRENCY_BUDGETS`` are monkeypatched so the cap math, the
reserved-priority slice, and the no-budget-no-regression path are deterministic.
"""

from __future__ import annotations

import pytest

from agent_utilities.core import gpu_group_budget as gb
from agent_utilities.core import model_capacity_autoscale as mod


class _ChatModel:
    def __init__(self, mid, base_url, gpu_group=None, level="normal"):
        self.id = mid
        self.base_url = base_url
        self.gpu_group = gpu_group
        self.intelligence_level = level


class _EmbedModel:
    def __init__(self, mid, base_url, gpu_group=None):
        self.id = mid
        self.base_url = base_url
        self.gpu_group = gpu_group


class _FakeConfig:
    """Minimal config: chat ``qwen`` + embed ``bge-m3`` share GPU ``gb10``."""

    def __init__(self, *, group="gb10", chat_cap=8, embed_cap=4):
        self._chat = _ChatModel("qwen3.5-9b", "http://vllm.arpa/v1", group)
        self._embed = _EmbedModel("bge-m3", "http://vllm-embed.arpa/v1", group)
        self.chat_models = [self._chat]
        self.embedding_models = [self._embed]
        self._chat_cap = chat_cap
        self._embed_cap = embed_cap

    def _resolve(self, model):
        key = (model or "").strip().lower()
        if key in ("", "chat", "default"):
            return self._chat
        if key in ("embedding", "embed"):
            return self._embed
        for m in (*self.chat_models, *self.embedding_models):
            if m.id == model:
                return m
        return None

    def model_endpoint(self, model):
        m = self._resolve(model)
        return (m.id, m.base_url) if m else (None, None)

    def model_capacity(self, model):
        m = self._resolve(model)
        if m is self._chat:
            return self._chat_cap
        if m is self._embed:
            return self._embed_cap
        return 1

    def gpu_group(self, model):
        m = self._resolve(model)
        if m is None:
            return None
        if m.gpu_group:
            return m.gpu_group.lower()
        from urllib.parse import urlsplit

        return urlsplit(m.base_url or "").netloc.lower() or None


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    mod.reset_adaptive_controllers()
    gb.reset_gpu_group_budgets()
    monkeypatch.setenv("KG_ADAPTIVE_CONCURRENCY", "1")
    monkeypatch.setenv("MODEL_MAX_CONCURRENCY", "512")
    yield
    mod.reset_adaptive_controllers()
    gb.reset_gpu_group_budgets()


# --- gpu_group resolution ---------------------------------------------------


def test_gpu_group_explicit_tag_wins_over_base_url(monkeypatch):
    cfg = _FakeConfig(group="gb10")
    monkeypatch.setattr("agent_utilities.core.config.config", cfg, raising=False)
    # both models tagged gb10 → same group despite different endpoints
    assert cfg.gpu_group("qwen3.5-9b") == "gb10"
    assert cfg.gpu_group("bge-m3") == "gb10"


def test_gpu_group_defaults_to_base_url_host_when_untagged(monkeypatch):
    cfg = _FakeConfig(group=None)
    monkeypatch.setattr("agent_utilities.core.config.config", cfg, raising=False)
    assert cfg.gpu_group("qwen3.5-9b") == "vllm.arpa"
    assert cfg.gpu_group("bge-m3") == "vllm-embed.arpa"  # different hosts → diff groups


# --- budget caps the SUM of member targets ----------------------------------


def _ramp(model, floor, fetcher, n=6):
    """Drive a model's controller up via fast polling, return final adaptive cap."""
    cap = mod.adaptive_capacity(model, floor, fetcher=fetcher)
    ctrl = mod._get_controller(model, floor)
    assert ctrl is not None
    ctrl.min_poll_interval_s = 0.0
    for _ in range(n):
        cap = mod.adaptive_capacity(model, floor, fetcher=fetcher)
    return cap


def _busy(_url):
    # vLLM running high, no capacity waiting → controller ramps UP.
    return (
        'vllm:num_requests_running{model_name="bge-m3"} 1000\n'
        'vllm:num_requests_running{model_name="qwen3.5-9b"} 1000\n'
    )


def test_budget_caps_sum_priority_keeps_floor_embedding_squeezed(monkeypatch):
    # GPU budget 10: chat floor 4 (priority), embed floor 4 (best-effort).
    cfg = _FakeConfig(group="gb10", chat_cap=4, embed_cap=4)
    monkeypatch.setattr("agent_utilities.core.config.config", cfg, raising=False)
    monkeypatch.setenv("GPU_CONCURRENCY_BUDGETS", '{"gb10": 10}')

    # Both ramp hard. Chat (priority) should be allowed to grow; embedding is capped
    # so chat's reserved floor (4) is always subtracted from embedding's allowance.
    _ramp("qwen3.5-9b", 4, _busy)
    embed_cap = _ramp("bge-m3", 4, _busy)

    # Embedding is squeezed toward its floor: budget(10) − chat floor(4) − chat
    # current target leaves little; never below its own floor of 4.
    assert embed_cap >= 4
    snap = mod.get_utilization("bge-m3")
    chat_snap = mod.get_utilization("qwen3.5-9b")
    # The SUM of resolved targets must respect the budget given demand: embedding's
    # allowed share = budget − chat reserved floor − (best-effort peers = none).
    assert embed_cap <= 10 - 4  # chat's reserved slice is protected
    # Chat always keeps at least its floor.
    assert chat_snap["current_target"] >= 4
    assert snap["group_budget"] == 10


def test_chat_idle_lets_embedding_reclaim_up_to_budget(monkeypatch):
    cfg = _FakeConfig(group="gb10", chat_cap=4, embed_cap=4)
    monkeypatch.setattr("agent_utilities.core.config.config", cfg, raising=False)
    monkeypatch.setenv("GPU_CONCURRENCY_BUDGETS", '{"gb10": 12}')

    # Chat registered (member) but idle at its floor; embedding ramps hard.
    mod.adaptive_capacity("qwen3.5-9b", 4, fetcher=lambda _u: "")  # register, idle
    embed_cap = _ramp("bge-m3", 4, _busy, n=12)

    # Only chat's reserved floor (4) is held back → embedding can reach budget − 4.
    assert embed_cap <= 12 - 4
    assert embed_cap > 4  # it reclaimed headroom beyond its own floor


# --- no budget configured → no regression -----------------------------------


def test_no_budget_configured_is_per_model_unchanged(monkeypatch):
    cfg = _FakeConfig(group="gb10", chat_cap=4, embed_cap=4)
    monkeypatch.setattr("agent_utilities.core.config.config", cfg, raising=False)
    monkeypatch.delenv("GPU_CONCURRENCY_BUDGETS", raising=False)

    # With no budget, both models ramp to their full per-model adaptive target.
    embed_cap = _ramp("bge-m3", 4, _busy, n=10)
    chat_cap = _ramp("qwen3.5-9b", 4, _busy, n=10)
    # Each grew well past its floor and their sum exceeds any small budget — proving
    # the cap is NOT applied when no budget is set.
    assert embed_cap > 4
    assert chat_cap > 4
    assert embed_cap + chat_cap > 8

    snap = mod.get_utilization("bge-m3")
    assert snap["gpu_group"] == "gb10"
    assert snap["group_budget"] is None  # no budget → fields are None


# --- get_utilization exposes the group fields -------------------------------


def test_utilization_exposes_group_fields(monkeypatch):
    cfg = _FakeConfig(group="gb10", chat_cap=4, embed_cap=4)
    monkeypatch.setattr("agent_utilities.core.config.config", cfg, raising=False)
    monkeypatch.setenv("GPU_CONCURRENCY_BUDGETS", '{"gb10": 10}')

    _ramp("qwen3.5-9b", 4, _busy)
    _ramp("bge-m3", 4, _busy)
    snap = mod.get_utilization("bge-m3")
    assert snap["gpu_group"] == "gb10"
    assert snap["group_budget"] == 10
    assert isinstance(snap["group_used"], int)
    assert isinstance(snap["group_allowed_for_this_model"], int)
    # embedding's allowed share never exceeds budget − chat's reserved floor
    assert snap["group_allowed_for_this_model"] <= 10 - 4


# --- coordinator unit (pure arithmetic) -------------------------------------


def test_coordinator_priority_reservation_math():
    budget = gb.GpuGroupBudget(group="g", budget=10)
    budget.upsert("chat", floor=4, priority=True)
    budget.upsert("embed", floor=2, priority=False)
    # chat demands a lot, embed demands a lot
    budget.report("chat", 8)
    budget.report("embed", 8)
    # embed allowed = budget(10) − chat reserved floor(4) − (no best-effort peers) = 6
    assert budget.allowed_for("embed") == 6
    # chat allowed = budget(10) − (no priority peers) − embed current target(8) = 2,
    # but chat is priority so it is floored at its own floor (4): guaranteed slice.
    assert budget.allowed_for("chat") == 4


def test_coordinator_unknown_member_returns_none():
    budget = gb.GpuGroupBudget(group="g", budget=10)
    assert budget.allowed_for("nope") is None


def test_reserved_role_classification(monkeypatch):
    monkeypatch.delenv("GPU_RESERVED_ROLES", raising=False)
    assert mod._role_hint("chat") == "chat"
    assert mod._role_hint("embedding") == "embedding"
    # chat is priority, embedding is best-effort by default
    assert gb._is_priority_role("chat", "chat") is True
    assert gb._is_priority_role("embedding", "embedding") is False
