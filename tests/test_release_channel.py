"""Tests for the Release-Channel System (CONCEPT:OS-5.13).

Covers channel resolution + the visibility gate, the decorator/registry, AND the
live wiring into KG-driven specialist designation: an ``edge``-tagged callable
node is hidden on the default ``stable`` channel and visible on ``edge``.

@pytest.mark.concept("OS-5.13")
"""

from __future__ import annotations

import pytest

from agent_utilities.core.release_channel import (
    ENV_RELEASE_CHANNEL,
    ChannelRegistry,
    ReleaseChannel,
    active_channel,
    channel_visible,
    component_visible,
    get_component_channel,
    release_channel,
    reset_active_channel,
    set_active_channel,
)

pytestmark = pytest.mark.concept("OS-5.13")


@pytest.fixture(autouse=True)
def _reset_channel():
    reset_active_channel()
    yield
    reset_active_channel()


# ---------------------------------------------------------------------------
# Resolution + parsing
# ---------------------------------------------------------------------------


class TestResolution:
    def test_default_is_stable(self, monkeypatch):
        monkeypatch.delenv(ENV_RELEASE_CHANNEL, raising=False)
        reset_active_channel()
        assert active_channel(refresh=True) == ReleaseChannel.STABLE

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv(ENV_RELEASE_CHANNEL, "edge")
        assert active_channel(refresh=True) == ReleaseChannel.EDGE

    def test_parse_aliases(self):
        assert ReleaseChannel.parse("canary") == ReleaseChannel.EDGE
        assert ReleaseChannel.parse("ga") == ReleaseChannel.STABLE
        assert ReleaseChannel.parse("preview") == ReleaseChannel.BETA

    def test_parse_unknown_defaults_stable(self):
        assert ReleaseChannel.parse("nonsense") == ReleaseChannel.STABLE


# ---------------------------------------------------------------------------
# Visibility gate
# ---------------------------------------------------------------------------


class TestVisibility:
    def test_edge_hidden_on_stable(self):
        assert channel_visible("edge", ReleaseChannel.STABLE) is False

    def test_edge_visible_on_edge(self):
        assert channel_visible("edge", ReleaseChannel.EDGE) is True

    def test_beta_visible_on_edge_and_beta(self):
        assert channel_visible("beta", ReleaseChannel.EDGE) is True
        assert channel_visible("beta", ReleaseChannel.BETA) is True
        assert channel_visible("beta", ReleaseChannel.STABLE) is False

    def test_stable_visible_everywhere(self):
        for ch in ReleaseChannel:
            assert channel_visible("stable", ch) is True


# ---------------------------------------------------------------------------
# Decorator + registry
# ---------------------------------------------------------------------------


class TestDecoratorRegistry:
    def test_decorator_stamps_channel(self):
        @release_channel("edge")
        class Experimental:
            pass

        assert get_component_channel(Experimental) == ReleaseChannel.EDGE

    def test_component_visible_uses_active(self):
        @release_channel("edge")
        def edge_feature():
            return 1

        set_active_channel("stable")
        assert component_visible(edge_feature) is False
        set_active_channel("edge")
        assert component_visible(edge_feature) is True

    def test_registry_filters_by_channel(self):
        reg = ChannelRegistry()
        reg.register("stable_tool", object(), channel="stable")
        reg.register("edge_tool", object(), channel="edge")

        set_active_channel("stable")
        active = reg.active()
        assert "stable_tool" in active
        assert "edge_tool" not in active

        set_active_channel("edge")
        active = reg.active()
        assert "edge_tool" in active and "stable_tool" in active

    def test_registry_all_ignores_channel(self):
        reg = ChannelRegistry()
        reg.register("a", object(), channel="edge")
        reg.register("b", object(), channel="stable")
        assert set(reg.all()) == {"a", "b"}


# ---------------------------------------------------------------------------
# LIVE-PATH: channel filters the designation index
# ---------------------------------------------------------------------------


class _FakeGraph:
    def __init__(self, nodes: dict[str, dict]):
        self._nodes = nodes

    def node_ids(self):
        return list(self._nodes)

    def _get_node_properties(self, nid):
        return self._nodes.get(nid, {})


class _FakeEngine:
    def __init__(self, nodes):
        self.graph = _FakeGraph(nodes)
        self.backend = None
        self._designation_index = None


class TestDesignationChannelLivePath:
    """Wire-first: capability_designation excludes off-channel callable nodes."""

    def _engine(self):
        emb = [0.1, 0.2, 0.3]
        return _FakeEngine(
            {
                "skill:stable": {
                    "type": "skill",
                    "embedding": emb,
                    "capabilities": ["search"],
                    "release_channel": "stable",
                },
                "skill:edge": {
                    "type": "skill",
                    "embedding": emb,
                    "capabilities": ["search"],
                    "release_channel": "edge",
                },
            }
        )

    def test_edge_node_excluded_on_stable_live_path(self):
        from agent_utilities.graph.routing.enrichers.capability_designation import (
            _callable_nodes_with_embeddings,
        )

        set_active_channel("stable")
        engine = self._engine()
        ids = {n["id"] for n in _callable_nodes_with_embeddings(engine)}
        assert "skill:stable" in ids
        assert "skill:edge" not in ids  # edge hidden on stable

    def test_edge_node_included_on_edge_live_path(self):
        from agent_utilities.graph.routing.enrichers.capability_designation import (
            _callable_nodes_with_embeddings,
        )

        set_active_channel("edge")
        engine = self._engine()
        ids = {n["id"] for n in _callable_nodes_with_embeddings(engine)}
        assert {"skill:stable", "skill:edge"} <= ids
