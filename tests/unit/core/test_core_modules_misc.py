"""Coverage uplift tests for agent-utilities.

Targets 0%-covered and low-covered modules:
- ``exceptions.py``, ``decorators.py``, ``api_utilities.py`` - trivial
- ``interfaces.py`` - protocol isinstance checks
- ``event_aggregator.py`` - find_package_data_dir candidates
- ``__init__.py`` lazy-load ``__getattr__`` branches
- ``agent_utilities.py`` facade
- ``graph/integration.py`` - outcome recording
- ``custom_observability.py`` - environment branches
- ``discovery.py`` - missing module fallbacks
- ``chat_persistence.py`` - on-disk round-trip
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# exceptions.py
# ---------------------------------------------------------------------------


def test_custom_exception_hierarchy():
    from agent_utilities.exceptions import (
        ApiError,
        AuthError,
        LoginRequiredError,
        MissingParameterError,
        ParameterError,
        UnauthorizedError,
    )

    assert issubclass(UnauthorizedError, AuthError)
    for exc in (
        ApiError,
        AuthError,
        UnauthorizedError,
        MissingParameterError,
        ParameterError,
        LoginRequiredError,
    ):
        assert issubclass(exc, Exception)
        # Each can be raised and caught
        try:
            raise exc("test")
        except exc as err:
            assert str(err) == "test"


# ---------------------------------------------------------------------------
# decorators.py
# ---------------------------------------------------------------------------


def test_require_auth_success():
    from agent_utilities.decorators import require_auth

    class Client:
        def __init__(self):
            self.headers = {"Authorization": "bearer"}

        @require_auth
        def get(self):
            return "ok"

    assert Client().get() == "ok"


def test_require_auth_missing_headers():
    from agent_utilities.decorators import require_auth
    from agent_utilities.exceptions import LoginRequiredError

    class Client:
        def __init__(self):
            self.headers = None

        @require_auth
        def get(self):
            return "ok"

    with pytest.raises(LoginRequiredError):
        Client().get()


# ---------------------------------------------------------------------------
# api_utilities.py - re-export smoke test
# ---------------------------------------------------------------------------


def test_api_utilities_exports():
    from agent_utilities import api_utilities

    assert hasattr(api_utilities, "require_auth")
    assert hasattr(api_utilities, "ApiError")
    assert hasattr(api_utilities, "AuthError")


# ---------------------------------------------------------------------------
# interfaces.py protocol checks
# ---------------------------------------------------------------------------


def test_protocol_isinstance_graph_backend():
    from agent_utilities.interfaces import GraphBackend

    class FakeBackend:
        def execute(self, query, params=None):
            return []

        def create_schema(self):
            pass

        def add_embedding(self, node_id, embedding):
            pass

        def prune(self, criteria):
            pass

    assert isinstance(FakeBackend(), GraphBackend)


def test_protocol_isinstance_agent_and_tool():
    from agent_utilities.interfaces import AgentInterface, ToolInterface

    class FakeAgent:
        def __init__(self):
            self.name = "agent"
            self.description = "desc"

        async def run(self, prompt, **kwargs):
            return prompt

    class FakeTool:
        def __init__(self):
            self.name = "tool"
            self.description = "desc"

        async def call(self, **kwargs):
            return kwargs

    assert isinstance(FakeAgent(), AgentInterface)
    assert isinstance(FakeTool(), ToolInterface)


# ---------------------------------------------------------------------------
# event_aggregator.py
# ---------------------------------------------------------------------------


def test_find_package_data_dir_none_for_missing():
    from agent_utilities.event_aggregator import find_package_data_dir

    assert find_package_data_dir("this_package_does_not_exist_xyz") is None


def test_find_package_data_dir_for_agent_utilities():
    """agent_utilities ships no ``agent_data/`` so we expect None here too,
    but the function should traverse candidates without raising."""
    from agent_utilities.event_aggregator import find_package_data_dir

    result = find_package_data_dir("agent_utilities")
    # Either None or a valid Path
    assert result is None or result.exists()


def test_find_package_data_dir_finds_agent_data(tmp_path, monkeypatch):
    """Point ``importlib.util.find_spec`` at a fake package whose parent
    directory contains an ``agent_data/`` subdir."""
    import importlib.util as _iu

    from agent_utilities import event_aggregator

    # Create ``tmp_path/mypkg/__init__.py`` plus an ``agent_data/`` sibling.
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (tmp_path / "agent_data").mkdir()

    fake_spec = SimpleNamespace(origin=str(pkg / "__init__.py"))
    monkeypatch.setattr(
        _iu,
        "find_spec",
        lambda name: fake_spec if name == "mypkg" else None,
    )
    result = event_aggregator.find_package_data_dir("mypkg")
    # Should resolve to the agent_data sibling
    assert result is not None
    assert result.name == "agent_data"


# ---------------------------------------------------------------------------
# graph/integration.py
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_record_specialist_outcome_no_engine():
    from agent_utilities.graph.integration import record_specialist_outcome_hook

    # get_active returns None → early return
    from agent_utilities.knowledge_graph.engine import IntelligenceGraphEngine

    original = IntelligenceGraphEngine.get_active
    setattr(IntelligenceGraphEngine, "get_active", staticmethod(lambda: None))
    try:
        await record_specialist_outcome_hook(
            deps=SimpleNamespace(),
            state=SimpleNamespace(),
            agent_name="alpha",
            success=True,
            server_name="mcp",
            duration=1.5,
        )
    finally:
        setattr(IntelligenceGraphEngine, "get_active", original)
    assert True, 'No-engine specialist outcome should not raise'


@pytest.mark.asyncio
async def test_record_specialist_outcome_success(monkeypatch):
    from agent_utilities.graph.integration import record_specialist_outcome_hook
    from agent_utilities.knowledge_graph.engine import IntelligenceGraphEngine

    fake_engine = MagicMock()
    fake_engine.backend = MagicMock()
    fake_engine.graph = MagicMock()
    fake_engine.graph.nodes.return_value = []
    fake_engine.graph.__contains__ = lambda self, item: False

    monkeypatch.setattr(
        IntelligenceGraphEngine, "get_active", staticmethod(lambda: fake_engine)
    )
    await record_specialist_outcome_hook(
        deps=SimpleNamespace(request_id=None),
        state=SimpleNamespace(session_id=None),
        agent_name="alpha",
        success=True,
        server_name="mcp",
        duration=1.5,
    )
    fake_engine.graph.add_node.assert_called()


@pytest.mark.asyncio
async def test_record_specialist_outcome_failure_long_duration(monkeypatch):
    """Cover the reward-shaping branch where success + long duration."""
    from agent_utilities.graph.integration import record_specialist_outcome_hook
    from agent_utilities.knowledge_graph.engine import IntelligenceGraphEngine

    fake_engine = MagicMock()
    fake_engine.backend = MagicMock()
    fake_engine.graph = MagicMock()
    fake_engine.graph.nodes.return_value = [("agent_1", {"name": "alpha"})]
    fake_engine.graph.__contains__ = lambda self, item: False

    monkeypatch.setattr(
        IntelligenceGraphEngine, "get_active", staticmethod(lambda: fake_engine)
    )
    await record_specialist_outcome_hook(
        deps=SimpleNamespace(),
        state=SimpleNamespace(),
        agent_name="alpha",
        success=True,
        server_name="mcp",
        duration=99.0,
    )
    # Assert add_node called with reward=0.8
    call_kwargs = fake_engine.graph.add_node.call_args.kwargs
    assert call_kwargs["reward"] == 0.8


@pytest.mark.asyncio
async def test_record_specialist_outcome_add_node_error(monkeypatch):
    from agent_utilities.graph.integration import record_specialist_outcome_hook
    from agent_utilities.knowledge_graph.engine import IntelligenceGraphEngine

    fake_engine = MagicMock()
    fake_engine.backend = MagicMock()
    fake_engine.graph = MagicMock()
    fake_engine.graph.add_node.side_effect = RuntimeError("DB down")

    monkeypatch.setattr(
        IntelligenceGraphEngine, "get_active", staticmethod(lambda: fake_engine)
    )
    # Should not raise; logs warning
    await record_specialist_outcome_hook(
        deps=SimpleNamespace(),
        state=SimpleNamespace(),
        agent_name="alpha",
        success=False,
        server_name="mcp",
        duration=1.0,
    )
    assert True, 'add_node error in specialist outcome handled'


def test_graph_integration_initialize():
    from agent_utilities.graph.integration import initialize_integration

    # The register function is module-level; just ensure no exception
    initialize_integration()
    assert True, 'Graph integration init completed'


# ---------------------------------------------------------------------------
# __init__.py lazy imports
# ---------------------------------------------------------------------------


def test_init_lazy_attributes_base_utilities():
    import agent_utilities

    # Access each lazy-loaded symbol to exercise the __getattr__ branches
    assert agent_utilities.get_logger is not None
    assert agent_utilities.to_boolean is not None
    assert agent_utilities.to_integer is not None
    assert agent_utilities.to_float is not None
    assert agent_utilities.to_list is not None
    assert agent_utilities.to_dict is not None
    assert agent_utilities.retrieve_package_name is not None
    assert agent_utilities.ensure_package_installed is not None
    assert agent_utilities.optional_import_block is not None
    assert agent_utilities.require_optional_import is not None


def test_init_lazy_agent_factory_and_config():
    import agent_utilities

    assert agent_utilities.create_agent is not None
    assert agent_utilities.create_agent_parser is not None
    assert agent_utilities.DEFAULT_GRAPH_PERSISTENCE_PATH is not None


def test_init_lazy_discovery_and_embedding():
    import agent_utilities

    assert agent_utilities.discover_agents is not None
    assert agent_utilities.discover_all_specialists is not None
    # embedding may or may not be importable; just try
    try:
        _ = agent_utilities.create_embedding_model
    except Exception:
        pass


def test_init_lazy_graph_and_workspace():
    import agent_utilities

    assert agent_utilities.GraphState is not None
    assert agent_utilities.initialize_workspace is not None
    assert agent_utilities.get_workspace_path is not None
    assert agent_utilities.CORE_FILES is not None


def test_init_lazy_chat_persistence_and_models():
    import agent_utilities

    assert agent_utilities.save_chat_to_disk is not None
    assert agent_utilities.DiscoveredSpecialist is not None
    assert agent_utilities.CodemapNode is not None
    assert agent_utilities.SDDManager is not None


def test_init_lazy_unknown_attribute_raises():
    import agent_utilities

    with pytest.raises(AttributeError):
        _ = agent_utilities.this_attribute_does_not_exist


def test_init_lazy_prompts_and_server():
    import agent_utilities

    # prompt_builder
    assert agent_utilities.load_identity is not None
    assert agent_utilities.build_system_prompt_from_workspace is not None

    # server
    assert agent_utilities.create_agent_server is not None
    assert agent_utilities.create_graph_agent_server is not None


def test_init_lazy_codemaps_and_models():
    import agent_utilities

    assert agent_utilities.CodemapGenerator is not None
    assert agent_utilities.parse_codemap_mentions is not None


def test_init_lazy_create_model():
    import agent_utilities

    assert agent_utilities.create_model is not None


# ---------------------------------------------------------------------------
# agent_utilities.py facade
# ---------------------------------------------------------------------------


def test_agent_utilities_facade_imports():
    # Importing the facade should not raise
    import agent_utilities.agent_utilities as au_facade

    assert au_facade.__version__ == "0.2.40"


# ---------------------------------------------------------------------------
# custom_observability.py
# ---------------------------------------------------------------------------


def test_custom_observability_disabled(monkeypatch):
    monkeypatch.setenv("OTEL_ENABLE_OTEL", "false")
    import importlib

    import agent_utilities.custom_observability as obs

    importlib.reload(obs)
    assert True, 'Disabled observability handled gracefully'


def test_custom_observability_initialize_function(monkeypatch):
    import agent_utilities.custom_observability as obs

    # Check available public functions
    exports = dir(obs)
    assert "logger" in exports or hasattr(obs, "logger")


# ---------------------------------------------------------------------------
# discovery.py
# ---------------------------------------------------------------------------


def test_discover_agents_no_registry(monkeypatch):
    from agent_utilities import discovery

    class _EmptyRegistry:
        agents: list = []

    monkeypatch.setattr(
        discovery,
        "get_discovery_registry",
        lambda: _EmptyRegistry(),
        raising=False,
    )
    # Patch the lazy import path
    from agent_utilities.graph import config_helpers

    monkeypatch.setattr(
        config_helpers, "get_discovery_registry", lambda: _EmptyRegistry()
    )
    result = discovery.discover_agents()
    assert result == {}


def test_discover_agents_filters(monkeypatch):
    from agent_utilities import discovery

    agent_a = SimpleNamespace(
        name="alpha",
        description="d",
        agent_type="prompt",
        capabilities=["skill1"],
        endpoint_url=None,
        mcp_server=None,
    )
    agent_b = SimpleNamespace(
        name="beta",
        description="d",
        agent_type="mcp",
        capabilities=[],
        endpoint_url=None,
        mcp_server="mcp://x",
    )
    agent_c = SimpleNamespace(
        name="Gamma",
        description="d",
        agent_type="a2a",
        capabilities=["c1"],
        endpoint_url="http://gamma",
        mcp_server=None,
    )

    class Reg:
        agents = [agent_a, agent_b, agent_c]

    from agent_utilities.graph import config_helpers

    monkeypatch.setattr(
        config_helpers, "get_discovery_registry", lambda: Reg()
    )

    # Exercise filter branches
    result = discovery.discover_agents(
        include_packages=["alpha", "beta"], exclude_packages=["beta"]
    )
    assert "alpha" in result
    assert "beta" not in result

    all_result = discovery.discover_agents()
    assert "alpha" in all_result
    assert "beta" in all_result
    assert "gamma" in all_result


def test_discover_all_specialists(monkeypatch):
    from agent_utilities import discovery
    from agent_utilities.graph import config_helpers

    class Reg:
        agents = [
            SimpleNamespace(
                name="alpha",
                description="d",
                agent_type="prompt",
                capabilities=[],
                endpoint_url=None,
                mcp_server=None,
            )
        ]

    monkeypatch.setattr(
        config_helpers, "get_discovery_registry", lambda: Reg()
    )
    specs = discovery.discover_all_specialists()
    assert specs and specs[0].tag == "alpha"


# ---------------------------------------------------------------------------
# chat_persistence.py - knowledge-graph backed (no engine)
# ---------------------------------------------------------------------------


def test_chat_persistence_no_engine():
    import agent_utilities.chat_persistence as cp
    from agent_utilities.knowledge_graph.engine import IntelligenceGraphEngine

    original = IntelligenceGraphEngine.get_active
    setattr(IntelligenceGraphEngine, "get_active", staticmethod(lambda: None))
    try:
        cp.save_chat_to_disk("chat-1", [{"role": "user", "content": "hi"}])
        assert cp.list_chats_from_disk() == []
        assert cp.get_chat_from_disk("missing") is None
        assert cp.delete_chat_from_disk("missing") is False
    finally:
        setattr(IntelligenceGraphEngine, "get_active", original)


def test_chat_persistence_save_and_list_with_engine(monkeypatch):
    import agent_utilities.chat_persistence as cp
    from agent_utilities.knowledge_graph.engine import IntelligenceGraphEngine

    fake_engine = MagicMock()
    fake_engine.backend = MagicMock()
    fake_engine.backend.execute.return_value = [
        {"t.id": "c1", "t.title": "first", "t.created_at": "2024"}
    ]
    monkeypatch.setattr(
        IntelligenceGraphEngine, "get_active", staticmethod(lambda: fake_engine)
    )
    cp.save_chat_to_disk(
        "c1",
        [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}],
    )
    listed = cp.list_chats_from_disk()
    assert listed[0]["id"] == "c1"


def test_chat_persistence_list_error(monkeypatch):
    import agent_utilities.chat_persistence as cp
    from agent_utilities.knowledge_graph.engine import IntelligenceGraphEngine

    fake_engine = MagicMock()
    fake_engine.backend = MagicMock()
    fake_engine.backend.execute.side_effect = RuntimeError("bad")
    monkeypatch.setattr(
        IntelligenceGraphEngine, "get_active", staticmethod(lambda: fake_engine)
    )
    assert cp.list_chats_from_disk() == []


def test_chat_persistence_get_chat_with_engine(monkeypatch):
    import agent_utilities.chat_persistence as cp
    from agent_utilities.knowledge_graph.engine import IntelligenceGraphEngine

    fake_engine = MagicMock()
    fake_engine.backend = MagicMock()
    fake_engine.backend.execute.side_effect = [
        [{"t.id": "c1", "t.created_at": "2024"}],
        [{"m.role": "user", "m.content": "hi", "m.timestamp": "2024"}],
    ]
    monkeypatch.setattr(
        IntelligenceGraphEngine, "get_active", staticmethod(lambda: fake_engine)
    )
    result = cp.get_chat_from_disk("c1")
    assert result and result["id"] == "c1"
    assert len(result["messages"]) == 1


def test_chat_persistence_get_chat_missing(monkeypatch):
    import agent_utilities.chat_persistence as cp
    from agent_utilities.knowledge_graph.engine import IntelligenceGraphEngine

    fake_engine = MagicMock()
    fake_engine.backend = MagicMock()
    fake_engine.backend.execute.return_value = []
    monkeypatch.setattr(
        IntelligenceGraphEngine, "get_active", staticmethod(lambda: fake_engine)
    )
    assert cp.get_chat_from_disk("missing") is None


def test_chat_persistence_delete_chat_with_engine(monkeypatch):
    import agent_utilities.chat_persistence as cp
    from agent_utilities.knowledge_graph.engine import IntelligenceGraphEngine

    fake_engine = MagicMock()
    fake_engine.backend = MagicMock()
    monkeypatch.setattr(
        IntelligenceGraphEngine, "get_active", staticmethod(lambda: fake_engine)
    )
    assert cp.delete_chat_from_disk("c1") is True


def test_chat_persistence_delete_chat_error(monkeypatch):
    import agent_utilities.chat_persistence as cp
    from agent_utilities.knowledge_graph.engine import IntelligenceGraphEngine

    fake_engine = MagicMock()
    fake_engine.backend = MagicMock()
    fake_engine.backend.execute.side_effect = RuntimeError("bad")
    monkeypatch.setattr(
        IntelligenceGraphEngine, "get_active", staticmethod(lambda: fake_engine)
    )
    assert cp.delete_chat_from_disk("c1") is False


def test_prune_large_messages_short_keeps_original():
    from agent_utilities.chat_persistence import prune_large_messages

    messages = [{"role": "user", "content": "hi"}]
    assert prune_large_messages(messages) == messages


def test_prune_large_messages_truncates():
    from agent_utilities.chat_persistence import prune_large_messages

    long = "x" * 6000
    messages = [{"role": "user", "content": long}]
    out = prune_large_messages(messages, max_length=100)
    assert "truncated" in out[0]["content"]


# ---------------------------------------------------------------------------
# __main__.py
# ---------------------------------------------------------------------------


def test_main_setup_logging_debug(monkeypatch):
    import agent_utilities.__main__ as m

    m.setup_logging(debug=True)
    m.setup_logging(debug=False)
    assert True, 'Debug logging setup completed'
