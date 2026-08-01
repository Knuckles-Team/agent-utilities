from __future__ import annotations

"""Integration tests for the per-turn model override middleware.

CONCEPT:AU-ECO.messaging.native-backend-abstraction Protocol Layer

These tests cover the full plumbing that lets the terminal UI (and any
other client) honour an ``x-agent-model-id`` header for the duration of
a single turn:

1. The FastAPI middleware (``_model_override_middleware``) populates
   ``request.state.requested_model_id`` and the
   ``REQUESTED_MODEL_ID_CTX`` ContextVar from the header.
2. Protocol wrappers (AG-UI, ``/stream``, and ACP graph execution)
   thread the value into ``GraphDeps.requested_model_id``.
3. ``pick_specialist_model`` prefers the user-requested model over the
   tier/tag routing heuristic, so the user's choice beats
   ``ModelRegistry.pick_for_task``.

Every test is mock-based: no real LLM is spun up and no HTTP traffic
leaves the process.
"""


from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from agent_utilities.core.config import config
from agent_utilities.graph.executor import pick_specialist_model
from agent_utilities.graph.state import REQUESTED_MODEL_ID_CTX, GraphDeps
from agent_utilities.models import ModelDefinition, ModelRegistry
from agent_utilities.orchestration.engine import AgentOrchestrationEngine
from agent_utilities.server import build_agent_app


@pytest.fixture(autouse=True)
def _no_ambient_jwt_auth(monkeypatch):
    """Keep this file's TestClient hermetic to the host's real identity provider
    AND to the server-minted-identity requirement ``register_graph_routes``
    wires in, neither of which this file exists to exercise.

    Two independent things stack up against an unauthenticated request here:

    1. ``config.auth_jwt_jwks_uri``/``auth_jwt_issuer``/``auth_jwt_audience``
       are process-global settings sourced from this machine's XDG runtime
       config (``core/config.py::_commit_xdg_environment_projection``) -- on
       a host that has ever run agent-utilities as a real deployed service
       behind Keycloak (see ``~/.config/agent-utilities/config.json``), those
       are non-empty here too, which would otherwise route a *presented*
       token through real JWKS verification (``security/auth.py::
       authenticate_header_values``).
    2. ``build_agent_app`` unconditionally calls
       ``gateway/graph_api.py::register_graph_routes``, which mounts
       ``ActorIdentityMiddleware`` via ``app.add_middleware(...)`` --
       Starlette middleware is never scoped to the ``/api`` prefix that
       function registers routes under, so it wraps the WHOLE app, including
       routes this file (and the real ``/stream``/``/ag-ui`` core routes)
       adds afterward. That middleware 401s any request without a valid
       Bearer identity regardless of ``auth_jwt_jwks_uri`` (see its
       docstring: "No valid identity -> 401 (health paths exempt)"), so
       clearing the JWKS settings alone is not sufficient here. None of
       these tests set an Authorization header -- they test the model-
       override middleware in isolation, per the module docstring -- so the
       identity middleware is replaced with a passthrough for this file only.
    """
    monkeypatch.setattr(config, "auth_jwt_jwks_uri", None)
    monkeypatch.setattr(config, "auth_jwt_issuer", None)
    monkeypatch.setattr(config, "auth_jwt_audience", None)

    class _PassthroughIdentityMiddleware:
        def __init__(self, app: Any) -> None:
            self.app = app

        async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
            await self.app(scope, receive, send)

    monkeypatch.setattr(
        "agent_utilities.security.request_identity.ActorIdentityMiddleware",
        _PassthroughIdentityMiddleware,
    )


@pytest.fixture
def mock_agent():
    """Minimal Agent stand-in used by build_agent_app via create_agent patch."""
    agent = MagicMock()
    agent.toolsets = []
    agent.to_a2a.return_value = MagicMock()
    return agent


@pytest.fixture
def registry() -> ModelRegistry:
    """Two-model registry so `pick_for_task` and the override can diverge."""
    return ModelRegistry(
        models=[
            ModelDefinition(
                id="cheap-local",
                name="Cheap Local",
                provider="openai",
                model_id="llama-3.2-3b-instruct",
                base_url="http://localhost:1234/v1",
                tier="light",
                is_default=True,
            ),
            ModelDefinition(
                id="premium-cloud",
                name="Premium Cloud",
                provider="openai",
                model_id="gpt-4o",
                api_key_env="OPENAI_API_KEY",
                tier="heavy",
                tags=["reasoning"],
            ),
        ]
    )


def _build_client(agent, **kwargs) -> TestClient:
    """Construct a TestClient without mounting the web UI or ACP.

    ``host`` is pinned to the loopback address so this test is hermetic to
    the ambient environment: a host whose XDG runtime config has ever
    projected ``HOST=0.0.0.0`` (e.g. after a real service deployment on the
    same machine, see ``core/config.py::_commit_xdg_environment_projection``)
    would otherwise make ``build_agent_app``'s non-loopback listener guard
    (``_http_boundary_settings``) reject construction with "A non-loopback
    REST listener requires direct TLS or a trusted TLS-terminating ingress"
    -- a real security guard doing exactly what it should for a genuine
    deployment, but not what this in-process ``TestClient`` (which never
    actually listens on the network) needs to exercise.
    """
    kwargs.setdefault("host", "127.0.0.1")
    with patch("agent_utilities.server.app.create_agent", return_value=(agent, [])):
        app = build_agent_app(
            enable_web_ui=False,
            enable_acp=False,
            enable_otel=False,
            graph_bundle=("graph", {"valid_domains": []}),
            **kwargs,
        )
    return TestClient(app)


def _probe_app(
    agent, registry: ModelRegistry | None = None
) -> tuple[FastAPI, list[dict[str, Any]]]:
    """Build an app and add a ``/__probe`` route that echoes request state.

    Returns ``(app, captures)`` where ``captures`` is a list appended to by
    the probe — one dict per request observed. This lets tests verify
    that the middleware populated ``request.state.requested_model_id``
    without needing to spin up the real AG-UI stack.
    """
    captures: list[dict[str, Any]] = []

    # host pinned to loopback for the same reason as _build_client above --
    # ambient XDG-projected HOST=0.0.0.0 would otherwise trip the real
    # non-loopback listener guard for this in-process-only TestClient.
    with patch("agent_utilities.server.app.create_agent", return_value=(agent, [])):
        app = build_agent_app(
            enable_web_ui=False,
            enable_acp=False,
            enable_otel=False,
            graph_bundle=("graph", {"valid_domains": []}),
            model_registry=registry,
            host="127.0.0.1",
        )

    @app.post("/__probe")
    async def _probe(request: Request):
        captures.append(
            {
                "requested_model_id": getattr(
                    request.state, "requested_model_id", None
                ),
                "ctx_value": REQUESTED_MODEL_ID_CTX.get(),
            }
        )
        return JSONResponse({"ok": True})

    return app, captures


def test_middleware_sets_state_from_header(mock_agent, registry):
    """An explicit ``x-agent-model-id`` header lands on ``request.state``."""
    app, captures = _probe_app(mock_agent, registry)
    client = TestClient(app)

    resp = client.post("/__probe", headers={"x-agent-model-id": "premium-cloud"})
    assert resp.status_code == 200
    assert len(captures) == 1
    assert captures[0]["requested_model_id"] == "premium-cloud"
    assert captures[0]["ctx_value"] == "premium-cloud"


def test_middleware_sets_state_none_when_absent(mock_agent, registry):
    """Missing header yields ``None`` on state and ContextVar alike."""
    app, captures = _probe_app(mock_agent, registry)
    client = TestClient(app)

    resp = client.post("/__probe")
    assert resp.status_code == 200
    assert len(captures) == 1
    assert captures[0]["requested_model_id"] is None
    assert captures[0]["ctx_value"] is None


def test_middleware_empty_header_treated_as_absent(mock_agent, registry):
    """An empty string header value is coerced to ``None``."""
    app, captures = _probe_app(mock_agent, registry)
    client = TestClient(app)

    resp = client.post("/__probe", headers={"x-agent-model-id": ""})
    assert resp.status_code == 200
    assert captures[0]["requested_model_id"] is None
    assert captures[0]["ctx_value"] is None


def test_middleware_contextvar_is_reset_after_request(mock_agent, registry):
    """Each request starts with a clean ContextVar even after a prior turn."""
    app, captures = _probe_app(mock_agent, registry)
    client = TestClient(app)

    client.post("/__probe", headers={"x-agent-model-id": "premium-cloud"})
    client.post("/__probe")  # no header → should be None, not leaked
    assert len(captures) == 2
    assert captures[0]["requested_model_id"] == "premium-cloud"
    assert captures[1]["requested_model_id"] is None


def test_stream_route_honors_header(mock_agent, registry):
    """``/stream`` forwards the header value into ``stream_graph``."""
    client = _build_client(mock_agent, model_registry=registry)

    captured: dict[str, Any] = {}

    async def fake_stream(*args, **kwargs):
        captured.update(kwargs)
        yield "data: ok\n\n"

    with patch(
        "agent_utilities.orchestration.engine.AgentOrchestrationEngine.stream_graph",
        side_effect=fake_stream,
    ):
        resp = client.post(
            "/stream",
            json={"query": "hi", "mode": "ask", "topology": "basic"},
            headers={"x-agent-model-id": "premium-cloud"},
        )
        assert resp.status_code == 200
        # Fully consume the stream so the generator runs.
        b"".join(resp.iter_bytes())

    assert captured.get("requested_model_id") == "premium-cloud"


def test_stream_route_no_header_passes_none(mock_agent, registry):
    """``/stream`` passes ``requested_model_id=None`` when no header."""
    client = _build_client(mock_agent, model_registry=registry)

    captured: dict[str, Any] = {}

    async def fake_stream(*args, **kwargs):
        captured.update(kwargs)
        yield "data: ok\n\n"

    with patch(
        "agent_utilities.orchestration.engine.AgentOrchestrationEngine.stream_graph",
        side_effect=fake_stream,
    ):
        resp = client.post(
            "/stream",
            json={"query": "hi"},
        )
        assert resp.status_code == 200
        b"".join(resp.iter_bytes())

    assert captured.get("requested_model_id") is None


def test_ag_ui_route_honors_header(mock_agent, registry):
    """AG-UI invokes ``agent.override(model=...)`` when the header is valid.

    This locks in the top-level contract: the header triggers a real
    override on the pydantic-ai agent for the duration of the AG-UI
    dispatch.
    """
    from fastapi.responses import StreamingResponse

    async def fake_iter():
        yield b'0:"hi"\n'

    # Instrument ``mock_agent.override`` so we can observe it being used.
    override_calls: list[dict[str, Any]] = []

    class _DummyCtx:
        def __enter__(self):
            return None

        def __exit__(self, *a):
            return False

    def override_side_effect(**kwargs):
        override_calls.append(kwargs)
        return _DummyCtx()

    mock_agent.override = MagicMock(side_effect=override_side_effect)

    client = _build_client(mock_agent, model_registry=registry)

    with patch("pydantic_ai.ui.ag_ui.AGUIAdapter") as mock_adapter_cls:
        adapter = mock_adapter_cls.return_value
        adapter.dispatch_request = MagicMock()

        async def _dispatch(*args, **kwargs):
            return StreamingResponse(fake_iter(), media_type="text/plain")

        adapter.dispatch_request.side_effect = _dispatch

        resp = client.post(
            "/ag-ui",
            json={"query": "hello"},
            headers={"x-agent-model-id": "premium-cloud"},
        )
        assert resp.status_code == 200
        b"".join(resp.iter_bytes())

    assert override_calls, "agent.override(...) was never called"
    assert "model" in override_calls[0]
    assert override_calls[0]["model"] is not None


def test_ag_ui_route_invalid_id_falls_back_to_default(mock_agent, registry):
    """Unknown model id → no override applied, adapter still dispatches."""
    from fastapi.responses import StreamingResponse

    async def fake_iter():
        yield b'0:"hi"\n'

    mock_agent.override = MagicMock()

    client = _build_client(mock_agent, model_registry=registry)

    with patch("pydantic_ai.ui.ag_ui.AGUIAdapter") as mock_adapter_cls:
        adapter = mock_adapter_cls.return_value
        adapter.dispatch_request = MagicMock()

        async def _dispatch(*args, **kwargs):
            return StreamingResponse(fake_iter(), media_type="text/plain")

        adapter.dispatch_request.side_effect = _dispatch

        resp = client.post(
            "/ag-ui",
            json={"query": "hello"},
            headers={"x-agent-model-id": "does-not-exist"},
        )
        assert resp.status_code == 200
        b"".join(resp.iter_bytes())

    # override must NOT be called for an unknown id — we fall back silently.
    mock_agent.override.assert_not_called()


def test_executor_prefers_requested_model_over_tier(registry):
    """``pick_specialist_model`` honours ``requested_model_id`` first.

    Given a ``researcher`` node (which the heuristic tags as ``light`` and
    would therefore pick ``cheap-local``), setting ``requested_model_id``
    to the ``heavy``-tier ``premium-cloud`` entry must win.
    """
    deps = GraphDeps(
        tag_prompts={},
        tag_env_vars={},
        mcp_toolsets=[],
        agent_model="DEFAULT",
        model_registry=registry,
        requested_model_id="premium-cloud",
    )

    captured: dict[str, Any] = {}

    def fake_create_model(**kwargs):
        captured.update(kwargs)
        return f"MODEL<{kwargs.get('model_id')}>"

    with patch(
        "agent_utilities.core.model_factory.create_model", side_effect=fake_create_model
    ):
        model = pick_specialist_model(deps, "researcher")

    assert model == "MODEL<gpt-4o>"
    assert captured["provider"] == "openai"
    assert captured["model_id"] == "gpt-4o"


def test_executor_falls_back_to_tier_when_requested_id_unknown(registry):
    """Unknown ``requested_model_id`` → heuristic tier routing still runs."""
    deps = GraphDeps(
        tag_prompts={},
        tag_env_vars={},
        mcp_toolsets=[],
        agent_model="DEFAULT",
        model_registry=registry,
        requested_model_id="no-such-id",
    )

    captured: dict[str, Any] = {}

    def fake_create_model(**kwargs):
        captured.update(kwargs)
        return f"MODEL<{kwargs.get('model_id')}>"

    # researcher is mapped to the 'light' tier heuristic, which matches
    # 'cheap-local'. An unknown requested id must NOT short-circuit that.
    # Mock get_discovery_registry so KG tier overrides don't interfere.
    empty_registry = MagicMock()
    empty_registry.agents = []
    with (
        patch(
            "agent_utilities.core.model_factory.create_model",
            side_effect=fake_create_model,
        ),
        patch(
            "agent_utilities.graph.executor.get_discovery_registry",
            return_value=empty_registry,
        ),
    ):
        model = pick_specialist_model(deps, "researcher")

    assert model == "MODEL<llama-3.2-3b-instruct>"


def test_executor_override_ignored_when_registry_empty():
    """With no registry, the override is silently ignored.

    A lone ``requested_model_id`` without a registry must not crash; the
    executor returns the legacy ``agent_model`` verbatim.

    ``GraphDeps.__post_init__`` eagerly resolves any string ``agent_model``
    through ``create_model()`` (CONCEPT boundary: every downstream
    ``Agent(model=ctx.deps.*_model)`` call must be governed, see its
    docstring), so by the time this assertion runs ``deps.agent_model`` is
    already the resolved model object, not the raw ``"DEFAULT"`` string
    passed to the constructor -- comparing against ``deps.agent_model``
    (identity, not the pre-init string) is what "verbatim" actually means
    post-resolution.
    """
    deps = GraphDeps(
        tag_prompts={},
        tag_env_vars={},
        mcp_toolsets=[],
        agent_model="DEFAULT",
        model_registry=None,
        requested_model_id="premium-cloud",
    )
    assert pick_specialist_model(deps, "researcher") is deps.agent_model


def test_invalid_model_id_falls_back_to_default(registry):
    """``pick_specialist_model`` never raises on an unknown requested id.

    This is the safety net for the '5. If the model id is invalid or
    unset, proceeds with the default' requirement.
    """
    deps = GraphDeps(
        tag_prompts={},
        tag_env_vars={},
        mcp_toolsets=[],
        agent_model="DEFAULT",
        model_registry=registry,
        requested_model_id="",  # empty string — treated like no override
    )

    def fake_create_model(**kwargs):
        return f"MODEL<{kwargs.get('model_id')}>"

    # Mock get_discovery_registry so KG tier overrides don't interfere.
    empty_registry = MagicMock()
    empty_registry.agents = []
    with (
        patch(
            "agent_utilities.core.model_factory.create_model",
            side_effect=fake_create_model,
        ),
        patch(
            "agent_utilities.graph.executor.get_discovery_registry",
            return_value=empty_registry,
        ),
    ):
        # researcher → light tier → cheap-local
        assert pick_specialist_model(deps, "researcher") == (
            "MODEL<llama-3.2-3b-instruct>"
        )


@pytest.mark.asyncio
async def test_stream_graph_reads_contextvar_fallback(registry):
    """When no explicit kwarg is passed, stream_graph reads the CV.

    This is the channel used by ACP (which sets the CV in the middleware
    and then the graph runs several layers deep behind an ACP adapter).
    """

    captured: dict[str, Any] = {}

    class _Graph:
        async def run(self, *, state, deps):
            captured["requested_model_id"] = deps.requested_model_id
            return None

    token = REQUESTED_MODEL_ID_CTX.set("premium-cloud")
    try:
        gen = AgentOrchestrationEngine().stream_graph(
            _Graph(),
            {"model_registry": registry},
            "hi",
            run_id="test-run",
        )
        # Drain the generator.
        async for _ in gen:
            pass
    finally:
        REQUESTED_MODEL_ID_CTX.reset(token)

    assert captured["requested_model_id"] == "premium-cloud"


@pytest.mark.asyncio
async def test_run_graph_kwarg_overrides_contextvar(registry):
    """An explicit kwarg wins over the ContextVar fallback."""

    captured: dict[str, Any] = {}

    class _Graph:
        async def run(self, *, state, deps):
            captured["requested_model_id"] = deps.requested_model_id
            return None

    token = REQUESTED_MODEL_ID_CTX.set("cheap-local")
    try:
        await AgentOrchestrationEngine().execute_graph(
            _Graph(),
            {"model_registry": registry},
            "hi",
            run_id="test-run",
            requested_model_id="premium-cloud",
        )
    finally:
        REQUESTED_MODEL_ID_CTX.reset(token)

    assert captured["requested_model_id"] == "premium-cloud"
