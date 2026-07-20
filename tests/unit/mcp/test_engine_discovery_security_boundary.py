"""Security contract for KG MCP discovery and the shared child probe."""

from __future__ import annotations

import ast
import asyncio
import contextlib
import inspect
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent_utilities.knowledge_graph.core.engine_ingestion import IngestionMixin
from agent_utilities.knowledge_graph.core.engine_mcp_discovery import (
    MCPDiscoveryError,
    MCPDiscoveryMixin,
)
from agent_utilities.mcp import multiplexer as multiplexer_module
from agent_utilities.mcp.multiplexer import (
    MCPMultiplexer,
    _bounded_tool_catalog,
)


class _DiscoveryHarness(MCPDiscoveryMixin):
    backend = None


class _ToolkitHarness:
    backend = None

    def __init__(self, *, fail_discovery: bool) -> None:
        self.fail_discovery = fail_discovery
        self.persisted: list[dict] = []

    def parse_mcp_config(self, _config_data: dict) -> list[dict]:
        return [
            {
                "name": "synthetic-child",
                "env": {"SYNTHETIC_TOKEN": "env://SYNTHETIC_TOKEN"},
                "tool_flags": ["fabricated"],
                "config_hash": "a" * 64,
            }
        ]

    async def discover_mcp_tools(self, _entry: dict, *, timeout: float) -> list[dict]:
        assert timeout == 30.0
        if self.fail_discovery:
            raise MCPDiscoveryError("mcp_discovery_unavailable")
        return []

    def check_server_freshness(self, _name: str, _identity: str) -> bool:
        return False

    def ingest_mcp_server(self, **kwargs) -> None:
        self.persisted.append(kwargs)


def _remote_config(*, credential_ref: str = "env://SYNTHETIC_TOKEN") -> dict:
    return {
        "mcpServers": {
            "synthetic-child": {
                "url": "https://mcp.example.invalid/mcp",
                "transport": "streamable-http",
                "headers": {"Authorization": credential_ref},
                "tls_profile_ref": "env://SYNTHETIC_TLS_PROFILE",
            }
        }
    }


def test_discovery_module_has_no_raw_mcp_transport_imports() -> None:
    tree = ast.parse(inspect.getsource(inspect.getmodule(MCPDiscoveryMixin)))
    imported_modules = {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert not any(
        name == "mcp" or name.startswith("mcp.") for name in imported_modules
    )
    assert not any(name == "mcp" or name.startswith("mcp.") for name in imported_names)


def test_server_persistence_schema_excludes_runtime_connection_material() -> None:
    schema_path = (
        Path(inspect.getsourcefile(IngestionMixin) or "").parents[2]
        / "models"
        / "schema_definition.py"
    )
    tree = ast.parse(schema_path.read_text(encoding="utf-8"))
    server_columns: dict[str, str] | None = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        keywords = {
            keyword.arg: keyword.value for keyword in node.keywords if keyword.arg
        }
        name_node = keywords.get("name")
        if not isinstance(name_node, ast.Constant) or name_node.value != "Server":
            continue
        columns_node = keywords.get("columns")
        if isinstance(columns_node, ast.Dict):
            server_columns = ast.literal_eval(columns_node)
            break

    assert server_columns is not None
    assert "source_ref" in server_columns
    assert not {"source_config", "command", "args", "env"} & server_columns.keys()
    assert "s.env" not in inspect.getsource(IngestionMixin.ingest_mcp_server)


def test_config_parser_accepts_only_one_current_transport_mode() -> None:
    harness = _DiscoveryHarness()
    with pytest.raises(ValueError, match="declaration is invalid"):
        harness.parse_mcp_config(
            {
                "mcpServers": {
                    "ambiguous": {
                        "command": "synthetic-child",
                        "url": "https://mcp.example.invalid/mcp",
                    }
                }
            }
        )


def test_config_parser_preserves_strict_provider_profile_selection() -> None:
    harness = _DiscoveryHarness()
    entry = harness.parse_mcp_config(
        {
            "mcpServers": {
                "synthetic-child": {
                    "command": "synthetic-child",
                    "provider_profile": "synthetic-provider",
                }
            }
        }
    )[0]

    assert entry["provider_profile"] == "synthetic-provider"
    with pytest.raises(ValueError, match="declaration is invalid"):
        harness.parse_mcp_config(
            {
                "mcpServers": {
                    "synthetic-child": {
                        "command": "synthetic-child",
                        "provider_profile": "INVALID",
                    }
                }
            }
        )
    with pytest.raises(ValueError, match="declaration is invalid"):
        harness.parse_mcp_config(
            {
                "mcpServers": {
                    "retired-alias": {
                        "url": "https://mcp.example.invalid/mcp",
                        "transport": "http",
                    }
                }
            }
        )


def test_freshness_identity_is_keyed_and_ignores_resolved_credentials() -> None:
    harness = _DiscoveryHarness()
    first = harness.parse_mcp_config(_remote_config(credential_ref="secret-one"))[0]
    rotated = harness.parse_mcp_config(_remote_config(credential_ref="secret-two"))[0]
    relocated_config = _remote_config(credential_ref="secret-two")
    relocated_config["mcpServers"]["synthetic-child"]["url"] = (
        "https://other.example.invalid/mcp"
    )
    relocated = harness.parse_mcp_config(relocated_config)[0]

    assert first["config_hash"] == rotated["config_hash"]
    assert first["config_hash"] != relocated["config_hash"]
    assert len(first["config_hash"]) == 64
    assert "secret" not in first["config_hash"]
    assert "example" not in first["config_hash"]


@pytest.mark.asyncio
async def test_discovery_reuses_canonical_probe_and_normalizes_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _DiscoveryHarness()
    entry = harness.parse_mcp_config(_remote_config())[0]
    probe = AsyncMock(
        return_value={
            "tools": [
                {
                    "name": "inspect",
                    "description": "Inspect safely",
                    "inputSchema": {"type": "object"},
                    "annotations": {"readOnlyHint": True},
                }
            ],
            "error": None,
        }
    )
    monkeypatch.setattr(MCPMultiplexer, "probe_declaration", probe)

    tools = await harness.discover_mcp_tools(entry, timeout=7.0)

    probe.assert_awaited_once_with("synthetic-child", entry, timeout=7.0)
    assert tools == [
        {
            "name": "inspect",
            "description": "Inspect safely",
            "input_schema": {"type": "object"},
            "annotations": {"readOnlyHint": True},
        }
    ]


@pytest.mark.asyncio
async def test_discovery_failure_is_closed_and_redacted(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    harness = _DiscoveryHarness()
    entry = harness.parse_mcp_config(_remote_config())[0]
    monkeypatch.setattr(
        MCPMultiplexer,
        "probe_declaration",
        AsyncMock(
            return_value={
                "tools": [],
                "error": "synthetic-secret https://private.example.invalid",
            }
        ),
    )

    with pytest.raises(MCPDiscoveryError, match="mcp_discovery_unavailable"):
        await harness.discover_mcp_tools(entry)

    assert "synthetic-secret" not in caplog.text
    assert "private.example.invalid" not in caplog.text


@pytest.mark.asyncio
async def test_toolkit_ingestion_does_not_synthesize_tools_after_probe_failure() -> (
    None
):
    harness = _ToolkitHarness(fail_discovery=True)
    summary = {
        "mcp_servers": 0,
        "tools_discovered": 0,
        "errors": [],
        "skipped": 0,
    }

    await IngestionMixin._ingest_mcp_from_config(
        harness,
        {"mcpServers": {}},
        "configured-source",
        summary,
    )

    assert harness.persisted == []
    assert summary["tools_discovered"] == 0
    assert summary["errors"] == ["MCP child discovery unavailable"]


@pytest.mark.asyncio
async def test_authoritative_empty_catalog_persists_only_neutral_references() -> None:
    harness = _ToolkitHarness(fail_discovery=False)
    summary = {
        "mcp_servers": 0,
        "tools_discovered": 0,
        "errors": [],
        "skipped": 0,
    }

    await IngestionMixin._ingest_mcp_from_config(
        harness,
        {"mcpServers": {}},
        "configured-source",
        summary,
    )

    assert len(harness.persisted) == 1
    persisted = harness.persisted[0]
    assert persisted["url"] == f"mcp-ref://{'a' * 64}"
    assert persisted["tools"] == []
    assert "source_config" not in persisted["resources"]
    assert "env" not in persisted["resources"]


@pytest.mark.asyncio
async def test_probe_rejects_inline_or_forged_credentials_before_connect() -> None:
    declaration = {
        "url": "https://mcp.example.invalid/mcp",
        "headers": {"Authorization": "synthetic-secret"},
        "_runtime_materialized_secret_keys": ["Authorization"],
        "_runtime_materialization_attestation": "0" * 64,
    }
    with pytest.raises(RuntimeError, match="runtime references"):
        await MCPMultiplexer.probe_declaration(
            "synthetic-child",
            declaration,
            timeout=1.0,
        )


def test_tool_catalog_enforces_count_size_and_depth_bounds() -> None:
    tool = SimpleNamespace(
        name="inspect",
        description="safe",
        inputSchema={"type": "object"},
        annotations=None,
    )
    with pytest.raises(RuntimeError, match="exceeded"):
        _bounded_tool_catalog([tool] * 2_049)

    oversized = SimpleNamespace(
        name="inspect",
        description="x" * (4 * 1024 * 1024),
        inputSchema={},
        annotations=None,
    )
    with pytest.raises(RuntimeError, match="exceeded"):
        _bounded_tool_catalog([oversized])

    schema: dict = {}
    cursor = schema
    for _ in range(40):
        child: dict = {}
        cursor["nested"] = child
        cursor = child
    too_deep = SimpleNamespace(
        name="inspect",
        description="safe",
        inputSchema=schema,
        annotations=None,
    )
    with pytest.raises(RuntimeError, match="exceeded"):
        _bounded_tool_catalog([too_deep])


@pytest.mark.asyncio
async def test_stdio_probe_delegates_only_allowlisted_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: dict = {}

    @contextlib.asynccontextmanager
    async def fake_stdio(parameters, *, errlog):
        observed.update(parameters.env or {})
        assert errlog is not None
        yield "read", "write"

    class FakeSession:
        def __init__(self, *_args, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def initialize(self):
            return None

    monkeypatch.setenv("UNRELATED_PRIVATE_TOKEN", "synthetic-secret")
    monkeypatch.setattr(multiplexer_module, "stdio_client", fake_stdio)
    monkeypatch.setattr(multiplexer_module, "ClientSession", FakeSession)
    multiplexer = MCPMultiplexer(tmp_path / "unused.json")

    async with contextlib.AsyncExitStack() as stack:
        await multiplexer._open_one_session(
            "synthetic-child",
            {
                "command": "synthetic-child",
                "args": [],
                "env": {"EXPLICIT_SETTING": "enabled"},
                "initialization_timeout": 1.0,
            },
            stack,
        )

    assert observed["EXPLICIT_SETTING"] == "enabled"
    assert "UNRELATED_PRIVATE_TOKEN" not in observed


@pytest.mark.asyncio
async def test_stdio_provider_profile_is_preflighted_off_loop_and_projected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: dict[str, str] = {}

    @contextlib.asynccontextmanager
    async def fake_stdio(parameters, *, errlog):
        observed.update(parameters.env or {})
        assert errlog is not None
        yield "read", "write"

    class FakeSession:
        def __init__(self, *_args, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def initialize(self):
            return None

    monkeypatch.setattr(multiplexer_module, "stdio_client", fake_stdio)
    monkeypatch.setattr(multiplexer_module, "ClientSession", FakeSession)
    parent_config_root = tmp_path / "parent-private-config"
    parent_config_root.mkdir()
    (parent_config_root / "config.json").write_text(
        '{"LANGFUSE_SECRET_KEY_REF":"env://UNRELATED_RUNTIME_SECRET"}',
        encoding="utf-8",
    )
    (parent_config_root / "runtime-secrets.json").write_text(
        '{"UNRELATED_RUNTIME_SECRET":"must-not-cross"}',
        encoding="utf-8",
    )
    monkeypatch.setenv("AGENT_UTILITIES_CONFIG_DIR", str(parent_config_root))
    monkeypatch.setenv("UNRELATED_RUNTIME_SECRET", "must-not-cross")
    case_variant_roots = {
        "agent_utilities_config_dir": parent_config_root,
        "xdg_config_home": parent_config_root / "xdg-private",
        "home": parent_config_root / "home-private",
    }
    for key, value in case_variant_roots.items():
        monkeypatch.setenv(key, str(value))
    resolver_threads: list[int] = []

    def prepare(profile_name: str):
        resolver_threads.append(threading.get_ident())
        return SimpleNamespace(
            environment={
                "AGENT_PROVIDER_PROFILE": profile_name,
                "PROVIDER_CONFIGS": '{"synthetic-provider":{}}',
                "SYNTHETIC_RUNTIME_VALUE": "ephemeral",
            },
            close=lambda: None,
        )

    monkeypatch.setattr(
        "agent_utilities.core.provider_runtime.prepare_provider_runtime_child_environment",
        prepare,
    )
    multiplexer = MCPMultiplexer(tmp_path / "unused.json")

    async with contextlib.AsyncExitStack() as stack:
        await multiplexer._open_one_session(
            "synthetic-child",
            {
                "command": "synthetic-child",
                "provider_profile": "synthetic-provider",
                "initialization_timeout": 1.0,
            },
            stack,
        )
        isolated_config_root = Path(observed["AGENT_UTILITIES_CONFIG_DIR"])
        assert isolated_config_root != parent_config_root
        assert list(isolated_config_root.iterdir()) == []
        assert "UNRELATED_RUNTIME_SECRET" not in observed
        assert not set(case_variant_roots).intersection(observed)

    assert observed["AGENT_PROVIDER_PROFILE"] == "synthetic-provider"
    assert observed["SYNTHETIC_RUNTIME_VALUE"] == "ephemeral"
    assert resolver_threads
    assert all(identifier != threading.get_ident() for identifier in resolver_threads)
    assert not isolated_config_root.exists()


@pytest.mark.asyncio
async def test_attested_child_cannot_override_parent_provider_selection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "agent_utilities.core.provider_runtime.prepare_provider_runtime_child_environment",
        lambda profile_name: SimpleNamespace(
            environment={
                "AGENT_PROVIDER_PROFILE": profile_name,
                "PROVIDER_CONFIGS": '{"synthetic-provider":{}}',
                "SYNTHETIC_RUNTIME_VALUE": "parent-controlled",
            },
            close=lambda: None,
        ),
    )
    declaration = multiplexer_module.attest_runtime_child_config(
        {
            "command": "synthetic-child",
            "provider_profile": "synthetic-provider",
            "env": {"SYNTHETIC_RUNTIME_VALUE": "changed-provider"},
        }
    )
    multiplexer = MCPMultiplexer(tmp_path / "unused.json")

    async with contextlib.AsyncExitStack() as stack:
        with pytest.raises(RuntimeError, match="parent-controlled"):
            await multiplexer._open_one_session(
                "synthetic-child",
                declaration,
                stack,
            )


@pytest.mark.asyncio
async def test_timed_out_provider_resolution_is_bounded_and_erased(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    closed = threading.Event()

    def delayed_prepare(profile_name: str):
        time.sleep(0.05)
        return SimpleNamespace(
            environment={"AGENT_PROVIDER_PROFILE": profile_name},
            close=closed.set,
        )

    monkeypatch.setattr(
        "agent_utilities.core.provider_runtime.prepare_provider_runtime_child_environment",
        delayed_prepare,
    )
    multiplexer = MCPMultiplexer(tmp_path / "unused.json")

    async with contextlib.AsyncExitStack() as stack:
        with pytest.raises(RuntimeError, match="profile is unavailable"):
            await multiplexer._open_one_session(
                "synthetic-child",
                {
                    "command": "synthetic-child",
                    "provider_profile": "synthetic-provider",
                    "initialization_timeout": 0.005,
                },
                stack,
            )

    assert await asyncio.to_thread(closed.wait, 1.0)


@pytest.mark.asyncio
async def test_provider_projection_closes_when_child_sandbox_creation_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    closed = False

    def prepare(profile_name: str):
        def close() -> None:
            nonlocal closed
            closed = True

        return SimpleNamespace(
            environment={"AGENT_PROVIDER_PROFILE": profile_name},
            close=close,
        )

    def fail_sandbox(_stack) -> dict[str, str]:
        raise OSError("synthetic sandbox creation failure")

    monkeypatch.setattr(
        "agent_utilities.core.provider_runtime.prepare_provider_runtime_child_environment",
        prepare,
    )
    monkeypatch.setattr(
        multiplexer_module,
        "_provider_child_sandbox_environment",
        fail_sandbox,
    )
    multiplexer = MCPMultiplexer(tmp_path / "unused.json")

    with pytest.raises(OSError, match="sandbox creation failure"):
        async with contextlib.AsyncExitStack() as stack:
            await multiplexer._open_one_session(
                "synthetic-child",
                {
                    "command": "synthetic-child",
                    "provider_profile": "synthetic-provider",
                    "initialization_timeout": 1.0,
                },
                stack,
            )

    assert closed is True


@pytest.mark.asyncio
async def test_provider_resolution_limit_is_process_wide_across_multiplexers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    active = 0
    peak = 0
    counter_lock = threading.Lock()

    def delayed_prepare(profile_name: str):
        nonlocal active, peak
        with counter_lock:
            active += 1
            peak = max(peak, active)
        try:
            time.sleep(0.05)
            return SimpleNamespace(
                environment={
                    "AGENT_PROVIDER_PROFILE": profile_name,
                    "PROVIDER_CONFIGS": '{"synthetic-provider":{}}',
                },
                close=lambda: None,
            )
        finally:
            with counter_lock:
                active -= 1

    @contextlib.asynccontextmanager
    async def fake_stdio(_parameters, *, errlog):
        assert errlog is not None
        yield "read", "write"

    class FakeSession:
        def __init__(self, *_args, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def initialize(self):
            return None

    monkeypatch.setattr(
        "agent_utilities.core.provider_runtime.prepare_provider_runtime_child_environment",
        delayed_prepare,
    )
    monkeypatch.setattr(multiplexer_module, "stdio_client", fake_stdio)
    monkeypatch.setattr(multiplexer_module, "ClientSession", FakeSession)
    multiplexers = [
        MCPMultiplexer(tmp_path / "unused-one.json"),
        MCPMultiplexer(tmp_path / "unused-two.json"),
    ]

    async def open_child(index: int) -> None:
        async with contextlib.AsyncExitStack() as stack:
            await multiplexers[index % 2]._open_one_session(
                f"synthetic-child-{index}",
                {
                    "command": "synthetic-child",
                    "provider_profile": "synthetic-provider",
                    "initialization_timeout": 2.0,
                },
                stack,
            )

    await asyncio.gather(*(open_child(index) for index in range(8)))

    assert 1 <= peak <= 4


@pytest.mark.asyncio
async def test_timed_out_provider_queue_stays_within_process_capacity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    release_workers = threading.Event()
    all_closed = threading.Event()
    counter_lock = threading.Lock()
    started = 0
    closed = 0
    submitted = 0

    def delayed_prepare(profile_name: str):
        nonlocal started
        with counter_lock:
            started += 1
        release_workers.wait(10.0)

        def close() -> None:
            nonlocal closed
            with counter_lock:
                closed += 1
                if closed == 8:
                    all_closed.set()

        return SimpleNamespace(
            environment={"AGENT_PROVIDER_PROFILE": profile_name},
            close=close,
        )

    original_submit = multiplexer_module._PROVIDER_RESOLUTION_EXECUTOR.submit

    def counted_submit(*args, **kwargs):
        nonlocal submitted
        with counter_lock:
            submitted += 1
        return original_submit(*args, **kwargs)

    monkeypatch.setattr(
        "agent_utilities.core.provider_runtime.prepare_provider_runtime_child_environment",
        delayed_prepare,
    )
    monkeypatch.setattr(
        multiplexer_module._PROVIDER_RESOLUTION_EXECUTOR,
        "submit",
        counted_submit,
    )
    multiplexers = [
        MCPMultiplexer(tmp_path / "unused-one.json"),
        MCPMultiplexer(tmp_path / "unused-two.json"),
    ]

    async def time_out_child(index: int) -> None:
        async with contextlib.AsyncExitStack() as stack:
            with pytest.raises(RuntimeError, match="profile is unavailable"):
                await multiplexers[index % 2]._open_one_session(
                    f"synthetic-child-{index}",
                    {
                        "command": "synthetic-child",
                        "provider_profile": "synthetic-provider",
                        "initialization_timeout": 0.05,
                    },
                    stack,
                )

    try:
        await asyncio.gather(*(time_out_child(index) for index in range(8)))
        with counter_lock:
            assert started == 4
            assert submitted == 8

        # All eight process-wide slots remain owned by the four running and
        # four queued resolutions. Further timed-out callers must fail before
        # submitting more work items to ThreadPoolExecutor's unbounded queue.
        await asyncio.gather(*(time_out_child(index) for index in range(8, 24)))
        with counter_lock:
            assert submitted == 8
    finally:
        release_workers.set()

    assert await asyncio.to_thread(all_closed.wait, 2.0)
