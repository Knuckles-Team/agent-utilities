#!/usr/bin/python
from __future__ import annotations

"""First-party Pydantic AI Harness adapter for Agent Client Protocol (ACP).

CONCEPT:AU-ECO.messaging.native-backend-abstraction

ACP is an editor-facing stdio JSON-RPC protocol.  It is deliberately served as
a subprocess entrypoint, not mounted into the Graph-OS ASGI application.  The
Harness adapter owns wire translation, streaming, approval prompts, rich
filesystem/shell presentation, cancellation, and session lifecycle.  Graph-OS
continues to own graph planning, execution modes, checkpoints, and provenance.
"""

import argparse
import asyncio
import hashlib
import logging
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter
from pydantic_ai import RunContext

from agent_utilities.core.config import (
    DEFAULT_ACP_SESSION_ROOT,
    DEFAULT_CUSTOM_SKILLS_DIRECTORY,
    DEFAULT_LLM_API_KEY,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL_ID,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_MCP_CONFIG,
    DEFAULT_MCP_URL,
)
from agent_utilities.core.contextual_model import create_context_agent
from agent_utilities.security.cli_secrets import RuntimeSecretReferenceAction

logger = logging.getLogger(__name__)

try:
    from pydantic_ai_harness.experimental.acp import (
        AcpSession,
        AcpSessionConfig,
        PydanticAIACPAgent,
        StoredSession,
    )

    _ACP_INSTALLED = True
except ModuleNotFoundError as exc:
    if exc.name not in {
        "acp",
        "pydantic_ai_harness",
        "pydantic_ai_harness.experimental",
        "pydantic_ai_harness.experimental.acp",
    }:
        raise
    _ACP_INSTALLED = False
    AcpSession = Any  # type: ignore[assignment,misc]
    AcpSessionConfig = Any  # type: ignore[assignment,misc]
    PydanticAIACPAgent = Any  # type: ignore[assignment,misc]
    StoredSession = Any  # type: ignore[assignment,misc]


class ACPUnavailableError(RuntimeError):
    """Raised when the optional first-party Harness ACP extra is absent."""


def _require_acp() -> None:
    if not _ACP_INSTALLED:
        raise ACPUnavailableError(
            'ACP requires the optional extra: uv add "pydantic-ai-harness[acp]"'
        )


class FileAcpSessionStore:
    """Durable, validated, atomic filesystem storage for Harness ACP sessions.

    Session IDs are hashed before becoming filenames, eliminating path traversal
    and avoiding disclosure through directory listings.  Files contain model
    history and the client-visible transcript, so the directory is private to
    the process owner.  This is conversation persistence, not graph checkpoint
    persistence; Graph-OS remains the checkpoint authority.
    """

    def __init__(self, root: Path) -> None:
        _require_acp()
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            self.root.chmod(0o700)
        except OSError:
            logger.warning("Could not restrict ACP session directory permissions")
        self._adapter = TypeAdapter(StoredSession)

    def _path(self, session_id: str) -> Path:
        digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
        return self.root / f"{digest}.json"

    def _save_sync(self, session_id: str, session: Any) -> None:
        payload = self._adapter.dump_json(session)
        path = self._path(session_id)
        fd, temporary = tempfile.mkstemp(
            dir=self.root,
            prefix=f".{path.stem}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, path)
        except BaseException:
            try:
                os.close(fd)
            except OSError:
                pass
            temporary_path.unlink(missing_ok=True)
            raise

    def _load_sync(self, session_id: str) -> Any | None:
        path = self._path(session_id)
        if not path.is_file():
            return None
        return self._adapter.validate_json(path.read_bytes())

    async def save(self, session_id: str, session: Any) -> None:
        """Atomically persist one committed ACP session."""
        await asyncio.to_thread(self._save_sync, session_id, session)

    async def load(self, session_id: str) -> Any | None:
        """Load and validate a session, or return ``None`` when unknown."""
        return await asyncio.to_thread(self._load_sync, session_id)


@dataclass(frozen=True, slots=True)
class HarnessACPConfig:
    """Agent Utilities configuration passed to the Harness ACP adapter."""

    session_store: FileAcpSessionStore
    models: tuple[str, ...] | None = None
    name: str | None = None
    version: str = "0.1.0"
    usage_limits: Any | None = None


@dataclass(frozen=True, slots=True)
class GraphACPDeps:
    """Per-session dependencies for the shared graph-wrapper agent."""

    session_id: str
    cwd: str
    graph: Any
    graph_config: dict[str, Any]
    mcp_toolsets: tuple[Any, ...]
    concurrency_manager: Any | None = None


def build_acp_config(
    session_root: Path | None = None,
    *,
    models: list[str] | tuple[str, ...] | None = None,
    name: str | None = None,
    version: str = "0.1.0",
    usage_limits: Any | None = None,
) -> HarnessACPConfig:
    """Build the stdio adapter configuration.

    Harness streams thinking and performs deferred-tool approval natively; no
    separate thinking or approval bridges are needed.  ACP modes and native
    plan persistence are intentionally absent because Graph-OS owns those
    execution semantics.
    """
    _require_acp()
    root = session_root or Path(DEFAULT_ACP_SESSION_ROOT)
    return HarnessACPConfig(
        session_store=FileAcpSessionStore(root),
        models=tuple(models) if models else None,
        name=name,
        version=version,
        usage_limits=usage_limits,
    )


def create_acp_agent(agent: Any, config: HarnessACPConfig) -> Any:
    """Adapt a flat Pydantic AI agent to the editor-facing ACP protocol."""
    _require_acp()
    return PydanticAIACPAgent(
        agent,
        name=config.name,
        version=config.version,
        session_store=config.session_store,
        models=config.models,
        usage_limits=config.usage_limits,
    )


def create_graph_acp_agent(
    agent: Any,
    config: HarnessACPConfig,
    graph_bundle: tuple[Any, Any] | None = None,
    mcp_toolsets: list[Any] | None = None,
    concurrency_manager: Any = None,
) -> Any:
    """Adapt an agent to ACP, routing turns through Pydantic Graph when supplied.

    Harness intentionally shares one immutable agent across sessions.  Its
    ``AcpSessionConfig`` hook injects session-scoped dependencies, replacing the
    legacy adapter's per-session agent factory without mutable globals.
    """
    if not graph_bundle:
        return create_acp_agent(agent, config)

    _require_acp()
    graph, graph_config = graph_bundle

    async def execute_graph(
        ctx: RunContext[GraphACPDeps],
        query: str,
        mode: str = "ask",
    ) -> str:
        """Execute the governed Graph-OS Pydantic Graph for this ACP session."""
        from agent_utilities.graph.protocol_agnostic_execution import (
            execute_graph as execute_graph_authority,
        )

        deps = ctx.deps
        if deps.concurrency_manager:
            await deps.concurrency_manager.acquire(
                deps.session_id,
                strategy="interrupt",
            )
        try:
            result = await execute_graph_authority(
                graph=deps.graph,
                config=deps.graph_config,
                query=query,
                mode=mode,
                mcp_toolsets=list(deps.mcp_toolsets),
                requested_model_id=getattr(ctx.model, "model_name", None),
            )
            results = result.get("results", result)
            if isinstance(results, dict):
                return str(results.get("output", results))
            return str(results)
        finally:
            if deps.concurrency_manager:
                await deps.concurrency_manager.release(deps.session_id)

    wrapper = create_context_agent(
        model=agent.model,
        deps_type=GraphACPDeps,
        system_prompt=(
            "You are the ACP boundary for a Graph-OS orchestrator. Call "
            "execute_graph exactly once for every user request. Pass the request "
            "verbatim and do not answer it yourself."
        ),
        tools=[execute_graph],
    )

    def session_config(session: AcpSession) -> AcpSessionConfig[GraphACPDeps]:
        if session.mcp_servers:
            import acp

            raise acp.RequestError.invalid_params(
                {
                    "reason": (
                        "client-offered MCP servers are not silently trusted; "
                        "configure MCP servers in the governed Graph-OS fleet"
                    ),
                    "mcp_server_count": len(session.mcp_servers),
                }
            )
        deps = GraphACPDeps(
            session_id=session.session_id,
            cwd=session.cwd,
            graph=graph,
            graph_config=dict(graph_config),
            mcp_toolsets=tuple(mcp_toolsets or graph_config.get("mcp_toolsets", ())),
            concurrency_manager=concurrency_manager,
        )
        return AcpSessionConfig(deps=deps)

    return PydanticAIACPAgent(
        wrapper,
        name=config.name,
        version=config.version,
        session_config=session_config,
        session_store=config.session_store,
        models=config.models,
        usage_limits=config.usage_limits,
    )


async def run_acp_agent(adapter: Any) -> None:
    """Serve a prepared Harness adapter over stdio JSON-RPC."""
    _require_acp()
    import acp

    await acp.run_agent(adapter, use_unstable_protocol=True)


def run_acp_agent_sync(adapter: Any) -> None:
    """Synchronous editor-process entrypoint for a prepared ACP adapter."""
    asyncio.run(run_acp_agent(adapter))


def is_acp_available() -> bool:
    """Return whether the first-party Harness ACP optional extra is importable."""
    return _ACP_INSTALLED


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Expose an Agent Utilities Pydantic AI agent to an ACP editor"
    )
    parser.add_argument("--provider", default=DEFAULT_LLM_PROVIDER)
    parser.add_argument("--model-id", default=DEFAULT_LLM_MODEL_ID)
    parser.add_argument("--base-url", default=DEFAULT_LLM_BASE_URL)
    parser.add_argument(
        "--api-key-ref",
        dest="api_key",
        action=RuntimeSecretReferenceAction,
        default=DEFAULT_LLM_API_KEY,
        help="Runtime secret reference for the model API key",
    )
    parser.add_argument("--mcp-url", default=DEFAULT_MCP_URL)
    parser.add_argument("--mcp-config", default=DEFAULT_MCP_CONFIG)
    parser.add_argument(
        "--custom-skills-directory",
        default=DEFAULT_CUSTOM_SKILLS_DIRECTORY,
    )
    parser.add_argument("--workspace")
    parser.add_argument(
        "--session-root",
        type=Path,
        default=Path(DEFAULT_ACP_SESSION_ROOT),
    )
    parser.add_argument(
        "--model",
        dest="models",
        action="append",
        help="Advertise a selectable Pydantic AI model id; repeat as needed",
    )
    parser.add_argument(
        "--graph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Route editor turns through the workspace Pydantic Graph",
    )
    return parser


def main() -> None:
    """Launch the dedicated ACP stdio subprocess used by compatible editors."""
    args = _parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _require_acp()

    if args.workspace:
        from agent_utilities.core import workspace as workspace_module

        workspace_module.WORKSPACE_DIR = args.workspace

    from agent_utilities.agent import create_agent

    graph_bundle = None
    if args.graph:
        from agent_utilities.graph.builder import initialize_graph_from_workspace

        graph_bundle = initialize_graph_from_workspace(
            mcp_config=args.mcp_config,
            router_model=args.model_id,
            agent_model=args.model_id,
            api_key=args.api_key,
            base_url=args.base_url,
            workspace=args.workspace,
        )

    agent, toolsets = create_agent(
        provider=args.provider,
        model_id=args.model_id,
        base_url=args.base_url,
        api_key=args.api_key,
        mcp_url=args.mcp_url,
        mcp_config=args.mcp_config,
        custom_skills_directory=args.custom_skills_directory,
        graph_bundle=graph_bundle,
        isolate_mcp=bool(graph_bundle),
    )
    config = build_acp_config(
        args.session_root,
        models=args.models,
        name="agent-utilities",
    )
    adapter = create_graph_acp_agent(
        agent,
        config,
        graph_bundle=graph_bundle,
        mcp_toolsets=toolsets,
    )
    run_acp_agent_sync(adapter)


if __name__ == "__main__":
    main()
