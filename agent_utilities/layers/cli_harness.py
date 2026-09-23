"""Shared drive loop for headless CLI harnesses (Claude Code, Codex, Grok).

A CLI adapter declares its binary, how a run is materialized into the leased
workspace (per-run MCP configuration and skills directory, RF-ADR-010 §6.5),
its argv/environment, and a table from the harness's JSONL record types to
parse handlers. :class:`CliHarness` owns the rest: binary resolution, the
launch, line decoding, the terminal-record check, process-exit typing and
cancellation by terminating the child.
"""

from __future__ import annotations

import abc
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict

from agent_utilities.layers.cli_process import (
    LaunchedProcess,
    ProcessLauncher,
    SubprocessLauncher,
    child_environment,
    confine,
)
from agent_utilities.layers.contracts import (
    UNAVAILABLE_USAGE,
    AccountMode,
    EnvironmentMode,
    HarnessDescriptor,
    HarnessError,
    HarnessNotConfigured,
    HarnessRunFailed,
    HarnessToolInventoryMismatch,
    McpEndpoint,
    ReconciliationSupport,
    RunSpec,
    UsageRecord,
)
from agent_utilities.layers.credentials import (
    CredentialResolver,
    SecretsCredentialResolver,
    api_key_for,
)
from agent_utilities.layers.session import DriveOutcome, HarnessRuntime, RunContext

#: Directory inside the leased workspace that holds per-run harness config.
RUN_CONFIG_DIR = ".au-run"


#: Descriptor fields every headless CLI adapter shares: both account modes,
#: a caller-managed host with a per-run workspace, provider session ids.
class _CliDescriptorDefaults(TypedDict):
    account_modes: frozenset[AccountMode]
    environment_modes: frozenset[EnvironmentMode]
    reconciliation: ReconciliationSupport


CLI_DESCRIPTOR_DEFAULTS = _CliDescriptorDefaults(
    account_modes=frozenset({"api_key", "subscription"}),
    environment_modes=frozenset({"caller-managed-host"}),
    reconciliation="provider_session",
)


@dataclass(slots=True)
class StreamState:
    """What the parse handlers learned from one harness stream."""

    output: str = ""
    usage: UsageRecord = UNAVAILABLE_USAGE
    terminal: bool = False
    failed: bool = False
    error: str = ""
    tool_calls: int = 0
    inventory_verified: bool = False
    text_chunks: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Invocation:
    argv: tuple[str, ...]
    env: Mapping[str, str]
    stdin_text: str


Handler = Callable[[RunContext, dict, StreamState], None]


def as_mapping(value: object) -> dict:
    """``value`` when it is a JSON object, else an empty mapping."""
    return value if isinstance(value, dict) else {}


def token_env_name(index: int) -> str:
    """Child variable carrying endpoint ``index``'s bearer token."""
    return f"AU_MCP_TOKEN_{index}"


class CliHarness(HarnessRuntime, abc.ABC):
    """Base for headless CLI adapters; subclasses supply the harness dialect."""

    binary: str
    descriptor: HarnessDescriptor
    record_handlers: Mapping[str, Handler]

    def __init__(
        self,
        *,
        launcher: ProcessLauncher | None = None,
        credentials: CredentialResolver | None = None,
    ) -> None:
        super().__init__()
        self._launcher = launcher or SubprocessLauncher()
        self._credentials = credentials or SecretsCredentialResolver()
        self._processes: dict[str, LaunchedProcess] = {}

    # -- dialect hooks ------------------------------------------------------

    @abc.abstractmethod
    def materialize(self, run: RunContext, config_dir: Path) -> None:
        """Write the per-run MCP config / skills inside the workspace."""

    @abc.abstractmethod
    def invocation(
        self, run: RunContext, binary_path: str, config_dir: Path
    ) -> Invocation:
        """argv, extra child environment and stdin for this run."""

    def describe(self) -> HarnessDescriptor:
        return self.descriptor

    def record_type(self, record: dict) -> str:
        """The dispatch key of one decoded JSONL record (``type`` by default)."""
        return str(record.get("type") or "")

    # -- HarnessRuntime -----------------------------------------------------

    def preflight(self, spec: RunSpec) -> None:
        self._launcher.resolve(self.binary)
        if spec.account_mode == "api_key" and not spec.account_ref:
            raise HarnessNotConfigured(
                f"{self.describe().name} api_key mode needs an account_ref"
            )

    async def drive(self, run: RunContext) -> DriveOutcome:
        binary_path = self._launcher.resolve(self.binary)
        workspace = confine(run.workspace, Path(run.workspace or "."))
        config_dir = confine(run.workspace, workspace / RUN_CONFIG_DIR)
        config_dir.mkdir(parents=True, exist_ok=True)
        self.materialize(run, config_dir)
        call = self.invocation(run, binary_path, config_dir)
        process = await self._launcher.launch(
            call.argv,
            cwd=str(workspace),
            env=child_environment(call.env),
            stdin_text=call.stdin_text,
        )
        self._processes[run.spec.run_id] = process
        state = StreamState()
        try:
            async for line in process.lines():
                self._consume(run, line, state)
        except BaseException:
            await process.terminate()
            raise
        code = await process.wait()
        return self._outcome(run, state, code, process.stderr_tail())

    async def interrupt(self, run: RunContext) -> None:
        process = self._processes.get(run.spec.run_id)
        if process is not None:
            await process.terminate()

    # -- shared helpers -----------------------------------------------------

    def endpoint_tokens(self, run: RunContext) -> dict[str, str]:
        """Resolved bearer tokens by child variable name (never written to disk)."""
        tokens: dict[str, str] = {}
        for index, endpoint in enumerate(run.spec.toolset.endpoints()):
            if endpoint.bearer_ref is None:
                continue
            value = self._credentials.resolve(endpoint.bearer_ref)
            if not value:
                raise HarnessNotConfigured(
                    f"MCP endpoint {endpoint.name!r} credential did not resolve"
                )
            tokens[token_env_name(index)] = value
        return tokens

    @property
    def credentials(self) -> CredentialResolver:
        return self._credentials

    def launch_env(self, run: RunContext, api_key_var: str) -> dict[str, str]:
        """Child variables: MCP tokens, plus the API key in ``api_key`` mode."""
        env = self.endpoint_tokens(run)
        if run.spec.account_mode == "api_key":
            name = self.describe().name
            env[api_key_var] = api_key_for(run.spec, self._credentials, name)
        return env

    def _consume(self, run: RunContext, line: str, state: StreamState) -> None:
        text = line.strip()
        if not text:
            return
        try:
            record = json.loads(text)
        except json.JSONDecodeError:
            run.emit("step", "claim", name="unparsed-line", detail=text[:2_000])
            return
        if not isinstance(record, dict):
            return
        handler = self.record_handlers.get(self.record_type(record))
        if handler is not None:
            handler(run, record, state)

    def _outcome(
        self, run: RunContext, state: StreamState, code: int, stderr: str
    ) -> DriveOutcome:
        if not (state.terminal and not state.failed and code == 0):
            reason = state.error or f"exit {code}: {stderr.strip()[-2_000:]}"
            if state.tool_calls == 0:
                raise HarnessRunFailed(f"{run.negotiated.harness} failed: {reason}")
            raise HarnessError(
                f"{run.negotiated.harness} failed after tool use: {reason}"
            )
        proves_inventory = self.describe().tool_proof == "startup_inventory"
        if proves_inventory and not state.inventory_verified:
            raise HarnessToolInventoryMismatch(
                f"{run.negotiated.harness} emitted no startup inventory"
            )
        return DriveOutcome(
            status="succeeded",
            output=state.output or "".join(state.text_chunks),
            usage=state.usage,
            provider_session=run.provider_session,
        )


def mcp_server_entries(endpoints: tuple[McpEndpoint, ...]) -> dict[str, dict]:
    """Claude-Code-compatible ``mcpServers`` entries; tokens stay in the env."""
    entries: dict[str, dict] = {}
    for index, endpoint in enumerate(endpoints):
        entry: dict[str, object] = {"type": endpoint.transport, "url": endpoint.url}
        if endpoint.bearer_ref is not None:
            entry["headers"] = {
                "Authorization": "Bearer ${" + token_env_name(index) + "}"
            }
        entries[endpoint.name] = entry
    return entries


def write_skills(run: RunContext, skills_root: Path) -> None:
    """Materialize the run's digest-pinned ``SKILL.md`` bodies (§6.5)."""
    for skill in run.spec.toolset.skills:
        directory = confine(run.workspace, skills_root / skill.name)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "SKILL.md").write_text(skill.body, encoding="utf-8")


__all__ = [
    "CLI_DESCRIPTOR_DEFAULTS",
    "RUN_CONFIG_DIR",
    "CliHarness",
    "Handler",
    "Invocation",
    "as_mapping",
    "StreamState",
    "mcp_server_entries",
    "token_env_name",
    "write_skills",
]
