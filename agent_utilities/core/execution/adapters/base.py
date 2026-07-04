"""CONCEPT:AU-ORCH.adapter.multi-cli-adapter-dispatch — Multi-CLI Agent Adapter Registry (declarative adapter contract).

Assimilated from open-design's ``RuntimeAgentDef`` pattern (apps/daemon/src/runtimes/types.ts):
agent CLI behaviour is **data**, not bespoke per-CLI code. Each installed coding-agent CLI (or
OpenAI-compatible local endpoint) is described by an :class:`AdapterDefinition`; the
:class:`~agent_utilities.core.execution.adapters.registry.AdapterRegistry` auto-detects which are
available on ``PATH`` and the engine dispatches a step to one of them.

Superiority delta vs the source: open-design binds one CLI per project; agent-utilities binds these
adapters into the multi-agent parallel engine (ORCH-1.8) and records the producing adapter as
provenance in the KG, so a single plan can fan out across heterogeneous CLI backends.

See ``.specify/design/orch-1.33-multi-cli-adapter-registry/design.md``.
"""

from __future__ import annotations

import enum
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


class StreamFormat(enum.StrEnum):
    """How an adapter's stdout stream is parsed into canonical events.

    Mirrors open-design's ``streamFormat`` discriminator; each value maps to a handler in
    :mod:`agent_utilities.core.execution.stream_handlers`.
    """

    PLAIN = "plain"  # raw text on stdout → one terminal text event
    JSONL = "jsonl"  # newline-delimited JSON objects → typed events
    # Reserved for E2 / later adapters (handlers added incrementally):
    ACP_JSON_RPC = "acp-json-rpc"


class PromptDelivery(enum.StrEnum):
    """How the composed prompt reaches the spawned process.

    ``args`` bakes it into argv; ``stdin-text`` writes it to stdin and closes; ``stdin-jsonl`` writes
    a JSONL message and *leaves stdin open* for mid-turn tool-result injection (EPIC 2 / ORCH-1.35).
    """

    ARGS = "args"
    STDIN_TEXT = "stdin-text"
    STDIN_JSONL = "stdin-jsonl"


class ExecEventType(enum.StrEnum):
    """Canonical event types every stream handler normalises to."""

    START = "start"
    TEXT_DELTA = "text_delta"
    TOOL_USE = "tool_use"
    ERROR = "error"
    END = "end"


@dataclass(slots=True)
class ExecEvent:
    """A single canonical execution event (provider/CLI-agnostic).

    The whole point of the adapter layer: every backend — a CLI subprocess or an HTTP provider
    (ORCH-1.34) — is normalised to a stream of these.
    """

    type: ExecEventType
    text: str = ""
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AdapterDefinition:
    """Declarative description of one agent-CLI runtime backend.

    Attributes:
        id: Stable adapter id (e.g. ``"claude-code"``, ``"ollama"``, ``"generic-cmd"``).
        bin: Executable name resolved on ``PATH``.
        version_args: Args that print a version (used for detection / capability probing).
        build_args: ``(model, prompt) -> list[str]`` producing the spawn argv. When prompt is
            delivered via stdin the prompt arg is omitted by the builder.
        stream_format: How stdout is parsed (:class:`StreamFormat`).
        prompt_delivery: How the prompt is delivered (:class:`PromptDelivery`).
        fallback_models: Model ids surfaced when live model listing is unavailable (graceful
            degradation — the picker never shows an empty list).
        model_override_env_var: Operator-level env var that pins the model for headless deploys
            (open-design ``defaultModelEnvVar``); consulted when no model is requested.
        list_models: Optional callable returning live model ids; falls back to ``fallback_models``.
        env: Extra environment to inject into the spawned process.
    """

    id: str
    bin: str
    version_args: tuple[str, ...] = ("--version",)
    build_args: Callable[[str, str], list[str]] = field(
        default=lambda model, prompt: (
            ([] if not model else ["-m", model]) + ([prompt] if prompt else [])
        )
    )
    stream_format: StreamFormat = StreamFormat.PLAIN
    prompt_delivery: PromptDelivery = PromptDelivery.ARGS
    fallback_models: tuple[str, ...] = ()
    model_override_env_var: str | None = None
    list_models: Callable[[], list[str]] | None = None
    env: dict[str, str] = field(default_factory=dict)

    def resolve_model(
        self, requested: str | None, env: dict[str, str] | None = None
    ) -> str:
        """Resolve the effective model id: explicit request > override env var > first fallback > ""."""
        import os

        if requested:
            return requested
        if self.model_override_env_var:
            val = (env or os.environ).get(self.model_override_env_var)
            if val:
                return val
        return self.fallback_models[0] if self.fallback_models else ""


@dataclass(slots=True)
class DetectedAdapter:
    """Result of probing an :class:`AdapterDefinition` on the host (open-design ``detectAgents``)."""

    id: str
    available: bool
    path: str | None = None
    version: str | None = None
    models: tuple[str, ...] = ()
    error: str | None = None
