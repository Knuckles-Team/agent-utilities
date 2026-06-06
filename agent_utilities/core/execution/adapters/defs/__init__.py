"""CONCEPT:ORCH-1.33 — Built-in adapter definitions.

Declarative descriptions of common agent-CLI / local-LLM backends. Each is pure data; the registry
detects which are actually installed. Add a backend by appending an :class:`AdapterDefinition` here
(or registering one at runtime) — no engine changes required.
"""

from __future__ import annotations

from ..base import AdapterDefinition, PromptDelivery, StreamFormat


def _claude_args(model: str, prompt: str) -> list[str]:
    # Claude Code: headless print mode, JSONL stream; prompt delivered via stdin (stream-json).
    args = ["-p", "--output-format", "stream-json", "--verbose"]
    if model:
        args += ["--model", model]
    return args


def _generic_cmd_args(model: str, prompt: str) -> list[str]:
    # Generic single-shot CLI: model as -m, prompt as the trailing positional arg.
    return (["-m", model] if model else []) + ([prompt] if prompt else [])


CLAUDE_CODE = AdapterDefinition(
    id="claude-code",
    bin="claude",
    version_args=("--version",),
    build_args=_claude_args,
    stream_format=StreamFormat.JSONL,
    prompt_delivery=PromptDelivery.STDIN_JSONL,
    fallback_models=(
        "claude-opus-4-8",
        "claude-sonnet-4-6",
        "claude-haiku-4-5-20251001",
    ),
    model_override_env_var="AGENT_UTILITIES_CLAUDE_MODEL",
)

# OpenAI-compatible local endpoints (Ollama, vLLM, LM Studio) are reached through the provider proxy
# (ORCH-1.34), but a CLI fronting them is described here for symmetry / direct spawn.
OLLAMA = AdapterDefinition(
    id="ollama",
    bin="ollama",
    version_args=("--version",),
    build_args=lambda model, prompt: [
        "run",
        model or "llama3",
        *([prompt] if prompt else []),
    ],
    stream_format=StreamFormat.PLAIN,
    prompt_delivery=PromptDelivery.ARGS,
    fallback_models=("llama3", "qwen2.5", "mistral"),
    model_override_env_var="AGENT_UTILITIES_OLLAMA_MODEL",
)

GENERIC_CMD = AdapterDefinition(
    id="generic-cmd",
    bin="true",  # placeholder bin; real deployments register their own generic adapter
    build_args=_generic_cmd_args,
    stream_format=StreamFormat.PLAIN,
    prompt_delivery=PromptDelivery.ARGS,
)

BUILTIN_ADAPTERS: tuple[AdapterDefinition, ...] = (CLAUDE_CODE, OLLAMA, GENERIC_CMD)

__all__ = ["BUILTIN_ADAPTERS", "CLAUDE_CODE", "OLLAMA", "GENERIC_CMD"]
