"""CONCEPT:AU-ORCH.adapter.builtin-adapter-definitions — Built-in adapter definitions.

The product supplies only a neutral command contract. Operators register concrete
runtime adapters and discovery callbacks without changing the execution engine.
"""

from __future__ import annotations

from ..base import AdapterDefinition, PromptDelivery, StreamFormat


def _generic_cmd_args(model: str, prompt: str) -> list[str]:
    # Generic single-shot CLI: model as -m, prompt as the trailing positional arg.
    return (["-m", model] if model else []) + ([prompt] if prompt else [])


GENERIC_CMD = AdapterDefinition(
    id="generic-cmd",
    bin="true",  # placeholder bin; real deployments register their own generic adapter
    build_args=_generic_cmd_args,
    stream_format=StreamFormat.PLAIN,
    prompt_delivery=PromptDelivery.ARGS,
)

BUILTIN_ADAPTERS: tuple[AdapterDefinition, ...] = (GENERIC_CMD,)

__all__ = ["BUILTIN_ADAPTERS", "GENERIC_CMD"]
