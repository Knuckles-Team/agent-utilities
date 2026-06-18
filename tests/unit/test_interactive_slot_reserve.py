"""Reserved interactive inference slot (CONCEPT:ORCH-1.59).

One capacity knob (``KG_LLM_CONCURRENCY``) with 1 slot always reserved for the interactive
path (messaging responder + graph-os-spawned pydantic-ai agents, which share the default
model). Background KG work is bounded to capacity − reserved.
"""

from __future__ import annotations

from agent_utilities.core.config import RESERVED_INTERACTIVE_INSTANCES, AgentConfig


def test_reserved_constant_is_one() -> None:
    assert RESERVED_INTERACTIVE_INSTANCES == 1


def _with_capacity(n: int) -> AgentConfig:
    # The field is alias-only, so set it by field name via model_copy.
    return AgentConfig().model_copy(update={"kg_llm_concurrency": n})


def test_background_concurrency_reserves_one() -> None:
    assert _with_capacity(8).background_llm_concurrency() == 7
    assert _with_capacity(4).background_llm_concurrency() == 3


def test_background_concurrency_floors_at_one() -> None:
    # Tiny endpoints (capacity 1) still leave background a usable slot.
    assert _with_capacity(1).background_llm_concurrency() == 1
