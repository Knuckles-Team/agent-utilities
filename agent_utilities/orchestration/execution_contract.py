"""Closed contracts for the GraphOS delegation execution route."""

from __future__ import annotations

from typing import Literal, cast

ExecutionMode = Literal["auto", "pydantic_graph"]

_EXECUTION_MODES = frozenset({"auto", "pydantic_graph"})


def validate_execution_mode(value: str | None) -> ExecutionMode:
    """Return one supported execution mode or fail before dispatch."""

    normalized = str(value or "auto").strip().casefold()
    if normalized not in _EXECUTION_MODES:
        raise ValueError("execution_mode must be one of: auto, pydantic_graph")
    return cast(ExecutionMode, normalized)


def validate_tool_contract(
    allowed_tools: list[str] | None,
    required_tools: list[str] | None,
) -> tuple[list[str] | None, list[str] | None]:
    """Normalize and validate least-privilege and required-provenance tool sets."""

    def normalized(values: list[str] | None) -> list[str] | None:
        if not values:
            return None
        result = [str(value).strip() for value in values if str(value).strip()]
        if len(result) != len(set(result)):
            raise ValueError("delegated tool lists must not contain duplicates")
        return result or None

    allowed = normalized(allowed_tools)
    required = normalized(required_tools)
    if required and not allowed:
        raise ValueError("required_tools requires an explicit allowed_tools catalog")
    if required and not set(required).issubset(set(allowed or [])):
        raise PermissionError("required_tools must be a subset of allowed_tools")
    return allowed, required


def validate_pydantic_graph_contract(
    execution_mode: ExecutionMode,
    *,
    skill_name: str | None,
    tool_server: str | None,
    allowed_tools: list[str] | None,
) -> None:
    """Require every authority input needed by the explicit graph route."""

    if execution_mode != "pydantic_graph":
        return
    missing: list[str] = []
    if not str(skill_name or "").strip():
        missing.append("skill_name")
    if not str(tool_server or "").strip():
        missing.append("tool_server")
    if not allowed_tools:
        missing.append("allowed_tools")
    if missing:
        raise ValueError(
            "execution_mode=pydantic_graph requires explicit " + ", ".join(missing)
        )


def missing_required_tools(
    required_tools: list[str] | None,
    observed_tools: list[str],
    *,
    observed_aliases: dict[str, str] | None = None,
) -> list[str]:
    """Return required tools with no matching recorded ToolCall."""

    if not required_tools:
        return []
    observed = {str(name).strip() for name in observed_tools if str(name).strip()}
    observed.update(
        str(alias).strip() for alias in (observed_aliases or {}) if str(alias).strip()
    )
    return [name for name in required_tools if name not in observed]
