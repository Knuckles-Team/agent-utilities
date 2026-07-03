"""CONCEPT:ORCH-1.28 — Composable Skills + Generic Environment Adapter.

Assimilated from predict-rlm (`Trampoline-AI/predict-rlm@edaddfe`, `src/predict_rlm/rlm_skills.py`)
and the AppWorld RLM-GEPA "less harness" thesis. Two pieces:

1. **Composable Skill** — upgrades skill-as-SOP from raw source-mounting to structured units that
   bundle ``instructions`` + ``packages`` + ``modules`` + ``tools``. :func:`merge_skills`
   deduplicates packages, concatenates instructions under per-skill headers, and raises on module/
   tool **name conflicts** (so composition is explicit, not silently last-wins).

2. **Generic Environment Adapter** — a minimal tool surface
   (``list_items`` / ``describe`` / ``call`` / ``SUBMIT``) over an external environment whose state
   and **evaluator are preserved**. The RLM supplies only the policy (the optimizable skill); the
   host owns scoring. This is the "expose a small set of tools, define a skill as a standard
   operating procedure, let the model decide how to proceed" pattern — same task, less harness.

Pure and dependency-free; fully unit-testable.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from pydantic import BaseModel, Field


class Skill(BaseModel):
    """A composable unit of agent capability (CONCEPT:ORCH-1.28)."""

    name: str
    instructions: str = ""
    packages: list[str] = Field(default_factory=list)  # PyPI deps the skill needs
    modules: dict[str, str] = Field(default_factory=dict)  # module_name -> source
    tools: dict[str, str] = Field(default_factory=dict)  # tool_name -> source


def merge_skills(skills: list[Skill], *, name: str = "merged") -> Skill:
    """Compose multiple skills into one, with explicit conflict detection.

    - ``packages``: order-preserving dedup.
    - ``instructions``: concatenated under ``## Skill: <name>`` headers.
    - ``modules`` / ``tools``: merged; a **name collision raises ValueError** (no silent override).
    """
    packages: list[str] = []
    instr_parts: list[str] = []
    modules: dict[str, str] = {}
    tools: dict[str, str] = {}
    for s in skills:
        for pkg in s.packages:
            if pkg not in packages:
                packages.append(pkg)
        if s.instructions.strip():
            instr_parts.append(f"## Skill: {s.name}\n{s.instructions.strip()}")
        for mname, src in s.modules.items():
            if mname in modules:
                raise ValueError(f"Module name conflict: {mname!r}")
            modules[mname] = src
        for tname, src in s.tools.items():
            if tname in tools:
                raise ValueError(f"Tool name conflict: {tname!r}")
            tools[tname] = src
    return Skill(
        name=name,
        instructions="\n\n".join(instr_parts),
        packages=packages,
        modules=modules,
        tools=tools,
    )


# ── Generic Environment Adapter ─────────────────────────────────────────────────


class EnvironmentAdapter(Protocol):
    """Minimal tool surface over an external environment (CONCEPT:ORCH-1.28).

    The adapter preserves the environment's state and evaluator; the RLM supplies only the policy.
    """

    def list_items(self) -> list[str]: ...

    def describe(self, item: str) -> str: ...

    def call(self, item: str, **kwargs: Any) -> Any: ...

    def submit(self, answer: Any = None) -> dict[str, Any]: ...


class RegistryEnvironmentAdapter:
    """A concrete generic adapter backed by a callable registry + an external evaluator.

    ``registry`` maps item name → callable. ``evaluator`` (preserved, host-owned) maps the submitted
    answer + the call log → a score dict. This keeps the small ``list/describe/call/SUBMIT`` surface
    while never reimplementing the environment's scoring.
    """

    def __init__(
        self,
        registry: dict[str, Callable[..., Any]],
        *,
        descriptions: dict[str, str] | None = None,
        evaluator: Callable[[Any, list[dict[str, Any]]], dict[str, Any]] | None = None,
    ) -> None:
        self._registry = registry
        self._descriptions = descriptions or {}
        self._evaluator = evaluator
        self.call_log: list[dict[str, Any]] = []

    def list_items(self) -> list[str]:
        return sorted(self._registry)

    def describe(self, item: str) -> str:
        return self._descriptions.get(item, f"{item}: (no description)")

    def call(self, item: str, **kwargs: Any) -> Any:
        if item not in self._registry:
            raise KeyError(f"Unknown environment item: {item!r}")
        result = self._registry[item](**kwargs)
        self.call_log.append({"item": item, "kwargs": kwargs, "result": result})
        return result

    def submit(self, answer: Any = None) -> dict[str, Any]:
        """Complete the task; hand off to the preserved evaluator (host owns scoring)."""
        if self._evaluator is None:
            return {"submitted": True, "answer": answer, "calls": len(self.call_log)}
        return self._evaluator(answer, self.call_log)
