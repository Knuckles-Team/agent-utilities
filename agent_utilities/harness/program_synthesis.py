#!/usr/bin/python
from __future__ import annotations

"""Inductive program synthesis over a typed primitive DSL with an Occam prior.

CONCEPT:AU-KG.coordination.inductive-program-synthesis-search — inductive program-synthesis search over a typed primitive composition DSL biased by a minimum-description-length prior so among programs that fit the examples the shortest is preferred a finite stand-in for the Solomonoff universal prior

AU's only "synthesis" was evolving scikit regressors (AHE-3.3) or prompt/agent
variants (AHE-3.2) — search over a fixed parametric or textual space, never over
*programs*. The paper grounds intelligence (§4) in search through hypothesis-space
under a low-complexity (Solomonoff/Occam) prior. This module searches a structured
program space — left-to-right compositions of provided pure primitive operations —
for the shortest program that fits a set of input→output examples, ranking
candidates with the MDL selection prior (``selection_operators.select_top_k`` in
``mdl`` mode). The primitive set is the finite, validatable stand-in for the
universal prior; the synthesized program can be rendered to source and validated in
the ORCH-1.38 sandbox before use.
"""

import itertools
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agent_utilities.harness.selection_operators import select_top_k

#: A primitive is a pure ``value -> value`` function, named in the DSL.
Primitive = Callable[[Any], Any]


@dataclass
class ProgramCandidate:
    """One synthesized program: an ordered composition of primitive op names."""

    ops: tuple[str, ...]
    score: float = 0.0  # fraction of examples satisfied
    length: int = 0  # description length (number of ops) — the MDL penalty axis

    def __post_init__(self) -> None:
        if not self.length:
            self.length = len(self.ops)

    def run(self, primitives: dict[str, Primitive], value: Any) -> Any:
        """Apply the composition left-to-right to ``value``."""
        for op in self.ops:
            value = primitives[op](value)
        return value

    def render(self) -> str:
        """A pure-Python source rendering for sandbox validation (ORCH-1.38)."""
        body = "x"
        for op in self.ops:
            body = f"{op}({body})"
        return f"def program(x):\n    return {body}\n"

    def as_dict(self) -> dict[str, Any]:
        return {"ops": list(self.ops), "score": self.score, "length": self.length}


def _score(cand: ProgramCandidate, primitives: dict[str, Primitive], examples) -> float:
    ok = 0
    for inp, want in examples:
        try:
            ok += 1 if cand.run(primitives, inp) == want else 0
        except Exception:  # noqa: BLE001 — a crashing program simply fails the example
            pass
    return ok / len(examples) if examples else 0.0


def synthesize(
    primitives: dict[str, Primitive],
    examples: list[tuple[Any, Any]],
    *,
    max_depth: int = 3,
    mdl_weight: float = 0.5,
    require_exact: bool = True,
) -> ProgramCandidate | None:
    """Search compositions up to ``max_depth`` for the shortest fitting program.

    Enumerates every composition of ``primitives`` (including the empty/identity
    program) up to ``max_depth`` ops, scores each on ``examples``, and selects the
    winner with the MDL prior — so among equally-correct programs the *shortest*
    wins (Occam). Returns ``None`` when ``require_exact`` and no program satisfies
    every example.
    """
    names = sorted(primitives)
    candidates: list[ProgramCandidate] = []
    for depth in range(max_depth + 1):
        for combo in itertools.product(names, repeat=depth):
            cand = ProgramCandidate(ops=combo)
            cand.score = _score(cand, primitives, examples)
            candidates.append(cand)

    rows = [{"cand": c, "score": c.score, "length": c.length} for c in candidates]
    best = select_top_k(
        rows,
        1,
        method="mdl",
        score_key="score",
        length_key="length",
        mdl_weight=mdl_weight,
    )
    if not best:
        return None
    winner: ProgramCandidate = best[0]["cand"]
    if require_exact and winner.score < 1.0:
        return None
    return winner


@dataclass
class SynthesisResult:
    """A synthesized program plus its sandbox-validation verdict."""

    program: ProgramCandidate
    source: str
    validated: bool = False
    detail: str = ""
    extras: dict[str, Any] = field(default_factory=dict)


def synthesize_and_validate(
    primitives: dict[str, Primitive],
    examples: list[tuple[Any, Any]],
    *,
    max_depth: int = 3,
    mdl_weight: float = 0.5,
    sandbox: Any = None,
) -> SynthesisResult | None:
    """Synthesize the shortest fitting program and (optionally) sandbox-validate it.

    When a ``sandbox`` with a ``validate(source) -> (ok, detail)`` callable is given,
    the rendered program source is checked before the result is returned; otherwise
    only synthesis runs (``validated`` stays ``False``).
    """
    program = synthesize(
        primitives, examples, max_depth=max_depth, mdl_weight=mdl_weight
    )
    if program is None:
        return None
    source = program.render()
    validated, detail = False, "not validated"
    if sandbox is not None:
        try:
            validated, detail = sandbox.validate(source)
        except Exception as exc:  # noqa: BLE001 — a sandbox failure is non-fatal
            validated, detail = False, f"sandbox error: {exc}"
    return SynthesisResult(
        program=program, source=source, validated=validated, detail=detail
    )
