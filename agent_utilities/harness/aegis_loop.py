"""AEGIS unified harness-evolution loop (CONCEPT:AHE-3.52).

HarnessX's AEGIS decomposes harness evolution into Digester → Planner → Evolver →
Critic. We wire our *existing* machinery into that shape — but the Critic is the
formal SHACL gate (AHE-3.53) reasoning over the harness ontology, and the
adaptation landscape (Planner) reads the **accumulated edit distribution from the
ontology**, so the loop sees cross-round concentration the paper's per-edit gate
cannot. The result: it self-corrects (diversifies the next edit) **before** the
sub-threshold-coupling tipping point that collapsed HarnessX's τ³-Bench run.

The stages are dependency-injected (an `evolver_fn`, optional `verifier_fn`) so
the loop runs offline with no LLM/engine — the same pattern as
`FastSlowController`/`SubstrateTrainer`.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agent_utilities.harness.harness_gate import GateVerdict, HarnessGate

# evolver_fn(landscape) -> a candidate edit dict {id, dimension, ...}
EvolverFn = Callable[[dict[str, Any]], dict[str, Any]]
# verifier_fn(edit) -> (ok, regressed_task_ids) — the ManifestVerifier check
VerifierFn = Callable[[dict[str, Any]], "tuple[bool, list[str]]"]


@dataclass
class AegisDecision:
    """One AEGIS round outcome."""

    round: int
    shipped: bool
    edit: dict[str, Any] | None
    reasons: list[str] = field(default_factory=list)


class AegisLoop:
    """Digester → Planner → Evolver → Critic over the harness ontology.

    The Critic runs the deterministic gate sequence: manifest verify (regressions,
    optional injected `verifier_fn`) → SHACL gate (concentration / no-regression /
    pathology) over the accumulated + candidate edits. An edit ships only if every
    check passes; the trace store (`shipped`) grows each round so the gate reasons
    over history.
    """

    def __init__(
        self,
        evolver_fn: EvolverFn,
        *,
        verifier_fn: VerifierFn | None = None,
        gate: HarnessGate | None = None,
    ) -> None:
        self._evolver = evolver_fn
        self._verifier = verifier_fn
        self._gate = gate or HarnessGate()
        self.shipped: list[dict[str, Any]] = []

    def adaptation_landscape(self) -> dict[str, Any]:
        """Planner: which dimensions are already heavily edited (under-exploration
        defense) — read from the accumulated ontology edits, not guessed."""
        used = Counter(e.get("dimension", "") for e in self.shipped)
        return {
            "used": dict(used),
            "shipped_count": len(self.shipped),
            "hot_dimension": (used.most_common(1)[0][0] if used else None),
        }

    def run_round(self, round_idx: int) -> AegisDecision:
        landscape = self.adaptation_landscape()
        candidate = dict(self._evolver(landscape))
        candidate.setdefault("round", round_idx)
        candidate.setdefault("status", "shipped")

        # Critic stage 1 — manifest verify (regressions).
        regresses: list[str] = []
        if self._verifier is not None:
            ok, regresses = self._verifier(candidate)
            if not ok:
                return AegisDecision(
                    round_idx, False, candidate, ["manifest verify failed"]
                )
        candidate["regresses"] = regresses

        # Critic stage 2 — SHACL gate over accumulated + candidate (the formal
        # seesaw + concentration gate; rejects BEFORE the coupling tipping point).
        trial = [*self.shipped, candidate]
        verdict: GateVerdict = self._gate.check_facts(trial)
        if not verdict.passed:
            return AegisDecision(round_idx, False, candidate, verdict.reasons)

        self.shipped.append(candidate)
        return AegisDecision(round_idx, True, candidate, [])

    def run(self, rounds: int) -> list[AegisDecision]:
        """Run `rounds` AEGIS rounds, returning each decision."""
        return [self.run_round(r) for r in range(1, rounds + 1)]
