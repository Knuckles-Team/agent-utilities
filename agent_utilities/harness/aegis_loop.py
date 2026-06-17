"""AEGIS unified harness-evolution loop (CONCEPT:AHE-3.52).

HarnessX's AEGIS decomposes harness evolution into Digester → Planner → Evolver →
Critic. We wire our *existing* machinery into that shape — but the Critic is the
formal SHACL gate (AHE-3.53) reasoning over the harness ontology, and the
adaptation landscape (Planner) reads the **accumulated edit distribution from the
ontology**, so the loop sees cross-round concentration the paper's per-edit gate
cannot. The result: it self-corrects (diversifies the next edit) **before** the
sub-threshold-coupling tipping point that collapsed HarnessX's τ³-Bench run.

The stages are dependency-injected (an `evolver_fn`, optional `verifier_fn`,
`smoke_fn`, `normalize_fn`) so the loop runs offline with no LLM/engine — the same
pattern as `FastSlowController`/`SubstrateTrainer`.

Beyond the offline self-correction, the loop now exercises three HarnessX
mechanisms at runtime that were previously only modelled:

  * **Complete deterministic gate sequence (CONCEPT:AHE-3.60):** manifest-verify →
    config-normalization (canonical-form dedup) → build/smoke test → SHACL gate.
  * **Selective invocation + reputation audit (CONCEPT:AHE-3.57):** an actionability
    threshold and patience-based idle early-stop, plus a per-dimension ship-outcome
    ledger that diverts exploration away from a *declining-yield* dimension — the
    paper's quantitative under-exploration defense, complementary to the
    concentration gate (over-concentration ≠ declining yield).
  * **Variant isolation via ensemble routing (CONCEPT:AHE-3.59):** when an edit
    improves one task cluster but regresses another, FORK a new variant scoped to
    the improved cluster instead of rejecting it (the single-harness seesaw
    stagnation the paper documents on heterogeneous benchmarks). The no-regression
    SHACL shape is then evaluated *per variant*, over each variant's own cluster.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agent_utilities.harness.harness_gate import GateVerdict, HarnessGate

# evolver_fn(landscape) -> a candidate edit dict {id, dimension, fixes?, regresses?}
EvolverFn = Callable[[dict[str, Any]], dict[str, Any]]
# verifier_fn(edit) -> (ok, regressed_task_ids) — the ManifestVerifier check
VerifierFn = Callable[[dict[str, Any]], "tuple[bool, list[str]]"]
# smoke_fn(edit) -> bool — does the edited processor/tool instantiate and run?
SmokeFn = Callable[[dict[str, Any]], bool]
# normalize_fn(edit) -> str — canonical form for config-normalization dedup.
NormalizeFn = Callable[[dict[str, Any]], str]


@dataclass
class AegisDecision:
    """One AEGIS round outcome."""

    round: int
    shipped: bool
    edit: dict[str, Any] | None
    reasons: list[str] = field(default_factory=list)
    variant_id: str | None = None
    forked: bool = False


class AegisLoop:
    """Digester → Planner → Evolver → Critic over the harness ontology.

    The Critic runs the deterministic gate sequence: manifest verify (regressions,
    optional injected ``verifier_fn``) → config-normalization → smoke test → SHACL
    gate (concentration / no-regression / pathology) over the accumulated +
    candidate edits (and forked variants). An edit ships only if every check
    passes; the trace store (``shipped``) grows each round so the gate reasons over
    history.

    Variant isolation is OFF by default (``variant_capacity=0`` ⇒ the legacy
    single-harness loop) and enabled by setting a positive capacity.
    """

    def __init__(
        self,
        evolver_fn: EvolverFn,
        *,
        verifier_fn: VerifierFn | None = None,
        gate: HarnessGate | None = None,
        smoke_fn: SmokeFn | None = None,
        normalize_fn: NormalizeFn | None = None,
        variant_capacity: int = 0,
        patience: int = 0,
        hit_rate_floor: float = 0.0,
    ) -> None:
        self._evolver = evolver_fn
        self._verifier = verifier_fn
        self._gate = gate or HarnessGate()
        self._smoke = smoke_fn
        self._normalize = normalize_fn
        self._variant_capacity = max(0, int(variant_capacity))
        self._patience = max(0, int(patience))
        self._hit_rate_floor = float(hit_rate_floor)

        self.shipped: list[dict[str, Any]] = []
        # CONCEPT:AHE-3.59 — variant pool: each {id, cluster:set(task_ids),
        # applies:[edit_ids], status}. Seeded lazily with a base variant.
        self.variants: list[dict[str, Any]] = []
        # CONCEPT:AHE-3.57 — per-dimension ship-outcome ledger (reputation audit).
        self._ledger: dict[str, list[bool]] = {}
        self._idle = 0
        self._seen_norm: set[str] = set()

    # ── Planner ──────────────────────────────────────────────────────────────
    def dimension_hit_rate(self, dimension: str, window: int = 3) -> float | None:
        """Recent ship hit-rate for a dimension (CONCEPT:AHE-3.57).

        ``None`` if the dimension has no recorded attempts yet. The reputation
        audit reads the last ``window`` attempts — a *declining-yield* signal the
        concentration gate cannot see (it counts ships, not their success rate).
        """
        hist = self._ledger.get(dimension)
        if not hist:
            return None
        recent = hist[-window:]
        return sum(1 for ok in recent if ok) / len(recent)

    def adaptation_landscape(self) -> dict[str, Any]:
        """Planner: which dimensions are heavily edited (under-exploration defense)
        AND which are *declining in yield* (reputation audit) — read from the
        accumulated ontology edits + the ship-outcome ledger, not guessed."""
        used = Counter(e.get("dimension", "") for e in self.shipped)
        discouraged: list[str] = []
        concerns: list[str] = []
        for dim, hist in self._ledger.items():
            if len(hist) < 2:
                continue
            hr = self.dimension_hit_rate(dim)
            # Shipped into in ≥2 of the last 3 attempts with hit-rate below the
            # floor ⇒ declining yield: discourage further edits there.
            if hr is not None and hr < self._hit_rate_floor:
                discouraged.append(dim)
                concerns.append(
                    f"strategy_concern: dimension {dim!r} hit-rate {hr:.2f} "
                    f"< floor {self._hit_rate_floor:.2f} — diverting exploration"
                )
        return {
            "used": dict(used),
            "shipped_count": len(self.shipped),
            "hot_dimension": (used.most_common(1)[0][0] if used else None),
            "discouraged_dimensions": discouraged,
            "strategy_concerns": concerns,
        }

    # ── Variant routing (CONCEPT:AHE-3.59) ───────────────────────────────────
    def _base_variant(self) -> dict[str, Any]:
        if not self.variants:
            self.variants.append(
                {
                    "id": "variant:base",
                    "cluster": set(),
                    "applies": [],
                    "status": "accepted",
                }
            )
        return self.variants[0]

    def _route_and_fork(
        self, candidate: dict[str, Any], fixes: set[str], regresses: set[str]
    ) -> tuple[dict[str, Any], bool]:
        """Pick the variant this edit applies to, forking on a mixed edit.

        Returns ``(variant, forked)``. A *mixed* edit (improves a subset, regresses
        others) forks a NEW variant scoped to the improved cluster — so its
        per-variant seesaw sees no within-scope regression — instead of being
        rejected outright (the paper's heterogeneous-task stagnation fix). When
        capacity is reached the lowest-cluster-size variant is retired.
        """
        base = self._base_variant()
        # An edit that regresses tasks OUTSIDE the base cluster but fixes others is
        # mixed → fork a variant scoped to its own fix-cluster.
        mixed = bool(fixes) and bool(regresses)
        if not mixed:
            base["cluster"] |= fixes
            return base, False
        vid = f"variant:{candidate.get('id', 'v')}-{len(self.variants)}"
        forked = {
            "id": vid,
            "cluster": set(fixes),  # scoped: only the tasks this edit improves
            "applies": [],
            "status": "accepted",
        }
        self.variants.append(forked)
        if self._variant_capacity and len(self.variants) > self._variant_capacity:
            # Retire the smallest-cluster non-base variant.
            retireable = [v for v in self.variants[1:] if v["id"] != vid]
            if retireable:
                victim = min(retireable, key=lambda v: len(v["cluster"]))
                self.variants.remove(victim)
        return forked, True

    def _variant_facts(self) -> list[dict[str, Any]]:
        """Project the variant pool into the gate's dict shape, with each variant's
        causesRegression SCOPED to its own cluster (the per-variant seesaw)."""
        facts: list[dict[str, Any]] = []
        for v in self.variants:
            facts.append(
                {
                    "id": v["id"],
                    "status": v.get("status", "accepted"),
                    "applies": list(v.get("applies", [])),
                }
            )
        return facts

    # ── Critic + gate sequence ───────────────────────────────────────────────
    def run_round(self, round_idx: int) -> AegisDecision:
        landscape = self.adaptation_landscape()

        # Selective invocation (CONCEPT:AHE-3.57): the Evolver may decline to
        # produce a candidate (empty landscape / nothing actionable) — short-circuit
        # the round as idle instead of forcing an edit.
        raw = self._evolver(landscape)
        if not raw:
            self._idle += 1
            return AegisDecision(round_idx, False, None, ["no actionable candidate"])
        candidate = dict(raw)
        candidate.setdefault("round", round_idx)
        candidate.setdefault("status", "shipped")
        dim = candidate.get("dimension", "")
        fixes = set(candidate.get("fixes", []) or [])
        regresses_decl = set(candidate.get("regresses", []) or [])

        # Critic stage 1 — manifest verify (regressions).
        regresses: list[str] = sorted(regresses_decl)
        if self._verifier is not None:
            ok, regresses = self._verifier(candidate)
            if not ok:
                self._record(dim, False)
                return AegisDecision(
                    round_idx, False, candidate, ["manifest verify failed"]
                )
        regresses_decl = set(regresses)
        candidate["regresses"] = sorted(regresses_decl)

        # Critic stage 2 — config-normalization (CONCEPT:AHE-3.60): canonical-form
        # dedup so a re-proposed identical edit is not double-counted toward
        # concentration (and cannot masquerade as fresh progress).
        if self._normalize is not None:
            norm = self._normalize(candidate)
            if norm in self._seen_norm:
                self._idle += 1
                return AegisDecision(
                    round_idx, False, candidate, ["duplicate edit (config-normalized)"]
                )

        # Critic stage 3 — build/smoke test (CONCEPT:AHE-3.60): does the edited
        # processor/tool instantiate and run? A failed smoke never reaches the gate.
        if self._smoke is not None:
            passed = bool(self._smoke(candidate))
            candidate["smoke_passed"] = passed
            if not passed:
                self._record(dim, False)
                return AegisDecision(round_idx, False, candidate, ["smoke test failed"])

        # Variant routing (CONCEPT:AHE-3.59) — fork on a mixed edit when isolation
        # is enabled, so the per-variant seesaw is scoped to the improved cluster.
        forked = False
        variant: dict[str, Any] | None = None
        variant_facts: list[dict[str, Any]] | None = None
        if self._variant_capacity > 0:
            variant, forked = self._route_and_fork(candidate, fixes, regresses_decl)
            variant["applies"] = [*variant.get("applies", []), candidate["id"]]
            # Within-scope regressions only: tasks the edit breaks that ARE in this
            # variant's cluster. Out-of-scope breaks route to other variants.
            scoped = sorted(regresses_decl & set(variant["cluster"]))
            candidate["regresses"] = scoped
            variant_facts = self._variant_facts()

        # Critic stage 4 — SHACL gate over accumulated + candidate (+ variants): the
        # formal seesaw + concentration gate; rejects BEFORE the coupling tipping
        # point. With variant isolation, the no-regression seesaw is evaluated per
        # variant over its own cluster.
        trial = [*self.shipped, candidate]
        verdict: GateVerdict = self._gate.check_facts(trial, variants=variant_facts)
        if not verdict.passed:
            if forked and variant is not None:
                self.variants.remove(variant)  # roll back the speculative fork
            self._record(dim, False)
            return AegisDecision(round_idx, False, candidate, verdict.reasons)

        self.shipped.append(candidate)
        self._record(dim, True)
        self._idle = 0
        if self._normalize is not None:
            self._seen_norm.add(self._normalize(candidate))
        return AegisDecision(
            round_idx,
            True,
            candidate,
            list(landscape.get("strategy_concerns", [])),
            variant_id=(variant["id"] if variant else None),
            forked=forked,
        )

    def _record(self, dimension: str, ok: bool) -> None:
        if dimension:
            self._ledger.setdefault(dimension, []).append(ok)

    def run(self, rounds: int) -> list[AegisDecision]:
        """Run up to ``rounds`` AEGIS rounds.

        Patience early-stop (CONCEPT:AHE-3.57): if ``patience`` is set and that many
        consecutive rounds ship nothing, the loop stops — no point burning rounds on
        an exhausted landscape.
        """
        decisions: list[AegisDecision] = []
        for r in range(1, rounds + 1):
            decisions.append(self.run_round(r))
            if self._patience and self._idle >= self._patience:
                break
        return decisions
