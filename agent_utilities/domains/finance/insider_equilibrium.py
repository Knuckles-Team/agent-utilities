"""Insider / stealth-trading equilibrium under dynamic legal risk — CONCEPT:KG-2.6.

Closed-form analytic layer that *deepens* the snapshot Kyle surveillance gate
(``surveillance_risk`` engine kernel, emerald-exchange CONCEPT:EE-042/EE-043) into
the full strategic model of **Qiao & Xia, "Insider and stealth trading with
dynamic legal risk"** (arXiv:2605.27684). Where the engine kernel answers *"how
toxic is the current flow?"*, this module answers *"how would a rational insider
trade against an endogenous, time-varying prosecution hazard, and which penalty
levers actually constrain them?"* — and is the model backing emerald's
``emerald_signals(action="insider_equilibrium")`` (CONCEPT:EE-044).

The math is a tractable, dependency-free reduction of the paper's continuous-time
game to linear insider strategies and a quadratic price-impact / detection cost,
which preserves the paper's three headline results exactly (see ``PolicyAnalysis``):

1. **Stealth + acceleration.** The insider scales intensity to the mispricing gap,
   and ``intensity_schedule`` rises toward the horizon as the *remaining* legal
   exposure decays — the paper's "trading accelerates as legal risk diminishes
   near the end of the window".
2. **Penalties alone don't substitute for enforcement.** The civil/financial
   penalty rate enters only the *denominator* of the optimal intensity and is
   scaled by enforcement effort ``e`` — so with weak enforcement, raising fines
   has vanishing effect.
3. **Criminal penalties are the binding lever.** A criminal (fixed, expected) cost
   enters the *numerator* subtractively and can drive equilibrium intensity to
   zero — a hard constraint that civil damages cannot replicate.

Pure Python (no numpy/scipy): this is a strategy/policy model, not a heavy numeric
kernel, so it lives in the finance domain alongside ``signal_fusion`` /
``regime_detector`` rather than the Rust engine. ``solve_equilibrium`` exposes a
``to_engine_args`` seam so the closed form can later be promoted to an engine
kernel without changing callers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

__all__ = [
    "InsiderEquilibriumInputs",
    "InsiderEquilibrium",
    "PolicyAnalysis",
    "solve_equilibrium",
    "intensity_schedule",
    "penalty_policy_analysis",
    "to_engine_args",
]


@dataclass(frozen=True)
class InsiderEquilibriumInputs:
    """Primitives of the continuous-time Kyle game with dynamic legal risk.

    All quantities are in the paper's normalized units. Defaults describe a
    moderately-enforced market where the insider holds a real informational edge.
    """

    sigma_v: float = 0.30  # std of the fundamental value v (informational edge)
    sigma_u: float = 1.00  # std of noise-trader order flow (the cover for stealth)
    gap_var: float | None = None  # E[(v-p)^2]; defaults to sigma_v**2 when None
    enforcement: float = 0.50  # regulator effort e ∈ [0,1] (surveillance budget)
    surveillance_kappa: float = 1.0  # how fast detection hazard grows with activity
    criminal_penalty: float = 0.0  # expected criminal cost C (fixed, hazard-scaled)
    civil_penalty_rate: float = 0.0  # civil damages as a multiple of gross profit
    horizon: float = 1.0  # trading window length T (continuous time)

    def resolved_gap_var(self) -> float:
        gv = self.gap_var if self.gap_var is not None else self.sigma_v**2
        return max(float(gv), 1e-12)


@dataclass
class InsiderEquilibrium:
    """Equilibrium of one (possibly time-sliced) insider trading problem."""

    intensity: float  # β* — optimal trade per unit of mispricing gap
    baseline_intensity: float  # β0 — Kyle intensity with NO legal risk
    kyle_lambda: float  # λ — market-maker price-impact coefficient
    detection_prob: float  # D(β*) ∈ [0,1] — endogenous prosecution hazard
    expected_profit: float  # E[π] gross of penalties, net of price impact
    expected_penalty: float  # E[penalty] = D·(criminal + civil)
    net_value: float  # E[π] − E[penalty] — the insider's objective at β*
    suppressed: bool  # True when legal risk drove β* to the zero floor
    binding_lever: str  # "criminal" | "civil" | "enforcement" | "none"

    def to_dict(self) -> dict[str, Any]:
        return {
            "intensity": self.intensity,
            "baseline_intensity": self.baseline_intensity,
            "kyle_lambda": self.kyle_lambda,
            "detection_prob": self.detection_prob,
            "expected_profit": self.expected_profit,
            "expected_penalty": self.expected_penalty,
            "net_value": self.net_value,
            "suppressed": self.suppressed,
            "binding_lever": self.binding_lever,
        }


@dataclass
class PolicyAnalysis:
    """Comparative-statics verdict on penalty design (the paper's §results)."""

    d_intensity_d_criminal: float  # ∂β*/∂C — criminal sensitivity (should be < 0)
    d_intensity_d_civil: float  # ∂β*/∂r_civ — civil sensitivity (≈0 when e→0)
    criminal_intensity_floor: float  # criminal cost that fully suppresses (β*→0)
    civil_only_min_intensity: float  # inf over r_civ of β* with C=0 (the floor fines hit)
    enforcement_gated: bool  # True ⇒ civil lever is throttled by weak enforcement
    verdict: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "d_intensity_d_criminal": self.d_intensity_d_criminal,
            "d_intensity_d_civil": self.d_intensity_d_civil,
            "criminal_intensity_floor": self.criminal_intensity_floor,
            "civil_only_min_intensity": self.civil_only_min_intensity,
            "enforcement_gated": self.enforcement_gated,
            "verdict": self.verdict,
        }


def _kyle_lambda(p: InsiderEquilibriumInputs) -> float:
    """Kyle's price-impact λ = ½·σ_v/σ_u (the single-auction closed form)."""
    su = max(p.sigma_u, 1e-9)
    return 0.5 * p.sigma_v / su


def solve_equilibrium(
    inputs: InsiderEquilibriumInputs | None = None,
    *,
    enforcement: float | None = None,
) -> InsiderEquilibrium:
    """Solve the insider's intensity β* in closed form.

    Maximizes ``J(β) = (β − λβ²)·Σ − e·κ·β·(C + r·β·Σ)`` where ``Σ`` is the
    mispricing variance, the first term is Kyle expected profit net of price
    impact, and ``e·κ·β`` is the activity-driven detection hazard. Setting
    ``dJ/dβ = 0`` gives

        β* = (Σ − e·κ·C) / (2·Σ·(λ + e·κ·r)).

    With no legal risk (``e=0``) this collapses to the Kyle baseline β0 = 1/(2λ)
    = σ_u/σ_v. The numerator carries the *criminal* cost (hard, can zero β*); the
    denominator carries the *civil* rate (soft, diminishing) — and both legal
    terms are gated by enforcement ``e``. ``enforcement`` overrides ``inputs`` so
    callers can sweep effort cheaply (used by ``intensity_schedule``).
    """
    p = inputs or InsiderEquilibriumInputs()
    e = p.enforcement if enforcement is None else float(enforcement)
    e = min(max(e, 0.0), 1.0)
    sigma = p.resolved_gap_var()
    lam = _kyle_lambda(p)
    kappa = max(p.surveillance_kappa, 0.0)
    crim = max(p.criminal_penalty, 0.0)
    civ = max(p.civil_penalty_rate, 0.0)

    baseline = 1.0 / (2.0 * lam) if lam > 0 else 0.0

    denom = 2.0 * sigma * (lam + e * kappa * civ)
    numer = sigma - e * kappa * crim
    beta = numer / denom if denom > 0 else 0.0
    suppressed = beta <= 0.0
    beta = max(beta, 0.0)

    # Endogenous detection hazard at the chosen intensity, bounded to a probability.
    detection = min(max(e * kappa * beta, 0.0), 1.0)
    gross = beta * sigma  # gross profit proxy E[(v-p)·x] = β·Σ
    expected_profit = (beta - lam * beta * beta) * sigma
    expected_penalty = detection * (crim + civ * gross)
    net_value = expected_profit - expected_penalty

    if suppressed:
        binding = "criminal"
    elif e <= 1e-9 or kappa <= 1e-9:
        binding = "none"
    elif e * kappa * crim > 0 and crim >= civ * gross:
        binding = "criminal"
    elif civ > 0:
        binding = "civil"
    else:
        binding = "enforcement"

    return InsiderEquilibrium(
        intensity=beta,
        baseline_intensity=baseline,
        kyle_lambda=lam,
        detection_prob=detection,
        expected_profit=expected_profit,
        expected_penalty=expected_penalty,
        net_value=net_value,
        suppressed=suppressed,
        binding_lever=binding,
    )


def intensity_schedule(
    inputs: InsiderEquilibriumInputs | None = None,
    *,
    steps: int = 10,
) -> list[dict[str, float]]:
    """Continuous-time β*(t) over [0, T] as remaining legal exposure decays.

    Effective enforcement decays linearly with the remaining window,
    ``e(t) = e0·(T − t)/T``: near the end there is little time left to detect and
    prosecute, so the hazard relaxes and the insider *accelerates*. Returns a list
    of ``{t, remaining, enforcement, intensity, detection_prob}`` samples — the
    intensity series is non-decreasing in ``t`` whenever a criminal cost is present
    (the paper's end-of-window acceleration).
    """
    p = inputs or InsiderEquilibriumInputs()
    n = max(int(steps), 1)
    out: list[dict[str, float]] = []
    for i in range(n + 1):
        t = p.horizon * i / n
        remaining = (p.horizon - t) / p.horizon if p.horizon > 0 else 0.0
        e_t = p.enforcement * remaining
        eq = solve_equilibrium(p, enforcement=e_t)
        out.append(
            {
                "t": t,
                "remaining": remaining,
                "enforcement": e_t,
                "intensity": eq.intensity,
                "detection_prob": eq.detection_prob,
            }
        )
    return out


def penalty_policy_analysis(
    inputs: InsiderEquilibriumInputs | None = None,
) -> PolicyAnalysis:
    """Reproduce the paper's penalty-design results via comparative statics.

    From ``β* = (Σ − eκC) / (2Σ(λ + eκr))``:

    * ``∂β*/∂C = −eκ / (2Σ(λ + eκr))`` — strictly negative, magnitude scaled by
      enforcement; the criminal lever bites and ``C ≥ Σ/(eκ)`` zeroes β*.
    * ``∂β*/∂r = −eκ(Σ − eκC) / (2Σ(λ + eκr)²)`` — also negative but *bounded*:
      as ``r → ∞`` β* → 0 only asymptotically, and the whole effect carries a
      factor ``e`` so weak enforcement throttles it. The civil-only floor
      (``C = 0, r → ∞``) is 0 in the limit but approached arbitrarily slowly,
      whereas a finite criminal cost reaches 0 exactly.
    """
    p = inputs or InsiderEquilibriumInputs()
    e = min(max(p.enforcement, 0.0), 1.0)
    sigma = p.resolved_gap_var()
    lam = _kyle_lambda(p)
    kappa = max(p.surveillance_kappa, 0.0)
    crim = max(p.criminal_penalty, 0.0)
    civ = max(p.civil_penalty_rate, 0.0)
    ek = e * kappa

    denom = 2.0 * sigma * (lam + ek * civ)
    d_crim = (-ek / denom) if denom > 0 else 0.0
    d_civ = (
        (-ek * (sigma - ek * crim) / (2.0 * sigma * (lam + ek * civ) ** 2))
        if (lam + ek * civ) > 0
        else 0.0
    )
    # Criminal cost that fully suppresses the insider: Σ − eκC ≤ 0 ⇒ C ≥ Σ/(eκ).
    crim_floor = (sigma / ek) if ek > 1e-12 else math.inf
    # Civil-only minimum intensity (C=0, r→∞): asymptotes to 0 but never reached.
    civ_only_min = 0.0
    enforcement_gated = e < 0.25  # weak-enforcement regime where fines lose bite

    if enforcement_gated:
        verdict = (
            "Enforcement is weak: civil/financial penalties have vanishing effect "
            "(∂β*/∂r is scaled by e). Raising fines cannot substitute for "
            "surveillance effort — only a criminal cost (or more enforcement) "
            "constrains the insider."
        )
    elif crim > 0 and crim >= crim_floor:
        verdict = (
            "Criminal penalty exceeds the suppression floor Σ/(eκ): equilibrium "
            "intensity is driven to zero — the binding, hard constraint."
        )
    else:
        verdict = (
            "Criminal cost enters β* subtractively (can reach zero exactly) while "
            "civil damages only inflate the denominator (diminishing, never zero). "
            "Criminal sanctions are the effective lever on aggressive intensity."
        )

    return PolicyAnalysis(
        d_intensity_d_criminal=d_crim,
        d_intensity_d_civil=d_civ,
        criminal_intensity_floor=crim_floor,
        civil_only_min_intensity=civ_only_min,
        enforcement_gated=enforcement_gated,
        verdict=verdict,
    )


def to_engine_args(inputs: InsiderEquilibriumInputs) -> dict[str, Any]:
    """Flatten inputs for a future ``engine.finance.insider_equilibrium`` kernel.

    Seam only — keeps the promotion path to the Rust engine explicit without
    coupling callers to the closed form (mirrors how portfolio optimization was
    promoted from a Python prototype to the engine).
    """
    return {
        "sigma_v": inputs.sigma_v,
        "sigma_u": inputs.sigma_u,
        "gap_var": inputs.resolved_gap_var(),
        "enforcement": inputs.enforcement,
        "surveillance_kappa": inputs.surveillance_kappa,
        "criminal_penalty": inputs.criminal_penalty,
        "civil_penalty_rate": inputs.civil_penalty_rate,
        "horizon": inputs.horizon,
    }
