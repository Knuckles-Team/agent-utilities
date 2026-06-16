"""Insider-equilibrium model tests — CONCEPT:KG-2.6 (distils arXiv:2605.27684).

Verifies the closed-form reduction reproduces the three headline results of
Qiao & Xia: Kyle baseline recovery with no legal risk, criminal penalties as the
hard (exactly-suppressing) lever, civil fines as a diminishing enforcement-gated
lever, and the end-of-window acceleration schedule.
"""

import math

from agent_utilities.domains.finance.insider_equilibrium import (
    InsiderEquilibriumInputs,
    intensity_schedule,
    penalty_policy_analysis,
    solve_equilibrium,
)


def test_no_legal_risk_recovers_kyle_baseline():
    # enforcement=0 ⇒ legal terms vanish ⇒ β* == Kyle baseline σ_u/σ_v.
    p = InsiderEquilibriumInputs(sigma_v=0.3, sigma_u=1.0, enforcement=0.0)
    eq = solve_equilibrium(p)
    assert math.isclose(eq.kyle_lambda, 0.5 * 0.3 / 1.0, rel_tol=1e-9)
    assert math.isclose(eq.baseline_intensity, 1.0 / (2 * eq.kyle_lambda), rel_tol=1e-9)
    assert math.isclose(eq.intensity, eq.baseline_intensity, rel_tol=1e-9)
    assert eq.binding_lever == "none"
    assert not eq.suppressed


def test_criminal_penalty_suppresses_intensity_exactly():
    base = solve_equilibrium(
        InsiderEquilibriumInputs(enforcement=0.6, surveillance_kappa=1.0)
    )
    # A small criminal cost lowers intensity; a large one drives it to zero.
    mild = solve_equilibrium(
        InsiderEquilibriumInputs(
            enforcement=0.6, surveillance_kappa=1.0, criminal_penalty=0.05
        )
    )
    assert mild.intensity < base.intensity
    huge = solve_equilibrium(
        InsiderEquilibriumInputs(
            enforcement=0.6, surveillance_kappa=1.0, criminal_penalty=100.0
        )
    )
    assert huge.intensity == 0.0
    assert huge.suppressed
    assert huge.binding_lever == "criminal"


def test_civil_penalty_diminishes_but_never_zeroes():
    base = solve_equilibrium(
        InsiderEquilibriumInputs(enforcement=0.6, surveillance_kappa=1.0)
    )
    civ = solve_equilibrium(
        InsiderEquilibriumInputs(
            enforcement=0.6, surveillance_kappa=1.0, civil_penalty_rate=2.0
        )
    )
    civ_more = solve_equilibrium(
        InsiderEquilibriumInputs(
            enforcement=0.6, surveillance_kappa=1.0, civil_penalty_rate=20.0
        )
    )
    # Civil fines lower intensity monotonically but never reach exactly zero.
    assert base.intensity > civ.intensity > civ_more.intensity > 0.0


def test_civil_lever_is_gated_by_enforcement():
    # With enforcement off, raising civil fines does nothing (the paper's result 2).
    no_enf_base = solve_equilibrium(InsiderEquilibriumInputs(enforcement=0.0))
    no_enf_fined = solve_equilibrium(
        InsiderEquilibriumInputs(enforcement=0.0, civil_penalty_rate=50.0)
    )
    assert math.isclose(no_enf_base.intensity, no_enf_fined.intensity, rel_tol=1e-9)


def test_policy_analysis_verdict_and_signs():
    weak = penalty_policy_analysis(InsiderEquilibriumInputs(enforcement=0.1))
    assert weak.enforcement_gated
    assert "enforcement" in weak.verdict.lower()

    strong = penalty_policy_analysis(
        InsiderEquilibriumInputs(enforcement=0.7, surveillance_kappa=1.0)
    )
    assert strong.d_intensity_d_criminal < 0.0  # criminal lever bites
    assert strong.d_intensity_d_civil <= 0.0  # civil lever bites (weaker)
    assert math.isfinite(strong.criminal_intensity_floor)
    assert strong.civil_only_min_intensity == 0.0


def test_schedule_accelerates_toward_horizon():
    # Criminal cost present ⇒ as remaining exposure decays, β*(t) rises (accelerates).
    p = InsiderEquilibriumInputs(
        enforcement=0.8, surveillance_kappa=1.0, criminal_penalty=0.05, horizon=1.0
    )
    sched = intensity_schedule(p, steps=8)
    intensities = [row["intensity"] for row in sched]
    assert intensities == sorted(intensities)  # non-decreasing in t
    # The final slice has no remaining exposure ⇒ enforcement→0 ⇒ baseline intensity.
    last = sched[-1]
    assert math.isclose(last["enforcement"], 0.0, abs_tol=1e-12)
    assert math.isclose(last["intensity"], solve_equilibrium(p, enforcement=0.0).intensity, rel_tol=1e-9)
