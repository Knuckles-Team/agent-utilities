"""Source-anchored AU scale-claim register and planted-drift fixture."""

from pathlib import Path

from scripts.check_scale_claims import check_claim_register


ROOT = Path(__file__).resolve().parents[2]


def test_scale_claim_register_is_source_anchored():
    assert check_claim_register(ROOT / "docs/scaling/scale_claims.md") == []


def test_planted_scale_claim_drift_is_rejected():
    errors = check_claim_register(ROOT / "tests/fixtures/scale_claims_planted_drift.md")
    assert any("required fragment is absent" in error for error in errors)

