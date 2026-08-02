"""Focused tests for the operator embedding-backfill report."""

from scripts.backfill_embeddings import _cost_estimate


def test_cost_estimate_subtracts_only_durable_embedding_progress():
    estimate = _cost_estimate(
        remaining=100,
        report={"scanned": 10, "embedded": 2, "skipped_no_text": 8},
        elapsed=4.0,
    )

    assert estimate == {
        "seconds_per_embedded": 2.0,
        "remaining_after": 98,
        "eta_seconds": 196.0,
    }
