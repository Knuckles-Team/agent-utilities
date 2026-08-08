"""Focused tests for the operator embedding-backfill report."""

from scripts.backfill_embeddings import _cost_estimate, _is_zero_progress_failure


def test_cost_estimate_subtracts_only_durable_embedding_progress():
    estimate = _cost_estimate(
        remaining=100,
        report={
            "scanned": 10,
            "embedded": 2,
            "skipped_no_text": 3,
            "deferred_no_text": 3,
            "conflicted": 5,
        },
        elapsed=4.0,
    )

    assert estimate == {
        "seconds_per_embedded": 2.0,
        "remaining_after": 95,
        "eta_seconds": 190.0,
    }


# ---------------------------------------------------------------------------
# D-CDX-101: zero durable progress must fail the process, unconditionally on
# WHY (a backend error and a lost OCC race are both "nothing happened").
# ---------------------------------------------------------------------------


def test_zero_progress_with_real_candidates_and_backend_errors_is_a_failure():
    """The exact production shape: scanned=200, every atomic commit raised
    (errored=200), embedded=0. This MUST be reported as a failure — this is
    the precise report shape the live pod produced under D-CDX-101 (every
    atomic property+ANN commit raised RuntimeError, the script still printed
    a clean report and exited 0)."""
    report = {
        "scanned": 200,
        "embedded": 0,
        "indexed": 0,
        "errored": 200,
        "skipped_no_text": 0,
        "deferred_no_text": 0,
        "conflicted": 0,
    }
    assert _is_zero_progress_failure(report) is True


def test_zero_progress_from_pure_occ_conflict_is_also_a_failure():
    """Zero durable progress from an ordinary lost race is STILL zero
    progress -- the exit code must not silently depend on WHY nothing landed."""
    report = {
        "scanned": 50,
        "embedded": 0,
        "indexed": 0,
        "errored": 0,
        "skipped_no_text": 0,
        "deferred_no_text": 0,
        "conflicted": 50,
    }
    assert _is_zero_progress_failure(report) is True


def test_partial_success_is_not_a_failure():
    """D-CDX-9 / task point 4: a partial success (some embedded, some not) is
    a REAL possible outcome and must stay visible as success, not get folded
    into the zero-progress failure path."""
    report = {
        "scanned": 200,
        "embedded": 37,
        "indexed": 37,
        "errored": 163,
        "skipped_no_text": 0,
        "deferred_no_text": 0,
        "conflicted": 0,
    }
    assert _is_zero_progress_failure(report) is False


def test_all_textless_page_is_not_a_failure():
    """A page with no embeddable candidates at all (every row textless) is
    not a failure -- there was nothing to embed, durable no_text progress is
    the correct, complete outcome."""
    report = {
        "scanned": 10,
        "embedded": 0,
        "indexed": 0,
        "errored": 0,
        "skipped_no_text": 10,
        "deferred_no_text": 10,
        "conflicted": 0,
    }
    assert _is_zero_progress_failure(report) is False


def test_empty_scan_is_not_a_failure():
    """Nothing eligible at all (e.g. the whole graph is already embedded) is
    not a failure."""
    report = {
        "scanned": 0,
        "embedded": 0,
        "indexed": 0,
        "errored": 0,
        "skipped_no_text": 0,
        "deferred_no_text": 0,
        "conflicted": 0,
    }
    assert _is_zero_progress_failure(report) is False
