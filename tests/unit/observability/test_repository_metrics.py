"""RMDD-19 — repository job metric catalog: bounded cardinality + no identifying labels."""

from __future__ import annotations

from agent_utilities.observability.repository_metrics import (
    FORBIDDEN_LABEL_NAMES,
    REPOSITORY_METRIC_LABELS,
    REPOSITORY_METRIC_REGISTRY,
)


def test_every_registered_metric_has_a_declared_label_set() -> None:
    assert set(REPOSITORY_METRIC_REGISTRY) == set(REPOSITORY_METRIC_LABELS)
    assert REPOSITORY_METRIC_REGISTRY  # non-empty: the catalog is populated


def test_no_metric_label_identifies_a_specific_job_path_or_credential() -> None:
    for name, labels in REPOSITORY_METRIC_LABELS.items():
        offending = set(labels) & FORBIDDEN_LABEL_NAMES
        assert not offending, f"{name} declares forbidden label(s): {offending}"


def test_label_sets_are_bounded_small_enums_not_open_cardinality() -> None:
    # Every repository metric label set is small (<=2 dimensions) -- a
    # concrete, checkable stand-in for "bounded cardinality" that would fail
    # loudly if a future edit added a wide-cardinality dimension by mistake.
    for name, labels in REPOSITORY_METRIC_LABELS.items():
        assert 0 < len(labels) <= 2, f"{name} has an unexpectedly wide label set: {labels}"


def test_instruments_are_callable_with_their_declared_labels() -> None:
    # Smoke-test every instrument end to end with its declared label
    # dimensions -- both the real prometheus_client path and the no-op
    # fallback expose the same .labels(...).inc()/.set()/.observe() surface.
    sample_values = {
        "repo": "repository-manager",
        "priority_class": "default",
        "resource_class": "light-check",
        "host_alias": "build-host-a",
        "outcome": "hit",
        "stage": "feedback",
        "retry_class": "transient",
        "drift_kind": "target_moved",
    }
    for name, instrument in REPOSITORY_METRIC_REGISTRY.items():
        labels = REPOSITORY_METRIC_LABELS[name]
        bound = instrument.labels(**{label: sample_values[label] for label in labels})
        for method, arg in (("inc", None), ("set", 1.0), ("observe", 1.0)):
            fn = getattr(bound, method, None)
            if fn is None:
                continue
            fn() if arg is None else fn(arg)
