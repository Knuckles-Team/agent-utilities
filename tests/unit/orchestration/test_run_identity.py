"""Strict current-contract tests for opaque orchestration run identifiers."""

from agent_utilities.knowledge_graph.workflow_store import _automatic_workflow_name
from agent_utilities.orchestration.run_identity import (
    is_run_id,
    new_run_id,
    require_run_id,
)


def test_run_ids_have_128_random_bits_and_do_not_collide() -> None:
    identifiers = {new_run_id() for _ in range(1_024)}

    assert len(identifiers) == 1_024
    assert all(is_run_id(identifier) for identifier in identifiers)
    assert all(len(identifier) == len("run:") + 32 for identifier in identifiers)


def test_short_or_noncanonical_run_ids_are_rejected() -> None:
    invalid = (
        "run:0123abcd",
        "run:" + "A" * 32,
        "run:" + "a" * 31,
        "run:" + "a" * 33,
        "trace:" + "a" * 32,
        "",
        None,
    )

    for value in invalid:
        assert not is_run_id(value)
        try:
            require_run_id(value)
        except ValueError as exc:
            assert str(exc) == "run_id_invalid"
        else:  # pragma: no cover - assertion branch
            raise AssertionError(f"accepted invalid run id: {value!r}")


def test_derived_workflow_identity_uses_the_full_opaque_run() -> None:
    first = "run:01234567" + "a" * 24
    second = "run:01234567" + "b" * 24

    first_name = _automatic_workflow_name(first)
    second_name = _automatic_workflow_name(second)

    assert first_name.startswith("auto:pref_run_")
    assert second_name.startswith("auto:pref_run_")
    assert first_name != second_name
    assert first not in first_name
    assert second not in second_name
