"""Focused contract tests for the AU-to-EG numeric boundary."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import agent_utilities.numeric as numeric


def _lookup(namespace: object, name: str) -> object:
    return getattr(namespace, name)


class _FakeKernel(SimpleNamespace):
    __kernel__ = "eg-numeric"

    def __init__(self) -> None:
        self.choice_calls: list[tuple[int, int, bool, object, int]] = []
        self.permutation_calls: list[tuple[int, int]] = []
        super().__init__(
            LinAlgError=ValueError,
            sum=lambda values: sum(values),
            sqrt=lambda values: [value**0.5 for value in values],
            norm=lambda values: sum(value * value for value in values) ** 0.5,
            where_=lambda condition, left, right: [
                a if flag else b
                for flag, a, b in zip(condition, left, right, strict=True)
            ],
            normal=lambda loc, scale, count, seed: [
                loc + scale * ((seed + index) % 7) / 7.0 for index in range(count)
            ],
            uniform=lambda low, high, count, seed: [
                low + (high - low) * ((seed + index) % 11) / 11.0
                for index in range(count)
            ],
            integers=lambda low, high, count, seed: [
                low + ((seed + index) % (high - low)) for index in range(count)
            ],
            choice_indices=self._choice_indices,
            permutation_indices=self._permutation_indices,
            array=lambda values: ("must-not-be-dispatched", values),
        )

    def _choice_indices(
        self,
        population: int,
        size: int,
        replace: bool,
        weights: object,
        seed: int,
    ) -> list[int]:
        self.choice_calls.append((population, size, replace, weights, seed))
        return list(range(size))

    def _permutation_indices(self, population: int, seed: int) -> list[int]:
        self.permutation_calls.append((population, seed))
        return list(reversed(range(population)))


class _ArrowVector:
    def to_pylist(self) -> list[float]:
        return [1.0, 2.0, 3.0]


def test_native_calls_only_convert_boundary_values() -> None:
    kernel = _FakeKernel()
    xp = numeric._XP(kernel)

    assert xp.sum((1, 2, 3)) == 6
    assert xp.sqrt([1.0, 4.0]) == [1.0, 2.0]
    assert xp.linalg.norm([3.0, 4.0]) == 5.0
    assert xp.linalg.LinAlgError is ValueError
    assert xp.where([True, False], [1, 2], [3, 4]) == [1, 4]
    assert xp.sum(_ArrowVector()) == 6.0


def test_mapping_is_not_treated_as_a_numeric_container() -> None:
    xp = numeric._XP(_FakeKernel())

    with pytest.raises(TypeError, match="builtin list/tuple"):
        xp.sum({"value": 1})


def test_native_result_is_not_wrapped_in_a_python_array_runtime() -> None:
    kernel = _FakeKernel()
    xp = numeric._XP(kernel)

    result = xp.sqrt((1.0, 4.0))

    assert type(result) is list
    assert all(type(item) is float for item in result)


def test_unsupported_surface_fails_closed_even_if_kernel_has_attribute() -> None:
    xp = numeric._XP(_FakeKernel())

    with pytest.raises(numeric.UnsupportedNumericOperationError, match=r"xp\.array"):
        _lookup(xp, "array")
    with pytest.raises(
        numeric.UnsupportedNumericOperationError, match=r"xp\.linalg\.eig"
    ):
        _lookup(xp.linalg, "eig")
    with pytest.raises(
        numeric.UnsupportedNumericOperationError, match=r"xp\.random\.seed"
    ):
        _lookup(xp.random, "seed")


@pytest.mark.parametrize(
    "operation",
    ("array", "asarray", "zeros", "stack", "save", "load"),
)
def test_audited_production_gaps_fail_closed_at_boundary(operation: str) -> None:
    """These are used by capability/finance/KG callers and require an EG contract."""

    xp = numeric._XP(_FakeKernel())

    with pytest.raises(numeric.UnsupportedNumericOperationError):
        _lookup(xp, operation)


def test_fact_deduper_uses_native_bounded_vectors() -> None:
    """A real vector caller uses native scalar/list operations directly."""

    from agent_utilities.knowledge_graph.extraction.fact_extractor import (
        ExtractedFact,
        FactDeduper,
    )

    deduper = FactDeduper(embed_fn=lambda _text: [1.0, 2.0])
    fact = ExtractedFact(subject="s", predicate="p", object="o")

    assert deduper.check(fact) == (False, 0.0)
    duplicate, similarity = deduper.check(fact)
    assert duplicate is True
    assert similarity == pytest.approx(1.0)


def test_seeded_random_adapter_is_deterministic_and_list_bounded() -> None:
    kernel = _FakeKernel()
    first = numeric._XP(kernel).random.default_rng(17)
    second = numeric._XP(kernel).random.default_rng(17)

    assert first.normal(size=(2, 2)) == second.normal(size=(2, 2))
    assert first.integers(0, 5, size=4) == second.integers(0, 5, size=4)
    assert isinstance(first.uniform(size=3), list)


def test_choice_and_shuffle_delegate_to_native_batch_indices() -> None:
    kernel = _FakeKernel()
    rng = numeric._XP(kernel).random.default_rng(17)

    assert rng.choice([10, 20, 30], size=2, replace=False, p=[1.0, 2.0, 3.0]) == [
        10,
        20,
    ]
    values = [1, 2, 3]
    rng.shuffle(values)

    assert kernel.choice_calls[0][:4] == (3, 2, False, [1.0, 2.0, 3.0])
    assert kernel.permutation_calls[0][0] == 3
    assert values == [3, 2, 1]


def test_numeric_boundary_rejects_deep_values_and_oversized_shapes() -> None:
    nested: object = 0
    for _ in range(numeric._MAX_NUMERIC_RANK + 1):
        nested = [nested]
    with pytest.raises(ValueError, match="rank"):
        numeric.to_builtin(nested)

    rng = numeric._XP(_FakeKernel()).random.default_rng(17)
    with pytest.raises(ValueError, match="element limit"):
        rng.normal(size=(1001, 1000))
    assert rng.normal(size=0) == []


def test_tolist_only_producers_are_not_a_general_numeric_boundary() -> None:
    class _ArrayLike:
        def tolist(self):
            return [1.0, 2.0]

    with pytest.raises(TypeError, match="Arrow values exposing to_pylist"):
        numeric.to_builtin(_ArrayLike())


def test_numeric_artifact_round_trip_is_versioned_and_builtin(tmp_path) -> None:
    path = tmp_path / "vectors.json"
    numeric.save_numeric_artifact(path, [[1.0, 2.0], [3.0, 4.0]])
    assert numeric.load_numeric_artifact(path) == [[1.0, 2.0], [3.0, 4.0]]
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload) == {"schema", "values", "digest"}
    payload["values"][0][0] = 99.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="digest"):
        numeric.load_numeric_artifact(path)

    numeric.save_numeric_artifact(path, [[1.0, 2.0], [3.0, 4.0]])
    path.write_text('{"schema":"wrong","values":[]}', encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        numeric.load_numeric_artifact(path)


def test_numeric_artifact_rejects_symlink_targets(tmp_path) -> None:
    target = tmp_path / "vectors.json"
    link = tmp_path / "link.json"
    numeric.save_numeric_artifact(target, [1.0])
    link.symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        numeric.save_numeric_artifact(link, [2.0])
    with pytest.raises(ValueError, match="unavailable"):
        numeric.load_numeric_artifact(link)


def test_production_adapter_has_no_module_table_fallback() -> None:
    source = open(numeric.__file__, encoding="utf-8").read()

    assert "sys.modules" not in source
    assert "import numpy" not in source


def test_audited_consumers_have_no_numpy_import_or_array_fallback() -> None:
    root = Path(__file__).parents[2] / "agent_utilities"
    audited = (
        root / "knowledge_graph/retrieval/capability_index.py",
        root / "knowledge_graph/retrieval/temporal_semantic_id.py",
        root / "knowledge_graph/memory/optimization_engine.py",
        root / "knowledge_graph/core/spectral_navigator.py",
        root / "knowledge_graph/core/world_model.py",
    )
    for path in audited:
        source = path.read_text(encoding="utf-8")
        assert "import numpy" not in source
        assert "sys.modules" not in source
