"""Unit tests for ``tests/wiring.py`` — the wiring-test helper kit.

CONCEPT:AU-AHE.evaluation.live-path-probe

These are honestly *unit* tests under the taxonomy in ``AGENTS.md``: they prove
the helpers behave. The proof that the helpers are actually *used* is the
exemplar wiring tests that import them (``tests/unit/**/test_*_wiring.py``).

The one exception is :func:`test_the_wiring_helpers_live_under_a_collected_path`,
which is a contract test over this repository's own test layout — the guard that
stops the "test file outside ``testpaths``" failure mode from recurring in the
very tree that is supposed to prevent it.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tests.wiring import (
    assert_collected_by_pytest,
    assert_not_faked,
    assert_surface,
    observe,
    observe_all,
    past_the_seam,
    require_module,
    surface_names,
    uncollected_test_files,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


class _Seam:
    """A stand-in for production code under observation."""

    def __init__(self) -> None:
        self.ran = 0

    def scan(self, text: str, *, strict: bool = False) -> str:
        self.ran += 1
        if text == "boom":
            raise ValueError("Security Alert: blocked")
        return f"scanned:{text}:{strict}"

    async def ascan(self, text: str) -> str:
        self.ran += 1
        return f"async:{text}"

    @staticmethod
    def helper(value: int) -> int:
        return value * 2

    @classmethod
    def build(cls) -> _Seam:
        return cls()


class TestObserve:
    def test_the_real_implementation_still_runs(self) -> None:
        """Pass-through, not substitution — this is what separates it from patch()."""
        seam = _Seam()

        with observe(_Seam, "scan") as seen:
            result = seam.scan("hello")

        assert result == "scanned:hello:False"
        assert seam.ran == 1
        assert seen.called
        assert seen.count == 1

    def test_records_arguments_by_name_including_defaults(self) -> None:
        seam = _Seam()

        with observe(_Seam, "scan") as seen:
            seam.scan("hello")

        call = seen.assert_called(why="self-test")
        assert call.arg("text") == "hello"
        # The default the caller left implicit is still visible — that is the
        # "did the live path default the integration ON?" question.
        assert call.arg("strict") is False

    def test_unknown_argument_name_raises_lookup_error(self) -> None:
        seam = _Seam()
        with observe(_Seam, "scan") as seen:
            seam.scan("hello")
        with pytest.raises(LookupError, match="nope"):
            seen.last.arg("nope")

    def test_exceptions_propagate_and_are_recorded(self) -> None:
        seam = _Seam()

        with observe(_Seam, "scan") as seen:
            with pytest.raises(ValueError):
                seam.scan("boom")

        assert isinstance(seen.last.exception, ValueError)
        # No result was ever produced, so the sentinel survives — a raising seam
        # is recorded as "reached", which is what the wiring question asks.
        assert repr(seen.last.result) == "<unset>"

    async def test_async_seams_are_observed_without_changing_awaitability(self) -> None:
        seam = _Seam()

        with observe(_Seam, "ascan") as seen:
            assert await seam.ascan("x") == "async:x"

        assert seen.count == 1

    def test_staticmethod_binding_is_preserved(self) -> None:
        with observe(_Seam, "helper") as seen:
            assert _Seam.helper(3) == 6
        assert seen.count == 1
        assert _Seam.helper(4) == 8  # restored intact

    def test_classmethod_binding_is_preserved(self) -> None:
        with observe(_Seam, "build") as seen:
            assert isinstance(_Seam.build(), _Seam)
        assert seen.count == 1
        assert isinstance(_Seam.build(), _Seam)

    def test_attribute_is_restored_exactly(self) -> None:
        original = _Seam.scan
        with observe(_Seam, "scan"):
            assert _Seam.scan is not original
        assert _Seam.scan is original

    def test_instance_attribute_observation_does_not_leak_to_the_class(self) -> None:
        seam = _Seam()
        with observe(seam, "scan") as seen:
            seam.scan("a")
            _Seam().scan("b")  # a different instance is untouched
        assert seen.count == 1
        assert "scan" not in vars(seam)

    def test_refuses_to_observe_something_already_mocked(self) -> None:
        """The rule "never mock the seam you are validating", enforced in code."""
        target = MagicMock()

        with pytest.raises(AssertionError, match="already a MagicMock"):
            with observe(target, "anything"):
                pass

    def test_generator_seams_are_refused_rather_than_silently_mistimed(self) -> None:
        class _Gen:
            def stream(self):
                yield 1

        with pytest.raises(NotImplementedError, match="generator function"):
            with observe(_Gen, "stream"):
                pass

    def test_assert_called_names_the_wiring_gap(self) -> None:
        with observe(_Seam, "scan") as seen:
            pass
        with pytest.raises(AssertionError, match="wiring gap"):
            seen.assert_called(why="the edge")

    def test_assert_called_checks_exact_call_count(self) -> None:
        seam = _Seam()
        with observe(_Seam, "scan") as seen:
            seam.scan("a")
            seam.scan("b")
        seen.assert_called(times=2)
        with pytest.raises(AssertionError, match="reached 2x, expected 1x"):
            seen.assert_called(times=1)

    def test_assert_not_called(self) -> None:
        seam = _Seam()
        with observe(_Seam, "scan") as seen:
            pass
        seen.assert_not_called()
        with observe(_Seam, "scan") as seen:
            seam.scan("a")
        with pytest.raises(AssertionError, match="should not have been"):
            seen.assert_not_called()

    def test_observe_all_keys_by_owner_and_attribute(self) -> None:
        seam = _Seam()
        with observe_all((_Seam, "scan"), (_Seam, "helper")) as seen:
            seam.scan("a")
            _Seam.helper(1)

        assert set(seen) == {"_Seam.scan", "_Seam.helper"}
        seen["_Seam.scan"].assert_called()
        seen["_Seam.helper"].assert_called()

    def test_observe_all_accepts_a_short_alias(self) -> None:
        """Module seams key on the full dotted path; alias keeps tests readable."""
        seam = _Seam()
        with observe_all((_Seam, "scan", "scan"), (_Seam, "helper", "double")) as seen:
            seam.scan("a")
            _Seam.helper(1)

        assert set(seen) == {"scan", "double"}
        seen["scan"].assert_called()


class TestPastTheSeam:
    def test_downstream_failure_is_tolerated(self) -> None:
        with past_the_seam(must_not_raise="Security Alert"):
            raise RuntimeError("no engine configured")

    def test_failure_at_the_seam_is_re_raised_as_an_assertion(self) -> None:
        with pytest.raises(AssertionError, match="failed AT the seam"):
            with past_the_seam(must_not_raise="Security Alert"):
                raise ValueError("Security Alert: blocked")

    def test_accepts_several_forbidden_messages(self) -> None:
        with pytest.raises(AssertionError, match="failed AT the seam"):
            with past_the_seam(must_not_raise=["Security Alert", "PolicyViolation"]):
                raise ValueError("PolicyViolation: nope")


class TestAssertSurface:
    def test_exact_match_passes(self) -> None:
        assert_surface({"a", "b"}, {"a", "b"}, surface="demo")

    def test_a_superset_fails_because_leakage_is_the_regression(self) -> None:
        with pytest.raises(AssertionError, match="UNEXPECTED"):
            assert_surface({"a", "b", "leaked"}, {"a", "b"}, surface="demo")

    def test_missing_members_are_reported_separately(self) -> None:
        with pytest.raises(AssertionError, match=r"MISSING \(1\)"):
            assert_surface({"a"}, {"a", "b"}, surface="demo")

    def test_invariant_violation_is_reported_before_ordinary_drift(self) -> None:
        """Losing mode-independent infrastructure is a different bug from drift."""
        with pytest.raises(AssertionError, match="dropped invariant members"):
            assert_surface(
                {"a"},
                {"a", "meta"},
                surface="demo",
                invariant={"meta"},
                parameterisation="mode=intent",
            )

    def test_parameterisation_is_named_in_the_failure(self) -> None:
        with pytest.raises(AssertionError, match=r"demo\[mode=intent\]"):
            assert_surface(
                {"a"}, {"a", "b"}, surface="demo", parameterisation="mode=intent"
            )

    def test_an_invariant_missing_from_expected_is_flagged_as_a_test_bug(self) -> None:
        with pytest.raises(AssertionError, match="test bug"):
            assert_surface({"a", "meta"}, {"a"}, surface="demo", invariant={"meta"})

    def test_surface_names_normalises_the_common_shapes(self) -> None:
        class _Route:
            def __init__(self, name: str) -> None:
                self.name = name

        assert surface_names({"a": 1, "b": 2}) == {"a", "b"}
        assert surface_names(["a", "b"]) == {"a", "b"}
        assert surface_names([_Route("a"), _Route("b")]) == {"a", "b"}


class TestGreenSignalFactories:
    def test_require_module_returns_a_real_module(self) -> None:
        assert require_module("json").dumps({}) == "{}"

    def test_require_module_fails_it_never_skips(self) -> None:
        """A missing optional extra must fail loudly, not vanish into a skip.

        ``pytest.fail.Exception`` — never ``Skipped``: the whole point is that
        the run is red, not quietly one test shorter.
        """
        with pytest.raises(pytest.fail.Exception) as exc_info:
            require_module("agent_utilities_definitely_not_installed", extra="ml")
        assert "has never" in str(exc_info.value)
        assert not isinstance(exc_info.value, pytest.skip.Exception)

    def test_assert_not_faked_rejects_mocks(self) -> None:
        assert_not_faked(object(), name="real")
        with pytest.raises(AssertionError, match="not the real object"):
            assert_not_faked(MagicMock(spec=[]), name="fake_module")


class TestCollectionGuard:
    def test_this_file_is_collected(self) -> None:
        assert_collected_by_pytest(__file__)

    def test_a_path_outside_testpaths_is_rejected(self) -> None:
        with pytest.raises(AssertionError, match="not under pytest.ini testpaths"):
            assert_collected_by_pytest(REPO_ROOT / "scripts" / "test_nowhere.py")

    def test_the_wiring_helpers_live_under_a_collected_path(self) -> None:
        """Contract test over our own layout, scoped to the trees this lane owns.

        Deliberately NOT the whole ``tests/`` tree: ~230 legacy files sit at
        ``tests/`` root outside ``testpaths`` and ratcheting that backlog belongs
        to the static-gate lane's ``check_wiring.py --check-test-collection``.
        This pins only that *new* wiring/contract tests land where CI runs them.
        """
        orphans = uncollected_test_files(REPO_ROOT / "tests" / "unit")
        assert orphans == []
