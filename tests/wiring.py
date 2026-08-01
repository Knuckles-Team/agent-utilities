"""Helpers that make a **wiring test** as cheap to write as a unit test.

CONCEPT:AU-AHE.evaluation.live-path-probe
CONCEPT:AU-AHE.evaluation.surface-contract-test

See ``AGENTS.md`` → *Wire-First — reachable ≠ invoked* for the taxonomy these
helpers implement (unit / **wiring** / **contract** / live-path) and the rules
they enforce. In one line: a **wiring test drives a real entrypoint and proves a
specific downstream seam was reached**, with no mock standing in for the seam.

The reason this module exists is economic. Every capability we shipped dead —
``PolicyEngine``, KV forking, the ``ocel_mode`` ``ChangeEnvelope``,
``AdmissionPolicy.decide``, the reasoning-topology package,
``NeuralRelationPrediction`` — had passing unit tests. The missing test was
always the same shape ("call the real entrypoint, assert the seam ran"), and it
was always more expensive to write than the unit test that proved nothing. These
helpers close that cost gap.

Import from tests as::

    from tests.wiring import assert_surface, observe, past_the_seam

What is here
------------
:func:`observe` / :func:`observe_all`
    Wrap a real attribute with a **pass-through recorder**. The production
    implementation still runs — this is the opposite of ``patch()``. It is how
    you honour *never mock the seam you are validating*: you watch the seam
    instead of replacing it. Refuses outright to observe an attribute that is
    already a ``Mock``.
:func:`past_the_seam`
    Run a live entrypoint that will legitimately fail *downstream* of the seam
    (no engine, no network), while still failing the test if the seam itself is
    what rejected the call.
:func:`assert_surface`
    Exact-set assertion for a public surface (MCP tool list, REST routes,
    capability catalogue) with an actionable diff, plus ``invariant=`` for the
    members that must be present in **every** parameterisation.
:func:`require_module` / :func:`assert_not_faked`
    Kill the two green-signal factories: a test that silently no-ops when an
    optional extra is missing, and a ``MagicMock(spec=[])`` standing in for a
    module that may no longer exist.
:func:`assert_collected_by_pytest`
    A test file CI never collects is worse than no test. Assert a path is inside
    ``pytest.ini``'s ``testpaths``.

What a wiring test **cannot** prove
-----------------------------------
It proves an **edge**: entrypoint → seam, under real construction. It does not
prove the seam is correct (that is the unit test), that the surface is complete
(that is the contract test), or that the thing works against real
infrastructure — real transports, real auth, real engines, real serialisation
(that is live validation, and nothing in this module substitutes for it).
"""

from __future__ import annotations

import configparser
import functools
import inspect
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import NonCallableMock

import pytest

__all__ = [
    "Call",
    "Observation",
    "assert_collected_by_pytest",
    "assert_not_faked",
    "assert_surface",
    "observe",
    "observe_all",
    "past_the_seam",
    "require_module",
    "surface_names",
    "uncollected_test_files",
]


class _Unset:
    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<unset>"


_UNSET = _Unset()

_REPO_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# observe — pass-through recording, never substitution
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Call:
    """One recorded invocation of an observed seam.

    ``result`` stays ``_UNSET`` when the call raised; ``exception`` is then set.
    Both the return value and the exception propagate to the caller unchanged —
    observing a seam never alters what the live path sees.
    """

    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    result: Any = _UNSET
    exception: BaseException | None = None
    _signature: inspect.Signature | None = None

    def arg(self, name: str) -> Any:
        """Look a positional-or-keyword argument up **by name**.

        Saves wiring tests from indexing into ``args`` and breaking the moment
        someone reorders a parameter. Defaults are applied, so an argument the
        live path left implicit still resolves — which is exactly the question
        "did the entrypoint pass a sensible default?" (Wire-First step 2).
        """
        if self._signature is None:  # pragma: no cover - defensive
            raise LookupError(f"no signature captured for argument {name!r}")
        bound = self._signature.bind(*self.args, **self.kwargs)
        bound.apply_defaults()
        try:
            return bound.arguments[name]
        except KeyError as exc:
            raise LookupError(
                f"{name!r} is not a parameter of the observed seam; "
                f"got {sorted(bound.arguments)}"
            ) from exc


@dataclass(slots=True)
class Observation:
    """The record of everything that reached one observed seam."""

    seam: str
    calls: list[Call] = field(default_factory=list)

    @property
    def called(self) -> bool:
        return bool(self.calls)

    @property
    def count(self) -> int:
        return len(self.calls)

    @property
    def last(self) -> Call:
        if not self.calls:
            raise AssertionError(f"{self.seam} was never called")
        return self.calls[-1]

    def assert_called(self, *, times: int | None = None, why: str = "") -> Call:
        """Assert the live path reached this seam; return the first call.

        ``why`` should name the *edge* being proved, e.g.
        ``"Orchestrator.dispatch_task must scan every task through PolicyEngine"``
        — the failure message is what a future reader will use to decide whether
        the wiring regressed or the test is stale.
        """
        suffix = f"\n  edge under test: {why}" if why else ""
        if not self.calls:
            raise AssertionError(
                f"wiring gap: {self.seam} was NOT reached from the live entrypoint. "
                f"The code is importable but never invoked.{suffix}"
            )
        if times is not None and self.count != times:
            raise AssertionError(
                f"{self.seam} was reached {self.count}x, expected {times}x.{suffix}"
            )
        return self.calls[0]

    def assert_not_called(self, *, why: str = "") -> None:
        suffix = f"\n  edge under test: {why}" if why else ""
        if self.calls:
            raise AssertionError(
                f"{self.seam} was reached {self.count}x but should not have been; "
                f"first call args={self.calls[0].args!r} "
                f"kwargs={self.calls[0].kwargs!r}.{suffix}"
            )


def _wrap(func: Callable[..., Any], observation: Observation) -> Callable[..., Any]:
    """Build a recording wrapper that preserves sync/async calling convention."""
    try:
        signature: inspect.Signature | None = inspect.signature(func)
    except (TypeError, ValueError):  # pragma: no cover - builtins etc.
        signature = None

    if inspect.isasyncgenfunction(func) or inspect.isgeneratorfunction(func):
        raise NotImplementedError(
            f"observe() cannot wrap the generator function {observation.seam}: "
            "recording it would change when the body runs. Observe a plain "
            "function on the same path, or assert the generator's side effect."
        )

    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            call = Call(args=args, kwargs=dict(kwargs), _signature=signature)
            observation.calls.append(call)
            try:
                call.result = await func(*args, **kwargs)
            except BaseException as exc:
                call.exception = exc
                raise
            return call.result

        return async_wrapper

    @functools.wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        call = Call(args=args, kwargs=dict(kwargs), _signature=signature)
        observation.calls.append(call)
        try:
            call.result = func(*args, **kwargs)
        except BaseException as exc:
            call.exception = exc
            raise
        return call.result

    return sync_wrapper


@contextmanager
def observe(target: Any, attribute: str) -> Iterator[Observation]:
    """Record every call to ``target.attribute`` **while still running it**.

    ``target`` may be a module, a class, or an instance. ``staticmethod`` and
    ``classmethod`` binding is preserved, and the original attribute is restored
    exactly (including the inherited-vs-own distinction) on exit.

    This is the primitive of a wiring test::

        with observe(PolicyEngine, "evaluate") as scan:
            await Orchestrator(engine=engine).dispatch_task("do the thing")
        scan.assert_called(why="every dispatch is policy-scanned")

    Note what did *not* happen: ``PolicyEngine.evaluate`` was not replaced. The
    real rule set ran, on the real object the real entrypoint constructed. A
    ``patch()`` here would assert only that the test's own mock was called.
    """
    seam = f"{getattr(target, '__name__', type(target).__name__)}.{attribute}"

    resolved = getattr(target, attribute)
    if isinstance(resolved, NonCallableMock) or type(resolved).__module__ == (
        "unittest.mock"
    ):
        raise AssertionError(
            f"refusing to observe {seam}: it is already a {type(resolved).__name__}. "
            "A wiring test must never mock the seam it is validating — the "
            "assertion would only prove the test's own mock was called."
        )

    own = vars(target).get(attribute, _UNSET) if hasattr(target, "__dict__") else _UNSET
    observation = Observation(seam=seam)

    if isinstance(own, staticmethod):
        replacement: Any = staticmethod(_wrap(own.__func__, observation))
    elif isinstance(own, classmethod):
        replacement = classmethod(_wrap(own.__func__, observation))
    else:
        if not callable(resolved):
            raise TypeError(f"{seam} is not callable; nothing to observe")
        replacement = _wrap(resolved, observation)

    setattr(target, attribute, replacement)
    try:
        yield observation
    finally:
        if isinstance(own, _Unset):
            # Attribute was inherited / dynamically resolved — remove our shim so
            # lookup falls back to where it came from.
            try:
                delattr(target, attribute)
            except AttributeError:  # pragma: no cover - defensive
                setattr(target, attribute, resolved)
        else:
            setattr(target, attribute, own)


@contextmanager
def observe_all(
    *seams: tuple[Any, str] | tuple[Any, str, str],
) -> Iterator[dict[str, Observation]]:
    """Observe several seams at once.

    Each entry is ``(target, attribute)`` — keyed by the full seam name — or
    ``(target, attribute, alias)`` to key it by something short. Use when one
    entrypoint must reach a *chain* of seams; proving each edge individually is
    what stops a refactor from quietly short-circuiting the middle of the path::

        with observe_all(
            (Orchestrator, "_scan_task", "scan"),
            (trace_module, "record_execution_trace", "trace"),
        ) as seen:
            await orchestrator.dispatch_task("...")
        seen["scan"].assert_called()
        seen["trace"].assert_called()
    """
    with ExitStack() as stack:
        observations: dict[str, Observation] = {}
        for entry in seams:
            target, attribute = entry[0], entry[1]
            alias = entry[2] if len(entry) == 3 else None
            observation = stack.enter_context(observe(target, attribute))
            observations[alias or observation.seam] = observation
        yield observations


@contextmanager
def past_the_seam(*, must_not_raise: str | Sequence[str] = ()) -> Iterator[None]:
    """Drive a live entrypoint that will fail *downstream* of the seam.

    Real entrypoints usually need infrastructure the test does not have (an
    engine, a socket, credentials). That is not a reason to fall back to mocking
    the entrypoint — the part being validated (entrypoint → seam) happens
    *before* the failure. This swallows the downstream failure but **re-raises**
    if the failure message matches ``must_not_raise``, i.e. if the seam itself
    is what rejected the call::

        with past_the_seam(must_not_raise="Security Alert"):
            await orchestrator.dispatch_task("summarize last night's logs")

    Be honest about the limit: this deliberately tolerates an exception, so pair
    it with an :func:`observe` assertion. On its own it proves nothing.
    """
    forbidden = (
        [must_not_raise] if isinstance(must_not_raise, str) else list(must_not_raise)
    )
    try:
        yield
    except Exception as exc:
        # Tolerating the downstream failure is the entire point of this helper —
        # the edge under test already happened before the exception was raised.
        message = str(exc)
        for needle in forbidden:
            if needle in message:
                raise AssertionError(
                    f"the entrypoint failed AT the seam, not past it: {needle!r} "
                    f"appeared in {type(exc).__name__}: {message}"
                ) from exc


# --------------------------------------------------------------------------- #
# contract / surface pinning
# --------------------------------------------------------------------------- #


def assert_surface(
    actual: Iterable[str],
    expected: Iterable[str],
    *,
    surface: str,
    invariant: Iterable[str] = (),
    parameterisation: str = "",
) -> None:
    """Pin a public surface **exactly**, with an actionable diff.

    ``expected`` is the complete intended set — an exact-set assertion, never a
    superset one. The regression that put 118 MCP tools where ~11 belonged would
    have passed every superset check ever written.

    ``invariant`` names members that must appear in *every* parameterisation of
    the surface (mode, profile, deployment shape). They are checked and reported
    **first and separately**, because "the surface drifted" and "mode-independent
    infrastructure vanished" are different bugs with different fixes.

    ``parameterisation`` labels the current variant (e.g. ``"MCP_TOOL_MODE=intent"``)
    so a parameterised failure names which variant broke.
    """
    actual_set = set(actual)
    expected_set = set(expected)
    invariant_set = set(invariant)
    where = f"{surface}[{parameterisation}]" if parameterisation else surface

    unmet = invariant_set - actual_set
    if unmet:
        raise AssertionError(
            f"{where} dropped invariant members {sorted(unmet)}.\n"
            "These must be present in EVERY parameterisation of this surface — "
            "if they are now mode/profile-dependent, that is the bug."
        )

    if not invariant_set <= expected_set:
        raise AssertionError(
            f"test bug: {where} declares invariant members "
            f"{sorted(invariant_set - expected_set)} that its own `expected` set "
            "omits."
        )

    missing = expected_set - actual_set
    unexpected = actual_set - expected_set
    if missing or unexpected:
        lines = [f"{where} is not the surface we intend to expose."]
        if missing:
            lines.append(f"  MISSING ({len(missing)}): {sorted(missing)}")
        if unexpected:
            preview = sorted(unexpected)
            shown = preview[:20]
            lines.append(
                f"  UNEXPECTED ({len(unexpected)}): {shown}"
                + (" …" if len(preview) > len(shown) else "")
            )
        lines.append(
            "  If this change is intended, update the expected set in the same "
            "commit — that edit is the reviewable record of a public-contract change."
        )
        raise AssertionError("\n".join(lines))


# --------------------------------------------------------------------------- #
# green-signal factories, closed
# --------------------------------------------------------------------------- #


def require_module(name: str, *, extra: str | None = None) -> ModuleType:
    """Import a module, **failing** the test if it is absent — never skipping.

    A test gated behind an optional extra that CI does not install has never run;
    it reports green forever. If a test genuinely may not be runnable everywhere,
    mark it (``@pytest.mark.live`` / ``slow``) so the exclusion is *declared* in
    ``pytest.ini`` and visible, rather than decided silently at import time by
    whatever happens to be installed.
    """
    try:
        return __import__(name, fromlist=["__name__"])
    except ImportError as exc:
        hint = (
            f" Install it (`pip install -e '.[{extra}]'`) or mark this test so its "
            "exclusion is declared in pytest.ini."
            if extra
            else " Install it or declare the exclusion via a pytest marker."
        )
        pytest.fail(
            f"required module {name!r} is not importable, so this test has never "
            f"actually run: {exc}.{hint}"
        )


def assert_not_faked(obj: Any, *, name: str) -> None:
    """Assert ``obj`` is a real object, not a mock standing in for a dead path.

    ``MagicMock(spec=[])`` against a module that may not exist keeps passing long
    after the real module is gone or renamed — it fakes the very thing whose
    existence the test is supposed to establish. Call this on anything you looked
    up dynamically before you assert against it.
    """
    if isinstance(obj, NonCallableMock) or type(obj).__module__ == "unittest.mock":
        raise AssertionError(
            f"{name} is a {type(obj).__name__}, not the real object. This test "
            "would keep passing against a path that no longer exists."
        )


def assert_collected_by_pytest(path: str | Path) -> None:
    """Assert ``path`` sits inside ``pytest.ini``'s ``testpaths``.

    A test file outside ``testpaths`` runs only when someone points pytest at it
    by hand — which is how the fleet's entire default transport stayed broken
    under a green suite. Use it in a meta-test over a directory you care about.
    """
    ini = _REPO_ROOT / "pytest.ini"
    parser = configparser.ConfigParser()
    parser.read(ini, encoding="utf-8")
    testpaths = [
        (_REPO_ROOT / raw).resolve()
        for raw in parser.get("pytest", "testpaths", fallback="").split()
    ]
    if not testpaths:  # pragma: no cover - defensive
        raise AssertionError(f"{ini} declares no testpaths")

    resolved = Path(path).resolve()
    if not any(resolved == root or root in resolved.parents for root in testpaths):
        rel = (
            resolved.relative_to(_REPO_ROOT)
            if _REPO_ROOT in resolved.parents
            else resolved
        )
        raise AssertionError(
            f"{rel} is not under pytest.ini testpaths "
            f"({[str(p.relative_to(_REPO_ROOT)) for p in testpaths]}), so the default "
            "suite never collects it. Move it under a collected path — a test CI "
            "does not run manufactures confidence instead of providing it."
        )


def uncollected_test_files(directory: str | Path) -> list[Path]:
    """Every ``test_*.py`` under ``directory`` that ``testpaths`` does not collect.

    The companion to :func:`assert_collected_by_pytest` for guarding a whole
    tree at once (see ``tests/unit/test_wiring_helpers.py``).
    """
    found: list[Path] = []
    for candidate in sorted(Path(directory).rglob("test_*.py")):
        try:
            assert_collected_by_pytest(candidate)
        except AssertionError:
            found.append(candidate)
    return found


def surface_names(
    mapping: Mapping[str, Any] | Iterable[Any], attr: str = "name"
) -> set[str]:
    """Normalise a surface into a set of names.

    Accepts a ``{name: obj}`` mapping (FastMCP's tool registry), an iterable of
    objects with a ``name``/``path`` attribute (Starlette routes), or an iterable
    of plain strings — so a contract test reads the same regardless of which
    shape the surface under test happens to hand back.
    """
    if isinstance(mapping, Mapping):
        return {str(key) for key in mapping}
    names: set[str] = set()
    for item in mapping:
        if isinstance(item, str):
            names.add(item)
        else:
            names.add(str(getattr(item, attr)))
    return names
