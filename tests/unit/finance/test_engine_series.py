"""Warning-clean regression coverage for engine-native finance time series."""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

from agent_utilities.domains.finance import engine_series


class _FakeTimeSeries:
    """Enough of the native time-series surface to prove the accelerated route."""

    def __init__(self) -> None:
        self._points: dict[str, dict[int, float]] = {}
        self.gap_fill_steps: list[int] = []

    def append(self, series_id: str, points: list[tuple[int, list[float]]]) -> None:
        self._points[series_id] = {timestamp: values[0] for timestamp, values in points}

    def gap_fill(
        self, series_id: str, start_ns: int, end_ns: int, step_ns: int
    ) -> list[tuple[int, float, int]]:
        self.gap_fill_steps.append(step_ns)
        points = self._points[series_id]
        rows: list[tuple[int, float, int]] = []
        last_value: float | None = None
        for timestamp in range(start_ns, end_ns, step_ns):
            if timestamp in points:
                last_value = points[timestamp]
            if last_value is not None:
                rows.append((timestamp, last_value, 0))
        return rows


class _FakeClient:
    def __init__(self) -> None:
        self.timeseries = _FakeTimeSeries()


def test_utc_gap_fill_reuses_the_already_canonical_series() -> None:
    """The common UTC path avoids rebuilding a large index before gap filling."""
    index = pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC")
    series = pd.Series([10.0, 20.0], index=index, name="close")

    assert engine_series._utc_series(series) is series


@pytest.mark.parametrize(
    "step",
    [
        pytest.param(1_000_000_000, id="bare-nanoseconds"),
        pytest.param("1s", id="frequency-string"),
        pytest.param(pd.to_timedelta(1, unit="s"), id="pandas-timedelta"),
    ],
)
def test_gap_fill_normalizes_supported_steps_without_generic_timedelta_warnings(
    monkeypatch: pytest.MonkeyPatch, step: str | int | pd.Timedelta
) -> None:
    """All supported forms retain exact LOCF parity on engine and fallback routes."""
    index = pd.date_range("2026-01-01", periods=4, freq="s", tz="UTC")
    series = pd.Series([10.0, 20.0, 40.0], index=index[[0, 1, 3]], name="close")
    expected = pd.Series([10.0, 20.0, 20.0, 40.0], index=index, name="close")
    client = _FakeClient()

    # Treat every warning as an error. The accelerated path must still execute;
    # otherwise its broad fallback handler could hide a parser regression.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        accelerated = engine_series.gap_fill_series(series, step, client=client)

    pd.testing.assert_series_equal(accelerated, expected, check_freq=False)
    assert client.timeseries.gap_fill_steps == [1_000_000_000]

    monkeypatch.setattr(engine_series, "_client", lambda: None)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fallback = engine_series.gap_fill_series(series, step)

    pd.testing.assert_series_equal(fallback, expected, check_freq=False)


@pytest.mark.parametrize(
    ("step", "step_ns", "canonical_frequency", "periods"),
    [
        pytest.param("1h", 3_600_000_000_000, "h", 3, id="canonical-hour"),
        pytest.param("1H", 3_600_000_000_000, "h", 3, id="legacy-hour"),
        pytest.param("1min", 60_000_000_000, "min", 121, id="canonical-minute"),
        pytest.param("1T", 60_000_000_000, "min", 121, id="legacy-minute"),
    ],
)
def test_gap_fill_normalizes_legacy_interval_aliases_without_warnings(
    monkeypatch: pytest.MonkeyPatch,
    step: str,
    step_ns: int,
    canonical_frequency: str,
    periods: int,
) -> None:
    """Canonical and legacy hour/minute strings retain exact warning-clean cadence."""
    index = pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC")
    series = pd.Series([10.0, 30.0], index=index[[0, 2]], name="close")
    expected_index = pd.date_range(
        index.min(), periods=periods, freq=canonical_frequency, tz="UTC"
    )
    expected = pd.Series(
        [10.0] * (periods - 1) + [30.0], index=expected_index, name="close"
    )
    client = _FakeClient()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        accelerated = engine_series.gap_fill_series(series, step, client=client)

    monkeypatch.setattr(engine_series, "_client", lambda: None)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fallback = engine_series.gap_fill_series(series, step)

    pd.testing.assert_series_equal(accelerated, expected, check_freq=False)
    pd.testing.assert_series_equal(fallback, expected, check_freq=False)
    assert client.timeseries.gap_fill_steps == [step_ns]


@pytest.mark.parametrize(
    ("index", "values", "expected_index", "expected_values"),
    [
        pytest.param(
            pd.DatetimeIndex(
                ["2026-03-08 00:00", "2026-03-08 03:00"],
                tz="America/New_York",
            ),
            [10.0, 30.0],
            pd.date_range("2026-03-08 05:00", periods=3, freq="h", tz="UTC"),
            [10.0, 10.0, 30.0],
            id="spring-forward-gap",
        ),
        pytest.param(
            pd.date_range(
                "2026-11-01 00:00", periods=4, freq="h", tz="America/New_York"
            )[[0, 2, 3]],
            [10.0, 30.0, 40.0],
            pd.date_range("2026-11-01 04:00", periods=4, freq="h", tz="UTC"),
            [10.0, 10.0, 30.0, 40.0],
            id="fall-back-fold",
        ),
    ],
)
def test_gap_fill_normalizes_new_york_dst_to_utc_with_engine_fallback_parity(
    monkeypatch: pytest.MonkeyPatch,
    index: pd.DatetimeIndex,
    values: list[float],
    expected_index: pd.DatetimeIndex,
    expected_values: list[float],
) -> None:
    """Both routes use UTC instants, including DST gaps and repeated local hours."""
    series = pd.Series(values, index=index, name="close")
    expected = pd.Series(expected_values, index=expected_index, name="close")
    client = _FakeClient()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        accelerated = engine_series.gap_fill_series(series, "1h", client=client)

    monkeypatch.setattr(engine_series, "_client", lambda: None)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fallback = engine_series.gap_fill_series(series, "1h")

    pd.testing.assert_series_equal(accelerated, expected, check_freq=False)
    pd.testing.assert_series_equal(fallback, expected, check_freq=False)


@pytest.mark.parametrize(
    "step",
    [
        pytest.param("", id="empty-string"),
        pytest.param("nonsense", id="malformed-string"),
        pytest.param("0H", id="zero-legacy-hour"),
        pytest.param("0min", id="zero-canonical-minute"),
        pytest.param(0, id="zero-nanoseconds"),
        pytest.param("-1h", id="negative-hour"),
        pytest.param(-1, id="negative-nanoseconds"),
    ],
)
def test_gap_fill_rejects_malformed_or_non_positive_intervals_before_any_work(
    monkeypatch: pytest.MonkeyPatch, step: str | int
) -> None:
    """D-CDX-96: every invalid interval form raises one typed error, pre-engine.

    Regression for the split where ``"nonsense"``/``""`` already raised but
    ``"0H"`` slipped through ``_normalize_step`` and only failed later as a
    ``ZeroDivisionError`` out of zero-frequency grid construction. Every form
    here must raise ``InvalidIntervalError`` before the engine client is even
    consulted (proven by a client whose methods explode if touched) and before
    the pandas fallback route runs.
    """
    index = pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC")
    series = pd.Series([10.0, 20.0], index=index, name="close")

    class _ExplodingClient:
        """Any access proves the invalid interval reached the engine path."""

        def __getattr__(self, name: str):  # noqa: D105
            raise AssertionError(
                f"engine client.{name} must never be touched for an invalid interval"
            )

    with pytest.raises(engine_series.InvalidIntervalError):
        engine_series.gap_fill_series(series, step, client=_ExplodingClient())

    monkeypatch.setattr(engine_series, "_client", lambda: None)
    with pytest.raises(engine_series.InvalidIntervalError):
        engine_series.gap_fill_series(series, step)


def test_gap_fill_rejects_invalid_interval_even_for_an_empty_series() -> None:
    """The interval is validated at the API boundary regardless of the data."""
    series = pd.Series(
        [], index=pd.DatetimeIndex([], tz="UTC"), name="close", dtype=float
    )

    with pytest.raises(engine_series.InvalidIntervalError):
        engine_series.gap_fill_series(series, "0H")
