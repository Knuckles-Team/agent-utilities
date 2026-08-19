"""Native sequence boundary tests for the finance regime detector."""

from __future__ import annotations

import pytest

pytest.importorskip("epistemic_graph.numeric")

from agent_utilities.domains.finance.regime_detector import RegimeDetector


class _Engine:
    def __init__(self) -> None:
        self.nodes: list[dict[str, object]] = []

    def add_node(
        self,
        node_id: str,
        node_type: str,
        properties: dict[str, object] | None = None,
        ephemeral: bool = False,
        *,
        session: object | None = None,
    ) -> None:
        self.nodes.append(
            {"node_id": node_id, "node_type": node_type, "properties": properties}
        )


class _ArrowLikeColumn(list[float]):
    def to_pylist(self) -> list[float]:
        return list(self)


class _ArrowLikeTable:
    empty = False

    def __init__(self, values: list[float]) -> None:
        self._values = values

    def __getitem__(self, name: str) -> _ArrowLikeColumn:
        if name != "Close":
            raise KeyError(name)
        return _ArrowLikeColumn(self._values)


def test_detect_close_prices_batches_native_statistics_and_persists() -> None:
    engine = _Engine()
    detector = RegimeDetector(engine)

    result = detector.detect_close_prices(
        [100.0 + index * 0.5 for index in range(60)], ticker="AAPL"
    )

    assert result == "bull_market"
    assert engine.nodes[0]["node_id"] == "Regime_AAPL"


def test_tabular_adapter_accepts_arrow_protocol_without_dataframe_runtime() -> None:
    detector = RegimeDetector()
    table = _ArrowLikeTable([100.0 + index * 0.5 for index in range(60)])

    assert detector.detect_regime(table, ticker="AAPL") == "bull_market"


def test_short_or_zero_denominator_window_fails_closed() -> None:
    detector = RegimeDetector()

    assert detector.detect_close_prices([1.0] * 49) == "unknown"
    assert detector.detect_close_prices([1.0] * 39 + [0.0] + [1.0] * 20) == "unknown"
