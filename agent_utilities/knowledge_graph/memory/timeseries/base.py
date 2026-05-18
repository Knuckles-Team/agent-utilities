import abc
from dataclasses import dataclass
from datetime import datetime


@dataclass
class TimeSeriesDataPoint:
    symbol: str
    timestamp: datetime
    metrics: dict[str, float]
    tags: dict[str, str] | None = None


class TimeSeriesBackend(abc.ABC):
    """Abstract base class for time-series memory backends."""

    @abc.abstractmethod
    def initialize(self) -> None:
        """Initialize the backend (e.g., create tables/buckets)."""
        pass

    @abc.abstractmethod
    def insert(self, points: list[TimeSeriesDataPoint]) -> None:
        """Insert a batch of time-series points."""
        pass

    @abc.abstractmethod
    def query(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        tags: dict[str, str] | None = None,
    ) -> list[TimeSeriesDataPoint]:
        """Query time-series points for a specific symbol within a time range."""
        pass

    @abc.abstractmethod
    def close(self) -> None:
        """Close the backend connection."""
        pass
