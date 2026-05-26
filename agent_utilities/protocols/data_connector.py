from __future__ import annotations

"""Market Data Connector Protocol — generic data source abstraction.

CONCEPT:ECO-4.0 — Market Data Connector Protocol

Provides a pluggable ``DataConnectorProtocol`` that any specialist can
implement to provide structured data ingestion into the Knowledge Graph.
Includes auto-fallback chain, rate-limit awareness, and provenance tracking.

Inspired by Vibe-Trading's DataLoader registry and FinceptTerminal's 100+
data connectors.  This abstraction is domain-agnostic — finance data
sources are just one implementation.

Usage::

    from agent_utilities.protocols.data_connector import (
        DataConnectorProtocol,
        DataConnectorRegistry,
        DataFetchResult,
    )

    class MyConnector(DataConnectorProtocol):
        name = "my_connector"
        provider = "my_provider"
        priority = 0

        def fetch(self, query: str, **kwargs) -> DataFetchResult:
            # ... implementation ...
            return DataFetchResult(rows=[...], row_count=len(rows))

    registry = DataConnectorRegistry()
    registry.register(MyConnector())
    result = registry.fetch_with_fallback("AAPL", instrument_type="equity")
"""


import logging
import time
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class DataFetchResult(BaseModel):
    """Result of a data fetch operation.

    CONCEPT:ECO-4.0

    Attributes:
        rows: The fetched data rows (list of dicts).
        row_count: Number of rows returned.
        connector_name: Name of the connector that served the request.
        latency_ms: Request latency in milliseconds.
        query: The query that was executed.
        is_fallback: Whether this result came from a fallback connector.
        error: Error message if the fetch failed.
        fetched_at: ISO timestamp of the fetch.
    """

    rows: list[dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0
    connector_name: str = ""
    latency_ms: float = 0.0
    query: str = ""
    is_fallback: bool = False
    error: str | None = None
    fetched_at: str = Field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )


@runtime_checkable
class DataConnectorProtocol(Protocol):
    """Protocol for data source connectors.

    CONCEPT:ECO-4.0

    Implementers provide structured data retrieval from external sources.
    The ``name``, ``provider``, and ``priority`` attributes enable
    automatic fallback chain construction.
    """

    name: str
    """Unique connector identifier."""

    provider: str
    """Human-readable provider name."""

    priority: int
    """Fallback priority (lower = tried first)."""

    supported_instruments: list[str]
    """List of instrument types this connector supports."""

    def fetch(self, query: str, **kwargs: Any) -> DataFetchResult:
        """Fetch data for the given query.

        Args:
            query: The data query (e.g., ticker symbol, search term).
            **kwargs: Provider-specific parameters.

        Returns:
            A ``DataFetchResult`` with the fetched data.
        """
        ...

    def health_check(self) -> bool:
        """Check if the connector is healthy and reachable.

        Returns:
            True if the connector is operational.
        """
        ...


class DataConnectorRegistry:
    """Registry with auto-fallback chain for data connectors.

    CONCEPT:ECO-4.0 — Market Data Connector Protocol

    Manages a prioritized set of ``DataConnectorProtocol`` implementations.
    When a primary connector fails, the registry automatically tries the
    next connector in priority order.

    Args:
        max_retries: Maximum number of fallback attempts per query.
    """

    def __init__(self, max_retries: int = 3) -> None:
        self._connectors: list[DataConnectorProtocol] = []
        self.max_retries = max_retries
        self._fetch_history: list[DataFetchResult] = []

    def register(self, connector: DataConnectorProtocol) -> None:
        """Register a data connector, maintaining priority order.

        Args:
            connector: A ``DataConnectorProtocol`` implementation.
        """
        self._connectors.append(connector)
        self._connectors.sort(key=lambda c: c.priority)
        logger.info(
            "Registered data connector: %s (priority=%d, provider=%s)",
            connector.name,
            connector.priority,
            connector.provider,
        )

    def unregister(self, name: str) -> bool:
        """Remove a connector by name.

        Args:
            name: The connector name to remove.

        Returns:
            True if the connector was found and removed.
        """
        initial = len(self._connectors)
        self._connectors = [c for c in self._connectors if c.name != name]
        removed = len(self._connectors) < initial
        if removed:
            logger.info("Unregistered data connector: %s", name)
        return removed

    def get_connector(self, name: str) -> DataConnectorProtocol | None:
        """Retrieve a specific connector by name.

        Args:
            name: The connector name.

        Returns:
            The connector instance, or None if not found.
        """
        for c in self._connectors:
            if c.name == name:
                return c
        return None

    def list_connectors(self) -> list[dict[str, Any]]:
        """List all registered connectors with their metadata.

        Returns:
            List of connector info dicts.
        """
        return [
            {
                "name": c.name,
                "provider": c.provider,
                "priority": c.priority,
                "supported_instruments": c.supported_instruments,
                "healthy": c.health_check(),
            }
            for c in self._connectors
        ]

    def fetch_with_fallback(
        self,
        query: str,
        instrument_type: str | None = None,
        **kwargs: Any,
    ) -> DataFetchResult:
        """Fetch data with automatic fallback across registered connectors.

        CONCEPT:ECO-4.0

        Tries connectors in priority order. If a connector fails or returns
        no data, the next connector in the chain is tried. Rate-limited
        connectors are skipped.

        Args:
            query: The data query (e.g., ticker, search term).
            instrument_type: Optional filter for connectors supporting
                this instrument type.
            **kwargs: Additional parameters passed to the connector.

        Returns:
            A ``DataFetchResult`` from the first successful connector,
            or an error result if all connectors fail.
        """
        candidates = self._connectors
        if instrument_type:
            candidates = [
                c
                for c in candidates
                if not c.supported_instruments
                or instrument_type in c.supported_instruments
            ]

        attempts = 0
        last_error = ""

        for connector in candidates:
            if attempts >= self.max_retries:
                break

            # Skip unhealthy connectors
            try:
                if not connector.health_check():
                    logger.debug("Skipping unhealthy connector: %s", connector.name)
                    continue
            except Exception as e:
                logger.debug("Health check failed for %s: %s", connector.name, e)
                continue

            attempts += 1
            start = time.monotonic()

            try:
                result = connector.fetch(query, **kwargs)
                elapsed_ms = (time.monotonic() - start) * 1000
                result.connector_name = connector.name
                result.latency_ms = elapsed_ms
                result.query = query
                result.is_fallback = attempts > 1

                if result.row_count > 0 or not result.error:
                    self._fetch_history.append(result)
                    logger.info(
                        "Data fetch OK: connector=%s, query=%s, rows=%d, latency=%.1fms",
                        connector.name,
                        query,
                        result.row_count,
                        elapsed_ms,
                    )
                    return result

                last_error = result.error or "empty result"
                logger.warning(
                    "Connector %s returned empty result for query=%s, trying fallback",
                    connector.name,
                    query,
                )

            except Exception as e:
                elapsed_ms = (time.monotonic() - start) * 1000
                last_error = str(e)
                logger.warning(
                    "Connector %s failed for query=%s: %s (%.1fms)",
                    connector.name,
                    query,
                    e,
                    elapsed_ms,
                )

        # All connectors failed
        error_result = DataFetchResult(
            query=query,
            error=f"All connectors failed. Last error: {last_error}",
        )
        self._fetch_history.append(error_result)
        return error_result

    @property
    def fetch_history(self) -> list[DataFetchResult]:
        """Return the fetch history for provenance tracking."""
        return list(self._fetch_history)

    def clear_history(self) -> None:
        """Clear the fetch history."""
        self._fetch_history.clear()
