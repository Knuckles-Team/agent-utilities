"""Async Data Aggregator — parallel fetching across all active widgets.

CONCEPT:GW-1.0 — Gateway Service Dashboard

Designed for all three frontends:
  - WebUI: Called by FastAPI endpoint, returns JSON
  - TUI: Called directly via ``async for data in aggregator.stream()``
  - GUI: Called via ``asyncio.run(aggregator.fetch_all())``
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor

from agent_utilities.gateway.config import ConfigManager
from agent_utilities.gateway.models import (
    DashboardLayout,
    ServiceConfig,
    WidgetData,
)
from agent_utilities.gateway.registry import Registry, get_registry

logger = logging.getLogger(__name__)


class Aggregator:
    """Fetches data from all configured service widgets in parallel.

    Uses a thread pool since most agent-package API clients are synchronous
    (requests-based). Each widget.fetch_data() call runs in its own thread.
    """

    def __init__(
        self,
        registry: Registry | None = None,
        config_manager: ConfigManager | None = None,
        max_workers: int = 10,
    ):
        self.registry = registry or get_registry()
        self.config_manager = config_manager or ConfigManager()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._cache: dict[str, tuple[WidgetData, float]] = {}
        self._cache_ttl: float = 10.0  # seconds

    async def fetch_all(self) -> dict[str, WidgetData]:
        """Fetch data from all configured services concurrently.

        Returns:
            Dict mapping service_id to WidgetData.
        """
        layout = self.config_manager.load()
        services = [
            svc for group in layout.groups for svc in group.services if svc.visible
        ]

        results: dict[str, WidgetData] = {}
        tasks = []

        for svc in services:
            # Check cache
            cached = self._get_cached(svc.id)
            if cached:
                results[svc.id] = cached
                continue
            tasks.append(self._fetch_one(svc))

        if tasks:
            fetched = await asyncio.gather(*tasks, return_exceptions=True)
            for svc, result in zip(
                [s for s in services if s.id not in results], fetched, strict=False
            ):
                if isinstance(result, BaseException):
                    logger.error("Widget %s failed: %s", svc.id, result)
                    results[svc.id] = WidgetData(status="error", error=str(result))
                else:
                    results[svc.id] = result
                    self._set_cached(svc.id, result)

        return results

    async def fetch_one(self, service_id: str) -> WidgetData:
        """Fetch data for a single service by ID."""
        services = self.config_manager.get_all_services()
        svc = next((s for s in services if s.id == service_id), None)
        if not svc:
            return WidgetData(status="error", error=f"Service '{service_id}' not found")
        return await self._fetch_one(svc)

    async def _fetch_one(self, config: ServiceConfig) -> WidgetData:
        """Fetch data for a single service using the thread pool."""
        widget = self.registry.get_widget(config.widget_type)
        if not widget:
            return WidgetData(
                status="error",
                error=f"No widget registered for type '{config.widget_type}'",
            )

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, widget._safe_fetch, config)

    async def stream(
        self, interval: float = 30.0
    ) -> AsyncIterator[dict[str, WidgetData]]:
        """Continuously stream dashboard data at the given interval.

        Usage (TUI/GUI)::

            async for data in aggregator.stream(interval=10):
                update_display(data)
        """
        while True:
            data = await self.fetch_all()
            yield data
            await asyncio.sleep(interval)

    async def health_check(self) -> dict[str, bool]:
        """Quick health check across all configured services."""
        layout = self.config_manager.load()
        services = [svc for group in layout.groups for svc in group.services]

        results: dict[str, bool] = {}
        loop = asyncio.get_event_loop()

        tasks = []
        for svc in services:
            widget = self.registry.get_widget(svc.widget_type)
            if widget:
                tasks.append(
                    loop.run_in_executor(self._executor, widget.check_health, svc)
                )
            else:
                results[svc.id] = False

        if tasks:
            health_results = await asyncio.gather(*tasks, return_exceptions=True)
            svc_with_widgets = [
                s for s in services if self.registry.get_widget(s.widget_type)
            ]
            for svc, result in zip(svc_with_widgets, health_results, strict=False):
                results[svc.id] = (
                    bool(result) if not isinstance(result, Exception) else False
                )

        return results

    def get_layout(self) -> DashboardLayout:
        """Get the current dashboard layout."""
        return self.config_manager.load()

    def save_layout(self, layout: DashboardLayout) -> None:
        """Save dashboard layout to YAML."""
        self.config_manager.save(layout)

    def _get_cached(self, service_id: str) -> WidgetData | None:
        """Get cached data if still fresh."""
        if service_id in self._cache:
            data, ts = self._cache[service_id]
            if time.time() - ts < self._cache_ttl:
                return data
        return None

    def _set_cached(self, service_id: str, data: WidgetData) -> None:
        """Cache widget data with timestamp."""
        self._cache[service_id] = (data, time.time())
