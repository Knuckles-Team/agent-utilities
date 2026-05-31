"""Tests for agent_utilities.gateway.aggregator — migrated from service-dashboard-core."""

import pytest

from agent_utilities.gateway.aggregator import Aggregator
from agent_utilities.gateway.config import ConfigManager
from agent_utilities.gateway.models import (
    DashboardLayout,
    ServiceCategory,
    ServiceConfig,
    ServiceGroup,
    WidgetData,
)
from agent_utilities.gateway.registry import Registry
from agent_utilities.gateway.widgets.base import BaseWidget


class MockWidget(BaseWidget):
    service_type = "portainer"
    display_name = "Portainer"

    def get_fields(self):
        return []

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        return WidgetData(
            status="healthy",
            fields={"cpu": 12.5, "memory": 45.2},
            raw={"summary": "Portainer healthy with 2 running containers."},
        )


@pytest.mark.asyncio
async def test_aggregator_fetch_all(tmp_path):
    # Setup registry with MockWidget
    registry = Registry()
    registry._widgets["portainer"] = MockWidget

    # Setup ConfigManager with one service
    config_file = tmp_path / "services.yaml"
    config_manager = ConfigManager(config_path=config_file)

    services = [
        ServiceConfig(
            id="portainer-test",
            name="Portainer Test",
            widget_type="portainer",
            url="http://localhost:9000",
            category=ServiceCategory.INFRASTRUCTURE,
        )
    ]
    group = ServiceGroup(name="Infrastructure", services=services)
    layout = DashboardLayout(groups=[group])
    config_manager.save(layout)

    # Run aggregator
    agg = Aggregator(registry=registry, config_manager=config_manager)
    results = await agg.fetch_all()

    assert "portainer-test" in results
    assert results["portainer-test"].status == "healthy"
    assert results["portainer-test"].fields["cpu"] == 12.5
    assert results["portainer-test"].raw is not None
    assert (
        results["portainer-test"].raw.get("summary")
        == "Portainer healthy with 2 running containers."
    )


@pytest.mark.asyncio
async def test_aggregator_fetch_one(tmp_path):
    registry = Registry()
    registry._widgets["portainer"] = MockWidget

    config_file = tmp_path / "services.yaml"
    config_manager = ConfigManager(config_path=config_file)

    services = [
        ServiceConfig(
            id="portainer-test",
            name="Portainer Test",
            widget_type="portainer",
            url="http://localhost:9000",
            category=ServiceCategory.INFRASTRUCTURE,
        )
    ]
    group = ServiceGroup(name="Infrastructure", services=services)
    layout = DashboardLayout(groups=[group])
    config_manager.save(layout)

    agg = Aggregator(registry=registry, config_manager=config_manager)
    result = await agg.fetch_one("portainer-test")
    assert result.status == "healthy"

    # Fetch non-existent
    result_fail = await agg.fetch_one("non-existent")
    assert result_fail.status == "error"
    assert result_fail.error is not None
    assert "not found" in result_fail.error
