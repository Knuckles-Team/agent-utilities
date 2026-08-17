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


@pytest.mark.asyncio
async def test_fetch_dashboard_subset_fetches_only_the_named_widgets(
    tmp_path, monkeypatch
):
    """BUG-019 (GOC-29): ``fetch_dashboard_subset`` is what lets a caller that
    already knows its subscribed widget set (the dashboard websocket, once a
    client has sent ``{"type": "subscribe", ...}``) fetch only those widgets
    instead of polling every configured service via ``fetch_all()`` and
    discarding the rest. Proves both that only the requested ids come back,
    AND that an unrequested widget is never fetched at all (a second,
    fetch-type widget that would raise if called is included precisely to
    catch a regression back to "fetch everything, filter after")."""

    from agent_utilities.gateway import api as gateway_api

    class OtherWidget(BaseWidget):
        service_type = "must-not-be-called"
        display_name = "Must Not Be Called"

        def get_fields(self):
            return []

        def fetch_data(self, config: ServiceConfig) -> WidgetData:
            raise AssertionError("an unsubscribed widget must never be fetched")

    registry = Registry()
    registry._widgets["portainer"] = MockWidget
    registry._widgets["must-not-be-called"] = OtherWidget

    config_file = tmp_path / "services.yaml"
    config_manager = ConfigManager(config_path=config_file)
    services = [
        ServiceConfig(
            id="portainer-test",
            name="Portainer Test",
            widget_type="portainer",
            url="http://localhost:9000",
            category=ServiceCategory.INFRASTRUCTURE,
        ),
        ServiceConfig(
            id="other-test",
            name="Other Test",
            widget_type="must-not-be-called",
            url="http://localhost:9001",
            category=ServiceCategory.INFRASTRUCTURE,
        ),
    ]
    group = ServiceGroup(name="Infrastructure", services=services)
    layout = DashboardLayout(groups=[group])
    config_manager.save(layout)

    agg = Aggregator(registry=registry, config_manager=config_manager)
    monkeypatch.setattr(gateway_api, "_aggregator", agg)

    results = await gateway_api.fetch_dashboard_subset({"portainer-test"})

    assert set(results) == {"portainer-test"}
    assert results["portainer-test"].status == "healthy"

    # An explicitly empty subscription fetches nothing.
    empty_results = await gateway_api.fetch_dashboard_subset(set())
    assert empty_results == {}
