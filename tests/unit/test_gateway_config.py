"""Tests for agent_utilities.gateway.config — migrated from service-dashboard-core."""

import json

from agent_utilities.gateway.config import ConfigManager
from agent_utilities.gateway.models import (
    DashboardLayout,
    ServiceCategory,
    ServiceConfig,
    ServiceGroup,
)


def test_config_manager_load_save(tmp_path):
    config_file = tmp_path / "services.yaml"
    mgr = ConfigManager(config_path=config_file)

    # Check loading when file doesn't exist (returns empty layout or runs auto-discover)
    layout = mgr.load()
    assert isinstance(layout, DashboardLayout)

    # Save a dummy layout
    services = [
        ServiceConfig(
            id="test-service",
            name="Test Service",
            widget_type="portainer",
            url="http://localhost:9000",
            category=ServiceCategory.INFRASTRUCTURE,
        )
    ]
    group = ServiceGroup(name="Infrastructure", services=services)
    layout = DashboardLayout(groups=[group])

    mgr.save(layout)
    assert config_file.exists()

    # Load it back
    loaded_layout = mgr.load()
    assert len(loaded_layout.groups) == 1
    assert loaded_layout.groups[0].name == "Infrastructure"
    assert loaded_layout.groups[0].services[0].name == "Test Service"
    assert loaded_layout.groups[0].services[0].id == "test-service"


def test_auto_discover(tmp_path, monkeypatch):
    mcp_config = tmp_path / "mcp_config.json"
    mcp_config.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "portainer-agent": {
                        "command": "uv",
                        "args": ["run", "portainer-agent"],
                        "env": {"PORTAINER_URL": "http://portainer.local"},
                    }
                }
            }
        )
    )

    # Patch path functions to use tmp_path
    monkeypatch.setattr(
        "agent_utilities.gateway.config.mcp_config_path", lambda: mcp_config
    )
    monkeypatch.setattr(
        "agent_utilities.gateway.config.services_config_path",
        lambda: tmp_path / "services.yaml",
    )

    mgr = ConfigManager(config_path=tmp_path / "services.yaml")
    layout = mgr.load()

    assert len(layout.groups) == 1
    assert layout.groups[0].name == ServiceCategory.INFRASTRUCTURE.value
    assert layout.groups[0].services[0].id == "portainer-agent"
    assert layout.groups[0].services[0].url == "http://portainer.local"
