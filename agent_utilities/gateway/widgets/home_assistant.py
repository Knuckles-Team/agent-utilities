"""Home Assistant widget — smart home device and automation status."""

from __future__ import annotations

import logging

from agent_utilities.gateway.models import ServiceCategory, ServiceConfig, WidgetData, WidgetField
from agent_utilities.gateway.widgets.base import BaseWidget

logger = logging.getLogger(__name__)


class Widget(BaseWidget):
    service_type = "home_assistant"
    display_name = "Home Assistant"
    icon = "home"
    category = ServiceCategory.LIFESTYLE
    description = "Smart home — devices, automations, and entity states"
    env_prefix = "HOME_ASSISTANT"
    default_url = "https://homeassistant.local.example.com"

    def get_fields(self) -> list[WidgetField]:
        return [
            WidgetField(key="entities", label="Entities", format="number"),
            WidgetField(key="lights_on", label="Lights On", format="number", highlight=True),
            WidgetField(key="automations", label="Automations", format="number"),
            WidgetField(key="switches_on", label="Switches", format="number"),
        ]

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        from home_assistant_agent.api_client import HomeAssistantApi

        url = self._resolve_url(config)
        token = self._resolve_token(config)
        client = HomeAssistantApi(base_url=url, token=token)

        try:
            states = client.get_states() or []
            entities = len(states)
            lights_on = sum(1 for s in states if s.get("entity_id", "").startswith("light.") and s.get("state") == "on")
            switches_on = sum(1 for s in states if s.get("entity_id", "").startswith("switch.") and s.get("state") == "on")
            automations = sum(1 for s in states if s.get("entity_id", "").startswith("automation."))
        except Exception as e:
            logger.debug("Home Assistant fetch: %s", e)
            return WidgetData(status="error", error=str(e))

        return WidgetData(
            fields={
                "entities": entities,
                "lights_on": lights_on,
                "automations": automations,
                "switches_on": switches_on,
            },
            status="ok",
        )
