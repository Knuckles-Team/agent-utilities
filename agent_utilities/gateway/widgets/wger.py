"""Wger widget — fitness and workout tracking."""

from __future__ import annotations

import logging

from agent_utilities.gateway.models import (
    ServiceCategory,
    ServiceConfig,
    WidgetData,
    WidgetField,
)
from agent_utilities.gateway.widgets.base import BaseWidget

logger = logging.getLogger(__name__)


class Widget(BaseWidget):
    service_type = "wger"
    display_name = "Wger"
    icon = "dumbbell"
    category = ServiceCategory.LIFESTYLE
    description = "Fitness tracker — workouts, exercises, and body measurements"
    env_prefix = "WGER"
    default_url = "https://wger.local.example.com"

    def get_fields(self) -> list[WidgetField]:
        return [
            WidgetField(key="workouts", label="Workouts", format="number"),
            WidgetField(key="exercises", label="Exercises", format="number"),
            WidgetField(key="routines", label="Routines", format="number"),
        ]

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        from wger_agent.api_client import WgerApi

        url = self._resolve_url(config)
        token = self._resolve_token(config)
        client = WgerApi(base_url=url, token=token)

        try:
            workouts = client.get_workouts() or []
            exercises = client.get_exercises() or []
        except Exception as e:
            logger.debug("Wger fetch: %s", e)
            return WidgetData(status="error", error=str(e))

        return WidgetData(
            fields={
                "workouts": len(workouts) if isinstance(workouts, list) else 0,
                "exercises": len(exercises) if isinstance(exercises, list) else 0,
                "routines": 0,
            },
            status="ok",
        )
