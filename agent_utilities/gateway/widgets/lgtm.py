"""LGTM widget — Loki/Grafana/Tempo/Mimir observability stack."""

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
    service_type = "lgtm"
    display_name = "LGTM Stack"
    icon = "activity"
    category = ServiceCategory.OBSERVABILITY
    description = "Observability — Grafana, Loki, Tempo, and Mimir metrics stack"
    env_prefix = "LGTM"

    def get_fields(self) -> list[WidgetField]:
        return [
            WidgetField(key="dashboards", label="Dashboards", format="number"),
            WidgetField(
                key="alerts_firing", label="Firing", format="number", highlight=True
            ),
            WidgetField(key="datasources", label="Sources", format="number"),
        ]

    def fetch_data(self, config: ServiceConfig) -> WidgetData:
        url = self._resolve_url(config)
        token = self._resolve_token(config)
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        try:
            with self._http_client(config, timeout=5.0, headers=headers) as client:
                resp = client.get(f"{url}/api/search?type=dash-db")
                dashboards = resp.json() if resp.status_code == 200 else []
                alerts_resp = client.get(f"{url}/api/v1/provisioning/alert-rules")
                alerts = alerts_resp.json() if alerts_resp.status_code == 200 else []
                firing = sum(
                    1
                    for alert in alerts
                    if isinstance(alert, dict) and alert.get("state") == "firing"
                )
                ds_resp = client.get(f"{url}/api/datasources")
                datasources = ds_resp.json() if ds_resp.status_code == 200 else []
        except Exception as e:
            logger.debug("LGTM fetch: %s", type(e).__name__)
            return self._error_data(e)

        return WidgetData(
            fields={
                "dashboards": len(dashboards) if isinstance(dashboards, list) else 0,
                "alerts_firing": firing,
                "datasources": len(datasources) if isinstance(datasources, list) else 0,
            },
            status="ok",
        )
