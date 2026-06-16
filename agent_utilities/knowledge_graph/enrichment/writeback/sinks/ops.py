"""Ops/observability write-back sinks (CONCEPT:KG-2.9).

DNS / Caddy / Uptime-Kuma / Kafka / Portainer / LGTM. Standard (reversible) sinks
write once their ``*_ENABLE_WRITE`` flag is set; destructive ones (kafka produce,
portainer deploy) are ``high_stakes`` → routed to the approval queue by the core.
All dry-run-first; tolerant of each connector's method surface.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from ..core import WritebackContext, WritebackResult, register_sink

logger = logging.getLogger(__name__)


def _resolve_client(ops: dict[str, Any], module: str) -> Any | None:
    client = ops.get("client")
    if client is not None:
        return client
    try:
        mod = __import__(f"{module}.auth", fromlist=["get_client"])
        return mod.get_client()
    except Exception:  # noqa: BLE001
        logger.debug("%s write client unavailable", module, exc_info=True)
        return None


class _OpsSink(ABC):
    domain = ""
    enable_flag = ""
    risk_tier = "standard"
    module = ""
    op_name = "create"

    @abstractmethod
    def _apply(self, client: Any, c: dict[str, Any]) -> None:
        """Perform the per-creation write against the resolved connector client."""

    def run(
        self, ctx: WritebackContext, ops: dict[str, Any], *, dry_run: bool
    ) -> WritebackResult:
        result = WritebackResult(target=self.domain)
        client = _resolve_client(ops, self.module)
        if client is None and not dry_run:
            result.skipped += 1
            return result
        for c in ops.get("creations") or []:
            if not c:
                continue
            if dry_run:
                result.proposals.append(
                    {
                        "op": self.op_name,
                        "target": self.domain,
                        "name": c.get("name") or c.get("topic") or c.get("zone"),
                    }
                )
                continue
            try:
                self._apply(client, c)
                result.created += 1
            except Exception:  # noqa: BLE001
                logger.debug("%s write failed", self.domain, exc_info=True)
                result.errors += 1
        return result


class TechnitiumSink(_OpsSink):
    domain = "technitium_dns"
    enable_flag = "TECHNITIUM_DNS_ENABLE_WRITE"
    op_name = "add_record"
    module = "technitium_dns_mcp"

    def _apply(self, client, c):
        client.add_record(
            c["zone"], c.get("name") or c["zone"], c.get("type", "A"), c.get("value")
        )


class CaddySink(_OpsSink):
    domain = "caddy"
    enable_flag = "CADDY_ENABLE_WRITE"
    op_name = "update_config"
    module = "caddy_mcp"

    def _apply(self, client, c):
        client.update_config(c.get("path", ""), c.get("value"))


class UptimeKumaSink(_OpsSink):
    domain = "uptime_kuma"
    enable_flag = "UPTIME_KUMA_ENABLE_WRITE"
    op_name = "add_monitor"
    module = "uptime_kuma_agent"

    def _apply(self, client, c):
        adder = getattr(client, "add_monitor", None) or client.create_monitor
        adder(
            {"name": c.get("name"), "url": c.get("url"), "type": c.get("type", "http")}
        )


class KafkaSink(_OpsSink):
    domain = "kafka"
    enable_flag = "KAFKA_ENABLE_WRITE"
    risk_tier = "high_stakes"  # produce/create_topic mutate the bus
    op_name = "kafka_write"
    module = "kafka_mcp"

    def _apply(self, client, c):
        if c.get("topic"):
            (getattr(client, "produce", None) or client.produce_record)(
                c["topic"], c.get("value")
            )
        else:
            (getattr(client, "create_topic", None) or client.create_topics)(c["name"])


class PortainerSink(_OpsSink):
    domain = "portainer"
    enable_flag = "PORTAINER_ENABLE_WRITE"
    risk_tier = "high_stakes"  # stack deploy
    op_name = "deploy_stack"
    module = "portainer_agent"

    def _apply(self, client, c):
        client.create_stack(
            name=c["name"], **{k: v for k, v in c.items() if k not in ("name", "type")}
        )


class LgtmSink(_OpsSink):
    domain = "lgtm"
    enable_flag = "LGTM_ENABLE_WRITE"
    op_name = "create_dashboard"
    module = "lgtm_mcp"

    def _apply(self, client, c):
        creator = getattr(client, "create_dashboard", None) or client.import_dashboard
        creator(c)


for _sink in (
    TechnitiumSink(),
    CaddySink(),
    UptimeKumaSink(),
    KafkaSink(),
    PortainerSink(),
    LgtmSink(),
):
    register_sink(_sink)
