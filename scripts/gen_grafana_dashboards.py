#!/usr/bin/env python3
"""Generate Agent OS Grafana dashboards as JSON (provisioned as code).

Builds the panels programmatically so the JSON is always valid and consistent,
then writes one file per dashboard into the LGTM Grafana provisioning dir
(CONCEPT:OS-5.23). Re-run to update; pairs with
``services/lgtm/grafana/provisioning/dashboards/dashboards.yml``.

Dashboards:
  * mcp-fleet-overview — every MCP/stack: up, probe, request/error rate, p95
    latency, plus per-stack container CPU/mem from cAdvisor. The "all Portainer
    stacks" view.
  * mcp-per-service    — templated by $stack: tool rate/latency/errors, in-flight,
    container CPU/mem, and a Loki logs panel.
  * host-infra         — node-exporter CPU/mem/disk per host.

Usage: gen_grafana_dashboards.py [--out DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROM = {"type": "prometheus", "uid": "prometheus"}
LOKI = {"type": "loki", "uid": "loki"}

_DEFAULT_OUT = Path(
    "/home/apps/workspace/services/lgtm/grafana/provisioning/dashboards/json"
)


def _target(expr: str, legend: str = "", instant: bool = False) -> dict:
    return {
        "datasource": PROM,
        "expr": expr,
        "legendFormat": legend,
        "instant": instant,
        "refId": "A",
    }


def _panel(
    pid: int,
    title: str,
    ptype: str,
    targets: list[dict],
    gp: dict,
    unit: str = "short",
    datasource: dict | None = None,
    options: dict | None = None,
) -> dict:
    p = {
        "id": pid,
        "title": title,
        "type": ptype,
        "datasource": datasource or PROM,
        "gridPos": gp,
        "targets": targets,
        "fieldConfig": {"defaults": {"unit": unit}, "overrides": []},
    }
    if options:
        p["options"] = options
    return p


def _gp(x: int, y: int, w: int, h: int) -> dict:
    return {"x": x, "y": y, "w": w, "h": h}


def _dashboard(uid: str, title: str, panels: list[dict], templating=None) -> dict:
    return {
        "uid": uid,
        "title": title,
        "tags": ["agent-os", "mcp"],
        "timezone": "browser",
        "schemaVersion": 39,
        "version": 1,
        "refresh": "30s",
        "time": {"from": "now-6h", "to": "now"},
        "templating": {"list": templating or []},
        "panels": panels,
    }


def fleet_overview() -> dict:
    panels = [
        _panel(
            1, "MCPs scrapeable", "stat",
            [_target('count(up{job="mcp-fleet"} == 1)', "up", instant=True)],
            _gp(0, 0, 6, 4),
        ),
        _panel(
            2, "Health probes passing", "stat",
            [_target('count(probe_success{job="blackbox-mcp"} == 1)', "ok", instant=True)],
            _gp(6, 0, 6, 4),
        ),
        _panel(
            3, "Tool calls/s (fleet)", "stat",
            [_target("sum(rate(agent_utilities_mcp_tool_calls_total[5m]))", "calls/s")],
            _gp(12, 0, 6, 4), unit="reqps",
        ),
        _panel(
            4, "Tool errors/s (fleet)", "stat",
            [_target(
                'sum(rate(agent_utilities_mcp_tool_calls_total{outcome="error"}[5m]))',
                "errors/s",
            )],
            _gp(18, 0, 6, 4), unit="reqps",
        ),
        _panel(
            5, "Per-MCP scrape + probe status", "table",
            [
                _target('up{job="mcp-fleet"}', "", instant=True),
                _target('probe_success{job="blackbox-mcp"}', "", instant=True),
            ],
            _gp(0, 4, 24, 7),
        ),
        _panel(
            6, "Request rate by stack", "timeseries",
            [_target(
                "sum by (stack) (rate(agent_utilities_mcp_tool_calls_total[5m]))",
                "{{stack}}",
            )],
            _gp(0, 11, 12, 8), unit="reqps",
        ),
        _panel(
            7, "Error rate by stack", "timeseries",
            [_target(
                'sum by (stack) (rate(agent_utilities_mcp_tool_calls_total{outcome="error"}[5m]))',
                "{{stack}}",
            )],
            _gp(12, 11, 12, 8), unit="reqps",
        ),
        _panel(
            8, "p95 tool latency by stack", "timeseries",
            [_target(
                "histogram_quantile(0.95, sum by (stack, le) "
                "(rate(agent_utilities_mcp_tool_duration_seconds_bucket[5m])))",
                "{{stack}}",
            )],
            _gp(0, 19, 12, 8), unit="s",
        ),
        _panel(
            9, "Container CPU by stack", "timeseries",
            [_target(
                'sum by (container_label_com_docker_stack_namespace) '
                '(rate(container_cpu_usage_seconds_total{container_label_com_docker_stack_namespace!=""}[5m]))',
                "{{container_label_com_docker_stack_namespace}}",
            )],
            _gp(12, 19, 12, 8), unit="percentunit",
        ),
        _panel(
            10, "Container memory by stack", "timeseries",
            [_target(
                'sum by (container_label_com_docker_stack_namespace) '
                '(container_memory_working_set_bytes{container_label_com_docker_stack_namespace!=""})',
                "{{container_label_com_docker_stack_namespace}}",
            )],
            _gp(0, 27, 12, 8), unit="bytes",
        ),
    ]
    return _dashboard("agentos-mcp-fleet", "MCP Fleet Overview", panels)


def per_service() -> dict:
    stack_var = {
        "name": "stack",
        "type": "query",
        "datasource": PROM,
        "query": 'label_values(up{job="mcp-fleet"}, stack)',
        "refresh": 2,
        "includeAll": False,
        "multi": False,
        "label": "MCP",
    }
    panels = [
        _panel(
            1, "Tool calls/s", "timeseries",
            [_target(
                'sum by (tool) (rate(agent_utilities_mcp_tool_calls_total{stack="$stack"}[5m]))',
                "{{tool}}",
            )],
            _gp(0, 0, 12, 8), unit="reqps",
        ),
        _panel(
            2, "Tool error rate", "timeseries",
            [_target(
                'sum by (tool) (rate(agent_utilities_mcp_tool_calls_total{stack="$stack",outcome="error"}[5m]))',
                "{{tool}}",
            )],
            _gp(12, 0, 12, 8), unit="reqps",
        ),
        _panel(
            3, "p95 tool latency", "timeseries",
            [_target(
                "histogram_quantile(0.95, sum by (tool, le) "
                '(rate(agent_utilities_mcp_tool_duration_seconds_bucket{stack="$stack"}[5m])))',
                "{{tool}}",
            )],
            _gp(0, 8, 12, 8), unit="s",
        ),
        _panel(
            4, "Tools in flight", "timeseries",
            [_target('agent_utilities_mcp_tool_in_flight{stack="$stack"}', "in flight")],
            _gp(12, 8, 12, 8),
        ),
        _panel(
            5, "Container CPU", "timeseries",
            [_target(
                'sum(rate(container_cpu_usage_seconds_total{container_label_com_docker_stack_namespace="$stack"}[5m]))',
                "cpu",
            )],
            _gp(0, 16, 8, 7), unit="percentunit",
        ),
        _panel(
            6, "Container memory", "timeseries",
            [_target(
                'sum(container_memory_working_set_bytes{container_label_com_docker_stack_namespace="$stack"})',
                "mem",
            )],
            _gp(8, 16, 8, 7), unit="bytes",
        ),
        _panel(
            7, "Logs", "logs",
            [{"datasource": LOKI, "expr": '{stack="$stack"}', "refId": "A"}],
            _gp(16, 16, 8, 7), datasource=LOKI,
            options={"showTime": True, "wrapLogMessage": True},
        ),
    ]
    return _dashboard("agentos-mcp-service", "MCP Per-Service", panels, [stack_var])


def host_infra() -> dict:
    panels = [
        _panel(
            1, "CPU by host", "timeseries",
            [_target(
                "100 - (avg by (instance) (rate(node_cpu_seconds_total{mode=\"idle\"}[5m])) * 100)",
                "{{instance}}",
            )],
            _gp(0, 0, 12, 8), unit="percent",
        ),
        _panel(
            2, "Memory used by host", "timeseries",
            [_target(
                "(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100",
                "{{instance}}",
            )],
            _gp(12, 0, 12, 8), unit="percent",
        ),
        _panel(
            3, "Disk free % by mount", "timeseries",
            [_target(
                '(node_filesystem_avail_bytes{fstype!~"tmpfs|overlay"} / '
                'node_filesystem_size_bytes{fstype!~"tmpfs|overlay"}) * 100',
                "{{instance}} {{mountpoint}}",
            )],
            _gp(0, 8, 24, 8), unit="percent",
        ),
    ]
    return _dashboard("agentos-host-infra", "Host & Infra", panels)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name, builder in (
        ("mcp-fleet-overview", fleet_overview),
        ("mcp-per-service", per_service),
        ("host-infra", host_infra),
    ):
        path = args.out / f"{name}.json"
        path.write_text(json.dumps(builder(), indent=2) + "\n")
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
