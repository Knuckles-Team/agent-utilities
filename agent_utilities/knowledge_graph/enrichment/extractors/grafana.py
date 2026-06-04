"""Grafana / observability source extractor (CONCEPT:KG-2.9).

Self-registering extractor that maps Grafana observability objects —
dashboards, panels, alert rules, and datasources — into the uniform
``ExtractionBatch`` shape (typed ``GraphNode`` + ``EnrichmentEdge``) so they
persist through the one generic writer with no edits to any shared hub file.

The Grafana client is **injected** (duck-typed) via ``config["client"]`` and is
expected to expose list-returning methods:

* ``client.dashboards()`` → list of dicts: ``uid``, ``title``, ``panels`` (each
  a dict with ``id``/``panel_id``, ``title``, ``targets``).
* ``client.alert_rules()`` → list of dicts: ``uid``, ``title``, ``condition``,
  ``labels``.
* ``client.datasources()`` → list of dicts: ``uid``, ``name``, ``type``.

All field access is tolerant of missing keys — this module performs **no
network calls** itself.
"""

from __future__ import annotations

import re
from typing import Any

from ..models import EnrichmentEdge, ExtractionBatch, GraphNode
from ..registry import register_source

CATEGORY = "grafana"

# Words that look like services in a title; tune-able heuristic.
_SERVICE_HINT = re.compile(r"[A-Za-z][A-Za-z0-9._-]{1,}")


def _get(record: Any, key: str, default: Any = None) -> Any:
    """Tolerant field access for dict records (or attr-style objects)."""
    if isinstance(record, dict):
        return record.get(key, default)
    return getattr(record, key, default)


def _call(client: Any, name: str) -> list:
    """Call a client method if present, returning a list (tolerant)."""
    method = getattr(client, name, None)
    if not callable(method):
        return []
    result = method()
    return list(result) if result else []


def _scalar(value: Any) -> str | None:
    """Normalise a value to a non-empty scalar string, else ``None``."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _service_from_labels(labels: Any) -> str | None:
    """Pull a service name from a labels mapping (service/app/job keys)."""
    if not isinstance(labels, dict):
        return None
    for key in ("service", "app", "job", "service_name"):
        name = _scalar(labels.get(key))
        if name:
            return name
    return None


def _service_from_title(title: Any) -> str | None:
    """Heuristically pull a service token from a free-text title.

    Looks for an explicit ``service=<name>`` / ``service: <name>`` marker first;
    only that explicit form is treated as a derivable service reference to avoid
    spurious MONITORS edges from arbitrary words.
    """
    text = _scalar(title)
    if not text:
        return None
    match = re.search(r"service\s*[:=]\s*([A-Za-z0-9._-]+)", text, re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def extract(config: Any) -> ExtractionBatch:
    """Extract Grafana observability objects into a uniform ``ExtractionBatch``.

    ``config`` is a dict (or attr-style object) carrying an injected ``client``.
    Dashboards become ``Dashboard`` nodes; their panels become ``Panel`` nodes
    linked ``PART_OF`` their dashboard. Alerts and datasources become ``Alert``
    and ``DataSource`` nodes. Panels/alerts that reference a service (via labels
    or an explicit ``service=`` marker in the title) emit a ``MONITORS`` edge to
    ``service:<name>``.
    """
    client = _get(config, "client")
    nodes: list[GraphNode] = []
    edges: list[EnrichmentEdge] = []
    if client is None:
        return ExtractionBatch(category=CATEGORY, nodes=nodes, edges=edges)

    # --- Datasources -------------------------------------------------------
    for ds in _call(client, "datasources"):
        uid = _scalar(_get(ds, "uid")) or _scalar(_get(ds, "name"))
        if not uid:
            continue
        nodes.append(
            GraphNode(
                id=f"datasource:{uid}",
                type="DataSource",
                props={
                    "name": _get(ds, "name"),
                    "ds_type": _scalar(_get(ds, "type")),
                },
            )
        )

    # --- Dashboards + Panels ----------------------------------------------
    for dash in _call(client, "dashboards"):
        dash_uid = _scalar(_get(dash, "uid"))
        if not dash_uid:
            continue
        dash_id = f"dashboard:{dash_uid}"
        nodes.append(
            GraphNode(
                id=dash_id,
                type="Dashboard",
                props={"title": _get(dash, "title")},
            )
        )
        for panel in _get(dash, "panels", []) or []:
            panel_id = _scalar(_get(panel, "id")) or _scalar(_get(panel, "panel_id"))
            if not panel_id:
                continue
            node_id = f"panel:{dash_uid}:{panel_id}"
            targets = _get(panel, "targets", []) or []
            nodes.append(
                GraphNode(
                    id=node_id,
                    type="Panel",
                    props={
                        "title": _get(panel, "title"),
                        "targets": list(targets),
                    },
                )
            )
            edges.append(
                EnrichmentEdge(source=node_id, target=dash_id, rel_type="PART_OF")
            )
            service = _service_from_labels(
                _get(panel, "labels")
            ) or _service_from_title(_get(panel, "title"))
            if service:
                edges.append(
                    EnrichmentEdge(
                        source=node_id,
                        target=f"service:{service}",
                        rel_type="MONITORS",
                    )
                )

    # --- Alert rules -------------------------------------------------------
    for alert in _call(client, "alert_rules"):
        uid = _scalar(_get(alert, "uid"))
        if not uid:
            continue
        node_id = f"alert:{uid}"
        nodes.append(
            GraphNode(
                id=node_id,
                type="Alert",
                props={
                    "title": _get(alert, "title"),
                    "condition": _scalar(_get(alert, "condition")),
                },
            )
        )
        service = _service_from_labels(_get(alert, "labels")) or _service_from_title(
            _get(alert, "title")
        )
        if service:
            edges.append(
                EnrichmentEdge(
                    source=node_id,
                    target=f"service:{service}",
                    rel_type="MONITORS",
                )
            )

    return ExtractionBatch(category=CATEGORY, nodes=nodes, edges=edges)


register_source(
    CATEGORY,
    extract,
    description="Grafana dashboards/alerts/datasources → KG",
)
