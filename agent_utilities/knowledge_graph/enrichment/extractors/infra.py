"""Infrastructure source extractor (CONCEPT:KG-2.9).

Self-registering enterprise source that maps the physical/runtime substrate into
the knowledge graph: tunnel-manager ``inventory.yaml`` hosts become ``Server``
nodes and Docker services become ``Service`` nodes, wired by ``RUNS_ON`` edges
so blast-radius / placement questions can be reasoned over the same single
``GraphBackend`` interface every other source uses.

The extractor is **pure and deterministic** — it never opens a network socket or
talks to the epistemic-graph daemon. It accepts an already-parsed inventory dict
(or a path to a YAML file it parses with :func:`yaml.safe_load`) plus an optional
list of service descriptors, and returns a uniform :class:`ExtractionBatch`.

Contract (see ``enrichment/registry.py``)::

    def extract(config) -> ExtractionBatch

``config`` may be a dict or any attribute-bearing object with:
  * ``inventory`` — a parsed inventory mapping OR a path string to a YAML file.
  * ``services``  — optional list of dicts ``{"name","image","replicas","node"}``.
"""

from __future__ import annotations

from typing import Any

from ..models import EnrichmentEdge, ExtractionBatch, GraphNode
from ..registry import register_source

CATEGORY = "infra"

# Common Ansible/tunnel-manager keys for a host's address.
_IP_KEYS = ("ansible_host", "ip", "ansible_ssh_host", "host", "address")
# Common keys naming the host a service runs on.
_NODE_KEYS = ("node", "host", "server", "hostname", "placement")


def _get(config: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a dict-like or attribute-bearing config object."""
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _load_inventory(inventory: Any) -> dict[str, Any]:
    """Return a parsed inventory mapping from a dict or a YAML path string."""
    if inventory is None:
        return {}
    if isinstance(inventory, str):
        import yaml

        with open(inventory, encoding="utf-8") as fh:
            inventory = yaml.safe_load(fh) or {}
    if not isinstance(inventory, dict):
        return {}
    return inventory


def _iter_host_maps(inventory: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Normalise tolerant inventory shapes into ``{host_name: host_vars}``.

    Handles:
      * flat ``{host: {ip: ...}}``
      * Ansible ``{all: {hosts: {host: {...}}}}`` (and nested ``children``)
      * a top-level ``hosts:`` / ``tunnels:`` mapping
    """
    hosts: dict[str, dict[str, Any]] = {}

    def _absorb(mapping: Any) -> None:
        if not isinstance(mapping, dict):
            return
        for name, vars_ in mapping.items():
            if not isinstance(name, str):
                continue
            hosts[name] = vars_ if isinstance(vars_, dict) else {}

    def _walk(node: Any) -> None:
        if not isinstance(node, dict):
            return
        if isinstance(node.get("hosts"), dict):
            _absorb(node["hosts"])
        children = node.get("children")
        if isinstance(children, dict):
            for child in children.values():
                _walk(child)

    # Ansible-style top-level group(s) (``all``, or arbitrary group names).
    grouped = False
    for key in ("all",):
        if isinstance(inventory.get(key), dict) and (
            "hosts" in inventory[key] or "children" in inventory[key]
        ):
            _walk(inventory[key])
            grouped = True

    if isinstance(inventory.get("hosts"), dict):
        _absorb(inventory["hosts"])
        grouped = True
    if isinstance(inventory.get("tunnels"), dict):
        _absorb(inventory["tunnels"])
        grouped = True

    if not grouped:
        # Treat the mapping itself as ``{host: vars}``; skip obvious non-host keys.
        for name, vars_ in inventory.items():
            if not isinstance(name, str) or name in ("vars", "children"):
                continue
            if isinstance(vars_, dict):
                hosts[name] = vars_
    return hosts


def _first(vars_: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for k in keys:
        if k in vars_ and vars_[k] not in (None, ""):
            return vars_[k]
    return None


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list | tuple | set):
        return [str(v) for v in value if v not in (None, "")]
    return [str(value)]


def extract(config: Any) -> ExtractionBatch:
    """Build an :class:`ExtractionBatch` of Servers, Services and RUNS_ON edges.

    Pure/deterministic: no network access, no daemon connection (CONCEPT:KG-2.9).
    """
    inventory = _load_inventory(_get(config, "inventory"))
    services = _get(config, "services") or []

    nodes: list[GraphNode] = []
    edges: list[EnrichmentEdge] = []
    known_servers: set[str] = set()

    # --- Servers (from inventory hosts) ---
    for name, vars_ in _iter_host_maps(inventory).items():
        server_id = f"server:{name}"
        known_servers.add(name)
        ip = _first(vars_, _IP_KEYS)
        roles = _as_list(vars_.get("roles") or vars_.get("role"))
        groups = _as_list(vars_.get("groups") or vars_.get("group"))
        nodes.append(
            GraphNode(
                id=server_id,
                type="Server",
                props={
                    "hostname": name,
                    "ip": str(ip) if ip is not None else None,
                    "roles": roles,
                    "groups": groups,
                },
            )
        )

    # --- Services (Docker) + RUNS_ON edges ---
    for svc in services:
        if not isinstance(svc, dict):
            continue
        svc_name = svc.get("name")
        if not svc_name:
            continue
        service_id = f"service:{svc_name}"
        replicas = svc.get("replicas")
        nodes.append(
            GraphNode(
                id=service_id,
                type="Service",
                props={
                    "image": svc.get("image"),
                    "replicas": int(replicas) if replicas is not None else None,
                },
            )
        )
        node_name = _first(svc, _NODE_KEYS)
        if node_name:
            target = f"server:{node_name}"
            edges.append(
                EnrichmentEdge(source=service_id, target=target, rel_type="RUNS_ON")
            )

    return ExtractionBatch(category=CATEGORY, nodes=nodes, edges=edges)


register_source(
    CATEGORY,
    extract,
    description="tunnel-manager inventory + Docker services → KG",
)
