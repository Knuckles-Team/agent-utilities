#!/usr/bin/python
from __future__ import annotations

"""Hardware inventory collector → KG infra ontology (CONCEPT:KG-2.9).

The real implementation behind the previously doc-only ``hardware-profile-sweep``
/ ``host-resource-sampler`` skills. Pulls host hardware/resource/network profiles
from the tunnel-manager ``tm_system`` MCP (CPU/mem/disk/GPU/NIC) and persists them
into the existing KG infra ontology (``HardwareNode``/``GPUAccelerator``/
``NetworkSubnet`` + ``RUNS_ON``), idempotent via the ingest manifest (re-collecting
an unchanged host is a no-op).

The ``tm_system`` caller is injected (any ``(host, action) -> dict`` callable) so
the collector is unit-testable offline; production passes a thin wrapper over the
tunnel-manager MCP client.
"""

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..ingestion.manifest import DeltaManifest

_CATEGORY = "infra"
TmCaller = Callable[[str, str], dict]


@dataclass
class HostProfile:
    """A collected hardware/resource profile for one host."""

    host_id: str
    cpu_cores: float = 0.0
    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0
    disk_total_gb: float = 0.0
    disk_used_pct: float = 0.0
    gpus: list[dict[str, Any]] = field(default_factory=list)
    subnets: list[str] = field(default_factory=list)
    os: str = ""
    role: str = ""
    region: str = ""
    security_findings: int = 0

    def content_hash(self) -> str:
        """Stable hash of the profile for idempotent re-persist."""
        payload = json.dumps(self.__dict__, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _f(d: dict, *keys: str, default: float = 0.0) -> float:
    for k in keys:
        if k in d:
            try:
                return float(d[k])
            except (TypeError, ValueError):
                continue
    return default


def parse_tm_system(host_id: str, info: dict[str, Any]) -> HostProfile:
    """Map a ``tm_system get_info``/``network_topology`` payload to a HostProfile.

    Tolerant of the various shapes tunnel-manager returns (nested ``cpu``/
    ``memory``/``disk`` dicts or flattened keys).
    """
    cpu = info.get("cpu", info)
    mem = info.get("memory", info)
    disk = info.get("disk", info)
    gpus = info.get("gpus") or info.get("accelerators") or []
    if isinstance(gpus, dict):
        gpus = [gpus]
    subnets = info.get("subnets") or []
    nets = info.get("network", {})
    if isinstance(nets, dict):
        subnets = subnets or nets.get("subnets", [])
    return HostProfile(
        host_id=host_id,
        cpu_cores=_f(cpu, "cores", "cpu_cores", "count", default=0.0),
        ram_total_gb=_f(mem, "total_gb", "ram_total_gb", "total", default=0.0),
        ram_available_gb=_f(
            mem, "available_gb", "ram_available_gb", "available", default=0.0
        ),
        disk_total_gb=_f(disk, "total_gb", "disk_total_gb", "total", default=0.0),
        disk_used_pct=_f(disk, "used_pct", "disk_used_pct", "percent", default=0.0),
        gpus=[g for g in gpus if isinstance(g, dict)],
        subnets=[str(s) for s in subnets if s],
        os=str(info.get("os", "") or ""),
        role=str(info.get("role", info.get("swarm_role", "")) or ""),
        region=str(info.get("region", "") or ""),
        security_findings=int(_f(info, "security_findings", default=0)),
    )


class InfraInventoryCollector:
    """Collect host profiles via tm_system and persist them into the KG."""

    def __init__(
        self,
        engine: Any,
        *,
        tm_caller: TmCaller | None = None,
        graph_name: str = "default",
        manifest: DeltaManifest | None = None,
    ) -> None:
        self.engine = engine
        self.tm_caller = tm_caller
        self.graph_name = graph_name
        self.manifest = manifest or DeltaManifest(
            backend=getattr(engine, "backend", None)
        )

    def collect_host(self, host_id: str) -> HostProfile | None:
        """Collect one host's profile via the injected tm_system caller."""
        if self.tm_caller is None:
            return None
        try:
            info = self.tm_caller(host_id, "get_info") or {}
        except Exception:  # noqa: BLE001
            return None
        if not isinstance(info, dict):
            return None
        return parse_tm_system(host_id, info)

    def collect_fleet(self, host_ids: list[str]) -> list[HostProfile]:
        """Collect profiles for many hosts (skips ones that fail)."""
        out: list[HostProfile] = []
        for h in host_ids:
            p = self.collect_host(h)
            if p is not None:
                out.append(p)
        return out

    def persist(self, profiles: list[HostProfile]) -> int:
        """Persist profiles into the KG infra ontology, idempotent via manifest.

        Returns the count of hosts actually (re)written (unchanged hosts skip).
        """
        add = getattr(self.engine, "add_node", None)
        link = getattr(self.engine, "link_nodes", None)
        if not callable(add):
            return 0
        known = self.manifest.load_for_graph(self.graph_name, _CATEGORY)
        written = 0
        for p in profiles:
            chash = p.content_hash()
            if known.get(p.host_id) == chash:
                continue  # unchanged → skip
            try:
                add(
                    p.host_id,
                    "hardwarenode",
                    properties={
                        "cpu_cores": p.cpu_cores,
                        "ram_total_gb": p.ram_total_gb,
                        "ram_available_gb": p.ram_available_gb,
                        "disk_total_gb": p.disk_total_gb,
                        "disk_used_pct": p.disk_used_pct,
                        "gpu": bool(p.gpus),
                        "gpu_count": len(p.gpus),
                        "os": p.os,
                        "swarm_role": p.role,
                        "region": p.region,
                        "security_findings": p.security_findings,
                        "concept": "KG-2.9",
                    },
                )
            except Exception:  # noqa: BLE001
                continue
            for i, gpu in enumerate(p.gpus):
                gid = f"{p.host_id}:gpu:{i}"
                try:
                    add(
                        gid,
                        "gpu_accelerator",
                        properties={
                            "gpu_model": gpu.get("model", gpu.get("name", "")),
                            "vram_gb": _f(gpu, "vram_gb", "memory_gb", default=0.0),
                            "concept": "KG-2.9",
                        },
                    )
                    if callable(link):
                        link(gid, p.host_id, "RUNS_ON", properties={"_rel": "RUNS_ON"})
                except Exception:  # noqa: BLE001
                    pass
            for subnet in p.subnets:
                sid = f"subnet:{subnet}"
                try:
                    add(sid, "networksubnet", properties={"cidr": subnet})
                    if callable(link):
                        link(
                            p.host_id, sid, "HAS_INTERFACE",
                            properties={"_rel": "HAS_INTERFACE"},
                        )
                except Exception:  # noqa: BLE001
                    pass
            self.manifest.record(self.graph_name, _CATEGORY, p.host_id, chash)
            written += 1
        return written


def collect_and_persist(
    engine: Any,
    host_ids: list[str],
    *,
    tm_caller: TmCaller | None = None,
    graph_name: str = "default",
) -> dict[str, int]:
    """Collect + persist a fleet's hardware inventory; return a summary."""
    collector = InfraInventoryCollector(
        engine, tm_caller=tm_caller, graph_name=graph_name
    )
    profiles = collector.collect_fleet(host_ids)
    written = collector.persist(profiles)
    return {"collected": len(profiles), "persisted": written}


__all__ = [
    "HostProfile",
    "InfraInventoryCollector",
    "parse_tm_system",
    "collect_and_persist",
]
