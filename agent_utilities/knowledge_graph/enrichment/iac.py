"""Infrastructure-as-Code extraction → Resource nodes (CONCEPT:KG-2.103).

A service's code is only half its definition; the other half is the IaC that
deploys it — the Dockerfile that builds its image, the Kubernetes manifests that
run it, the Terraform that provisions its cloud resources. We parse those into
``Resource`` nodes and link them to the deployed ``Service`` (``provisions``), so
the one ontology-driven KG spans code → infra → live topology in a single graph.
"""

from __future__ import annotations

import re
from pathlib import Path

from .models import EnrichmentEdge, GraphNode

_DOCKERFILE_RE = re.compile(r"(^|/)(dockerfile)(\.|$)", re.IGNORECASE)
_FROM_RE = re.compile(r"^\s*FROM\s+(\S+)", re.IGNORECASE | re.MULTILINE)
_EXPOSE_RE = re.compile(r"^\s*EXPOSE\s+(\d+)", re.IGNORECASE | re.MULTILINE)
# Terraform `resource "aws_s3_bucket" "logs" {`.
_TF_RESOURCE_RE = re.compile(r'resource\s+"([^"]+)"\s+"([^"]+)"')
# K8s `kind:` / `metadata:\n  name:` (best-effort line scan, no YAML dep).
_K8S_KIND_RE = re.compile(r"^kind:\s*(\S+)", re.MULTILINE)
_K8S_NAME_RE = re.compile(r"^\s+name:\s*(\S+)", re.MULTILINE)

# IaC file kinds → how to read them.
_IAC_EXTENSIONS = {".tf", ".yaml", ".yml"}


def discover_iac_files(root: str | Path) -> list[Path]:
    """Find IaC files under ``root``: Dockerfiles, Terraform, and K8s/Kustomize
    YAML (skipping vendored/build dirs)."""
    root = Path(root)
    skip = {
        ".git",
        "node_modules",
        ".venv",
        "venv",
        "vendor",
        "target",
        "dist",
        "build",
    }
    if root.is_file():
        return [root] if _is_iac_file(root) else []
    out: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file() or any(part in skip for part in p.parts):
            continue
        if _is_iac_file(p):
            out.append(p)
    return sorted(out)


def _is_iac_file(p: Path) -> bool:
    return bool(_DOCKERFILE_RE.search(p.name)) or p.suffix.lower() in _IAC_EXTENSIONS


def extract_iac(
    files: list[tuple[str, str]],
) -> tuple[list[GraphNode], list[EnrichmentEdge]]:
    """Parse IaC files into ``Resource`` nodes (CONCEPT:KG-2.103).

    ``files`` is ``[(path, text), ...]``. Returns the Resource nodes; edges is
    reserved for intra-IaC links (empty for now)."""
    nodes: dict[str, GraphNode] = {}
    edges: list[EnrichmentEdge] = []

    def add(rid: str, kind: str, name: str, path: str, **extra: str) -> None:
        nodes.setdefault(
            rid,
            GraphNode(
                id=rid,
                type="Resource",
                props={"kind": kind, "name": name, "file_path": path, **extra},
            ),
        )

    for path, text in files:
        base = Path(path).name
        if _DOCKERFILE_RE.search(base):
            fm = _FROM_RE.findall(text)
            ports = _EXPOSE_RE.findall(text)
            add(
                f"resource:container_image:{path}",
                "container_image",
                base,
                path,
                base_image=(fm[-1] if fm else ""),
                ports=",".join(ports),
            )
        elif path.endswith(".tf"):
            for rtype, rname in _TF_RESOURCE_RE.findall(text):
                add(f"resource:{rtype}:{rname}", rtype, rname, path)
        elif path.endswith((".yaml", ".yml")):
            # One manifest per `kind:` (multi-doc files have several).
            for kind in _K8S_KIND_RE.findall(text):
                nm = _K8S_NAME_RE.search(text)
                name = nm.group(1) if nm else base
                add(f"resource:{kind.lower()}:{name}", kind.lower(), name, path)

    return list(nodes.values()), edges


def link_resources_to_service(
    resources: list[GraphNode], service_id: str
) -> list[EnrichmentEdge]:
    """Link each ``Resource`` to the deployed ``Service`` it provisions
    (``provisions``) — the IaC↔topology bridge (CONCEPT:KG-2.103)."""
    if not service_id:
        return []
    return [
        EnrichmentEdge(source=r.id, target=service_id, rel_type="PROVISIONS")
        for r in resources
    ]
