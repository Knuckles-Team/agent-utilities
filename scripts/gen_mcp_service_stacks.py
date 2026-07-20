#!/usr/bin/env python3
"""Generate GitOps stack dirs for *-mcp connectors that aren't deployed yet.

For every service in ``deploy/mcp-fleet.registry.yml`` that lacks a
``<services-dir>/<name>/`` directory, generate the standard Portainer swarm
stack (``compose.yml`` + ``AGENTS.md`` + ``README.md``) matching the convention
of the already-deployed connectors (e.g. ``services/github-mcp``):

  image:   <runtime image prefix>/<package>:<runtime tag>
  command: [<name>]            # the package's console script
  transport: streamable-http on :8000, /health healthcheck
  network, authentication, policy endpoint, and placement are operator inputs

These dirs become individual GitLab repos (push-to-create) and Portainer GitOps
swarm stacks. Idempotent: never overwrites an existing service dir.

Usage:
    python scripts/gen_mcp_service_stacks.py \
        --registry deploy/mcp-fleet.registry.yml \
        --services-dir /path/to/workspace/services
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

_SERVICE_COMPONENT = re.compile(r"^[a-z0-9](?:[a-z0-9._-]{0,126}[a-z0-9])?$")
_PLACEMENT_COMPONENT = re.compile(
    r"^[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,126}[A-Za-z0-9])?$"
)
_IMAGE_PREFIX = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._:/-]{0,253}[A-Za-z0-9])?$")
_IMAGE_TAG = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$")

COMPOSE_TMPL = """version: '3.8'
services:
  {name}:
    image: {image_prefix}/{package}:{image_tag}
    hostname: {name}
    restart: always
    networks:
    - {network}
    environment:
    - PYTHONUNBUFFERED=1
    - HOST=0.0.0.0
    - PORT=8000
    - TRANSPORT=streamable-http
    # Authentication and policy authority are runtime configuration.
    - AUTH_TYPE=jwt
    - FASTMCP_SERVER_AUTH_JWT_AUDIENCE=${{FASTMCP_SERVER_AUTH_JWT_AUDIENCE:?required}}
    - FASTMCP_SERVER_AUTH_JWT_ISSUER=${{FASTMCP_SERVER_AUTH_JWT_ISSUER:?required}}
    - FASTMCP_SERVER_AUTH_JWT_JWKS_URI=${{FASTMCP_SERVER_AUTH_JWT_JWKS_URI:?required}}
    - EUNOMIA_TYPE=remote
    - EUNOMIA_REMOTE_URL=${{EUNOMIA_REMOTE_URL:?required}}
    command:
    - {name}
    healthcheck:
      test: ["CMD", "python3", "-c", "import socket; socket.create_connection(('localhost', 8000), timeout=5)"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s
    logging:
      driver: json-file
      options:
        max-size: 10m
        max-file: '3'
    deploy:
      placement:
        constraints:
        - node.labels.{placement_label} == ${{SERVER:-{node}}}
      restart_policy:
        condition: any
networks:
  {network}:
    external: true
"""

README_TMPL = """# {name}

GitOps stack for the `{name}` Model Context Protocol server, served over
streamable-http on port 8000.

## Deploy

```
cd existing_repo
test -n "${{GIT_REMOTE_URL:?set the target repository URL}}"
git remote add origin "$GIT_REMOTE_URL"
git branch -M main
git push -u origin main
```

Then create or redeploy the target platform's Swarm stack from this repository.
The public/service URL is supplied by the target environment.

Per-service credentials and configuration are injected by the target platform's
secret controller, not committed here.
"""

AGENTS_TMPL = """# AGENTS.md - AI Agent Context

## Role in Agent OS Architecture
The `{name}` service is an active operational MCP server running within the Swarm
overlay network (image selected by deployment configuration).

### Intent & Function
- **Ecosystem Capability**: Model Context Protocol adapter exposed over
  streamable-http on port 8000.
- **Network Access**: Resolved from the runtime service registry.
- **Integration Layer**: AI agents reach it through GraphOS's `{name}` fleet
  tool surface to automate its domain.

### How to Interact
1. **MCP / HTTP**: Resolve the endpoint from the runtime fleet registry.
2. **Lifecycle**: Use the deployment platform's standard lifecycle interface.
3. **Config/Secrets**: Injected by the deployment secret controller.
"""


def parse_registry(path: Path) -> list[tuple[str, str]]:
    """Return [(name, package), ...] from the registry YAML (no yaml dep needed)."""
    services: list[tuple[str, str]] = []
    name = pkg = None
    for line in path.read_text().splitlines():
        m = re.match(r"\s+- name:\s*(\S+)", line)
        if m:
            if name and pkg:
                services.append((name, pkg))
            name, pkg = m.group(1), None
            continue
        m = re.match(r"\s+package:\s*(\S+)", line)
        if m and name:
            pkg = m.group(1)
    if name and pkg:
        services.append((name, pkg))
    return services


def _require_match(value: str, pattern: re.Pattern[str], option: str) -> str:
    """Reject values that could escape a path or alter generated YAML."""
    clean = value.strip()
    if not pattern.fullmatch(clean) or ".." in clean:
        raise ValueError(f"{option} contains unsupported characters")
    return clean


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registry", required=True, type=Path)
    ap.add_argument("--services-dir", required=True, type=Path)
    ap.add_argument("--image-prefix", required=True)
    ap.add_argument("--image-tag", default="latest")
    ap.add_argument("--network", required=True)
    ap.add_argument(
        "--placement-nodes",
        required=True,
        help="comma-separated node-label values",
    )
    ap.add_argument("--placement-label", default="name")
    args = ap.parse_args()

    services = parse_registry(args.registry)
    try:
        image_prefix = _require_match(
            args.image_prefix.rstrip("/"), _IMAGE_PREFIX, "--image-prefix"
        )
        image_tag = _require_match(args.image_tag, _IMAGE_TAG, "--image-tag")
        network = _require_match(args.network, _SERVICE_COMPONENT, "--network")
        placement_label = _require_match(
            args.placement_label, _PLACEMENT_COMPONENT, "--placement-label"
        )
        placement_nodes = [
            _require_match(value, _PLACEMENT_COMPONENT, "--placement-nodes")
            for value in args.placement_nodes.split(",")
            if value.strip()
        ]
    except ValueError as exc:
        ap.error(str(exc))
    if not placement_nodes:
        ap.error("--placement-nodes must contain at least one value")
    generated: list[str] = []
    idx = 0
    for name, package in services:
        try:
            name = _require_match(name, _SERVICE_COMPONENT, "registry service name")
            package = _require_match(
                package, _SERVICE_COMPONENT, "registry package name"
            )
        except ValueError as exc:
            ap.error(str(exc))
        dest = args.services_dir / name
        if dest.exists():
            continue  # already deployed / present
        node = placement_nodes[idx % len(placement_nodes)]
        idx += 1
        dest.mkdir(parents=True)
        (dest / "compose.yml").write_text(
            COMPOSE_TMPL.format(
                name=name,
                package=package,
                node=node,
                image_prefix=image_prefix,
                image_tag=image_tag,
                network=network,
                placement_label=placement_label,
            )
        )
        (dest / "README.md").write_text(README_TMPL.format(name=name, package=package))
        (dest / "AGENTS.md").write_text(AGENTS_TMPL.format(name=name, package=package))
        generated.append(f"{name} ({package} -> configured placement)")

    print(f"Generated {len(generated)} new stack dirs:")
    for g in generated:
        print(f"  + {g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
