#!/usr/bin/env python3
"""Generate GitOps stack dirs for *-mcp connectors that aren't deployed yet.

For every service in ``deploy/mcp-fleet.registry.yml`` that lacks a
``<services-dir>/<name>/`` directory, generate the standard Portainer swarm
stack (``compose.yml`` + ``AGENTS.md`` + ``README.md``) matching the convention
of the already-deployed connectors (e.g. ``services/github-mcp``):

  image:   knucklessg1/<package>:latest
  command: [<name>]            # the package's console script
  transport: streamable-http on :8000, /health healthcheck
  networks: caddy/cloudflare/internet, dns 10.0.0.199
  swarm placement round-robin over the misc-MCP workers (GR1080 excluded — down)

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

PLACEMENT_NODES = ["R710", "RW710"]  # stateless MCP workers; GR1080 is down

COMPOSE_TMPL = """version: '3.8'
services:
  {name}:
    image: knucklessg1/{package}:latest
    hostname: {name}
    restart: always
    networks:
    - caddy
    - cloudflare
    - internet
    dns:
    - 10.0.0.199
    environment:
    - PYTHONUNBUFFERED=1
    - HOST=0.0.0.0
    - PORT=8000
    - TRANSPORT=streamable-http
    command:
    - {name}
    healthcheck:
      test:
      - CMD
      - python3
      - -c
      - import urllib.request; urllib.request.urlopen('http://localhost:8000/health')
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
        - node.labels.name == ${{SERVER:-{node}}}
      restart_policy:
        condition: any
networks:
  cloudflare:
    external: true
  internet:
    external: true
  caddy:
    external: true
"""

README_TMPL = """# {name}

Portainer GitOps stack for the `{name}` Model Context Protocol server
(`knucklessg1/{package}`), served over streamable-http on port 8000.

## Deploy

```
cd existing_repo
git remote add origin http://gitlab.arpa/homelab/containers/services/{name}.git
git branch -M main
git push -u origin main
```

Then create/redeploy the Portainer **swarm** stack from this repository
(GitOps auto-sync). Reachable internally at `http://{name}.arpa`.

Per-service credentials/config are injected as Portainer stack environment
variables (or from OpenBao at runtime), not committed here.
"""

AGENTS_TMPL = """# AGENTS.md - AI Agent Context

## Role in Agent OS Architecture
The `{name}` service is an active operational MCP server running within the Swarm
overlay network (image `knucklessg1/{package}`).

### Intent & Function
- **Ecosystem Capability**: Model Context Protocol adapter exposed over
  streamable-http on port 8000.
- **LAN Access**: Reachable internally at `http://{name}.arpa`.
- **Integration Layer**: AI agents reach it via the `{name}` tool surface (or
  through the `mcp-multiplexer`) to automate its domain.

### How to Interact
1. **MCP / HTTP**: Connect to the internal endpoint `http://{name}.arpa` (or the
   mapped host port from the fleet registry).
2. **Lifecycle**: Use `portainer-agent` / `container-manager-mcp` to check
   replication, scale, or trigger updates.
3. **Config/Secrets**: Injected as Portainer stack env / OpenBao at runtime.
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registry", required=True, type=Path)
    ap.add_argument("--services-dir", required=True, type=Path)
    args = ap.parse_args()

    services = parse_registry(args.registry)
    generated: list[str] = []
    idx = 0
    for name, package in services:
        dest = args.services_dir / name
        if dest.exists():
            continue  # already deployed / present
        node = PLACEMENT_NODES[idx % len(PLACEMENT_NODES)]
        idx += 1
        dest.mkdir(parents=True)
        (dest / "compose.yml").write_text(
            COMPOSE_TMPL.format(name=name, package=package, node=node)
        )
        (dest / "README.md").write_text(README_TMPL.format(name=name, package=package))
        (dest / "AGENTS.md").write_text(AGENTS_TMPL.format(name=name, package=package))
        generated.append(f"{name} (knucklessg1/{package} -> {node})")

    print(f"Generated {len(generated)} new stack dirs:")
    for g in generated:
        print(f"  + {g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
