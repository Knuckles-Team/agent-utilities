#!/usr/bin/env python3
"""Interactive deployment wizard for agent-utilities.

Asks the operator about their use case, then recommends and (optionally) writes a
complete deployment: the XDG ``config.json``, a backend ``.env``, and the run
artifacts for the chosen target (uvx/uv, Docker Compose, or Kubernetes).

Self-contained — standard library only, no third-party imports — so it runs from
the skill directory without installing anything.

Usage::

    python deploy_wizard.py                       # fully interactive
    python deploy_wizard.py --use-case prod-scale # preset, still confirms
    python deploy_wizard.py --use-case test --non-interactive --emit uvx
    python deploy_wizard.py --use-case dev --apply --output-dir ./deploy

By default the wizard is a DRY RUN: it prints what it would write. Pass
``--apply`` to actually write files (it backs up any existing config.json first).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

# ───────────────────────────────────────────────────────────────────────────────
# Use-case profiles. Each profile is a recommended baseline; every value can be
# overridden interactively. Keys map onto config.json fields and backend env vars.
# ───────────────────────────────────────────────────────────────────────────────

TIERS = ["test", "dev", "prod-small", "prod-scale"]

TIER_BLURB = {
    "test": "Just testing / CI — ephemeral, zero disk, zero infra.",
    "dev": "Local dev — one user, persistent, no external servers.",
    "prod-small": "Small production — a team / single node, durable + secured.",
    "prod-scale": "Production at scale — thousands of users, multi-node, HA.",
}

PROFILES: dict[str, dict] = {
    "test": {
        "graph_backend": "memory",  # pure in-memory EpistemicGraph (no disk)
        "graph_backend_l2": None,
        "graph_db_uri": None,
        "app_profile": None,
        "deploy": "uvx",
        "host": "127.0.0.1",
        "port": 9000,
        "enable_web_ui": False,
        "enable_api_auth": False,
        "secrets_backend": "inmemory",
        "a2a_broker": "in-memory",
        "a2a_storage": "in-memory",
        "kafka_bootstrap_servers": None,
        "enable_otel": False,
        "max_concurrent_agents": 2,
        "debug": True,
    },
    "dev": {
        "graph_backend": "tiered",  # epistemic_graph L1 + LadybugDB L2 (no server)
        "graph_backend_l2": "ladybug",
        "graph_db_uri": None,
        "app_profile": None,
        "deploy": "uvx",
        "host": "0.0.0.0",
        "port": 9000,
        "enable_web_ui": True,
        "enable_api_auth": False,
        "secrets_backend": "sqlite",
        "a2a_broker": "in-memory",
        "a2a_storage": "in-memory",
        "kafka_bootstrap_servers": None,
        "enable_otel": False,
        "max_concurrent_agents": 5,
        "debug": False,
    },
    "prod-small": {
        "graph_backend": "tiered",  # L1 + PostgreSQL durable L2
        "graph_backend_l2": "postgresql",
        "graph_db_uri": "postgresql://agent:agent@pggraph:5432/agent_kg",
        "app_profile": "production",
        "deploy": "docker",
        "host": "0.0.0.0",
        "port": 9000,
        "enable_web_ui": True,
        "enable_api_auth": True,
        "secrets_backend": "vault",
        "a2a_broker": "nats",
        "a2a_storage": "postgresql",
        "kafka_bootstrap_servers": "redpanda:9092",
        "enable_otel": True,
        "max_concurrent_agents": 16,
        "debug": False,
    },
    "prod-scale": {
        "graph_backend": "tiered",  # L1 + pooled PostgreSQL/pgGraph durable L2
        "graph_backend_l2": "postgresql",
        "graph_db_uri": "postgresql://agent:agent@pggraph:5432/agent_kg",
        "app_profile": "production",
        "deploy": "kubernetes",
        "host": "0.0.0.0",
        "port": 9000,
        "enable_web_ui": True,
        "enable_api_auth": True,
        "secrets_backend": "vault",
        "a2a_broker": "kafka",
        "a2a_storage": "postgresql",
        "kafka_bootstrap_servers": "redpanda-0:9092,redpanda-1:9092,redpanda-2:9092",
        "enable_otel": True,
        "max_concurrent_agents": 64,
        "debug": False,
    },
}

DEPLOY_TARGETS = ["uvx", "docker", "kubernetes"]

RECOMMENDED_EXTRAS = {
    "test": "mcp",
    "dev": "mcp,graph",
    "prod-small": "mcp,graph,postgresql,auth,vault,messaging",
    "prod-scale": "all",
}


# ───────────────────────────────────────────────────────────────────────────────
# Prompt helpers
# ───────────────────────────────────────────────────────────────────────────────


def _interactive() -> bool:
    return sys.stdin.isatty() and not NON_INTERACTIVE


def ask(label: str, default):
    """Prompt with a default; returns the default in non-interactive mode."""
    if not _interactive():
        return default
    shown = "" if default is None else str(default)
    raw = input(f"  {label} [{shown}]: ").strip()
    if raw == "":
        return default
    return raw


def ask_bool(label: str, default: bool) -> bool:
    if not _interactive():
        return default
    d = "Y/n" if default else "y/N"
    raw = input(f"  {label} ({d}): ").strip().lower()
    if raw == "":
        return default
    return raw in ("y", "yes", "true", "1")


def ask_choice(label: str, choices: list[str], default: str) -> str:
    if not _interactive():
        return default
    while True:
        raw = input(f"  {label} {choices} [{default}]: ").strip()
        if raw == "":
            return default
        if raw in choices:
            return raw
        print(f"    ! choose one of {choices}")


def section(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m")


# ───────────────────────────────────────────────────────────────────────────────
# Interview
# ───────────────────────────────────────────────────────────────────────────────


def pick_tier(preset: str | None) -> str:
    if preset:
        return preset
    section("1. What are you deploying for?")
    for i, t in enumerate(TIERS, 1):
        print(f"  {i}) {t:<11} — {TIER_BLURB[t]}")
    if not _interactive():
        return "dev"
    while True:
        raw = input("  Choose 1-4 [2]: ").strip() or "2"
        if raw in ("1", "2", "3", "4"):
            return TIERS[int(raw) - 1]
        if raw in TIERS:
            return raw
        print("    ! enter 1-4")


def interview(tier: str) -> dict:
    s = dict(PROFILES[tier])
    s["tier"] = tier

    print(f"\nRecommended baseline for \033[1m{tier}\033[0m "
          f"({TIER_BLURB[tier]}):")
    print(f"  backend   : {_backend_summary(s)}")
    print(f"  deploy    : {s['deploy']}")
    print(f"  web UI    : {s['enable_web_ui']}   auth: {s['enable_api_auth']}   "
          f"secrets: {s['secrets_backend']}")
    print(f"  a2a       : broker={s['a2a_broker']} storage={s['a2a_storage']}")
    print(f"  otel      : {s['enable_otel']}   max_concurrent_agents: "
          f"{s['max_concurrent_agents']}")
    if s["app_profile"]:
        print("  APP_PROFILE=production → the profile guard will REQUIRE a "
              "durable Postgres L2, a real broker, and Kafka.")

    if not ask_bool("\nCustomize these settings?", default=_interactive() and tier.startswith("prod")):
        return s

    section("2. Knowledge-graph backend")
    print("  memory     → pure in-memory, ephemeral (tests/CI)")
    print("  tiered     → epistemic_graph L1 + L2 store (recommended)")
    print("  postgresql → single PostgreSQL backend (no L1 compute tier)")
    s["graph_backend"] = ask_choice(
        "GRAPH_BACKEND", ["memory", "tiered", "postgresql"], s["graph_backend"]
    )
    if s["graph_backend"] == "tiered":
        print("    L2 (durable tier): ladybug = embedded/no server; "
              "postgresql = durable/shardable")
        s["graph_backend_l2"] = ask_choice(
            "GRAPH_BACKEND_L2", ["ladybug", "postgresql"], s["graph_backend_l2"] or "ladybug"
        )
        if s["graph_backend_l2"] == "postgresql":
            s["graph_db_uri"] = ask("GRAPH_DB_URI", s["graph_db_uri"]
                                    or "postgresql://agent:agent@pggraph:5432/agent_kg")
        else:
            s["graph_db_uri"] = None
    elif s["graph_backend"] == "postgresql":
        s["graph_backend_l2"] = None
        s["graph_db_uri"] = ask("GRAPH_DB_URI", s["graph_db_uri"]
                                or "postgresql://agent:agent@pggraph:5432/agent_kg")
    else:
        s["graph_backend_l2"] = None
        s["graph_db_uri"] = None

    section("3. Deployment target")
    print("  uvx        → ephemeral run via uv (fastest; dev/test)")
    print("  docker     → docker compose (single node, durable)")
    print("  kubernetes → generated manifests (HA, multi-node)")
    s["deploy"] = ask_choice("deploy", DEPLOY_TARGETS, s["deploy"])

    section("4. Server & access")
    s["host"] = ask("host", s["host"])
    s["port"] = int(ask("port", s["port"]))
    s["enable_web_ui"] = ask_bool("enable web UI", s["enable_web_ui"])
    s["enable_api_auth"] = ask_bool("enable API auth (JWT/OIDC)", s["enable_api_auth"])
    if s["enable_api_auth"]:
        s["oidc_config_url"] = ask("OIDC discovery URL (blank = static token)",
                                   s.get("oidc_config_url"))

    section("5. Secrets, messaging & observability")
    s["secrets_backend"] = ask_choice(
        "secrets_backend", ["inmemory", "sqlite", "vault"], s["secrets_backend"]
    )
    if s["secrets_backend"] == "vault":
        s["vault_url"] = ask("vault_url", s.get("vault_url") or "http://openbao:8200")
    s["a2a_broker"] = ask_choice(
        "a2a_broker", ["in-memory", "nats", "kafka"], s["a2a_broker"]
    )
    s["a2a_storage"] = ask_choice(
        "a2a_storage", ["in-memory", "postgresql", "redis"], s["a2a_storage"]
    )
    if s["a2a_broker"] == "kafka":
        s["kafka_bootstrap_servers"] = ask(
            "kafka_bootstrap_servers", s["kafka_bootstrap_servers"] or "redpanda:9092"
        )
    s["enable_otel"] = ask_bool("enable OpenTelemetry", s["enable_otel"])
    if s["enable_otel"]:
        s["otel_endpoint"] = ask("otel OTLP endpoint",
                                 s.get("otel_endpoint") or "http://otel-collector:4318")

    section("6. Capacity")
    s["max_concurrent_agents"] = int(ask("max_concurrent_agents", s["max_concurrent_agents"]))

    section("7. Models (LLM gateway)")
    s["llm_base_url"] = ask("llm_base_url", s.get("llm_base_url") or "http://vllm.arpa/v1")
    s["model_id"] = ask("default model_id (blank = auto-route)", s.get("model_id"))

    _warn_production_safety(s)
    return s


def _backend_summary(s: dict) -> str:
    if s["graph_backend"] == "tiered":
        l2 = s.get("graph_backend_l2") or ("postgresql" if s.get("graph_db_uri") else "ladybug")
        return f"tiered (epistemic_graph + {l2})"
    return s["graph_backend"]


def _warn_production_safety(s: dict) -> None:
    if s.get("app_profile") != "production":
        return
    problems = []
    l2 = s.get("graph_backend_l2")
    if s["graph_backend"] == "tiered" and l2 != "postgresql" and not s.get("graph_db_uri"):
        problems.append("backend resolves to a single-host LadybugDB L2")
    if s["graph_backend"] in ("memory", "file", "ladybug"):
        problems.append(f"GRAPH_BACKEND={s['graph_backend']} is single-host")
    if s["a2a_broker"] in ("in-memory",):
        problems.append("a2a_broker=in-memory loses messages on restart")
    if not s.get("kafka_bootstrap_servers"):
        problems.append("kafka_bootstrap_servers unset (no durable event ledger)")
    if problems:
        print("\n  \033[33m⚠ APP_PROFILE=production will be REJECTED by the profile "
              "guard:\033[0m")
        for p in problems:
            print(f"    - {p}")
        print("  Set a Postgres L2 (GRAPH_DB_URI), a real broker, and Kafka, or "
              "drop APP_PROFILE.")


# ───────────────────────────────────────────────────────────────────────────────
# Artifact builders
# ───────────────────────────────────────────────────────────────────────────────


def build_config_json(s: dict) -> dict:
    cfg = {
        "default_agent_name": "Agent",
        "host": s["host"],
        "port": s["port"],
        "debug": s.get("debug", False),
        "enable_web_ui": s["enable_web_ui"],
        "enable_api_auth": s["enable_api_auth"],
        "routing_strategy": "hybrid",
        # graph_backend is the authoritative selector (also exported via env)
        "graph_backend": s["graph_backend"],
        "graph_backend_l1": "epistemic_graph",
        "graph_backend_l2": s.get("graph_backend_l2"),
        "graph_db_uri": s.get("graph_db_uri"),
        "secrets_backend": s["secrets_backend"],
        "a2a_broker": s["a2a_broker"],
        "a2a_storage": s["a2a_storage"],
        "kafka_bootstrap_servers": s.get("kafka_bootstrap_servers"),
        "enable_otel": s["enable_otel"],
        "max_concurrent_agents": s["max_concurrent_agents"],
        "llm_base_url": s.get("llm_base_url", "http://vllm.arpa/v1"),
        "model_id": s.get("model_id"),
    }
    if s.get("oidc_config_url"):
        cfg["oidc_config_url"] = s["oidc_config_url"]
    if s.get("vault_url"):
        cfg["vault_url"] = s["vault_url"]
    if s.get("otel_endpoint"):
        cfg["otel_exporter_otlp_endpoint"] = s["otel_endpoint"]
    return {k: v for k, v in cfg.items() if v is not None}


def build_env(s: dict) -> str:
    lines = ["# agent-utilities backend environment (authoritative for backend selection)"]
    lines.append(f"GRAPH_BACKEND={s['graph_backend']}")
    if s["graph_backend"] == "tiered":
        lines.append("GRAPH_BACKEND_L1=epistemic_graph")
        if s.get("graph_backend_l2"):
            lines.append(f"GRAPH_BACKEND_L2={s['graph_backend_l2']}")
    if s.get("graph_db_uri"):
        lines.append(f"GRAPH_DB_URI={s['graph_db_uri']}")
    if s.get("app_profile"):
        lines.append(f"APP_PROFILE={s['app_profile']}")
    if s.get("kafka_bootstrap_servers"):
        lines.append(f"KAFKA_BOOTSTRAP_SERVERS={s['kafka_bootstrap_servers']}")
    lines.append("KG_DAEMON_ROLE=auto")
    return "\n".join(lines) + "\n"


def build_run_commands(s: dict, extras: str) -> str:
    spec = f'agent-utilities[{extras}]'
    if s["deploy"] == "uvx":
        return (
            f"# Ephemeral run with uv (no venv to manage):\n"
            f"uvx --from '{spec}' graph-os --transport stdio\n"
            f"uvx --from '{spec}' graph-os --transport streamable-http "
            f"--host {s['host']} --port 8004\n"
            f"# Full agent server (REST + gateway + UI):\n"
            f"uv run --with '{spec}' python -m agent_utilities\n"
            f"# Multiplexer (aggregate many child MCP servers):\n"
            f"uvx --from '{spec}' mcp-multiplexer --config mcp_config.json "
            f"--transport stdio\n"
        )
    if s["deploy"] == "docker":
        return (
            "# Bring up the MCP server (+ Postgres L2 if selected):\n"
            "docker compose --env-file deploy.env -f docker/mcp.compose.yml up -d\n"
            + ("docker compose -f docker/pggraph.compose.yml up -d\n"
               if s.get("graph_backend_l2") == "postgresql" else "")
        )
    return (
        "# Apply the generated Kubernetes manifests:\n"
        "kubectl apply -f k8s/\n"
        "kubectl -n agent-utilities rollout status deploy/agent-utilities\n"
    )


def build_k8s(s: dict, extras: str) -> str:
    """A minimal but complete K8s manifest set (no charts exist in-repo)."""
    env_items = [("GRAPH_BACKEND", s["graph_backend"])]
    if s["graph_backend"] == "tiered":
        env_items.append(("GRAPH_BACKEND_L1", "epistemic_graph"))
        if s.get("graph_backend_l2"):
            env_items.append(("GRAPH_BACKEND_L2", s["graph_backend_l2"]))
    if s.get("graph_db_uri"):
        env_items.append(("GRAPH_DB_URI", s["graph_db_uri"]))
    if s.get("app_profile"):
        env_items.append(("APP_PROFILE", s["app_profile"]))
    if s.get("kafka_bootstrap_servers"):
        env_items.append(("KAFKA_BOOTSTRAP_SERVERS", s["kafka_bootstrap_servers"]))
    env_yaml = "\n".join(
        f"            - name: {k}\n              value: \"{v}\"" for k, v in env_items
    )
    replicas = 3 if s["tier"] == "prod-scale" else 1
    return f"""# Generated by deploy_wizard.py — tier={s['tier']}
apiVersion: v1
kind: Namespace
metadata:
  name: agent-utilities
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-utilities
  namespace: agent-utilities
spec:
  replicas: {replicas}
  selector:
    matchLabels: {{app: agent-utilities}}
  template:
    metadata:
      labels: {{app: agent-utilities}}
    spec:
      containers:
        - name: graph-os
          image: ghcr.io/knuckles-team/agent-utilities:latest
          args: ["python", "-m", "agent_utilities"]
          ports:
            - containerPort: {s['port']}
          env:
{env_yaml}
          readinessProbe:
            httpGet: {{path: /health, port: {s['port']}}}
            initialDelaySeconds: 10
          resources:
            requests: {{cpu: "1", memory: "2Gi"}}
            limits: {{cpu: "2", memory: "4Gi"}}
---
apiVersion: v1
kind: Service
metadata:
  name: agent-utilities
  namespace: agent-utilities
spec:
  selector: {{app: agent-utilities}}
  ports:
    - port: 80
      targetPort: {s['port']}
"""


def build_compose_override(s: dict) -> str:
    return f"""# Generated by deploy_wizard.py — tier={s['tier']}
# Use with: docker compose --env-file deploy.env -f docker/mcp.compose.yml \\
#   -f docker-compose.override.yml up -d
services:
  kg-server-mcp:
    env_file: [deploy.env]
    environment:
      - HOST={s['host']}
      - PORT={s['port']}
      - TRANSPORT=streamable-http
"""


# ───────────────────────────────────────────────────────────────────────────────
# Emit
# ───────────────────────────────────────────────────────────────────────────────


def _xdg_config_path() -> Path:
    override = os.environ.get("AGENT_UTILITIES_CONFIG_DIR")
    base = Path(override) if override else Path.home() / ".config" / "agent-utilities"
    return base / "config.json"


def emit(s: dict, emit_what: str, out_dir: Path, apply: bool) -> None:
    extras = RECOMMENDED_EXTRAS[s["tier"]]
    config = build_config_json(s)
    env = build_env(s)
    config_path = _xdg_config_path()

    def write(path: Path, content: str, *, is_config=False) -> None:
        if not apply:
            print(f"\n# --- would write {path} ---")
            print(content if len(content) < 1600 else content[:1600] + "\n…(truncated)")
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        if is_config and path.exists():
            backup = path.with_suffix(f".json.bak-{int(time.time())}")
            shutil.copy2(path, backup)
            print(f"  backed up existing config → {backup}")
        path.write_text(content, encoding="utf-8")
        print(f"  wrote {path}")

    section("Plan" if not apply else "Applying")
    print(f"  tier={s['tier']}  backend={_backend_summary(s)}  deploy={s['deploy']}  "
          f"extras=[{extras}]")

    want = {"config", "env"} | (
        {emit_what} if emit_what != "all" else {"uvx", "docker", "kubernetes"}
    )
    # config + env always
    write(config_path, json.dumps(config, indent=2) + "\n", is_config=True)
    out_dir.mkdir(parents=True, exist_ok=True) if apply else None
    write(out_dir / "deploy.env", env)

    if s["deploy"] in want or emit_what == "all":
        if s["deploy"] == "docker" or emit_what == "all":
            write(out_dir / "docker-compose.override.yml", build_compose_override(s))
        if s["deploy"] == "kubernetes" or emit_what == "all":
            write(out_dir / "k8s" / "agent-utilities.yaml", build_k8s(s, extras))

    section("Run it")
    print(build_run_commands(s, extras))
    if not apply:
        print("\n(DRY RUN — re-run with --apply to write these files.)")


# ───────────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────────

NON_INTERACTIVE = False


def main() -> int:
    global NON_INTERACTIVE
    p = argparse.ArgumentParser(description="Interactive agent-utilities deployment wizard")
    p.add_argument("--use-case", choices=TIERS, help="Preset tier (skips the first prompt)")
    p.add_argument("--deploy", choices=DEPLOY_TARGETS, help="Override deployment target")
    p.add_argument("--emit", choices=["uvx", "docker", "kubernetes", "all"], default="all",
                   help="Which run artifacts to emit (config.json + .env always emitted)")
    p.add_argument("--output-dir", default="./deploy", help="Where to write artifacts")
    p.add_argument("--non-interactive", action="store_true",
                   help="Accept all recommended defaults, no prompts")
    p.add_argument("--apply", action="store_true",
                   help="Actually write files (default is a dry run)")
    args = p.parse_args()
    NON_INTERACTIVE = args.non_interactive

    print("\033[1magent-utilities deployment wizard\033[0m")
    print("Maps your use case → backend, deploy target, and config.json.\n")

    tier = pick_tier(args.use_case)
    settings = interview(tier)
    if args.deploy:
        settings["deploy"] = args.deploy

    emit(settings, args.emit, Path(args.output_dir), args.apply)
    return 0


if __name__ == "__main__":
    sys.exit(main())
