#!/usr/bin/env python3
"""Fail when a removed compatibility surface returns to shipped artifacts.

The project has one current contract. This gate intentionally scans runtime code,
tests, documentation, deployment assets, and repository guidance so a deleted
environment switch, alias, or raw graph endpoint cannot survive on a secondary
surface after its implementation is removed.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOTS = (
    ROOT / "agent_utilities",
    ROOT / "scripts",
    ROOT / "tests",
    ROOT / "docs",
    ROOT / "deploy",
    ROOT / ".github",
)
SCAN_FILES = (
    ROOT / ".env.example",
    ROOT / "AGENTS.head.md",
    ROOT / "AGENTS.md",
    ROOT / "README.md",
    ROOT / "pyproject.toml",
)
TEXT_SUFFIXES = {
    ".cfg",
    ".env",
    ".html",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".yaml",
    ".yml",
}

# Split each spelling so the gate does not match its own source.
RETIRED_IDENTIFIERS: tuple[str, ...] = (
    "ENGINE_" + "MODE",
    "ENGINE_" + "ENDPOINT",
    "EPISTEMIC_GRAPH_" + "AUTOSTART",
    "GRAPH_SERVICE_" + "SOCKET",
    "GRAPH_SERVICE_TCP_" + "ADDR",
    "GRAPH_" + "BACKEND",
    "GRAPH_" + "AUTHORITY",
    "DURABLE_" + "EXECUTION_DB",
    "DurableExecution" + "Manager",
    "PostgresCheckpoint" + "Store",
    "SQLiteCheckpoint" + "Store",
    "A2A_BROKER_" + "URL",
    "A2A_STORAGE_" + "URL",
    '"a2a_broker_' + 'url"',
    '"a2a_storage_' + 'url"',
    '"a2a_broker": "in-' + 'memory"',
    '"a2a_storage": "in-' + 'memory"',
    '"graph_direct_' + 'execution"',
    '"secrets_backend": "in' + 'memory"',
    "PERMISSIONS_SIGNING_" + "KEY=",
    '"permissions_signing_' + 'key"',
    "ElasticContext" + "Manager",
    "mark_system_" + "synced",
    "is_system_" + "synced",
    "start_sdd_" + "watcher",
    "knowledge_graph/self_" + "model.py",
    "SelfModel" + "Node",
    "compute_diversity_" + "metrics",
    "diversity_preserving_" + "consolidation",
    "embedding_health_" + "check",
    "install-" + "skills",
    "KG_" + "SERVED_PROFILE",
    "kg_" + "served_profile",
    "KG_ENVELOPE_" + "LEGACY_ADAPTER",
    "kg_envelope_" + "legacy_adapter",
    "CONNECTOR_" + "DEFAULT_PUBLIC",
    "CONNECTOR_MANIFEST_" + "REQUIRE_ENTERPRISE",
    "UV_" + "SYSTEM_CERTS",
    "Centralized" + "CypherMiddleware",
    "enterprise_" + "required_sources",
    "secrets_vault_" + "url",
    "secrets_vault_" + "mount",
    "get_mcp_agent_" + "registry",
    "dev_execute_" + "agent.py",
    "dev_orchestration_" + "seam_e2e.py",
    "AUTH_JWT_ALLOW_" + "INSECURE_HTTP",
    "auth_jwt_allow_" + "insecure_http",
    "SERVER_ALLOW_UNAUTHENTICATED_" + "REMOTE",
    "server_allow_unauthenticated_" + "remote",
    "SERVER_ALLOW_WILDCARD_" + "HOSTS",
    "server_allow_wildcard_" + "hosts",
    "MCP_ALLOW_UNAUTHENTICATED_" + "REMOTE",
    "mcp_allow_unauthenticated_" + "remote",
    "KG_INGEST_ENGINE_" + "ENDPOINT",
    "kg_ingest_engine_" + "endpoint",
    "main = " + "mcp_server",
    "kg_server:" + "main",
    "agent-utilities-" + "kg",
    "KG_SERVER_" + "HOST",
    "KG_SERVER_" + "PORT",
    "KG" + "Coordinator",
    "kg_" + "coordinator",
    "migrate_" + "legacy_asset",
    "imprint_" + "connection",
    "set_default_" + "connection",
    "BulkMigration" + "Result",
    "fetch_" + "asset",
    "uv run " + "graph-os",
    "MCP_" + "MULTIPLEXER_MODE",
    "mcp_" + "multiplexer_mode",
    "MCP_DYNAMIC_" + "ALWAYS_ON",
    "mcp_dynamic_" + "always_on",
    "FLEET_MCP_" + "SCHEME",
    "fleet_mcp_" + "scheme",
    "FLEET_MCP_" + "DOMAIN",
    "GRAPH_SERVICE_CHECKPOINT_" + "SECS",
    "graph_service_checkpoint_" + "secs",
    "GRAPH_SERVICE_CHECKPOINT_" + "INTERVAL",
    "GRAPH_COMPUTE_" + "FALLBACK",
    "graph_compute_" + "fallback",
    "KG_ALLOW_" + "FULL_SCAN",
    "kg_allow_" + "full_scan",
    "Query" + "Tier",
    "WorkingSet" + "Manager",
    "KafkaGraphSync" + "Daemon",
    "TieredGraph" + "Backend",
    "GRAPH_" + "BACKEND_L1",
    "GRAPH_" + "BACKEND_L2",
    "GRAPH_ROUTING_" + "STRATEGY",
    "graph_routing_" + "strategy",
    "WORKING_SET_" + "EVICTION_RATIO",
    "WORKING_SET_" + "MAX_EDGES",
    "WORKING_SET_" + "MAX_NODES",
    "QUERY_ROUTER_" + "L1_THRESHOLD",
    "RESIDENTS_PER_" + "L0_SHARD",
    "l0_shards_" + "for",
    "reconcile_" + "to_durable",
    "reconcile_" + "durable",
    "checkout_" + "subgraph",
    "hydrate_" + "compute_engine",
    "sync_" + "embeddings",
    "load_" + "subgraph",
    "query_nx_" + "fallback",
    "core/subgraph_" + "checkout.py",
    "core/kafka_graph_" + "sync.py",
    "core/working_set_" + "manager.py",
    "strategies/query_" + "tier.py",
    "L0/L1/" + "L2/L3",
    "L1 compute " + "graph",
    "L0 compute " + "tier",
    "L3 durable " + "mirror",
    "Tiered Graph " + "Engine",
    "compute " + "mirror",
    "tiered " + "backend",
    "HAVE_" + "KERNEL",
    "KERNEL_" + "SOURCE",
    "numeric-" + "kernel = [",
    "from_research_" + "artifact",
    '_check_extension("' + "pggraph" + '")',
    "classify_" + "legacy",
    "migrate_legacy_" + "mementos",
    "legacy_source_" + "migrated",
    "legacy_" + "migrated",
    "_migrate_legacy_" + "sqlite",
    "SQLite" + "Backend",
    "secrets." + "db",
    "AGENT_SECRETS_" + "MASTER_KEY",
    "SECRETS_BACKEND=" + "inmemory",
    'backend="' + "inmemory" + '"',
    "agent-utilities.dev/ontology" + "/",
    "EtlResult." + "coerce",
    "EtlResult." + "count_of",
    "envelope: str = " + "Field(",
    "legacy GoalStatus " + "view",
    "ENABLE_API_" + "AUTH",
    "enable_api_" + "auth",
    "AGENT_API_" + "KEY",
    "agent_api_" + "key",
    "verify_api_key_" + "only",
    "DEVELOPER_HOST_TOOLS_" + "ENABLED",
    "developer_host_tools_" + "enabled",
    "launch_agent_in_" + "terminal",
    "core/agent_" + "launcher.py",
    "agent_run_shell_" + "command",
    "run_shell_" + "command",
    "DEFAULT_TERMINAL_" + "AGENT",
    "default_terminal_" + "agent",
    "execute_shell_" + "command",
    "Retry" + "Manager",
    "Retry" + "Config",
    "Success" + "Check",
    "ShellCheck" + "Result",
    "run_shell_with_" + "diagnostics",
    "replace_in_" + "file",
    "create_" + "worktree",
    "remove_" + "worktree",
    "export_knowledge_" + "base",
    "update_agents_" + "md",
    "init_agents_" + "md",
    "append_note_to_" + "file",
    "_enforce_admin_" + "scope",
    "legacy_observations_v1_" + "get_many",
    "parse_concept_" + "id",
    "canonicalize_concept_" + "id",
    "build_alias_" + "index",
    "observed_project_" + "namespaces",
    "derive_part_of_" + "edges",
    "LEGACY_" + "PILLAR",
    "PROJECT_" + "NAMESPACES",
    "PILLAR_" + "MAP",
    "migrate_concepts_" + "hierarchy.py",
    "plan_concept_" + "migration.py",
    "apply_concept_" + "migration.py",
    "check_no_legacy_" + "markers.py",
    "reserve_concepts_" + "hook.py",
    "concept reserve --" + "ns",
    ":flat" + "Id",
    ":dotted" + "Id",
    "agent_utilities/exceptions" + ".py",
    "agent_utilities/mcp_" + "utilities.py",
    "agent_utilities/graph/" + "steps.py",
    "knowledge_graph/core/ingest_" + "engine.py",
    "custom_" + "nodes",
    "mcp_servers_" + "config",
    "AgentTrace" + "Node",
    "KG_INGEST_GRAPH_" + "ROUTING",
    "kg_ingest_graph_" + "routing",
    "routing_" + "enabled",
    "build_designation_" + "index",
    "CapabilityIndex" + "Watcher",
    "get_designation_" + "index",
    "core.chat_persistence import compact_" + "messages",
    "chat_persistence.compact_" + "messages",
    "_from_config_" + "json",
    "satisfies" + "Compliance",
    "conformsTo" + "Standard",
    ".local.example" + ".com",
)
RETIRED_PATHS: tuple[str, ...] = (
    "agent_utilities/core/agent_" + "launcher.py",
    "agent_utilities/exceptions" + ".py",
    "agent_utilities/graph/" + "steps.py",
    "agent_utilities/knowledge_graph/core/ingest_" + "engine.py",
    "agent_utilities/orchestration/durable_" + "execution.py",
    "agent_utilities/mcp_" + "utilities.py",
    "agent_utilities/mcp/kg_" + "coordinator.py",
    "scripts/apply_concept_" + "migration.py",
    "scripts/autocurate_" + "repo.py",
    "scripts/check_concept_" + "gaps.py",
    "scripts/check_no_legacy_" + "markers.py",
    "scripts/consolidate_" + "concepts.py",
    "scripts/curate_" + "batches.py",
    "scripts/inject_concept_" + "ids.py",
    "scripts/migrate_concepts_" + "hierarchy.py",
    "scripts/plan_concept_" + "migration.py",
    "scripts/reserve_concepts_" + "hook.py",
    "tests/unit/mcp/test_kg_" + "coordinator.py",
    "tests/unit/test_kg_" + "coordinator_client_role.py",
)
PATH_RETIRED_IDENTIFIERS: tuple[tuple[str, str], ...] = (
    (
        "agent_utilities/base_utilities.py",
        "from agent_utilities.core.config import " + "setting",
    ),
    (
        "agent_utilities/tools/dynamic_tool_orchestrator.py",
        "defaults to returning " + "all tools",
    ),
    (
        "agent_utilities/tools/dynamic_tool_orchestrator.py",
        "# Fallback: if zero matches " + "found",
    ),
    (
        "docs/guides/dynamic-tool-selection.md",
        "falls back to exposing the " + "complete set",
    ),
    (
        "docs/examples/graph-os-mcp-examples.md",
        "exhaustive examples of every possible tool " + "configuration",
    ),
    (
        "agent_utilities/mcp/multiplexer.py",
        "stdio_client(" + "server_params)",
    ),
    (
        "agent_utilities/tools/developer_tools.py",
        "async def apply_" + "edits(",
    ),
    (
        "agent_utilities/tools/developer_tools.py",
        "async def create_" + "file(",
    ),
    (
        "agent_utilities/tools/developer_tools.py",
        "async def delete_" + "file(",
    ),
    (
        "agent_utilities/tools/developer_tools.py",
        "async def replace_in_" + "file(",
    ),
    (
        "agent_utilities/tools/developer_tools.py",
        "async def run_shell_with_" + "diagnostics(",
    ),
    (
        "agent_utilities/tools/git_tools.py",
        "async def create_" + "worktree(",
    ),
    (
        "agent_utilities/tools/git_tools.py",
        "async def remove_" + "worktree(",
    ),
    (
        "agent_utilities/tools/knowledge_tools.py",
        "async def export_knowledge_" + "base(",
    ),
    (
        "agent_utilities/tools/memory_tools.py",
        "async def init_agents_" + "md(",
    ),
    (
        "agent_utilities/tools/memory_tools.py",
        "async def update_agents_" + "md(",
    ),
    (
        "agent_utilities/tools/workspace_tools.py",
        "async def append_note_to_" + "file(",
    ),
    (
        "agent_utilities/tools/workspace_tools.py",
        "async def create_" + "skill(",
    ),
    (
        "agent_utilities/tools/workspace_tools.py",
        "async def delete_" + "skill(",
    ),
    (
        "agent_utilities/tools/workspace_tools.py",
        "async def edit_" + "skill(",
    ),
    (
        "agent_utilities/core/chat_persistence.py",
        "def compact_" + "messages(",
    ),
)
PATH_REQUIRED_IDENTIFIERS: tuple[tuple[str, str], ...] = (
    (
        "agent_utilities/base_utilities.py",
        "from agent_utilities.core._env import setting",
    ),
    (
        "agent_utilities/mcp/server_factory.py",
        "if query_filter and not reject_all:",
    ),
    (
        "agent_utilities/tools/dynamic_tool_orchestrator.py",
        "No match or query failure returns no tools.",
    ),
    (
        "agent_utilities/mcp/multiplexer.py",
        "stdio_client(server_params, errlog=child_error_sink)",
    ),
    (
        "tests/unit/mcp/test_dynamic_tool_selection.py",
        "def test_dynamic_visibility_transform_kg_no_match_fails_closed(",
    ),
    (
        "tests/unit/mcp/test_dynamic_tool_selection.py",
        "def test_dynamic_visibility_transform_without_active_graph_fails_closed(",
    ),
    (
        "tests/unit/mcp/test_dynamic_tool_selection.py",
        "def test_dynamic_visibility_transform_kg_error_fails_closed(",
    ),
)
RAW_ROUTE_FRAGMENTS: tuple[str, ...] = (
    '"/' + 'cypher"',
    "'/" + "cypher'",
    "POST /" + "cypher",
)

# One exact README sentence names the rejected launch keys so operators can
# diagnose an old configuration. No other occurrence is accepted.
_README_RETIRED_KEY_LINE = (
    f"`{RETIRED_IDENTIFIERS[0]}`, `{RETIRED_IDENTIFIERS[1]}`, and "
    f"`{RETIRED_IDENTIFIERS[2]}` are retired and"
)


def _iter_files() -> list[Path]:
    roots = [str(root.relative_to(ROOT)) for root in SCAN_ROOTS if root.exists()]
    result = subprocess.run(
        ["rg", "--files", *roots],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    files = [path for path in SCAN_FILES if path.is_file()]
    for name in result.stdout.splitlines():
        path = ROOT / name
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            files.append(path)
    return sorted(set(files))


def _check_repository_with_rg() -> list[str]:
    """Scan repository roots in one native ripgrep pass.

    WSL workspaces on NTFS make thousands of Python ``read_text`` calls
    disproportionately expensive. Ripgrep batches that I/O while preserving the
    exact same line-level contract.
    """
    needles = RETIRED_IDENTIFIERS + RAW_ROUTE_FRAGMENTS
    roots = [str(path.relative_to(ROOT)) for path in SCAN_ROOTS if path.exists()]
    command = ["rg", "-n", "-F", "--no-heading", "--color=never"]
    for needle in needles:
        command.extend(("-e", needle))
    for suffix in sorted(TEXT_SUFFIXES):
        command.extend(("--glob", f"*{suffix}"))
    command.extend(roots)
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if result.returncode not in {0, 1}:
        return [f"repository scan failed (ripgrep exit {result.returncode})"]

    violations: list[str] = []
    for match in result.stdout.splitlines():
        name, line_number, line = match.split(":", 2)
        path = ROOT / name
        for needle in needles:
            if needle not in line:
                continue
            if path == ROOT / "README.md" and line == _README_RETIRED_KEY_LINE:
                continue
            violations.append(f"{name}:{line_number}: retired surface {needle!r}")

    # Explicit root files have suffixes such as ``.example`` and are not all
    # selected by the root globs above.
    violations.extend(
        check(ROOT, paths=[path for path in SCAN_FILES if path.is_file()])
    )
    for relative in RETIRED_PATHS:
        if (ROOT / relative).exists():
            violations.append(f"{relative}: retired path exists")
    for relative, needle in PATH_RETIRED_IDENTIFIERS:
        path = ROOT / relative
        if not path.is_file():
            continue
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if needle in line:
                violations.append(
                    f"{relative}:{line_number}: retired surface {needle!r}"
                )
    for relative, needle in PATH_REQUIRED_IDENTIFIERS:
        path = ROOT / relative
        if not path.is_file():
            violations.append(f"{relative}: required current surface is missing")
            continue
        if needle not in path.read_text(encoding="utf-8"):
            violations.append(
                f"{relative}: required current surface {needle!r} is missing"
            )
    return violations


def check(root: Path = ROOT, *, paths: Iterable[Path] | None = None) -> list[str]:
    if paths is None and root == ROOT:
        return _check_repository_with_rg()
    violations: list[str] = []
    needles = RETIRED_IDENTIFIERS + RAW_ROUTE_FRAGMENTS
    inspected = _iter_files() if paths is None else sorted(set(paths))
    for path in inspected:
        relative = path.relative_to(root).as_posix()
        if relative in RETIRED_PATHS:
            violations.append(f"{relative}: retired path exists")
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as exc:
            violations.append(
                f"{path.relative_to(root)}: could not inspect ({type(exc).__name__})"
            )
            continue
        for line_number, line in enumerate(lines, start=1):
            for needle in needles:
                if needle in line:
                    if path == ROOT / "README.md" and line == _README_RETIRED_KEY_LINE:
                        continue
                    violations.append(
                        f"{path.relative_to(root)}:{line_number}: retired surface {needle!r}"
                    )
            for retired_path, needle in PATH_RETIRED_IDENTIFIERS:
                if relative == retired_path and needle in line:
                    violations.append(
                        f"{relative}:{line_number}: retired surface {needle!r}"
                    )
    return violations


def main() -> int:
    violations = check()
    if violations:
        print("Current-only contract violations:", file=sys.stderr)
        for violation in violations:
            print(f"  - {violation}", file=sys.stderr)
        return 1
    print("Current-only contract: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
