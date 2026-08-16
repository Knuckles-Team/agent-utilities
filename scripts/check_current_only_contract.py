#!/usr/bin/env python3
"""Fail when a removed compatibility surface returns to shipped artifacts.

The project has one current contract. This gate intentionally scans runtime code,
tests, documentation, deployment assets, and repository guidance so a deleted
environment switch, alias, or raw graph endpoint cannot survive on a secondary
surface after its implementation is removed.

D-MQR-11: a handful of RETIRED_PATHS/RETIRED_IDENTIFIERS entries are not gaps
in this gate's coverage but *documented, cross-repo-blocked, deliberately
carried* debt (BUG-032/GOC-59 shape) -- the module docstrings right below say
so in terms ("do not delete the file or this entry to silence it"). Before
this fix the gate could not tell that population apart from a genuine new
regression: both exited 1, both printed as "violations", so a reader (human
or the merge queue) saw one undifferentiated pile and had no way to trust
"zero" was ever reachable or that a change actually introduced something new.
``ACCEPTED_RESIDUALS`` (below) makes that split real: an accepted residual is
reported as carried, non-blocking INFO with its rationale and owner; anything
not on that list still fails loudly, exactly as before. The list is typed
data with required ``reason``/``owner`` fields (``AcceptedResidual.__post_init__``
rejects a blank one) specifically so an entry cannot be added -- or a real
violation silenced -- without a documented reason.
"""

from __future__ import annotations

<<<<<<< HEAD
=======
import subprocess
import argparse
>>>>>>> 4c0d900a910fc1f8f1339df8f68d44570efeb858
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts._git_scan import tracked_or_walked  # noqa: E402

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
    "imprint_" + "connection",
    "set_default_" + "connection",
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
    "core.chat_persistence import compact_" + "messages",
    "chat_persistence.compact_" + "messages",
    "_from_config_" + "json",
    "satisfies" + "Compliance",
    "conformsTo" + "Standard",
    ".local.example" + ".com",
)
RETIRED_PATHS: tuple[str, ...] = (
    "agent_utilities/core/agent_" + "launcher.py",
    # GOC-59 (BUG-032 shape): accepted residual, NOT a gap in this gate's
    # coverage. This module is a back-compat re-export shim over
    # ``agent_utilities.core.exceptions`` (its own docstring says so). It has
    # zero in-repo importers, but ``agents/microsoft-agent`` -- a real,
    # actively-developed repo with a live GitHub origin remote
    # (github.com/Knuckles-Team/microsoft-agent) -- imports it at
    # ``tests/test_auth_coverage.py:5``
    # (``from agent_utilities.exceptions import AuthError, UnauthorizedError``).
    # Deleting it would break that repo's test suite; migrating that one
    # caller onto ``agent_utilities.core.exceptions`` is a separate,
    # cross-repo change, not a Phase-0 publication-unblock item. Carried and
    # owned, not silently cleared.
    "agent_utilities/exceptions" + ".py",
    "agent_utilities/graph/" + "steps.py",
    "agent_utilities/knowledge_graph/core/ingest_" + "engine.py",
    # BUG-032 (GOC-59): accepted residual, NOT a gap in this gate's coverage.
    # Deletion is blocked by live cross-repo importers -- 15+ files across
    # 7+ repos under agents/, plus scaffold_package.py:1650,1978 (which
    # emits `from agent_utilities.mcp_utilities import ...` into every
    # newly scaffolded package). A fleet-wide migration off this module is
    # a separate program, not a Phase-0 publication-unblock item. This
    # finding is meant to keep reporting until that migration lands --
    # do not delete the file or this entry to silence it.
    "agent_utilities/mcp_" + "utilities.py",
    "agent_utilities/mcp/kg_" + "coordinator.py",
    "scripts/apply_concept_" + "migration.py",
    "scripts/autocurate_" + "repo.py",
    # BUG-032 (GOC-59): accepted residual, same shape as mcp_utilities.py
    # above. Referenced by the `.pre-commit-config.yaml` of all 61
    # `agents/*` packages plus agent-webui, agent-terminal-ui, geniusbot,
    # and the scaffolder -- deleting it would break the fleet's gates
    # wholesale. Carried and owned, not silently cleared.
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


@dataclass(frozen=True)
class AcceptedResidual:
    """One deliberately-carried retired-surface finding (BUG-032/GOC-59 shape).

    ``relative`` + ``needle`` identify exactly the finding this record covers
    -- ``needle=None`` matches a whole ``RETIRED_PATHS`` "retired path exists"
    finding; a string ``needle`` matches a "retired surface" text match at
    that path (any line -- line numbers drift as unrelated content nearby
    changes, so this does not pin one). ``reason`` and ``owner`` are
    mandatory and validated non-blank: a record cannot be constructed without
    them, so an entry cannot be added -- or a real violation silenced -- by
    dropping the explanation.
    """

    relative: str
    needle: str | None
    owner: str
    reason: str

    def __post_init__(self) -> None:
        if not self.relative.strip():
            raise ValueError("AcceptedResidual.relative must not be blank")
        if not self.owner.strip():
            raise ValueError("AcceptedResidual.owner must not be blank")
        if not self.reason.strip():
            raise ValueError("AcceptedResidual.reason must not be blank")

    def matches(self, relative: str, needle: str | None) -> bool:
        return self.relative == relative and self.needle == needle

    def describe(self, message: str) -> str:
        return f"{message} -- ACCEPTED RESIDUAL: {self.reason} [owner: {self.owner}]"


# Every entry below already carries a documented rationale + owner at its
# RETIRED_PATHS definition above (or, for phase10-cutover-runbook.md, in that
# file's own 2026-08-09 accepted-residual banner) -- this registry formalizes
# that existing documentation into typed, machine-checked data rather than
# inventing a new judgment call. Nothing else in RETIRED_PATHS/
# RETIRED_IDENTIFIERS is listed here: an undocumented retired surface still
# fails the gate exactly as before (see check_report()/main()).
ACCEPTED_RESIDUALS: tuple[AcceptedResidual, ...] = (
    AcceptedResidual(
        relative="agent_utilities/exceptions" + ".py",
        needle=None,
        owner="BUG-032 / GOC-59",
        reason=(
            "Back-compat re-export shim over agent_utilities.core.exceptions, "
            "kept alive by a live cross-repo importer this repo does not own "
            "(agents/microsoft-agent/tests/test_auth_coverage.py:5). Migrating "
            "that one caller is a separate, cross-repo change."
        ),
    ),
    AcceptedResidual(
        relative="docs/concepts.yaml",
        needle="agent_utilities/mcp_" + "utilities.py",
        owner="BUG-032 / GOC-59",
        reason=(
            "docs/concepts.yaml is GENERATED by scripts/build_concepts_yaml.py "
            "from the CONCEPT: markers in the tree, so it necessarily lists the "
            "code_paths of the accepted-residual module above. A generated "
            "reference inherits the acceptance of what it references -- it is "
            "not independently fixable, and hand-editing the file is forbidden "
            "by its own header. Clears automatically when mcp_utilities.py is "
            "finally deleted."
        ),
    ),
    AcceptedResidual(
        relative="agent_utilities/mcp_" + "utilities.py",
        needle=None,
        owner="BUG-032 / GOC-59",
        reason=(
            "Back-compat re-export shim; deletion is blocked by 15+ live "
            "importers across 7+ agents/* repos plus universal-skills' "
            "scaffold_package.py, which emits an import of this module into "
            "every newly scaffolded package. A fleet-wide migration off it is "
            "a separate program."
        ),
    ),
    AcceptedResidual(
        relative="scripts/check_no_legacy_" + "markers.py",
        needle=None,
        owner="BUG-032 / GOC-59",
        reason=(
            "Referenced by the .pre-commit-config.yaml of all 61 agents/* "
            "packages plus agent-webui, agent-terminal-ui, and geniusbot; "
            "deleting it would break the fleet's gates wholesale."
        ),
    ),
    # The two entries below are the necessary, accurate consequence of
    # keeping the accepted-residual script named directly above this comment
    # (split so this comment itself does not trip the needle it discusses):
    # a fast-tier forwarder's TARGET constant and a docstring naming its
    # canonical location. Treating the file as accepted but these literal,
    # correct references to it as violations would leave the gate permanently red
    # for naming the file correctly.
    AcceptedResidual(
        relative="scripts/security/check_no_legacy_markers_gate.py",
        needle="check_no_legacy_" + "markers.py",
        owner="BUG-032 / GOC-59",
        reason=(
            "Fast-tier forwarder TARGET constant that must literally name the "
            "accepted-residual script above to invoke it."
        ),
    ),
    AcceptedResidual(
        relative="scripts/security_contract.py",
        needle="check_no_legacy_" + "markers.py",
        owner="BUG-032 / GOC-59",
        reason=(
            "Docstring names the accepted-residual script above as the "
            "worked example of the locate-au-and-run pattern every fleet "
            "consumer reaches it through."
        ),
    ),
    AcceptedResidual(
        relative="docs/operations/phase10-cutover-runbook.md",
        needle="GRAPH_" + "BACKEND",
        owner="GOC-59",
        reason=(
            "Dated, point-in-time runbook intentionally names this retired "
            "key as evidence of what was found live and as the exact "
            "guard-rail command to detect its reappearance (see the file's "
            "own 2026-08-09 accepted-residual banner)."
        ),
    ),
    AcceptedResidual(
        relative="docs/operations/phase10-cutover-runbook.md",
        needle="ENGINE_" + "MODE",
        owner="GOC-59",
        reason=(
            "Same runbook, documenting a retired key confirmed present on "
            "the graph-os-host drifted twin so the cutover can detect and "
            "strip it; see the file's own 2026-08-09 accepted-residual banner."
        ),
    ),
    AcceptedResidual(
        relative="docs/operations/phase10-cutover-runbook.md",
        needle="ENGINE_" + "ENDPOINT",
        owner="GOC-59",
        reason=(
            "Same runbook, same drifted-twin evidence as the retired engine-"
            "mode key above; see the file's own 2026-08-09 accepted-residual "
            "banner."
        ),
    ),
    AcceptedResidual(
        relative="docs/operations/phase10-cutover-runbook.md",
        needle="GRAPH_SERVICE_TCP_" + "ADDR",
        owner="GOC-59",
        reason=(
            "Same runbook, same drifted-twin evidence as the retired engine-"
            "mode key above; see the file's own 2026-08-09 accepted-residual "
            "banner."
        ),
    ),
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


# D-CIM-5: this check used to shell out to ``rg`` (ripgrep) for both file
# discovery (``_iter_files``) and the bulk content scan (a since-removed
# ``_check_repository_with_rg``), with the docstring's rationale that "WSL
# workspaces on NTFS make thousands of Python ``read_text`` calls
# disproportionately expensive." Neither path degraded gracefully when the
# binary was absent: ``subprocess.run([..., "rg", ...], check=True, ...)``
# raises a raw, unhandled ``FileNotFoundError`` straight out of ``main()`` —
# exactly the "degraded read explodes instead of refusing" anti-pattern this
# project codifies against. This environment has no ``rg`` installed and
# installing one is out of scope for a governance script.
#
# Decision: remove the ``rg`` dependency rather than declare the binary a
# hard requirement. Two reasons beyond portability: (1) the described
# NTFS/WSL slowdown does not apply to this deployment target; (2) auditing
# the two implementations for this fix found they had already DRIFTED —
# the ripgrep path additionally ran the ``PATH_REQUIRED_IDENTIFIERS`` check
# (asserting a current surface is present, not just that a retired one is
# absent) that the plain-Python ``check()`` path silently lacked, so calling
# ``check(root, paths=[...])`` directly (as every existing test in
# ``tests/gates/test_current_only_contract_gate.py`` already does) was
# ALREADY exercising an incomplete contract even before ``rg`` went missing
# here. A single implementation cannot drift from itself. ``_iter_files``
# now walks ``SCAN_ROOTS`` with ``Path.rglob`` instead of ``rg --files``, and
# ``check()`` runs ``PATH_REQUIRED_IDENTIFIERS`` (scoped to the real
# repository root, exactly as the removed ripgrep path scoped it, so the
# tmp_path-based unit tests are unaffected).
#
# BUG-043 follow-up: ``rg --files`` respected ``.gitignore`` by construction,
# so the ``Path.rglob`` replacement above silently NARROWED what this gate is
# safe against — a raw filesystem walk over ``_SKIP_DIR_NAMES`` alone (no
# ``.venv``, ``node_modules``, ``build``, ``dist``, ``target``,
# ``target-isolated``, ...) can pick up a retired identifier surviving in
# gitignored, generated build output and flag it as if it were live source,
# or — the opposite and equally real failure — miss a retired identifier
# that DOES live in tracked source but happens to sit inside a name not on
# the hand-maintained skip list. ``_iter_files`` now prefers the git-tracked
# file set (matching what actually ships/reviews), falling back to the
# ``_SKIP_DIR_NAMES``-filtered walk only when a scan root is not inside a
# git working tree (e.g. the ``tmp_path``-based unit tests).
_SKIP_DIR_NAMES = frozenset(
    {"__pycache__", ".git", ".mypy_cache", ".pytest_cache", ".ruff_cache"}
)


def _tracked_or_walked(scan_root: Path) -> list[Path]:
    return tracked_or_walked(scan_root, root=ROOT)


def _iter_files() -> list[Path]:
    files: set[Path] = {path for path in SCAN_FILES if path.is_file()}
    for scan_root in SCAN_ROOTS:
        if not scan_root.exists():
            continue
        for path in _tracked_or_walked(scan_root):
            if any(part in _SKIP_DIR_NAMES for part in path.parts):
                continue
            if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
                files.add(path)
    return sorted(files)


class ContractReport(NamedTuple):
    """The two populations D-MQR-11 requires be kept visibly separate."""

    new: list[str]
    accepted: list[str]


def _accepted(relative: str, needle: str | None) -> AcceptedResidual | None:
    for residual in ACCEPTED_RESIDUALS:
        if residual.matches(relative, needle):
            return residual
    return None


def check_report(
    root: Path = ROOT, *, paths: Iterable[Path] | None = None
) -> ContractReport:
    new: list[str] = []
    accepted: list[str] = []
    needles = RETIRED_IDENTIFIERS + RAW_ROUTE_FRAGMENTS
    inspected = _iter_files() if paths is None else sorted(set(paths))
    for path in inspected:
        relative = path.relative_to(root).as_posix()
        if relative in RETIRED_PATHS:
            message = f"{relative}: retired path exists"
            residual = _accepted(relative, None)
            (accepted if residual is not None else new).append(
                residual.describe(message) if residual is not None else message
            )
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as exc:
            new.append(
                f"{path.relative_to(root)}: could not inspect ({type(exc).__name__})"
            )
            continue
        for line_number, line in enumerate(lines, start=1):
            for needle in needles:
                if needle in line:
                    if path == ROOT / "README.md" and line == _README_RETIRED_KEY_LINE:
                        continue
                    message = (
                        f"{path.relative_to(root)}:{line_number}: "
                        f"retired surface {needle!r}"
                    )
                    residual = _accepted(relative, needle)
                    (accepted if residual is not None else new).append(
                        residual.describe(message) if residual is not None else message
                    )
            for retired_path, path_needle in PATH_RETIRED_IDENTIFIERS:
                if relative == retired_path and path_needle in line:
                    message = (
                        f"{relative}:{line_number}: retired surface {path_needle!r}"
                    )
                    residual = _accepted(relative, path_needle)
                    (accepted if residual is not None else new).append(
                        residual.describe(message) if residual is not None else message
                    )
    if root == ROOT:
        # Scoped to the real repository root only (not a tmp_path fixture,
        # which cannot contain these real repo-relative files) -- matches
        # how the removed ripgrep path scoped this same check.
        for relative, needle in PATH_REQUIRED_IDENTIFIERS:
            path = ROOT / relative
            if not path.is_file():
                new.append(f"{relative}: required current surface is missing")
                continue
            if needle not in path.read_text(encoding="utf-8"):
                new.append(
                    f"{relative}: required current surface {needle!r} is missing"
                )
    return ContractReport(new=new, accepted=accepted)


def check(root: Path = ROOT, *, paths: Iterable[Path] | None = None) -> list[str]:
    """Backward-compatible surface: genuinely NEW (non-accepted) violations
    only -- this is what drives ``main()``'s exit code. An accepted residual
    (see ``ACCEPTED_RESIDUALS``) never appears here; use ``check_report()``
    to see both populations."""
    return check_report(root, paths=paths).new


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="check-current-only-contract")
    parser.add_argument(
        "--new-only",
        action="store_true",
        help=(
            "Emit ONLY blocking (new) violations. Suppresses the accepted-residual "
            "listing and the summary line entirely. This is the machine-readable "
            "mode the merge queue's differential contract gate consumes: that gate "
            "diffs the check's COMBINED output between the base ref and the "
            "candidate, so any carried-residual text -- even though it is "
            "non-blocking and goes to stderr -- makes a candidate that legitimately "
            "regenerates a file (e.g. docs/concepts.yaml) look like it introduced a "
            "NEW violation. Humans running the gate directly still see everything."
        ),
    )
    args = parser.parse_args(argv)
    report = check_report()
    if not args.new_only:
        if report.accepted:
            print(
                f"Current-only contract: {len(report.accepted)} accepted residual(s) "
                "carried (documented rationale + owner; non-blocking):",
                file=sys.stderr,
            )
            for item in report.accepted:
                print(f"  - {item}", file=sys.stderr)
    if report.new:
        print("Current-only contract violations:", file=sys.stderr)
        for violation in report.new:
            print(f"  - {violation}", file=sys.stderr)
    if not args.new_only:
        print(
            f"Current-only contract: {len(report.accepted)} accepted residual(s) "
            f"carried; {len(report.new)} new violation(s)"
        )
    if report.new:
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
