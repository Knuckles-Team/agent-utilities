#!/usr/bin/env python3
"""Fail when a removed compatibility surface returns to shipped artifacts.

The project has one current contract. This gate intentionally scans runtime code,
tests, documentation, deployment assets, and repository guidance so a deleted
environment switch, alias, or raw graph endpoint cannot survive on a secondary
surface after its implementation is removed.

WD10-R-RESIDZERO: the ``ACCEPTED_RESIDUALS`` allowlist mechanism this gate
used to carry (D-MQR-11, BUG-032/GOC-59 shape: a typed registry of specific
``relative``/``needle`` pairs, each exempted from failing and printed instead
as carried, non-blocking INFO) is retired. It was an enumeration -- a list of
paths and needles that happened to be failing on the day each entry was
added, with no rule that told a reader what else belonged in it. Measured on
merged main, every remaining entry pointed at ONE file:
``docs/operations/phase10-cutover-runbook.md``, a dated, point-in-time
incident runbook that intentionally names retired configuration keys as
evidence of what was found live and as the exact detection command for their
reappearance. Deleting those names from the runbook would destroy its audit
value without changing anything the gate actually protects against, but a
path-keyed allowlist entry is still a ratchet: silent, unbounded, and blind
to *why* an exemption exists.

The replacement is a category, not a list: ``_is_dated_historical_record``
below exempts a file from retired-surface IDENTIFIER scanning only when BOTH
(a) it lives under ``docs/`` -- this gate's purpose is live configuration and
runtime-surface drift, and documentation is categorically not that -- AND
(b) it carries ``DATED_HISTORICAL_RECORD_MARKER`` as an intrinsic,
machine-readable, human-visible declaration near its own top. The marker
lives in the document itself, so a reviewer sees it in the same diff that
adds a retired name, and a renamed or copied copy of the document keeps the
exemption; a path enumerated in this script instead would silently stop
covering it. The exemption is narrow on purpose: it skips identifier/text
matching only, never ``RETIRED_PATHS`` (a retired path actually reappearing
still fails) or ``PATH_REQUIRED_IDENTIFIERS`` (a required current surface
going missing still fails) -- both of those test something a documentation
marker cannot legitimately excuse. There is no "carried, non-blocking INFO"
population anymore: a file either matches the category and is out of scope,
or it does not and a match fails loudly, exactly like every other retired
surface.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _git_subprocess_env import strip_inherited_git_repository_env  # noqa: E402

# BUG-180: `_tracked_or_walked`'s `git -C <scan_root> ls-files` call below
# inherits a real `git commit`/`git push`'s GIT_DIR/GIT_INDEX_FILE otherwise,
# which win over `-C`'s path and silently redirect it to the wrong
# repository/index -- the same class as D-LGI-1, but in a script invoked
# directly as its own git hook rather than under pytest (where
# tests/conftest.py already protects the session). Strip once, at import
# time, before any subprocess call in this process.
strip_inherited_git_repository_env()

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
    # NOT "GRAPH_SERVICE_TCP_ADDR" -- 2026-08-25: commit 2e399171e legitimately
    # reactivated this exact name for a new, unrelated meaning (the
    # auto-started engine child's own --tcp-addr flag; see the matching
    # removal + comment in agent_utilities/core/config.py's
    # _RETIRED_CONFIGURATION_KEYS). Re-adding it here would fail this gate on
    # every live reference in graph_compute.py/.env.example/runtime-
    # configuration.md.
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
    # WD10-P-AUPUSH: the back-compat re-export shim over
    # ``agent_utilities.core.exceptions`` this path used to name is DELETED
    # (its sole external importer, agents/microsoft-agent/tests/
    # test_auth_coverage.py:5, was migrated onto
    # ``agent_utilities.core.exceptions`` directly first). No longer an
    # ACCEPTED_RESIDUALS entry -- this is now a plain tripwire against the
    # shim reappearing, same as every other RETIRED_PATHS entry.
    "agent_utilities/exceptions" + ".py",
    "agent_utilities/graph/" + "steps.py",
    "agent_utilities/knowledge_graph/core/ingest_" + "engine.py",
    # WD10-P-AUPUSH: the back-compat re-export shim this path used to name
    # is DELETED. Its 12 live importers (aris-mcp, firefly-iii-mcp x1,
    # freshrss-agent x4, hdhomerun-mcp x6) were migrated onto the canonical
    # submodules it forwarded to, and universal-skills' scaffold_package.py
    # no longer emits an import of it into new packages. No longer an
    # ACCEPTED_RESIDUALS entry -- this is now a plain tripwire against the
    # shim reappearing, same as every other RETIRED_PATHS entry.
    "agent_utilities/mcp_" + "utilities.py",
    "agent_utilities/mcp/kg_" + "coordinator.py",
    "scripts/apply_concept_" + "migration.py",
    "scripts/autocurate_" + "repo.py",
    # WD10-P-AUPUSH: scripts/check_no_legacy_markers.py was WRONGLY listed
    # here. It is not retired surface -- it is a live gate the
    # `.pre-commit-config.yaml` of all 61 `agents/*` packages plus
    # agent-webui, agent-terminal-ui, geniusbot, and the scaffolder still
    # invoke by this exact name today. The RETIRED_PATHS/RETIRED_IDENTIFIERS
    # contract exists to catch surface that was REMOVED and should stay
    # removed; a file 65 repos actively depend on does not meet that
    # definition regardless of what its name contains. Fixed the contract
    # (removed the entry, here and in RETIRED_IDENTIFIERS above, plus the
    # 3 ACCEPTED_RESIDUALS entries this false positive required) rather
    # than migrating or deleting anything -- there is nothing to migrate.
    "scripts/consolidate_" + "concepts.py",
    "scripts/curate_" + "batches.py",
    "scripts/inject_concept_" + "ids.py",
    "scripts/migrate_concepts_" + "hierarchy.py",
    "scripts/plan_concept_" + "migration.py",
    "scripts/reserve_concepts_" + "hook.py",
    "tests/unit/mcp/test_kg_" + "coordinator.py",
    "tests/unit/test_kg_" + "coordinator_client_role.py",
)


# A dated historical record (an incident runbook, a postmortem, or similar
# point-in-time document) that legitimately names retired configuration
# surface as evidence of what was found live -- and as the exact detection
# command for its reappearance -- declares that fact about itself with this
# exact marker, near its own top. This is the category rule in code: any
# ``docs/`` file carrying it, present or future, whatever its filename, is
# exempt from retired-surface IDENTIFIER scanning (see
# ``_is_dated_historical_record`` below). It is deliberately NOT a path or a
# needle -- a rename or a copy of the document keeps the marker and stays
# covered, unlike an entry in a list here.
DATED_HISTORICAL_RECORD_MARKER = "CURRENT-ONLY-CONTRACT: DATED-HISTORICAL-RECORD"
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
    try:
        out = subprocess.run(
            ["git", "-C", str(scan_root), "ls-files"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        tracked = [scan_root / line for line in out.splitlines() if line]
        if tracked:
            return [p for p in tracked if p.is_file()]
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return [
        p
        for p in scan_root.rglob("*")
        if p.is_file() and not any(part in _SKIP_DIR_NAMES for part in p.parts)
    ]


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
    """Kept as a NamedTuple (rather than a bare list) so the shape is stable
    for callers that unpack it, even though there is now only one
    population: every retired-surface match this gate finds is a violation.
    There is no second, carried/non-blocking bucket any more -- that was the
    ``ACCEPTED_RESIDUALS`` mechanism, removed by WD10-R-RESIDZERO (see the
    module docstring)."""

    new: list[str]


def _is_dated_historical_record(relative: str, lines: list[str]) -> bool:
    """Category rule, not an enumeration: a ``docs/`` file that declares
    itself a dated historical record, via ``DATED_HISTORICAL_RECORD_MARKER``
    near its own top, is a point-in-time incident/runbook document that may
    legitimately name retired configuration surface as evidence -- not live
    configuration this gate exists to police. Only ``docs/`` files qualify
    (this gate's purpose is runtime/config surface, and a marker inside
    runtime code should never be able to buy an exemption); only the marker
    -- not the path, not the filename -- decides, so this covers any current
    or future file of the same kind without naming one."""

    if not relative.startswith("docs/"):
        return False
    return any(DATED_HISTORICAL_RECORD_MARKER in line for line in lines[:10])


_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_]+$")
_IDENTIFIER_NEEDLE_PATTERN: dict[str, re.Pattern[str]] = {
    needle: re.compile(r"\b" + re.escape(needle) + r"\b")
    for needle in RETIRED_IDENTIFIERS
    if _IDENTIFIER_RE.match(needle)
}

# Check every line with two compiled alternations instead of running one regex
# or substring search per needle.  At the current tree size the former
# O(files x lines x needles) loop performs roughly 350 million Python-level
# matches and cannot finish inside the merge queue's 55-second forwarder
# budget. Sorting longest-first, together with the surrounding word
# boundaries, ensures an extended current identifier does not also match its
# shorter retired base name; this preserves `_needle_matches`.
_ALL_NEEDLES = RETIRED_IDENTIFIERS + RAW_ROUTE_FRAGMENTS
_NEEDLE_ORDER = {needle: index for index, needle in enumerate(_ALL_NEEDLES)}
_IDENTIFIER_ALTERNATION = re.compile(
    r"\b(?:"
    + "|".join(
        re.escape(needle)
        for needle in sorted(_IDENTIFIER_NEEDLE_PATTERN, key=len, reverse=True)
    )
    + r")\b"
)
_LITERAL_NEEDLES = tuple(
    needle for needle in _ALL_NEEDLES if needle not in _IDENTIFIER_NEEDLE_PATTERN
)
_LITERAL_ALTERNATION = re.compile(
    "|".join(
        re.escape(needle) for needle in sorted(_LITERAL_NEEDLES, key=len, reverse=True)
    )
)


def _needle_matches(needle: str, line: str) -> bool:
    """Plain substring, except a pure-identifier needle (only
    ``[A-Za-z0-9_]``) requires a word boundary on both sides.

    Without this, a retired bare identifier also matches as a substring of
    an unrelated, CURRENT identifier that merely contains it as a suffix
    (a longer, differently-prefixed env var name) or is a same-stem helper
    with an extra prefix/infix word -- neither carries the retired meaning.
    (Concretely, this closed two 2026-08-28 false positives: a retired bare
    config key matching inside an unrelated, differently-prefixed env var
    name that happens to end the same way, and a retired bare helper name
    matching inside an unrelated function whose name happens to contain it
    as a middle segment -- see the wD9-CIGATE report for the exact
    identifiers and files.) ``\\b`` does not insert a boundary between ``_``
    and a letter/digit (both are word characters), so a same-suffix
    differently-prefixed name still correctly does NOT match, while a real
    bare/quoted occurrence of the exact retired token still does. Needles
    that already embed non-identifier punctuation (quotes, ``=``, ...) are
    unaffected and keep the original plain-substring check.
    """

    pattern = _IDENTIFIER_NEEDLE_PATTERN.get(needle)
    if pattern is not None:
        return pattern.search(line) is not None
    return needle in line


def _matching_needles(line: str) -> list[str]:
    """Return each matching needle once, in the legacy declaration order."""
    matches = {match.group(0) for match in _IDENTIFIER_ALTERNATION.finditer(line)}
    matches.update(match.group(0) for match in _LITERAL_ALTERNATION.finditer(line))
    return sorted(matches, key=_NEEDLE_ORDER.__getitem__)


def check_report(
    root: Path = ROOT, *, paths: Iterable[Path] | None = None
) -> ContractReport:
    new: list[str] = []
    inspected = _iter_files() if paths is None else sorted(set(paths))
    for path in inspected:
        relative = path.relative_to(root).as_posix()
        if relative in RETIRED_PATHS:
            # The dated-historical-record exemption never applies here: a
            # retired PATH actually reappearing on disk is a fact about the
            # tree, not a documentation choice, so no marker can excuse it.
            new.append(f"{relative}: retired path exists")
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as exc:
            new.append(
                f"{path.relative_to(root)}: could not inspect ({type(exc).__name__})"
            )
            continue
        if _is_dated_historical_record(relative, lines):
            continue
        for line_number, line in enumerate(lines, start=1):
            for needle in _matching_needles(line):
                if path == ROOT / "README.md" and line == _README_RETIRED_KEY_LINE:
                    continue
                new.append(
                    f"{path.relative_to(root)}:{line_number}: "
                    f"retired surface {needle!r}"
                )
            for retired_path, path_needle in PATH_RETIRED_IDENTIFIERS:
                if relative == retired_path and path_needle in line:
                    new.append(
                        f"{relative}:{line_number}: retired surface {path_needle!r}"
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
    return ContractReport(new=new)


def check(root: Path = ROOT, *, paths: Iterable[Path] | None = None) -> list[str]:
    """Every violation this gate finds -- this is what drives ``main()``'s
    exit code."""
    return check_report(root, paths=paths).new


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="check-current-only-contract")
    parser.add_argument(
        "--new-only",
        action="store_true",
        help=(
            "Emit ONLY the violation listing, no summary line. Kept for the merge "
            "queue's differential contract gate, which diffs the check's COMBINED "
            "output between the base ref and the candidate -- the summary line's "
            "count would otherwise itself look like a diff. Humans running the gate "
            "directly still see the summary."
        ),
    )
    args = parser.parse_args(argv)
    report = check_report()
    if report.new:
        print("Current-only contract violations:", file=sys.stderr)
        for violation in report.new:
            print(f"  - {violation}", file=sys.stderr)
    if not args.new_only:
        print(f"Current-only contract: {len(report.new)} violation(s)")
    if report.new:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
