"""Auto-extracted graph-os MCP tools: analysis_tools (register_analysis_tools).

Split out of kg_server._build_server to deepen the MCP surface into focused
modules without changing tool behavior or names.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from pydantic import Field

from agent_utilities.mcp import kg_server
from agent_utilities.models.evidence_bundle import EvidenceBundle
from agent_utilities.orchestration.response_format import (
    ResponseFormat,
    validate_response_format,
)
from agent_utilities.security.error_surface import public_error_json, public_error_text

logger = logging.getLogger(__name__)


_MCP_REGISTRATION_MAX_BYTES = 128 * 1024
_MCP_CONFIG_MAX_BYTES = 1024 * 1024
_MCP_REGISTRATION_MAX_ITEMS = 2048
_MCP_REGISTRATION_MAX_DEPTH = 16
_MCP_SERVER_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_RUNTIME_REFERENCE_RE = re.compile(
    r"^(?:\$\{[A-Za-z_][A-Za-z0-9_]*\}|(?:vault|env|secret)://[A-Za-z0-9_./#-]+)$"
)
_URI_LITERAL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_WINDOWS_ABSOLUTE_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")
_INLINE_SECRET_RE = re.compile(
    r"(?i)^(?:bearer\s+\S+|basic\s+\S+|sk-[A-Za-z0-9_-]{8,}|gh[pousr]_[A-Za-z0-9_-]{8,})$"
)
_EMAIL_LITERAL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
_SENSITIVE_MCP_KEY_PARTS = frozenset(
    {
        "authorization",
        "credential",
        "credentials",
        "email",
        "identity",
        "password",
        "secret",
        "tenant",
        "token",
        "user",
        "username",
    }
)
_ENDPOINT_MCP_KEY_PARTS = frozenset(
    {
        "address",
        "baseurl",
        "broker",
        "brokers",
        "endpoint",
        "endpoints",
        "host",
        "hostname",
        "hosts",
        "server",
        "servers",
        "uri",
        "uris",
        "url",
        "urls",
    }
)
_PATH_MCP_KEY_PARTS = frozenset(
    {
        "bundle",
        "ca",
        "cert",
        "certificate",
        "cwd",
        "directory",
        "dir",
        "file",
        "keyfile",
        "path",
        "root",
        "workspace",
    }
)
_SAFE_INLINE_MCP_ENV_KEYS = frozenset(
    {
        "APP_PROFILE",
        "FASTMCP_LOG_LEVEL",
        "LOG_LEVEL",
        "MCP_CLIENT_AUTH",
        "MCP_TOOL_MODE",
        "NO_COLOR",
        "PYTHONUNBUFFERED",
        "UV_NATIVE_TLS",
    }
)
_SAFE_INLINE_MCP_ENV_VALUE_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_analysis_core_handler: Any | None = None
_EXTERNAL_MAPPING_POLICY_FIELDS = frozenset(
    {
        "access",
        "edge_property_allowlist",
        "edge_type_overrides",
        "identity_property",
        "property_allowlist",
        "type_overrides",
    }
)
_EXTERNAL_MAPPING_POLICY_MAX_BYTES = 4 * 1024 * 1024


def _reject_nonfinite_json(_value: str) -> None:
    raise ValueError("non-finite JSON constants are not supported")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON keys are not supported")
        value[key] = item
    return value


def _property_graph_sync_policy(declaration: dict[str, Any]) -> dict[str, Any]:
    """Return the bounded, non-secret sync policy bound into mapping approval."""

    def bounded(name: str, default: int, lower: int, upper: int) -> int:
        value = declaration.get(name, default)
        if isinstance(value, bool):
            raise ValueError("external graph sync bounds must be integers")
        parsed = int(value)
        if not lower <= parsed <= upper:
            raise ValueError("external graph sync bounds are out of range")
        return parsed

    mode = str(declaration.get("sync_mode") or "auto")
    if mode not in {"auto", "cdc", "snapshot"}:
        raise ValueError("external graph sync mode is invalid")
    reconcile = declaration.get("reconcile_deletions", True)
    allow_empty = declaration.get("allow_empty_snapshot", False)
    if not isinstance(reconcile, bool) or not isinstance(allow_empty, bool):
        raise ValueError("external graph reconciliation policy must be boolean")
    row_bytes = bounded("ingest_max_row_bytes", 1_048_576, 256, 8_388_608)
    total_bytes = bounded("ingest_max_total_bytes", 16_777_216, 256, 67_108_864)
    if total_bytes < row_bytes:
        raise ValueError("external graph total byte bound must cover one row")
    return {
        "allow_empty_snapshot": allow_empty,
        "max_collection_items": bounded(
            "ingest_max_collection_items", 10_000, 1, 100_000
        ),
        "max_nesting_depth": bounded("ingest_max_nesting_depth", 16, 1, 64),
        "max_pages": bounded("ingest_max_pages", 100, 1, 1_000),
        "max_row_bytes": row_bytes,
        "max_total_bytes": total_bytes,
        "page_size": bounded("ingest_page_size", 500, 1, 1_000),
        "reconcile_deletions": reconcile,
        "sync_mode": mode,
    }


def _configured_external_graph_declaration(name: str) -> dict[str, Any]:
    """Resolve one reference-only declaration without returning it publicly."""

    from agent_utilities.core.config import config as runtime_config

    selected: dict[str, Any] = {}
    collections = (
        getattr(runtime_config, "external_graph_connectors", []) or [],
        getattr(runtime_config, "kg_connections", []) or [],
    )
    for declarations in collections:
        for configured in declarations:
            candidate = (
                configured.model_dump(exclude_defaults=True)
                if hasattr(configured, "model_dump")
                else dict(configured)
                if isinstance(configured, dict)
                else {}
            )
            if str(candidate.get("name") or "") == name:
                selected = candidate
    return selected


def _resolved_external_mapping_policy(
    store: Any, declaration: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    """Resolve, validate, and hash a property-graph policy behind its ref."""

    policy_ref = str(declaration.get("mapping_policy_ref") or "")
    policy: dict[str, Any] = {}
    if policy_ref:
        raw = store.resolve_ref(policy_ref)
        if (
            not isinstance(raw, str)
            or not raw
            or len(raw.encode("utf-8")) > _EXTERNAL_MAPPING_POLICY_MAX_BYTES
        ):
            raise ValueError("external mapping policy is missing or exceeds its bound")

        parsed = json.loads(
            raw,
            parse_constant=_reject_nonfinite_json,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
        if not isinstance(parsed, dict):
            raise ValueError("external mapping policy must be an object")
        policy = parsed
    if set(policy).difference(_EXTERNAL_MAPPING_POLICY_FIELDS):
        raise ValueError("external mapping policy contains unsupported inline material")
    for field in ("access", "type_overrides", "edge_type_overrides"):
        if field in policy and not isinstance(policy[field], dict):
            raise ValueError("external mapping policy has an invalid mapping field")
    access = policy.get("access")
    if isinstance(access, dict):
        if set(access).difference({"group_ids", "is_public", "markings"}):
            raise ValueError(
                "external mapping policy access cannot contain inline identities"
            )
        if "is_public" in access and not isinstance(access["is_public"], bool):
            raise ValueError("external mapping policy access is invalid")
        for field in ("group_ids", "markings"):
            value = access.get(field)
            if value is not None and (
                not isinstance(value, list)
                or any(not isinstance(item, str) for item in value)
            ):
                raise ValueError("external mapping policy access is invalid")
    for field in ("type_overrides", "edge_type_overrides"):
        value = policy.get(field)
        if isinstance(value, dict) and any(
            not isinstance(key, str) or not isinstance(item, str)
            for key, item in value.items()
        ):
            raise ValueError("external mapping policy has an invalid type mapping")
    for field in ("property_allowlist", "edge_property_allowlist"):
        value = policy.get(field)
        if value is not None and (
            not isinstance(value, list)
            or any(not isinstance(item, str) for item in value)
        ):
            raise ValueError("external mapping policy has an invalid allowlist")
    if "identity_property" in policy and not isinstance(
        policy["identity_property"], str
    ):
        raise ValueError("external mapping policy has an invalid identity property")

    from agent_utilities.knowledge_graph.ingestion.external_graph_schema import (
        external_mapping_policy_digest,
    )

    approval_policy = {
        **policy,
        "sync": _property_graph_sync_policy(declaration),
    }
    return policy, external_mapping_policy_digest(approval_policy)


async def execute_focused_analysis(
    action: str,
    query: str = "",
    top_k: int = 10,
    node_id: str = "",
    depth: int = 2,
    target: str = "",
) -> EvidenceBundle:
    """Execute one canonical focused action through the shared analysis kernel."""

    if _analysis_core_handler is None:
        raise RuntimeError("analysis tools are not registered")
    raw = await _analysis_core_handler(
        action=action,
        query=query,
        top_k=top_k,
        node_id=node_id,
        depth=depth,
        target=target,
    )
    return EvidenceBundle.from_payload(raw, operation=action)


def _configuration_key_is_sensitive(
    env_key: str, metadata: dict[str, Any] | None = None
) -> bool:
    """Classify durable settings whose values must not cross the MCP surface."""

    parts = _normalised_key_parts(env_key)
    if bool((metadata or {}).get("secret")):
        return True
    if parts & (
        _SENSITIVE_MCP_KEY_PARTS | _ENDPOINT_MCP_KEY_PARTS | _PATH_MCP_KEY_PARTS
    ):
        return True
    if env_key.upper() == "MCP_CONFIG":
        return True
    if "id" in parts and parts & {
        "actor",
        "agent",
        "client",
        "identity",
        "tenant",
        "user",
    }:
        return True
    return "key" in parts and bool(
        parts
        & {"api", "auth", "client", "encryption", "hmac", "private", "signing", "tls"}
    )


def _runtime_reference(value: Any) -> bool:
    """Return whether ``value`` is a non-literal runtime reference.

    MCP clients already expand ``${VAR}`` placeholders.  Secret-store URI refs
    are also accepted for child servers which resolve them themselves.  Defaults
    inside placeholders are intentionally rejected: they put the supposedly
    external value back into the durable JSON document.
    """

    return isinstance(value, str) and bool(_RUNTIME_REFERENCE_RE.fullmatch(value))


def _normalised_key_parts(key: str) -> set[str]:
    normalised = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key).lower()
    parts = {part for part in re.split(r"[^a-z0-9]+", normalised) if part}
    if "base" in parts and "url" in parts:
        parts.add("baseurl")
    if "key" in parts and "file" in parts:
        parts.add("keyfile")
    return parts


def _validate_mcp_server_definition(definition: Any) -> dict[str, Any]:
    """Validate one durable MCP server declaration without retaining literals.

    Commands and neutral feature flags remain inline.  Endpoint, credential,
    identity, TLS/path material, and values which visibly look like credentials
    or PII must be runtime references.  The recursive bounds also prevent a
    small administrative request from causing excessive parsing/walk work.
    """

    if not isinstance(definition, dict):
        raise ValueError("MCP server definition must be an object")

    args = definition.get("args")
    if args is not None and not isinstance(args, list):
        raise ValueError("MCP server args must be a list")
    expects_reference = False
    for argument in args or []:
        if not isinstance(argument, str):
            raise ValueError("MCP server arguments must be strings")
        if expects_reference:
            if not _runtime_reference(argument):
                raise ValueError(
                    "sensitive MCP command arguments must be runtime references"
                )
            expects_reference = False
            continue
        if not argument.startswith("-"):
            continue
        option, separator, inline_value = argument.partition("=")
        option_name = option.lstrip("-")
        option_sensitive = _configuration_key_is_sensitive(option_name) or (
            option_name.lower()
            in {
                "config",
                "connection-string",
                "dsn",
                "env",
                "h",
                "header",
                "proxy",
            }
        )
        if option_sensitive and separator:
            if not _runtime_reference(inline_value):
                raise ValueError(
                    "sensitive MCP command arguments must be runtime references"
                )
        elif option_sensitive:
            expects_reference = True
    if expects_reference:
        raise ValueError("sensitive MCP command argument is missing its value")

    seen = 0

    def _walk(
        value: Any, *, key: str = "", depth: int = 0, in_env: bool = False
    ) -> None:
        nonlocal seen
        seen += 1
        if seen > _MCP_REGISTRATION_MAX_ITEMS:
            raise ValueError("MCP server definition is too large")
        if depth > _MCP_REGISTRATION_MAX_DEPTH:
            raise ValueError("MCP server definition is too deeply nested")

        parts = _normalised_key_parts(key)
        requires_reference = bool(
            parts
            & (_SENSITIVE_MCP_KEY_PARTS | _ENDPOINT_MCP_KEY_PARTS | _PATH_MCP_KEY_PARTS)
        )
        requires_reference = requires_reference or (
            "id" in parts
            and bool(parts & {"actor", "agent", "client", "identity", "tenant", "user"})
        )
        requires_reference = requires_reference or (
            "key" in parts
            and bool(
                parts
                & {
                    "api",
                    "auth",
                    "client",
                    "encryption",
                    "hmac",
                    "private",
                    "signing",
                    "tls",
                }
            )
        )
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                if not isinstance(child_key, str) or len(child_key) > 128:
                    raise ValueError("MCP server definition contains an invalid key")
                _walk(
                    child_value,
                    key=child_key,
                    depth=depth + 1,
                    in_env=in_env or key.lower() == "env",
                )
            return
        if isinstance(value, list):
            for child in value:
                _walk(child, key=key, depth=depth + 1, in_env=in_env)
            return
        if not isinstance(value, str) or not value:
            if requires_reference and value not in (None, ""):
                raise ValueError("sensitive MCP values must be runtime references")
            return
        if _runtime_reference(value):
            return
        if in_env and not (
            key.upper() in _SAFE_INLINE_MCP_ENV_KEYS
            and _SAFE_INLINE_MCP_ENV_VALUE_RE.fullmatch(value)
        ):
            raise ValueError(
                "MCP environment values must be runtime references unless the "
                "setting is a bounded non-sensitive mode"
            )
        if requires_reference:
            raise ValueError("sensitive MCP values must be runtime references")
        if (
            _URI_LITERAL_RE.match(value)
            or value.startswith(("/", "~/", "~\\"))
            or _WINDOWS_ABSOLUTE_PATH_RE.match(value)
            or _INLINE_SECRET_RE.match(value)
            or _EMAIL_LITERAL_RE.match(value)
        ):
            raise ValueError(
                "endpoint, credential, identity, and path literals are not "
                "durable MCP configuration"
            )

    _walk(definition)
    return definition


def _workspace_mcp_config_path() -> Path:
    """Resolve the active MCP config strictly beneath the declared workspace.

    Registration is a workspace-scoped write operation. It uses only ``MCP_CONFIG``
    (when set) or ``<workspace>/mcp_config.json`` and rejects traversal and every
    existing symlink component.
    """

    from agent_utilities.core.config import setting
    from agent_utilities.core.workspace import get_agent_workspace

    root = get_agent_workspace().resolve(strict=True)
    configured = str(setting("MCP_CONFIG", "") or "").strip()
    if configured and ("\x00" in configured or "$" in configured):
        raise ValueError("MCP_CONFIG must resolve before registration")
    if _WINDOWS_ABSOLUTE_PATH_RE.match(configured):
        raise PermissionError("MCP config must use the active workspace namespace")
    requested = Path(configured).expanduser() if configured else Path("mcp_config.json")
    if ".." in requested.parts:
        raise PermissionError("MCP config traversal is not permitted")
    candidate = requested if requested.is_absolute() else root / requested
    candidate = candidate.absolute()
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        raise PermissionError(
            "MCP config must be inside the active workspace"
        ) from None
    if candidate.suffix.lower() != ".json":
        raise ValueError("MCP config must be a JSON file")

    current = root
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise PermissionError("symlinked MCP config paths are not writable")
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError:
        raise PermissionError(
            "MCP config must resolve inside the active workspace"
        ) from None
    return candidate


def _read_mcp_config_document(path: Path) -> dict[str, Any]:
    """Read a bounded regular JSON file without following the final symlink."""

    import stat

    if path.is_symlink():
        raise PermissionError("symlinked MCP config files are not readable")
    if not path.exists():
        return {}
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("MCP config must be a regular file")
        with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
            descriptor = -1
            raw = stream.read(_MCP_CONFIG_MAX_BYTES + 1)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if len(raw.encode("utf-8")) > _MCP_CONFIG_MAX_BYTES:
        raise ValueError("MCP config exceeds the size limit")
    document = json.loads(raw) if raw.strip() else {}
    if not isinstance(document, dict):
        raise ValueError("MCP config must be a JSON object")
    servers = document.get("mcpServers", {})
    if not isinstance(servers, dict):
        raise ValueError("mcpServers must be a JSON object")
    return document


def _atomic_private_json_write(path: Path, document: dict[str, Any]) -> None:
    """Atomically replace ``path`` with a private regular JSON file."""

    import tempfile

    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.parent.is_symlink() or path.is_symlink():
        raise PermissionError("symlinked MCP config paths are not writable")
    descriptor, raw_tmp = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    tmp_path = Path(raw_tmp)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(document, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if path.parent.is_symlink() or path.is_symlink():
            raise PermissionError("symlinked MCP config paths are not writable")
        os.replace(tmp_path, path)
        try:
            path.chmod(0o600)
        except OSError:
            # The atomic temp file is already created private on POSIX.  Some
            # Windows filesystems do not implement POSIX mode changes.
            pass
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def _register_mcp_server(name: str, raw_definition: str) -> None:
    """Validate and atomically persist one named MCP server declaration."""

    if not _MCP_SERVER_NAME_RE.fullmatch((name or "").strip()):
        raise ValueError("invalid MCP server name")
    encoded = (raw_definition or "").encode("utf-8")
    if not encoded or len(encoded) > _MCP_REGISTRATION_MAX_BYTES:
        raise ValueError("MCP server definition has an invalid size")
    definition = _validate_mcp_server_definition(json.loads(raw_definition))
    path = _workspace_mcp_config_path()
    document = _read_mcp_config_document(path)
    servers = document.setdefault("mcpServers", {})
    for existing_name, existing_definition in servers.items():
        if not _MCP_SERVER_NAME_RE.fullmatch(str(existing_name)):
            raise ValueError("existing MCP configuration has an invalid server name")
        _validate_mcp_server_definition(existing_definition)
    servers[name.strip()] = definition
    _atomic_private_json_write(path, document)


logger = logging.getLogger(__name__)


def register_analysis_tools(mcp):
    """Register the analysis_tools group on the given FastMCP server."""

    async def _run_analysis_action(
        action: str = Field(
            default="inspect",
            description="Ops/structural action: inspect | enrichment_coverage | process_writeback | placement_plan | infra_sweep | security_scan. (Codebase→graph_code, research→graph_research, eval→graph_evaluate, Q&A→graph_explain, traces→graph_observe.)",
        ),
        query: str = Field(default="", description="Query or path for the analysis."),
        top_k: int = Field(
            default=10, description="Number of results or complexity budget."
        ),
        node_id: str = Field(
            default="",
            description="Specific node ID to analyze (e.g., for blast_radius).",
        ),
        depth: int = Field(
            default=2, description="Depth of traversal (e.g., for blast_radius)."
        ),
        target: str = Field(
            default="", description="Target for the analysis or inspection."
        ),
    ) -> str:
        """Execute complex analysis across the Knowledge Graph. Enables advanced semantic synthesis, causal dependency mapping, and structural inspection."""
        engine = kg_server._get_engine()
        if not engine:
            return "Error: IntelligenceGraphEngine not active."
        try:
            if action in (
                "synthesize",
                "deep_extract",
                "background_research",
                "relevance_sweep",
            ):
                job_id = engine.submit_task(
                    target_path=query or target or "none",
                    is_codebase=False,
                    task_type=action,
                    provenance={
                        "top_k": top_k,
                        "node_id": node_id,
                        "depth": depth,
                        "target": target,
                    },
                    skip_dedupe=True,
                )
                return f"Job submitted as '{job_id}'. Use graph_ingest(action='status', job_id='{job_id}') to check the result."
            elif action == "blast_radius":
                if not node_id:
                    return "Error: node_id required for blast_radius"
                radius = engine.get_blast_radius(node_id, depth)
                if not radius:
                    return f"No dependencies found for {node_id} within depth {depth}."
                return "\n".join(
                    [
                        f"[{n['node_type']}] {n['id']} (Depth: {n['depth']})"
                        for n in radius
                    ]
                )
            elif action == "inspect":
                # Structural/subgraph inspection (KG-2.134 docs): a node's own
                # properties + its immediate neighbors + degree. No such method
                # exists on IntelligenceGraphEngine — build the snapshot from
                # REAL, already-wired read primitives instead of inventing one:
                # ``query_cypher`` (parameterized — never f-string the target
                # into Cypher) for properties, falling back to the bounded
                # single-node reader, plus ``graph_compute`` for O(1) neighbor/
                # degree lookups (never a whole-graph scan).
                import json as _json

                ident = (target or query or node_id or "").strip()
                if not ident:
                    return "Error: target (or query/node_id) required for inspect"

                props: dict[str, Any] = {}
                try:
                    rows = engine.query_cypher(
                        "MATCH (n {id: $ident}) RETURN n AS node, labels(n) AS labels LIMIT 1",
                        {"ident": ident},
                    )
                except Exception as e:  # noqa: BLE001 — fall back below
                    logger.warning(
                        "inspect: query_cypher lookup failed for %s (exception_type=%s)",
                        ident,
                        type(e).__name__,
                    )
                    rows = None
                if rows:
                    node = rows[0].get("node")
                    if isinstance(node, dict):
                        props = dict(node)
                    labels = rows[0].get("labels") or []
                    if labels and "node_type" not in props:
                        props["node_type"] = labels[0]
                if not props:
                    from agent_utilities.knowledge_graph.core.bounded_read import (
                        get_node_data,
                    )

                    props = get_node_data(engine.graph_compute, ident) or {}

                neighbors: list[str] = []
                degree = 0
                try:
                    neighbors = list(engine.graph_compute.neighbors(ident))
                    degree = engine.graph_compute.degree(ident)
                except Exception as e:  # noqa: BLE001 — best-effort structural read
                    logger.warning(
                        "inspect: neighbor/degree lookup failed for %s (exception_type=%s)",
                        ident,
                        type(e).__name__,
                    )

                if not props and not neighbors:
                    return f"No node found for {ident!r}."

                limit = top_k if isinstance(top_k, int) and top_k > 0 else 10
                return _json.dumps(
                    {
                        "id": ident,
                        "properties": props,
                        "degree": degree,
                        "neighbor_count": len(neighbors),
                        "neighbors": neighbors[:limit],
                    },
                    indent=2,
                    default=str,
                )
            # ── KG-2.8: Per-category enrichment coverage gauge ──
            elif action == "enrichment_coverage":
                import json as _json

                from agent_utilities.knowledge_graph.enrichment.query import (
                    enrichment_coverage,
                )

                backend = getattr(engine, "backend", None)
                if backend is None:
                    return "Error: no graph backend available."
                gname = getattr(
                    getattr(engine, "graph_compute", None), "graph_name", None
                )
                return _json.dumps(
                    enrichment_coverage(backend, graph_name=gname), indent=2
                )
            # ── KG-2.8: Outbound process-intelligence writeback ──
            elif action == "process_writeback":
                # Push KG-derived process intelligence back INTO Camunda instances
                # + ARIS models via the unified write-back core (target=process).
                # target=camunda|aris|both (default both); query=optional process ids.
                import json as _json

                from agent_utilities.knowledge_graph.enrichment.writeback import (
                    run_writeback,
                )

                scope = (target or "both").strip().lower()
                process_ids = (
                    [p.strip() for p in query.split(",") if p.strip()]
                    if query
                    else None
                )
                backend = getattr(engine, "backend", None)
                return _json.dumps(
                    run_writeback(
                        "process",
                        backend=backend,
                        engine=engine,
                        dry_run=False,
                        scope=scope,
                        process_ids=process_ids,
                    )
                )
            # ── KG-2.7: Startup Context Generation ──
            elif action == "context":
                try:
                    from agent_utilities.knowledge_graph.memory import (
                        build_startup_payload,
                    )

                    payload = build_startup_payload(
                        engine,
                        agent=target or None,
                        cwd=query or None,
                        budget_chars=top_k * 1000 if top_k != 10 else 24000,
                    )
                    return payload.text
                except Exception as e:
                    return public_error_text(e)
            elif action == "evaluate_alpha":
                from agent_utilities.knowledge_graph.core.quant_tasks import (
                    execute_quant_task,
                )

                res = execute_quant_task(
                    engine, "run_qlib_backtest", {"target": target or query}
                )
                return json.dumps(res)
            elif action in (
                "evaluate",
                "evolve_model",
                "forecast",
                "causal",
                "invariant",
            ):
                # BUG-5: these used to be hardcoded canned-success strings that did
                # nothing regardless of input — a silent no-op masquerading as a
                # real result. No real implementation of any of these exists on
                # THIS surface (confirmed by source search), so fail honestly with
                # a pointer to the real tool/service instead of faking success.
                # forecast/evolve_model stay out of agent-utilities on purpose
                # (heavy ML training belongs in data-science-mcp — anti-sprawl).
                _NOT_IMPLEMENTED_HINT = {
                    "evaluate": (
                        "'evaluate_alpha' (quant backtests), 'evaluate_harness', "
                        "or 'check_constraints' on this same graph_evaluate/graph_analyze surface"
                    ),
                    "evolve_model": (
                        "the data-science-mcp model-training/evolution surface "
                        "(heavy ML training does not belong in agent-utilities)"
                    ),
                    "forecast": (
                        "engine_timeseries (native TSDB) or "
                        "graph_mine_deep(action='deep_forecast'), which delegates to data-science-mcp"
                    ),
                    "causal": (
                        "graph_ops_causal (agent_utilities/mcp/tools/ops_causal_tools.py) — "
                        "a real root-cause/causal-graph implementation already exists there"
                    ),
                    "invariant": (
                        "agent_utilities.knowledge_graph.core.formal_reasoning_core."
                        "FiniteStateMachine (add_invariant/validate_invariants) directly, "
                        "or 'check_constraints' on this surface for a different kind of check"
                    ),
                }
                return json.dumps(
                    {
                        "status": "not_implemented",
                        "error": (
                            f"Action '{action}' is not implemented on this surface — "
                            f"use {_NOT_IMPLEMENTED_HINT[action]}."
                        ),
                        "action": action,
                    }
                )
            elif action == "security_scan":
                # BUG-5: was a hardcoded canned-success string; no real scan ever ran.
                return json.dumps(
                    {
                        "status": "not_implemented",
                        "error": (
                            "Action 'security_scan' is not implemented on this surface — "
                            "use the security-vulnerability-scan / security-patch-sweep "
                            "skill, or engine_rbac / graph_audit for KG-native access and "
                            "integrity checks."
                        ),
                        "action": action,
                        "target": target,
                    }
                )
            elif action == "placement_plan":
                # Multi-objective workload placement over the infra subgraph
                # (efficiency/security/cost/resilience), propose-only (CONCEPT:AU-KG.ingest.enterprise-source-extractor).
                import json as _json

                from agent_utilities.knowledge_graph.infra import optimize_from_graph

                return _json.dumps(optimize_from_graph(engine), indent=2, default=str)
            elif action == "infra_sweep":
                # Hardware inventory sweep → KG infra ontology (CONCEPT:AU-KG.ingest.enterprise-source-extractor).
                # `target`/`query` carries a comma-separated host id list.
                import json as _json

                from agent_utilities.knowledge_graph.infra import collect_and_persist

                host_ids = [
                    h.strip() for h in (target or query or "").split(",") if h.strip()
                ]
                return _json.dumps(
                    collect_and_persist(engine, host_ids), indent=2, default=str
                )
            elif action == "specialize":
                # SAI factory (CONCEPT:AU-AHE.harness.sai-controller): ground a learned world model in
                # persisted WorldModelTransition history and specialize its config,
                # returning adaptation-speed metrics (AHE-3.27) + superhuman
                # certification (SAFE-1.6). On-demand twin of the KG_SAI_FACTORY tick,
                # so the closed loop is reachable through the gateway, not just the daemon.
                import json as _json

                from agent_utilities.harness.superhuman_gate import SuperhumanCertifier
                from agent_utilities.harness.world_model_task import (
                    specialize_world_model_from_engine,
                )

                summary = specialize_world_model_from_engine(
                    engine, certifier=SuperhumanCertifier()
                )
                if summary is None:
                    return _json.dumps(
                        {
                            "status": "noop",
                            "reason": "insufficient WorldModelTransition history to specialize",
                        }
                    )
                return _json.dumps({"status": "ok", **summary}, default=str)
            elif action == "world_model_rollout":
                # CONCEPT:AU-KG.compute.world-model-forward-simulation — forward-simulate the learned world model with
                # persistent latent rollout memory (carry the predicted latent across
                # steps so the imagined trajectory stays on-manifold instead of
                # re-deriving from the bare next-state string each step). Grounds in
                # persisted WorldModelTransition history, rolls a fixed policy forward,
                # and persists the imagined trajectory as a WorldModelRollout node.
                from agent_utilities.knowledge_graph.core.world_model import WorldModel

                world_model = WorldModel.from_engine(engine, latent=True)
                start = (query or "").strip()
                horizon = int(top_k) if top_k else 8
                repeat_action = "advance"
                traj = world_model.rollout(start, lambda _s: repeat_action, horizon)
                rollout_id = world_model.persist_rollout(traj)
                return json.dumps(
                    {
                        "status": "ok",
                        "start": start,
                        "horizon": horizon,
                        "rollout_id": rollout_id,
                        "expected_return": round(world_model.expected_return(traj), 4),
                        "total_drift": round(sum(t.drift for t in traj), 4),
                        "steps": [t.as_dict() for t in traj],
                    },
                    default=str,
                )
            elif action == "research_ingest":
                # KG-2.33 — deep-research ingestion: fetch a paper/URL, run the
                # research pipeline (orchestrator + citation subagents), and persist
                # it into the KG. ``query`` carries the URL or paper id.
                from agent_utilities.knowledge_graph.research.research_intelligence_engine import (  # noqa: E501
                    ResearchIntelligenceEngine,
                )

                if not query:
                    return "Error: research_ingest needs a URL/paper id in `query`."
                rie = ResearchIntelligenceEngine(engine)
                return await rie.ingest_url(query)
            elif action == "evolve_variants":
                from agent_utilities.harness.agentic_evolution_engine import (
                    AgenticEvolutionEngine,
                )

                if not query:
                    return "Error: evolve_variants needs a base_id in `query`."
                aee = AgenticEvolutionEngine(engine)
                result = aee.run_evolution_cycle(
                    base_id=query,
                    task_text=node_id or "",
                    top_k=top_k if top_k else 3,
                )
                return json.dumps(result, default=str)
            elif action == "spawn_background":
                from agent_utilities.harness.background_spawner import (
                    BackgroundAgentSpawner,
                )

                if not engine:
                    return "Error: spawn_background requires an active engine."
                if not query:
                    return (
                        "Error: spawn_background needs a task description in `query`."
                    )
                spawner = BackgroundAgentSpawner(engine)
                team = spawner.orchestrator.synthesize_team(
                    query=query,
                    domain=target or "background_operations",
                    complexity=depth if depth > 0 else 4,
                )
                return json.dumps(
                    {
                        "status": "ok",
                        "team_id": team.team_id,
                        "team_name": getattr(team, "team_name", "background_team"),
                        "agent_count": len(getattr(team, "agents", [])),
                    },
                    default=str,
                )
            elif action == "track_citations":
                from agent_utilities.harness.citation_tracker import CitationTracker

                if not query:
                    return (
                        "Error: track_citations needs agent response text in `query`."
                    )
                tracker = CitationTracker()
                citations = tracker.extract_citations(query)
                if not citations:
                    return json.dumps(
                        {"status": "no_citations", "total": 0, "citations": []}
                    )
                citation_data = [
                    {
                        "source_id": c.source_id,
                        "citation_type": c.citation_type,
                        "raw_text": c.raw_text,
                        "confidence": c.confidence,
                    }
                    for c in citations
                ]
                report = tracker.evaluate_citations(
                    citations,
                    retrieved_doc_ids=set(json.loads(target)) if target else None,
                    gold_doc_ids=set(json.loads(node_id)) if node_id else None,
                )
                return json.dumps(
                    {
                        "status": "extracted",
                        "total_citations": report.total_citations,
                        "precision": report.precision,
                        "recall": report.recall,
                        "f1": report.f1,
                        "citations": citation_data,
                        "hallucinated_citations": report.hallucinated_citations,
                        "uncited_evidence": report.uncited_evidence,
                        "citation_types": report.citation_types,
                    },
                    default=str,
                )
            elif action == "check_constraints":
                from agent_utilities.harness.constraint_engine import (
                    ConstraintEngine,
                )

                if not query:
                    return "Error: check_constraints needs a tool_name in `query`."
                if not engine:
                    return "Error: check_constraints requires a knowledge engine to instantiate ConstraintEngine."
                ce = ConstraintEngine(knowledge_engine=engine)
                allowed, violations = ce.check_tool_call(
                    tool_name=query,
                    args={"target": target} if target else None,
                )
                result = {
                    "allowed": allowed,
                    "tool_name": query,
                    "violations": [
                        {
                            "constraint_id": v.constraint_id,
                            "violation_context": v.violation_context,
                            "timestamp": v.timestamp,
                            "auto_blocked": v.auto_blocked,
                        }
                        for v in violations
                    ],
                }
                return json.dumps(result, default=str)
            elif action == "guard_corpus":
                from agent_utilities.harness.corpus_collapse_guard import (
                    CorpusCollapseGuard,
                )

                guard = CorpusCollapseGuard()
                return json.dumps(guard.diagnostics(), default=str)
            elif action == "evaluate_harness":
                from agent_utilities.harness.evaluation_engine import EvaluationEngine

                if not query:
                    return "Error: evaluate_harness needs a trajectory_id in `query`."
                eval_engine = EvaluationEngine(engine)
                result = eval_engine.evaluate_and_decompose(
                    trajectory_id=query,
                    steps=[],
                    goal_achieved=True,
                    reasoning_effort=0.5,
                )
                return json.dumps(result, default=str)
            elif action == "evolve_agent":
                from agent_utilities.harness.evidence_corpus import EvidenceCorpus
                from agent_utilities.harness.evolve_agent import EvolveAgent

                if not query:
                    return "Error: evolve_agent needs an evidence corpus ID or path in `query`."
                try:
                    workspace_path = os.getcwd()  # Fallback; ideally passed as param
                    evolve = EvolveAgent(
                        workspace_path=workspace_path,
                        registry=None,
                        knowledge_engine=engine,
                    )
                    # Best-effort: construct minimal EvidenceCorpus from query.
                    # In real usage, this would load from .specify/ or KG.
                    evidence = EvidenceCorpus(
                        round_id=query,
                        benchmark_score=0.5,
                        pass_rate=0.5,
                        total_tasks=0,
                    )
                    manifest = await evolve.evolve(evidence)
                    return json.dumps(manifest.model_dump(), default=str)
                except Exception as e:
                    return public_error_text(e)
            elif action == "recursive_distill":
                from agent_utilities.harness.recursive_distill import RecursiveDistiller

                if not engine:
                    return "Error: recursive_distill requires an active engine."
                # RecursiveDistiller needs external-compute injections (corpus_source,
                # trainer, evaluate_model, promote). Report what it expects so the
                # caller can wire a distillation daemon (CONCEPT:AU-AHE.optimization.recursive-distillation-loop).
                return json.dumps(
                    {
                        "status": "needs_injection",
                        "entry": "RecursiveDistiller.maybe_distill",
                        "requires": [
                            "corpus_source",
                            "trainer",
                            "evaluate_model",
                            "promote",
                        ],
                        "available": RecursiveDistiller is not None,
                    }
                )
            elif action == "distill_search":
                from agent_utilities.harness.search_distillation import (
                    SearchDistillationHarvester,
                )

                if not query:
                    return "Error: distill_search needs a prompt in `query`."
                harvester = SearchDistillationHarvester(engine)
                candidates = [
                    (f"candidate_{i}", float(i) / max(1, top_k))
                    for i in range(1, top_k + 1)
                ]
                rows, pairs = harvester.harvest_candidates(query, candidates)
                result = {
                    "sft_rows": [
                        {
                            "prompt": r.prompt,
                            "completion": r.completion,
                            "score": r.score,
                            "source": r.source,
                            "synthetic": r.synthetic,
                        }
                        for r in rows
                    ],
                    "preference_pairs": [
                        {"prompt": p.prompt, "chosen": p.chosen, "rejected": p.rejected}
                        for p in pairs
                    ],
                }
                return json.dumps(result, default=str)
            elif action == "extract_claims":
                # CONCEPT:AU-KG.enrichment.entity-claim-extraction — entity-claim extraction for MAGMA epistemic view.
                # Extracts entities, claims, and implicit relationships from document
                # content using deterministic + pack-driven inference, then persists
                # to the KG. ``query`` carries the content to analyze.
                from agent_utilities.knowledge_graph.kb.entity_claim_extractor import (
                    EntityClaimExtractor,
                )

                if not query:
                    return "Error: extract_claims needs document content in `query`."
                ece = EntityClaimExtractor(engine)
                ext_result = ece.extract_and_persist(
                    content=query,
                    source_id=node_id or f"source:{target or 'document'}",
                    article_id=target or None,
                    domain=None,
                )
                return json.dumps(ext_result.model_dump(), default=str)
            elif action == "contradictions":
                # CONCEPT:AU-KG.research.explicit-node-node-contradiction — explicit node↔node contradiction/friction surface
                # (the night-shift Critic): retrieve topically-similar existing nodes
                # and flag those that OPPOSE the new claim in `query`. Propose-only —
                # never auto-resolves; returns FRICTION findings for human judgment.
                from agent_utilities.knowledge_graph.adaptation.contradiction_detector import (  # noqa: E501
                    Claim,
                    ContradictionDetector,
                )

                if not query:
                    return "Error: contradictions needs the new claim text in `query`."
                neighbours = engine.search_hybrid(query, top_k=top_k) or []
                existing = [
                    Claim(
                        id=str(n.get("id") or (n.get("node", {}) or {}).get("id") or i),
                        text=str(
                            n.get("description")
                            or n.get("name")
                            or (n.get("node", {}) or {}).get("description")
                            or ""
                        ),
                    )
                    for i, n in enumerate(neighbours)
                    if isinstance(n, dict)
                ]
                new_claim = Claim(id=node_id or "new", text=query)
                findings = ContradictionDetector().check(new_claim, existing)

                # CONCEPT:AU-KG.retrieval.graph-engineering-canonical-prompts — the
                # graph-maintenance canonical prompt, wired onto this EXISTING
                # contradiction/TMS path as a best-effort LLM recommendation layered
                # on top of the deterministic detector above (propose-only, same
                # contract). Resolved ONCE (not per finding) and degrades to no
                # "maintenance" key at all with no LLM configured — identical JSON
                # shape to before this was added.
                from agent_utilities.knowledge_graph.retrieval.graph_engineering import (
                    narrate_maintenance_action,
                    resolve_llm_fn,
                )

                existing_by_id = {c.id: c.text for c in existing}
                llm_fn = resolve_llm_fn() if findings else None
                results = []
                for f in findings:
                    entry: dict[str, Any] = {
                        "new_id": f.new_id,
                        "conflict_id": f.conflict_id,
                        "similarity": round(f.similarity, 3),
                        "severity": f.severity,
                        "reason": f.reason,
                    }
                    maintenance = narrate_maintenance_action(
                        f,
                        new_text=new_claim.text,
                        existing_text=existing_by_id.get(f.conflict_id, ""),
                        llm_fn=llm_fn,
                    )
                    if maintenance:
                        entry["maintenance"] = maintenance
                    results.append(entry)
                return json.dumps(results, default=str)
            elif action == "evolve_code":
                # CONCEPT:AU-KG.retrieval.monte-carlo-graph-search — Monte-Carlo GRAPH search code evolution (MLEvolve)
                # driven by a REAL LLM coder (CONCEPT:AU-ORCH.execution.drop-rlm-completion-client RLM). Each search node
                # is coded by the LLM from the step plan + prior code; a deterministic
                # refinement is the offline fallback. Run in a worker thread so the
                # sync RLM client has its own event loop.
                from agent_utilities.harness.agentic_evolution_engine import (
                    AgenticEvolutionEngine,
                )

                if not query:
                    return "Error: evolve_code needs a task description in `query`."

                def _llm_coder(plan: str, prior_code: str | None) -> tuple[str, str]:
                    try:
                        from agent_utilities.rlm.client import RLM

                        prompt = (
                            "Improve the code solution for this task. Return ONLY the "
                            "full updated Python code, no prose.\n"
                            f"Task: {query}\nStep plan: {plan}\n"
                            f"Current code:\n{prior_code or '(none)'}"
                        )
                        resp = RLM().completion(prompt)
                        if resp.ok and resp.response.strip():
                            return (plan, resp.response)
                    except Exception:  # noqa: BLE001 — offline / LLM error -> fallback
                        pass
                    return (plan, f"{prior_code or ''}\n# step for: {plan}".strip())

                result = await asyncio.to_thread(
                    lambda: AgenticEvolutionEngine(engine).evolve_via_graph_search(
                        query, num_steps=top_k, coder_fn=_llm_coder
                    )
                )
                return json.dumps(result, default=str)
            elif action == "night_shift":
                # CONCEPT:AU-KG.research.run-one-autonomous-night — run one autonomous night-shift cycle over a
                # local markdown vault: scout→catalog→cartograph→critique→edit
                # (the second-brain swarm). `target` is the vault root; sources
                # dropped in <vault>/0-raw|sources are refined into linked atomic
                # notes with [FRICTION] surfaced + a morning briefing. Schedule it
                # via cron for the overnight pattern. Propose-only; never deletes.
                from agent_utilities.knowledge_graph.research.night_shift import (
                    NightShiftSwarm,
                )

                if not target:
                    return "Error: night_shift needs the vault root path in `target`."

                def _llm_extract(source_text: str) -> list[str]:
                    # Real LLM Cataloger (CONCEPT:AU-ORCH.execution.drop-rlm-completion-client RLM): split a source into
                    # atomic ideas; deterministic paragraph/sentence splitter fallback.
                    try:
                        from agent_utilities.rlm.client import RLM

                        prompt = (
                            "Extract the atomic ideas from the text below as a list, "
                            "one self-contained claim per line:\n\n" + source_text
                        )
                        resp = RLM().completion(prompt)
                        if resp.ok and resp.response.strip():
                            atoms = [
                                line.lstrip("0123456789.-) \t").strip()
                                for line in resp.response.splitlines()
                                if line.strip()
                            ]
                            if atoms:
                                return atoms
                    except Exception:  # noqa: BLE001 — offline / LLM error -> fallback
                        pass
                    from agent_utilities.knowledge_graph.research.night_shift import (
                        default_extract,
                    )

                    return default_extract(source_text)

                shift_report = await asyncio.to_thread(
                    lambda: NightShiftSwarm(target, extract_fn=_llm_extract).run_shift()
                )
                return json.dumps(
                    {
                        "sources_ingested": shift_report.sources_ingested,
                        "atoms_created": shift_report.atoms_created,
                        "links_added": shift_report.links_added,
                        "frictions": shift_report.frictions,
                        "briefing_path": shift_report.briefing_path,
                    },
                    default=str,
                )
            elif action == "recommend":
                # CONCEPT:AU-KG.retrieval.pauserec-implicit-reasoning-generative — PauseRec implicit-reasoning generative recommender:
                # retrieve candidate items, assign them semantic IDs, then recommend the
                # next items via a latent-reasoning budget + a text↔SID bridge (no
                # brittle explicit CoT). `query` is the user intent / history summary.
                from agent_utilities.knowledge_graph.retrieval.generative_recommender import (  # noqa: E501
                    ImplicitReasoningRecommender,
                )
                from agent_utilities.knowledge_graph.retrieval.temporal_semantic_id import (  # noqa: E501
                    TemporalSemanticIdEncoder,
                )

                if not query:
                    return "Error: recommend needs a query/intent in `query`."
                candidates = engine.search_hybrid(query, top_k=max(top_k * 4, 20)) or []
                items = []
                for c in candidates:
                    if not isinstance(c, dict):
                        continue
                    inner = c.get("node", c)
                    inner = inner if isinstance(inner, dict) else {}
                    emb = inner.get("embedding")
                    cid = str(inner.get("id") or c.get("id") or "")
                    if emb and cid:
                        items.append((cid, emb))
                if not items:
                    return json.dumps([])
                embed_model = getattr(
                    getattr(engine, "hybrid_retriever", None), "embed_model", None
                )
                qemb = None
                if embed_model is not None:
                    try:
                        qemb = embed_model.get_text_embedding(query)
                    except Exception:  # noqa: BLE001 — embedder down -> anchor on top item
                        qemb = None
                recommender = ImplicitReasoningRecommender(TemporalSemanticIdEncoder())
                recommender.fit_catalog(items)
                recs = recommender.recommend(qemb or items[0][1], top_k=top_k)
                return json.dumps(
                    [
                        {
                            "item_id": r.item_id,
                            "semantic_id": list(r.semantic_id),
                            "score": r.score,
                        }
                        for r in recs
                    ],
                    default=str,
                )
            elif action == "assimilation_benchmark":
                # CONCEPT:AU-AHE.assimilation.empirical-parity-evidence-assimilation — measured empirical-parity evidence: run each
                # assimilated paper's mechanism vs a baseline on a controlled task and
                # report the real lift + claim-reproduced verdict (the proof that we
                # got feature parity, not just shipped the mechanism). Deterministic,
                # CPU; the trained-pause-token bench runs when torch is present.
                from agent_utilities.harness.assimilation_benchmark import (
                    run_all as _bench_run_all,
                )
                from agent_utilities.harness.assimilation_benchmark import (
                    to_markdown as _bench_md,
                )

                bench_results = _bench_run_all(seed=int(top_k) if top_k else 0)
                return json.dumps(
                    {
                        "reproduced": sum(
                            1 for r in bench_results if r.claim_reproduced
                        ),
                        "total": len(bench_results),
                        "results": [
                            {
                                "name": r.name,
                                "metric": r.metric,
                                "baseline": r.baseline,
                                "ours": r.ours,
                                "lift": r.lift,
                                "claim_reproduced": r.claim_reproduced,
                            }
                            for r in bench_results
                        ],
                        "markdown": _bench_md(bench_results),
                    },
                    default=str,
                )
            elif action == "latent_efficiency_benchmark":
                # CONCEPT:AU-AHE.harness.empirical-evidence-that-latent — measured lift for the latent-native memory
                # mechanisms: latent rollout memory (KG-2.73b) reduces trajectory
                # drift vs a memoryless rollout, and the ontology-type prior (KG-2.44b)
                # improves top-k neighbourhood coherence vs flat cosine. Deterministic,
                # CPU; the on-demand twin of the latent-native enhancements' evidence.
                from agent_utilities.harness.latent_efficiency_benchmark import (
                    run_all as _lat_run_all,
                )
                from agent_utilities.harness.latent_efficiency_benchmark import (
                    to_markdown as _lat_md,
                )

                lat_results = _lat_run_all(seed=int(top_k) if top_k else 0)
                return json.dumps(
                    {
                        "reproduced": sum(1 for r in lat_results if r.claim_reproduced),
                        "total": len(lat_results),
                        "results": [
                            {
                                "name": r.name,
                                "metric": r.metric,
                                "baseline": r.baseline,
                                "ours": r.ours,
                                "lift": r.lift,
                                "claim_reproduced": r.claim_reproduced,
                            }
                            for r in lat_results
                        ],
                        "markdown": _lat_md(lat_results),
                    },
                    default=str,
                )
            elif action == "infer_links":
                from agent_utilities.knowledge_graph.kb.link_inference import (
                    infer_links,
                )
                from agent_utilities.models.schema_pack_loader import get_active_pack

                if not query:
                    return "Error: infer_links needs content text in `query`."
                if not node_id:
                    return "Error: infer_links needs a source node ID in `node_id`."

                schema_pack = get_active_pack()
                if not schema_pack or not getattr(schema_pack, "link_inference", None):
                    return "Error: no active schema pack with link_inference rules available."

                rules = schema_pack.link_inference
                extracted = infer_links(query, node_id, rules)

                return json.dumps(
                    [
                        {
                            "source_name": rel.source_name,
                            "target_name": rel.target_name,
                            "relationship_type": rel.relationship_type,
                            "confidence": rel.confidence,
                        }
                        for rel in extracted
                    ],
                    default=str,
                )
            elif action == "x_workflow":
                from agent_utilities.knowledge_graph.kb.x_workflows import (
                    register_x_workflows,
                )

                if not engine:
                    return (
                        "Error: x_workflow requires an active IntelligenceGraphEngine."
                    )
                force = query.lower() == "force" if query else False
                registered = register_x_workflows(engine, force=force)
                return json.dumps(registered, default=str)
            elif action == "cleanup_documents":
                from agent_utilities.knowledge_graph.maintenance.document_cleanup import (
                    DocumentCleanup,
                )

                cleanup = DocumentCleanup(engine)
                result = await cleanup.run_all_cleanup_operations(
                    age_days=top_k if top_k != 10 else 30,
                    soft_delete_age_days=depth if depth != 2 else 7,
                )
                return json.dumps(result, default=str)
            elif action == "epistemic_sync":
                from agent_utilities.workflows.epistemic_sync import (
                    EpistemicSyncWorkflow,
                )

                workflow = EpistemicSyncWorkflow()
                await workflow.run_sync_cycle()
                return json.dumps(
                    {
                        "status": "sync_cycle_completed",
                        "message": "Epistemic Sync cycle executed successfully. Check logs for details on entities ingested and mutations flushed.",
                    }
                )
            elif action == "pick_skill":
                from agent_utilities.workflows.skill_picker import (
                    SkillCandidate,
                    SkillPicker,
                )

                if not query:
                    return "Error: pick_skill needs a skill query in `query`."
                picker = SkillPicker()
                # Without a skill registry endpoint or hardcoded candidates,
                # we cannot populate the candidate list. Placeholder shows the API.
                skill_candidates: list[SkillCandidate] = []
                ranked = picker.rank(query, skill_candidates)
                return json.dumps(
                    [
                        {
                            "name": s.candidate.name,
                            "score": s.score,
                            "breakdown": s.breakdown,
                            "scenario": s.candidate.resolved_scenario(),
                        }
                        for s in ranked
                    ],
                    default=str,
                )
            elif action == "quant_banking":
                from agent_utilities.domains.finance.banking import KYCAMLEngine

                if not query:
                    return "Error: quant_banking needs a transaction_id in `query`."
                # Use query as transaction_id; derive account_id and amount from context
                # or use sensible defaults for a compliance check
                engine_instance = KYCAMLEngine()
                alert = engine_instance.check_transaction(
                    transaction_id=query,
                    account_id=f"account:{query[:8]}",
                    amount=float(target)
                    if target and target.replace(".", "").isdigit()
                    else 10000.0,
                )
                if alert is None:
                    return json.dumps({"status": "compliant", "transaction_id": query})
                return json.dumps(
                    {
                        "status": "alert",
                        "alert_id": alert.id,
                        "transaction_id": alert.transaction_id,
                        "account_id": alert.account_id,
                        "severity": alert.severity.value,
                        "alert_type": alert.alert_type,
                        "amount": alert.amount,
                    },
                    default=str,
                )
            elif action == "quant_arb":
                from agent_utilities.domains.finance.cross_market_arb import (
                    EventArbitrageEngine,
                )

                if not query:
                    return "Error: quant_arb needs market parameters in `query` (JSON: {model_probability, market_a_price, market_b_price} or comma-separated values)."
                try:
                    if query.startswith("{"):
                        params = json.loads(query)
                        model_prob = float(params.get("model_probability", 0.5))
                        market_a = float(params.get("market_a_price", 0.5))
                        market_b = float(params.get("market_b_price", 0.5))
                        exec_costs = float(params.get("execution_costs", 0.08))
                    else:
                        parts = query.split(",")
                        model_prob = float(parts[0].strip())
                        market_a = float(parts[1].strip()) if len(parts) > 1 else 0.5
                        market_b = float(parts[2].strip()) if len(parts) > 2 else 0.5
                        exec_costs = float(parts[3].strip()) if len(parts) > 3 else 0.08
                except (ValueError, IndexError, json.JSONDecodeError) as e:
                    return public_error_text(e, code="invalid_request")
                result = EventArbitrageEngine.evaluate_dual_markets(
                    model_probability=model_prob,
                    market_a_price=market_a,
                    market_b_price=market_b,
                    execution_costs=exec_costs,
                )
                return json.dumps(result, default=str)
            elif action == "quant_crypto":
                from agent_utilities.domains.finance.crypto_connector import (
                    CryptoConnector,
                )

                if not query:
                    return "Error: quant_crypto needs a symbol in `query` (e.g., 'BTC/USD')."
                connector = CryptoConnector()
                result = connector.get_asset_context(query)
                return json.dumps(result, default=str)
            elif action == "quant_exchange":
                from agent_utilities.domains.finance.exchange_bridge import (
                    ExchangeBridge,
                )

                if not query:
                    return "Error: quant_exchange needs a symbol (e.g., BTC/USDT or AAPL) in `query`."
                bridge = ExchangeBridge(paper_mode=True)
                exec_result = bridge.execute(
                    symbol=query,
                    side="buy",
                    qty=float(target.split(":")[1])
                    if target and ":" in target
                    else 1.0,
                    order_type="market",
                    limit_price=None,
                )
                return json.dumps(
                    {
                        "order_id": exec_result.order_id,
                        "status": exec_result.status,
                        "filled_qty": exec_result.filled_qty,
                        "average_price": exec_result.average_price,
                        "fees": exec_result.fees,
                        "exchange": exec_result.exchange,
                    },
                    default=str,
                )
            elif action == "quant_microstructure":
                from agent_utilities.domains.finance.microstructure import (
                    ConvergenceFilter,
                    MicroPriceCalculator,
                    OrderBookImbalance,
                )

                if not query:
                    return "Error: quant_microstructure needs order book data in `query` (JSON: {bid_price, ask_price, bid_volume, ask_volume}) or set via target/depth."
                try:
                    import json as _json

                    if isinstance(query, str):
                        try:
                            params = _json.loads(query)
                        except Exception:
                            params = {}
                    else:
                        params = query if isinstance(query, dict) else {}
                    bid_price = float(
                        params.get(
                            "bid_price", target.split(",")[0] if target else 99.5
                        )
                    )
                    ask_price = float(
                        params.get(
                            "ask_price",
                            target.split(",")[1] if target and "," in target else 100.5,
                        )
                    )
                    bid_volume = float(params.get("bid_volume", top_k * 100))
                    ask_volume = float(params.get("ask_volume", depth * 100))

                    obi = OrderBookImbalance.calculate(bid_volume, ask_volume)
                    spread = ask_price - bid_price
                    micro_price = MicroPriceCalculator.calculate(
                        bid_price, ask_price, bid_volume, ask_volume
                    )
                    micro_price_from_imbalance = MicroPriceCalculator.from_imbalance(
                        (bid_price + ask_price) / 2.0, spread, obi
                    )
                    is_consensus = ConvergenceFilter.check_agreement(
                        [True] * min(5, max(1, int(obi * 5 + 2.5))), threshold=5
                    )
                    result = {
                        "order_book": {
                            "bid_price": bid_price,
                            "ask_price": ask_price,
                            "bid_volume": bid_volume,
                            "ask_volume": ask_volume,
                            "spread": spread,
                        },
                        "imbalance": {"obi": float(obi), "consensus": is_consensus},
                        "micro_price": {
                            "direct_calculation": float(micro_price),
                            "from_imbalance": float(micro_price_from_imbalance),
                        },
                        "status": "ok",
                    }
                    return _json.dumps(result, default=str)
                except Exception as e:
                    return public_error_text(e)
            elif action == "quant_strategy":
                from agent_utilities.domains.finance.strategy_engine import (
                    StrategyEngine,
                    StrategyMetrics,
                )

                if not query:
                    return "Error: quant_strategy needs a strategy_id in `query`."
                if engine is None:
                    return "Error: quant_strategy requires an active knowledge graph engine."
                se = StrategyEngine(engine)
                metrics = StrategyMetrics(
                    sharpe=2.5,
                    max_drawdown=-0.10,
                    win_rate=0.55,
                    profit_factor=1.5,
                    total_trades=max(100, top_k),
                )
                promotable = se.record_backtest(query, metrics)
                return json.dumps(
                    {
                        "strategy_id": query,
                        "promotable": promotable,
                        "metrics": {
                            "sharpe": metrics.sharpe,
                            "max_drawdown": metrics.max_drawdown,
                            "win_rate": metrics.win_rate,
                            "profit_factor": metrics.profit_factor,
                            "total_trades": metrics.total_trades,
                        },
                    },
                    default=str,
                )
            elif action == "quant_regime":
                from agent_utilities.domains.finance.regime_detector import (
                    RegimeDetector,
                )

                try:
                    import pandas as pd
                except ImportError:
                    return "Error: pandas is required for quant_regime."

                if not query:
                    return "Error: quant_regime needs a ticker symbol in `query`."

                # Create synthetic OHLC data for demonstration
                # In production, this would ingest real market data
                from agent_utilities.numeric import xp as np

                dates = pd.date_range(end=pd.Timestamp.now(), periods=100)
                base_price = 100.0
                returns = np.random.normal(0.0005, 0.02, 100)
                close_prices = base_price * np.cumprod(1 + returns)
                df = pd.DataFrame(
                    {
                        "Close": close_prices,
                        "High": close_prices * 1.02,
                        "Low": close_prices * 0.98,
                        "Open": np.roll(close_prices, 1)[1:].tolist() + [base_price],
                    },
                    index=dates,
                )

                detector = RegimeDetector(engine)
                regime = detector.detect_regime(df, ticker=query)
                return regime
            elif action == "quant_insider":
                # CONCEPT:AU-KG.research.research-pipeline-runner — Kyle insider-trading equilibrium + enforcement
                # policy analysis. `query` = optional JSON of InsiderEquilibriumInputs
                # overrides (sigma_v, enforcement, criminal_penalty, …).
                import json as _json

                from agent_utilities.domains.finance.insider_equilibrium import (
                    InsiderEquilibriumInputs,
                    penalty_policy_analysis,
                    solve_equilibrium,
                )

                try:
                    overrides = _json.loads(query) if query else {}
                except Exception:
                    overrides = {}
                inputs = InsiderEquilibriumInputs(
                    **{
                        k: v
                        for k, v in overrides.items()
                        if k in InsiderEquilibriumInputs.__dataclass_fields__
                    }
                )
                import dataclasses as _dc

                def _ser(o):
                    return (
                        _dc.asdict(o)
                        if _dc.is_dataclass(o) and not isinstance(o, type)
                        else o
                    )

                eq = solve_equilibrium(inputs)
                policy = penalty_policy_analysis(inputs)
                return _json.dumps(
                    {"status": "ok", "equilibrium": _ser(eq), "policy": _ser(policy)},
                    default=str,
                )
            elif action == "workforce_plan":
                from agent_utilities.domains.hr.workforce_manager import (
                    WorkforceManager,
                )

                wm = WorkforceManager()
                result = wm.get_workforce_summary()
                return json.dumps(result, default=str)
            elif action == "close":
                # Background OWL-RL + SHACL closure (KG-2.6): promote recent nodes
                # to RDF, materialize implied edges via the reasoner, validate
                # against shapes. On-demand twin of the maintenance-tick closure.
                from agent_utilities.knowledge_graph.maintenance.owl_closure import (
                    run_closure,
                )

                summary = run_closure(
                    engine, limit=top_k * 200 if top_k != 10 else 2000
                )
                return json.dumps(summary, default=str)
            elif action == "call_graph":
                # CONCEPT:EG-KG.compute.type-scope-resolved-call — the type/scope-resolved call/inheritance graph
                # for a symbol. Returns the resolved edges (with their strategy +
                # confidence) the Rust resolver bound and the OWL layer reasons over.
                # `node_id` = the symbol id; `target` = direction
                # (callees | callers | inherits). Reads run in the engine backend.
                import json as _json

                if not node_id:
                    return "Error: call_graph needs a symbol id in `node_id`."
                backend = getattr(engine, "backend", None)
                if backend is None:
                    return "Error: no graph backend available."
                direction = (target or "callees").strip().lower()
                if direction == "callers":
                    query = (
                        "MATCH (t)-[r]->(s {id: $id}) "
                        "WHERE type(r) IN ['calls', 'CALLS'] "
                        "RETURN t.id AS id, t.id AS node, type(r) AS rel, "
                        "r.strategy AS strategy, r.confidence AS confidence"
                    )
                elif direction == "inherits":
                    query = (
                        "MATCH (s {id: $id})-[r]->(t) "
                        "WHERE type(r) IN ['inherits', 'INHERITS', 'realizes', 'REALIZES'] "
                        "RETURN t.id AS id, t.id AS node, type(r) AS rel, "
                        "r.strategy AS strategy, r.confidence AS confidence"
                    )
                else:  # callees (default)
                    direction = "callees"
                    query = (
                        "MATCH (s {id: $id})-[r]->(t) "
                        "WHERE type(r) IN ['calls', 'CALLS'] "
                        "RETURN t.id AS id, t.id AS node, type(r) AS rel, "
                        "r.strategy AS strategy, r.confidence AS confidence"
                    )
                try:
                    rows = engine.query_cypher(query, {"id": node_id})
                except Exception as e:
                    return public_error_json(e)
                return _json.dumps(
                    {
                        "status": "ok",
                        "node_id": node_id,
                        "direction": direction,
                        "edges": [
                            {
                                "node": r.get("node"),
                                "rel": r.get("rel"),
                                "strategy": r.get("strategy"),
                                "confidence": r.get("confidence"),
                            }
                            for r in (rows or [])
                        ],
                    },
                    default=str,
                )
            elif action == "similar_code":
                # CONCEPT:EG-KG.compute.model-free-similar-code — model-free similar-code lookup. Returns the
                # symbol's `similar_to` neighbours (MinHash/LSH near-clones) with
                # their score — works with the embedder offline (no accelerator needed).
                # `node_id` = the symbol id.
                import json as _json

                if not node_id:
                    return "Error: similar_code needs a symbol id in `node_id`."
                if getattr(engine, "backend", None) is None:
                    return "Error: no graph backend available."
                # similar_to is symmetric, so match it in either direction.
                query = (
                    "MATCH (s {id: $id})-[r]-(t) "
                    "WHERE type(r) IN ['similar_to', 'SIMILAR_TO'] "
                    "RETURN t.id AS id, t.id AS node, r.score AS score"
                )
                try:
                    rows = engine.query_cypher(query, {"id": node_id})
                except Exception as e:
                    return public_error_json(e)
                neighbours = [
                    {"node": r.get("node"), "score": r.get("score")}
                    for r in (rows or [])
                ]
                neighbours.sort(key=lambda n: float(n["score"] or 0), reverse=True)
                return _json.dumps(
                    {
                        "status": "ok",
                        "node_id": node_id,
                        "embedder_free": True,
                        "similar": neighbours[: top_k if top_k else 10],
                    },
                    default=str,
                )
            elif action == "routes":
                # CONCEPT:AU-KG.compute.http-route-graph — the HTTP route graph: each Route (method+path),
                # its handler Code symbol, and the deployed Service that serves it
                # (Code –serves→ Route –servedBy→ Service). Reads run in the engine.
                import json as _json

                if getattr(engine, "backend", None) is None:
                    return "Error: no graph backend available."
                query = (
                    "MATCH (h)-[r2]->(rt:Route) "
                    "WHERE type(r2) IN ['SERVES', 'serves'] "
                    "OPTIONAL MATCH (rt)-[r3]->(svc) "
                    "WHERE type(r3) IN ['SERVED_BY', 'served_by'] "
                    "RETURN rt.id AS id, rt.id AS route, rt.method AS method, "
                    "rt.path AS path, "
                    "h.id AS handler, svc.id AS service"
                )
                try:
                    rows = engine.query_cypher(query, {})
                except Exception as e:
                    return public_error_json(e)
                return _json.dumps(
                    {
                        "status": "ok",
                        "routes": [
                            {
                                "route": r.get("route"),
                                "method": r.get("method"),
                                "path": r.get("path"),
                                "handler": r.get("handler"),
                                "service": r.get("service"),
                            }
                            for r in (rows or [])
                        ],
                    },
                    default=str,
                )
            elif action == "change_coupling":
                # CONCEPT:AU-KG.ingest.mine-git-history-files — mine git history for files that change together
                # (hidden coupling the AST can't see) and persist symmetric
                # FILE_CHANGES_WITH edges. `target`/`query` = the repo work-tree path.
                import json as _json

                from agent_utilities.knowledge_graph.enrichment.git_coupling import (
                    change_coupling_for_repo,
                )

                repo = (target or query or "").strip()
                if not repo:
                    return "Error: change_coupling needs a repo path in `target`."
                edges = change_coupling_for_repo(
                    repo, min_support=depth if depth > 1 else 3
                )
                written = 0
                for edge in edges:
                    engine.link_nodes(
                        edge.source,
                        edge.target,
                        edge.rel_type,
                        properties=edge.props,
                    )
                    written += 1
                return _json.dumps(
                    {"status": "ok", "repo": repo, "coupled_pairs": written}
                )
            elif action == "code_evolution":
                # CONCEPT:AU-KG.enrichment.query-ingested-commit-history — query the ingested commit-history graph
                # (KG-2.282) for codebase EVOLUTION: file timelines, subsystem
                # ownership, churn hotspots, and change-coupling. `target` = the
                # mode (file|owners|hotspots|coupled), `query` = the file path /
                # subsystem path substring, `top_k` = result cap.
                import json as _json

                from agent_utilities.knowledge_graph.enrichment.git_history import (
                    query_evolution,
                )

                if getattr(engine, "backend", None) is None:
                    return "Error: no graph backend available."
                mode = (target or "file").strip() or "file"
                return _json.dumps(
                    query_evolution(engine, mode, query.strip(), top_k or 20),
                    default=str,
                )
            elif action == "adr":
                # CONCEPT:AU-KG.compute.adr-crud — Architecture Decision Record CRUD. `query` = the
                # decision title (create); empty = list. `target` = status; `node_id`
                # = the decision text.
                import json as _json
                import re as _re

                if getattr(engine, "backend", None) is None:
                    return "Error: no graph backend available."
                if query:
                    slug = _re.sub(r"[^a-z0-9]+", "-", query.lower()).strip("-")
                    adr_id = f"adr:{slug}"
                    engine.add_node(
                        adr_id,
                        "ArchitectureDecisionRecord",
                        {
                            "title": query,
                            "status": target or "proposed",
                            "decision": node_id or "",
                        },
                    )
                    return _json.dumps({"status": "ok", "adr_id": adr_id})
                try:
                    rows = engine.query_cypher(
                        "MATCH (a:ArchitectureDecisionRecord) "
                        "RETURN a.id AS id, a.title AS title, a.status AS status",
                        {},
                    )
                except Exception as e:
                    return public_error_json(e)
                return _json.dumps(
                    {
                        "status": "ok",
                        "adrs": [
                            {
                                "id": r.get("id"),
                                "title": r.get("title"),
                                "status": r.get("status"),
                            }
                            for r in (rows or [])
                        ],
                    },
                    default=str,
                )
            elif action == "harness_gate":
                # CONCEPT:AU-AHE.evaluation.parity-surpass-scoreboard — the formal harness-evolution gate (the seesaw
                # HarnessX lacks): validate a candidate harness-evolution state
                # against the concentration / no-regression / pathology SHACL shapes.
                # `query` = JSON {edits:[{id,dimension,round,status?,regresses?}],
                # variants?:[{id,status,applies}], pathologies?:[{id,kind,exhibited_by}]}.
                import json as _json

                from agent_utilities.harness.harness_gate import HarnessGate

                try:
                    facts = _json.loads(query) if query else {}
                except Exception:
                    return "Error: harness_gate needs JSON harness-evolution facts in `query`."
                verdict = HarnessGate().check_facts(
                    facts.get("edits", []) or [],
                    variants=facts.get("variants"),
                    pathologies=facts.get("pathologies"),
                )
                return _json.dumps(
                    {
                        "status": "ok",
                        "ships": verdict.passed,
                        "reasons": verdict.reasons,
                    }
                )
            elif action == "harness_evolve":
                # CONCEPT:AU-AHE.harness.run-aegis-loop-over — run the AEGIS loop over a provided edit sequence
                # (offline, no LLM): the gate fires across rounds so concentration is
                # blocked BEFORE the tipping point. `query` = JSON {edits:[{dimension,...}]}.
                import json as _json

                from agent_utilities.harness.aegis_loop import AegisLoop

                try:
                    seq = (_json.loads(query) or {}).get("edits", []) if query else []
                except Exception:
                    return "Error: harness_evolve needs JSON {edits:[…]} in `query`."
                pending = list(seq)

                def _replay_evolver(_landscape, _q=pending):
                    return dict(_q.pop(0)) if _q else {"id": "noop", "dimension": "D0"}

                loop = AegisLoop(_replay_evolver)
                decisions = loop.run(rounds=len(seq) or 1)
                return _json.dumps(
                    {
                        "status": "ok",
                        "decisions": [
                            {"round": d.round, "ships": d.shipped, "reasons": d.reasons}
                            for d in decisions
                        ],
                        "shipped": sum(1 for d in decisions if d.shipped),
                    }
                )
            elif action == "harness_certify":
                # CONCEPT:AU-AHE.harness.kg-held-out-certification/KG-2.108 — held-out certification + ARA-Seal of a
                # promoted variant. `query` = JSON {held_out_rewards:[…], human_baseline,
                # variant_id?}.
                import json as _json

                from agent_utilities.harness.co_evolution import CrossHarnessCoEvolution
                from agent_utilities.harness.harness_grounding import seal_variant

                try:
                    payload = _json.loads(query) if query else {}
                except Exception:
                    return "Error: harness_certify needs JSON in `query`."
                cert = CrossHarnessCoEvolution().certify_promotion(
                    [float(x) for x in payload.get("held_out_rewards", [])],
                    payload.get("human_baseline"),
                )
                _, _, level = seal_variant(
                    payload.get("variant_id", "harness_variant:adhoc"), cert
                )
                return _json.dumps(
                    {
                        "status": "ok",
                        "certified": cert.certified,
                        "seal_level": level,
                        "ci_lower": cert.ci_lower,
                        "mean_reward": cert.mean_reward,
                    },
                    default=str,
                )
            elif action == "harness_benchmark":
                # CONCEPT:AU-AHE.evaluation.parity-surpass-scoreboard — the parity-and-surpass scoreboard vs HarnessX.
                import json as _json

                from agent_utilities.harness.harness_foundry_benchmark import (
                    run_all as _hf_run,
                )
                from agent_utilities.harness.harness_foundry_benchmark import (
                    to_markdown as _hf_md,
                )

                results = _hf_run()
                return _json.dumps(
                    {
                        "status": "ok",
                        "reproduced": sum(1 for r in results if r.claim_reproduced),
                        "total": len(results),
                        "results": [
                            {
                                "name": r.name,
                                "baseline": r.baseline,
                                "ours": r.ours,
                                "lift": r.lift,
                                "claim_reproduced": r.claim_reproduced,
                            }
                            for r in results
                        ],
                        "markdown": _hf_md(results),
                    },
                    default=str,
                )
            elif action == "code_context":
                # CONCEPT:AU-KG.retrieval.synthesized-cited-answer — the synthesized, cited "how does this code
                # work / where is it used / what breaks if I change it" answer.
                # Composes the call graph (KG-2.100), similar-code (KG-2.101),
                # routes (KG-2.102), change-coupling (KG-2.104), CONCEPT: markers
                # and docs into ONE grounded explanation with file:line citations,
                # so the agent queries the KG instead of grep-then-read. `query` =
                # the question/area/symbol; `target` = intent (how|usage|impact);
                # `node_id` = optional exact :Code anchor; `top_k`/`depth` budget.
                import json as _json

                from agent_utilities.knowledge_graph.retrieval.code_context import (
                    build_code_context,
                )

                cross = target.strip().lower().endswith("+xrepo")
                intent = (target or "how").strip().lower().replace(
                    "+xrepo", ""
                ) or "how"
                result = build_code_context(
                    engine,
                    query=query,
                    intent=intent,
                    node_id=node_id,
                    top_k=top_k,
                    depth=depth,
                    cross_repo=cross or intent == "usage",
                )
                return EvidenceBundle.from_code_context_answer(result).model_dump_json()
            elif action == "executable_rag":
                # CONCEPT:AU-KG.retrieval.memory-first-retrieval — the executable multi-hop RAG
                # interpreter, exposed over MCP for the first time (previously library-only).
                # `query` = the question; `top_k` = retrieval width per step; `target`="planner"
                # opts into LLM plan synthesis (default: the deterministic linear plan). Always
                # returns an EvidenceBundle (no legacy consumer to keep byte-identical, so this
                # defaults straight to the wrapped shape — CONCEPT:evidence-bundle-envelope).
                import json as _json

                from agent_utilities.knowledge_graph.retrieval.hybrid_retriever import (
                    HybridRetriever,
                )

                if not query.strip():
                    return "Error: executable_rag needs a question in `query`."
                use_planner = (target or "").strip().lower() == "planner"
                retriever = HybridRetriever(engine)
                rag_result = retriever.retrieve_executable(
                    query, top_k=top_k, use_planner=use_planner
                )
                bundle = EvidenceBundle.from_rag_result(rag_result)
                return _json.dumps(bundle.model_dump(), default=str)
            elif action == "cross_repo_usages":
                # CONCEPT:AU-KG.retrieval.every-usage-published-symbol — every usage of a published symbol across the
                # whole fleet in one query (name-anchored callers grouped by repo).
                # `query`/`target` = the symbol name; `top_k` = max usages.
                import json as _json

                from agent_utilities.knowledge_graph.retrieval.code_context import (
                    cross_repo_usages,
                )

                symbol = (query or target or "").strip()
                if not symbol:
                    return "Error: cross_repo_usages needs a symbol name in `query`."
                return _json.dumps(
                    cross_repo_usages(engine, symbol, limit=top_k or 200),
                    default=str,
                )
            elif action == "code_metrics":
                # CONCEPT:AU-KG.retrieval.structural-analytics — Graphify-style structural analytics over the
                # :Code call/inheritance subgraph: god nodes (degree hubs), Louvain
                # communities (via the engine's ephemeral detector KG-2.58),
                # surprising cross-community connections, and language/relation/
                # confidence distributions. `target` = optional scope substring
                # (file_path / source_system) to focus one repo; `top_k` = how many
                # god nodes / communities / bridges to surface. Reuses the durable
                # resolved graph — not a one-shot NetworkX notebook.
                import json as _json

                from agent_utilities.knowledge_graph.retrieval.code_metrics import (
                    build_code_metrics,
                )

                return _json.dumps(
                    build_code_metrics(
                        engine, scope=(target or query).strip(), top_k=top_k
                    ),
                    default=str,
                )
            elif action == "arch_report":
                # CONCEPT:AU-KG.retrieval.architecture-report — a regenerable architecture report (the
                # GRAPH_REPORT.md analog): summary, god nodes, community hubs,
                # surprising connections and dependency cycles, rendered as Markdown
                # plus structured metrics and persisted as an ArchitectureReport node.
                # `target` = optional scope substring; `top_k` = section sizes.
                import json as _json

                from agent_utilities.knowledge_graph.retrieval.code_metrics import (
                    build_arch_report,
                )

                scope = (target or query).strip()
                arch_report: dict[str, Any] = build_arch_report(
                    engine, scope=scope, top_k=top_k
                )
                # Persist the report as a durable node (best-effort) so it is
                # queryable + refreshable, exceeding Graphify's static file.
                if arch_report.get("status") == "ok":
                    try:
                        rid = f"arch_report:{scope or 'all'}"
                        engine.add_node(
                            rid,
                            {
                                "label": "ArchitectureReport",
                                "scope": scope or "all",
                                "markdown": arch_report["markdown"],
                                "node_count": arch_report["metrics"]["nodes"],
                                "community_count": arch_report["metrics"][
                                    "community_count"
                                ],
                            },
                        )
                        arch_report["report_node_id"] = rid
                    except Exception as _e:  # noqa: BLE001
                        arch_report["persist_warning"] = public_error_text(_e)
                return _json.dumps(arch_report, default=str)
            elif action == "explain":
                # CONCEPT:AU-KG.retrieval.route-question-its-domain — the universal context plane: route a question
                # to its DOMAIN provider (code | ops | …) and return one grounded,
                # cited answer. `query` = the question; `target` = "domain:intent"
                # (e.g. "ops:why", "code:usage") or just an intent (domain inferred);
                # empty target/domain infers both. This is the cockpit: more domains
                # = more providers on this one plane, not new subsystems.
                import json as _json

                from agent_utilities.knowledge_graph.retrieval.context_plane import (
                    list_context_domains,
                    synthesize_context,
                )

                spec = (target or "").strip()
                if spec in ("", "domains", "list"):
                    domain, intent = "", ""
                    if spec in ("domains", "list"):
                        return _json.dumps(
                            {"status": "ok", "domains": list_context_domains()}
                        )
                elif ":" in spec:
                    domain, _, intent = spec.partition(":")
                else:
                    domain, intent = "", spec  # treat a bare target as the intent
                return _json.dumps(
                    synthesize_context(
                        engine,
                        domain=domain,
                        query=query,
                        intent=intent,
                        node_id=node_id,
                        top_k=top_k,
                        depth=depth,
                    ),
                    default=str,
                )
            # ── KG-2.316/2.318: memory→weights distillation EXPORT + LIVE DS-MCP
            # dispatch (train_model over graph_workflows) + graph_jobs status poll ──
            elif action == "distill_memory":
                import json as _json

                from agent_utilities.knowledge_graph.memory.weights_distillation import (
                    distill_memory_to_weights,
                )

                # `query` may carry a JSON params object (base_model/scopes/method/
                # adapter_rank/time_window_days/target_entities/submit/…, or
                # `poll_job_id` to read a submitted job's live train state back —
                # CONCEPT:AU-KG.memory.live-data-science-mcp); `target` is the base model shorthand; `top_k`
                # overrides max_examples.
                distill_params: dict[str, Any] = {}
                q = (query or "").strip()
                if q.startswith("{"):
                    try:
                        loaded = _json.loads(q)
                        if isinstance(loaded, dict):
                            distill_params = loaded
                    except (TypeError, ValueError):
                        distill_params = {}
                if isinstance(target, str) and target:
                    distill_params.setdefault("base_model", target)
                if isinstance(top_k, int) and top_k and top_k != 10:
                    distill_params.setdefault("max_examples", top_k)
                submit = bool(distill_params.pop("submit", False))
                return _json.dumps(
                    distill_memory_to_weights(
                        engine, params=distill_params, submit=submit
                    ),
                    default=str,
                )
            else:
                return f"Error: Unknown analyze action '{action}'"
        except Exception as e:
            return public_error_text(e)

    global _analysis_core_handler
    _analysis_core_handler = _run_analysis_action

    @mcp.tool(
        name="graph_analyze",
        description=(
            "Structural and operational KG analysis. Actions: inspect | "
            "enrichment_coverage | process_writeback | placement_plan | "
            "infra_sweep | security_scan. Use graph_code, graph_research, "
            "graph_evaluate, graph_explain, or graph_observe for their focused domains. "
            "Returns the sole typed EvidenceBundle response."
        ),
        tags=["graph-os", "analyze"],
    )
    async def graph_analyze(
        action: str = Field(
            default="inspect",
            description="inspect | enrichment_coverage | process_writeback | placement_plan | infra_sweep | security_scan",
        ),
        query: str = Field(default="", description="Query or path for the analysis."),
        top_k: int = Field(default=10, description="Result or complexity bound."),
        node_id: str = Field(default="", description="Optional anchor node id."),
        depth: int = Field(default=2, description="Traversal depth."),
        target: str = Field(default="", description="Analysis target."),
    ) -> EvidenceBundle:
        allowed = {
            "inspect",
            "enrichment_coverage",
            "process_writeback",
            "placement_plan",
            "infra_sweep",
            "security_scan",
        }
        if action not in allowed:
            return EvidenceBundle.from_payload(
                {
                    "error": "action belongs to a focused graph tool",
                    "action": action,
                },
                operation="graph_analyze",
            )
        return await execute_focused_analysis(
            action=action,
            query=query,
            top_k=top_k,
            node_id=node_id,
            depth=depth,
            target=target,
        )

    kg_server.REGISTERED_TOOLS["graph_analyze"] = graph_analyze

    @mcp.tool(
        name="graph_orchestrate",
        description=(
            "Execute one named agent against a task through the governed graph-os delegation "
            "runtime. Returns the output, run/session handles, tool-call provenance linkage, "
            "and execution-flow Mermaid diagram when available."
        ),
        tags=["graph-os", "orchestrate", "agent"],
    )
    async def graph_orchestrate(
        task: str = Field(default="", description="Task for the delegated agent."),
        agent_name: str = Field(default="", description="Registered agent name."),
        max_steps: int = Field(
            default=30, description="Maximum delegated tool-loop steps."
        ),
        context: str = Field(
            default="",
            description="Curated inline context injected into the delegated agent.",
        ),
        budget_tokens: int = Field(
            default=0,
            description="Hard total-token budget; zero uses the runtime default.",
        ),
        context_ref: str = Field(
            default="",
            description="Persisted ContextBlob id to resolve and inject.",
        ),
        allowed_tools: str = Field(
            default="",
            description="Comma-separated least-privilege tool allow-list.",
        ),
        cred_ref: str = Field(
            default="",
            description="Reference to an ephemeral credential in the secrets backend.",
        ),
        open_channel: bool = Field(
            default=False,
            description="Open a native bidirectional message channel for this run.",
        ),
        reasoning_effort: str = Field(
            default="",
            description="CONCEPT:AU-ORCH.execution.delegation-reasoning-off — reasoning is an OPT-IN capability "
            "(like RLM). A delegated agent runs with chain-of-thought OFF by default (deterministic "
            "tool loops don't need it and it stacks ~18x per-turn latency). Set an effort "
            "('low'/'medium'/'high') to turn reasoning ON for an execution that genuinely needs "
            "deliberation (action='execute_agent'). Empty = off / inherit the model's setting.",
        ),
        model_class: str = Field(
            default="standard",
            description="Required configured model class: economy | standard.",
        ),
        response_format: ResponseFormat = Field(
            default="text",
            description=(
                "Response contract: text for ordinary prose or json for one "
                "Pydantic-validated JSON object."
            ),
        ),
    ) -> str:
        """Execute a single governed agent delegation."""
        response_format = validate_response_format(response_format)
        engine = kg_server._get_engine()
        if engine is None:
            return "Error: IntelligenceGraphEngine not active."
        try:
            from agent_utilities.orchestration.manager import Orchestrator

            result = await Orchestrator(engine).execute_agent(
                agent_name=agent_name,
                task=task,
                max_steps=max_steps,
                return_mermaid=True,
                context=context or None,
                budget_tokens=budget_tokens or None,
                context_ref=context_ref or None,
                allowed_tools=(
                    [name.strip() for name in allowed_tools.split(",") if name.strip()]
                    or None
                ),
                cred_ref=cred_ref or None,
                open_channel=open_channel,
                reasoning_effort=reasoning_effort or None,
                model_class=model_class,
                response_format=response_format,
            )
            try:
                payload = json.loads(result)
            except (TypeError, ValueError):
                payload = {"output": str(result), "mermaid": None}
            if not isinstance(payload, dict):
                payload = {"output": str(result), "mermaid": None}
            payload.setdefault("output", "")
            payload.setdefault("mermaid", None)
            return json.dumps(payload, default=str)
        except PermissionError:
            raise
        except Exception as exc:
            return public_error_text(exc)

    kg_server.REGISTERED_TOOLS["graph_orchestrate"] = graph_orchestrate

    @mcp.tool(
        name="graph_configure",
        description="Manage backend configurations, system credentials, and tool registration within the unified agent ecosystem.",
        tags=["graph-os", "configure"],
    )
    def graph_configure(
        action: str = Field(
            default="register_mcp",
            description=(
                "Configuration operation. Core actions: set_secret, vault_sync, "
                "register_mcp, install_hooks, uninstall_hooks, harness_fence, "
                "schema_pack, schema_candidates, add_connection, remove_connection, "
                "list_connections, mirror_status, reconcile, "
                "generate_config, config_doctor, config_reference, get_config, "
                "set_config, list_config, system_doctor, health, and preflight. Universal "
                "external-source lifecycle actions are discover_connection_schema, "
                "propose_connection_mapping, approve_connection_mapping, "
                "connection_mapping_status, external_graph_doctor, and "
                "ingest_connection. Durable connection declarations accept neutral "
                "aliases and runtime secret references only; endpoint, credential, "
                "identity, query, TLS material, and local-path literals are rejected. "
                "GraphQL sources use a read-only runtime adapter; "
                "their connection, mapping, auth, TLS, and variables documents remain "
                "separate refs, and every ingest rechecks the approved schema and "
                "mapping-policy digests."
            ),
        ),
        config_key: str = Field(
            default="",
            description="The key or ID of the configuration/secret (for 'schema_pack', the pack name e.g. 'research-state'; for connection actions, the connection name).",
        ),
        config_value: str = Field(
            default="",
            description="JSON string containing the payload or secret value.",
        ),
    ) -> str:
        """Manage backend configurations and abstract credentials. Allows dynamic registry updates and credential injection during agent provisioning."""
        try:
            if action == "set_secret":
                from agent_utilities.security.secrets_client import (
                    create_secrets_client,
                )
                from agent_utilities.security.xai_auth import get_secrets_client_for_xai

                if config_key.startswith("xai/"):
                    client = get_secrets_client_for_xai()
                else:
                    client = create_secrets_client()
                client.set(config_key, config_value)
                return json.dumps(
                    {"status": "success", "action": "set_secret", "stored": True}
                )
            if action == "vault_sync":
                # CONCEPT:AU-OS.deployment.vault-seed-service — read-existing + seed a service's secrets.
                # config_key=service; config_value=JSON
                # {"env_keys":[...],"values":{KEY:VAL},"overwrite":bool}.
                from agent_utilities.security.secrets_client import (
                    create_secrets_client,
                )

                payload = json.loads(config_value) if config_value else {}
                env_keys = payload.get("env_keys", [])
                client = create_secrets_client()
                result = client.vault_sync(
                    config_key,
                    env_keys,
                    values=payload.get("values"),
                    overwrite=bool(payload.get("overwrite", False)),
                )
                result.update({"status": "success", "action": "vault_sync"})
                return json.dumps(result)
            if action == "register_mcp":
                try:
                    _register_mcp_server(config_key, config_value)
                except PermissionError:
                    raise
                except Exception as exc:
                    return json.dumps(
                        {
                            "error": "MCP registration rejected",
                            "error_type": type(exc).__name__,
                        }
                    )
                return json.dumps(
                    {
                        "status": "success",
                        "action": "register_mcp",
                        "server": config_key,
                    }
                )
            # ── CONCEPT:AU-KG.backend.multi-connection-registry: Named multi-connection graph registry ──
            if action in (
                "add_connection",
                "remove_connection",
                "list_connections",
            ):
                registry = kg_server.get_connection_registry()
                if action == "list_connections":
                    return json.dumps(registry.status(), default=str)
                if not config_key:
                    return json.dumps(
                        {"error": f"config_key (connection name) required for {action}"}
                    )
                if action == "add_connection":
                    try:
                        spec = json.loads(config_value) if config_value else {}
                    except Exception:
                        return json.dumps(
                            {"error": "config_value must contain valid JSON"}
                        )
                    if not isinstance(spec, dict):
                        return json.dumps(
                            {
                                "error": "config_value must be a JSON object (backend spec)"
                            }
                        )
                    declared_name = spec.pop("name", None)
                    if declared_name not in (None, config_key):
                        return json.dumps(
                            {
                                "error": (
                                    "connection name is authoritative in config_key; "
                                    "a payload name cannot select another alias"
                                )
                            }
                        )
                    if not spec:
                        # AgentConfig is the reference-only declarative plane.
                        # An operator may activate one of those declarations by
                        # alias without copying any profile reference into MCP
                        # arguments or traces.
                        try:
                            candidate = _configured_external_graph_declaration(
                                config_key
                            )
                            if candidate:
                                candidate.pop("name", None)
                                candidate["role"] = str(candidate.get("role") or "read")
                                spec = candidate
                        except Exception as exc:
                            return json.dumps(
                                {
                                    "error": "configured connection lookup failed",
                                    "error_type": type(exc).__name__,
                                }
                            )
                    if not spec:
                        return json.dumps(
                            {
                                "error": (
                                    "add_connection requires a reference-only JSON "
                                    "declaration or a matching AgentConfig alias"
                                )
                            }
                        )
                    try:
                        from agent_utilities.knowledge_graph.core.connection_registry import (
                            validate_persistable_connection_spec,
                        )

                        validate_persistable_connection_spec(spec)
                    except Exception as exc:
                        return json.dumps(
                            {
                                "error": "connection registration is not persistence-safe",
                                "error_type": type(exc).__name__,
                            }
                        )
                    try:
                        name = registry.register(config_key, spec)
                    except Exception as exc:
                        return json.dumps(
                            {
                                "error": "connection registration failed",
                                "error_type": type(exc).__name__,
                            }
                        )
                    # CONCEPT:AU-KG.backend.connection-registry — persist the connection list to config.json so
                    # it survives restart (re-seeded from config.kg_connections).
                    from agent_utilities.core.config import save_config_item

                    save_config_item("kg_connections", registry.export_specs())
                    return json.dumps(
                        {
                            "status": "success",
                            "action": action,
                            "connection": name,
                            "role": registry.role(name),
                            "persisted": True,
                        }
                    )
                if action == "remove_connection":
                    removed = registry.remove(config_key)
                    if removed:
                        from agent_utilities.core.config import save_config_item

                        save_config_item("kg_connections", registry.export_specs())
                    return json.dumps(
                        {
                            "status": "success" if removed else "not_found",
                            "action": action,
                            "connection": config_key,
                            "persisted": bool(removed),
                        }
                    )
            # ── CONCEPT:AU-KG.backend.multi-connection-registry: discover/profile an external graph + map ──
            if action in (
                "approve_connection_mapping",
                "connection_mapping_status",
                "discover_connection_schema",
                "external_graph_doctor",
                "profile_connection",
                "ingest_connection",
                "propose_connection_mapping",
            ):
                if not config_key:
                    return json.dumps(
                        {"error": f"config_key (connection name) required for {action}"}
                    )
                registry = kg_server.get_connection_registry()
                if action in {
                    "approve_connection_mapping",
                    "connection_mapping_status",
                }:
                    from agent_utilities.knowledge_graph.ingestion.external_graph_schema import (
                        approve_mapping_profile,
                        mapping_profile_status,
                    )
                    from agent_utilities.security.secrets_client import (
                        create_secrets_client,
                    )

                    store = create_secrets_client()
                    if action == "connection_mapping_status":
                        if config_value:
                            return json.dumps(
                                {"error": "connection_mapping_status takes no payload"}
                            )
                        try:
                            from agent_utilities.knowledge_graph.ingestion.external_graph_schema import (
                                normalize_backend_kind,
                            )

                            backend_kind = normalize_backend_kind(
                                registry.backend_kind(config_key)
                            )
                            if backend_kind == "graphql":
                                from agent_utilities.knowledge_graph.ingestion.graphql_connection import (
                                    GraphQLSourceAdapter,
                                    graphql_mapping_profile_status,
                                )

                                source = registry.get_engine(config_key)
                                if not isinstance(source, GraphQLSourceAdapter):
                                    raise TypeError("registered source is not GraphQL")
                                status = graphql_mapping_profile_status(
                                    source,
                                    connection=config_key,
                                    secret_store=store,
                                )
                            else:
                                declaration = _configured_external_graph_declaration(
                                    config_key
                                )
                                _policy, current_policy_digest = (
                                    _resolved_external_mapping_policy(
                                        store, declaration
                                    )
                                )
                                status = mapping_profile_status(
                                    config_key,
                                    secret_store=store,
                                    runtime_policy_digest=current_policy_digest,
                                )
                            return json.dumps(
                                status,
                                default=str,
                            )
                        except Exception as exc:
                            return json.dumps(
                                {
                                    "error": "mapping status lookup failed",
                                    "error_type": type(exc).__name__,
                                }
                            )
                    try:
                        options = json.loads(config_value) if config_value else {}
                        if not isinstance(options, dict):
                            raise ValueError("approval payload must be an object")
                        if "approver_ref" in options:
                            return json.dumps(
                                {
                                    "error": "approver identity is derived from authenticated context"
                                }
                            )
                        if set(options).difference(
                            {
                                "mapping_digest",
                                "proposal_id",
                                "proposal_version",
                                "schema_digest",
                            }
                        ):
                            return json.dumps(
                                {
                                    "error": (
                                        "mapping approval accepts only the exact "
                                        "proposal version and digest tuple"
                                    )
                                }
                            )
                        from agent_utilities.knowledge_graph.core.session import (
                            resolve_session,
                        )

                        approval_session = resolve_session(required_scope="kg:admin")
                        approval_actor = (
                            approval_session.actor.actor_id
                            if approval_session.actor is not None
                            else "authenticated-operator"
                        )
                        result = approve_mapping_profile(
                            connection=config_key,
                            proposal_id=str(options.get("proposal_id") or ""),
                            proposal_version=int(options.get("proposal_version") or 0),
                            schema_digest=str(options.get("schema_digest") or ""),
                            mapping_digest=str(options.get("mapping_digest") or ""),
                            secret_store=store,
                            approver_ref=approval_actor,
                        )
                    except Exception as exc:
                        return json.dumps(
                            {
                                "error": "mapping approval failed",
                                "error_type": type(exc).__name__,
                            }
                        )
                    return json.dumps(result, default=str)
                if action == "ingest_connection":
                    try:
                        options = json.loads(config_value) if config_value else {}
                    except Exception:
                        return json.dumps(
                            {"error": "config_value must be a JSON object"}
                        )
                    if not isinstance(options, dict):
                        return json.dumps(
                            {"error": "config_value must be a JSON object"}
                        )
                    try:
                        declared = _configured_external_graph_declaration(config_key)
                    except Exception as exc:
                        return json.dumps(
                            {
                                "error": "configured source lookup failed",
                                "error_type": type(exc).__name__,
                            }
                        )
                    try:
                        from agent_utilities.knowledge_graph.ingestion.external_graph_schema import (
                            normalize_backend_kind,
                        )

                        backend_kind = normalize_backend_kind(
                            registry.backend_kind(config_key)
                        )
                    except Exception as exc:
                        return json.dumps(
                            {
                                "error": "external source declaration is invalid",
                                "error_type": type(exc).__name__,
                            }
                        )
                    if backend_kind == "graphql":
                        allowed = {
                            "contextual",
                            "dry_run",
                            "max_depth",
                            "max_records",
                            "max_types",
                            "operation",
                            "variables_ref",
                        }
                        if set(options).difference(allowed) or "variables" in options:
                            return json.dumps(
                                {
                                    "error": (
                                        "GraphQL ingestion accepts bounded policy choices "
                                        "and a variables_ref only"
                                    )
                                }
                            )
                        try:
                            source = registry.get_engine(config_key)
                            from agent_utilities.knowledge_graph.ingestion.graphql_connection import (
                                GraphQLSourceAdapter,
                                ingest_registered_graphql,
                            )
                            from agent_utilities.security.secrets_client import (
                                create_secrets_client,
                            )

                            if not isinstance(source, GraphQLSourceAdapter):
                                raise TypeError("registered source is not GraphQL")
                            result = ingest_registered_graphql(
                                registry.get_engine(None),
                                source,
                                connection=config_key,
                                secret_store=create_secrets_client(),
                                operation=str(
                                    options.get("operation")
                                    or declared.get("ingest_operation")
                                    or source.ingest_operation
                                    or ""
                                ),
                                variables_ref=str(
                                    options.get("variables_ref")
                                    or declared.get("variables_ref")
                                    or ""
                                ),
                                max_records=int(
                                    options.get("max_records")
                                    or declared.get("ingest_max_records")
                                    or source.ingest_max_records
                                    or 1_000
                                ),
                                max_types=int(
                                    options.get("max_types")
                                    or declared.get("discovery_max_types")
                                    or source.discovery_max_types
                                    or 200
                                ),
                                max_depth=int(
                                    options.get("max_depth")
                                    or declared.get("discovery_max_depth")
                                    or source.discovery_max_depth
                                    or 6
                                ),
                                contextual=bool(
                                    options.get(
                                        "contextual",
                                        declared.get("contextual", source.contextual),
                                    )
                                ),
                                dry_run=bool(options.get("dry_run", False)),
                            )
                        except Exception as exc:  # noqa: BLE001 — safe type only
                            return json.dumps(
                                {
                                    "error": "GraphQL document ingestion failed",
                                    "error_type": type(exc).__name__,
                                }
                            )
                        return json.dumps(result, default=str)
                    allowed = {
                        "classification",
                        "dry_run",
                        "legal_hold",
                        "max_records",
                        "retention",
                        "source_alias",
                        "tenant",
                    }
                    if set(options).difference(allowed):
                        return json.dumps(
                            {
                                "error": (
                                    "external graph ingestion accepts only aliases and "
                                    "bounded governance choices; profiles, queries, "
                                    "variables, ontology, endpoints, paths, and "
                                    "credentials stay behind configured runtime refs"
                                )
                            }
                        )
                    from agent_utilities.knowledge_graph.ingestion.external_graph import (
                        ExternalGraphIngestionRequest,
                        ingest_registered_graph,
                    )
                    from agent_utilities.security.secrets_client import (
                        create_secrets_client,
                    )

                    try:
                        _runtime_policy, runtime_policy_digest = (
                            _resolved_external_mapping_policy(
                                create_secrets_client(), declared
                            )
                        )
                        request = ExternalGraphIngestionRequest(
                            connection=config_key,
                            source_alias=str(
                                options.get("source_alias")
                                or declared.get("source_alias")
                                or config_key
                            ),
                            profile_ref="",
                            variables={},
                            runtime_policy_digest=runtime_policy_digest,
                            max_records=int(
                                options.get("max_records")
                                or declared.get("ingest_max_records")
                                or 1_000
                            ),
                            page_size=int(declared.get("ingest_page_size") or 500),
                            max_pages=int(declared.get("ingest_max_pages") or 100),
                            max_row_bytes=int(
                                declared.get("ingest_max_row_bytes") or 1_048_576
                            ),
                            max_total_bytes=int(
                                declared.get("ingest_max_total_bytes") or 16_777_216
                            ),
                            max_nesting_depth=int(
                                declared.get("ingest_max_nesting_depth") or 16
                            ),
                            max_collection_items=int(
                                declared.get("ingest_max_collection_items") or 10_000
                            ),
                            sync_mode=str(declared.get("sync_mode") or "auto"),
                            reconcile_deletions=bool(
                                declared.get("reconcile_deletions", True)
                            ),
                            allow_empty_snapshot=bool(
                                declared.get("allow_empty_snapshot", False)
                            ),
                            classification=options.get(
                                "classification", "confidential"
                            ),
                            retention=str(options.get("retention") or "P30D"),
                            legal_hold=bool(options.get("legal_hold", False)),
                            tenant=str(options.get("tenant") or ""),
                            dry_run=bool(options.get("dry_run", False)),
                        )
                        result = ingest_registered_graph(
                            registry.get_engine(None), registry, request
                        )
                    except Exception as exc:  # noqa: BLE001 — safe type only
                        return json.dumps(
                            {
                                "error": "external graph ingestion failed",
                                "error_type": type(exc).__name__,
                            }
                        )
                    return json.dumps(result, default=str)
                try:
                    ext_engine = registry.get_engine(config_key)
                except Exception as e:
                    return json.dumps(
                        {
                            "error": "external graph connection unavailable",
                            "error_type": type(e).__name__,
                        }
                    )
                try:
                    options = json.loads(config_value) if config_value else {}
                except Exception:
                    return json.dumps({"error": "config_value must be a JSON object"})
                if not isinstance(options, dict):
                    return json.dumps({"error": "config_value must be a JSON object"})
                from agent_utilities.knowledge_graph.ingestion.external_graph_schema import (
                    discover_external_schema,
                    external_graph_readiness,
                    governed_semantic_mapping_enricher,
                    propose_mapping_profile,
                )
                from agent_utilities.security.secrets_client import (
                    create_secrets_client,
                )

                backend = registry.backend_kind(config_key)
                try:
                    from agent_utilities.knowledge_graph.ingestion.external_graph_schema import (
                        normalize_backend_kind,
                    )

                    backend_kind = normalize_backend_kind(backend)
                except Exception as exc:
                    return json.dumps(
                        {
                            "error": "external source declaration is invalid",
                            "error_type": type(exc).__name__,
                        }
                    )
                connector_config: dict[str, Any] = {}
                try:
                    connector_config = _configured_external_graph_declaration(
                        config_key
                    )
                except Exception as exc:
                    return json.dumps(
                        {
                            "error": "configured source lookup failed",
                            "error_type": type(exc).__name__,
                        }
                    )
                max_types = max(
                    1,
                    min(
                        int(
                            options.get("max_types")
                            or connector_config.get("discovery_max_types")
                            or getattr(ext_engine, "discovery_max_types", None)
                            or 200
                        ),
                        500,
                    ),
                )
                max_depth = max(
                    1,
                    min(
                        int(
                            options.get("max_depth")
                            or connector_config.get("discovery_max_depth")
                            or getattr(ext_engine, "discovery_max_depth", None)
                            or 6
                        ),
                        12,
                    ),
                )
                if (
                    backend_kind == "graphql"
                    and action
                    in {
                        "discover_connection_schema",
                        "external_graph_doctor",
                        "profile_connection",
                    }
                    and set(options).difference({"max_depth", "max_types"})
                ):
                    return json.dumps(
                        {
                            "error": (
                                "GraphQL discovery actions accept bounded discovery "
                                "limits only; source material comes from runtime refs"
                            )
                        }
                    )
                if (
                    backend_kind != "graphql"
                    and action
                    in {
                        "discover_connection_schema",
                        "external_graph_doctor",
                        "profile_connection",
                    }
                    and set(options).difference({"max_types"})
                ):
                    return json.dumps(
                        {
                            "error": (
                                "external graph discovery actions accept a bounded "
                                "max_types value only; source material stays behind "
                                "configured runtime refs"
                            )
                        }
                    )
                if action in {"discover_connection_schema", "profile_connection"}:
                    try:
                        if backend_kind == "graphql":
                            schema, capabilities, _accepted = ext_engine.discover(
                                max_types=max_types, max_depth=max_depth
                            )
                        else:
                            schema, capabilities = discover_external_schema(
                                ext_engine, backend=backend, max_types=max_types
                            )
                    except Exception as exc:
                        return json.dumps(
                            {
                                "error": "external schema discovery failed",
                                "error_type": type(exc).__name__,
                            }
                        )
                    return json.dumps(
                        {
                            "status": "success",
                            "connection": config_key,
                            "schema": schema.public_dict(),
                            "capabilities": capabilities.public_dict(),
                        },
                        default=str,
                    )
                store = create_secrets_client()
                runtime_policy: dict[str, Any] = {}
                runtime_policy_digest = ""
                if backend_kind != "graphql":
                    try:
                        runtime_policy, runtime_policy_digest = (
                            _resolved_external_mapping_policy(store, connector_config)
                        )
                    except Exception as exc:
                        return json.dumps(
                            {
                                "error": "secret-backed mapping policy resolution failed",
                                "error_type": type(exc).__name__,
                            }
                        )
                if action == "external_graph_doctor":
                    if backend_kind == "graphql":
                        from agent_utilities.knowledge_graph.ingestion.graphql_connection import (
                            graphql_source_readiness,
                        )

                        return json.dumps(
                            graphql_source_readiness(
                                ext_engine,
                                connection=config_key,
                                secret_store=store,
                                max_types=max_types,
                                max_depth=max_depth,
                            ),
                            default=str,
                        )
                    return json.dumps(
                        external_graph_readiness(
                            ext_engine,
                            backend=backend,
                            connection=config_key,
                            secret_store=store,
                            runtime_policy_digest=runtime_policy_digest,
                            max_types=max_types,
                        ),
                        default=str,
                    )
                if backend_kind == "graphql":
                    if set(options).difference({"max_depth", "max_types"}):
                        return json.dumps(
                            {
                                "error": (
                                    "GraphQL mapping proposals resolve query, mapping, "
                                    "governance, auth, and TLS policy from runtime refs"
                                )
                            }
                        )
                    try:
                        from agent_utilities.knowledge_graph.ingestion.graphql_connection import (
                            GraphQLSourceAdapter,
                            propose_graphql_mapping_profile,
                        )

                        if not isinstance(ext_engine, GraphQLSourceAdapter):
                            raise TypeError("registered source is not GraphQL")
                        result = propose_graphql_mapping_profile(
                            ext_engine,
                            connection=config_key,
                            source_alias=ext_engine.source_alias,
                            secret_store=store,
                            max_types=max_types,
                            max_depth=max_depth,
                        )
                    except Exception as exc:
                        return json.dumps(
                            {
                                "error": "GraphQL mapping proposal failed",
                                "error_type": type(exc).__name__,
                            }
                        )
                    return json.dumps(result, default=str)
                if set(options).difference({"max_types", "source_alias"}):
                    return json.dumps(
                        {
                            "error": (
                                "external graph mapping proposals accept aliases and "
                                "bounded discovery choices only; mapping, ontology, "
                                "endpoint, path, identity, and credential material "
                                "must come from configured runtime refs"
                            )
                        }
                    )
                max_types = int(
                    connector_config.get("discovery_max_types") or max_types
                )
                from agent_utilities.knowledge_graph.core.connection_profiler import (
                    _our_ontology_vocabulary,
                )

                authority = registry.get_engine(None)
                vocabulary = _our_ontology_vocabulary(authority, None)
                try:
                    semantic_enricher = None
                    semantic_context_session = None
                    if bool(connector_config.get("semantic_mapping", False)):
                        from agent_utilities.knowledge_graph.core.session import (
                            resolve_session,
                        )

                        semantic_context_session = resolve_session(
                            required_scope="kg:read"
                        )
                        semantic_enricher = governed_semantic_mapping_enricher
                    result = propose_mapping_profile(
                        ext_engine,
                        backend=backend,
                        connection=config_key,
                        source_alias=str(
                            connector_config.get("source_alias")
                            or options.get("source_alias")
                            or config_key
                        ),
                        ontology_classes=vocabulary,
                        secret_store=store,
                        access=(
                            runtime_policy.get("access")
                            if isinstance(runtime_policy.get("access"), dict)
                            else None
                        ),
                        property_allowlist=(
                            list(runtime_policy.get("property_allowlist") or []) or None
                        ),
                        edge_property_allowlist=(
                            list(runtime_policy.get("edge_property_allowlist") or [])
                            or None
                        ),
                        type_overrides=(
                            runtime_policy.get("type_overrides")
                            if isinstance(runtime_policy.get("type_overrides"), dict)
                            else None
                        ),
                        edge_type_overrides=(
                            runtime_policy.get("edge_type_overrides")
                            if isinstance(
                                runtime_policy.get("edge_type_overrides"), dict
                            )
                            else None
                        ),
                        identity_property=str(
                            runtime_policy.get("identity_property") or ""
                        )
                        or None,
                        runtime_policy_digest=runtime_policy_digest,
                        page_size=int(connector_config.get("ingest_page_size") or 500),
                        max_pages=int(connector_config.get("ingest_max_pages") or 100),
                        max_row_bytes=int(
                            connector_config.get("ingest_max_row_bytes") or 1_048_576
                        ),
                        max_total_bytes=int(
                            connector_config.get("ingest_max_total_bytes") or 16_777_216
                        ),
                        max_nesting_depth=int(
                            connector_config.get("ingest_max_nesting_depth") or 16
                        ),
                        max_collection_items=int(
                            connector_config.get("ingest_max_collection_items")
                            or 10_000
                        ),
                        sync_mode=str(connector_config.get("sync_mode") or "auto"),
                        reconcile_deletions=bool(
                            connector_config.get("reconcile_deletions", True)
                        ),
                        allow_empty_snapshot=bool(
                            connector_config.get("allow_empty_snapshot", False)
                        ),
                        max_types=max_types,
                        semantic_enricher=semantic_enricher,
                        context_session=semantic_context_session,
                    )
                except Exception as exc:
                    return json.dumps(
                        {
                            "error": "mapping proposal failed",
                            "error_type": type(exc).__name__,
                        }
                    )
                return json.dumps(result, default=str)
            # ── CONCEPT:AU-KG.backend.mirror-health-repair: Concurrent N-way mirroring health/repair ──
            if action in ("mirror_status", "reconcile"):
                from agent_utilities.knowledge_graph.backends import (
                    get_active_backend,
                )
                from agent_utilities.knowledge_graph.backends.fanout_backend import (
                    FanOutBackend,
                )

                backend = get_active_backend()
                # Locate the FanOutBackend created automatically when one or more
                # projections are configured. Also unwrap a BrainGuarded proxy.
                cand = getattr(backend, "inner", backend)
                fan = cand if isinstance(cand, FanOutBackend) else None
                if fan is None:
                    return json.dumps(
                        {
                            "error": "No fanout projection active (configure "
                            "GRAPH_MIRROR_TARGETS or a role=mirror connection).",
                            "backend": type(backend).__name__,
                        }
                    )
                inner = fan
                if action == "mirror_status":
                    return json.dumps(inner.durability_stats(), default=str)
                # reconcile — full authority→mirror drift repair (config_key =
                # optional single mirror name; empty = all mirrors).
                return json.dumps(inner.reconcile(config_key or None), default=str)
            # ── CONCEPT:AU-KG.query.stardog-instance-data: Stardog instance-data push / pull / query ──
            if action in ("push_to_stardog", "pull_from_stardog", "stardog_sparql"):
                try:
                    opts = json.loads(config_value) if config_value else {}
                except Exception:
                    return json.dumps({"error": "config_value must contain valid JSON"})
                if not isinstance(opts, dict):
                    # stardog_sparql also accepts a bare query string in config_value.
                    if action == "stardog_sparql" and isinstance(config_value, str):
                        opts = {"query": config_value}
                    else:
                        return json.dumps(
                            {"error": "config_value must be a JSON object"}
                        )

                inline_connection_fields = {
                    "database",
                    "endpoint",
                    "password",
                    "username",
                }
                if inline_connection_fields.intersection(opts):
                    return json.dumps(
                        {
                            "error": (
                                "inline Stardog connection material is not accepted; "
                                "use a registered connection alias backed by secret references"
                            )
                        }
                    )

                def _resolve_stardog_backend():
                    """Resolve Stardog exclusively through a registered alias."""
                    name = config_key or opts.get("connection")
                    if not isinstance(name, str) or not name.strip():
                        raise ValueError("registered Stardog connection is required")
                    eng = kg_server.get_connection_registry().get_engine(name.strip())
                    be = getattr(eng, "backend", eng)
                    return getattr(be, "_authority", be)

                try:
                    sd_backend = _resolve_stardog_backend()
                except Exception as exc:
                    return json.dumps(
                        {
                            "error": "registered Stardog connection unavailable",
                            "error_type": type(exc).__name__,
                        }
                    )

                if action == "stardog_sparql":
                    query = opts.get("query")
                    if not query:
                        return json.dumps(
                            {"error": "config_value.query (a SPARQL string) required"}
                        )
                    return json.dumps(
                        {"results": sd_backend.execute_sparql(query)}, default=str
                    )

                authority = kg_server.get_connection_registry().get_engine(None)
                if action == "push_to_stardog":
                    from agent_utilities.knowledge_graph.integrations.stardog_sync import (  # noqa: E501
                        push_to_stardog,
                    )

                    return json.dumps(
                        push_to_stardog(
                            authority, sd_backend, sources=opts.get("sources")
                        ),
                        default=str,
                    )
                # pull_from_stardog
                from agent_utilities.knowledge_graph.integrations.stardog_sync import (
                    pull_from_stardog,
                )

                return json.dumps(
                    pull_from_stardog(
                        sd_backend,
                        authority,
                        graph_uri=opts.get("graph_uri"),
                        source=opts.get("source"),
                        limit=int(opts.get("limit", 10_000)),
                    ),
                    default=str,
                )
            # ── Database environment provisioning (Stardog + pg-age) ──
            if action in ("setup_databases", "verify_databases"):
                from agent_utilities.knowledge_graph.setup import (
                    setup_environment,
                    verify_postgres,
                )

                try:
                    opts = json.loads(config_value) if config_value else {}
                except Exception:
                    return json.dumps({"error": "config_value must contain valid JSON"})
                if not isinstance(opts, dict):
                    return json.dumps(
                        {"error": "config_value must be a JSON object of options"}
                    )
                if "dsn" in opts or "://" in (config_key or ""):
                    return json.dumps(
                        {
                            "error": (
                                "inline database endpoints are not accepted; configure "
                                "the runtime connection through a secret-backed profile"
                            )
                        }
                    )
                connection_profile_ref = opts.get("connection_profile_ref")
                if connection_profile_ref and not _runtime_reference(
                    connection_profile_ref
                ):
                    return json.dumps(
                        {
                            "error": (
                                "connection_profile_ref must be a runtime secret "
                                "reference"
                            )
                        }
                    )
                if config_key and config_key not in {"dev", "prod"}:
                    return json.dumps(
                        {
                            "error": (
                                "config_key must be a deployment profile alias; "
                                "database endpoints belong in the secret-backed runtime profile"
                            )
                        }
                    )
                if action == "verify_databases":
                    return json.dumps(
                        verify_postgres(connection_profile_ref),
                        default=str,
                    )
                # setup_databases — config_key is a profile shortcut ('dev'/'prod').
                profile = opts.get("profile") or config_key or "dev"
                return json.dumps(
                    setup_environment(
                        profile=profile,
                        postgres_mode=opts.get("postgres_mode", "managed_image"),
                        connection_profile_ref=connection_profile_ref,
                        sparql_target=opts.get("sparql_target"),
                        mirror_targets=opts.get("mirror_targets"),
                        do_backfill=opts.get("do_backfill", True),
                    ),
                    default=str,
                )
            # ── Full-deployment config: generate / validate / document ──
            if action in ("generate_config", "config_doctor", "config_reference"):
                from agent_utilities.deployment import (
                    config_doctor,
                    config_reference,
                    write_config,
                )

                try:
                    opts = json.loads(config_value) if config_value else {}
                except Exception:
                    return json.dumps({"error": "config_value must contain valid JSON"})
                if not isinstance(opts, dict):
                    return json.dumps({"error": "config_value must be a JSON object"})
                if action == "config_reference":
                    return json.dumps(config_reference(), default=str)
                # profile shortcut via config_key ('tiny'/'single-node-prod'/'enterprise')
                profile = opts.get("profile") or config_key or None
                if action == "generate_config":
                    if opts.get("out"):
                        return json.dumps({"error": "remote_path_not_allowed"})
                    return json.dumps(
                        write_config(
                            profile or "tiny",
                        ),
                        default=str,
                    )
                # config_doctor
                if opts.get("config"):
                    return json.dumps({"error": "remote_path_not_allowed"})
                return json.dumps(config_doctor(profile), default=str)
            # ── CONCEPT:AU-KG.backend.connection-registry: generic live config get / set / list ──
            if action in ("get_config", "set_config", "list_config"):
                from agent_utilities.deployment import (
                    config_reference,
                    is_restart_required,
                )

                known: dict[str, dict] = {}
                for section in config_reference():
                    for f in section.get("fields", []):
                        known[str(f.get("env") or "").upper()] = f

                if action == "list_config":
                    out = {}
                    for env_key, meta in known.items():
                        val = os.environ.get(env_key)
                        out[env_key] = (
                            "***"
                            if (_configuration_key_is_sensitive(env_key, meta) and val)
                            else val
                        )
                    return json.dumps({"config": out, "count": len(out)}, default=str)

                if not config_key:
                    return json.dumps(
                        {"error": f"config_key (env name) required for {action}"}
                    )
                env_key = config_key.upper()
                if env_key not in known:
                    return json.dumps(
                        {"error": "Unknown config key (see config_reference)"}
                    )
                if action == "get_config":
                    val = os.environ.get(env_key)
                    if _configuration_key_is_sensitive(env_key, known[env_key]) and val:
                        val = "***"
                    return json.dumps(
                        {
                            "key": env_key,
                            "value": val,
                            "restart_required": is_restart_required(env_key),
                        },
                        default=str,
                    )
                # set_config — persist to config.json + apply live (or flag restart).
                if _configuration_key_is_sensitive(env_key, known[env_key]):
                    if not env_key.endswith("_REF") or not _runtime_reference(
                        config_value
                    ):
                        return json.dumps(
                            {
                                "error": (
                                    "sensitive settings cannot be persisted inline; "
                                    "use the secret store and a reference-capable setting"
                                )
                            }
                        )
                parsed = config_value
                if config_value and config_value.strip()[:1] in '[{"':
                    try:
                        parsed = json.loads(config_value)
                    except Exception:
                        parsed = config_value
                from agent_utilities.core.config import save_config_item

                save_config_item(env_key, parsed)
                restart = is_restart_required(env_key)
                return json.dumps(
                    {
                        "status": "success",
                        "key": env_key,
                        "applied_live": not restart,
                        "restart_required": restart,
                    },
                    default=str,
                )
            # ── Authenticated runtime health (CONCEPT:AU-OS.config.two-surfaces-by-default) ──
            # The MCP twin of GET /health and GET /health/ready: dispatches into
            # the SAME shared core those unauthenticated HTTP routes use, so this
            # tool call, the REST route, and the gateway's own /health never
            # drift. Unlike the raw HTTP routes this goes through the normal
            # authenticated tool-dispatch path (_execute_tool's verified
            # GraphSession requirement) rather than being unauthenticated.
            if action == "health":
                from agent_utilities.observability.runtime_health import collect_health

                return json.dumps(collect_health(), default=str)
            # ── Holistic deployment health sweep (brew/flutter-doctor style) ──
            if action == "system_doctor":
                from agent_utilities.deployment import run_doctor

                try:
                    opts = json.loads(config_value) if config_value else {}
                except Exception:
                    return json.dumps({"error": "config_value must contain valid JSON"})
                if not isinstance(opts, dict):
                    return json.dumps({"error": "config_value must be a JSON object"})
                return json.dumps(
                    run_doctor(
                        opts.get("only"),
                        fix=opts.get("fix", False),
                        live=opts.get("live", False),
                    ),
                    default=str,
                )
            if action == "preflight":
                from agent_utilities.deployment.preflight import run_preflight

                profile = config_key or "tiny"
                try:
                    opts = json.loads(config_value) if config_value else {}
                except Exception:
                    return json.dumps({"error": "config_value must contain valid JSON"})
                if not isinstance(opts, dict):
                    return json.dumps({"error": "config_value must be a JSON object"})
                return json.dumps(
                    run_preflight(profile, opts.get("components")),
                    default=str,
                )
            # ── KG-2.7 / ECO-4.6: Memory Hook Management ──
            if action == "harness_fence":
                # CONCEPT:AU-OS.deployment.governance-derived-claude-code — write a governance-derived Claude Code
                # permission fence (settings.json + .claudeignore). config_key =
                # target Claude config dir (default $XDG_CONFIG_HOME/claude); config_value =
                # optional {"policy": path, "dry_run": bool}.
                try:
                    from pathlib import Path as _Path

                    from agent_utilities.claude_harness.claude_fence import write_fence
                    from agent_utilities.orchestration.action_policy import ActionPolicy

                    opts = json.loads(config_value) if config_value else {}
                    if not isinstance(opts, dict):
                        opts = {}
                    target = config_key or str(_Path.home() / ".claude")
                    policy_path = opts.get("policy")
                    policy = (
                        ActionPolicy(policy_path=policy_path)
                        if policy_path
                        else ActionPolicy()
                    )
                    return json.dumps(
                        write_fence(target, policy, dry_run=bool(opts.get("dry_run"))),
                        default=str,
                    )
                except PermissionError:
                    raise
                except Exception as exc:
                    return json.dumps(
                        {
                            "error": "harness fence update failed",
                            "error_type": type(exc).__name__,
                        }
                    )
            if action == "install_hooks":
                try:
                    from agent_utilities.ecosystem.hook_installer import HookInstaller

                    installer = HookInstaller()
                    agents = config_value.split(",") if config_value else None
                    results = installer.install(agents)
                    return json.dumps(
                        {
                            "status": "success",
                            "results": results,
                            "installed": installer.installed,
                            "errors": installer.errors,
                        }
                    )
                except PermissionError:
                    raise
                except Exception as exc:
                    return json.dumps(
                        {
                            "error": "hook installation failed",
                            "error_type": type(exc).__name__,
                        }
                    )
            if action == "uninstall_hooks":
                try:
                    from agent_utilities.ecosystem.hook_installer import HookInstaller

                    agents = config_value.split(",") if config_value else None
                    results = HookInstaller().uninstall(agents)
                    return json.dumps({"status": "success", "results": results})
                except PermissionError:
                    raise
                except Exception as exc:
                    return json.dumps(
                        {
                            "error": "hook removal failed",
                            "error_type": type(exc).__name__,
                        }
                    )
            if action == "doctor":
                try:
                    from agent_utilities.ecosystem.hook_installer import HookInstaller

                    return json.dumps(HookInstaller().doctor(), default=str)
                except PermissionError:
                    raise
                except Exception as exc:
                    return json.dumps(
                        {
                            "error": "configuration doctor failed",
                            "error_type": type(exc).__name__,
                        }
                    )
            # ── CONCEPT:AU-ORCH.routing.role-specialized-model-routing: Role-Specialized Model Routing ──
            if action == "set_role_routing":
                try:
                    from pathlib import Path

                    from agent_utilities.core.config import config as _cfg
                    from agent_utilities.models.model_registry import (
                        ModelRegistry,
                        RoleSpec,
                    )

                    payload = json.loads(config_value) if config_value else {}
                    reg_path = getattr(_cfg, "model_registry_path", None)
                    if not reg_path or not Path(reg_path).is_file():
                        return json.dumps(
                            {
                                "error": (
                                    "No model_registry_path configured; cannot "
                                    "persist role_routing."
                                )
                            }
                        )
                    registry = ModelRegistry.load_from_file(reg_path)
                    for rname, spec in payload.items():
                        registry.role_routing[rname] = RoleSpec.model_validate(spec)
                    Path(reg_path).write_text(
                        json.dumps(registry.model_dump(), indent=2)
                    )
                    return json.dumps(
                        {
                            "status": "success",
                            "action": "set_role_routing",
                            "roles": list(payload.keys()),
                        }
                    )
                except PermissionError:
                    raise
                except Exception as exc:
                    return json.dumps(
                        {
                            "error": "role routing update failed",
                            "error_type": type(exc).__name__,
                        }
                    )
            # ── KG-2.35: Schema-Pack lifecycle (get/set the active domain pack) ──
            if action == "schema_pack":
                from agent_utilities.models.schema_pack_loader import (
                    get_active_pack,
                    set_active_pack,
                )
                from agent_utilities.models.schema_packs import list_schema_packs

                if config_key:
                    pack = set_active_pack(config_key)
                    return json.dumps(
                        {
                            "status": "success",
                            "action": "schema_pack",
                            "active": pack.name,
                            "signature": pack.signature(),
                        }
                    )
                active = get_active_pack()
                return json.dumps(
                    {
                        "status": "success",
                        "action": "schema_pack",
                        "active": active.name,
                        "signature": active.signature(),
                        "available": list_schema_packs(),
                    }
                )
            # ── KG-2.35: review out-of-pack candidate types seen on write ──
            if action == "schema_candidates":
                from agent_utilities.models.schema_pack_audit import (
                    SchemaCandidateAuditor,
                )

                try:
                    limit = int(config_value) if config_value else 100
                except ValueError:
                    limit = 100
                return json.dumps(
                    {
                        "status": "success",
                        "action": "schema_candidates",
                        "candidates": SchemaCandidateAuditor.instance().review(limit),
                    }
                )
            return json.dumps({"error": "unknown configuration action"})
        except PermissionError:
            # Authorization and filesystem-boundary denials are policy results,
            # not successful MCP payloads.  Preserve fail-closed dispatch.
            raise PermissionError("configuration operation denied") from None
        except Exception as exc:
            logger.warning("graph_configure operation failed (%s)", type(exc).__name__)
            return json.dumps(
                {
                    "error": "configuration operation failed",
                    "error_type": type(exc).__name__,
                }
            )

    kg_server.REGISTERED_TOOLS["graph_configure"] = graph_configure
