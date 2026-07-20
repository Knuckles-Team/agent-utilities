#!/usr/bin/env python3
"""Check the bounded, reference-only universal external-graph contract."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "agent_utilities/core/config.py"
REGISTRY = ROOT / "agent_utilities/knowledge_graph/core/connection_registry.py"
SCHEMA = ROOT / "agent_utilities/knowledge_graph/ingestion/external_graph_schema.py"
INGEST = ROOT / "agent_utilities/knowledge_graph/ingestion/external_graph.py"
GRAPHQL = ROOT / "agent_utilities/knowledge_graph/ingestion/graphql_connection.py"
GRAPHQL_CONNECTOR = (
    ROOT / "agent_utilities/protocols/source_connectors/connectors/graphql_document.py"
)
TOOLS = ROOT / "agent_utilities/mcp/tools/analysis_tools.py"
DOCTOR = ROOT / "agent_utilities/deployment/doctor.py"
MANIFEST_GATE = (
    ROOT / "agent_utilities/knowledge_graph/ontology/connector_manifest_gate.py"
)
NATIVE_MANIFEST = (
    ROOT
    / "agent_utilities/knowledge_graph/ontology/connector_manifests"
    / "native-source-connectors/connector_manifest.yml"
)
EXAMPLE_CONFIG = ROOT / "docs/examples/config.json"
DOCS = (
    ROOT / "docs/architecture/universal-external-graph-connectors.md",
    ROOT / "docs/architecture/privacy-safe-external-ingestion.md",
)
CI = ROOT / ".github/workflows/guardrails.yml"

REQUIRED_BACKENDS = frozenset(
    {
        "neo4j",
        "opencypher",
        "age",
        "ladybug",
        "epistemic_graph",
        "graphql",
    }
)
_ENVIRONMENT_LITERAL_RE = re.compile(
    r"(?i)(?:https?|bolt|neo4j(?:\+s)?|postgres(?:ql)?)://|"
    r"(?:/home/|/Users/|[A-Z]:[\\/]Users[\\/])|"
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"
)


def environment_literal_violations(path: Path, text: str) -> list[str]:
    """Return locations of checked-in endpoint, path, or email literals."""

    return [
        f"{path.relative_to(ROOT)}:{line_number}: environment-specific literal"
        for line_number, line in enumerate(text.splitlines(), 1)
        if _ENVIRONMENT_LITERAL_RE.search(line)
    ]


def _require_markers(
    failures: list[str], path: Path, text: str, markers: tuple[str, ...]
) -> None:
    for marker in markers:
        if marker not in text:
            failures.append(f"{path.relative_to(ROOT)}: missing {marker!r}")


def violations() -> list[str]:
    failures: list[str] = []
    sources = {
        path: path.read_text(encoding="utf-8")
        for path in (
            CONFIG,
            REGISTRY,
            SCHEMA,
            INGEST,
            GRAPHQL,
            GRAPHQL_CONNECTOR,
            TOOLS,
            DOCTOR,
            MANIFEST_GATE,
        )
    }

    config = sources[CONFIG]
    missing_backends = sorted(
        backend for backend in REQUIRED_BACKENDS if f'"{backend}"' not in config
    )
    if missing_backends:
        failures.append(f"AgentConfig is missing external backends: {missing_backends}")
    _require_markers(
        failures,
        CONFIG,
        config,
        (
            "class ExternalGraphConnectorConfig(BaseModel):",
            'model_config = ConfigDict(extra="forbid")',
            "connection_profile_ref: str",
            "ingest_page_size: int = 500",
            "ingest_max_pages: int = 100",
            "ingest_max_row_bytes: int = 1_048_576",
            "ingest_max_total_bytes: int = 16_777_216",
            "ingest_max_nesting_depth: int = 16",
            "ingest_max_collection_items: int = 10_000",
            'sync_mode: Literal["auto", "cdc", "snapshot"]',
            "reconcile_deletions: bool = True",
            "allow_empty_snapshot: bool = False",
            "def _validate_external_graph_declaration_identities(",
            "names and source_alias values must be unique",
            'schema_drift_policy: Literal["fail_closed"]',
            "require_approval: Literal[True]",
        ),
    )

    _require_markers(
        failures,
        REGISTRY,
        sources[REGISTRY],
        (
            "_EXTERNAL_PROPERTY_GRAPH_FIELDS",
            '"ingest_page_size"',
            '"ingest_max_pages"',
            '"ingest_max_row_bytes"',
            '"ingest_max_total_bytes"',
            '"ingest_max_nesting_depth"',
            '"ingest_max_collection_items"',
            '"reconcile_deletions"',
            '"allow_empty_snapshot"',
            '"sync_mode"',
            "_EXTERNAL_GRAPHQL_FIELDS",
            "set(spec).difference(allowed)",
            "persistent external graph declarations contain unsupported inline material",
            "persistent connection backend selectors disagree",
            'required_refs = ["connection_profile_ref"]',
            'if backend_kind == "opencypher":',
            'build_spec["backend_type"] = "neo4j"',
        ),
    )
    _require_markers(
        failures,
        SCHEMA,
        sources[SCHEMA],
        (
            "class GraphQLDiscoveryAdapter:",
            "allow_introspection: bool = True",
            "if allow_introspection:",
            "probe_limit = limit + 1",
            "def external_mapping_policy_digest(",
            '"identity_hmac_key_ref": str(',
            '"id_path": str(node_mapping.get("id_path") or "id")',
            '"type_path": str(node_mapping.get("type_path") or "type")',
            '"version_path": str(node_mapping.get("version_path") or "version")',
            'edge_mapping.get("properties_path") or "properties"',
            '"source_path": str(edge_mapping.get("source_path") or "source")',
            '"target_path": str(edge_mapping.get("target_path") or "target")',
            '"type_path": str(edge_mapping.get("type_path") or "type")',
            '"runtime_policy_digest": runtime_policy_digest',
            '"mapping_drift": mapping_drift',
            'self.kind in {"neo4j", "opencypher"}',
            "class RemoteEpistemicGraphReadAdapter:",
            "class LadybugDiscoveryAdapter(",
        ),
    )
    _require_markers(
        failures,
        INGEST,
        sources[INGEST],
        (
            'runtime_policy_digest: str = ""',
            "External graph mapping policy drift requires a new proposal",
            '"material-version"',
            "payload,",
            "ingest_envelope(authority_engine, envelope)",
            "ChangeEnvelope(",
            'precheck_source("external_graph")',
            "runtime profiles cannot embed identity key material",
            "External graph CDC event cursor did not advance",
            'params.update({"offset": 0, "limit": max_records + 1})',
            "External graph stable snapshot token changed during paging",
            "snapshot_identity_complete = False",
            "exceeded the per-row byte bound",
            "exceeded the cumulative byte bound",
            "exceeded the nesting-depth bound",
            "exceeded the collection-size bound",
        ),
    )
    _require_markers(
        failures,
        DOCTOR,
        sources[DOCTOR],
        (
            "external_graph_source_aliases_unique",
            "external_graph_connection_names_unique",
        ),
    )
    if '"identity_hmac_key": identity_key' in sources[SCHEMA]:
        failures.append(
            "external_graph_schema.py: resolved identity key material enters a profile"
        )
    _require_markers(
        failures,
        MANIFEST_GATE,
        sources[MANIFEST_GATE],
        (
            '"external_graph": "native-source-connectors"',
            '"external_graph_ingestion_v1"',
            '"agent_utilities.knowledge_graph.ingestion.external_graph"',
            'NATIVE_FINGERPRINT_FORMAT = "agent-utilities-local-module-closure-v1"',
            "def native_activation_fingerprint(",
            "def native_activation_fingerprint_modules(",
        ),
    )
    _require_markers(
        failures,
        NATIVE_MANIFEST,
        NATIVE_MANIFEST.read_text(encoding="utf-8"),
        (
            "preset: native-external-graph",
            "tool: external_graph",
            "interface: external_graph_ingestion_v1",
        ),
    )
    _require_markers(
        failures,
        GRAPHQL,
        sources[GRAPHQL],
        (
            '"allow_introspection": bool(',
            "IngestionEngine(kg_engine=authority_engine",
            '"incremental": True',
            "GraphQL mapping policy changed and requires a new approval",
            "def graphql_mapping_profile_status(",
            "_GRAPHQL_READ_BOOTSTRAP",
            "def _generate_mapping_policy(",
        ),
    )
    _require_markers(
        failures,
        GRAPHQL_CONNECTOR,
        sources[GRAPHQL_CONNECTOR],
        (
            '_CHECKPOINT_FORMAT = "graphql-snapshot-checkpoint/v1"',
            "snapshot_authoritative",
            'operation="delete"',
            '"snapshot_reconciliation": True',
        ),
    )
    _require_markers(
        failures,
        TOOLS,
        sources[TOOLS],
        (
            "_EXTERNAL_MAPPING_POLICY_FIELDS",
            "_configured_external_graph_declaration(",
            "_resolved_external_mapping_policy(",
            "runtime_policy_digest=current_policy_digest",
            "runtime_policy_digest=runtime_policy_digest",
            "profiles, queries,",
            "variables, ontology, endpoints, paths",
        ),
    )
    _require_markers(
        failures,
        DOCTOR,
        sources[DOCTOR],
        (
            '"mapping_policy_drift": mapping_policy_drift',
            '"capability_bundle_ready"',
            '"sync_policy": sync_policy',
            "graphql_mapping_profile_status(",
        ),
    )

    for path in (SCHEMA, INGEST, GRAPHQL, REGISTRY, *DOCS):
        failures.extend(
            environment_literal_violations(path, path.read_text(encoding="utf-8"))
        )

    example = json.loads(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    if example.get("external_graph_connectors") != []:
        failures.append(
            "docs/examples/config.json: external_graph_connectors must stay empty"
        )
    forbidden_profile_patterns = (
        "agent_utilities/knowledge_graph/ingestion/external*.json",
        "agent_utilities/knowledge_graph/ingestion/external*.yaml",
        "agent_utilities/knowledge_graph/ingestion/external*.yml",
        "agent_utilities/knowledge_graph/ingestion/external*.ttl",
        "agent_utilities/knowledge_graph/ingestion/external*.owl",
        "docs/examples/external*profile*",
        "docs/examples/external*ontology*",
    )
    for pattern in forbidden_profile_patterns:
        for path in ROOT.glob(pattern):
            failures.append(
                f"{path.relative_to(ROOT)}: bundled external profile or ontology"
            )

    for path in DOCS:
        text = path.read_text(encoding="utf-8")
        _require_markers(
            failures,
            path,
            text,
            (
                "Neo4j/openCypher",
                "Apache AGE",
                "LadybugDB",
                "remote epistemic-graph",
                "GraphQL",
                "ChangeEnvelope",
                "mapping-policy",
            ),
        )
    if "check_external_graph_contract.py" not in CI.read_text(encoding="utf-8"):
        failures.append("CI does not execute the external-graph contract gate")
    return failures


def main() -> int:
    failures = violations()
    if failures:
        print("External graph contract gate failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print("External graph contract gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
