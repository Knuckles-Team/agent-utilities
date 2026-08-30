#!/usr/bin/env python3
"""Fail closed when connector ChangeEnvelope ingestion bypasses engine atomicity."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INGEST = ROOT / "agent_utilities/knowledge_graph/ingestion/envelope_ingest.py"
CONFIG = ROOT / "agent_utilities/core/config.py"
PROFILE_GUARD = ROOT / "agent_utilities/core/profile_guard.py"
SOURCE_SYNC = ROOT / "agent_utilities/knowledge_graph/core/source_sync.py"
CHUNKED_DRAIN = ROOT / "agent_utilities/knowledge_graph/core/chunked_drain.py"
HYDRATION = ROOT / "agent_utilities/knowledge_graph/core/hydration.py"
MATERIALIZE = ROOT / "agent_utilities/knowledge_graph/enrichment/materialize.py"
DOCUMENT_PROCESSOR = (
    ROOT / "agent_utilities/knowledge_graph/ontology/document_processing.py"
)
INGESTION_ENGINE = ROOT / "agent_utilities/knowledge_graph/ingestion/engine.py"
WORLD_MODEL = ROOT / "agent_utilities/automation/worldmodel_pipeline.py"
FEED_SOURCES = ROOT / "agent_utilities/automation/feed_sources.py"
RESEARCH_PIPELINE = ROOT / "agent_utilities/automation/research_pipeline.py"
RESEARCH_FEED = ROOT / "agent_utilities/knowledge_graph/research/feed_grading.py"
RESEARCH_COHORT = ROOT / "agent_utilities/knowledge_graph/research/cohort.py"
ENGINE_TASKS = ROOT / "agent_utilities/knowledge_graph/core/engine_tasks.py"
CI = ROOT / ".github/workflows/advisory.yml"

_RETIRED_SEQUENTIAL_SYMBOLS = {
    "_ingest_envelope_legacy",
    "_apply_write",
    "_lineage_write",
    "_emit_cdc",
    "_advance_watermark",
    "_read_watermark",
    "_reconcile_legacy",
}


def _dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _calls(function: ast.AST | None) -> set[str]:
    if function is None:
        return set()
    return {
        _dotted(node.func) for node in ast.walk(function) if isinstance(node, ast.Call)
    }


def _class_node(tree: ast.Module, name: str) -> ast.ClassDef | None:
    return next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == name
        ),
        None,
    )


def _class_method(class_node: ast.ClassDef | None, name: str) -> ast.FunctionDef | None:
    if class_node is None:
        return None
    return next(
        (
            node
            for node in class_node.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        ),
        None,
    )


def _ingest_call_failures(
    public: ast.FunctionDef, native: ast.FunctionDef
) -> list[str]:
    failures: list[str] = []
    public_calls = _calls(public)
    if "_apply_native_change_envelope" not in public_calls:
        failures.append(
            "ingest_envelope does not delegate to native ApplyChangeEnvelope"
        )
    leaked = sorted(_RETIRED_SEQUENTIAL_SYMBOLS.intersection(public_calls))
    if leaked:
        failures.append(
            f"public ingest_envelope calls sequential durability steps: {leaked}"
        )

    native_calls = _calls(native)
    if not any(call.endswith("changes.apply") for call in native_calls):
        failures.append(
            "native ingestion does not invoke the generated changes.apply client"
        )
    return failures


def _ingest_marker_failures(source: str) -> list[str]:
    return [
        f"native ingestion guard is missing {marker}"
        for marker in (
            "NativeChangeEnvelopeUnavailable",
            'supports("ApplyChangeEnvelope")',
            'supports("GetChangeCursor")',
            "def read_change_cursor(",
            "class NativeChangeEnvelopeEngineProxy:",
            "def ingest_graph_slice(",
        )
        if marker not in source
    ]


def _ingest_failures() -> tuple[list[str], bool]:
    source = INGEST.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(INGEST))
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    public = functions.get("ingest_envelope")
    native = functions.get("_apply_native_change_envelope")
    if public is None or native is None:
        return ["native/public ChangeEnvelope functions are incomplete"], True

    failures = _ingest_call_failures(public, native)
    failures.extend(_ingest_marker_failures(source))
    return failures, False


def _retired_source_failures() -> list[str]:
    retired_sources = {
        "envelope_ingest": INGEST.read_text(encoding="utf-8"),
        "source_sync": SOURCE_SYNC.read_text(encoding="utf-8"),
        "typed_config": CONFIG.read_text(encoding="utf-8"),
        "profile_guard": PROFILE_GUARD.read_text(encoding="utf-8"),
    }
    failures: list[str] = []
    for label, text in retired_sources.items():
        if ("KG_ENVELOPE_" + "LEGACY_ADAPTER") in text or (
            "kg_envelope_" + "legacy_adapter"
        ) in text:
            failures.append(
                f"{label} still exposes the retired envelope adapter switch"
            )
        leaked_symbols = sorted(
            symbol for symbol in _RETIRED_SEQUENTIAL_SYMBOLS if symbol in text
        )
        if leaked_symbols:
            failures.append(
                f"{label} still contains retired sequential symbols: {leaked_symbols}"
            )
    if "check_native_change_envelope_boundary.py" not in CI.read_text(encoding="utf-8"):
        failures.append("CI does not execute the native ChangeEnvelope boundary gate")
    return failures


def _retired_import_failures() -> list[str]:
    failures: list[str] = []
    for relative in (
        "agent_utilities/knowledge_graph/core/source_sync.py",
        "agent_utilities/knowledge_graph/ingestion/external_graph.py",
        "agent_utilities/knowledge_graph/ingestion/engine.py",
    ):
        text = (ROOT / relative).read_text(encoding="utf-8")
        if "_ingest_envelope_legacy" in text:
            failures.append(f"{relative} imports the retired sequential adapter")
    return failures


def _source_sync_functions(
    sync_tree: ast.Module,
) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in sync_tree.body
        if isinstance(node, ast.FunctionDef)
        and (
            node.name.startswith("_sync_")
            or node.name in {"_write_fleet_nodes", "_reconcile"}
        )
    }


def _source_sync_leanix_failures(sync_tree: ast.Module) -> list[str]:
    leanix = next(
        (
            node
            for node in sync_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_sync_leanix"
        ),
        None,
    )
    if leanix is None:
        return ["source_sync lacks the LeanIX ChangeEnvelope handler"]
    if "engine.ingest_external_batch" in _calls(leanix):
        return ["LeanIX bypasses ApplyChangeEnvelope for graph rows"]
    return []


def _source_sync_function_failures(
    sync_functions: dict[str, ast.FunctionDef],
) -> list[str]:
    forbidden = {
        "engine.ingest_external_batch",
        "engine.add_node",
        "engine.link_nodes",
        "_write_watermark",
    }
    failures: list[str] = []
    for name, function in sync_functions.items():
        leaked = sorted(_calls(function).intersection(forbidden))
        if leaked:
            failures.append(
                f"{name} contains durable native-boundary bypasses: {leaked}"
            )
    return failures


def _source_sync_package_failures(
    sync_functions: dict[str, ast.FunctionDef],
) -> list[str]:
    package_install = sync_functions.get("_sync_package_install")
    if package_install is None:
        return ["source_sync lacks the package-install orchestrator"]
    package_calls = _calls(package_install)
    if package_calls != {"sync_package_install"}:
        return [
            "package_install orchestration allowlist gained non-delegation calls: "
            f"{sorted(package_calls)}"
        ]
    return []


def _source_sync_marker_failures(source: str) -> list[str]:
    failures: list[str] = []
    if (
        'ORCHESTRATION_ONLY_SOURCES: frozenset[str] = frozenset({"package_install"})'
        not in source
    ):
        failures.append("source_sync orchestration-only allowlist is not exact")
    if "LEGACY_BATCH_SOURCES" in source or "NON_ENVELOPE_PIPELINE_SOURCES" in source:
        failures.append("source_sync still exposes a durable legacy migration bucket")
    return failures


def _source_sync_failures() -> list[str]:
    source = SOURCE_SYNC.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(SOURCE_SYNC))
    sync_functions = _source_sync_functions(tree)
    failures = _source_sync_leanix_failures(tree)
    failures.extend(_source_sync_function_failures(sync_functions))
    failures.extend(_source_sync_package_failures(sync_functions))
    failures.extend(
        _source_sync_marker_failures(SOURCE_SYNC.read_text(encoding="utf-8"))
    )
    return failures


def _hydration_failures() -> list[str]:
    source = HYDRATION.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(HYDRATION))
    hydration_class = _class_node(tree, "HydrationManager")
    hydrate_source = _class_method(hydration_class, "hydrate_source")
    if "NativeChangeEnvelopeEngineProxy" not in _calls(hydrate_source):
        return [
            "generic HydrationManager dispatch does not wrap batch writers in the "
            "native ChangeEnvelope proxy"
        ]
    return []


def _materialize_failures() -> list[str]:
    source = MATERIALIZE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(MATERIALIZE))
    materialize_source = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "materialize_source"
        ),
        None,
    )
    calls = _calls(materialize_source)
    failures: list[str] = []
    if "ingest_graph_slice" not in calls:
        failures.append("materialize_source bypasses the native graph-slice envelope")
    if "write_batch" in calls:
        failures.append("materialize_source still invokes the direct batch writer")
    return failures


def _chunked_drain_failures() -> list[str]:
    source = CHUNKED_DRAIN.read_text(encoding="utf-8")
    failures: list[str] = []
    if "_write_watermark" in source or "_read_watermark" in source:
        failures.append("chunked drain bypasses the native typed source cursor")
    for marker in (
        "_ingest_graph_slice_via_envelope",
        "_read_envelope_watermark",
        '"failed": report.failed',
    ):
        if marker not in source:
            failures.append(f"chunked drain native cursor boundary is missing {marker}")
    return failures


def _document_processor_failures() -> list[str]:
    source = DOCUMENT_PROCESSOR.read_text(encoding="utf-8")
    return [
        f"DocumentProcessor native slice is missing {marker}"
        for marker in ("def _persist_native(", 'record["_nodes"]', "ingest_envelope(")
        if marker not in source
    ]


def _ingestion_methods() -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    source = INGESTION_ENGINE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(INGESTION_ENGINE))
    ingestion_class = _class_node(tree, "IngestionEngine")
    if ingestion_class is None:
        return {}
    return {
        node.name: node
        for node in ingestion_class.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }


def _enrich_failures(methods: dict[str, ast.AST]) -> list[str]:
    calls = _calls(methods.get("_enrich_text"))
    failures: list[str] = []
    if "ingest_graph_slice" not in calls:
        failures.append("shared text enrichment bypasses the native graph slice")
    for bypass in ("self.backend.add_node", "self.backend.add_edge"):
        if bypass in calls:
            failures.append(f"shared text enrichment writes directly: {bypass}")
    return failures


def _document_ingest_failures(methods: dict[str, ast.AST]) -> list[str]:
    calls = _calls(methods.get("_ingest_document_file"))
    failures: list[str] = []
    if "ingest_graph_slice" not in calls:
        failures.append("generic document ingestion bypasses the native graph slice")
    if "persistence_reference" not in calls:
        failures.append("generic document ingestion persists a raw source location")
    for bypass in ("backend.add_node", "backend.add_edge"):
        if bypass in calls:
            failures.append(f"generic document ingestion writes directly: {bypass}")
    return failures


def _paper_acquisition_failures(methods: dict[str, ast.AST]) -> list[str]:
    calls = _calls(methods.get("_acquire_referenced_papers"))
    failures: list[str] = []
    if "ingest_graph_slice" not in calls:
        failures.append("research-roundup links bypass the native graph slice")
    if "self.backend.add_edge" in calls:
        failures.append("research-roundup links write directly")
    return failures


def _connector_processor_failures(method: ast.AST | None) -> list[str]:
    processors = (
        [
            node
            for node in ast.walk(method)
            if isinstance(node, ast.Call)
            and _dotted(node.func).endswith("DocumentProcessor")
        ]
        if method is not None
        else []
    )
    if not processors or any(
        not any(keyword.arg == "engine" for keyword in call.keywords)
        for call in processors
    ):
        return ["connector DocumentProcessor lacks native engine authority"]
    return []


def _connector_failures(methods: dict[str, ast.AST]) -> list[str]:
    method = methods.get("_ingest_connector")
    failures = _connector_processor_failures(method)
    calls = _calls(method)
    for required in ("read_change_cursor", "ingest_graph_slice"):
        if required not in calls:
            failures.append(f"generic connector lacks native {required} boundary")
    for bypass in ("self.manifest.get", "self.manifest.record"):
        if bypass in calls:
            failures.append(
                f"generic connector retains split cursor authority: {bypass}"
            )
    return failures


def _ingestion_engine_failures() -> list[str]:
    methods = _ingestion_methods()
    failures = _enrich_failures(methods)
    failures.extend(_document_ingest_failures(methods))
    failures.extend(_paper_acquisition_failures(methods))
    failures.extend(_connector_failures(methods))
    return failures


def _world_model_failures() -> list[str]:
    source = WORLD_MODEL.read_text(encoding="utf-8")
    failures: list[str] = []
    for bypass in (
        "engine.graph.add_node(",
        "self.engine._upsert_node(",
        "engine.add_edge(",
    ):
        if bypass in source:
            failures.append(
                f"world-model external ingest bypasses native envelope: {bypass}"
            )
    if "._enrich_text(" in source:
        failures.append(
            "world-model external ingest invokes the legacy direct-write enrichment seam"
        )
    return failures


def _research_functions(
    tree: ast.AST,
) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }
    runner = _class_node(tree, "ResearchPipelineRunner")
    if runner is not None:
        functions.update(
            {
                node.name: node
                for node in runner.body
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            }
        )
    return functions


def _research_paper_failures(
    functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef],
) -> list[str]:
    failures: list[str] = []
    for name in ("ingest_paper_full", "ingest_paper_marginal"):
        calls = _calls(functions.get(name))
        if "_commit_research_paper_slice" not in calls:
            failures.append(f"{name} bypasses the native research paper slice")
        leaked = sorted(
            calls.intersection(
                {
                    "self.engine.graph.add_node",
                    "self.engine.graph.add_edge",
                    "self.engine._upsert_node",
                    "bridge.ingest_paper",
                    "bridge.ingest_paper_abstract_only",
                    "kb_engine.ingest_directory",
                }
            )
        )
        if leaked:
            failures.append(
                f"{name} contains research graph-authority bypasses: {leaked}"
            )
    return failures


def _research_document_failures(
    functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef],
) -> list[str]:
    failures: list[str] = []
    for name in ("ingest_local_file", "ingest_url"):
        if "DocumentProcessor" not in _calls(functions.get(name)):
            failures.append(f"{name} bypasses native DocumentProcessor ingestion")
    return failures


def _research_slice_failures(
    functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef], source: str
) -> list[str]:
    calls = _calls(functions.get("_commit_research_paper_slice"))
    failures: list[str] = []
    if "ingest_graph_slice" not in calls:
        failures.append("research paper slice does not invoke ingest_graph_slice")
    for marker in (
        "persistence_reference(",
        'name="[REDACTED_PERSON]"',
        "DocumentProcessor(",
    ):
        if marker not in source:
            failures.append(f"research native/privacy boundary is missing {marker}")
    for bypass in (
        "ScholarXKGBridge",
        "KBIngestionEngine",
        "self.engine.graph.add_node(",
        "self.engine.graph.add_edge(",
        "self.engine._upsert_node(",
    ):
        if bypass in source:
            failures.append(
                f"research paper graph authority bypasses native envelope: {bypass}"
            )
    return failures


def _research_pipeline_failures() -> list[str]:
    source = RESEARCH_PIPELINE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(RESEARCH_PIPELINE))
    functions = _research_functions(tree)
    failures = _research_paper_failures(functions)
    failures.extend(_research_document_failures(functions))
    failures.extend(_research_slice_failures(functions, source))
    return failures


def _research_feed_failures() -> list[str]:
    source = RESEARCH_FEED.read_text(encoding="utf-8")
    if '"authors": author_refs' not in source:
        return ["research fetch work item persists raw author identities"]
    return []


def _research_cohort_failures() -> list[str]:
    source = RESEARCH_COHORT.read_text(encoding="utf-8")
    failures: list[str] = []
    for marker in (
        "def _commit_cohort_state(",
        "ingest_graph_slice(",
        "def resolve_ephemeral_paper_pdf(",
    ):
        if marker not in source:
            failures.append(f"research cohort native boundary is missing {marker}")
    for bypass in ("engine.add_node(", '"pdf_path"', "_paper_pdf_path"):
        if bypass in source:
            failures.append(
                f"research cohort persists through a forbidden seam: {bypass}"
            )
    return failures


def _research_engine_task_failures() -> list[str]:
    source = ENGINE_TASKS.read_text(encoding="utf-8")
    failures: list[str] = []
    if "resolve_ephemeral_paper_pdf(" not in source:
        failures.append("research worker does not resolve paper files ephemerally")
    if 'paper.get("pdf_path")' in source:
        failures.append("research worker accepts a durable local paper path")
    tree = ast.parse(source, filename=str(ENGINE_TASKS))
    task_processors = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _dotted(node.func).endswith("DocumentProcessor")
    ]
    if any(
        not any(keyword.arg == "engine" for keyword in call.keywords)
        for call in task_processors
    ):
        failures.append("worker DocumentProcessor lacks native engine authority")
    return failures


def _external_batch_failures() -> list[str]:
    failures: list[str] = []
    for path in (ROOT / "agent_utilities").rglob("*.py"):
        if path.resolve() == HYDRATION.resolve():
            continue
        module_tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for call in (
            node for node in ast.walk(module_tree) if isinstance(node, ast.Call)
        ):
            dotted = _dotted(call.func)
            if dotted.endswith(".ingest_external_batch"):
                relative = path.relative_to(ROOT)
                failures.append(
                    "external batch bypass outside native-proxied hydration: "
                    f"{relative}:{call.lineno} ({dotted})"
                )
    return failures


def _feed_source_failures() -> list[str]:
    source = FEED_SOURCES.read_text(encoding="utf-8")
    return [
        f"feed registry bypasses native envelope: {bypass}"
        for bypass in ("engine.add_node(", "DETACH DELETE")
        if bypass in source
    ]


def violations() -> list[str]:
    failures, incomplete = _ingest_failures()
    if incomplete:
        return failures

    failures.extend(_retired_source_failures())
    failures.extend(_retired_import_failures())
    failures.extend(_source_sync_failures())
    failures.extend(_hydration_failures())
    failures.extend(_materialize_failures())
    failures.extend(_chunked_drain_failures())
    failures.extend(_document_processor_failures())
    failures.extend(_ingestion_engine_failures())
    failures.extend(_world_model_failures())
    failures.extend(_research_pipeline_failures())
    failures.extend(_research_feed_failures())
    failures.extend(_research_cohort_failures())
    failures.extend(_research_engine_task_failures())
    failures.extend(_external_batch_failures())
    failures.extend(_feed_source_failures())
    return failures


def main() -> int:
    failures = violations()
    if failures:
        print("Native ChangeEnvelope boundary gate failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print("Native ChangeEnvelope boundary gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
