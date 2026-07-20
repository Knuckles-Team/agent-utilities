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
CI = ROOT / ".github/workflows/guardrails.yml"

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


def violations() -> list[str]:
    failures: list[str] = []
    tree = ast.parse(INGEST.read_text(encoding="utf-8"), filename=str(INGEST))
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    public = functions.get("ingest_envelope")
    native = functions.get("_apply_native_change_envelope")
    if public is None or native is None:
        failures.append("native/public ChangeEnvelope functions are incomplete")
        return failures

    public_calls = {
        _dotted(node.func) for node in ast.walk(public) if isinstance(node, ast.Call)
    }
    if "_apply_native_change_envelope" not in public_calls:
        failures.append(
            "ingest_envelope does not delegate to native ApplyChangeEnvelope"
        )
    leaked = sorted(_RETIRED_SEQUENTIAL_SYMBOLS.intersection(public_calls))
    if leaked:
        failures.append(
            f"public ingest_envelope calls sequential durability steps: {leaked}"
        )

    native_calls = {
        _dotted(node.func) for node in ast.walk(native) if isinstance(node, ast.Call)
    }
    if not any(call.endswith("changes.apply") for call in native_calls):
        failures.append(
            "native ingestion does not invoke the generated changes.apply client"
        )

    source = INGEST.read_text(encoding="utf-8")
    for marker in (
        "NativeChangeEnvelopeUnavailable",
        'supports("ApplyChangeEnvelope")',
        'supports("GetChangeCursor")',
        "def read_change_cursor(",
        "class NativeChangeEnvelopeEngineProxy:",
        "def ingest_graph_slice(",
    ):
        if marker not in source:
            failures.append(f"native ingestion guard is missing {marker}")

    retired_sources = {
        "envelope_ingest": source,
        "source_sync": SOURCE_SYNC.read_text(encoding="utf-8"),
        "typed_config": CONFIG.read_text(encoding="utf-8"),
        "profile_guard": PROFILE_GUARD.read_text(encoding="utf-8"),
    }
    for label, text in retired_sources.items():
        if ("KG_ENVELOPE_" + "LEGACY_ADAPTER") in text or (
            "kg_envelope_" + "legacy_adapter"
        ) in text:
            failures.append(f"{label} still exposes the retired envelope adapter switch")
        leaked_symbols = sorted(
            symbol for symbol in _RETIRED_SEQUENTIAL_SYMBOLS if symbol in text
        )
        if leaked_symbols:
            failures.append(
                f"{label} still contains retired sequential symbols: {leaked_symbols}"
            )
    if "check_native_change_envelope_boundary.py" not in CI.read_text(encoding="utf-8"):
        failures.append("CI does not execute the native ChangeEnvelope boundary gate")

    for relative in (
        "agent_utilities/knowledge_graph/core/source_sync.py",
        "agent_utilities/knowledge_graph/ingestion/external_graph.py",
        "agent_utilities/knowledge_graph/ingestion/engine.py",
    ):
        text = (ROOT / relative).read_text(encoding="utf-8")
        if "_ingest_envelope_legacy" in text:
            failures.append(f"{relative} imports the retired sequential adapter")

    sync_tree = ast.parse(
        SOURCE_SYNC.read_text(encoding="utf-8"), filename=str(SOURCE_SYNC)
    )
    leanix = next(
        (
            node
            for node in sync_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_sync_leanix"
        ),
        None,
    )
    if leanix is None:
        failures.append("source_sync lacks the LeanIX ChangeEnvelope handler")
    else:
        leanix_calls = {
            _dotted(node.func)
            for node in ast.walk(leanix)
            if isinstance(node, ast.Call)
        }
        if "engine.ingest_external_batch" in leanix_calls:
            failures.append("LeanIX bypasses ApplyChangeEnvelope for graph rows")

    sync_functions = {
        node.name: node
        for node in sync_tree.body
        if isinstance(node, ast.FunctionDef)
        and (
            node.name.startswith("_sync_")
            or node.name in {"_write_fleet_nodes", "_reconcile"}
        )
    }
    forbidden = {
        "engine.ingest_external_batch",
        "engine.add_node",
        "engine.link_nodes",
        "_write_watermark",
    }
    for name, function in sync_functions.items():
        calls = {
            _dotted(node.func)
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
        }
        leaked = sorted(calls.intersection(forbidden))
        if leaked:
            failures.append(
                f"{name} contains durable native-boundary bypasses: {leaked}"
            )

    package_install = sync_functions.get("_sync_package_install")
    if package_install is None:
        failures.append("source_sync lacks the package-install orchestrator")
    else:
        package_calls = {
            _dotted(node.func)
            for node in ast.walk(package_install)
            if isinstance(node, ast.Call)
        }
        if package_calls != {"sync_package_install"}:
            failures.append(
                "package_install orchestration allowlist gained non-delegation calls: "
                f"{sorted(package_calls)}"
            )

    sync_source = SOURCE_SYNC.read_text(encoding="utf-8")
    if (
        'ORCHESTRATION_ONLY_SOURCES: frozenset[str] = frozenset({"package_install"})'
        not in sync_source
    ):
        failures.append("source_sync orchestration-only allowlist is not exact")
    if (
        "LEGACY_BATCH_SOURCES" in sync_source
        or "NON_ENVELOPE_PIPELINE_SOURCES" in sync_source
    ):
        failures.append("source_sync still exposes a durable legacy migration bucket")

    hydration_tree = ast.parse(
        HYDRATION.read_text(encoding="utf-8"), filename=str(HYDRATION)
    )
    hydration_class = next(
        (
            node
            for node in hydration_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "HydrationManager"
        ),
        None,
    )
    hydrate_source = (
        next(
            (
                node
                for node in hydration_class.body
                if isinstance(node, ast.FunctionDef) and node.name == "hydrate_source"
            ),
            None,
        )
        if hydration_class is not None
        else None
    )
    hydrate_calls = (
        {
            _dotted(node.func)
            for node in ast.walk(hydrate_source)
            if isinstance(node, ast.Call)
        }
        if hydrate_source is not None
        else set()
    )
    if "NativeChangeEnvelopeEngineProxy" not in hydrate_calls:
        failures.append(
            "generic HydrationManager dispatch does not wrap batch writers in the "
            "native ChangeEnvelope proxy"
        )

    materialize_tree = ast.parse(
        MATERIALIZE.read_text(encoding="utf-8"), filename=str(MATERIALIZE)
    )
    materialize_source = next(
        (
            node
            for node in materialize_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "materialize_source"
        ),
        None,
    )
    materialize_calls = (
        {
            _dotted(node.func)
            for node in ast.walk(materialize_source)
            if isinstance(node, ast.Call)
        }
        if materialize_source is not None
        else set()
    )
    if "ingest_graph_slice" not in materialize_calls:
        failures.append("materialize_source bypasses the native graph-slice envelope")
    if "write_batch" in materialize_calls:
        failures.append("materialize_source still invokes the direct batch writer")

    chunked_source = CHUNKED_DRAIN.read_text(encoding="utf-8")
    if "_write_watermark" in chunked_source or "_read_watermark" in chunked_source:
        failures.append("chunked drain bypasses the native typed source cursor")
    for marker in (
        "_ingest_graph_slice_via_envelope",
        "_read_envelope_watermark",
        '"failed": report.failed',
    ):
        if marker not in chunked_source:
            failures.append(f"chunked drain native cursor boundary is missing {marker}")

    document_source = DOCUMENT_PROCESSOR.read_text(encoding="utf-8")
    for marker in ("def _persist_native(", 'record["_nodes"]', "ingest_envelope("):
        if marker not in document_source:
            failures.append(f"DocumentProcessor native slice is missing {marker}")

    engine_tree = ast.parse(
        INGESTION_ENGINE.read_text(encoding="utf-8"), filename=str(INGESTION_ENGINE)
    )
    ingestion_class = next(
        (
            node
            for node in engine_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "IngestionEngine"
        ),
        None,
    )
    ingestion_methods = (
        {
            node.name: node
            for node in ingestion_class.body
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        }
        if ingestion_class is not None
        else {}
    )
    enrich_text = ingestion_methods.get("_enrich_text")
    enrich_calls = (
        {
            _dotted(node.func)
            for node in ast.walk(enrich_text)
            if isinstance(node, ast.Call)
        }
        if enrich_text is not None
        else set()
    )
    if "ingest_graph_slice" not in enrich_calls:
        failures.append("shared text enrichment bypasses the native graph slice")
    for bypass in ("self.backend.add_node", "self.backend.add_edge"):
        if bypass in enrich_calls:
            failures.append(f"shared text enrichment writes directly: {bypass}")

    document_method = ingestion_methods.get("_ingest_document_file")
    document_calls = (
        {
            _dotted(node.func)
            for node in ast.walk(document_method)
            if isinstance(node, ast.Call)
        }
        if document_method is not None
        else set()
    )
    if "ingest_graph_slice" not in document_calls:
        failures.append("generic document ingestion bypasses the native graph slice")
    if "persistence_reference" not in document_calls:
        failures.append("generic document ingestion persists a raw source location")
    for bypass in ("backend.add_node", "backend.add_edge"):
        if bypass in document_calls:
            failures.append(f"generic document ingestion writes directly: {bypass}")

    acquire_method = ingestion_methods.get("_acquire_referenced_papers")
    acquire_calls = (
        {
            _dotted(node.func)
            for node in ast.walk(acquire_method)
            if isinstance(node, ast.Call)
        }
        if acquire_method is not None
        else set()
    )
    if "ingest_graph_slice" not in acquire_calls:
        failures.append("research-roundup links bypass the native graph slice")
    if "self.backend.add_edge" in acquire_calls:
        failures.append("research-roundup links write directly")

    connector_method = ingestion_methods.get("_ingest_connector")
    connector_processors = (
        [
            node
            for node in ast.walk(connector_method)
            if isinstance(node, ast.Call)
            and _dotted(node.func).endswith("DocumentProcessor")
        ]
        if connector_method is not None
        else []
    )
    if not connector_processors or any(
        not any(keyword.arg == "engine" for keyword in call.keywords)
        for call in connector_processors
    ):
        failures.append("connector DocumentProcessor lacks native engine authority")
    connector_calls = (
        {
            _dotted(node.func)
            for node in ast.walk(connector_method)
            if isinstance(node, ast.Call)
        }
        if connector_method is not None
        else set()
    )
    for required in ("read_change_cursor", "ingest_graph_slice"):
        if required not in connector_calls:
            failures.append(f"generic connector lacks native {required} boundary")
    for bypass in ("self.manifest.get", "self.manifest.record"):
        if bypass in connector_calls:
            failures.append(
                f"generic connector retains split cursor authority: {bypass}"
            )

    world_source = WORLD_MODEL.read_text(encoding="utf-8")
    for bypass in (
        "engine.graph.add_node(",
        "self.engine._upsert_node(",
        "engine.add_edge(",
    ):
        if bypass in world_source:
            failures.append(
                f"world-model external ingest bypasses native envelope: {bypass}"
            )
    if "._enrich_text(" in world_source:
        failures.append(
            "world-model external ingest invokes the legacy direct-write enrichment seam"
        )

    research_source = RESEARCH_PIPELINE.read_text(encoding="utf-8")
    research_tree = ast.parse(research_source, filename=str(RESEARCH_PIPELINE))
    research_functions = {
        node.name: node
        for node in research_tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }
    runner = next(
        (
            node
            for node in research_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "ResearchPipelineRunner"
        ),
        None,
    )
    if runner is not None:
        research_functions.update(
            {
                node.name: node
                for node in runner.body
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            }
        )
    for name in ("ingest_paper_full", "ingest_paper_marginal"):
        function = research_functions.get(name)
        calls = (
            {
                _dotted(node.func)
                for node in ast.walk(function)
                if isinstance(node, ast.Call)
            }
            if function is not None
            else set()
        )
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

    for name in ("ingest_local_file", "ingest_url"):
        function = research_functions.get(name)
        calls = (
            {
                _dotted(node.func)
                for node in ast.walk(function)
                if isinstance(node, ast.Call)
            }
            if function is not None
            else set()
        )
        if "DocumentProcessor" not in calls:
            failures.append(f"{name} bypasses native DocumentProcessor ingestion")

    paper_slice = research_functions.get("_commit_research_paper_slice")
    paper_calls = (
        {
            _dotted(node.func)
            for node in ast.walk(paper_slice)
            if isinstance(node, ast.Call)
        }
        if paper_slice is not None
        else set()
    )
    if "ingest_graph_slice" not in paper_calls:
        failures.append("research paper slice does not invoke ingest_graph_slice")
    for marker in (
        "persistence_reference(",
        'name="[REDACTED_PERSON]"',
        "DocumentProcessor(",
    ):
        if marker not in research_source:
            failures.append(f"research native/privacy boundary is missing {marker}")
    for bypass in (
        "ScholarXKGBridge",
        "KBIngestionEngine",
        "self.engine.graph.add_node(",
        "self.engine.graph.add_edge(",
        "self.engine._upsert_node(",
    ):
        if bypass in research_source:
            failures.append(
                f"research paper graph authority bypasses native envelope: {bypass}"
            )

    research_feed = RESEARCH_FEED.read_text(encoding="utf-8")
    if '"authors": author_refs' not in research_feed:
        failures.append("research fetch work item persists raw author identities")

    cohort_source = RESEARCH_COHORT.read_text(encoding="utf-8")
    for marker in (
        "def _commit_cohort_state(",
        "ingest_graph_slice(",
        "def resolve_ephemeral_paper_pdf(",
    ):
        if marker not in cohort_source:
            failures.append(f"research cohort native boundary is missing {marker}")
    for bypass in (
        "engine.add_node(",
        '"pdf_path"',
        "_paper_pdf_path",
    ):
        if bypass in cohort_source:
            failures.append(
                f"research cohort persists through a forbidden seam: {bypass}"
            )

    engine_tasks = ENGINE_TASKS.read_text(encoding="utf-8")
    if "resolve_ephemeral_paper_pdf(" not in engine_tasks:
        failures.append("research worker does not resolve paper files ephemerally")
    if 'paper.get("pdf_path")' in engine_tasks:
        failures.append("research worker accepts a durable local paper path")
    engine_tasks_tree = ast.parse(engine_tasks, filename=str(ENGINE_TASKS))
    task_processors = [
        node
        for node in ast.walk(engine_tasks_tree)
        if isinstance(node, ast.Call)
        and _dotted(node.func).endswith("DocumentProcessor")
    ]
    if any(
        not any(keyword.arg == "engine" for keyword in call.keywords)
        for call in task_processors
    ):
        failures.append("worker DocumentProcessor lacks native engine authority")

    # Repository-wide external-batch regression fence. Hydration handlers retain
    # the compatibility-shaped call only because source_sync supplies the native
    # ChangeEnvelope proxy; no other production module may call it directly.
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

    feed_source = FEED_SOURCES.read_text(encoding="utf-8")
    for bypass in ("engine.add_node(", "DETACH DELETE"):
        if bypass in feed_source:
            failures.append(f"feed registry bypasses native envelope: {bypass}")
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
