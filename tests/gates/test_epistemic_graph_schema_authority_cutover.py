"""Enforce the final one-writer core-schema boundary between AU and EG.

Operator ruling 2026-09-24 (EH-470..EH-473): epistemic-graph owns ALL ontology
lifecycle, SHACL, RDF and OWL semantics; agent-utilities is only the agent
orchestration plane. AU therefore ships no shape/ontology documents, declares no
RDF/SHACL/OWL library, and imports none — in runtime code, tests or scripts.
"""

import ast
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MIGRATED_ONTOLOGY_NAMES = frozenset(
    {
        "ontology_a2a.ttl",
        "ontology_action.ttl",
        "ontology.ttl",
        "ontology_archimate.ttl",
        "ontology_argumentation.ttl",
        "ontology_calendar.ttl",
        "ontology_capability.ttl",
        "ontology_company.ttl",
        "ontology_company_infra.ttl",
        "ontology_concepts.ttl",
        "ontology_documentation.ttl",
        "ontology_energy_geopolitics.ttl",
        "ontology_enterprise.ttl",
        "ontology_government.ttl",
        "ontology_harness.ttl",
        "ontology_hr.ttl",
        "ontology_identity.ttl",
        "ontology_infrastructure.ttl",
        "ontology_medical.ttl",
        "ontology_native_source_connector.ttl",
        "ontology_orchestration.ttl",
        "ontology_personal.ttl",
        "ontology_process_intelligence.ttl",
        "ontology_sdd.ttl",
        "ontology_sdlc_lifecycle.ttl",
        "ontology_software.ttl",
        "ontology_system.ttl",
        "ontology_trm.ttl",
        "ontology_worldview.ttl",
    }
)
MIGRATED_ASSETS = frozenset(
    {
        *(
            Path("agent_utilities/knowledge_graph") / name
            for name in MIGRATED_ONTOLOGY_NAMES
        ),
        Path("agent_utilities/knowledge_graph/shapes/governance.shapes.ttl"),
        # EH-470: moved to EG core sources (`core:*-shapes@1`) or deleted as dead.
        *(
            Path("agent_utilities/knowledge_graph/shapes") / f"{name}.shapes.ttl"
            for name in (
                "argumentation",
                "documentation",
                "feed",
                "harness",
                "portfolio_intelligence",
                "process_intelligence",
                "sdlc_lifecycle",
                "temporal",
            )
        ),
        Path("agent_utilities/ontology/shapes/governance.shapes.ttl"),
        Path("agent_utilities/content.py"),
    }
)
PROVEN_CUTOVER_LOCAL_AUTHORITIES = frozenset(
    {
        Path("agent_utilities/knowledge_graph/core/shacl_validator.py"),
        Path("agent_utilities/knowledge_graph/core/owl_bridge.py"),
        Path("agent_utilities/knowledge_graph/core/ontology_federation.py"),
        Path("agent_utilities/knowledge_graph/core/ontology_loader.py"),
        Path("agent_utilities/knowledge_graph/backends/owl/owlready2_backend.py"),
        Path("agent_utilities/knowledge_graph/maintenance/owl_closure.py"),
        Path("agent_utilities/knowledge_graph/ontology/activation.py"),
        Path("agent_utilities/knowledge_graph/ontology/axioms.py"),
        Path("agent_utilities/knowledge_graph/ontology/capability_hierarchy.py"),
        Path("agent_utilities/knowledge_graph/ontology/evolution.py"),
        Path("agent_utilities/knowledge_graph/ontology/pack_pipeline.py"),
        Path("agent_utilities/knowledge_graph/pipeline/phases/owl_reasoning.py"),
    }
)
GENERATED_GRAPH_SCHEMA_DTO_EXPORTS = frozenset(
    {
        "GraphSchemaOpAttach",
        "GraphSchemaOpAttachPack",
        "GraphSchemaOpDetach",
        "GraphSchemaOp",
        "SchemaSourceOriginViewCore",
        "SchemaSourceOriginViewOperator",
        "SchemaSourceOriginViewAdmin",
        "SchemaSourceOriginViewPack",
        "SchemaSourceOriginViewIngestion",
        "SchemaSourceOriginView",
        "GraphSchemaSourceView",
        "GraphSchemaCommitted",
        "GraphSchemaSourcesView",
    }
)
GENERATED_GRAPH_SCHEMA_REQUEST_EXPORTS = frozenset(
    {
        "GraphSchemaRequest",
        "GraphSchemaListRequest",
        "ShaclValidateRequest",
        "OwlReasonRequest",
        "OwlReasonResult",
        "OwlPropertyFact",
        "OwlExplainRequest",
        "OwlExplainResult",
        "ProofNodeWire",
        "RunDatalogReasoningRequest",
        "DatalogReasoningResult",
        "send_graph_schema",
        "send_graph_schema_list",
        "send_shacl_validate",
        "send_owl_reason",
        "send_owl_explain",
        "send_run_datalog_reasoning",
    }
)
GENERATED_SHACL_REPORT_EXPORTS = frozenset(
    {
        "ShaclSeverity",
        "ShaclValidationResult",
        "ShaclValidationReport",
    }
)
REFERENCE_ROOTS = (
    Path("agent_utilities"),
    Path("docs"),
    Path("scripts"),
    Path("tests"),
    Path("AGENTS.head.md"),
    Path("AGENTS.md"),
    Path("MANIFEST.in"),
    Path("mkdocs.yml"),
    Path("pyproject.toml"),
)
TEXT_SUFFIXES = frozenset(
    {".in", ".json", ".md", ".py", ".toml", ".ttl", ".yaml", ".yml"}
)
MIGRATED_REFERENCE_TOKENS = frozenset(
    {
        "agent_utilities.content",
        "agent_utilities/ontology/shapes",
        "knowledge_graph/shapes/",
        "agent_utilities/knowledge_graph/shapes/governance.shapes.ttl",
        "knowledge_graph/shapes/governance.shapes.ttl",
        "knowledge_graph.core.owl_bridge",
        "knowledge_graph.core.ontology_loader",
        "knowledge_graph.core.shacl_validator",
        "knowledge_graph.maintenance.owl_closure",
        "knowledge_graph.ontology.evolution",
        "knowledge_graph.ontology.pack_pipeline",
    }
)


def _text_paths(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix in TEXT_SUFFIXES
    )


def _stale_references() -> list[str]:
    this_file = Path(__file__).resolve()
    findings: list[str] = []
    for relative_root in REFERENCE_ROOTS:
        findings.extend(
            _stale_references_in_path(ROOT / relative_root, this_file=this_file)
        )
    return findings


def _stale_references_in_path(root: Path, *, this_file: Path) -> list[str]:
    findings: list[str] = []
    for path in _text_paths(root):
        if path.resolve() == this_file:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        findings.extend(
            f"{path.relative_to(ROOT)}: {token}"
            for token in sorted(MIGRATED_REFERENCE_TOKENS)
            if token in text
        )
    return findings


def test_generated_graph_schema_is_the_only_core_schema_contract() -> None:
    from epistemic_graph.generated import graph_schema, rdf_report, reasoning

    missing_dtos = sorted(
        name
        for name in GENERATED_GRAPH_SCHEMA_DTO_EXPORTS
        if not hasattr(graph_schema, name)
    )
    missing_requests = sorted(
        name
        for name in GENERATED_GRAPH_SCHEMA_REQUEST_EXPORTS
        if not hasattr(reasoning, name)
    )
    missing_reports = sorted(
        name for name in GENERATED_SHACL_REPORT_EXPORTS if not hasattr(rdf_report, name)
    )
    assert not missing_dtos, f"generated GraphSchema DTOs missing: {missing_dtos}"
    assert not missing_requests, (
        f"generated GraphSchema requests/senders missing: {missing_requests}"
    )
    assert not missing_reports, (
        f"generated SHACL report DTOs missing: {missing_reports}"
    )

    request_fields = reasoning.ShaclValidateRequest.model_fields
    assert set(request_fields) == {"data_graph", "shapes", "data_triples"}
    assert request_fields["data_graph"].annotation == str | None
    assert request_fields["shapes"].annotation == str | None
    assert request_fields["data_graph"].default is None
    assert request_fields["shapes"].default is None
    # EH-472: typed triples are optional; AU sends them instead of RDF text.
    assert not request_fields["data_triples"].is_required()

    report_fields = rdf_report.ShaclValidationReport.model_fields
    assert set(report_fields) == {
        "schema_digests",
        "composed_digest",
        "conforms",
        "results",
    }
    assert report_fields["schema_digests"].annotation == list[str]
    assert report_fields["composed_digest"].annotation == str | None
    assert report_fields["conforms"].annotation is bool
    assert report_fields["results"].annotation == list[rdf_report.ShaclValidationResult]
    assert report_fields["composed_digest"].default is None
    assert all(
        report_fields[name].is_required()
        for name in ("schema_digests", "conforms", "results")
    )

    sources_fields = graph_schema.GraphSchemaSourcesView.model_fields
    assert set(sources_fields) == {
        "schema_version",
        "graph",
        "core_catalog_digest",
        "composed_digest",
        "core_sources",
        "dynamic_sources",
    }
    assert sources_fields["schema_version"].annotation is int
    assert sources_fields["graph"].annotation is str
    assert sources_fields["core_catalog_digest"].annotation is str
    assert sources_fields["composed_digest"].annotation is str
    assert (
        sources_fields["core_sources"].annotation
        == list[graph_schema.GraphSchemaSourceView]
    )
    assert (
        sources_fields["dynamic_sources"].annotation
        == list[graph_schema.GraphSchemaSourceView]
    )
    assert all(field.is_required() for field in sources_fields.values())

    owl_fields = reasoning.OwlReasonResult.model_fields
    assert set(owl_fields) == {
        "schema_digests",
        "direct_subclasses",
        "subclasses",
        "subclass_conf",
        "instances",
        "instance_conf",
        "property_facts",
        "consistent",
        "unsatisfiable",
    }
    assert owl_fields["schema_digests"].annotation == list[str]
    assert owl_fields["property_facts"].annotation == list[reasoning.OwlPropertyFact]
    assert owl_fields["consistent"].annotation is bool
    assert all(field.is_required() for field in owl_fields.values())

    explain_fields = reasoning.OwlExplainResult.model_fields
    assert set(explain_fields) == {
        "schema_digests",
        "found",
        "tree",
        "consistent",
        "unsatisfiable",
    }
    assert explain_fields["tree"].annotation == reasoning.ProofNodeWire | None
    assert explain_fields["tree"].default is None

    datalog_fields = reasoning.DatalogReasoningResult.model_fields
    assert set(datalog_fields) == {
        "schema_digests",
        "inferred_count",
        "inferred_triples",
    }
    assert datalog_fields["schema_digests"].annotation == list[str]
    assert datalog_fields["inferred_count"].annotation is int
    assert datalog_fields["inferred_triples"].annotation == list[dict[str, str]]

    roundtrip_values = (
        reasoning.ShaclValidateRequest(),
        rdf_report.ShaclValidationReport(
            schema_digests=["sha256:schema"],
            composed_digest="sha256:schema",
            conforms=True,
            results=[],
        ),
        graph_schema.GraphSchemaSourcesView(
            schema_version=1,
            graph="tenant:test",
            core_catalog_digest="sha256:core",
            composed_digest="sha256:composed",
            core_sources=[],
            dynamic_sources=[],
        ),
        reasoning.OwlReasonResult(
            schema_digests=["sha256:schema"],
            direct_subclasses=[],
            subclasses=[],
            subclass_conf=[],
            instances=[],
            instance_conf=[],
            property_facts=[],
            consistent=True,
            unsatisfiable=[],
        ),
        reasoning.OwlExplainResult(
            schema_digests=["sha256:schema"],
            found=False,
            tree=None,
            consistent=True,
            unsatisfiable=[],
        ),
        reasoning.DatalogReasoningResult(
            schema_digests=["sha256:schema"],
            inferred_count=0,
            inferred_triples=[],
        ),
    )
    for value in roundtrip_values:
        assert type(value).model_validate(value.model_dump(mode="json")) == value


def test_migrated_au_core_schema_authority_is_absent() -> None:
    retained = sorted(
        str(relative) for relative in MIGRATED_ASSETS if (ROOT / relative).exists()
    )
    assert not retained, f"migrated AU schema assets still exist: {retained}"

    references = _stale_references()
    assert not references, "migrated AU schema references still exist:\n" + "\n".join(
        references
    )


def test_proven_local_ontology_and_shape_authorities_are_absent() -> None:
    retained = sorted(
        str(relative)
        for relative in PROVEN_CUTOVER_LOCAL_AUTHORITIES
        if (ROOT / relative).exists()
    )
    assert not retained, f"proven local AU semantic authorities still exist: {retained}"

    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    forbidden_dependencies = sorted(
        dependency
        for dependency in ("pyshacl", "owlrl", "owlready2")
        if dependency in project.casefold()
    )
    assert not forbidden_dependencies, (
        "AU still declares dependencies owned by the proven GraphSchema/SHACL cutover: "
        f"{forbidden_dependencies}"
    )


# ── EH-472/EH-473: no RDF/SHACL/OWL library anywhere in AU ─────────────────────

FORBIDDEN_SEMANTIC_LIBRARIES = frozenset({"rdflib", "pyshacl", "owlrl", "owlready2"})
SEMANTIC_SCAN_ROOTS = (Path("agent_utilities"), Path("tests"), Path("scripts"))
_DYNAMIC_IMPORTERS = frozenset(
    {"importorskip", "import_module", "find_spec", "__import__"}
)
_SEMANTIC_DOCUMENT_SUFFIXES = frozenset({".ttl", ".owl", ".nt", ".rdf", ".jsonld"})


def _callee(func: ast.expr) -> str:
    if isinstance(func, ast.Attribute):
        return func.attr
    return func.id if isinstance(func, ast.Name) else ""


def _imported_names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom) and node.level == 0:
        return [node.module or ""]
    if (
        isinstance(node, ast.Call)
        and _callee(node.func) in _DYNAMIC_IMPORTERS
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    ):
        return [node.args[0].value]
    return []


def semantic_library_imports(root: Path, *, relative_to: Path) -> list[str]:
    """Every static or dynamic import of a forbidden library under ``root``."""
    findings: list[str] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        findings.extend(
            f"{path.relative_to(relative_to)}:{node.lineno}: {name}"
            for node in ast.walk(tree)
            for name in _imported_names(node)
            if name.split(".", 1)[0] in FORBIDDEN_SEMANTIC_LIBRARIES
        )
    return findings


def test_au_runtime_tests_and_scripts_import_no_semantic_library() -> None:
    findings = [
        finding
        for root in SEMANTIC_SCAN_ROOTS
        for finding in semantic_library_imports(ROOT / root, relative_to=ROOT)
    ]
    assert not findings, "AU imports an EG-owned semantic library:\n" + "\n".join(
        findings
    )


def test_the_semantic_import_gate_catches_a_planted_import(tmp_path: Path) -> None:
    """Known-bad proof: the scan reports every import shape and ignores prose."""
    planted = {
        "static.py": "import rdflib\n",
        "from_import.py": "from pyshacl import validate\n",
        "skip.py": "import pytest\n\nowlrl = pytest.importorskip('owlrl')\n",
        "dynamic.py": "import importlib\n\nimportlib.import_module('owlready2.reasoning')\n",
        "prose.py": "# rdflib is gone\nNOTE = 'pyshacl was removed'\n",
    }
    for name, text in planted.items():
        (tmp_path / name).write_text(text, encoding="utf-8")

    findings = semantic_library_imports(tmp_path, relative_to=tmp_path)

    assert findings == [
        "dynamic.py:3: owlready2.reasoning",
        "from_import.py:1: pyshacl",
        "skip.py:3: owlrl",
        "static.py:1: rdflib",
    ]


def _declared_requirements() -> list[str]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    declared = list(project["project"].get("dependencies", []))
    for values in project["project"].get("optional-dependencies", {}).values():
        declared.extend(values)
    for values in project.get("dependency-groups", {}).values():
        declared.extend(value for value in values if isinstance(value, str))
    return declared


def test_packaging_declares_and_installs_no_semantic_library() -> None:
    declared = [
        requirement
        for requirement in _declared_requirements()
        if any(
            requirement.casefold().startswith(name)
            for name in FORBIDDEN_SEMANTIC_LIBRARIES
        )
    ]
    assert not declared, f"AU declares EG-owned semantic libraries: {declared}"
    dockerfile = (ROOT / "docker" / "graphos-unified.Dockerfile").read_text(
        encoding="utf-8"
    )
    smoke_imports = sorted(
        name for name in FORBIDDEN_SEMANTIC_LIBRARIES if f"import {name}" in dockerfile
    )
    assert not smoke_imports, f"the unified image imports {smoke_imports}"


def test_au_ships_no_ontology_or_shape_document() -> None:
    shipped = sorted(
        str(path.relative_to(ROOT))
        for path in (ROOT / "agent_utilities").rglob("*")
        if path.suffix in _SEMANTIC_DOCUMENT_SUFFIXES
    )
    assert not shipped, f"AU ships EG-owned semantic documents: {shipped}"
