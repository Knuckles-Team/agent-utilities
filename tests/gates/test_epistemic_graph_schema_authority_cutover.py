"""Enforce the final one-writer core-schema boundary between AU and EG."""

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
COMPONENT_PACK_ASSET = Path("agent_utilities/ontology/shapes/governance.shapes.ttl")
MIGRATED_REFERENCE_TOKENS = frozenset(
    {
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
        if path.resolve() == this_file or path == ROOT / COMPONENT_PACK_ASSET:
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
    assert set(request_fields) == {"data_graph", "shapes"}
    assert request_fields["data_graph"].annotation == str | None
    assert request_fields["shapes"].annotation == str | None
    assert request_fields["data_graph"].default is None
    assert request_fields["shapes"].default is None

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
