"""Auto-extracted graph-os MCP tools: ontology_tools (register_ontology_tools).

Split out of kg_server._build_server to deepen the MCP surface into focused
modules without changing tool behavior or names.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import Field

from agent_utilities.mcp import kg_server

logger = logging.getLogger(__name__)


def _run_coro(coro: Any) -> Any:
    """Run an async coroutine from a sync MCP handler, loop-running or not.

    The concept tools are registered sync (FastMCP runs them off the event loop)
    but ``kg_server._execute_tool`` is async. When no loop is running we
    ``asyncio.run``; when one is, we run on a worker thread with its own loop so
    we never re-enter a running loop.
    """
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(lambda: asyncio.run(coro)).result()


def register_ontology_tools(mcp):
    """Register the ontology_tools group on the given FastMCP server."""

    @mcp.tool(
        name="ontology_property_types",
        description="List the ontology property-type registry and resolve/validate a Palantir-style type ref (CONCEPT:KG-2.47).",
        tags=["graph-os", "ontology"],
    )
    def ontology_property_types(
        action: str = Field(
            default="list",
            description="'list' all type names, 'describe' a type, 'column_type' a type's column DDL string, or 'validate' a value.",
        ),
        type_ref: str = Field(
            default="", description="A type ref, e.g. 'array<string>' or 'vector<768>'."
        ),
        value: str = Field(
            default="", description="JSON-encoded value for action='validate'."
        ),
    ) -> str:
        """List/describe ontology property types and resolve/validate a type reference."""
        from agent_utilities.knowledge_graph.ontology.property_types import (
            column_type_for,
            get_property_type,
            list_property_types,
            validate_value,
        )

        try:
            if action == "list":
                return json.dumps({"property_types": list_property_types()})
            if action == "describe":
                pt = get_property_type(type_ref)
                if pt is None:
                    return json.dumps({"error": f"unknown type: {type_ref!r}"})
                return json.dumps(pt.model_dump(), default=str)
            if action == "column_type":
                return json.dumps(
                    {"type_ref": type_ref, "column_type": column_type_for(type_ref)}
                )
            if action == "validate":
                parsed = json.loads(value) if value else None
                return json.dumps(
                    {"type_ref": type_ref, "valid": validate_value(type_ref, parsed)}
                )
            return json.dumps({"error": f"unknown action: {action!r}"})
        except Exception as e:  # noqa: BLE001
            return json.dumps({"error": str(e)})

    kg_server.REGISTERED_TOOLS["ontology_property_types"] = ontology_property_types

    @mcp.tool(
        name="ontology_value_types",
        description="List/describe constrained ontology value types and validate or coerce a value (CONCEPT:KG-2.39).",
        tags=["graph-os", "ontology"],
    )
    def ontology_value_types(
        action: str = Field(
            default="list", description="'list' | 'describe' | 'validate' | 'coerce'."
        ),
        name: str = Field(
            default="", description="The value-type name, e.g. 'EmailAddress'."
        ),
        value: str = Field(
            default="", description="JSON-encoded value for validate/coerce."
        ),
    ) -> str:
        """List/describe value types and validate or coerce a value through one."""
        from agent_utilities.knowledge_graph.ontology.value_types import (
            coerce_value_type,
            get_value_type,
            list_value_types,
            validate_value_type,
        )

        try:
            if action == "list":
                return json.dumps({"value_types": list_value_types()})
            if action == "describe":
                vt = get_value_type(name)
                if vt is None:
                    return json.dumps({"error": f"unknown value type: {name!r}"})
                return json.dumps(vt.model_dump(), default=str)
            parsed = json.loads(value) if value else None
            if action == "validate":
                return json.dumps(
                    {"name": name, "valid": validate_value_type(name, parsed)}
                )
            if action == "coerce":
                return json.dumps(
                    {"name": name, "value": coerce_value_type(name, parsed)},
                    default=str,
                )
            return json.dumps({"error": f"unknown action: {action!r}"})
        except Exception as e:  # noqa: BLE001
            return json.dumps({"error": str(e)})

    kg_server.REGISTERED_TOOLS["ontology_value_types"] = ontology_value_types

    @mcp.tool(
        name="ontology_interface",
        description="Ontology interfaces: resolve implementers (targeting), check conformance, or emit OWL (CONCEPT:KG-2.38). Set registry='enterprise' to operate on the enterprise-standard contracts (CONCEPT:KG-2.49).",
        tags=["graph-os", "ontology"],
    )
    def ontology_interface(
        action: str = Field(
            default="list",
            description="'list' interfaces, 'implementers' (resolve an interface/type to concrete types), 'conforms' (check an object), or 'owl'.",
        ),
        name: str = Field(default="", description="Interface or concrete type name."),
        object_json: str = Field(
            default="{}", description="JSON object dict for action='conforms'."
        ),
        registry: str = Field(
            default="structural",
            description="Which interface registry: 'structural' (built-in shapes) or 'enterprise' (enterprise-standard contracts, CONCEPT:KG-2.49).",
        ),
    ) -> str:
        """Resolve interface targeting, check conformance, or emit interface OWL/SHACL."""
        from agent_utilities.knowledge_graph.ontology.interfaces import (
            DEFAULT_INTERFACE_REGISTRY,
            target_object_types,
        )
        from agent_utilities.knowledge_graph.standardization.standards import (
            ENTERPRISE_STANDARD_REGISTRY,
        )

        reg = (
            ENTERPRISE_STANDARD_REGISTRY
            if str(registry).lower() == "enterprise"
            else DEFAULT_INTERFACE_REGISTRY
        )
        try:
            if action == "list":
                return json.dumps(
                    {
                        "registry": registry,
                        "interfaces": [i.name for i in reg.list_interfaces()],
                    }
                )
            if action == "implementers":
                impls = (
                    reg.resolve_target(name)
                    if reg is not DEFAULT_INTERFACE_REGISTRY
                    else target_object_types(name)
                )
                return json.dumps({"target": name, "implementers": impls})
            if action == "conforms":
                obj = json.loads(object_json) if object_json else {}
                return json.dumps(
                    {"interface": name, "conforms": reg.conforms(obj, name)}
                )
            if action == "owl":
                return json.dumps({"owl": reg.to_owl()})
            return json.dumps({"error": f"unknown action: {action!r}"})
        except Exception as e:  # noqa: BLE001
            return json.dumps({"error": str(e)})

    kg_server.REGISTERED_TOOLS["ontology_interface"] = ontology_interface

    @mcp.tool(
        name="ontology_sampling_profile",
        description=(
            "Task-aware LLM sampling profiles (CONCEPT:ORCH-1.58/KG-2.94): list/describe "
            "the per-task-class profiles, 'resolve' the profile that would be picked for a "
            "prompt/role, 'set' a profile (SHACL-validated), 'evolve' one round, or emit OWL."
        ),
        tags=["graph-os", "ontology"],
    )
    def ontology_sampling_profile(
        action: str = Field(
            default="list",
            description="'list' | 'describe' | 'resolve' | 'set' | 'evolve' | 'owl'.",
        ),
        task_class: str = Field(
            default="", description="Task class for describe/set/evolve."
        ),
        task_text: str = Field(
            default="", description="Free-text prompt for action='resolve'."
        ),
        role: str = Field(
            default="", description="Functional role for action='resolve'."
        ),
        profile_json: str = Field(
            default="{}", description="JSON SamplingProfile dict for action='set'."
        ),
    ) -> str:
        """List/describe/resolve/set/evolve sampling profiles, or emit their OWL."""
        from agent_utilities.agent.sampling_profile import (
            SamplingProfile,
            resolve_sampling_profile,
        )
        from agent_utilities.knowledge_graph.ontology.value_types import (
            sampling_profile_violations,
        )
        from agent_utilities.models.model_registry import (
            _DEFAULT_TASK_PROFILES,
            inference_owl_ttl,
            load_active_registry,
        )

        try:
            registry = load_active_registry()
            if action == "list":
                effective = {**_DEFAULT_TASK_PROFILES, **registry.task_class_profiles}
                return json.dumps(
                    {"profiles": {k: v.model_dump() for k, v in effective.items()}},
                    default=str,
                )
            if action == "describe":
                return json.dumps(
                    registry.pick_profile_for_task(task_class).model_dump(), default=str
                )
            if action == "resolve":
                prof = resolve_sampling_profile(task_text or None, role=role or None)
                return json.dumps(prof.model_dump(), default=str)
            if action == "set":
                data = json.loads(profile_json) if profile_json else {}
                if task_class:
                    data.setdefault("task_class", task_class)
                profile = SamplingProfile.model_validate(data)
                violations = sampling_profile_violations(profile.model_dump())
                if violations:
                    return json.dumps(
                        {"error": "SHACL bound violation", "violations": violations}
                    )
                registry.set_task_profile(profile)
                return json.dumps({"set": profile.model_dump()}, default=str)
            if action == "evolve":
                from agent_utilities.harness.variant_pool import VariantPool
                from agent_utilities.knowledge_graph.core.engine import (
                    IntelligenceGraphEngine,
                )

                engine = IntelligenceGraphEngine.get_active()
                kg = getattr(engine, "kg", None) or getattr(engine, "_kg", None)
                ci = kg.retrieval if kg is not None else None
                if ci is None:
                    return json.dumps(
                        {"error": "no capability index available to score"}
                    )
                vp = VariantPool.__new__(VariantPool)

                # Live eval: reward = mean capability-index reward already recorded for
                # this profile's prior outcomes; absent history, neutral 0.5 keeps the
                # incumbent. The daemon/evolve loop feeds real outcomes over time.
                def _evaluator(p: SamplingProfile) -> float:
                    return ci.reward_of(vp._profile_id(task_class, p))

                promoted = vp.evolve_profile(registry, task_class, ci, _evaluator)
                return json.dumps({"promoted": promoted.model_dump()}, default=str)
            if action == "owl":
                return json.dumps({"owl": inference_owl_ttl(registry)})
            return json.dumps({"error": f"unknown action: {action!r}"})
        except Exception as e:  # noqa: BLE001
            return json.dumps({"error": str(e)})

    kg_server.REGISTERED_TOOLS["ontology_sampling_profile"] = ontology_sampling_profile

    @mcp.tool(
        name="ontology_leanix_sync",
        description="Discover the live LeanIX metamodel and mirror it natively as OWL/RDF: regenerates ontology_leanix.ttl (every fact sheet type, relation, field) and registers the types for OWL promotion (CONCEPT:KG-2.9). dry_run=true previews without writing.",
        tags=["graph-os", "ontology"],
    )
    def ontology_leanix_sync(
        dry_run: bool = Field(
            default=True,
            description="Preview the generated ontology without writing (default). Set false to apply.",
        ),
    ) -> str:
        """Compile the live LeanIX data model into OWL and apply it to both reasoning layers."""
        from agent_utilities.knowledge_graph.ontology.leanix_metamodel import (
            sync_leanix_ontology,
        )

        try:
            return json.dumps(sync_leanix_ontology(dry_run=bool(dry_run)))
        except Exception as e:  # noqa: BLE001
            return json.dumps({"error": str(e)})

    kg_server.REGISTERED_TOOLS["ontology_leanix_sync"] = ontology_leanix_sync

    @mcp.tool(
        name="graph_ontology",
        description=(
            "Hosted-ontology lifecycle CRUD (CONCEPT:KG-2.265) — manage arbitrary "
            "OWL/RDF ontologies hosted in the running KG. action='load' (parse + "
            "SHACL-validate + register a .ttl/OWL from a file path, URL, or raw "
            "turtle text via `source`/`source_type`, idempotent on iri+version, and "
            "load its axioms into the native reasoner), 'list' (every hosted "
            "ontology with metadata: iri/version/#classes/#properties/#axioms/"
            "loaded_at/active), 'get' (inspect one ontology's classes/properties/"
            "axioms; serialize=true returns turtle), 'update' (load a NEW version, "
            "superseding prior — versioned/bi-temporal), 'delete' (unload from the "
            "hosted set + deactivate), 'validate' (run the valid/connected/SHACL "
            "gate on a candidate WITHOUT committing), 'activate'/'deactivate' "
            "(toggle participation in reasoning)."
        ),
        tags=["graph-os", "ontology", "lifecycle"],
    )
    def graph_ontology(
        action: str = Field(
            default="list",
            description="load | list | get | update | delete | validate | activate | deactivate.",
        ),
        source: str = Field(
            default="",
            description="For load/update/validate: a .ttl/OWL file path, an HTTP(S) URL, or raw turtle/RDF text.",
        ),
        source_type: str = Field(
            default="auto",
            description="How to read `source`: 'file' | 'url' | 'text' | 'auto' (sniff).",
        ),
        iri: str = Field(
            default="",
            description="Ontology IRI (get/update/delete/activate/deactivate; optional override for load).",
        ),
        version: str = Field(
            default="",
            description="Ontology version (defaults to '1.0.0' on load; omit on get/delete to target the newest).",
        ),
        serialize: bool = Field(
            default=False,
            description="For action='get': also return the ontology re-serialized to turtle.",
        ),
        active_only: bool = Field(
            default=False,
            description="For action='list': only ontologies currently active for reasoning.",
        ),
        drop_inferences: bool = Field(
            default=False,
            description="For action='delete': also attempt to drop materialized inferences (engine-gap aware).",
        ),
    ) -> str:
        """Load / list / inspect / version / unload / validate hosted ontologies."""
        from agent_utilities.knowledge_graph.ontology.lifecycle import OntologyLifecycle

        try:
            try:
                engine = kg_server._get_engine()
            except Exception:  # noqa: BLE001 — offline → registry-only operations
                engine = None
            lc = OntologyLifecycle(engine=engine)

            if action == "load":
                if not source:
                    return json.dumps({"error": "load requires `source`"})
                return json.dumps(
                    lc.load(
                        source,
                        source_type=source_type,
                        version=version or None,
                        iri=iri or None,
                    ),
                    default=str,
                )
            if action == "list":
                return json.dumps(
                    lc.list_ontologies(active_only=bool(active_only)), default=str
                )
            if action == "get":
                if not iri:
                    return json.dumps({"error": "get requires `iri`"})
                return json.dumps(
                    lc.get(iri, version=version or None, serialize=bool(serialize)),
                    default=str,
                )
            if action == "update":
                if not (source and iri and version):
                    return json.dumps(
                        {"error": "update requires `source`, `iri`, and `version`"}
                    )
                return json.dumps(
                    lc.update(
                        source, iri=iri, version=version, source_type=source_type
                    ),
                    default=str,
                )
            if action == "delete":
                if not iri:
                    return json.dumps({"error": "delete requires `iri`"})
                return json.dumps(
                    lc.delete(
                        iri,
                        version=version or None,
                        drop_inferences=bool(drop_inferences),
                    ),
                    default=str,
                )
            if action == "validate":
                if not source:
                    return json.dumps({"error": "validate requires `source`"})
                return json.dumps(
                    lc.validate(source, source_type=source_type), default=str
                )
            if action in ("activate", "deactivate"):
                if not iri:
                    return json.dumps({"error": f"{action} requires `iri`"})
                return json.dumps(
                    lc.set_active(
                        iri, version=version or None, active=(action == "activate")
                    ),
                    default=str,
                )
            return json.dumps({"error": f"unknown action: {action!r}"})
        except Exception as e:  # noqa: BLE001
            return json.dumps({"error": str(e)})

    kg_server.REGISTERED_TOOLS["graph_ontology"] = graph_ontology

    @mcp.tool(
        name="graph_writeback",
        description="Backfeed KG-derived knowledge into an external system-of-record (CONCEPT:KG-2.8/2.9). target='leanix'|'servicenow'|'erpnext'|'process'|'capability'. ops: inferences_json [{source,rel_type,target}] (relationships), enrichments_json [{node,patches,tag}], creations_json [{type,name}] (inventory CIs/items/fact sheets), retirements_json [{node}]. Fail-closed: live writes need the target's enable flag (e.g. LEANIX_ENABLE_WRITE / SERVICENOW_ENABLE_WRITE / ERPNEXT_ENABLE_WRITE / KG_PROCESS_WRITEBACK); dry_run=true (default) previews the exact proposed writes.",
        tags=["graph-os", "writeback"],
    )
    def graph_writeback(
        target: str = Field(
            default="leanix",
            description="Write-back target: leanix | servicenow | erpnext | process | capability | (any registered sink).",
        ),
        action: str = Field(
            default="write",
            description="'write' (default), 'proposals' (list queued high-stakes proposals), or 'approve' (apply proposal_id).",
        ),
        proposal_id: str = Field(
            default="",
            description="For action='approve': the queued high-stakes proposal id to apply.",
        ),
        inferences_json: str = Field(
            default="[]",
            description="JSON list of inferred edges [{source,rel_type,target}] to write as upstream relations.",
        ),
        enrichments_json: str = Field(
            default="[]",
            description="JSON list of enrichments [{node, patches, tag}] onto existing records.",
        ),
        creations_json: str = Field(
            default="[]",
            description="JSON list of new records [{type,name,...}] to create upstream (inventory CIs/items).",
        ),
        retirements_json: str = Field(
            default="[]",
            description="JSON list [{node}] to retire/decommission upstream (highest risk).",
        ),
        process_ids_json: str = Field(
            default="[]",
            description="For target=process: JSON list of process ids to narrow to.",
        ),
        inventory: bool = Field(
            default=False,
            description="If true, collect the KG's reconciled inventory (infra/topology + LeanIX + TRM, deduped via ALIGNED_WITH) and create the items missing from the target CMDB/ERP.",
        ),
        findings: bool = Field(
            default=False,
            description="If true, file the KG's risk findings (TRM TechnologyRisk: EOL/vuln) as issues in the target tracker (gitlab/github/plane). Pass project context via creations_json[0] or the route.",
        ),
        dry_run: bool = Field(
            default=True,
            description="Preview proposed writes without mutating the system-of-record (default). Set false to apply.",
        ),
    ) -> str:
        """Unified fail-closed write-back to any target system (dry-run-first)."""
        from agent_utilities.knowledge_graph.enrichment.writeback import (
            ProposalQueue,
            approve_proposal,
            push_findings,
            push_inventory,
            run_writeback,
        )

        try:
            try:
                engine = kg_server._get_engine()
            except Exception:  # noqa: BLE001 - offline → no backend resolver
                engine = None
            backend = getattr(engine, "backend", None) if engine is not None else None
            if str(action) == "proposals":
                return json.dumps({"proposals": ProposalQueue().list(status="pending")})
            if str(action) == "approve":
                return json.dumps(
                    approve_proposal(str(proposal_id), backend=backend, engine=engine)
                )
            if bool(inventory):
                return json.dumps(
                    push_inventory(
                        str(target),
                        backend=backend,
                        engine=engine,
                        dry_run=bool(dry_run),
                    )
                )
            if bool(findings):
                project = (
                    json.loads(creations_json)[0]
                    if creations_json and creations_json != "[]"
                    else None
                )
                return json.dumps(
                    push_findings(
                        str(target),
                        backend=backend,
                        engine=engine,
                        project=project,
                        dry_run=bool(dry_run),
                    )
                )
            ops = {
                "inferences": json.loads(inferences_json) if inferences_json else [],
                "enrichments": json.loads(enrichments_json) if enrichments_json else [],
                "creations": json.loads(creations_json) if creations_json else [],
                "retirements": json.loads(retirements_json) if retirements_json else [],
                "process_ids": json.loads(process_ids_json)
                if process_ids_json
                else None,
            }
            return json.dumps(
                run_writeback(
                    str(target),
                    backend=backend,
                    engine=engine,
                    dry_run=bool(dry_run),
                    **ops,
                )
            )
        except Exception as e:  # noqa: BLE001
            return json.dumps({"error": str(e)})

    kg_server.REGISTERED_TOOLS["graph_writeback"] = graph_writeback

    @mcp.tool(
        name="spec_ticket",
        description="Link a KG SDD spec/feature to a Plane/Jira work item and make agents assignable (CONCEPT:KG-2.9). action='link' (push spec content onto the item, link it, assign to user/agent/act-as-user, comment) or 'pull' (read items assigned to a user — 'what do I own?'). Fail-closed (PLANE_ENABLE_WRITE/JIRA_ENABLE_WRITE), dry_run=true previews.",
        tags=["graph-os", "writeback", "sdd"],
    )
    def spec_ticket(
        action: str = Field(default="link", description="'link' or 'pull'."),
        target: str = Field(default="plane", description="'plane' or 'jira'."),
        spec_json: str = Field(
            default="{}",
            description="For action='link': the spec dict {feature_id,title,user_stories,...}.",
        ),
        issue_id: str = Field(
            default="", description="The Plane work-item id / Jira issue key."
        ),
        project_id: str = Field(
            default="", description="Plane project id (for plane)."
        ),
        assignee: str = Field(default="", description="Explicit user id to assign."),
        agent: str = Field(
            default="", description="Agent id to assign (maps via AGENT_USER_MAP)."
        ),
        comment: str = Field(default="", description="Optional comment to post."),
        user: str = Field(
            default="", description="For action='pull': user whose items to read."
        ),
        dry_run: bool = Field(
            default=True, description="Preview without writing (default)."
        ),
    ) -> str:
        """Spec↔ticket↔agent linking + assignment + assigned-items read."""
        from agent_utilities.knowledge_graph.enrichment.writeback import (
            link_spec,
            pull_assigned,
        )

        try:
            try:
                engine = kg_server._get_engine()
            except Exception:  # noqa: BLE001
                engine = None
            backend = getattr(engine, "backend", None) if engine is not None else None
            if str(action) == "pull":
                return json.dumps(
                    pull_assigned(
                        str(target), user=user or None, project_id=project_id or None
                    )
                )
            spec = json.loads(spec_json) if spec_json else {}
            return json.dumps(
                link_spec(
                    spec,
                    target=str(target),
                    issue_id=str(issue_id),
                    project_id=project_id or None,
                    assignee=assignee or None,
                    agent=agent or None,
                    comment=comment or None,
                    backend=backend,
                    dry_run=bool(dry_run),
                )
            )
        except Exception as e:  # noqa: BLE001
            return json.dumps({"error": str(e)})

    kg_server.REGISTERED_TOOLS["spec_ticket"] = spec_ticket

    @mcp.tool(
        name="concept_registry",
        description=(
            "Atomically claim/list/release concept ids across parallel sessions & worktrees "
            "(CONCEPT:OS-5.42). action='reserve' mints the next free id in a namespace (a pillar "
            "like 'KG-2'/'OS-5' or a package prefix like 'KEY') and appends it to the committed, "
            "merge=union ledger so two sessions never collide; 'list' shows reservations; "
            "'release' frees one; 'reconcile' marks landed/expired. The ledger is authoritative; "
            "reservations are also projected into the KG when the gateway is healthy."
        ),
        tags=["graph-os", "governance", "concept"],
    )
    def concept_registry(
        action: str = Field(
            default="list",
            description="'reserve', 'list', 'release', or 'reconcile'.",
        ),
        namespace: str = Field(
            default="",
            description="For 'reserve': pillar ('KG-2','OS-5') or package prefix ('KEY','GL').",
        ),
        session_id: str = Field(
            default="", description="Claiming session id (defaults to host:pid)."
        ),
        design_doc: str = Field(
            default="",
            description="Optional design-doc path recorded with the reservation.",
        ),
        concept_id: str = Field(
            default="", description="For 'release': the id to free."
        ),
        status: str = Field(
            default="",
            description="For 'list': filter by status (reserved/landed/expired).",
        ),
        ttl_seconds: int = Field(
            default=86_400, description="Reservation TTL before it is reclaimable."
        ),
        repo: str = Field(
            default="",
            description="Repo root whose ledger to use (defaults to agent-utilities).",
        ),
    ) -> str:
        """Concept-ID reservation ledger operations (see docs/concept_coordination.md)."""
        import os
        import socket
        from pathlib import Path

        from agent_utilities.governance import concept_allocator as ca

        try:
            repo_root = Path(repo).expanduser().resolve() if repo else ca.REPO_ROOT
            if str(action) == "list":
                return json.dumps(
                    {
                        "reservations": ca.list_reservations(
                            repo_root=repo_root, status=status or None
                        )
                    }
                )
            if str(action) == "reconcile":
                return json.dumps(ca.reconcile(repo_root=repo_root))
            if str(action) == "release":
                if not concept_id:
                    return json.dumps({"error": "release requires concept_id"})
                return json.dumps(
                    {"released": ca.release_concept_id(concept_id, repo_root=repo_root)}
                )
            if str(action) == "reserve":
                if not namespace:
                    return json.dumps({"error": "reserve requires namespace"})
                sid = session_id or f"{socket.gethostname()}:{os.getpid()}"
                record = ca.reserve_concept_id(
                    namespace,
                    session_id=sid,
                    design_doc=design_doc or None,
                    ttl_seconds=int(ttl_seconds),
                    repo_root=repo_root,
                )
                # Best-effort projection into the KG for queryability — the ledger
                # remains the authoritative claim regardless of gateway health.
                try:
                    from agent_utilities.mcp.kg_coordinator import KGCoordinator

                    if KGCoordinator.is_server_healthy():
                        _run_coro(
                            kg_server._execute_tool(
                                "graph_write",
                                action="add_node",
                                node_id=record["id"],
                                node_type="ConceptReservation",
                                properties=json.dumps(record),
                            )
                        )
                        record["kg_projected"] = True
                except Exception:  # noqa: BLE001 - projection is advisory
                    record["kg_projected"] = False
                return json.dumps(record)
            return json.dumps({"error": f"unknown action: {action!r}"})
        except Exception as e:  # noqa: BLE001
            return json.dumps({"error": str(e)})

    kg_server.REGISTERED_TOOLS["concept_registry"] = concept_registry

    @mcp.tool(
        name="source_sync",
        description="THE canonical connector→KG ingestion tool (CONCEPT:KG-2.9) — one entrypoint for every external source. source='leanix'|'camunda'|'servicenow'|'gitlab'|… (any registered hydration/materialize source), OR source='all' to sweep EVERY configured connector in one pass (the fleet-wide background-ingest sweep). mode='delta' (only changes since the watermark, default), 'full' (re-mirror all), or 'reconcile' (tombstone records deleted upstream). Delta-capable sources do incremental sync; all others fall back to a full hydrate, and a generic write-layer content-hash delta means unchanged entities are skipped (no re-write, no re-reason) for ALL sources even on a full fetch. ids=[...] narrows to specific records (webhook-driven). (graph_hydrate is a thin alias of this tool; graph_ingest covers path/URL/document content.)",
        tags=["graph-os", "ingestion"],
    )
    def source_sync(
        source: str = Field(
            default="leanix",
            description="Registered source to sync (e.g. 'leanix', 'camunda', 'servicenow').",
        ),
        mode: str = Field(
            default="delta",
            description="'delta' (watermark poll), 'full' (re-mirror all), or 'reconcile' (tombstone deletions).",
        ),
        ids_json: str = Field(
            default="[]",
            description="JSON list of record ids to narrow the sync (webhook delta).",
        ),
    ) -> str:
        """Run a delta/full/reconcile sync for any registered source against the live engine."""
        from agent_utilities.knowledge_graph.core.source_sync import sync_source

        try:
            ids = json.loads(ids_json) if ids_json else []
            try:
                engine = kg_server._get_engine()
            except Exception:  # noqa: BLE001
                engine = None
            if engine is None:
                return json.dumps({"status": "skipped", "reason": "no active engine"})
            return json.dumps(
                sync_source(engine, str(source), mode=str(mode), ids=ids or None)
            )
        except Exception as e:  # noqa: BLE001
            return json.dumps({"error": str(e)})

    kg_server.REGISTERED_TOOLS["source_sync"] = source_sync

    @mcp.tool(
        name="graph_etl",
        description=(
            "Unified ETL pipeline between systems over the canonical KG hub "
            "(CONCEPT:KG-2.98). One source→(ontological transform)→sink interface that "
            "composes the existing ingestion, write-back, and graph-store machinery. "
            "action='run' (default): pull `source` into the KG (any registered "
            "ingestion source: leanix/servicenow/egeria/camunda/aris/gitlab/…; mode "
            "delta|full|reconcile) and/or load `sink` from the KG — `sink` is either a "
            "write-back system-of-record (leanix/servicenow/egeria/… → dry_run-first + "
            "approval gate; pass ops_json with inferences/enrichments/creations/"
            "retirements) OR a graph store (stardog/neo4j/age/jena_fuseki or a "
            "registered connection name → full data load; SPARQL stores partition into "
            "urn:source:<system> named graphs; sources_json filters the subset). Omit "
            "either side for a one-directional run. action='list': available sources, "
            "write-back sinks, and graph-store backends. action='lineage': recorded "
            "ETL runs (impact analysis), filterable by source/sink (CONCEPT:KG-2.99)."
        ),
        tags=["graph-os", "ingestion", "etl"],
    )
    def graph_etl(
        action: str = Field(default="run", description="'run' | 'list' | 'lineage'."),
        source: str = Field(
            default="", description="Ingestion source to pull into the KG (inbound)."
        ),
        sink: str = Field(
            default="",
            description="Write-back domain or graph-store/connection name (outbound).",
        ),
        mode: str = Field(
            default="delta", description="Inbound sync mode: delta|full|reconcile."
        ),
        sources_json: str = Field(
            default="[]",
            description="JSON list of source systems to filter a graph-store push.",
        ),
        ids_json: str = Field(
            default="[]",
            description="JSON list of record ids to narrow the inbound sync.",
        ),
        ops_json: str = Field(
            default="{}",
            description="JSON write-back payload (inferences/enrichments/creations/…).",
        ),
        dry_run: bool = Field(
            default=True, description="Write-back dry-run (fail-closed default)."
        ),
        limit: int = Field(default=200, description="Max rows for action='lineage'."),
    ) -> str:
        """Run / inspect a unified ETL flow over the canonical KG hub."""
        from agent_utilities.knowledge_graph.enrichment.registry import (
            discover_extractors,
            list_sources,
        )
        from agent_utilities.knowledge_graph.enrichment.writeback.core import (
            get_sink,
            list_sinks,
        )

        try:
            engine = kg_server._get_engine()
        except Exception:  # noqa: BLE001
            engine = None

        try:
            if action == "list":
                discover_extractors()
                names = sorted({s.category for s in list_sources()})
                reg = kg_server.get_connection_registry()
                backends = sorted(
                    set(reg.names())
                    | {"stardog", "neo4j", "falkordb", "age", "jena_fuseki"}
                )
                return json.dumps(
                    {"sources": names, "sinks": list_sinks(), "backends": backends}
                )

            if engine is None:
                return json.dumps({"status": "skipped", "reason": "no active engine"})

            if action == "lineage":
                from agent_utilities.knowledge_graph.etl import query_lineage

                return json.dumps(
                    {
                        "runs": query_lineage(
                            engine,
                            source=source or None,
                            sink=sink or None,
                            limit=int(limit),
                        )
                    },
                    default=str,
                )

            # action == "run"
            from agent_utilities.knowledge_graph.etl import run_etl

            ids = json.loads(ids_json) if ids_json else []
            srcs = json.loads(sources_json) if sources_json else []
            ops = json.loads(ops_json) if ops_json else {}

            # Resolve a graph-store sink backend (write-back sinks need none).
            sink_backend = None
            if sink and get_sink(sink) is None:
                reg = kg_server.get_connection_registry()
                if sink in reg.names():
                    be = getattr(reg.get_engine(sink), "backend", None)
                    sink_backend = getattr(be, "_authority", be)
                else:
                    from agent_utilities.knowledge_graph.backends import create_backend

                    sink_backend = create_backend(backend_type=sink)

            return json.dumps(
                run_etl(
                    engine,
                    source=source or None,
                    mode=str(mode),
                    ids=ids or None,
                    sink=sink or None,
                    sink_backend=sink_backend,
                    sources=srcs or None,
                    dry_run=bool(dry_run),
                    ops=ops or None,
                ),
                default=str,
            )
        except Exception as e:  # noqa: BLE001
            return json.dumps({"error": str(e)})

    kg_server.REGISTERED_TOOLS["graph_etl"] = graph_etl

    @mcp.tool(
        name="ontology_function",
        description="Typed, versioned ontology functions: list or invoke through the governed runtime (CONCEPT:KG-2.41).",
        tags=["graph-os", "ontology"],
    )
    def ontology_function(
        action: str = Field(
            default="list", description="'list' registered functions or 'invoke' one."
        ),
        name: str = Field(default="", description="Function name for action='invoke'."),
        params: str = Field(
            default="{}", description="JSON-encoded typed input params."
        ),
        version: str = Field(default="", description="Optional pinned semver version."),
        actor: str = Field(
            default="mcp:caller",
            description="Invoking actor id (recorded in the audit entry).",
        ),
    ) -> str:
        """List registered ontology functions or invoke one with typed params."""
        from agent_utilities.knowledge_graph.ontology.functions import (
            DEFAULT_FUNCTION_REGISTRY,
        )

        try:
            if action == "list":
                return json.dumps(
                    [
                        {
                            "name": s.name,
                            "version": s.version,
                            "kind": str(s.kind),
                            "released": s.released,
                            "inputs": [p.model_dump() for p in s.inputs],
                            "output": str(s.output),
                            "description": s.description,
                        }
                        for s in DEFAULT_FUNCTION_REGISTRY.list_functions()
                    ],
                    default=str,
                )
            if action == "invoke":
                actor_id = actor or "mcp:caller"
                ont = kg_server._ontology_system()
                parsed = json.loads(params) if params else {}
                result = ont.invoke_function(
                    name, parsed, version or None, actor_id=actor_id
                )
                return json.dumps(result.model_dump(), default=str)
            return json.dumps({"error": f"unknown action: {action!r}"})
        except Exception as e:  # noqa: BLE001
            return json.dumps({"error": str(e)})

    kg_server.REGISTERED_TOOLS["ontology_function"] = ontology_function

    @mcp.tool(
        name="ontology_derive",
        description="Compute derived (function/cypher/sparql/embedding-backed) properties live at read time (CONCEPT:KG-2.40).",
        tags=["graph-os", "ontology"],
    )
    def ontology_derive(
        action: str = Field(
            default="compute",
            description="'list' declarations, 'compute' one property, 'compute_all', "
            "or 'discover_extensions' (propose ontology .ttl extensions from a text "
            "sample, CONCEPT:KG-2.259).",
        ),
        object_json: str = Field(
            default="{}", description="JSON object dict the property is computed for."
        ),
        name: str = Field(
            default="", description="Derived-property name for action='compute'."
        ),
        object_type: str = Field(
            default="",
            description="Optional object type for declaration resolution; the "
            "content/source type for action='discover_extensions'.",
        ),
        sample_text: str = Field(
            default="",
            description="Representative document text for action='discover_extensions'.",
        ),
    ) -> str:
        """Compute derived properties / discover ontology extensions."""
        from agent_utilities.knowledge_graph.ontology.derived_properties import (
            DEFAULT_DERIVED_REGISTRY,
        )

        try:
            if action == "discover_extensions":
                # Ontology-aware schema discovery (KG-2.259): propose .ttl extensions
                # from a text sample, diffed against the live ontology. Human/SHACL-
                # gated — returns a proposal, never auto-merges.
                from agent_utilities.knowledge_graph.enrichment.cards import (
                    make_lite_llm_fn,
                )
                from agent_utilities.knowledge_graph.extraction.schema_discovery import (
                    discover_schema_extensions,
                    discovery_report,
                )

                texts = [sample_text] if sample_text else []
                discovered = discover_schema_extensions(
                    texts, object_type or "document", make_lite_llm_fn()
                )
                return json.dumps(discovery_report(discovered), default=str)
            if action == "list":
                return json.dumps(
                    [
                        {
                            "name": d.name,
                            "object_type": d.object_type,
                            "backing": str(d.backing),
                            "output_type": str(d.output_type),
                            "description": d.description,
                        }
                        for d in DEFAULT_DERIVED_REGISTRY.list_all()
                    ],
                    default=str,
                )
            ont = kg_server._ontology_system()
            obj = json.loads(object_json) if object_json else {}
            otype = object_type or None
            if action == "compute":
                res = ont.derive(obj, name, object_type=otype)
                return json.dumps(res.model_dump(), default=str)
            if action == "compute_all":
                return json.dumps(ont.derive_all(obj, object_type=otype), default=str)
            return json.dumps({"error": f"unknown action: {action!r}"})
        except Exception as e:  # noqa: BLE001
            return json.dumps({"error": str(e)})

    kg_server.REGISTERED_TOOLS["ontology_derive"] = ontology_derive

    @mcp.tool(
        name="ontology_link_materialize",
        description="Reify a many-to-many ontology link as a (junction_node, edge_a, edge_b) triple and write it (CONCEPT:KG-2.26).",
        tags=["graph-os", "ontology"],
    )
    async def ontology_link_materialize(
        action: str = Field(
            default="materialize",
            description="'types' to list link types, or 'materialize' a junction.",
        ),
        link_name: str = Field(
            default="", description="The junction link type name, e.g. 'agent_skill'."
        ),
        source_id: str = Field(default="", description="Source endpoint node id."),
        target_id: str = Field(default="", description="Target endpoint node id."),
        properties: str = Field(
            default="{}", description="JSON-encoded junction (link) properties."
        ),
    ) -> str:
        """List link types or reify + persist a M:N link via the graph_write path."""
        from agent_utilities.knowledge_graph.ontology.links import DEFAULT_LINK_REGISTRY

        try:
            if action == "types":
                return json.dumps(
                    [
                        {
                            "name": link.name,
                            "source_type": str(link.source_type),
                            "target_type": str(link.target_type),
                            "edge_type": str(link.edge_type),
                            "cardinality": str(link.cardinality),
                            "is_junction": link.name
                            in {j.name for j in DEFAULT_LINK_REGISTRY.junctions()},
                        }
                        for link in DEFAULT_LINK_REGISTRY.list_links()
                    ],
                    default=str,
                )
            ont = kg_server._ontology_system()
            props = json.loads(properties) if properties else {}
            node, edge_a, edge_b = ont.materialize_link(
                link_name, source_id, target_id, props
            )
            # Persist via the existing graph_write add_node / add_edge primitives.
            await kg_server._execute_tool(
                "graph_write",
                action="add_node",
                node_type=str(node.type),
                node_id=node.id,
                properties=json.dumps(
                    {"name": node.name, **(node.metadata or {})}, default=str
                ),
            )
            for edge in (edge_a, edge_b):
                await kg_server._execute_tool(
                    "graph_write",
                    action="add_edge",
                    source_id=edge.source,
                    target_id=edge.target,
                    rel_type=str(edge.type),
                    properties=json.dumps(edge.metadata or {}, default=str),
                )
            return json.dumps(
                {
                    "junction_id": node.id,
                    "edge_a_type": str(edge_a.type),
                    "edge_b_type": str(edge_b.type),
                },
                default=str,
            )
        except Exception as e:  # noqa: BLE001
            return json.dumps({"error": str(e)})

    kg_server.REGISTERED_TOOLS["ontology_link_materialize"] = ontology_link_materialize

    @mcp.tool(
        name="object_edits",
        description=(
            "Durable object-edit ledger (CONCEPT:KG-2.43): record a structured edit "
            "(property_set/link_add/link_remove/object_create/object_delete), revert "
            "an edit, or read per-object history / as_of snapshot. For a property_set, "
            "pass 'expect' to record the edit ONLY if the object still matches those "
            "values (CONCEPT:KG-2.142) — optimistic concurrency for concurrent agent "
            "object-shaping so two agents editing the same object never lose a write."
        ),
        tags=["graph-os", "ontology"],
    )
    def object_edits(
        action: str = Field(
            default="history",
            description="'record' an edit | 'revert' an edit by id | 'history' per object | 'as_of' snapshot.",
        ),
        object_id: str = Field(
            default="", description="Target object id (record/history/as_of)."
        ),
        edit_type: str = Field(
            default="property_set",
            description="property_set|link_add|link_remove|object_create|object_delete (for action='record').",
        ),
        properties_json: str = Field(
            default="{}",
            description="JSON property map (record property_set/object_create).",
        ),
        link_target: str = Field(
            default="", description="Link target id (record link_add/link_remove)."
        ),
        link_label: str = Field(
            default="related", description="Link label (record link_add/link_remove)."
        ),
        edit_id: str = Field(default="", description="Edit id (action='revert')."),
        ts: float = Field(default=0.0, description="Unix timestamp (action='as_of')."),
        actor: str = Field(
            default="system", description="Acting principal recorded on the edit."
        ),
        expect: dict = Field(
            default_factory=dict,
            description=(
                "For action='record' edit_type='property_set': field→expected current "
                "value the object must still match for the set to apply (missing field "
                "reads as null). When non-empty the set goes through an atomic "
                "compare-and-set (CONCEPT:KG-2.142): the edit is recorded ONLY if it "
                "wins; an unapplied set returns {'applied': false} and records nothing. "
                "Empty (default) = unconditional set, identical to prior behavior. "
                "e.g. {'status': 'pending'}."
            ),
        ),
    ) -> str:
        """Record / revert object edits and read per-object edit history or an as_of snapshot.

        CONCEPT:KG-2.142 optimistic-concurrency for object property edits.
        ``action='record'`` with a non-empty ``expect`` is the object-layer
        optimistic-concurrency primitive: the property_set is
        applied through the engine's atomic ``compare_and_set_node_fields`` and the
        ledger edit is recorded **only if the precondition still holds** — use it
        when concurrent agents shape the same object so one never clobbers another.
        """
        from agent_utilities.knowledge_graph.ontology.edits import (
            Edit,
            EditType,
            revert_edit,
        )

        def _as_dict(v: Any) -> dict:
            # Omitted dict params arrive as the unresolved FastMCP ``FieldInfo``
            # (default_factory is not resolved by the internal/REST dispatcher);
            # coerce anything non-dict — and a JSON-string some clients send.
            if isinstance(v, dict):
                return v
            if isinstance(v, str) and v.strip():
                try:
                    parsed = json.loads(v)
                    return parsed if isinstance(parsed, dict) else {}
                except (ValueError, TypeError):
                    return {}
            return {}

        try:
            ont = kg_server._ontology_system()
            ledger = ont.edits
            if action == "record":
                etype = EditType(edit_type)
                if etype in (EditType.LINK_ADD, EditType.LINK_REMOVE):
                    edit = Edit(
                        actor=actor,
                        edit_type=etype,
                        object_id=object_id,
                        link_source=object_id,
                        link_label=link_label,
                        link_target=link_target,
                    )
                else:
                    props = json.loads(properties_json) if properties_json else {}
                    conditions = _as_dict(expect)
                    if etype == EditType.PROPERTY_SET and conditions:
                        # CONCEPT:KG-2.142 — atomic optimistic-concurrency property
                        # set. The object id IS the node id (the ledger persists the
                        # edit's target as MERGE (t {id: object_id})), so we condition
                        # on the SAME node the edit targets. Apply the set ONLY if the
                        # node still matches ``expect`` (missing field ≡ null), under
                        # the engine write lock. If we lose the race we record NOTHING
                        # and surface applied=false — never a misleading audit edit.
                        engine = kg_server._get_engine()
                        backend = getattr(engine, "backend", None)
                        if backend is None:
                            return json.dumps(
                                {
                                    "action": "compare_and_set",
                                    "object_id": object_id,
                                    "applied": False,
                                    "error": "no engine backend for conditional set",
                                }
                            )
                        applied = bool(
                            backend.compare_and_set_node_fields(
                                object_id, conditions, dict(props)
                            )
                        )
                        if not applied:
                            return json.dumps(
                                {
                                    "action": "compare_and_set",
                                    "object_id": object_id,
                                    "applied": False,
                                }
                            )
                        edit = Edit(
                            actor=actor,
                            edit_type=etype,
                            object_id=object_id,
                            after=dict(props),
                        )
                        recorded = ledger.record(edit)
                        payload = recorded.model_dump()
                        payload["applied"] = True
                        return json.dumps(payload, default=str)
                    edit = Edit(
                        actor=actor,
                        edit_type=etype,
                        object_id=object_id,
                        after=dict(props),
                    )
                recorded = ledger.record(edit)
                return json.dumps(recorded.model_dump(), default=str)
            if action == "revert":
                comp = revert_edit(ledger, edit_id, actor=actor)
                return json.dumps(comp.model_dump(), default=str)
            if action == "history":
                return json.dumps(
                    {
                        "object_id": object_id,
                        "history": [e.model_dump() for e in ledger.history(object_id)],
                    },
                    default=str,
                )
            if action == "as_of":
                return json.dumps(
                    {"object_id": object_id, "snapshot": ledger.as_of(object_id, ts)},
                    default=str,
                )
            return json.dumps({"error": f"unknown action: {action!r}"})
        except Exception as e:  # noqa: BLE001
            return json.dumps({"error": str(e)})

    kg_server.REGISTERED_TOOLS["object_edits"] = object_edits

    @mcp.tool(
        name="object_index",
        description="Object Index Lifecycle / Object Data Funnel (CONCEPT:KG-2.44): batch/incremental sync of the live search index from source nodes, report staleness, or reindex stale objects.",
        tags=["graph-os", "ontology"],
    )
    def object_index(
        action: str = Field(
            default="status",
            description="'sync' (batch rebuild) | 'reindex' (reconcile stale) | 'status' (live/tombstone counts).",
        ),
        nodes_json: str = Field(
            default="[]",
            description="JSON list of source node mappings (sync/reindex).",
        ),
    ) -> str:
        """Sync / reindex the live object search index and report staleness."""
        try:
            ont = kg_server._ontology_system()
            funnel = ont.index_funnel
            if action == "sync":
                nodes = json.loads(nodes_json) if nodes_json else []
                return json.dumps(funnel.batch_sync(nodes).as_dict())
            if action == "reindex":
                nodes = json.loads(nodes_json) if nodes_json else []
                return json.dumps(funnel.reconcile(nodes).as_dict())
            if action == "status":
                return json.dumps(
                    {
                        "live_size": len(funnel),
                        "tombstones": funnel.tombstone_count,
                        "indexed_ids": sorted(funnel.live_ids()),
                    }
                )
            return json.dumps({"error": f"unknown action: {action!r}"})
        except Exception as e:  # noqa: BLE001
            return json.dumps({"error": str(e)})

    kg_server.REGISTERED_TOOLS["object_index"] = object_index

    @mcp.tool(
        name="object_permissioning",
        description="Fine-grained object permissioning (CONCEPT:KG-2.46): redact an object, materialize a restricted view, or attach a mandatory marking. Actor is resolved from the ambient context — never from caller-supplied clearance.",
        tags=["graph-os", "ontology"],
    )
    def object_permissioning(
        action: str = Field(
            default="restricted_view",
            description="'redact' one object | 'restricted_view' an object set | 'mark' attach a marking.",
        ),
        objects_json: str = Field(
            default="[]", description="JSON list of object dicts (restricted_view)."
        ),
        object_json: str = Field(
            default="{}", description="JSON object dict (redact)."
        ),
        node_id: str = Field(default="", description="Node id (action='mark')."),
        marking: str = Field(default="", description="Marking name (action='mark')."),
        mask: bool = Field(
            default=False,
            description="Mask withheld properties instead of dropping them.",
        ),
    ) -> str:
        """Redact / restrict / mark objects for the AMBIENT actor (no spoofable clearance)."""
        from agent_utilities.knowledge_graph.ontology.permissioning import (
            apply_marking,
            redact_object,
            restricted_view,
        )

        try:
            # actor=None -> resolved from the ambient ActorContext set by the
            # dispatcher's use_actor(); callers cannot inject their own clearance.
            if action == "redact":
                obj = json.loads(object_json) if object_json else {}
                return json.dumps(redact_object(obj, None, mask=mask), default=str)
            if action == "restricted_view":
                objs = json.loads(objects_json) if objects_json else []
                return json.dumps(restricted_view(objs, None, mask=mask), default=str)
            if action == "mark":
                apply_marking(node_id, marking)
                return json.dumps(
                    {"node_id": node_id, "marking": marking, "applied": True}
                )
            return json.dumps({"error": f"unknown action: {action!r}"})
        except Exception as e:  # noqa: BLE001
            return json.dumps({"error": str(e)})

    kg_server.REGISTERED_TOOLS["object_permissioning"] = object_permissioning

    @mcp.tool(
        name="graph_share",
        description="Share a private node (CONCEPT:KG-2.60). Data is private-to-its-owner by default; this is the explicit promotion path. action='org' shares with the owner's org (in-place), 'commons' promotes a copy into the shared cross-org commons graph (share by WHERE placed), 'mark' attaches a mandatory marking (share by HOW placed), 'private' restricts it back. Actor/owner is the ambient identity — never caller-supplied.",
        tags=["graph-os", "tenancy"],
    )
    def graph_share(
        action: str = Field(
            default="org",
            description="'org' share with my org | 'commons' promote to the shared commons graph | 'mark' attach a marking | 'private' restrict back to me.",
        ),
        node_id: str = Field(default="", description="Id of the node to share."),
        marking: str = Field(default="", description="Marking name (action='mark')."),
    ) -> str:
        """Explicit, private-by-default sharing for the AMBIENT actor (KG-2.60)."""
        from agent_utilities.knowledge_graph.core import tenant_sharing as _ts

        if not node_id:
            return json.dumps({"error": "node_id is required"})
        try:
            if action == "org":
                _ts.share_with_org(node_id)
                return json.dumps({"node_id": node_id, "shared_scope": "org"})
            if action == "commons":
                ok = _ts.promote_to_commons(node_id)
                return json.dumps(
                    {"node_id": node_id, "shared_scope": "commons", "promoted": ok}
                )
            if action == "mark":
                if not marking:
                    return json.dumps(
                        {"error": "marking is required for action='mark'"}
                    )
                _ts.share(node_id, marking)
                return json.dumps({"node_id": node_id, "marking": marking})
            if action == "private":
                _ts.make_private(node_id)
                return json.dumps({"node_id": node_id, "shared_scope": "private"})
            return json.dumps({"error": f"unknown action: {action!r}"})
        except Exception as e:  # noqa: BLE001
            return json.dumps({"error": str(e)})

    kg_server.REGISTERED_TOOLS["graph_share"] = graph_share

    @mcp.tool(
        name="object_set",
        description="Object Set Service (CONCEPT:KG-2.45/2.38): search/filter/search_around/pivot/aggregate and union/intersect/subtract over Foundry-style object sets.",
        tags=["graph-os", "ontology"],
    )
    def object_set(
        action: str = Field(
            default="of_type",
            description="of_type|from_ids|search|filter|search_around|pivot|aggregate|union|intersect|subtract.",
        ),
        type_or_interface: str = Field(
            default="", description="Object type / interface (of_type)."
        ),
        ids_json: str = Field(
            default="[]",
            description="JSON list of ids (from_ids / set algebra 'other').",
        ),
        query: str = Field(default="", description="Search query (search)."),
        link_type: str = Field(
            default="", description="Link type (search_around/pivot); empty = any."
        ),
        hops: int = Field(default=1, description="Hop count (search_around)."),
        direction: str = Field(
            default="out", description="out|in|both (search_around/pivot)."
        ),
        group_by: str = Field(
            default="", description="Group-by property (pivot/aggregate)."
        ),
        metric: str = Field(
            default="count", description="count|sum|avg|min|max (aggregate)."
        ),
        field: str = Field(
            default="", description="Numeric field (aggregate sum/avg/min/max)."
        ),
        limit: int = Field(default=50, description="Result limit (search)."),
    ) -> str:
        """Compute over a Foundry-style object set: search/filter/traverse/pivot/aggregate/algebra."""
        try:
            ont = kg_server._ontology_system()
            if action == "from_ids" or action in ("union", "intersect", "subtract"):
                base = ont.object_set(json.loads(ids_json) if ids_json else [])
            else:
                base = ont.object_set_of_type(type_or_interface)

            if action in ("of_type", "from_ids"):
                return json.dumps({"ids": base.ids(), "count": base.count()})
            if action == "search":
                res = base.search(query, limit=limit)
                return json.dumps({"ids": res.ids(), "count": res.count()})
            if action == "search_around":
                res = base.search_around(
                    link_type or None, hops=hops, direction=direction
                )
                return json.dumps({"ids": res.ids(), "count": res.count()})
            if action == "pivot":
                piv = base.pivot(link_type or None, group_by, direction=direction)
                return json.dumps(
                    {
                        "link_type": piv.link_type,
                        "group_by": piv.group_by,
                        "groups": piv.groups,
                    },
                    default=str,
                )
            if action == "aggregate":
                agg = base.aggregate(
                    metric, field=field or None, group_by=group_by or None
                )
                return json.dumps(
                    {
                        "metric": agg.metric,
                        "field": agg.field,
                        "group_by": agg.group_by,
                        "groups": {str(k): v for k, v in agg.groups.items()},
                        "total_objects": agg.total_objects,
                    },
                    default=str,
                )
            if action in ("union", "intersect", "subtract"):
                other = (
                    ont.object_set_of_type(type_or_interface)
                    if type_or_interface
                    else ont.object_set([])
                )
                combined = getattr(base, action)(other)
                return json.dumps({"ids": combined.ids(), "count": combined.count()})
            return json.dumps({"error": f"unknown action: {action!r}"})
        except Exception as e:  # noqa: BLE001
            return json.dumps({"error": str(e)})

    kg_server.REGISTERED_TOOLS["object_set"] = object_set

    @mcp.tool(
        name="document_process",
        description="Document → ontology processing (CONCEPT:KG-2.48): extract → chunk(overlap) → embed → materialize a Document + linked Chunk objects through the live graph write path.",
        tags=["graph-os", "ontology"],
    )
    def document_process(
        document: str = Field(
            description="A file path or raw text content to process."
        ),
        text: str = Field(
            default="", description="Optional pre-extracted text (OCR/external)."
        ),
        source: str = Field(default="", description="Provenance label (path/URL)."),
        chunk_size: int = Field(
            default=800, description="Target chunk size in characters."
        ),
        overlap: int = Field(
            default=120, description="Overlap characters between chunks."
        ),
        contextual: bool = Field(
            default=False,
            description="Enable contextual-retrieval enrichment (CONCEPT:KG-2.50): situate each chunk within the document and embed context+chunk for better recall.",
        ),
    ) -> str:
        """Process a document into Document + Chunk ontology objects through the live graph."""
        from agent_utilities.knowledge_graph.facade import KnowledgeGraph
        from agent_utilities.knowledge_graph.ontology.document_processing import (
            ChunkingConfig,
            DocumentProcessor,
        )

        try:
            engine = None
            try:
                engine = kg_server._get_engine()
            except Exception:  # pragma: no cover - defensive
                engine = None
            backend = getattr(engine, "backend", None) if engine is not None else None
            kg = KnowledgeGraph()
            if backend is not None:
                kg._store = backend
            proc = DocumentProcessor(
                kg,
                chunking=ChunkingConfig(chunk_size=chunk_size, overlap=overlap),
                contextual=contextual,
            )
            result = proc.process(document, text=text or None, source=source)
            return json.dumps(
                {
                    "document_id": result.document_id,
                    "chunk_count": result.chunk_count,
                    "persisted": result.persisted,
                    "edges": len(result.edges),
                },
                default=str,
            )
        except Exception as e:  # noqa: BLE001
            return json.dumps({"error": str(e)})

    kg_server.REGISTERED_TOOLS["document_process"] = document_process

    # Quant trading system tool (CONCEPT:ECO-4.0): debate/regime/data/execute/
    # portfolio over the finance engines. Registered onto the MCP server AND the
    # shared kg_server.REGISTERED_TOOLS map so the gateway REST twin (/quant) reaches it.
    # The finance domain needs the optional `[finance]` extra (scipy/pandas/statsmodels);
    # the lean serving image omits it, so guard the registration so kg_server still boots
    # — the quant tool is simply absent there. See AGENTS.md "Dependency discipline".
    try:
        from agent_utilities.domains.finance.quant_mcp_tools import (
            register_quant_tools,
        )

        kg_server.REGISTERED_TOOLS["quant"] = register_quant_tools(mcp, None)
    except ImportError:
        logger.info(
            "quant tools skipped (finance extra not installed) — "
            "install agent-utilities[finance] to enable the `quant` MCP tool"
        )

    @mcp.tool(
        name="source_connector",
        description="Document-source connectors (CONCEPT:ECO-4.25–4.29, KG-2.59): list registered connectors, or run one (filesystem/web/rest/database/mcp:<package>/mcp_tool — mcp_tool drives any fleet MCP server's listing tool as a paginated source) to ingest its documents into the KG as Document+Chunk objects with contextual enrichment (KG-2.50) and external permission sync (ECO-4.28).",
        tags=["graph-os", "ecosystem", "connectors"],
    )
    async def source_connector(
        action: str = Field(
            default="list",
            description="One of: 'list' (registered connector types), 'run' (build + ingest a connector).",
        ),
        source_type: str = Field(
            default="",
            description="Connector type for 'run' (filesystem/web/rest/database/mcp:<package>/mcp_tool).",
        ),
        config: dict = Field(
            default_factory=dict,
            description="Connector configuration dict for 'run' (e.g. {'root': '/docs'} or {'base_url': 'https://…'}).",
        ),
        connector_id: str = Field(
            default="",
            description="Stable id for incremental checkpoint storage (optional).",
        ),
        contextual: bool = Field(
            default=True,
            description="Enable contextual-retrieval enrichment (CONCEPT:KG-2.50).",
        ),
        incremental: bool = Field(
            default=True,
            description="Use the connector's resumable poll (CONCEPT:ECO-4.26) vs a full load.",
        ),
    ) -> str:
        """List or run a document-source connector (CONCEPT:ECO-4.25–4.29)."""
        from agent_utilities.knowledge_graph.facade import KnowledgeGraph

        try:
            if action == "list":
                from agent_utilities.protocols.source_connectors import list_sources

                return json.dumps({"connectors": list_sources()})

            if action == "run":
                if not source_type:
                    return json.dumps(
                        {"error": "source_type is required for action='run'"}
                    )
                engine = None
                try:
                    engine = kg_server._get_engine()
                except Exception:  # pragma: no cover - defensive
                    engine = None
                backend = (
                    getattr(engine, "backend", None) if engine is not None else None
                )
                kg = KnowledgeGraph()
                if backend is not None:
                    kg._store = backend
                result = await kg.ontology.run_connector(
                    source_type,
                    dict(config or {}),
                    connector_id=connector_id or None,
                    contextual=contextual,
                    incremental=incremental,
                )
                return json.dumps(result, default=str)

            return json.dumps(
                {"error": f"unknown action {action!r}; use 'list' or 'run'"}
            )
        except Exception as e:  # noqa: BLE001
            return json.dumps({"error": str(e)})

    kg_server.REGISTERED_TOOLS["source_connector"] = source_connector
