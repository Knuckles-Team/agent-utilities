"""Auto-extracted graph-os MCP tools: write_ingest_tools (register_write_ingest_tools).

Split out of kg_server._build_server to deepen the MCP surface into focused
modules without changing tool behavior or names.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import Field

from agent_utilities.mcp import kg_server


def _parse_source_specs(raw: str, spec_cls: Any) -> list[Any]:
    """Parse skill-graph source specs from a JSON list or ``kind=uri,...`` shorthand.

    JSON form: ``[{"kind": "web", "uri": "https://x", "options": {"max_depth": 2}}]``.
    Shorthand: ``web=https://x,pdf=/a.pdf`` (no per-source options).
    """
    raw = (raw or "").strip()
    if not raw:
        return []
    if raw.startswith("["):
        return [spec_cls.from_dict(d) for d in json.loads(raw)]
    return [spec_cls.parse(tok) for tok in raw.split(",") if tok.strip()]


def register_write_ingest_tools(mcp):
    """Register the write_ingest_tools group on the given FastMCP server."""

    @mcp.tool(
        name="graph_write",
        description="Write nodes, relationships, or register external graphs to the Knowledge Graph.",
        tags=["graph-os", "write", "mutation"],
    )
    def graph_write(
        action: str = Field(
            description="Action to perform (add_node, add_edge, delete_node, delete_edge, register_external_graph, bulk_ingest, store_memory, recall_memory, log_chat, submit_sdd, register_execution, check_loop)."
        ),
        node_id: str = Field(
            default="", description="The unique identifier for the node."
        ),
        node_type: str = Field(
            default="", description="The type or label of the node."
        ),
        properties: str = Field(
            default="{}", description="JSON-encoded dictionary of properties."
        ),
        source_id: str = Field(
            default="", description="The source node ID for an edge."
        ),
        target_id: str = Field(
            default="", description="The target node ID for an edge."
        ),
        rel_type: str = Field(
            default="", description="The relationship type for an edge."
        ),
        endpoint_url: str = Field(
            default="", description="URL for external graph registration."
        ),
        graph_type: str = Field(
            default="",
            description="Type of external graph (e.g., 'sparql', 'graphql').",
        ),
        agent_id: str = Field(
            default="", description="ID of the agent performing the action."
        ),
        nodes: str = Field(
            default="[]",
            description="JSON-encoded list of nodes or tags for bulk operations.",
        ),
        target: str = Field(
            default="",
            description=(
                "CONCEPT:KG-2.63 — named graph connection to write to (default = primary). "
                "Use a registered connection name, or 'all' (or a comma-separated list) to "
                "mirror the SAME write to several backends. Fan-out requires an explicit "
                "multi-target value; the default and a single named target stay single-write."
            ),
        ),
    ) -> str:
        """Write nodes, relationships, or register external graphs. This is the primary mutation interface for the Knowledge Graph."""

        def _write_with_engine(engine: Any) -> str:
            if not engine:
                return "Error: IntelligenceGraphEngine not active."
            try:
                props = json.loads(properties) if properties else {}

                if action == "add_node":
                    if not node_id or not node_type:
                        return "Error: node_id and node_type required"
                    engine.add_node(node_id, node_type, props)
                    return f"Node {node_id} added."
                elif action == "add_edge":
                    if not source_id or not target_id or not rel_type:
                        return "Error: source_id, target_id, and rel_type required"
                    engine.link_nodes(source_id, target_id, rel_type, props)
                    return f"Edge {source_id} -> {target_id} added."
                elif action == "delete_node":
                    engine.delete_node(node_id)
                    return f"Node {node_id} deleted."
                elif action == "delete_edge":
                    engine.delete_edge(source_id, target_id, rel_type)
                    return f"Edge {source_id} -> {target_id} deleted."
                elif action == "register_external_graph":
                    if not endpoint_url:
                        return "Error: endpoint_url required"
                    engine.add_node(
                        endpoint_url, "ExternalGraphReference", {"type": graph_type}
                    )
                    return f"Registered external graph at {endpoint_url}"
                elif action == "bulk_ingest":
                    nodes_list = json.loads(nodes) if nodes else []
                    for n in nodes_list:
                        engine.add_node(
                            n.get("id"), n.get("type", "Node"), n.get("properties", {})
                        )
                    return f"Bulk ingested {len(nodes_list)} nodes."
                elif action in ("store_memory", "recall_memory"):
                    try:
                        from agent_utilities.memory.manager import MemoryManager

                        mm = MemoryManager(engine)
                        if action == "store_memory":
                            mm.store(
                                agent_id=agent_id,
                                content=properties,
                                memory_type=node_type,
                                tags=json.loads(nodes) if nodes else [],
                            )
                            return "Memory stored."
                        else:
                            res = mm.recall(
                                query=properties, memory_type=node_type, top_k=5
                            )
                            return "\n".join([str(r) for r in res])
                    except ImportError:
                        return "Error: memory module not available"
                elif action in (
                    "log_chat",
                    "submit_sdd",
                    "register_execution",
                    "check_loop",
                ):
                    if action == "log_chat":
                        engine.add_node(
                            f"chat_{agent_id}_{hash(properties)}",
                            "ChatLog",
                            {"content": properties, "agent_id": agent_id},
                        )
                        return "Chat logged."
                    elif action == "submit_sdd":
                        engine.add_node(
                            f"sdd_{agent_id}_{hash(properties)}",
                            "SDD",
                            {"content": properties, "agent_id": agent_id},
                        )
                        return "SDD submitted."
                    elif action == "register_execution":
                        engine.add_node(
                            f"exec_{agent_id}", "Execution", {"status": "running"}
                        )
                        return "Execution registered."
                    elif action == "check_loop":
                        return "Loop status: OK"
                    return f"Error: Action '{action}' not implemented."
                else:
                    return f"Error: Unknown write action '{action}'"
            except Exception as e:
                return f"Write error: {str(e)}"

        # CONCEPT:KG-2.63 — resolve target connection(s). Writes only fan out on
        # an EXPLICIT multi-target request ('all' or a list); the default and a
        # single named target stay single-write to avoid accidental multi-store
        # writes.
        try:
            entries, errors, fanout = kg_server._resolve_target_engines(target)
        except Exception as e:
            return f"Write error: {str(e)}"

        # CONCEPT:KG-2.89 — role enforcement: a 'read' (data source) or 'mirror'
        # (fan-out replica) connection rejects direct target= writes. Mirrors are
        # written only through the fan-out outbox, never here.
        registry = kg_server.get_connection_registry()
        errors = dict(errors)
        writable = []
        for name, eng in entries:
            if registry.is_writable(name):
                writable.append((name, eng))
            else:
                errors[name] = (
                    f"connection '{name}' is read-only (role={registry.role(name)})"
                )

        if not fanout:
            if not writable:
                return json.dumps(
                    {"error": errors.get(entries[0][0], "target is read-only")},
                    default=str,
                )
            return _write_with_engine(writable[0][1])

        # Fan-out — per-target timeout so one slow backend can't stall the set.
        results, fan_errors = kg_server.fanout_execute(
            writable, lambda name, eng: _write_with_engine(eng)
        )
        return json.dumps(
            {"targets": results, "errors": {**errors, **fan_errors}}, default=str
        )

    kg_server.REGISTERED_TOOLS["graph_write"] = graph_write

    @mcp.tool(
        name="graph_feedback",
        description=(
            "Record a human correction so the brain learns: correction_type "
            "'outcome' adjusts an entity's reward, 'rule' persists a durable "
            "governance/voice/source rule consulted at retrieval time, 'eval' "
            "adds a regression case, 'reads_avoided' closes the code_context "
            "reads-avoided loop (target_id=the answer's capability_id, "
            "corrected_value=JSON {reads_avoided,files_read,correct,query}) so the "
            "code retriever learns which answers replace a file read (CONCEPT:AHE-3.61). "
            "This is how 'this was wrong, here's the fix' becomes future behaviour "
            "(CONCEPT:KG-2.8)."
        ),
        tags=["graph-os", "feedback", "learning"],
    )
    def graph_feedback(
        correction_type: str = Field(
            description="One of: outcome | rule | eval | reads_avoided."
        ),
        target_id: str = Field(
            description="Entity/episode/query the correction is about."
        ),
        corrected_value: str = Field(
            default="",
            description="The corrected value (reward, expected output, etc.).",
        ),
        reason: str = Field(default="", description="Why — the human's explanation."),
        rule_scope: str = Field(
            default="governance",
            description="For rule corrections: governance | voice | source | preference.",
        ),
        rule_kind: str = Field(
            default="forbid",
            description="For rule corrections: forbid | prefer | demote.",
        ),
        actor_id: str = Field(
            default="human", description="Who issued the correction."
        ),
    ) -> str:
        """Record a human correction (outcome/rule/eval) and apply it durably."""
        engine = kg_server._get_engine()
        if not engine:
            return "Error: IntelligenceGraphEngine not active."
        try:
            from agent_utilities.knowledge_graph.adaptation.feedback import (
                FeedbackService,
            )

            service = FeedbackService.from_engine(engine)
            result = service.record_correction(
                correction_type,
                target_id,
                corrected_value=corrected_value or None,
                reason=reason,
                actor_id=actor_id,
                rule_scope=rule_scope,
                rule_kind=rule_kind,
            )
            return json.dumps(result.as_dict())
        except Exception as e:
            return f"Feedback error: {str(e)}"

    kg_server.REGISTERED_TOOLS["graph_feedback"] = graph_feedback

    @mcp.tool(
        name="graph_ingest",
        description="Smart ingestion for codebases, documents, directories, and conversation logs. Also handles corpus management and job status.",
        tags=["graph-os", "ingest"],
    )
    async def graph_ingest(
        target_path: str = Field(
            default="", description="Path or JSON list of paths to ingest."
        ),
        max_depth: int = Field(
            default=3, description="Maximum directory depth for codebase ingestion."
        ),
        agent_id: str = Field(
            default="", description="ID of the agent performing the ingestion."
        ),
        action: str = Field(
            default="ingest",
            description="Action to perform (ingest, ingest_url, archivebox_sync, skill_workflows, fact_extract, distill, import_pack, ingest_knowledge_pack, agent_toolkit, corpus, jobs, job_status, status, cancel, clear, prioritize, rebuild_indexes, observe, materialize, materialize_source, sync, reflect). 'ingest_url' content-aware single-URL ingest (CONCEPT:KG-2.7): target_path=URL → fetch via the unified resolver (ArchiveBox→crawl4ai→requests) into a Document, and for a research roundup (auto-detected, or forced with description='extract_papers' / disabled with 'no_papers') download the cited papers via scholarx and ingest them too, linking page→paper; runs inline. 'archivebox_sync' pulls preserved ArchiveBox snapshots into the KG (corpus_name='full' = pull ALL, else delta; base_path=JSON list of snapshot ids to select). 'skill_workflows' ingests the universal-skills workflow corpus (workflows/<domain>/<name>/SKILL.md) into the KG as dispatchable WorkflowDefinition DAGs (+WorkflowStep depends_on edges +USES_SKILL links) in the exact WorkflowStore shape execute_workflow reads, so kg-delegation-router / graph_orchestrate execute_workflow can discover and fire them; target_path optionally overrides the corpus root, default=installed universal_skills package; idempotent (content-addressed re-ingest is a no-op); runs as a BACKGROUND job (returns a job_id immediately — the full corpus takes ~150s, over the call ceiling — poll with action=job_status job_id=<id>). 'materialize_source' runs an enterprise source extractor (corpus_name=category, e.g. 'camunda'/'aris'/'egeria'; description=optional JSON extractor config), persists its BusinessProcess/BusinessTask/FLOWS_TO batch into the graph via an in-process vendor client, then runs one OWL reasoning cycle so the new process structure folds into the cross-vendor crosswalk. 'fact_extract' turns a document (description=raw text, or target_path=file) into atomic (subject)-[predicate]->(object) fact edges with confidence/evidence/tags, dedups them, persists to the graph, and returns the facts + JSONL. 'extract_submit'/'extract_jobs'/'extract_status'/'extract_pause'/'extract_resume'/'extract_jsonl' run extraction as a GPU-slot-scheduled job (preempt/backfill/resume on the single GPU) addressed by job_id; max_depth sets rounds. 'distill' exports a KG subgraph to a portable skill-graph (target_path=out dir; corpus_name=seed node id OR description=query; max_depth=hop depth). 'import_pack' re-ingests a distilled skill-graph dir back into the KG (target_path=dir; corpus_name='dedup' to merge duplicates). 'build_skill_graph' runs the UNIFIED skill-graph pipeline (CONCEPT:KG-2.7): acquire from ANY source kind into one standardized skill-graph (corpus_name=name; target_path=output parent dir; base_path=JSON list of sources [{kind,uri,options}] OR 'kind=uri,kind=uri' shorthand over web/pdf/office/dir/url_reader/rest/database/mcp_tool/generated/kg_query; description=optional human description) — always writes the offline corpus + a sources.json provenance/freshness manifest, and ALSO ingests into the KG when the daemon is reachable (degrades cleanly otherwise). 'skill_graph_status' reports freshness of an existing skill-graph (target_path=dir; corpus_name='quick' to skip network sources). 'rebuild_skill_graph' re-acquires from the recorded sources and bumps the version (target_path=dir). Queue control: 'cancel' (job_id), 'clear' (target_path=status filter pending|running|completed|failed|cancelled|zombie|all, default completed), 'prioritize' (job_id, target_path=high|normal).",
        ),
        job_id: str = Field(
            default="", description="ID of the job to check status for."
        ),
        corpus_name: str = Field(
            default="", description="Name of the corpus to add/update."
        ),
        base_path: str = Field(default="", description="Base path for the corpus."),
        description: str = Field(default="", description="Description of the corpus."),
        content_type: str = Field(
            default="",
            description="Internal override only — leave empty. The content type (codebase, document, config, prompt, skill, mcp_server, kb, conversation, policy) is auto-detected from the path, and heavy types (codebase/document) always run on the async job queue. Only set this to force a specific category for an ambiguous path.",
        ),
    ) -> str:
        """Smart ingestion tool to populate the Knowledge Graph with codebases, documents, and memory observations. Monitors async ingestion jobs."""
        engine = kg_server._get_engine()
        if not engine:
            return "Error: IntelligenceGraphEngine not active."

        try:
            if action == "ingest":
                from agent_utilities.knowledge_graph.ingestion.engine import (
                    ContentType,
                    IngestionEngine,
                    IngestionManifest,
                )

                if not target_path:
                    return "Error: target_path required for ingest action"

                # Parse one-or-many paths (JSON list, comma-separated, or single).
                raw = target_path.strip()
                paths = (
                    json.loads(raw)
                    if raw.startswith("[")
                    else [p.strip() for p in raw.split(",") if p.strip()]
                    if "," in raw
                    else [raw]
                )
                paths = [p.strip() for p in paths if isinstance(p, str) and p.strip()]
                if not paths:
                    return "Error: target_path required for ingest action"

                # ``content_type`` is auto-detected per path and is NOT an
                # agent-facing concern (CONCEPT:KG-2.7 ContentType.classify is the
                # single source of truth). It survives only as an internal override
                # for genuinely ambiguous paths; ``isinstance(str)`` filters out the
                # unresolved FastMCP ``FieldInfo`` default. Whatever the type, heavy
                # categories ALWAYS route through the async durable queue so an
                # ingest call can never block the caller for minutes — the old
                # "explicit content_type → synchronous IngestionEngine" branch was a
                # footgun that did exactly that.
                override = (
                    content_type.strip().lower()
                    if (content_type and isinstance(content_type, str))
                    else ""
                )

                def resolve_ct(p: str) -> ContentType:
                    if override:
                        try:
                            return ContentType(override)
                        except ValueError:
                            pass
                    return ContentType.classify(p)

                # DOCUMENT/CODEBASE are slow (chunk+embed / tree-sitter parse) and
                # are handled by the background task worker → enqueue, never block.
                # The remaining lightweight categories (config/prompt/skill/
                # mcp_server/kb/conversation/policy/…) are fast and are only routed
                # by the unified IngestionEngine, so they run inline.
                async_types = {ContentType.DOCUMENT, ContentType.CODEBASE}
                async_jobs: list[str] = []
                sync_out: list[str] = []
                ing: IngestionEngine | None = None
                for p in paths:
                    ct = resolve_ct(p)
                    if ct in async_types:
                        t_type = (
                            "codebase" if ct == ContentType.CODEBASE else "document"
                        )
                        jid = engine.submit_task(
                            target_path=p,
                            is_codebase=(t_type == "codebase"),
                            provenance={
                                "agent_id": agent_id,
                                "max_depth": max_depth,
                            },
                            task_type=t_type,
                        )
                        async_jobs.append(jid)
                    else:
                        if ing is None:
                            ing = IngestionEngine(kg_engine=engine)
                        r = await ing.ingest(
                            IngestionManifest(
                                content_type=ct,
                                source_uri=p,
                                max_depth=max_depth,
                                metadata={"agent_id": agent_id},
                            )
                        )
                        sync_out.append(
                            f"[{ct.value}] {p}: {r.status} (+{r.nodes_created}n/+{r.edges_created}e"
                            f"{', ' + str(r.details.get('cards_pending')) + ' cards pending' if r.details.get('cards_pending') else ''}"
                            f"{'; ' + r.error if r.error else ''})"
                        )

                msgs: list[str] = []
                if async_jobs:
                    label = (
                        f"Started ingestion job {async_jobs[0]} for {paths[0]}"
                        if len(async_jobs) == 1
                        else f"Submitted {len(async_jobs)} jobs: {', '.join(async_jobs)}"
                    )
                    msgs.append(label)
                if sync_out:
                    msgs.append(" | ".join(sync_out))
                return " ; ".join(msgs) if msgs else "Nothing to ingest."

            elif action == "ingest_url":
                # Content-aware single-URL ingest (CONCEPT:KG-2.7): fetch via the
                # unified resolver (ArchiveBox→crawl4ai→requests) → Document, and —
                # for a research roundup (auto-detected, or forced via
                # description='extract_papers') — download the papers it cites and
                # ingest them too. Runs as a BACKGROUND job (fetch + paper downloads
                # can exceed the call ceiling): returns a job_id; poll with
                # action=job_status. The gateway host daemon's task workers process
                # it through the unified _ingest_document path.
                if not target_path:
                    return "Error: target_path (a URL) required for ingest_url"
                url = target_path.strip()
                prov: dict[str, Any] = {"agent_id": agent_id, "source_url": url}
                flag = (description or "").strip().lower()
                if flag in ("extract_papers", "papers", "extract_papers=true", "true"):
                    prov["extract_papers"] = True
                elif flag in ("no_papers", "extract_papers=false", "false"):
                    prov["extract_papers"] = False
                jid = engine.submit_task(
                    target_path=url,
                    is_codebase=False,
                    provenance=prov,
                    task_type="content_url",
                )
                return (
                    f"Submitted content-aware URL ingest job {jid} for {url} "
                    f"(poll: action=job_status job_id={jid})."
                )

            elif action == "archivebox_sync":
                # Pull preserved ArchiveBox snapshots into the KG (CONCEPT:KG-2.7).
                # corpus_name selects the mode: 'full' = pull ALL, else delta;
                # base_path = JSON list of specific snapshot ids to sync.
                from agent_utilities.knowledge_graph.core.source_sync import (
                    sync_source,
                )

                mode = (corpus_name or "delta").strip().lower()
                ids = None
                if base_path.strip().startswith("["):
                    ids = [str(x) for x in json.loads(base_path)]
                res_d = sync_source(
                    engine,
                    "archivebox",
                    mode="full" if mode == "full" else mode,
                    ids=ids,
                )
                return json.dumps(res_d)

            elif action == "gitlab_sync":
                # Index whole GitLab instance(s) as a resolved code graph (KG-2.9g).
                # corpus_name = mode ('full' = re-index all, else delta);
                # base_path = JSON list of project ids to narrow to.
                from agent_utilities.knowledge_graph.core.source_sync import (
                    sync_source,
                )

                mode = (corpus_name or "delta").strip().lower()
                ids = None
                if base_path.strip().startswith("["):
                    ids = [str(x) for x in json.loads(base_path)]
                res_d = sync_source(
                    engine,
                    "gitlab",
                    mode="full" if mode == "full" else mode,
                    ids=ids,
                )
                return json.dumps(res_d)

            elif action == "gitlab_webhook":
                # Near-real-time incremental re-index from a GitLab push/MR webhook
                # (KG-2.9g): description = the raw webhook JSON payload.
                from agent_utilities.knowledge_graph.core.gitlab_indexer import (
                    handle_gitlab_webhook,
                )

                try:
                    payload = json.loads(description) if description else {}
                except (ValueError, TypeError):
                    return json.dumps(
                        {"status": "ignored", "reason": "invalid payload JSON"}
                    )
                return json.dumps(handle_gitlab_webhook(engine, payload))

            elif action == "corpus":
                if not corpus_name:
                    return "Error: corpus_name required"
                engine.add_node(
                    f"corpus_{corpus_name}",
                    "Corpus",
                    base_path=base_path,
                    description=description,
                )
                return f"Corpus {corpus_name} added/updated."

            elif action == "jobs":
                import json as _json

                from agent_utilities.knowledge_graph.core.engine_tasks import (
                    _decode_metadata,
                )

                jobs = engine.query_cypher(
                    "MATCH (t:Task) RETURN t.id as id, t.status as status, t.metadata as meta LIMIT 20"
                )
                lines = []
                for j in jobs or []:
                    meta = _decode_metadata(j.get("meta"))
                    target = meta.get("target", "unknown")
                    dur = meta.get("duration_ms")
                    dur_s = f" {dur / 1000:.1f}s" if dur else ""
                    lines.append(f"{j['id']}: {j['status']} ({target}){dur_s}")
                # Per-category metrics breakdown (time/nodes/edges/failures) —
                # the harness-style view, pollable over MCP (CONCEPT:KG-2.8).
                breakdown = {}
                if hasattr(engine, "aggregate_ingest_metrics"):
                    try:
                        _b = engine.aggregate_ingest_metrics()
                        breakdown = _b if isinstance(_b, dict) else {}
                    except Exception:  # noqa: BLE001
                        breakdown = {}
                head = (
                    "\n".join(lines) if lines else "No active or recent ingestion jobs."
                )
                return (
                    head
                    + "\n\n=== per-category metrics ===\n"
                    + _json.dumps(breakdown, indent=2)
                    if breakdown
                    else head
                )

            elif action in ("job_status", "status"):
                if not job_id:
                    return "Error: job_id required"
                import json as _json

                from agent_utilities.knowledge_graph.core.engine_tasks import (
                    _decode_metadata,
                )

                jobs = engine.query_cypher(
                    "MATCH (t:Task) WHERE t.id = $job_id RETURN t.status as status, t.metadata as meta",
                    {"job_id": job_id},
                )
                if not jobs:
                    return f"Job {job_id} not found."
                status = jobs[0]["status"]
                meta = _decode_metadata(jobs[0].get("meta"))
                metrics = {
                    k: meta[k]
                    for k in (
                        "type",
                        "content_type",
                        "duration_ms",
                        "nodes_added",
                        "nodes_created",
                        "edges_added",
                        "edges_created",
                        "cards_pending",
                        "error",
                    )
                    if k in meta
                }
                return f"Job {job_id} status: {status}\n" + _json.dumps(
                    metrics, indent=2
                )

            elif action == "cancel":
                import json as _json

                if not job_id:
                    return "Error: job_id required for cancel"
                return _json.dumps(engine.cancel_task(job_id), indent=2)

            elif action == "clear":
                # ``target_path`` carries the status filter:
                # pending|running|completed|failed|cancelled|zombie|all (default
                # 'completed' — the safe default that never drops queued work).
                import json as _json

                tp = target_path if isinstance(target_path, str) else ""
                return _json.dumps(
                    engine.clear_tasks((tp or "completed").strip().lower()), indent=2
                )

            elif action == "prioritize":
                # ``target_path`` carries the level: 'high' (default) | 'normal'.
                import json as _json

                if not job_id:
                    return "Error: job_id required for prioritize"
                tp = target_path if isinstance(target_path, str) else ""
                return _json.dumps(
                    engine.prioritize_task(job_id, (tp or "high").strip().lower()),
                    indent=2,
                )

            elif action == "rebuild_indexes":
                engine.build_indexes()
                return "Indexes rebuilt successfully."

            # ── KG-2.7: Observational Memory Bridge Actions ──
            elif action == "observe":
                try:
                    from pathlib import Path as _Path

                    from agent_utilities.knowledge_graph.memory.observer import (
                        observe_from_file,
                    )

                    if not target_path:
                        return "Error: target_path required (path to JSONL transcript)"
                    result = observe_from_file(
                        engine, _Path(target_path), source=agent_id or "mcp"
                    )
                    return result or "No new observations extracted."
                except Exception as e:
                    return f"Observe error: {e}"

            elif action == "materialize":
                try:
                    from agent_utilities.knowledge_graph.memory import (
                        materialize_memory,
                    )

                    paths = materialize_memory(engine)
                    return json.dumps(
                        {
                            "status": "materialized",
                            "files": {k: str(v) for k, v in paths.items()},
                        }
                    )
                except Exception as e:
                    return f"Materialize error: {e}"

            elif action == "sync":
                try:
                    from agent_utilities.knowledge_graph.memory import (
                        ingest_memory_edits,
                    )

                    results = ingest_memory_edits(engine)
                    return (
                        json.dumps({"status": "synced", "ingested": results})
                        if results
                        else "No edits detected."
                    )
                except Exception as e:
                    return f"Sync error: {e}"

            elif action == "reflect":
                try:
                    from agent_utilities.knowledge_graph.memory import (
                        run_reflector,
                    )

                    result = run_reflector(engine)
                    return result or "No observations to reflect on."
                except Exception as e:
                    return f"Reflect error: {e}"

            elif action == "materialize_source":
                # CONCEPT:KG-2.9 — persist an enterprise source extractor
                # (camunda/aris/egeria/…) INTO the graph, then run one OWL
                # reasoning cycle so the new BusinessProcess/BusinessTask/
                # FLOWS_TO structure folds into the cross-vendor crosswalk
                # natively. corpus_name=category; description=optional JSON
                # extractor config; an in-process vendor client is resolved
                # from the connector package's auth.get_client().
                try:
                    from agent_utilities.knowledge_graph.enrichment.materialize import (
                        run_materialize_source,
                    )

                    category = (corpus_name or "").strip()
                    if not category:
                        return json.dumps(
                            {
                                "error": "materialize_source requires corpus_name "
                                "(the extractor category, e.g. 'camunda' or 'aris')"
                            }
                        )
                    extractor_config = (
                        json.loads(description)
                        if description and description.strip().startswith("{")
                        else None
                    )
                    # Shared core — same path the unified ``source_sync`` uses.
                    return json.dumps(
                        run_materialize_source(
                            engine, category, config=extractor_config
                        ),
                        default=str,
                    )
                except Exception as e:
                    return f"Materialize source error: {e}"

            elif action == "skill_workflows":
                # CONCEPT:KG-2.97 — ingest the universal-skills workflow corpus
                # (workflows/<domain>/<name>/SKILL.md) as dispatchable
                # WorkflowDefinition DAGs so kg-delegation-router /
                # execute_workflow can discover & fire them. ``target_path`` is
                # an optional explicit corpus root (a dir that is/contains
                # ``workflows/``); default = installed universal_skills package.
                #
                # Durable per-node writes for the full corpus (~315 workflows)
                # take ~150s — over the MCP call ceiling — and the backend can't
                # bulk-write durably here, so this enqueues a BACKGROUND job (run
                # by the task worker, off the request path) and returns its id;
                # poll with ``action=job_status job_id=<id>``.
                try:
                    root = target_path if isinstance(target_path, str) else ""
                    jid = engine.submit_task(
                        target_path=root or "universal-skills",
                        is_codebase=False,
                        provenance={"agent_id": agent_id},
                        task_type="skill_workflows",
                    )
                    return json.dumps(
                        {
                            "job_id": jid,
                            "status": "submitted",
                            "message": (
                                f"Skill-workflow ingest enqueued as background job "
                                f"{jid}; poll with graph_ingest action=job_status "
                                f"job_id={jid}."
                            ),
                        }
                    )
                except Exception as e:
                    return f"Skill-workflow ingest error: {e}"

            elif action == "curate_wiki":
                # CONCEPT:KG-2.19 — delta-skip continuous ingest of a self-curating wiki dir.
                try:
                    from agent_utilities.knowledge_graph.ingestion.wiki_curator import (
                        curate_wiki,
                    )

                    if not target_path:
                        return json.dumps(
                            {"error": "curate_wiki requires target_path (the wiki dir)"}
                        )
                    summary = curate_wiki(engine, target_path)
                    return json.dumps(summary, default=str)
                except Exception as e:
                    return f"Wiki curation error: {e}"

            elif action == "distill":
                # CONCEPT:AHE-3.9 — Distill a coherent KG subgraph OUT into a
                # portable skill-graph: a reference/ markdown tree + a
                # kg_manifest.json provenance record (round-trippable via the
                # 'ingest_knowledge_pack' action). The output dir is consumable
                # verbatim by skill-graph-builder as a local-directory source.
                # Param overloads (mirroring agent_toolkit's reuse of fields):
                #   target_path  -> output directory (required)
                #   corpus_name  -> seed node id      (anchor by id)
                #   description  -> natural-language query (semantic anchor)
                #   max_depth    -> BFS hop depth
                try:
                    from agent_utilities.knowledge_graph.distillation import (
                        SkillGraphDistiller,
                    )

                    if not target_path:
                        return json.dumps(
                            {"error": "distill requires target_path (output dir)"}
                        )
                    seed = corpus_name or None
                    query = description or None
                    if not (seed or query):
                        return json.dumps(
                            {
                                "error": "distill requires a seed (corpus_name=node_id) "
                                "or query (description=text)"
                            }
                        )
                    # content_type="workflow" → distill a graph-native skill-WORKFLOW
                    # (procedure step-DAG) instead of a documentation skill-graph.
                    as_workflow = (content_type or "").strip().lower() == "workflow"
                    distiller = await SkillGraphDistiller.connect()
                    try:
                        if as_workflow:
                            wf = await distiller.distill_workflow(
                                seed=seed,
                                query=query,
                                depth=max_depth,
                                out_dir=target_path,
                            )
                            payload = {
                                "kind": "skill-workflow",
                                "name": wf["name"],
                                "steps": wf["steps"],
                            }
                        else:
                            manifest = await distiller.distill(
                                seed=seed,
                                query=query,
                                depth=max_depth,
                                out_dir=target_path,
                            )
                            payload = {
                                "kind": "skill-graph",
                                "stats": manifest["stats"],
                            }
                    finally:
                        await distiller.close()
                    return json.dumps(
                        {
                            "status": "distilled",
                            "out_dir": target_path,
                            "manifest": f"{target_path.rstrip('/')}/kg_manifest.json",
                            **payload,
                        },
                        default=str,
                    )
                except Exception as e:
                    return f"Distill error: {e}"

            elif action in (
                "build_skill_graph",
                "skill_graph_status",
                "rebuild_skill_graph",
            ):
                # CONCEPT:KG-2.7 — the unified skill-graph pipeline: acquire from any
                # source kind (web/pdf/office/dir/url_reader/rest/database/mcp_tool/
                # generated/kg_query) into a standardized skill-graph with a
                # sources.json provenance/freshness manifest, hybrid-auto KG ingest,
                # and a staleness/rebuild loop. Heavy/blocking work runs off the event
                # loop via a worker thread.
                import asyncio

                from agent_utilities.knowledge_graph.distillation import (
                    SkillGraphPipeline,
                    SourceSpec,
                )

                pipe = SkillGraphPipeline()
                if action == "build_skill_graph":
                    if not (corpus_name and target_path):
                        return json.dumps(
                            {
                                "error": "build_skill_graph requires corpus_name (name) "
                                "and target_path (output parent dir); base_path = JSON "
                                "list of sources or 'kind=uri,kind=uri' shorthand."
                            }
                        )
                    try:
                        specs = _parse_source_specs(base_path, SourceSpec)
                    except ValueError as exc:
                        return json.dumps({"error": str(exc)})
                    if not specs:
                        return json.dumps({"error": "no sources provided in base_path"})
                    sg_built = await asyncio.to_thread(
                        lambda: pipe.build(
                            name=corpus_name,
                            specs=specs,
                            out_dir=target_path,
                            description=description or None,
                        )
                    )
                    return json.dumps(sg_built, default=str)
                if action == "skill_graph_status":
                    if not target_path:
                        return json.dumps(
                            {"error": "skill_graph_status requires target_path (dir)"}
                        )
                    quick = corpus_name.strip().lower() == "quick"
                    sg_report = await asyncio.to_thread(
                        lambda: pipe.status(target_path, quick=quick)
                    )
                    return json.dumps(sg_report, default=str)
                # rebuild_skill_graph
                if not target_path:
                    return json.dumps(
                        {"error": "rebuild_skill_graph requires target_path (dir)"}
                    )
                sg_rebuilt = await asyncio.to_thread(lambda: pipe.rebuild(target_path))
                return json.dumps(sg_rebuilt, default=str)

            elif action == "agent_toolkit":
                sources = (
                    json.loads(target_path)
                    if target_path.startswith("[")
                    else [target_path]
                )
                # Use `description` param as optional agent_card_path override
                agent_card_path = (
                    description if description else "/.well-known/agent.json"
                )
                result = await engine.ingest_agent_toolkit(
                    sources, agent_card_path=agent_card_path
                )
                return json.dumps(result, default=str)

            elif action == "ingest_knowledge_pack":
                from pathlib import Path

                import yaml

                from agent_utilities.models.knowledge_pack import (
                    KnowledgePackBundle,
                    KnowledgePackHydrator,
                    KnowledgePackImporter,
                )

                if not target_path:
                    return "Error: target_path required for ingest_knowledge_pack"

                path = Path(target_path)
                if not path.exists() or not path.is_file():
                    return f"Error: knowledge pack file not found at {target_path}"

                with open(path, encoding="utf-8") as f:
                    if path.suffix in [".yaml", ".yml"]:
                        data = yaml.safe_load(f)
                    else:
                        data = json.load(f)

                bundle = KnowledgePackBundle.from_dict(data)
                await KnowledgePackHydrator.hydrate(bundle)
                KnowledgePackImporter.seed_into_kg(bundle, engine)
                return f"Knowledge pack from {target_path} hydrated and ingested."

            elif action == "import_pack":
                # CONCEPT:AHE-3.9 — Round-trip import of a distilled skill-graph
                # package (reference/ + kg_manifest.json): reconstruct the original
                # subgraph here, preserving node ids + edges. The inverse of
                # 'distill'. ``corpus_name="dedup"`` runs the IdeaBlock dedup-merge.

                from agent_utilities.knowledge_graph.distillation import (
                    import_skill_graph_pack,
                )

                if not target_path:
                    return json.dumps(
                        {"error": "import_pack requires target_path (skill-graph dir)"}
                    )
                try:
                    stats = import_skill_graph_pack(
                        engine, target_path, dedup=(corpus_name == "dedup")
                    )
                    return json.dumps(
                        {"status": "imported", "stats": stats}, default=str
                    )
                except Exception as e:  # noqa: BLE001
                    return f"Import error: {e}"

            elif action == "fact_extract":
                # CONCEPT:KG-2.64 — document → atomic-triple fact extraction.
                # Streams (subject)-[predicate]->(object) edges carrying
                # confidence/evidence_span/tags, dedups them semantically with
                # our own embedder, persists them as graph edges (variant node
                # names merged), and returns the facts + JSONL (upstream parity).
                # Text source: ``description`` (raw text) or ``target_path``
                # (local file, else treated as raw text). Single round + dedup
                # (multi-round recall is opt-in over the REST surface).
                from pathlib import Path

                from agent_utilities.knowledge_graph.extraction import (
                    ExtractedFact,
                    extract_facts,
                    facts_to_jsonl,
                    persist_facts,
                )
                from agent_utilities.knowledge_graph.extraction.job_manager import (
                    EngineStoreAdapter,
                )

                text = description or ""
                source_file = ""
                if not text and target_path:
                    p = Path(target_path)
                    if p.exists() and p.is_file():
                        text = p.read_text(encoding="utf-8", errors="ignore")
                        source_file = target_path
                    else:
                        text = target_path
                if not text.strip():
                    return json.dumps(
                        {
                            "error": "fact_extract requires text (description=) "
                            "or a readable file (target_path=)"
                        }
                    )

                facts: list[ExtractedFact] = []
                async for ev in extract_facts(text, rounds=1, source_file=source_file):
                    if ev["type"] == "fact":
                        facts.append(ExtractedFact(**ev["fact"]))

                stats = persist_facts(EngineStoreAdapter(engine), facts)
                unique = sum(1 for f in facts if not f.is_duplicate)
                return json.dumps(
                    {
                        "status": "extracted",
                        "facts": [f.model_dump() for f in facts],
                        "jsonl": facts_to_jsonl(facts),
                        "stats": {
                            **stats,
                            "total_facts": len(facts),
                            "unique_facts": unique,
                            "duplicate_facts": len(facts) - unique,
                        },
                    },
                    default=str,
                )

            elif action in (
                "extract_submit",
                "extract_jobs",
                "extract_status",
                "extract_pause",
                "extract_resume",
                "extract_jsonl",
            ):
                # CONCEPT:KG-2.65 — GPU-slot-scheduled fact extraction. Unlike the
                # inline 'fact_extract', these submit a job that runs on the single
                # GPU inference slot with preempt/backfill/resume, so concurrent
                # submissions don't oversubscribe the GPU. job_id addresses a job.

                mgr = kg_server._get_extraction_manager(engine)

                if action == "extract_submit":
                    text = description or ""
                    if not text and target_path:
                        from pathlib import Path

                        p = Path(target_path)
                        text = (
                            p.read_text(encoding="utf-8", errors="ignore")
                            if p.exists() and p.is_file()
                            else target_path
                        )
                    if not text.strip():
                        return json.dumps(
                            {
                                "error": "extract_submit requires description= or target_path="
                            }
                        )
                    jid = await mgr.submit(
                        text=text, rounds=max(1, min(10, max_depth or 1))
                    )
                    return json.dumps({"status": "submitted", "job_id": jid})

                if action == "extract_jobs":
                    return json.dumps({"jobs": mgr.jobs()}, default=str)

                if not job_id:
                    return json.dumps({"error": f"{action} requires job_id"})

                if action == "extract_status":
                    return json.dumps(
                        mgr.status(job_id) or {"error": "no such job"}, default=str
                    )
                if action == "extract_jsonl":
                    return mgr.jsonl(job_id)
                if action == "extract_pause":
                    await mgr.pause(job_id)
                    return json.dumps({"status": "paused", "job_id": job_id})
                # extract_resume
                await mgr.resume(job_id)
                return json.dumps({"status": "resumed", "job_id": job_id})

            else:
                return f"Error: Unknown ingest action '{action}'"
        except Exception as e:
            return f"Ingest error: {str(e)}"

    kg_server.REGISTERED_TOOLS["graph_ingest"] = graph_ingest

    @mcp.tool(
        name="usage_query",
        description=(
            "Query usage/cost/observability analytics (CONCEPT:ECO-4.41): token "
            "counts, cost, model/tool/skill/db-call usage, session browser, "
            "activity heatmap, full-text search, and Langfuse trace links. One "
            "store covers both ingested agent logs and our own runtime telemetry."
        ),
        tags=["graph-os", "observability", "usage"],
    )
    async def usage_query(
        action: str = Field(
            default="summary",
            description=(
                "summary | by_model | by_project | by_agent | tools | activity | "
                "sessions | session_detail | top_sessions | search | traces"
            ),
        ),
        from_date: str = Field(default="", description="ISO start (started_at >=)."),
        to_date: str = Field(default="", description="ISO end (started_at <=)."),
        project: str = Field(default="", description="Filter by project."),
        agent: str = Field(default="", description="Filter by agent type."),
        model: str = Field(default="", description="Filter by model."),
        origin: str = Field(
            default="", description="ingested | runtime (omit for both)."
        ),
        tenant_id: str = Field(default="", description="Tenant scope."),
        session_id: str = Field(default="", description="For action=session_detail."),
        query: str = Field(default="", description="For action=search (FTS)."),
        limit: int = Field(default=50, description="Row cap for list actions."),
    ) -> str:
        """Read-side analytics over the usage store. Returns JSON."""
        import json as _json

        from agent_utilities.usage.service import get_usage_service

        svc = get_usage_service()
        f = {
            k: v
            for k, v in {
                "from_date": from_date,
                "to_date": to_date,
                "project": project,
                "agent": agent,
                "model": model,
                "origin": origin,
                "tenant_id": tenant_id,
            }.items()
            if v
        }
        try:
            if action == "summary":
                out: Any = svc.summary(**f).model_dump()
            elif action == "by_model":
                out = [e.model_dump() for e in svc.by_model(**f)]
            elif action == "by_project":
                out = [e.model_dump() for e in svc.by_project(**f)]
            elif action == "by_agent":
                out = [e.model_dump() for e in svc.by_agent(**f)]
            elif action == "tools":
                out = [e.model_dump() for e in svc.tools(**f)]
            elif action == "activity":
                out = [e.model_dump() for e in svc.activity(**f)]
            elif action == "sessions":
                out = [e.model_dump() for e in svc.sessions(limit=limit, **f)]
            elif action == "top_sessions":
                out = [e.model_dump() for e in svc.top_sessions(limit=limit, **f)]
            elif action == "session_detail":
                if not session_id:
                    return "Error: session_id required for session_detail"
                detail = svc.session_detail(session_id)
                out = detail.model_dump() if detail else None
            elif action == "search":
                if not query:
                    return "Error: query required for search"
                out = [e.model_dump() for e in svc.search(query, limit=limit)]
            else:
                return f"Error: unknown usage_query action '{action}'"
            return _json.dumps(out, default=str)
        except Exception as e:  # noqa: BLE001
            return f"usage_query error: {e}"

    kg_server.REGISTERED_TOOLS["usage_query"] = usage_query

    @mcp.tool(
        name="ingest_sessions",
        description=(
            "Ingest AI agent chat/session history into the usage store + KG "
            "(CONCEPT:ECO-4.42). 'collect' auto-detects installed agents on THIS "
            "host and parses their local logs (use when the engine is local). "
            "'upload' accepts pre-parsed session bundles as JSON so a CLIENT can "
            "parse its own logs and push them to a REMOTE/central engine that has "
            "no filesystem access to the client — closing the remote-ingest gap. "
            "'paths' ingests explicit files/dirs."
        ),
        tags=["graph-os", "ingest", "observability"],
    )
    async def ingest_sessions(
        action: str = Field(default="collect", description="collect | upload | paths"),
        bundles_json: str = Field(
            default="",
            description="For action=upload: JSON array of ParsedSessionBundle objects.",
        ),
        target_path: str = Field(
            default="", description="For action=paths: JSON list or comma paths."
        ),
        tenant_id: str = Field(default="", description="Tenant scope for the rows."),
    ) -> str:
        """Client-parses, server-sinks ingestion of agent session logs."""
        import json as _json

        try:
            if action == "collect":
                from agent_utilities.ingestion.collector import collect_local_sessions

                return _json.dumps(collect_local_sessions(), default=str)
            if action == "upload":
                from agent_utilities.usage.models import ParsedSessionBundle
                from agent_utilities.usage.recorder import get_usage_recorder

                raw = _json.loads(bundles_json) if bundles_json else []
                recorder = get_usage_recorder()
                ok = 0
                for item in raw:
                    bundle = ParsedSessionBundle.model_validate(item)
                    if tenant_id:
                        bundle.session.tenant_id = tenant_id
                    if recorder.record_bundle(bundle):
                        ok += 1
                return _json.dumps({"received": len(raw), "ingested": ok})
            if action == "paths":
                from agent_utilities.ingestion.collector import collect_paths

                raw = target_path.strip()
                paths = (
                    _json.loads(raw)
                    if raw.startswith("[")
                    else [p.strip() for p in raw.split(",") if p.strip()]
                )
                return _json.dumps(collect_paths(paths), default=str)
            return f"Error: unknown ingest_sessions action '{action}'"
        except Exception as e:  # noqa: BLE001
            return f"ingest_sessions error: {e}"

    kg_server.REGISTERED_TOOLS["ingest_sessions"] = ingest_sessions
