#!/usr/bin/python
"""Unified Intelligence Graph CLI.

CONCEPT:AU-KG.query.object-graph-mapper

Command-line interface for running the Intelligence Pipeline
and querying the graph (Agents, Tools, Code, Memory).
"""

import argparse
import asyncio
import logging
from pathlib import Path

from agent_utilities.core.paths import kg_db_path
from agent_utilities.models.knowledge_graph import PipelineConfig

from .core.engine import IntelligenceGraphEngine
from .pipeline import IntelligencePipeline


async def main():
    parser = argparse.ArgumentParser(description="Unified Intelligence Graph CLI")
    parser.add_argument(
        "--maintain", action="store_true", help="Run the full intelligence pipeline"
    )
    parser.add_argument(
        "--bootstrap-workspace",
        nargs="?",
        const="DEFAULT",
        help="Parse XDG workspace.yml (or custom path), clone missing projects, and ingest all projects natively.",
    )
    parser.add_argument(
        "--status", action="store_true", help="Show current graph metrics"
    )
    parser.add_argument(
        "--search",
        type=str,
        help="Hybrid search across agents, tools, code, and memories",
    )
    parser.add_argument(
        "--impact", type=str, help="Analyze topological impact of a symbol or file"
    )
    parser.add_argument("--memory", type=str, help="Search memories by keyword")
    parser.add_argument("--add-memory", type=str, help="Add a new memory entry")
    parser.add_argument("--get-memory", type=str, help="Retrieve a memory by ID")
    parser.add_argument("--delete-memory", type=str, help="Delete a memory by ID")
    parser.add_argument(
        "--update-memory",
        type=str,
        help="Update a memory (requires --id and --content)",
    )
    parser.add_argument("--id", type=str, help="Memory ID for update")
    parser.add_argument("--content", type=str, help="New content for memory update")

    parser.add_argument(
        "--stage-to-queue",
        type=str,
        help="Job ID to serialize and stage the graph instead of persisting directly",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    config = PipelineConfig(
        workspace_path=str(Path.cwd()),
        ladybug_path=str(kg_db_path()),
    )

    if args.stage_to_queue:
        config.persist_to_ladybug = False
        # Isolate this bulk-ingest subprocess's scratch symbol graph in its own
        # tenant so concurrent repo ingests don't saturate the shared "__commons__"
        # tenant (which the main process uses for the task queue + graph-writer).
        # Only the final staged payload is merged into "__commons__" downstream.
        pipeline = IntelligencePipeline(
            config, graph_name=f"stage_{args.stage_to_queue}"
        )
    else:
        pipeline = IntelligencePipeline(config)

    if args.bootstrap_workspace:
        from agent_utilities.core.workspace_config import clone_missing_projects

        print("Bootstrapping workspace...")
        if args.bootstrap_workspace == "DEFAULT":
            project_paths = clone_missing_projects()
        else:
            project_paths = clone_missing_projects(yml_path=args.bootstrap_workspace)

        if not project_paths:
            print("No projects found in workspace.yml to bootstrap.")
            return

        print(
            f"Discovered {len(project_paths)} projects. Running IntelligencePipeline on each..."
        )
        total_nodes = 0
        total_edges = 0
        for path in project_paths:
            print(f"Ingesting {path}...")
            if not path.exists() or not path.is_dir():
                print(f"  Skipping {path} (not a valid directory)")
                continue

            try:
                proj_config = PipelineConfig(
                    workspace_path=str(path),
                    ladybug_path=str(kg_db_path()),
                )
                proj_pipeline = IntelligencePipeline(proj_config)
                metadata = await proj_pipeline.run()
                total_nodes += metadata.node_count
                total_edges += metadata.edge_count
                print(
                    f"  Success: {metadata.node_count} nodes, {metadata.edge_count} edges."
                )
            except Exception as e:
                print(f"  Failed to ingest {path}: {e}")

        print(
            f"Workspace Bootstrap Complete! Ingested {total_nodes} nodes and {total_edges} edges across {len(project_paths)} projects."
        )

    elif args.maintain:
        metadata = await pipeline.run()
        print(
            f"Intelligence Graph Updated: {metadata.node_count} nodes, {metadata.edge_count} edges."
        )
        if args.stage_to_queue:
            from agent_utilities.core.config import config as app_config
            from agent_utilities.core.paths import data_dir
            from agent_utilities.knowledge_graph.core.queue_backend import (
                create_task_queue,
            )

            # Same selection contract as the engine (CONCEPT:AU-KG.backend.selectable-queue-backend):
            # TASK_QUEUE_BACKEND, fail-loud when explicit, auto otherwise.
            queue_db_path = data_dir() / "kg_task_queue.db"
            queue, _ = create_task_queue(app_config, str(queue_db_path))

            nodes = []
            for nid, data in pipeline.graph.nodes(data=True):
                node = data.copy()
                node["id"] = nid
                nodes.append(node)

            edges = []
            for u, v, data in pipeline.graph.edges(data=True):
                edge = data.copy()
                edge["source"] = u
                edge["target"] = v
                edges.append(edge)

            queue.put_staged_graph(args.stage_to_queue, nodes, edges)
            print(f"Graph staged to queue for job {args.stage_to_queue}")

            # Drop the ephemeral scratch tenant — its payload now lives in the
            # staging queue and will be merged into "__commons__" by the graph-writer.
            try:
                graph_name = getattr(pipeline, "graph_name", None)
                client = getattr(pipeline.graph, "_client", None)
                if graph_name and graph_name != "__commons__" and client is not None:
                    client.tenants.delete(graph_name)
            except Exception as e:
                logging.getLogger(__name__).debug(
                    f"Scratch tenant cleanup skipped: {e}"
                )

    elif args.status:
        await pipeline.run()
        print(f"Nodes: {pipeline.metadata.node_count}")
        print(f"Edges: {pipeline.metadata.edge_count}")
        print(f"Agents: {pipeline.metadata.agent_count}")
        print(f"Tools: {pipeline.metadata.tool_count}")

    elif args.search:
        await pipeline.run()
        engine = IntelligenceGraphEngine(graph=pipeline.graph)
        results = engine.search_hybrid(args.search)
        if results:
            for r in results:
                print(
                    f"[{r.get('type', 'node').upper()}] [ID: {r.get('id')}] {r.get('name')}: {r.get('description', '')[:100]}..."
                )
        else:
            print(f"No results found for '{args.search}'.")

    elif args.impact:
        await pipeline.run()
        engine = IntelligenceGraphEngine(graph=pipeline.graph)
        impact = engine.query_impact(args.impact)
        if impact:
            print(f"Impact Set for '{args.impact}':")
            for node in impact:
                print(
                    f"- [{node.get('type')}] {node.get('id')} (File: {node.get('file_path', 'N/A')})"
                )
        else:
            print(f"No impact found for '{args.impact}'.")

    elif args.memory:
        await pipeline.run()
        engine = IntelligenceGraphEngine(graph=pipeline.graph)
        memories = engine.search_hybrid(args.memory)
        # Filter for memory nodes
        memories = [m for m in memories if m.get("type") == "memory"]
        if memories:
            for m in memories:
                print(
                    f"[{m.get('timestamp', 'N/A')}] {m.get('name')}: {m.get('description', '')[:100]}..."
                )
        else:
            print(f"No memories found matching '{args.memory}'.")

    elif args.add_memory:
        await pipeline.run()
        engine = IntelligenceGraphEngine(
            graph=pipeline.graph, db_path=config.ladybug_path
        )
        mem_id = engine.add_memory(args.add_memory)
        print(f"Memory added with ID: {mem_id}")

    elif args.get_memory:
        await pipeline.run()
        engine = IntelligenceGraphEngine(graph=pipeline.graph)
        memory = engine.get_memory(args.get_memory)
        if memory:
            print(f"ID: {memory['id']}")
            print(f"Name: {memory['name']}")
            print(f"Category: {memory['category']}")
            print(f"Content: {memory['description']}")
        else:
            print(f"Memory '{args.get_memory}' not found.")

    elif args.delete_memory:
        await pipeline.run()
        engine = IntelligenceGraphEngine(
            graph=pipeline.graph, db_path=config.ladybug_path
        )
        engine.delete_memory(args.delete_memory)
        print(f"Memory '{args.delete_memory}' deleted.")

    elif args.update_memory:
        if not args.id or not args.content:
            print("Error: --update-memory requires --id and --content.")
            return
        await pipeline.run()
        engine = IntelligenceGraphEngine(
            graph=pipeline.graph, db_path=config.ladybug_path
        )
        engine.update_memory(args.id, description=args.content)
        print(f"Memory '{args.id}' updated.")

    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
