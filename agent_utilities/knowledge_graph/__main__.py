#!/usr/bin/python
# coding: utf-8
"""Unified Intelligence Graph CLI.

Command-line interface for running the Unified Intelligence Pipeline
and querying the graph (Agents, Tools, Code, Memory).
"""

import asyncio
import argparse
import logging
from pathlib import Path

from ..models.knowledge_graph import PipelineConfig
from .pipeline import IntelligencePipeline
from .engine import IntelligenceGraphEngine
from ..workspace import get_agent_workspace


async def main():
    parser = argparse.ArgumentParser(description="Unified Intelligence Graph CLI")
    parser.add_argument(
        "--maintain", action="store_true", help="Run the full intelligence pipeline"
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

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    agent_ws = get_agent_workspace()
    config = PipelineConfig(
        workspace_path=str(Path.cwd()),
        ladybug_path=str(agent_ws / "knowledge_graph.db"),
    )

    pipeline = IntelligencePipeline(config)

    if args.maintain:
        metadata = await pipeline.run()
        print(
            f"Intelligence Graph Updated: {metadata.node_count} nodes, {metadata.edge_count} edges."
        )

    elif args.status:
        await pipeline.run()
        print(f"Nodes: {pipeline.metadata.node_count}")
        print(f"Edges: {pipeline.metadata.edge_count}")
        print(f"Agents: {pipeline.metadata.agent_count}")
        print(f"Tools: {pipeline.metadata.tool_count}")

    elif args.search:
        await pipeline.run()
        engine = IntelligenceGraphEngine(pipeline.graph)
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
        engine = IntelligenceGraphEngine(pipeline.graph)
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
        engine = IntelligenceGraphEngine(pipeline.graph)
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
        engine = IntelligenceGraphEngine(pipeline.graph, db_path=config.ladybug_path)
        mem_id = engine.add_memory(args.add_memory)
        print(f"Memory added with ID: {mem_id}")

    elif args.get_memory:
        await pipeline.run()
        engine = IntelligenceGraphEngine(pipeline.graph)
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
        engine = IntelligenceGraphEngine(pipeline.graph, db_path=config.ladybug_path)
        engine.delete_memory(args.delete_memory)
        print(f"Memory '{args.delete_memory}' deleted.")

    elif args.update_memory:
        if not args.id or not args.content:
            print("Error: --update-memory requires --id and --content.")
            return
        await pipeline.run()
        engine = IntelligenceGraphEngine(pipeline.graph, db_path=config.ladybug_path)
        engine.update_memory(args.id, description=args.content)
        print(f"Memory '{args.id}' updated.")

    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
