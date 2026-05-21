#!/usr/bin/python
from __future__ import annotations

"""CLI for the Cross-Agent Observational Memory Bridge.

CONCEPT:KG-2.1 -- Observational Memory Bridge
CONCEPT:ECO-4.0 -- Cross-Agent Memory Hook Installer

Provides subcommands for memory materialization, startup context generation,
transcript observation, reflection, hook installation, and diagnostics.

Usage:
    agent-utilities-memory context --for codex --cwd /home/user/project
    agent-utilities-memory recall --query "database decision"
    agent-utilities-memory observe --source claude
    agent-utilities-memory reflect
    agent-utilities-memory install --agents claude,codex,grok
    agent-utilities-memory doctor
    agent-utilities-memory export --format json
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _get_engine():
    """Lazily initialize the IntelligenceGraphEngine singleton."""
    import networkx as nx

    from agent_utilities.core.paths import ensure_dirs, kg_db_path
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    engine = IntelligenceGraphEngine.get_active()
    if engine is not None:
        return engine

    ensure_dirs()
    db_path = str(kg_db_path())
    try:
        from agent_utilities.knowledge_graph.backends import create_backend

        backend = create_backend(backend_type="ladybug", db_path=db_path)
    except Exception:
        backend = None

    graph = nx.MultiDiGraph()
    return IntelligenceGraphEngine(graph=graph, backend=backend)


def cmd_context(args: argparse.Namespace) -> None:
    """Generate startup context payload for an agent."""
    from agent_utilities.knowledge_graph.memory.startup_context import (
        build_startup_payload,
    )

    engine = _get_engine()
    payload = build_startup_payload(
        engine,
        budget_chars=args.budget_chars,
        agent=args.agent,
        cwd=args.cwd or os.getcwd(),
        task=args.task,
    )
    print(payload.text)


def cmd_recall(args: argparse.Namespace) -> None:
    """Search KG memory or expand a startup handle."""
    from agent_utilities.knowledge_graph.memory.startup_context import (
        StartupContextBuilder,
    )

    engine = _get_engine()
    builder = StartupContextBuilder(engine)

    if args.handle:
        try:
            print(builder.recall_handle(args.handle))
        except KeyError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.query:
        results = builder.recall_query(args.query, limit=args.limit)
        if results:
            print("\n".join(results))
        else:
            print("No results found.", file=sys.stderr)
    else:
        print("Error: --query or --handle required", file=sys.stderr)
        sys.exit(1)


def cmd_observe(args: argparse.Namespace) -> None:
    """Process transcripts into observations."""
    from agent_utilities.knowledge_graph.memory.observer import observe_from_file

    engine = _get_engine()

    if args.file:
        path = Path(args.file)
        result = observe_from_file(
            engine, path, source=args.source, dry_run=args.dry_run
        )
        if result:
            print(result)
        else:
            print("No new observations extracted.", file=sys.stderr)
    else:
        print("Error: --file required for observe", file=sys.stderr)
        sys.exit(1)


def cmd_reflect(args: argparse.Namespace) -> None:
    """Run the reflection cycle to condense observations."""
    from agent_utilities.knowledge_graph.memory.reflector import run_reflector

    engine = _get_engine()
    result = run_reflector(engine, dry_run=args.dry_run)
    if result:
        print(result)
    else:
        print("No observations to reflect on.", file=sys.stderr)


def cmd_materialize(args: argparse.Namespace) -> None:
    """Materialize KG memory state into Markdown files."""
    from agent_utilities.knowledge_graph.memory.memory_materializer import (
        materialize_memory,
    )

    engine = _get_engine()
    paths = materialize_memory(engine)
    for name, path in paths.items():
        print(f"  {name}: {path}")
    print(f"\nMaterialized {len(paths)} files.")


def cmd_sync(args: argparse.Namespace) -> None:
    """Bidirectional sync: detect edits and ingest back to KG."""
    from agent_utilities.knowledge_graph.memory.memory_materializer import (
        MemoryMaterializer,
        ingest_memory_edits,
    )

    engine = _get_engine()
    materializer = MemoryMaterializer(engine)

    edited = materializer.detect_edits()
    if not edited:
        print("No manual edits detected.")
        return

    print(f"Detected edits in: {', '.join(edited)}")
    results = ingest_memory_edits(engine)
    for name, count in results.items():
        print(f"  {name}: {count} items ingested")


def cmd_install(args: argparse.Namespace) -> None:
    """Install memory hooks into external agents."""
    from agent_utilities.ecosystem.hook_installer import HookInstaller

    installer = HookInstaller()
    agents = args.agents.split(",") if args.agents else None
    results = installer.install(agents)

    for agent, status in results.items():
        emoji = (
            "\u2705"
            if status == "installed"
            else ("\U0001f504" if status == "integrated" else "\u274c")
        )
        print(f"  {emoji} {agent}: {status}")

    print(
        f"\nInstalled: {len(installer.installed)}, "
        f"Skipped: {len(installer.skipped)}, "
        f"Errors: {len(installer.errors)}"
    )


def cmd_uninstall(args: argparse.Namespace) -> None:
    """Remove memory hooks from external agents."""
    from agent_utilities.ecosystem.hook_installer import HookInstaller

    installer = HookInstaller()
    agents = args.agents.split(",") if args.agents else None
    results = installer.uninstall(agents)
    for agent, status in results.items():
        print(f"  {agent}: {status}")


def cmd_doctor(args: argparse.Namespace) -> None:
    """Verify all hook installations and memory health."""
    from agent_utilities.ecosystem.hook_installer import HookInstaller
    from agent_utilities.knowledge_graph.memory.memory_materializer import (
        memory_dir,
    )

    print("=== Hook Installation Status ===\n")
    installer = HookInstaller()
    report = installer.doctor()
    for agent, info in report.items():
        status = info.get("status", "unknown")
        emoji = {
            "healthy": "\u2705",
            "not_installed": "\u26aa",
            "integrated": "\U0001f504",
            "stale": "\u26a0\ufe0f",
        }.get(status, "\u2753")
        path = info.get("path", "N/A")
        print(f"  {emoji} {info['name']:20s} {status:15s} {path}")

    print("\n=== Memory Directory ===\n")
    mem_dir = memory_dir()
    print(f"  Path: {mem_dir}")
    if mem_dir.exists():
        for f in sorted(mem_dir.iterdir()):
            if not f.name.startswith("."):
                print(f"  \u2705 {f.name} ({f.stat().st_size:,} bytes)")
    else:
        print("  \u274c Directory does not exist (run `materialize` first)")


def cmd_export(args: argparse.Namespace) -> None:
    """Export materialized memory in various formats."""
    from agent_utilities.knowledge_graph.memory.memory_materializer import memory_dir

    mem_dir = memory_dir()
    if not mem_dir.exists():
        print(
            "Error: No materialized memory found. Run `materialize` first.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.format == "json":
        data = {}
        for f in sorted(mem_dir.iterdir()):
            if f.suffix == ".md":
                data[f.name] = f.read_text(encoding="utf-8")
        print(json.dumps(data, indent=2))
    elif args.format == "markdown":
        for f in sorted(mem_dir.iterdir()):
            if f.suffix == ".md":
                print(f"# === {f.name} ===\n")
                print(f.read_text(encoding="utf-8"))
                print("\n---\n")
    else:
        print(f"Error: Unknown format '{args.format}'", file=sys.stderr)
        sys.exit(1)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="agent-utilities-memory",
        description="Cross-Agent Observational Memory Bridge (KG-2.10)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # context
    p = sub.add_parser("context", help="Generate startup context for an agent")
    p.add_argument(
        "--for", dest="agent", default="", help="Agent name (claude, codex, grok)"
    )
    p.add_argument("--cwd", default="", help="Current working directory")
    p.add_argument("--task", default="", help="Current task description")
    p.add_argument("--budget-chars", type=int, default=24000, help="Max payload chars")

    # recall
    p = sub.add_parser("recall", help="Search KG memory or expand a handle")
    p.add_argument("--query", default="", help="Natural language search query")
    p.add_argument("--handle", default="", help="Startup handle to expand")
    p.add_argument("--limit", type=int, default=8, help="Max results")

    # observe
    p = sub.add_parser("observe", help="Process transcripts into observations")
    p.add_argument("--source", default="unknown", help="Source agent name")
    p.add_argument("--file", default="", help="Path to JSONL transcript file")
    p.add_argument("--dry-run", action="store_true", help="Print without persisting")

    # reflect
    p = sub.add_parser("reflect", help="Run reflection cycle")
    p.add_argument("--dry-run", action="store_true", help="Print without persisting")

    # materialize
    sub.add_parser("materialize", help="Render KG memory to Markdown files")

    # sync
    sub.add_parser("sync", help="Detect edits and ingest back to KG")

    # install
    p = sub.add_parser("install", help="Install hooks into external agents")
    p.add_argument(
        "--agents", default="", help="Comma-separated agent names (empty=all)"
    )

    # uninstall
    p = sub.add_parser("uninstall", help="Remove hooks from external agents")
    p.add_argument(
        "--agents", default="", help="Comma-separated agent names (empty=all)"
    )

    # doctor
    sub.add_parser("doctor", help="Verify hook installations and memory health")

    # export
    p = sub.add_parser("export", help="Export materialized memory")
    p.add_argument("--format", choices=["json", "markdown"], default="json")

    return parser


def main() -> None:
    """Entry point for agent-utilities-memory CLI."""
    parser = build_parser()
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    commands = {
        "context": cmd_context,
        "recall": cmd_recall,
        "observe": cmd_observe,
        "reflect": cmd_reflect,
        "materialize": cmd_materialize,
        "sync": cmd_sync,
        "install": cmd_install,
        "uninstall": cmd_uninstall,
        "doctor": cmd_doctor,
        "export": cmd_export,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
