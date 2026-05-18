#!/usr/bin/env python3
import sys
from pathlib import Path


def submit_diff(patch_file: str):
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    engine = IntelligenceGraphEngine.get_active()
    if not engine:
        import networkx as nx
        from agent_utilities.core.paths import ensure_dirs, kg_db_path
        from agent_utilities.knowledge_graph.backends import create_backend

        ensure_dirs()
        db_path = str(kg_db_path())
        backend = create_backend(backend_type="ladybug", db_path=db_path)
        graph = nx.MultiDiGraph()
        engine = IntelligenceGraphEngine(graph=graph, backend=backend)

    target_path = Path(patch_file).resolve()
    job_id = engine.submit_task(
        target_path=str(target_path),
        is_codebase=False,
        provenance={"source": "git_hook"},
        task_type="diff",
    )
    print(f"Submitted diff task {job_id}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        submit_diff(sys.argv[1])
    else:
        print("Usage: submit_diff.py <patch_file>")
