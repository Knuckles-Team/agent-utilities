import asyncio

from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine


async def main():
    engine = IntelligenceGraphEngine.get_active()
    if not engine:
        import networkx as nx

        from agent_utilities.core.paths import ensure_dirs
        from agent_utilities.knowledge_graph.backends import create_backend

        ensure_dirs()
        backend = create_backend(backend_type="ladybug")
        graph = nx.MultiDiGraph()
        engine = IntelligenceGraphEngine(graph=graph, backend=backend)
    if engine.backend:
        engine.backend.create_schema()
        print("Schema created.")
    else:
        print("No backend available.")


if __name__ == "__main__":
    asyncio.run(main())
