import asyncio
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from agent_utilities.mcp.kg_server import _get_engine


async def ingest_codebase(target: Path, engine):
    from agent_utilities.core.paths import kg_db_path
    from agent_utilities.knowledge_graph.pipeline import IntelligencePipeline
    from agent_utilities.models.knowledge_graph import PipelineConfig

    print(f"Ingesting codebase: {target}")
    config = PipelineConfig(
        workspace_path=str(target),
        ladybug_path=str(kg_db_path()),
    )
    pipeline = IntelligencePipeline(config, backend=engine.backend)
    metadata = await pipeline.run()
    print(f"  -> Added {metadata.node_count} nodes, {metadata.edge_count} edges")


def ingest_document(target: Path, engine, provenance):
    from llama_index.core import SimpleDirectoryReader

    from agent_utilities.core.embedding_utilities import create_embedding_model

    print(f"Ingesting document: {target}")
    embed_model = create_embedding_model()
    if target.is_dir():
        docs = SimpleDirectoryReader(input_dir=str(target), recursive=True).load_data()
    else:
        docs = SimpleDirectoryReader(input_files=[str(target)]).load_data()

    created = []
    ingestion_timestamp = datetime.now(UTC).isoformat()
    for idx, doc in enumerate(docs):
        chunk_text = doc.text
        if not chunk_text.strip():
            continue
        file_path = doc.metadata.get("file_path", str(target))
        raw_id = f"{file_path}::{chunk_text}".encode()
        nid = f"doc-{hashlib.sha256(raw_id).hexdigest()[:8]}"
        existing = engine.query_cypher(
            "MATCH (n:Article {id: $nid}) RETURN n.id as id", {"nid": nid}
        )
        if existing:
            engine.backend.execute(
                "MATCH (n:Article {id: $nid}) SET n.last_seen_timestamp = $ts",
                {"nid": nid, "ts": ingestion_timestamp},
            )
            created.append(nid)
            continue
        embedding = embed_model.get_text_embedding(chunk_text)
        props = {
            "content": chunk_text,
            "embedding": embedding,
            "metadata": json.dumps(doc.metadata),
            "last_seen_timestamp": ingestion_timestamp,
            "target_path": str(target),
            "chunk_index": idx,
        }
        props.update(provenance)
        engine.add_node(nid, "Article", properties=props)
        created.append(nid)

    engine.backend.execute(
        "MATCH (n:Article) WHERE n.target_path = $target AND n.last_seen_timestamp < $ts DETACH DELETE n",
        {"target": str(target), "ts": ingestion_timestamp},
    )
    print(f"  -> Added {len(created)} document chunks")


async def main():
    try:
        engine = _get_engine()
        print("Engine initialized.")
        stats = engine.query_cypher(
            "MATCH (n) RETURN n.type AS type, count(*) AS count ORDER BY count DESC LIMIT 50"
        )
        print("Initial Stats:", stats)

        with open("/home/apps/workspace/scratch/paths.txt") as f:
            paths = [line.strip() for line in f if line.strip()]

        provenance = {
            "agent_id": "test-ingest-script",
            "session_id": "test-session",
            "workspace_path": "/home/apps/workspace",
            "timestamp": datetime.now(UTC).isoformat(),
            "source": "mcp",
        }

        for t_path in paths:
            target = Path(t_path)
            if not target.exists():
                print(f"Skipping {t_path}, does not exist.")
                continue

            is_codebase = False
            if target.is_dir():
                for indicator in [".git", "pyproject.toml", "package.json", "setup.py"]:
                    if (target / indicator).exists():
                        is_codebase = True
                        break

            try:
                if is_codebase:
                    await ingest_codebase(target, engine)
                elif target.is_file():
                    ingest_document(target, engine, provenance)
                else:
                    print(f"Skipping {target}: not a codebase and not a file.")
            except Exception as e:
                print(f"Failed to ingest {target}: {e}")

        final_stats = engine.query_cypher(
            "MATCH (n) RETURN n.type AS type, count(*) AS count ORDER BY count DESC LIMIT 50"
        )
        print("Final Stats:", final_stats)
    except Exception as e:
        print(f"Top-level Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
