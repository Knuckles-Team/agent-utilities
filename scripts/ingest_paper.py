import sys
from agent_utilities.mcp.kg_server import _get_engine, _provenance_props

def ingest(path):
    engine = _get_engine()
    provenance = _provenance_props()
    engine.submit_task(path, is_codebase=False, provenance=provenance)
    print("Ingested task submitted.")

if __name__ == "__main__":
    ingest(sys.argv[1])
