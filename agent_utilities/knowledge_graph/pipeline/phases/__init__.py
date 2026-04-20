from .scan import scan_phase
from .parse import parse_phase
from .resolve import resolve_phase
from .mro import mro_phase
from .reference import reference_phase
from .communities import communities_phase
from .centrality import centrality_phase
from .embedding import embedding_phase
from .registry import registry_phase
from .memory import memory_phase
from .sync import sync_phase
from .knowledge_base import knowledge_base_phase

PHASES = [
    memory_phase,
    scan_phase,
    parse_phase,
    registry_phase,
    resolve_phase,
    mro_phase,
    reference_phase,
    communities_phase,
    centrality_phase,
    embedding_phase,
    sync_phase,
    knowledge_base_phase,
]
