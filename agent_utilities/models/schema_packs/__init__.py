"""Pre-built Schema Packs for common domains.

CONCEPT:KG-2.2 — Schema Packs

Registry of domain-specific KG profiles. Use ``get_schema_pack()``
to instantiate a named pack, or ``list_schema_packs()`` to discover
available packs.
"""

from __future__ import annotations

from .biomedical import BiomedicalSchemaPack
from .core import CoreSchemaPack
from .evolution import EvolutionSchemaPack
from .finance import FinanceSchemaPack
from .research import ResearchSchemaPack

# --- Pack Registry ---

_REGISTRY: dict[str, type] = {
    "core": CoreSchemaPack,
    "research-state": ResearchSchemaPack,
    "biomedical": BiomedicalSchemaPack,
    "finance": FinanceSchemaPack,
    "evolution": EvolutionSchemaPack,
}


def get_schema_pack(
    name: str,
) -> (
    CoreSchemaPack
    | ResearchSchemaPack
    | BiomedicalSchemaPack
    | FinanceSchemaPack
    | EvolutionSchemaPack
):
    """Instantiate a named schema pack.

    Args:
        name: Pack identifier (e.g. ``research-state``, ``biomedical``).

    Returns:
        An instantiated ``SchemaPack`` subclass.

    Raises:
        KeyError: If the pack name is not registered.
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise KeyError(f"Unknown schema pack '{name}'. Available packs: {available}")
    return _REGISTRY[name]()  # type: ignore[return-value]


def list_schema_packs() -> list[dict[str, str]]:
    """Return metadata for all registered schema packs.

    Returns:
        List of dicts with ``name`` and ``description`` keys.
    """
    result = []
    for name, cls in sorted(_REGISTRY.items()):
        instance = cls()
        result.append({"name": name, "description": instance.description})
    return result


def register_schema_pack(name: str, pack_cls: type) -> None:
    """Register a custom schema pack at runtime.

    Args:
        name: Pack identifier.
        pack_cls: A ``SchemaPack`` subclass.
    """
    _REGISTRY[name] = pack_cls


__all__ = [
    "BiomedicalSchemaPack",
    "CoreSchemaPack",
    "FinanceSchemaPack",
    "ResearchSchemaPack",
    "EvolutionSchemaPack",
    "get_schema_pack",
    "list_schema_packs",
    "register_schema_pack",
]
