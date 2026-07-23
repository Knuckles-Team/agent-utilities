#!/usr/bin/python
"""Argument Interchange Format (AIF) support (CONCEPT:AU-KG.epistemic.aif).

Typed AIF argument-map objects + import (JSON/AIF-db-shaped -> graph)/export
(graph -> AIF JSON)/``to_dung()`` bridge layered over the engine's existing
Claim/Evidence/BeliefState + Dung argumentation — see :mod:`.aif`.
"""

from .aif import (
    AIF_NODE_TYPES,
    NODE_TYPE_CLASS,
    SCHEME_CLASS_BY_KIND,
    SCHEME_NODE_TYPES,
    AIFEdge,
    AIFNode,
    ArgumentMap,
    DungProjection,
    add_scheme,
    export_argument_map,
    from_aifdb_json,
    import_argument_map,
    to_aifdb_json,
    to_dung,
    validate_argument_map,
)

__all__ = [
    "AIF_NODE_TYPES",
    "SCHEME_NODE_TYPES",
    "NODE_TYPE_CLASS",
    "SCHEME_CLASS_BY_KIND",
    "AIFNode",
    "AIFEdge",
    "ArgumentMap",
    "DungProjection",
    "validate_argument_map",
    "from_aifdb_json",
    "to_aifdb_json",
    "to_dung",
    "import_argument_map",
    "export_argument_map",
    "add_scheme",
]
