"""Design-pattern + implementation-style detection (CONCEPT:KG-2.8 Phase 2).

Deterministic detection over the structural facts the Rust parser extracts
(bases, methods, decorators, abstractness). Pure functions — fast, no LLM. The
results tag code entities so questions like "what design patterns are used" and
"how is this implemented" become graph facts. LLM capability cards (``cards.py``)
add the prose narrative on top.
"""

from __future__ import annotations

from .models import CodeEntity

_ABC_BASES = {"ABC", "ABCMeta", "Protocol", "abc.ABC", "typing.Protocol"}
_ENUM_BASES = {"Enum", "IntEnum", "StrEnum", "Flag", "enum.Enum"}
_EXC_BASES = {"Exception", "BaseException", "RuntimeError", "ValueError"}
_MODEL_BASES = {"BaseModel", "pydantic.BaseModel"}


def _base_set(c: CodeEntity) -> set[str]:
    out: set[str] = set()
    for b in c.bases:
        out.add(b)
        out.add(b.rsplit(".", 1)[-1])  # short name
    return out


def detect_class_patterns(c: CodeEntity) -> list[str]:
    """Return design-pattern / style tags for a class entity."""
    if c.kind != "class":
        return []
    tags: list[str] = []
    bases = _base_set(c)
    methods = set(c.methods)
    decos = {d.split("(")[0].rsplit(".", 1)[-1] for d in c.decorators}
    name = c.name

    if c.is_abstract or bases & _ABC_BASES:
        tags.append("AbstractBaseClass")
    if bases & _MODEL_BASES or "dataclass" in decos:
        tags.append("DataModel")
    if bases & _ENUM_BASES:
        tags.append("Enumeration")
    if bases & _EXC_BASES or name.endswith(("Error", "Exception")):
        tags.append("Exception")
    if {"__enter__", "__exit__"} <= methods or {"__aenter__", "__aexit__"} <= methods:
        tags.append("ContextManager")
    if "__new__" in methods or name.endswith("Singleton") or "instance" in methods:
        tags.append("Singleton")
    if {"__iter__", "__next__"} & methods:
        tags.append("Iterator")
    if "__call__" in methods:
        tags.append("Callable")
    if name.endswith(("Factory", "Builder")) or any(
        m.startswith(("create_", "build_", "make_")) for m in methods
    ):
        tags.append("Factory")
    if name.endswith(("Strategy", "Policy", "Handler", "Backend", "Provider")):
        tags.append("Strategy")
    if name.endswith(("Repository", "Store", "DAO", "Dao")):
        tags.append("Repository")
    if name.endswith(("Manager", "Coordinator", "Orchestrator")):
        tags.append("Manager")
    if name.endswith(("Adapter", "Wrapper", "Proxy")):
        tags.append("Adapter")
    if name.endswith(("Observer", "Listener", "Subscriber")) or any(
        m.startswith(("on_", "handle_", "notify")) for m in methods
    ):
        tags.append("Observer")
    if name.endswith(("Mixin",)):
        tags.append("Mixin")
    return tags


def detect_function_patterns(c: CodeEntity) -> list[str]:
    """Return style tags for a function entity."""
    if c.kind != "function":
        return []
    tags: list[str] = []
    decos = {d.split("(")[0].rsplit(".", 1)[-1] for d in c.decorators}
    name = c.name
    if "property" in decos:
        tags.append("Property")
    if "contextmanager" in decos or "asynccontextmanager" in decos:
        tags.append("ContextManager")
    if "cached_property" in decos or "lru_cache" in decos or "cache" in decos:
        tags.append("Memoized")
    if "staticmethod" in decos:
        tags.append("StaticMethod")
    if "classmethod" in decos:
        tags.append("ClassMethod")
    if name.startswith(("create_", "build_", "make_", "from_")):
        tags.append("Factory")
    if name.startswith(("get_", "fetch_", "load_", "read_")):
        tags.append("Accessor")
    return tags


def detect_patterns(c: CodeEntity) -> list[str]:
    """Dispatch by kind; returns de-duplicated, order-preserving tags."""
    raw = detect_class_patterns(c) + detect_function_patterns(c)
    seen: set[str] = set()
    out: list[str] = []
    for t in raw:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out
