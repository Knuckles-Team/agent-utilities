"""Characterization tests for patterns.py (CX-AU-06).

Pins OBSERVED behaviour of ``detect_class_patterns`` (pre-refactor CCN 28) and
``detect_function_patterns`` (pre-refactor CCN 13) exactly as they stand today.
No behaviour is changed by this commit.
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.enrichment.models import CodeEntity
from agent_utilities.knowledge_graph.enrichment.patterns import (
    detect_class_patterns,
    detect_function_patterns,
    detect_patterns,
)


def _class(
    name: str = "Thing",
    *,
    bases: list[str] | None = None,
    methods: list[str] | None = None,
    decorators: list[str] | None = None,
    is_abstract: bool = False,
) -> CodeEntity:
    return CodeEntity(
        id=f"code:{name}",
        name=name,
        qualname=name,
        kind="class",
        file_path="m.py",
        line=1,
        ast_hash="h",
        bases=bases or [],
        methods=methods or [],
        decorators=decorators or [],
        is_abstract=is_abstract,
    )


def _func(
    name: str = "thing",
    *,
    decorators: list[str] | None = None,
) -> CodeEntity:
    return CodeEntity(
        id=f"code:{name}",
        name=name,
        qualname=name,
        kind="function",
        file_path="m.py",
        line=1,
        ast_hash="h",
        decorators=decorators or [],
    )


# ── detect_class_patterns ────────────────────────────────────────────────


def test_non_class_kind_returns_empty() -> None:
    c = _func("whatever")
    assert not (detect_class_patterns(c) != [])


def test_is_abstract_flag_tags_abstract_base_class() -> None:
    c = _class(is_abstract=True)
    assert "AbstractBaseClass" in detect_class_patterns(c)


def test_abc_base_tags_abstract_base_class() -> None:
    c = _class(bases=["ABC"])
    assert "AbstractBaseClass" in detect_class_patterns(c)


def test_pydantic_base_model_tags_data_model() -> None:
    c = _class(bases=["BaseModel"])
    assert "DataModel" in detect_class_patterns(c)


def test_dataclass_decorator_tags_data_model() -> None:
    c = _class(decorators=["dataclass"])
    assert "DataModel" in detect_class_patterns(c)


def test_enum_base_tags_enumeration() -> None:
    c = _class(bases=["Enum"])
    assert "Enumeration" in detect_class_patterns(c)


def test_exception_base_tags_exception() -> None:
    c = _class(bases=["Exception"])
    assert "Exception" in detect_class_patterns(c)


def test_name_ending_error_tags_exception_without_base() -> None:
    c = _class(name="ParseError")
    assert "Exception" in detect_class_patterns(c)


def test_sync_context_manager_methods_tag_context_manager() -> None:
    c = _class(methods=["__enter__", "__exit__"])
    assert "ContextManager" in detect_class_patterns(c)


def test_async_context_manager_methods_tag_context_manager() -> None:
    c = _class(methods=["__aenter__", "__aexit__"])
    assert "ContextManager" in detect_class_patterns(c)


def test_dunder_new_tags_singleton() -> None:
    c = _class(methods=["__new__"])
    assert "Singleton" in detect_class_patterns(c)


def test_name_ending_singleton_tags_singleton() -> None:
    c = _class(name="ConfigSingleton")
    assert "Singleton" in detect_class_patterns(c)


def test_iterator_methods_tag_iterator() -> None:
    c = _class(methods=["__iter__", "__next__"])
    assert "Iterator" in detect_class_patterns(c)


def test_call_method_tags_callable() -> None:
    c = _class(methods=["__call__"])
    assert "Callable" in detect_class_patterns(c)


def test_factory_name_suffix_tags_factory() -> None:
    c = _class(name="WidgetFactory")
    assert "Factory" in detect_class_patterns(c)


def test_create_prefixed_method_tags_factory() -> None:
    c = _class(methods=["create_widget"])
    assert "Factory" in detect_class_patterns(c)


def test_strategy_name_suffix_tags_strategy() -> None:
    c = _class(name="RetryPolicy")
    assert "Strategy" in detect_class_patterns(c)


def test_repository_name_suffix_tags_repository() -> None:
    c = _class(name="UserRepository")
    assert "Repository" in detect_class_patterns(c)


def test_manager_name_suffix_tags_manager() -> None:
    c = _class(name="ConnectionManager")
    assert "Manager" in detect_class_patterns(c)


def test_adapter_name_suffix_tags_adapter() -> None:
    c = _class(name="LegacyAdapter")
    assert "Adapter" in detect_class_patterns(c)


def test_observer_name_suffix_tags_observer() -> None:
    c = _class(name="EventListener")
    assert "Observer" in detect_class_patterns(c)


def test_on_prefixed_method_tags_observer() -> None:
    c = _class(methods=["on_change"])
    assert "Observer" in detect_class_patterns(c)


def test_mixin_name_suffix_tags_mixin() -> None:
    c = _class(name="LoggingMixin")
    assert "Mixin" in detect_class_patterns(c)


def test_plain_class_with_no_signals_has_no_tags() -> None:
    c = _class(name="Widget")
    assert detect_class_patterns(c) == []


def test_tag_order_is_detection_order_and_deduplicated_by_detect_patterns() -> None:
    # A class matching several independent rules keeps the source's own
    # if-chain order -- this is the ordering invariant a naive rewrite
    # (e.g. a dict keyed by rule, iterated in arbitrary order) could invert.
    c = _class(
        name="ThingManager",
        bases=["ABC"],
        methods=["__call__"],
        is_abstract=True,
    )
    tags = detect_class_patterns(c)
    assert tags == ["AbstractBaseClass", "Callable", "Manager"]


# ── detect_function_patterns ─────────────────────────────────────────────


def test_non_function_kind_returns_empty() -> None:
    c = _class("Whatever")
    assert not (detect_function_patterns(c) != [])


def test_property_decorator_tags_property() -> None:
    f = _func(decorators=["property"])
    assert "Property" in detect_function_patterns(f)


def test_contextmanager_decorator_tags_context_manager() -> None:
    f = _func(decorators=["contextmanager"])
    assert "ContextManager" in detect_function_patterns(f)


def test_asynccontextmanager_decorator_tags_context_manager() -> None:
    f = _func(decorators=["asynccontextmanager"])
    assert "ContextManager" in detect_function_patterns(f)


def test_lru_cache_decorator_tags_memoized() -> None:
    f = _func(decorators=["lru_cache"])
    assert "Memoized" in detect_function_patterns(f)


def test_staticmethod_decorator_tags_static_method() -> None:
    f = _func(decorators=["staticmethod"])
    assert "StaticMethod" in detect_function_patterns(f)


def test_classmethod_decorator_tags_class_method() -> None:
    f = _func(decorators=["classmethod"])
    assert "ClassMethod" in detect_function_patterns(f)


def test_create_prefixed_name_tags_factory() -> None:
    f = _func(name="create_widget")
    assert "Factory" in detect_function_patterns(f)


def test_get_prefixed_name_tags_accessor() -> None:
    f = _func(name="get_widget")
    assert "Accessor" in detect_function_patterns(f)


def test_plain_function_has_no_tags() -> None:
    f = _func(name="compute")
    assert detect_function_patterns(f) == []


# ── detect_patterns (dispatch + dedupe) ──────────────────────────────────


def test_detect_patterns_dispatches_by_kind() -> None:
    assert detect_patterns(_class(name="UserRepository")) == ["Repository"]
    assert detect_patterns(_func(name="get_widget")) == ["Accessor"]


def test_detect_patterns_dedupes_preserving_first_occurrence_order() -> None:
    # Both name endings that map to "Factory" fire; the combined list must
    # de-duplicate while keeping first-seen order (source uses a `seen` set).
    c = _class(name="WidgetFactory", methods=["create_widget"])
    assert detect_patterns(c) == ["Factory"]
