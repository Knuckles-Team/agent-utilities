"""CONCEPT:AU-OS.deployment.agent-factory-autoload

Regression test for B-22: ``agent_utilities/__init__.py``'s module-level ``__getattr__``
(PEP 562 lazy-import registry) had a stale entry -- ``SemanticCompactor`` pointed at
``.knowledge_graph.memory.memory_compaction``, a module that does not exist, so
``agent_utilities.SemanticCompactor`` raised ``ModuleNotFoundError`` at attribute-access
time for anyone who imported it that way (the real location is
``.knowledge_graph.memory.agent_context``).

The package keeps its complete lazy-import surface in the declarative
``_LAZY_MODULE_EXPORTS`` registry and derives ``_LAZY_EXPORTS`` from it. This test
audits both layers, then asserts each resolved module imports cleanly and actually
defines every name it re-exports -- so a *future* lazy-import entry that drifts from
its target (a rename, a moved module) fails this test instead of only failing for
whichever caller happens to touch that specific attribute first.
"""

from __future__ import annotations

import importlib

import agent_utilities

_EXPECTED_LAZY_MODULE_COUNT = 41
_EXPECTED_LAZY_EXPORT_COUNT = 117


def _getattr_import_targets() -> list[tuple[str, list[str]]]:
    """Normalize the package's declarative registry into absolute module targets."""
    return [
        (
            importlib.util.resolve_name(module_name, agent_utilities.__name__),
            names.split(),
        )
        for module_name, names in agent_utilities._LAZY_MODULE_EXPORTS
    ]


def _flatten_targets(
    targets: list[tuple[str, list[str]]],
) -> list[tuple[str, str]]:
    return [(module_name, name) for module_name, names in targets for name in names]


def _duplicate_names(names: list[str]) -> list[str]:
    return sorted(name for name in set(names) if names.count(name) > 1)


def _expected_dispatch() -> list[tuple[str, tuple[str, str]]]:
    return [
        (name, (module_name, name))
        for module_name, names in agent_utilities._LAZY_MODULE_EXPORTS
        for name in names.split()
    ]


def _assert_registry_shape(targets: list[tuple[str, list[str]]]) -> None:
    module_names = [module_name for module_name, _ in targets]
    assert len(targets) == _EXPECTED_LAZY_MODULE_COUNT, (
        "sanity check: expected the complete 41-module lazy-import surface; "
        f"found {len(targets)} registry entries"
    )
    assert len(set(module_names)) == _EXPECTED_LAZY_MODULE_COUNT, (
        f"duplicate lazy-import modules: {module_names}"
    )

    flattened_targets = _flatten_targets(targets)
    assert len(flattened_targets) == _EXPECTED_LAZY_EXPORT_COUNT, (
        f"sanity check: expected all 117 lazy exports; found {len(flattened_targets)}"
    )

    export_names = [name for _, name in flattened_targets]
    duplicate_names = _duplicate_names(export_names)
    assert not duplicate_names, f"duplicate lazy export names: {duplicate_names}"

    assert list(agent_utilities._LAZY_EXPORTS.items()) == _expected_dispatch(), (
        "_LAZY_EXPORTS must contain every registry export exactly once, "
        "in registry order"
    )


def _target_failures(module_name: str, names: list[str]) -> list[str]:
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        return [f"{module_name!r} (importing {names}): {type(exc).__name__}: {exc}"]
    return [
        f"{module_name!r} has no attribute {name!r}"
        for name in names
        if not hasattr(module, name)
    ]


def _lazy_import_failures(targets: list[tuple[str, list[str]]]) -> list[str]:
    return [
        failure
        for module_name, names in targets
        for failure in _target_failures(module_name, names)
    ]


def test_every_lazy_import_target_module_exists_and_exports_its_names():
    targets = _getattr_import_targets()
    _assert_registry_shape(targets)

    failures = _lazy_import_failures(targets)
    assert not failures, (
        "broken lazy-import entries in agent_utilities.__getattr__:\n"
        + "\n".join(failures)
    )


def test_semantic_compactor_resolves_via_the_package_lazy_import():
    """Pin the exact B-20/B-22 regression: ``agent_utilities.SemanticCompactor`` must
    resolve to the real class, not raise ``ModuleNotFoundError``."""
    from agent_utilities.knowledge_graph.memory.agent_context import (
        SemanticCompactor as direct,
    )

    assert agent_utilities.SemanticCompactor is direct
