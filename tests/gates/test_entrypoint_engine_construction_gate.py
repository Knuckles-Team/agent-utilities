"""Meta-tests for the entrypoint engine-authority gate (D-WD-7 regression).

CONCEPT:AU-ECO.ui.one-engine-authority
"""

from __future__ import annotations

from pathlib import Path

from scripts.check_entrypoint_engine_construction import (
    agent_server_files,
    entrypoint_trees,
    scan_tree,
)


def _write(root: Path, relative: str, source: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def test_live_entrypoint_trees_are_clean() -> None:
    """Whichever entrypoint trees exist on THIS checkout must already pass.

    Degrades gracefully (empty dict, so no assertion trips) when a sibling
    repo (agent-webui, agent-terminal-ui, geniusbot) isn't cloned alongside
    agent-utilities -- exactly like check_coupling.py's existing convention.
    """
    repo_root = Path(__file__).resolve().parents[2].parent
    for name, path in entrypoint_trees(repo_root).items():
        found = scan_tree(path)
        assert not found, (
            f"{name} has entrypoint engine-construction violations: {found}"
        )
    for server_file in agent_server_files(repo_root):
        from scripts.check_entrypoint_engine_construction import _violations_in_source

        violations = _violations_in_source(
            server_file.read_text(encoding="utf-8", errors="ignore")
        )
        assert not violations, f"{server_file}: {violations}"


def test_gate_catches_the_exact_d_wd_7_shape(tmp_path: Path) -> None:
    """The gate must fail on the exact code shape D-WD-7 found in agent-webui."""
    _write(
        tmp_path,
        "api_extensions.py",
        """
from agent_utilities.knowledge_graph.backends import create_backend
from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine


def get_engine():
    backend = create_backend(backend_type='ladybug', db_path='/tmp/x')
    return IntelligenceGraphEngine(graph=None, backend=backend)
""",
    )
    found = scan_tree(tmp_path)
    assert "api_extensions.py" in found
    joined = "\n".join(found["api_extensions.py"])
    assert "create_backend(backend_type=...)" in joined
    assert "IntelligenceGraphEngine(...)" in joined


def test_gate_allows_the_sanctioned_seam(tmp_path: Path) -> None:
    """get_active()/get_or_create() are the sanctioned path and must never be flagged."""
    _write(
        tmp_path,
        "api_extensions.py",
        """
from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine


def get_engine():
    engine = IntelligenceGraphEngine.get_active()
    if not engine:
        engine = IntelligenceGraphEngine.get_or_create(defer_background_start=True)
    return engine
""",
    )
    assert scan_tree(tmp_path) == {}


def test_gate_does_not_confuse_messaging_registry_create_backend_with_the_graph_factory(
    tmp_path: Path,
) -> None:
    """``MessagingRegistry.create_backend`` is an unrelated same-named method.

    ``registry.create_backend("discord")`` (an attribute call on a
    MessagingRegistry instance) must NOT be flagged just because the graph
    package also exports a function of the same name -- this was a real false
    positive the first version of this gate produced against
    ``agent_utilities/messaging/service.py``.
    """
    _write(
        tmp_path,
        "service.py",
        """
class MessagingRegistry:
    def create_backend(self, platform, config=None):
        return object()


def connect(registry, platform):
    return registry.create_backend(platform)
""",
    )
    assert scan_tree(tmp_path) == {}


def test_gate_exempts_test_files(tmp_path: Path) -> None:
    """A fixture engine built for a test is not the serving path this gate guards."""
    _write(
        tmp_path,
        "__tests__/test_something.py",
        """
from agent_utilities.knowledge_graph.backends import create_backend
from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

backend = create_backend(backend_type='memory')
engine = IntelligenceGraphEngine(graph=None, backend=backend)
""",
    )
    assert scan_tree(tmp_path) == {}


def test_gate_requires_a_real_import_before_flagging_a_same_named_bare_call(
    tmp_path: Path,
) -> None:
    """A bare ``create_backend(...)``/``IntelligenceGraphEngine(...)`` call that is
    NOT the graph symbol (no matching import) must not be flagged either --
    the import-gate is what makes the AST scan precise instead of a brittle
    name-substring match."""
    _write(
        tmp_path,
        "unrelated.py",
        """
def create_backend(backend_type=None):
    return backend_type


def IntelligenceGraphEngine(x):
    return x


create_backend(backend_type="whatever")
IntelligenceGraphEngine(1)
""",
    )
    assert scan_tree(tmp_path) == {}
