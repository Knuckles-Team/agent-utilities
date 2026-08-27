"""Characterization test for ``_editable_source_root`` (CX-AU-07).

Pins the OBSERVED behaviour of the pre-refactor function (CCN 48,
``agent_utilities/core/providers.py``) before it is decomposed into
per-concern helper functions (direct_url.json parsing, project-root
resolution, pyproject.toml declared-root extraction, candidate-root
validation/selection, and owned-path enumeration). This test must be
green against the unmodified function; if it is not, the test is wrong,
not the code. It is not touched again in the refactor commit.

No prior test in this repo exercises this function directly (only
indirectly, via real editable installs at runtime through
``provider_registrations``/``resolve_skill_provider_dirs``) -- every
scenario below is built from a real filesystem tree under ``tmp_path``
with a minimal fake ``Distribution`` standing in for
``importlib.metadata.Distribution`` (only ``read_text`` is used by the
function under test).
"""

from __future__ import annotations

import json

import pytest

from agent_utilities.core.providers import (
    ProviderRegistrationError,
    _editable_source_root,
)


class _FakeDistribution:
    """Stands in for importlib.metadata.Distribution: only read_text is used."""

    def __init__(self, direct_url_text: str | None):
        self._direct_url_text = direct_url_text

    def read_text(self, filename):
        assert filename == "direct_url.json"
        return self._direct_url_text


def _direct_url_json(project_path, editable=True):
    return json.dumps(
        {
            "url": project_path.as_uri(),
            "dir_info": {"editable": editable},
        }
    )


def _dist_for(project_path, editable=True):
    return _FakeDistribution(_direct_url_json(project_path, editable=editable))


def test_missing_direct_url_metadata_raises(tmp_path):
    dist = _FakeDistribution(None)
    with pytest.raises(ProviderRegistrationError, match="metadata is unavailable"):
        _editable_source_root("pkg", dist)


def test_malformed_direct_url_json_raises(tmp_path):
    dist = _FakeDistribution("{not valid json")
    with pytest.raises(ProviderRegistrationError, match="metadata is unavailable"):
        _editable_source_root("pkg", dist)


def test_non_editable_dir_info_raises(tmp_path):
    project = tmp_path / "proj"
    (project / "pkg").mkdir(parents=True)
    dist = _dist_for(project, editable=False)
    with pytest.raises(ProviderRegistrationError, match="no auditable file manifest"):
        _editable_source_root("pkg", dist)


def test_non_file_url_scheme_raises(tmp_path):
    dist = _FakeDistribution(
        json.dumps(
            {"url": "https://example.invalid/pkg", "dir_info": {"editable": True}}
        )
    )
    with pytest.raises(ProviderRegistrationError, match="must be local"):
        _editable_source_root("pkg", dist)


def test_nonexistent_project_path_raises(tmp_path):
    missing = tmp_path / "does-not-exist"
    dist = _dist_for(missing)
    with pytest.raises(ProviderRegistrationError, match="project is unavailable"):
        _editable_source_root("pkg", dist)


def test_project_path_is_a_symlink_raises(tmp_path):
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real_dir)
    dist = _dist_for(link)
    with pytest.raises(ProviderRegistrationError, match="not a regular directory"):
        _editable_source_root("pkg", dist)


def test_project_path_is_a_file_not_directory_raises(tmp_path):
    a_file = tmp_path / "notadir"
    a_file.write_text("x")
    dist = _dist_for(a_file)
    with pytest.raises(ProviderRegistrationError, match="not a regular directory"):
        _editable_source_root("pkg", dist)


def test_module_found_directly_under_project_root_no_pyproject(tmp_path):
    project = tmp_path / "proj"
    pkg_dir = project / "pkg"
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "__init__.py").write_text("")
    (pkg_dir / "mod.py").write_text("x = 1")
    dist = _dist_for(project)

    root, owned = _editable_source_root("pkg", dist)

    assert root == pkg_dir.resolve()
    assert owned == frozenset({"__init__.py", "mod.py"})


def test_module_found_under_src_layout(tmp_path):
    project = tmp_path / "proj"
    pkg_dir = project / "src" / "pkg"
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "__init__.py").write_text("")
    dist = _dist_for(project)

    root, owned = _editable_source_root("pkg", dist)

    assert root == pkg_dir.resolve()
    assert owned == frozenset({"__init__.py"})


def test_pyproject_declared_root_is_honored(tmp_path):
    project = tmp_path / "proj"
    pkg_dir = project / "lib" / "pkg"
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "__init__.py").write_text("")
    (project / "pyproject.toml").write_text(
        '[tool.setuptools.packages.find]\nwhere = ["lib"]\n'
    )
    dist = _dist_for(project)

    root, owned = _editable_source_root("pkg", dist)

    assert root == pkg_dir.resolve()
    assert owned == frozenset({"__init__.py"})


def test_pyproject_declared_root_absolute_is_unsafe(tmp_path):
    project = tmp_path / "proj"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        '[tool.setuptools.packages.find]\nwhere = ["/etc"]\n'
    )
    dist = _dist_for(project)
    with pytest.raises(ProviderRegistrationError, match="package root is unsafe"):
        _editable_source_root("pkg", dist)


def test_pyproject_declared_root_parent_traversal_is_unsafe(tmp_path):
    project = tmp_path / "proj"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        '[tool.setuptools.packages.find]\nwhere = ["../escape"]\n'
    )
    dist = _dist_for(project)
    with pytest.raises(ProviderRegistrationError, match="package root is unsafe"):
        _editable_source_root("pkg", dist)


def test_ambiguous_ownership_across_multiple_roots_raises(tmp_path):
    project = tmp_path / "proj"
    (project / "pkg").mkdir(parents=True)
    (project / "pkg" / "a.py").write_text("")
    (project / "src" / "pkg").mkdir(parents=True)
    (project / "src" / "pkg" / "b.py").write_text("")
    dist = _dist_for(project)
    with pytest.raises(ProviderRegistrationError, match="not uniquely owned"):
        _editable_source_root("pkg", dist)


def test_symlink_inside_candidate_path_raises(tmp_path):
    project = tmp_path / "proj"
    real_target = tmp_path / "real_pkg"
    real_target.mkdir()
    (real_target / "a.py").write_text("")
    project.mkdir()
    (project / "pkg").symlink_to(real_target)
    dist = _dist_for(project)
    with pytest.raises(ProviderRegistrationError, match="contains a link"):
        _editable_source_root("pkg", dist)


def test_module_directory_with_no_regular_files_raises(tmp_path):
    project = tmp_path / "proj"
    pkg_dir = project / "pkg"
    (pkg_dir / "subdir").mkdir(parents=True)
    dist = _dist_for(project)
    with pytest.raises(ProviderRegistrationError, match="has no assets"):
        _editable_source_root("pkg", dist)


def test_nested_module_target_dotted_path(tmp_path):
    project = tmp_path / "proj"
    pkg_dir = project / "outer" / "inner"
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "leaf.py").write_text("")
    dist = _dist_for(project)

    root, owned = _editable_source_root("outer.inner", dist)

    assert root == pkg_dir.resolve()
    assert owned == frozenset({"leaf.py"})
