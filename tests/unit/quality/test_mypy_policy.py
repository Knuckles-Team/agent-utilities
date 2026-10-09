"""Tests for the AU-QUAL-R005.1 mypy exclusion policy model."""

import pytest
from pydantic import ValidationError

from agent_utilities.quality.mypy_policy import MypyExclusionPolicy


def test_accepts_non_test_exclusions() -> None:
    policy = MypyExclusionPolicy(excluded_paths=("build/", "dist/"))
    assert policy.excluded_paths == ("build/", "dist/")


def test_refuses_tests_directory_exclusion() -> None:
    """AU-QUAL-R005: tests/ may not be excluded from mypy coverage."""
    with pytest.raises(ValidationError, match="AU-QUAL-R005"):
        MypyExclusionPolicy(excluded_paths=("tests/",))


def test_refuses_test_file_glob_exclusion() -> None:
    with pytest.raises(ValidationError, match="AU-QUAL-R005"):
        MypyExclusionPolicy(excluded_paths=("*_test.py",))
