"""Typed mypy-exclusion policy model.

CONCEPT:AU-QUAL.gates.mypy-test-inclusion

AU-QUAL-R005 requires that agent-utilities' mypy configuration includes
``tests/`` rather than excluding it, with every reported error resolved by a
real type fix rather than a new ``# type: ignore``, an ``Any`` cast, or an
additional exclusion.

This is the ``.1`` slice for that row: a typed model that *refuses* to
represent a mypy exclusion policy that still carves tests out of coverage.
The remaining mypy errors across the real test suite are out of scope for
this slice (AU-QUAL-R005.2+); see ``specs/au-boundary-quality/tasks.md`` for
the recorded split.
"""

from __future__ import annotations

from pydantic import BaseModel, field_validator

_DISALLOWED_SUBSTRINGS = ("tests/", "tests\\", "test_*", "*_test.py")


class MypyExclusionPolicy(BaseModel):
    """The set of path globs a mypy run is allowed to exclude.

    Construction refuses any pattern that would carve the ``tests/`` tree
    (or a test-file naming convention) out of mypy coverage, per
    AU-QUAL-R005.
    """

    excluded_paths: tuple[str, ...] = ()

    @field_validator("excluded_paths")
    @classmethod
    def _refuse_test_exclusions(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for pattern in value:
            lowered = pattern.lower()
            if any(bad in lowered for bad in _DISALLOWED_SUBSTRINGS):
                raise ValueError(
                    "mypy exclusion policy refuses a pattern that excludes "
                    f"test files from coverage: {pattern!r} (AU-QUAL-R005)"
                )
        return value
