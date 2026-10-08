"""Shared fake for configuration-source TOCTOU tests."""

from __future__ import annotations

import os
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any


def fstat_changing_on_second_call() -> Callable[[int], Any]:
    """An ``os.fstat`` whose second call reports a changed modification time.

    Configuration readers stat a source before and after the read; this fake
    simulates a file replaced between the two calls.
    """
    real_fstat = os.fstat
    calls = 0

    def changed_fstat(descriptor: int) -> Any:
        nonlocal calls
        calls += 1
        metadata = real_fstat(descriptor)
        if calls != 2:
            return metadata
        return SimpleNamespace(
            st_mode=metadata.st_mode,
            st_size=metadata.st_size,
            st_uid=metadata.st_uid,
            st_dev=metadata.st_dev,
            st_ino=metadata.st_ino,
            st_mtime_ns=metadata.st_mtime_ns + 1,
        )

    return changed_fstat
