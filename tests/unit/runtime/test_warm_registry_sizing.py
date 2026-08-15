"""BUG-110 regression: ``compute_warm_parent_count`` must size against the
cgroup CPU/memory envelope, not the host's (mirrors the U-64 fix for
``compute_ingest_worker_count`` — this function's own docstring already said
it "mirrors" that one, and it turned out to mirror the bug too)."""

from __future__ import annotations

from agent_utilities.runtime import warm_registry


def test_compute_warm_parent_count_bounded_by_cgroup_not_host(monkeypatch):
    """A live pod's 1.5-CPU/4-GiB cgroup on a 192-core host must size the
    warm-parent pool from the cgroup, not from the host's much larger view."""
    import agent_utilities.core.cgroup_resources as cgroup_resources

    monkeypatch.setattr(cgroup_resources, "effective_cpu_cores", lambda: 1.5)
    monkeypatch.setattr(
        cgroup_resources, "effective_memory_limit_bytes", lambda: 4 * (1024**3)
    )
    # max_cpu = max(2, int(1.5)) = 2; max_mem = max(1, int(4/1.5)) = 2 -> ceiling 2
    assert warm_registry.compute_warm_parent_count(None) == 2


def test_compute_warm_parent_count_configured_is_clamped_to_cgroup_ceiling(
    monkeypatch,
):
    """An explicit override must not force more parents than the cgroup can
    schedule -- the same U-64 contract, applied here."""
    import agent_utilities.core.cgroup_resources as cgroup_resources

    monkeypatch.setattr(cgroup_resources, "effective_cpu_cores", lambda: 1.5)
    monkeypatch.setattr(
        cgroup_resources, "effective_memory_limit_bytes", lambda: 4 * (1024**3)
    )
    assert warm_registry.compute_warm_parent_count(69) == 2


def test_compute_warm_parent_count_scales_up_on_a_generous_cgroup(monkeypatch):
    """Sanity check the ceiling genuinely tracks the cgroup both ways."""
    import agent_utilities.core.cgroup_resources as cgroup_resources

    monkeypatch.setattr(cgroup_resources, "effective_cpu_cores", lambda: 16.0)
    monkeypatch.setattr(
        cgroup_resources, "effective_memory_limit_bytes", lambda: 64 * (1024**3)
    )
    # max_cpu = 16; max_mem = int(64/1.5) = 42 -> ceiling min(16, 42) = 16
    assert warm_registry.compute_warm_parent_count(None) == 16
