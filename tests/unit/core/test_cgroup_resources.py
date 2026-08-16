"""Tests for :mod:`agent_utilities.core.cgroup_resources` (U-64 / BUG-110).

Covers cgroup v2/v1 CPU + memory parsing directly against real files on disk
(via ``tmp_path`` + monkeypatching the module's file-path constants), and the
"tighter of cgroup vs host" composition in ``effective_cpu_cores``/
``effective_memory_limit_bytes``.
"""

from __future__ import annotations

from agent_utilities.core import cgroup_resources as cg


def _write(path, content):
    path.write_text(content)
    return str(path)


# --- cgroup v2 CPU -----------------------------------------------------------


def test_cgroup_v2_cpu_max_parses_quota_over_period(tmp_path, monkeypatch):
    p = _write(tmp_path / "cpu.max", "150000 100000\n")
    monkeypatch.setattr(cg, "_V2_CPU_MAX", p)
    assert cg.cgroup_cpu_limit_cores() == 1.5


def test_cgroup_v2_cpu_max_unset_returns_none(tmp_path, monkeypatch):
    p = _write(tmp_path / "cpu.max", "max 100000\n")
    monkeypatch.setattr(cg, "_V2_CPU_MAX", p)
    monkeypatch.setattr(cg, "_V1_CFS_QUOTA", str(tmp_path / "nope"))
    assert cg.cgroup_cpu_limit_cores() is None


def test_cgroup_missing_file_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(cg, "_V2_CPU_MAX", str(tmp_path / "does-not-exist"))
    monkeypatch.setattr(cg, "_V1_CFS_QUOTA", str(tmp_path / "also-missing"))
    monkeypatch.setattr(cg, "_V1_CFS_PERIOD", str(tmp_path / "also-missing-2"))
    assert cg.cgroup_cpu_limit_cores() is None


# --- cgroup v1 CPU -----------------------------------------------------------


def test_cgroup_v1_cpu_quota_over_period(tmp_path, monkeypatch):
    monkeypatch.setattr(cg, "_V2_CPU_MAX", str(tmp_path / "no-v2"))
    quota = _write(tmp_path / "cfs_quota_us", "150000\n")
    period = _write(tmp_path / "cfs_period_us", "100000\n")
    monkeypatch.setattr(cg, "_V1_CFS_QUOTA", quota)
    monkeypatch.setattr(cg, "_V1_CFS_PERIOD", period)
    assert cg.cgroup_cpu_limit_cores() == 1.5


def test_cgroup_v1_cpu_unlimited_quota_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(cg, "_V2_CPU_MAX", str(tmp_path / "no-v2"))
    quota = _write(tmp_path / "cfs_quota_us", "-1\n")
    period = _write(tmp_path / "cfs_period_us", "100000\n")
    monkeypatch.setattr(cg, "_V1_CFS_QUOTA", quota)
    monkeypatch.setattr(cg, "_V1_CFS_PERIOD", period)
    assert cg.cgroup_cpu_limit_cores() is None


# --- memory ------------------------------------------------------------------


def test_cgroup_v2_memory_max_parses_bytes(tmp_path, monkeypatch):
    p = _write(tmp_path / "memory.max", str(4 * 1024**3) + "\n")
    monkeypatch.setattr(cg, "_V2_MEM_MAX", p)
    assert cg.cgroup_memory_limit_bytes() == 4 * 1024**3


def test_cgroup_v2_memory_max_unset_returns_none(tmp_path, monkeypatch):
    p = _write(tmp_path / "memory.max", "max\n")
    monkeypatch.setattr(cg, "_V2_MEM_MAX", p)
    monkeypatch.setattr(cg, "_V1_MEM_LIMIT", str(tmp_path / "nope"))
    assert cg.cgroup_memory_limit_bytes() is None


def test_cgroup_v1_memory_unlimited_sentinel_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(cg, "_V2_MEM_MAX", str(tmp_path / "no-v2"))
    p = _write(tmp_path / "memory.limit_in_bytes", "9223372036854771712\n")
    monkeypatch.setattr(cg, "_V1_MEM_LIMIT", p)
    assert cg.cgroup_memory_limit_bytes() is None


# --- effective_* composition (cgroup vs host) --------------------------------


def test_effective_cpu_cores_uses_cgroup_when_tighter(monkeypatch):
    monkeypatch.setattr(cg, "cgroup_cpu_limit_cores", lambda: 1.5)
    monkeypatch.setattr(cg.os, "cpu_count", lambda: 192)
    assert cg.effective_cpu_cores() == 1.5


def test_effective_cpu_cores_falls_back_to_host_when_no_cgroup(monkeypatch):
    monkeypatch.setattr(cg, "cgroup_cpu_limit_cores", lambda: None)
    monkeypatch.setattr(cg.os, "cpu_count", lambda: 8)
    assert cg.effective_cpu_cores() == 8.0


def test_effective_memory_uses_the_tighter_of_cgroup_and_host(monkeypatch):
    monkeypatch.setattr(cg, "cgroup_memory_limit_bytes", lambda: 4 * 1024**3)

    class _FakeVM:
        available = 500 * 1024**3

    class _FakePsutil:
        @staticmethod
        def virtual_memory():
            return _FakeVM()

    monkeypatch.setitem(__import__("sys").modules, "psutil", _FakePsutil())
    assert cg.effective_memory_limit_bytes() == 4 * 1024**3
