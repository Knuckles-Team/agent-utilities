"""CONCEPT:KG-2.16 — concurrent parse + entity-build across sibling connections (#1+#2)."""

from __future__ import annotations

import threading

from agent_utilities.knowledge_graph.enrichment.extractors.code_test import (
    extract_source_parallel,
)


def _parse_result(name: str) -> dict:
    return {
        "nodes": [
            {
                "properties": {
                    "name": name,
                    "symbol_type": "Function",
                    "kind_detail": "function",
                    "language": "java",
                    "line": "1",
                    "ast_hash": name,
                }
            }
        ]
    }


class _FakeClient:
    """A sibling connection whose parse_files records concurrency."""

    def __init__(self, tracker):
        self._tracker = tracker
        self.closed = False
        self.graph = self

    def parse_files(self, files):
        self._tracker.enter()
        try:
            # one result per input file, in order
            return [_parse_result(fp.rsplit("/", 1)[-1].split(".")[0]) for fp, _src in files]
        finally:
            self._tracker.exit()

    def close(self):
        self.closed = True


class _Tracker:
    def __init__(self):
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def enter(self):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        # brief hold so overlap is observable
        import time

        time.sleep(0.02)

    def exit(self):
        with self._lock:
            self.active -= 1


class _FakeCompute:
    supports_batch_parse = True

    def __init__(self, n_clients):
        self.tracker = _Tracker()
        self._n = n_clients
        self.made = []

    def make_parse_clients(self, n):
        self.made = [_FakeClient(self.tracker) for _ in range(min(n, self._n))]
        return self.made

    def parse_files(self, files):  # serial fallback path
        return [_parse_result(fp.rsplit("/", 1)[-1].split(".")[0]) for fp, _src in files]


def test_parallel_parse_preserves_order_and_runs_concurrently():
    files = [(f"/r/F{i}.java", f"class F{i} {{}}\n") for i in range(20)]
    gc = _FakeCompute(n_clients=4)
    results = extract_source_parallel(files, gc, concurrency=4, chunk=2)

    # one ExtractionResult per file, in input order
    assert len(results) == 20
    names = [r.code[0].name for r in results]
    assert names == [f"F{i}" for i in range(20)]
    # genuinely concurrent (more than one parse RPC in flight at once)
    assert gc.tracker.max_active >= 2
    # all sibling connections closed
    assert all(c.closed for c in gc.made)


def test_falls_back_to_serial_when_no_pool():
    files = [(f"/r/F{i}.java", "class F {}\n") for i in range(5)]
    gc = _FakeCompute(n_clients=1)  # only 1 client → can't parallelise
    results = extract_source_parallel(files, gc, concurrency=4, chunk=2)
    assert len(results) == 5
    assert results[0].code[0].name == "F0"
