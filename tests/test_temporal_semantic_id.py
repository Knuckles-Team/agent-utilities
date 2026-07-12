"""Tests for the ChronoID-style temporal semantic ID encoder (CONCEPT:AU-KG.query.chronoid-fits-residual-quantization)."""

from __future__ import annotations

import subprocess
import sys

import pytest

from agent_utilities.knowledge_graph.retrieval.temporal_semantic_id import (
    TemporalSemanticIdEncoder,
)
from agent_utilities.numeric import xp as np

_SECONDS_PER_DAY = 86400.0


def _clustered_embeddings(
    n_clusters: int = 4, per_cluster: int = 20, dim: int = 8, seed: int = 1
) -> list[list[float]]:
    """Build small synthetic clustered vectors with a fixed numpy seed."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_clusters, dim)) * 5.0
    rows: list[list[float]] = []
    for c in range(n_clusters):
        pts = centers[c] + rng.normal(scale=0.1, size=(per_cluster, dim))
        rows.extend(pts.tolist())
    return rows


def _fitted_encoder(**kwargs) -> TemporalSemanticIdEncoder:
    enc = TemporalSemanticIdEncoder(seed=0, **kwargs)
    enc.fit(_clustered_embeddings())
    return enc


def test_not_fitted_then_fitted():
    enc = TemporalSemanticIdEncoder(seed=0)
    assert enc.is_fitted is False
    with pytest.raises(RuntimeError):
        enc.encode_content([0.0] * 8)
    enc.fit(_clustered_embeddings())
    assert enc.is_fitted is True


def test_encode_length_is_one_plus_n_codebooks():
    enc = _fitted_encoder(n_codebooks=3)
    vec = _clustered_embeddings()[0]
    sid = enc.encode(vec, event_time_epoch=1000.0, now_epoch=1000.0)
    assert isinstance(sid, tuple)
    assert len(sid) == 1 + 3


def test_encode_content_length_is_n_codebooks():
    enc = _fitted_encoder(n_codebooks=2)
    codes = enc.encode_content(_clustered_embeddings()[0])
    assert len(codes) == 2


def test_determinism_same_inputs_same_code():
    enc = _fitted_encoder()
    vec = _clustered_embeddings()[5]
    a = enc.encode(vec, event_time_epoch=500_000.0, now_epoch=1_000_000.0)
    b = enc.encode(vec, event_time_epoch=500_000.0, now_epoch=1_000_000.0)
    assert a == b


def test_two_fits_same_seed_same_codebooks():
    vec = _clustered_embeddings()[7]
    one = _fitted_encoder().encode_content(vec)
    two = _fitted_encoder().encode_content(vec)
    assert one == two


def test_recent_event_yields_smaller_bucket_than_older():
    enc = _fitted_encoder(n_time_buckets=16, time_span_days=365.0)
    now = 1_000_000_000.0
    recent = now - 1.0 * _SECONDS_PER_DAY  # ~1 day old
    older = now - 300.0 * _SECONDS_PER_DAY  # ~300 days old
    b_recent = enc.time_bucket(recent, now_epoch=now)
    b_older = enc.time_bucket(older, now_epoch=now)
    assert b_recent < b_older


def test_most_recent_is_bucket_zero_and_future_clamps_to_zero():
    enc = _fitted_encoder()
    now = 2_000_000.0
    assert enc.time_bucket(now, now_epoch=now) == 0
    # Future timestamp (negative age) clamps to the most-recent bucket.
    assert enc.time_bucket(now + 10_000.0, now_epoch=now) == 0


def test_old_event_clamps_to_last_known_bucket():
    enc = _fitted_encoder(n_time_buckets=16, time_span_days=365.0)
    now = 1_000_000_000.0
    ancient = now - 10_000.0 * _SECONDS_PER_DAY
    # Last *known* bucket is n_time_buckets - 2 (the final one is "unknown").
    assert enc.time_bucket(ancient, now_epoch=now) == enc.n_time_buckets - 2


def test_unknown_event_time_uses_dedicated_last_bucket():
    enc = _fitted_encoder(n_time_buckets=16)
    bucket = enc.time_bucket(None, now_epoch=1234.0)
    assert bucket == enc.n_time_buckets - 1
    # Encoding with unknown time leads with the unknown bucket token.
    sid = enc.encode(_clustered_embeddings()[0], None, now_epoch=1234.0)
    assert sid[0] == enc.n_time_buckets - 1


def test_codes_within_codebook_range():
    enc = _fitted_encoder(n_codebooks=3, codebook_size=64)
    for vec in _clustered_embeddings():
        for code in enc.encode_content(vec):
            assert 0 <= code < 64


def test_fewer_samples_than_codebook_size_still_works():
    # 3 samples but codebook_size 64 -> clusters cap at n_samples.
    enc = TemporalSemanticIdEncoder(codebook_size=64, n_codebooks=2, seed=0)
    enc.fit([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    codes = enc.encode_content([1.0, 0.0, 0.0])
    assert len(codes) == 2
    for code in codes:
        assert 0 <= code < 64


def test_nan_input_does_not_break_encoding():
    enc = _fitted_encoder()
    dim = 8
    bad = [float("nan")] * dim
    codes = enc.encode_content(bad)
    assert len(codes) == enc.n_codebooks


def test_dim_mismatch_raises():
    enc = _fitted_encoder()
    with pytest.raises(ValueError):
        enc.encode_content([1.0, 2.0, 3.0])


def test_engine_query_imports_clean_without_numeric_kernel():
    """The messaging socket-listener imports ``engine_query`` (which pulls in
    ``temporal_semantic_id``) without ever touching the encoder's numeric code
    paths. That import must NOT require the epistemic-graph numeric kernel —
    only invoking ``TemporalSemanticIdEncoder`` methods should.

    Runs in a clean subprocess (a shared pytest session already has
    ``agent_utilities.numeric`` cached in ``sys.modules``) so this is a true
    cold-import check. The kernel modules are poisoned to ``None`` in
    ``sys.modules`` before the import, which forces Python's import system to
    raise ``ImportError`` for them — simulating a kernel-absent (lean/headless)
    environment, matching the ``agent-utilities-messaging`` deployment that
    runs BAKED agent-utilities without ``epistemic-graph[numeric]``.
    """
    probe = (
        "import sys, json\n"
        # Poison the kernel modules so any import of them raises ImportError,
        # regardless of whether epistemic-graph[numeric] is actually installed
        # in this dev environment.
        "sys.modules['epistemic_graph.numeric'] = None\n"
        "sys.modules['numeric'] = None\n"
        "import agent_utilities.knowledge_graph.orchestration.engine_query\n"
        "import agent_utilities.knowledge_graph.retrieval.temporal_semantic_id as tsi\n"
        "assert tsi.np is None, 'expected the numeric shim to be None without the kernel'\n"
        "print(json.dumps({'ok': True}))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"engine_query import failed without the numeric kernel:\n{result.stderr}"
    )
    assert '{"ok": true}' in result.stdout.strip().splitlines()[-1].lower()
