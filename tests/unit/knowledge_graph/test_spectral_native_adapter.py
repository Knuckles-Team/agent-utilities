"""The bounded EG spectral result preserves the AU navigator contract."""

import pytest

from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine
from agent_utilities.knowledge_graph.core.spectral_navigator import (
    SpectralClusterNavigator,
)
from agent_utilities.knowledge_graph.core.topological_analysis_engine import (
    TopologicalAnalysisEngine,
)
from agent_utilities.knowledge_graph.retrieval.semantic_retrieval_engine import (
    KGNativeRetrievalRetriever,
)

_VECTORS = [
    [1.0, 0.0],
    [0.98, 0.02],
    [0.96, 0.04],
    [0.0, 1.0],
    [0.02, 0.98],
    [0.04, 0.96],
]


def _native_result():
    return {
        "n_rows": 6,
        "n_clusters": 2,
        "labels": [0, 0, 0, 1, 1, 1],
        "clusters": [
            {
                "cluster_id": 0,
                "members": [0, 1, 2],
                "centroid": [0.98, 0.02],
                "coherence": 0.999,
            },
            {
                "cluster_id": 1,
                "members": [3, 4, 5],
                "centroid": [0.02, 0.98],
                "coherence": 0.999,
            },
        ],
    }


def test_small_matrix_uses_native_result_and_preserves_dto():
    calls = []

    def native(vectors, max_k):
        calls.append((vectors, max_k))
        return _native_result()

    clusters = SpectralClusterNavigator(native_cluster=native).cluster(
        _VECTORS, max_k=4, domain="research"
    )
    assert calls == [(_VECTORS, 4)]
    assert [row.indices for row in clusters] == [[0, 1, 2], [3, 4, 5]]
    assert [row.label for row in clusters] == [
        "research_cluster_0",
        "research_cluster_1",
    ]
    assert all(row.cluster_id.startswith("sc_") for row in clusters)
    assert clusters[0].centroid == [0.98, 0.02]
    assert clusters[0].coherence == 0.999


def test_native_error_does_not_run_a_second_local_algorithm(monkeypatch):
    navigator = SpectralClusterNavigator(
        native_cluster=lambda _vectors, _max_k: (_ for _ in ()).throw(RuntimeError("EG down"))
    )
    monkeypatch.setattr(
        navigator,
        "_cosine_similarity_matrix",
        lambda _vectors: pytest.fail("local spectral algorithm ran"),
    )
    with pytest.raises(RuntimeError, match="EG down"):
        navigator.cluster(_VECTORS)


def test_oversize_matrix_uses_existing_local_path(monkeypatch):
    navigator = SpectralClusterNavigator(
        native_cluster=lambda _vectors, _max_k: pytest.fail("oversize native call")
    )

    def local(_vectors):
        raise RuntimeError("local spectral path")

    monkeypatch.setattr(navigator, "_cosine_similarity_matrix", local)
    with pytest.raises(RuntimeError, match="local spectral path"):
        navigator.cluster([[1.0, 0.0]] * 65)


def test_64_row_boundary_is_native():
    vectors = [[1.0, 0.0]] * 32 + [[0.0, 1.0]] * 32
    calls = []

    def native(rows, max_k):
        calls.append((len(rows), max_k))
        return {
            "n_rows": 64,
            "n_clusters": 2,
            "labels": [0] * 32 + [1] * 32,
            "clusters": [
                {
                    "cluster_id": cluster_id,
                    "members": list(range(start, start + 32)),
                    "centroid": centroid,
                    "coherence": 1.0,
                }
                for cluster_id, start, centroid in (
                    (0, 0, [1.0, 0.0]),
                    (1, 32, [0.0, 1.0]),
                )
            ],
        }

    clusters = SpectralClusterNavigator(native_cluster=native).cluster(vectors)
    assert calls == [(64, 10)]
    assert [len(row.indices) for row in clusters] == [32, 32]


def test_malformed_native_partition_is_rejected():
    result = _native_result()
    result["clusters"][0]["members"] = [0, 1, 2, 2]
    navigator = SpectralClusterNavigator(native_cluster=lambda _vectors, _max_k: result)
    with pytest.raises(ValueError, match="assigns a row twice"):
        navigator.cluster(_VECTORS)


def test_native_singleton_is_filtered_by_au_minimum_size():
    result = _native_result()
    result["labels"] = [0, 0, 0, 0, 0, 1]
    result["clusters"][0]["members"] = [0, 1, 2, 3, 4]
    result["clusters"][1]["members"] = [5]
    result["clusters"][1]["coherence"] = 1.0
    clusters = SpectralClusterNavigator(native_cluster=lambda _vectors, _max_k: result).cluster(
        _VECTORS
    )
    assert [row.indices for row in clusters] == [[0, 1, 2, 3, 4]]


def test_topology_facade_wires_native_graph_method():
    class Graph:
        def spectral_cluster(self, vectors, max_k):
            assert vectors == _VECTORS and max_k == 4
            return _native_result()

    clusters = TopologicalAnalysisEngine(Graph()).build_spectral_clusters(
        _VECTORS, max_k=4
    )
    assert [row.indices for row in clusters] == [[0, 1, 2], [3, 4, 5]]
    nested = type("Engine", (), {"graph_compute": Graph()})()
    clusters = TopologicalAnalysisEngine(nested).build_spectral_clusters(
        _VECTORS, max_k=4
    )
    assert [row.indices for row in clusters] == [[0, 1, 2], [3, 4, 5]]


def test_retrieval_facade_wires_native_graph_method():
    class Compute:
        def spectral_cluster(self, vectors, max_k):
            assert vectors == _VECTORS and max_k == 10
            return _native_result()

    engine = type("Engine", (), {"graph_compute": Compute()})()
    retriever = KGNativeRetrievalRetriever(engine)
    clusters = retriever._spectral_nav.cluster(_VECTORS)
    assert [row.indices for row in clusters] == [[0, 1, 2], [3, 4, 5]]


def test_graph_compute_sends_read_only_spectral_request():
    calls = []

    class Mining:
        def cluster(self, **kwargs):
            calls.append(kwargs)
            return _native_result()

    engine = GraphComputeEngine.__new__(GraphComputeEngine)
    engine._client = type("Client", (), {"mining": Mining()})()
    assert engine.spectral_cluster(_VECTORS, 4) == _native_result()
    assert calls == [
        {
            "features": _VECTORS,
            "algorithm": "spectral",
            "k": 4,
            "seed": 42,
            "writeback": False,
        }
    ]
