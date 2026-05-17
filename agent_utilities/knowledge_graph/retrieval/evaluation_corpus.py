#!/usr/bin/python
from __future__ import annotations

"""CONCEPT:KG-2.3 — Fixed Corpus Evaluation Mode.

Enables reproducible agent benchmarking by constraining retrieval to a
curated, frozen document set. Inspired by BrowseComp-Plus (arXiv:2508.06600),
which proves that fixed corpora are essential for fair, reproducible
evaluation of deep-research agents.

Key capabilities:
    - **EvaluationCorpus**: Named set of document IDs with optional
      query-answer pairs for benchmark evaluation.
    - **CorpusManager**: CRUD operations for corpora, including freeze
      semantics that make a corpus immutable for reproducibility.
    - **Constrained retrieval**: ``HybridRetriever.retrieve_hybrid()``
      accepts a ``corpus_id`` parameter to restrict search scope.

Usage::

    manager = CorpusManager(engine)

    # Create a corpus from ingested documents
    corpus_id = manager.create_corpus(
        name="browsecomp-v1",
        document_ids=["doc-001", "doc-002", "doc-003"],
        queries=[
            {"query": "What is X?", "answer": "Y", "gold_docs": ["doc-001"]},
        ],
    )

    # Freeze for reproducible benchmarking
    manager.freeze_corpus(corpus_id)

    # Use in retrieval
    results = retriever.retrieve_hybrid("test query", corpus_id=corpus_id)

See docs/pillars/2_epistemic_knowledge_graph/KG-2.3-Graph_Integrity_And_Retrieval.md
"""

import json
import logging
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ..core.engine import IntelligenceGraphEngine

logger = logging.getLogger(__name__)


class CorpusQuery(BaseModel):
    """A benchmark query with expected answer and gold document references.

    Used for evaluation: the gold_doc_ids identify which documents in the
    corpus contain the answer, enabling nDCG and recall computation.
    """

    query: str = Field(description="The benchmark query text")
    answer: str = Field(default="", description="Expected answer for scoring")
    gold_doc_ids: list[str] = Field(
        default_factory=list,
        description="Document IDs that contain evidence for the answer",
    )
    difficulty: str = Field(
        default="medium",
        description="Estimated difficulty: easy, medium, hard",
    )


class EvaluationCorpus(BaseModel):
    """A named, versioned set of documents for reproducible evaluation.

    CONCEPT:KG-2.3 — Fixed Corpus Evaluation Mode (BrowseComp-Plus)

    When ``frozen`` is True, the corpus is immutable: document IDs and
    queries cannot be modified. This guarantees that benchmark results
    are comparable across runs and agent versions.
    """

    corpus_id: str = Field(description="Unique identifier for this corpus")
    name: str = Field(description="Human-readable corpus name")
    description: str = Field(default="", description="Purpose and contents")
    document_ids: list[str] = Field(
        default_factory=list,
        description="IDs of KG nodes in this corpus",
    )
    queries: list[CorpusQuery] = Field(
        default_factory=list,
        description="Benchmark queries with gold answers",
    )
    frozen: bool = Field(
        default=False,
        description="Whether this corpus is immutable",
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
    )
    frozen_at: str | None = Field(
        default=None,
        description="Timestamp when the corpus was frozen",
    )
    document_count: int = Field(
        default=0,
        description="Number of documents (set automatically)",
    )
    query_count: int = Field(
        default=0,
        description="Number of queries (set automatically)",
    )


class CorpusManager:
    """Manages evaluation corpora for reproducible benchmarking.

    CONCEPT:KG-2.3 — Fixed Corpus Evaluation Mode

    Persists corpora as KG nodes with type ``EvaluationCorpus`` for
    discovery via ``kg_search`` and ``kg_query``.

    Args:
        engine: The ``IntelligenceGraphEngine`` instance.
    """

    CORPUS_NODE_TYPE = "EvaluationCorpus"

    def __init__(self, engine: IntelligenceGraphEngine) -> None:
        self._engine = engine

    def create_corpus(
        self,
        name: str,
        document_ids: list[str],
        queries: list[dict[str, Any]] | None = None,
        description: str = "",
    ) -> str:
        """Create a new evaluation corpus.

        Args:
            name: Human-readable name for the corpus.
            document_ids: List of KG node IDs to include.
            queries: Optional benchmark queries (dicts with query/answer/gold_doc_ids).
            description: Purpose description.

        Returns:
            The generated corpus_id.
        """
        corpus_id = f"corpus-{uuid.uuid4().hex[:12]}"
        parsed_queries = [
            CorpusQuery(**q) if isinstance(q, dict) else q for q in (queries or [])
        ]

        corpus = EvaluationCorpus(
            corpus_id=corpus_id,
            name=name,
            description=description,
            document_ids=document_ids,
            queries=parsed_queries,
            document_count=len(document_ids),
            query_count=len(parsed_queries),
        )

        # Persist to KG
        props = corpus.model_dump(mode="json")
        # Serialize nested models as JSON strings for KG storage
        props["queries"] = json.dumps([q.model_dump() for q in parsed_queries])
        props["document_ids"] = json.dumps(document_ids)

        self._engine.add_node(
            corpus_id,
            self.CORPUS_NODE_TYPE,
            properties=props,
        )

        logger.info(
            "Created evaluation corpus %r (%s docs, %s queries)",
            name,
            len(document_ids),
            len(parsed_queries),
        )
        return corpus_id

    def freeze_corpus(self, corpus_id: str) -> bool:
        """Freeze a corpus to make it immutable for benchmarking.

        Args:
            corpus_id: The corpus to freeze.

        Returns:
            True if frozen successfully, False if already frozen or not found.
        """
        corpus = self.get_corpus(corpus_id)
        if corpus is None:
            logger.warning("Corpus %s not found", corpus_id)
            return False

        if corpus.frozen:
            logger.info("Corpus %s is already frozen", corpus_id)
            return False

        # Update the KG node
        self._engine.add_node(
            corpus_id,
            self.CORPUS_NODE_TYPE,
            properties={
                "frozen": True,
                "frozen_at": datetime.now(UTC).isoformat(),
            },
        )
        logger.info("Froze evaluation corpus %s", corpus_id)
        return True

    def get_corpus(self, corpus_id: str) -> EvaluationCorpus | None:
        """Load a corpus by ID.

        Args:
            corpus_id: The corpus identifier.

        Returns:
            EvaluationCorpus or None if not found.
        """
        try:
            results = self._engine.query_cypher(
                "MATCH (n) WHERE n.id = $cid AND n.type = $ctype RETURN n LIMIT 1",
                {"cid": corpus_id, "ctype": self.CORPUS_NODE_TYPE},
            )
            if not results:
                return None

            data = results[0].get("n", results[0])

            # Deserialize JSON-encoded fields
            doc_ids = data.get("document_ids", "[]")
            if isinstance(doc_ids, str):
                doc_ids = json.loads(doc_ids)

            queries_raw = data.get("queries", "[]")
            if isinstance(queries_raw, str):
                queries_raw = json.loads(queries_raw)

            return EvaluationCorpus(
                corpus_id=data.get("corpus_id", corpus_id),
                name=data.get("name", ""),
                description=data.get("description", ""),
                document_ids=doc_ids,
                queries=[CorpusQuery(**q) for q in queries_raw],
                frozen=data.get("frozen", False),
                created_at=data.get("created_at", ""),
                frozen_at=data.get("frozen_at"),
                document_count=len(doc_ids),
                query_count=len(queries_raw),
            )
        except Exception as e:
            logger.warning("Failed to load corpus %s: %s", corpus_id, e)
            return None

    def list_corpora(self) -> list[EvaluationCorpus]:
        """List all evaluation corpora.

        Returns:
            List of EvaluationCorpus objects.
        """
        try:
            results = self._engine.query_cypher(
                "MATCH (n) WHERE n.type = $ctype RETURN n",
                {"ctype": self.CORPUS_NODE_TYPE},
            )
            corpora = []
            for row in results:
                data = row.get("n", row)
                cid = data.get("corpus_id", data.get("id", ""))
                corpus = self.get_corpus(cid)
                if corpus:
                    corpora.append(corpus)
            return corpora
        except Exception as e:
            logger.warning("Failed to list corpora: %s", e)
            return []

    def get_document_ids(self, corpus_id: str) -> set[str]:
        """Get the set of document IDs in a corpus.

        This is the primary interface used by ``HybridRetriever`` to
        constrain search scope.

        Args:
            corpus_id: The corpus identifier.

        Returns:
            Set of document IDs, empty if corpus not found.
        """
        corpus = self.get_corpus(corpus_id)
        if corpus is None:
            return set()
        return set(corpus.document_ids)
