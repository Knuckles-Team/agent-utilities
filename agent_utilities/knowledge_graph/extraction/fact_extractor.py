"""Document → atomic-triple fact extraction core (CONCEPT:AU-KG.enrichment.atomic-triple-extraction).

Turns document text into a stream of ``ExtractedFact`` edges via a self-hosted
chat model, dedups them semantically (across rounds AND files) using our own
embedder, and persists them as graph edges.

Provenance: the prompt craft (canonical-entity forcing → graph connectivity),
the incremental streaming JSON parser, and the seed-varied multi-round recall
loop are assimilated from the open-source ``knowledge-graph-extractor`` (hanxiao,
MIT). The serving stack (vLLM), the embedder, the dedup math, and the graph
persistence are all our own — no second model is loaded for dedup, and rounds
default to 1 so the GPU cost is opt-in.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from collections.abc import AsyncGenerator, Callable, Iterable
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .extraction_schema import ExtractionSchema

logger = logging.getLogger(__name__)

# An extraction LLM call can legitimately run longer than a card summary (it
# emits up to thousands of JSON tokens), but it must still be bounded so one
# stalled vLLM request never wedges a job. One correct value, not an env knob
# (config discipline). Matches the upstream 300s read budget.
_EXTRACT_READ_TIMEOUT_S = 300.0
_EXTRACT_CONNECT_TIMEOUT_S = 10.0
_EXTRACT_MAX_TOKENS = 8192

# Dedup fields the caller may compare on (parity with upstream UI options).
DEDUP_FIELDS = ("triple", "title", "description", "title+desc", "triple+title", "all")


class ExtractedFact(BaseModel):
    """One knowledge-graph edge: ``(subject) --[predicate]--> (object)``.

    Field set is wire-compatible with the upstream ``facts.jsonl`` schema so our
    export is byte-for-byte comparable, while ``is_duplicate`` / ``source_file``
    carry our streaming + multi-file provenance.
    """

    subject: str
    predicate: str
    object: str
    title: str = ""
    description: str = ""
    evidence_span: str = ""
    confidence: int = 0  # 0..100 (upstream parity); persisted as 0..1 on the edge
    tags: list[str] = Field(default_factory=list)
    is_duplicate: bool = False
    source_file: str = ""

    @staticmethod
    def normalize_key(s: str) -> str:
        """Canonical node key for merging surface-form variants.

        NFKC-compose, lowercase, collapse whitespace, strip wrapping
        punctuation — so ``"The Jina AI team"`` and ``"jina ai team"`` resolve to
        one node. Ports the upstream frontend ``normKey`` to the persistence
        layer so merging happens in the graph, not just the viewport.
        """
        s = unicodedata.normalize("NFKC", s or "").lower()
        s = re.sub(r"[\s ]+", " ", s)
        s = re.sub(r"^[\s\"'`([{]+|[\s\"'`)\]}.,;:!?]+$", "", s)
        return s.strip()

    def dedup_text(self, field: str = "triple") -> str:
        """The text used for semantic-dedup embedding, per the chosen field."""
        if field == "title":
            return self.title
        if field == "description":
            return self.description
        if field == "title+desc":
            return f"{self.title} {self.description}"
        if field == "triple+title":
            return f"{self.subject} {self.predicate} {self.object} {self.title}"
        if field == "all":
            return f"{self.title} {self.description} {self.subject} {self.predicate} {self.object}"
        # default: "triple"
        return f"{self.subject} {self.predicate} {self.object}"


# --------------------------------------------------------------------------- #
# G1 — the extraction prompt + JSON schema
# --------------------------------------------------------------------------- #

FACT_EXTRACTION_PROMPT = """Extract a knowledge graph from the document. Return a JSON object with key
"facts" containing 0-15 atomic relationship facts. Each fact is ONE edge:
a (subject) --[predicate]--> (object) triple plus human-readable context.
Long, dense documents (articles, papers, profiles) typically warrant 8-15
facts; short or generic pages 0-3.

The single most important rule -- THIS DRIVES GRAPH CONNECTIVITY:
subject and object MUST be canonical ENTITIES or short atomic VALUES, never
prose. They are graph nodes: the same entity must come out identical every
time so edges connect. Put narrative, evidence and nuance in the description,
NOT in subject/object.

subject / object rules:
- Use the shortest canonical name of a real entity: a person, organisation,
  product/model, dataset, method, paper, place, technology, or a concrete
  atomic value (a date, a number+unit, a version, a metric score).
- Strip articles, roles, and qualifiers: "the Jina AI team" -> "Jina AI";
  "a model called jina-embeddings-v3" -> "jina-embeddings-v3".
- Use the canonical surface form, not a pronoun or paraphrase. Reuse the exact
  same string for the same entity across every fact (this is how nodes merge).
- Never put a sentence or clause in subject/object. If the value is inherently
  descriptive (e.g. "trained on 2B multilingual tokens"), make object the
  atomic value ("2B multilingual tokens") and explain in description.
- Prefer relationships that link TWO named entities (entity-entity edges) --
  these are what make the graph rich. Entity-value edges are fine too.

Each fact:
  {
  "title": "<one natural sentence <=140 chars stating the fact, ending with the value when possible>",
  "description": "<2-3 sentences <=350 chars carrying the answer + evidence: entities, relation, value, date/number/source detail, and an inline verbatim quote when it disambiguates. Avoid restating the title verbatim.>",
  "subject": "<canonical entity name, short>",
  "predicate": "<precise snake_case relation, <=32 chars>",
  "object": "<canonical entity name OR short atomic value>",
  "evidence_span": "<verbatim 1-3 sentence quote, substring of the doc text above>",
  "confidence": <0..100 integer>,
  "tags": ["<entity/topic/year tags, lowercase, alphanumeric+hyphen>", ...]
  }
Coverage priorities -- extract a fact for EACH of the following when grounded in the text:
- Every named person + their role / position / affiliation (even if named once).
- Every named organisation, product, model, dataset, or method + how it relates
  to other named entities (built_by, based_on, trained_on, outperforms, part_of).
- Every concrete date/version + the event or release it marks.
- Every named place + what happened or is located there.
- Every quantitative result: metric scores, sizes, token counts, speedups,
  prices -- as entity --[has_metric/scored]--> value edges.
- Every cross-entity relationship: X built Y, X based on Y, X collaborated with
  Y, X acquired Y, X cites Y, X compared against Y.
Anti-patterns -- do NOT do these:
- Don't put descriptive sentences into subject or object. That creates dead-end
  nodes that never connect. Keep nodes short and canonical.
- Don't only extract facts about the dominant entity. Secondary entities named
  once still warrant their own edge.
- Don't fill the budget with generic boilerplate (tagline, copyright, nav) at
  the expense of specific, connectable relationships deeper in the body.
Predicate guidance:
- Always choose the MOST precise snake_case predicate (<=32 chars) for the actual
  relation -- accuracy matters more than reusing a known term. Coin a new one
  freely whenever it fits better.
- The following are only EXAMPLES of the style/granularity (NOT a fixed list,
  NOT a menu to pick from): built_by, based_on, trained_on, fine_tuned_from,
  released_on, outperforms, evaluated_on, scored, integrates_with, acquired_by,
  authored_by, cites, position_held, successor_of, used_for. Do not force a
  relation into one of these if a more specific predicate describes it better.
- AVOID vague catch-alls like affiliated_with or related_to.
Title and description constraints (CRITICAL -- items violating these are dropped):
- title and description MUST read as natural standalone fact statements.
- They MUST NOT mention the document, the dataset, or this task.
Fact constraints:
- Favor specificity (proper nouns, versions, numbers) over generic claims.
- Skip the doc entirely (return empty facts list) for navigation pages, login
  walls, error pages, very short or generic content.
- evidence_span must be a verbatim substring of the doc text supplied above.

Output ONLY the JSON object."""


FACT_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "subject": {"type": "string"},
                    "predicate": {"type": "string"},
                    "object": {"type": "string"},
                    "evidence_span": {"type": "string"},
                    "confidence": {"type": "integer"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "title",
                    "description",
                    "subject",
                    "predicate",
                    "object",
                    "evidence_span",
                    "confidence",
                    "tags",
                ],
            },
        }
    },
    "required": ["facts"],
}


# --------------------------------------------------------------------------- #
# G2 — incremental streaming fact parser
# --------------------------------------------------------------------------- #

_FACT_START = re.compile(r'\{\s*"title"\s*:')


def parse_facts_incremental(text: str, seen: set[int]) -> list[dict[str, Any]]:
    """Extract complete fact objects from a *partial* LLM stream.

    Scans for ``{"title":`` openers, brace-matches each to its close, and emits
    any newly-completed object whose string hash isn't in ``seen`` — so facts
    surface to the UI as they finish rather than after the whole array. Cheap
    (regex + one linear scan per opener); safe on truncated tails.
    """
    new_facts: list[dict[str, Any]] = []
    for m in _FACT_START.finditer(text):
        start = m.start()
        depth = 0
        i = start
        n = len(text)
        while i < n:
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    obj_str = text[start : i + 1]
                    h = hash(obj_str)
                    if h not in seen:
                        try:
                            fact = json.loads(obj_str)
                        except json.JSONDecodeError:
                            pass
                        else:
                            if "title" in fact and "subject" in fact:
                                seen.add(h)
                                new_facts.append(fact)
                    break
            i += 1
    return new_facts


# --------------------------------------------------------------------------- #
# G4 — per-fact semantic dedup (reuses our embedder, vectorized)
# --------------------------------------------------------------------------- #

EmbedFn = Callable[[str], "list[float]"]


def _default_embed_fn() -> EmbedFn:
    """Build an embed fn from the canonical factory (no second model loaded).

    Reuses ``create_embedding_model`` — the same bge/nomic embedder the rest of
    the KG uses — so dedup adds no extra model to memory. Returns L2-normalized
    vectors so a dot product is a cosine.
    """
    from agent_utilities.core.embedding_utilities import create_embedding_model

    model = create_embedding_model()

    def _embed(text: str) -> list[float]:
        vec = model.get_text_embedding(text or " ")
        # normalize so dot == cosine; tolerate already-normalized vectors
        norm = sum(v * v for v in vec) ** 0.5
        if norm == 0.0:
            return list(vec)
        return [v / norm for v in vec]

    return _embed


class FactDeduper:
    """Accumulating semantic-dedup index over extracted facts.

    Holds the normalized embeddings of every *unique* fact kept so far and, for
    each new fact, returns the max cosine to the set in a single matvec (BLAS via
    numpy when present, pure-Python fallback otherwise). Survives across rounds
    and files; ``rehydrate`` rebuilds the corpus from prior facts on resume.
    """

    def __init__(
        self,
        embed_fn: EmbedFn | None = None,
        *,
        field: str = "triple",
        threshold: float = 0.90,
    ) -> None:
        if field not in DEDUP_FIELDS:
            field = "triple"
        self._embed_fn = embed_fn
        self.field = field
        self.threshold = threshold
        self._rows: list[list[float]] = []
        try:  # vectorize when numpy is available; degrade gracefully otherwise
            from agent_utilities.numeric import xp as np  # noqa: F401

            self._np = np
            self._mat: Any = None  # lazily-stacked (n, d) matrix
        except Exception:  # pragma: no cover - numpy is a core dep but stay safe
            self._np = None
            self._mat = None

    @property
    def embed_fn(self) -> EmbedFn:
        if self._embed_fn is None:
            self._embed_fn = _default_embed_fn()
        return self._embed_fn

    def __len__(self) -> int:
        return len(self._rows)

    def _add_vec(self, vec: list[float]) -> None:
        self._rows.append(vec)
        self._mat = None  # invalidate stacked cache

    def check(self, fact: ExtractedFact) -> tuple[bool, float]:
        """Return ``(is_duplicate, max_similarity)`` for ``fact``.

        On a non-duplicate, the fact's embedding is added to the corpus so later
        facts dedup against it. On a duplicate, nothing is added (the survivor
        already represents the cluster).
        """
        vec = self.embed_fn(fact.dedup_text(self.field))
        if not self._rows:
            self._add_vec(vec)
            return False, 0.0
        if self._np is not None:
            np = self._np
            if self._mat is None:
                self._mat = np.asarray(self._rows, dtype="float32")
            q = np.asarray(vec, dtype="float32")
            sims = self._mat @ q
            max_sim = float(sims.max())
        else:  # pragma: no cover - pure-Python fallback
            max_sim = max(
                sum(a * b for a, b in zip(row, vec, strict=False)) for row in self._rows
            )
        if max_sim >= self.threshold:
            return True, max_sim
        self._add_vec(vec)
        return False, max_sim

    def rehydrate(self, facts: Iterable[ExtractedFact]) -> None:
        """Seed the corpus from previously-kept (non-duplicate) facts on resume."""
        for f in facts:
            if f.is_duplicate:
                continue
            self._add_vec(self.embed_fn(f.dedup_text(self.field)))


# --------------------------------------------------------------------------- #
# Streaming LLM call (reuses the configured chat model — vLLM)
# --------------------------------------------------------------------------- #

StreamFn = Callable[[str, int], AsyncGenerator[str, None]]


def make_streaming_extract_fn(
    model: str | None = None,
    base_url: str | None = None,
) -> StreamFn:
    """Factory: an async fn ``(prompt, seed) -> AsyncGenerator[str]`` of deltas.

    Backed by the configured chat model (``vllm.arpa``) with the JSON-schema
    response format and the sampling profile tuned for factual extraction. Lazy
    so importing this module never requires the OpenAI client; on any failure it
    yields nothing and the caller degrades to zero facts for that round.
    """

    async def _stream(prompt: str, seed: int) -> AsyncGenerator[str, None]:
        try:
            from openai import AsyncOpenAI

            from agent_utilities.core.config import config, setting

            cfg = config.default_chat_model
            client = AsyncOpenAI(
                base_url=base_url
                or (cfg.base_url if cfg else None)
                or "http://vllm.arpa/v1",
                api_key=(cfg.api_key if cfg else None) or "not-needed",
                timeout=_EXTRACT_READ_TIMEOUT_S,
                # Retry transient backend errors (502/503/429/timeout) with the
                # SDK's exponential backoff — a momentarily overloaded vLLM must
                # not silently zero out a chunk's facts. Tunable via env.
                max_retries=int(setting("KG_EXTRACT_MAX_RETRIES", "4")),
            )
            model_id = model or (cfg.id if cfg else None) or "default"
            stream = await client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=_EXTRACT_MAX_TOKENS,
                stream=True,
                seed=seed,
                temperature=0.7,
                top_p=0.8,
                presence_penalty=1.5,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "kg_facts",
                        "strict": True,
                        "schema": FACT_JSON_SCHEMA,
                    },
                },
                extra_body={
                    "top_k": 20,
                    "min_p": 0.0,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as e:  # noqa: BLE001 — never let a bad call wedge a job
            logger.warning("fact extraction stream failed (%s); 0 facts", e)
            return

    return _stream


# --------------------------------------------------------------------------- #
# G2+G4 — the multi-round recall loop (the public entry point)
# --------------------------------------------------------------------------- #


def _seed_for_round(round_num: int, base: int = 1000) -> int:
    """Deterministic per-round seed (``Math.random`` is unavailable in this
    runtime, and a fixed schedule keeps extraction reproducible/testable)."""
    return base + round_num * 7919  # a prime stride so rounds diverge


async def extract_facts(
    text: str,
    *,
    rounds: int = 1,
    dedup: bool = True,
    dedup_field: str = "triple",
    dedup_threshold: float = 0.90,
    source_file: str = "",
    deduper: FactDeduper | None = None,
    stream_fn: StreamFn | None = None,
    prompt: str = FACT_EXTRACTION_PROMPT,
    schema: ExtractionSchema | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """Extract facts from ``text``, yielding events as they stream.

    Emits, in order, dicts shaped like the upstream SSE taxonomy so any frontend
    can render live: ``round_start`` → many ``fact`` (+ periodic ``metrics``) →
    ``round_end`` per round, then a final ``done``. ``rounds`` defaults to 1 —
    seed-varied multi-round recall (each round re-asks with a fresh seed and the
    accumulated deduper suppresses repeats) is opt-in so the GPU cost is a
    deliberate choice. The same ``deduper`` may be passed across documents to
    dedup across files.

    ``schema`` (CONCEPT:AU-KG.retrieval.mmr-diversification) — when an :class:`ExtractionSchema` is supplied,
    its OWL classes + ``rdfs:domain/range`` are spliced into the prompt so the
    model extracts ontology-typed, direction-constrained triples (we exceed
    sift-kg's flat closed-vocab by sourcing it from the formal ontology and
    keeping a controlled-overflow escape for off-ontology facts). ``None`` keeps
    the unchanged free-vocab behaviour, so generic content never regresses.
    """
    if rounds < 1:
        rounds = 1
    # Ontology-guided extraction: prepend the schema block so the closed-vocab +
    # typed-relation guidance frames every round. Soft-closed (prompt prefers,
    # never forbids) to preserve recall on off-ontology facts.
    if schema is not None and not schema.is_empty:
        prompt = f"{schema.prompt_block()}\n{prompt}"
    if stream_fn is None:
        stream_fn = make_streaming_extract_fn()
    if dedup and deduper is None:
        deduper = FactDeduper(field=dedup_field, threshold=dedup_threshold)

    total_facts = 0
    total_dupes = 0

    for r in range(1, rounds + 1):
        seed = _seed_for_round(r)
        full_prompt = f"{prompt}\n\nDocument:\n  text: {text}"
        yield {"type": "round_start", "round": r, "seed": seed}

        buf = ""
        seen_hashes: set[int] = set()
        round_facts = 0
        round_dupes = 0
        token_count = 0

        async for delta in stream_fn(full_prompt, seed):
            buf += delta
            token_count += 1
            for raw in parse_facts_incremental(buf, seen_hashes):
                fact = _coerce_fact(raw, source_file)
                if fact is None:
                    continue
                is_dup = False
                max_sim = 0.0
                if dedup and deduper is not None:
                    is_dup, max_sim = deduper.check(fact)
                fact.is_duplicate = is_dup
                total_facts += 1
                round_facts += 1
                if is_dup:
                    total_dupes += 1
                    round_dupes += 1
                yield {
                    "type": "fact",
                    "round": r,
                    "fact": fact.model_dump(),
                    "is_duplicate": is_dup,
                    "max_similarity": round(max_sim, 4),
                }
            if token_count % 64 == 0:
                yield {"type": "metrics", "round": r, "tokens": token_count}

        yield {
            "type": "round_end",
            "round": r,
            "round_facts": round_facts,
            "round_dupes": round_dupes,
            "tokens": token_count,
        }

    yield {
        "type": "done",
        "total_facts": total_facts,
        "duplicate_facts": total_dupes,
        "unique_facts": total_facts - total_dupes,
    }


def _coerce_fact(raw: dict[str, Any], source_file: str) -> ExtractedFact | None:
    """Validate a raw LLM fact dict into an ``ExtractedFact`` (drop malformed)."""
    try:
        subject = str(raw.get("subject", "")).strip()
        obj = str(raw.get("object", "")).strip()
        predicate = str(raw.get("predicate", "")).strip()
        if not subject or not obj or not predicate:
            return None
        conf = raw.get("confidence", 0)
        try:
            conf = int(conf)
        except (TypeError, ValueError):
            conf = 0
        tags = raw.get("tags") or []
        if not isinstance(tags, list):
            tags = []
        return ExtractedFact(
            subject=subject,
            predicate=predicate,
            object=obj,
            title=str(raw.get("title", "")),
            description=str(raw.get("description", "")),
            evidence_span=str(raw.get("evidence_span", "")),
            confidence=max(0, min(100, conf)),
            tags=[str(t) for t in tags],
            source_file=source_file,
        )
    except Exception:  # noqa: BLE001 - malformed fact is skipped, not fatal
        return None


# --------------------------------------------------------------------------- #
# Persistence + export
# --------------------------------------------------------------------------- #


def aggregate_confidence(confidences: Iterable[float]) -> float:
    """Product-complement confidence ``1 − ∏(1 − cᵢ)`` over members (CONCEPT:AU-KG.ingest.observability-queries-opik-cannot).

    Each ``cᵢ`` is a 0..1 confidence. Independent weak mentions *reinforce*: two
    0.5 mentions combine to 0.75, three to 0.875 — corroboration raises the edge's
    confidence rather than averaging it down (sift-kg ``knowledge_graph.py:362``).
    """
    comp = 1.0
    for c in confidences:
        comp *= 1.0 - max(0.0, min(1.0, c))
    return 1.0 - comp


def persist_facts(store: Any, facts: Iterable[ExtractedFact]) -> dict[str, int]:
    """Write facts to the graph as ``subject -[predicate]-> object`` edges.

    Nodes are keyed by ``normalize_key`` so surface-form variants merge. Repeated
    mentions of the same ``(subject, predicate, object)`` triple are **merged into
    one edge** (CONCEPT:AU-KG.ingest.observability-queries-opik-cannot): the edge's ``confidence`` is the
    product-complement aggregate (corroboration reinforces), ``support_count`` is
    the number of mentions backing it, and ``weight`` is set to that count so
    well-supported edges rank above singletons — all populating fields the engine
    ``EdgeData`` already carries (``weight``/``confidence``). Aggregation is
    client-side over the in-batch facts (already resident), so it costs no extra
    engine round-trips. Duplicates (``is_duplicate``) are skipped.
    """
    nodes = 0
    seen_nodes: set[str] = set()
    groups: dict[tuple[str, str, str], list[ExtractedFact]] = {}
    order: list[tuple[str, str, str]] = []
    for f in facts:
        if f.is_duplicate:
            continue
        s_key = ExtractedFact.normalize_key(f.subject) or "?"
        o_key = ExtractedFact.normalize_key(f.object) or "?"
        for key, label in ((s_key, f.subject), (o_key, f.object)):
            if key not in seen_nodes:
                store.add_node(key, label=label)
                seen_nodes.add(key)
                nodes += 1
        edge_key = (s_key, f.predicate, o_key)
        if edge_key not in groups:
            groups[edge_key] = []
            order.append(edge_key)
        groups[edge_key].append(f)

    edges = 0
    for edge_key in order:
        members = groups[edge_key]
        s_key, predicate, o_key = edge_key
        support_count = len(members)
        agg_conf = aggregate_confidence(m.confidence / 100.0 for m in members)
        # Representative narrative fields come from the highest-confidence mention;
        # tags union; sources deduped (support_documents = distinct sources).
        rep = max(members, key=lambda m: m.confidence)
        tags = sorted({t for m in members for t in (m.tags or [])})
        sources = sorted({m.source_file for m in members if m.source_file})
        store.add_edge(
            s_key,
            o_key,
            rel_type=predicate,
            confidence=agg_conf,
            weight=float(support_count),
            support_count=support_count,
            support_documents=len(sources),
            evidence_span=rep.evidence_span,
            title=rep.title,
            description=rep.description,
            tags=tags,
            source_file=",".join(sources),
            provenance="fact_extractor",
        )
        edges += 1
    return {"nodes": nodes, "edges": edges}


def facts_to_jsonl(facts: Iterable[ExtractedFact]) -> str:
    """Serialize facts to newline-delimited JSON (upstream ``facts.jsonl`` parity)."""
    return "\n".join(
        json.dumps(f.model_dump(), ensure_ascii=False, separators=(",", ":"))
        for f in facts
    )
