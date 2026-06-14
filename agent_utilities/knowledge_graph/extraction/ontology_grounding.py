"""Ontology grounding for extracted facts (CONCEPT:KG-2.74).

The last hop of the universal multi-modal ingestion funnel:

    reader -> structure-router -> {open | schema} extraction
        -> *ontology grounding*  -> background OWL-RL/SHACL closure

A fact extractor (``extraction.fact_extractor``) emits canonical-entity triples
``(subject) --[predicate]--> (object)`` whose ``subject``/``object`` are short
surface strings ("a vendor on a bill", "a supplier in CMDB", "a company in a
post"). Those three name the *same* real-world kind, yet land as three
unrelated nodes unless something maps each surface form onto a shared OWL class.
This module is that something.

``ground_fact(subject, predicate, obj)`` returns
``{subject_type, object_type, predicate}`` where each ``*_type`` is the canonical
OWL class (a :class:`~agent_utilities.models.knowledge_graph.RegistryNodeType`
value such as ``"organization"``/``"person"``/``"place"``) the entity grounds
onto -- so a *vendor*, a *supplier* and a *company* all converge on
``organization`` and connect in the graph. Grounding reuses the existing
ontology fabric rather than reinventing it:

* **Value types** (``ontology.value_types`` -- KG-2.39): an atomic object that
  *coerces* through a built-in value type (``ISOCurrencyCode``, ``Percentage``,
  ``EmailAddress``, ``URL``, ``E164PhoneNumber``, ``Probability``) is grounded
  to that value type's name instead of a node class, and the coerced/normalized
  value is returned -- so ``"USD"`` on a bill grounds to ``ISOCurrencyCode`` and
  a literal e-mail to ``EmailAddress``. Uses ``coerce_value_type`` /
  ``validate_value_type``.
* **Interfaces** (``ontology.interfaces`` -- KG-2.38): when a candidate class is
  named where a target *interface* is expected, ``InterfaceRegistry`` resolves
  it -- a fact whose object dict ``conforms`` to ``Locatable`` is annotated as
  ``Locatable`` (the abstract shared shape), and ``resolve_target`` expands an
  interface name to its concrete implementers. This is what lets cross-modal
  entities sharing a shape unify even when their concrete node types differ.
* **Node classes**: a synonym lexicon maps common surface words
  ("vendor"->organization, "supplier"->organization, "company"->organization,
  "buyer"->person, "city"->place, "invoice"->document, "bank account"->account,
  ...) onto the live ``RegistryNodeType`` vocabulary, so heterogeneous extractors
  converge on the same OWL class.

Everything here is **best-effort and never raises** -- it runs inside the
best-effort ingest enrichment path, so a grounding miss degrades to a ``None``
type (the fact is still persisted, just un-typed) rather than breaking ingest.

Provenance: the value-type/interface registries are ours (KG-2.38/2.39); this
module is the convergence layer that binds extracted surface forms to them.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Iterable, Mapping
from typing import Any

logger = logging.getLogger(__name__)

# Node/edge property keys the grounding writes onto persisted facts. Kept as
# module constants (config discipline: one correct value, not an env knob) so
# the persistence wiring and any reader agree on the names.
SUBJECT_TYPE_KEY = "ontology_type"
OBJECT_TYPE_KEY = "ontology_type"
EDGE_SUBJECT_TYPE_KEY = "subject_type"
EDGE_OBJECT_TYPE_KEY = "object_type"


def _norm(text: str) -> str:
    """NFKC-compose, lowercase, collapse whitespace, strip wrapping punctuation.

    Mirrors ``ExtractedFact.normalize_key`` so a surface form grounds the same
    way it is keyed in the graph.
    """
    s = unicodedata.normalize("NFKC", text or "").lower()
    s = re.sub(r"[\s ]+", " ", s)
    s = re.sub(r"^[\s\"'`([{]+|[\s\"'`)\]}.,;:!?]+$", "", s)
    return s.strip()


# --------------------------------------------------------------------------- #
# Node-class synonym lexicon -- cross-modal surface form -> canonical OWL class.
#
# Values are RegistryNodeType *values* (lowercase strings). Resolved lazily and
# validated against the live enum at import so a typo can't ground onto a
# non-existent class. This is the convergence table: a "vendor" on a bill, a
# "supplier" in CMDB and a "company" in a post all map to ``organization``.
# --------------------------------------------------------------------------- #
_RAW_CLASS_SYNONYMS: dict[str, str] = {
    # organizations / companies / suppliers (the headline convergence)
    "organization": "organization",
    "organisation": "organization",
    "org": "organization",
    "company": "organization",
    "corporation": "organization",
    "corp": "organization",
    "business": "organization",
    "firm": "organization",
    "vendor": "organization",
    "supplier": "organization",
    "seller": "organization",
    "merchant": "organization",
    "manufacturer": "organization",
    "institution": "organization",
    "agency": "organization",
    "team": "organization",
    "employer": "organization",
    # people
    "person": "person",
    "people": "person",
    "individual": "person",
    "human": "person",
    "buyer": "person",
    "customer": "person",
    "client": "person",
    "employee": "person",
    "author": "person",
    "founder": "person",
    "contact": "person",
    "user": "person",
    # places
    "place": "place",
    "location": "place",
    "city": "place",
    "town": "place",
    "country": "place",
    "region": "place",
    "address": "place",
    "site": "place",
    "venue": "place",
    # documents / records
    "document": "document",
    "doc": "document",
    "invoice": "document",
    "bill": "document",
    "receipt": "document",
    "report": "document",
    "article": "article",
    "paper": "document",
    "contract": "document",
    "record": "document",
    # finance
    "account": "account",
    "bank account": "account",
    "financial instrument": "financial_instrument",
    "instrument": "financial_instrument",
    "security": "financial_instrument",
    "stock": "financial_instrument",
    "transaction": "financial_transaction",
    "payment": "financial_transaction",
    # datasets / software / events
    "dataset": "dataset",
    "data set": "dataset",
    "software": "software_project",
    "software project": "software_project",
    "project": "software_project",
    "product": "software_project",
    "model": "software_project",
    "event": "event",
    "incident": "incident",
    "role": "role",
    "system": "system",
    "regulation": "regulation",
    "procedure": "procedure",
    "policy": "policy",
}

# Word-level hints scanned when no whole-string lexicon hit lands (e.g. a label
# like "Acme Corp Ltd"). Ordered most-specific first.
_CLASS_WORD_HINTS: tuple[tuple[str, str], ...] = (
    ("inc", "organization"),
    ("llc", "organization"),
    ("ltd", "organization"),
    ("gmbh", "organization"),
    ("corp", "organization"),
    ("co", "organization"),
    ("plc", "organization"),
    ("university", "organization"),
    ("institute", "organization"),
    ("bank", "organization"),
)


def _build_class_synonyms() -> dict[str, str]:
    """Validate the raw lexicon against the live ``RegistryNodeType`` vocabulary.

    Any target that is not an actual node-type value is dropped (with a warning)
    so grounding can only ever emit a real OWL class. Import-time, best-effort:
    if the enum can't be imported we keep the raw map (string values are still
    meaningful as types).
    """
    try:
        from ...models.knowledge_graph import RegistryNodeType

        valid = {t.value for t in RegistryNodeType}
    except Exception:  # noqa: BLE001 - never fail import over a typing lookup
        return dict(_RAW_CLASS_SYNONYMS)
    resolved: dict[str, str] = {}
    for surface, target in _RAW_CLASS_SYNONYMS.items():
        if target in valid:
            resolved[surface] = target
        else:  # pragma: no cover - guards against a future enum rename
            logger.debug("grounding lexicon target %r not a node type; skipped", target)
    return resolved


CLASS_SYNONYMS: dict[str, str] = _build_class_synonyms()

# Value types tried (in order) when grounding an atomic object literal. Names
# index ``ontology.value_types.VALUE_TYPES``; the first whose ``validate``
# passes wins. Currency/percentage first because they are the most discriminating
# on a bill/financial document.
_VALUE_TYPE_ORDER: tuple[str, ...] = (
    "EmailAddress",
    "URL",
    "E164PhoneNumber",
    "ISOCurrencyCode",
)


def _ground_value_type(literal: str) -> tuple[str, Any] | None:
    """Try to ground an atomic ``literal`` onto a built-in value type.

    Returns ``(value_type_name, coerced_value)`` on the first match, else None.
    Best-effort: a missing value_types module or any coercion error -> None.
    """
    if not literal:
        return None
    try:
        from ..ontology.value_types import coerce_value_type, validate_value_type
    except Exception:  # noqa: BLE001
        return None
    for name in _VALUE_TYPE_ORDER:
        try:
            if validate_value_type(name, literal):
                return name, coerce_value_type(name, literal)
        except Exception:  # noqa: BLE001 - a bad value type never breaks grounding
            continue
    return None


def _ground_node_class(surface: str) -> str | None:
    """Map a surface form onto a canonical OWL node class, or None.

    Tries: exact normalized lexicon hit -> head-noun lexicon hit (last word of
    a phrase, e.g. "acme corp" -> "corp") -> word-level company-suffix hints.
    """
    norm = _norm(surface)
    if not norm:
        return None
    hit = CLASS_SYNONYMS.get(norm)
    if hit:
        return hit
    words = norm.split(" ")
    if len(words) > 1:
        # Head noun of a short noun phrase ("purchase order" -> "order").
        head = CLASS_SYNONYMS.get(words[-1])
        if head:
            return head
    word_set = {w.strip(".") for w in words}
    for token, cls in _CLASS_WORD_HINTS:
        if token in word_set:
            return cls
    return None


def _ground_interface(
    surface: str,
    object_dict: Mapping[str, Any] | None,
) -> str | None:
    """Ground onto an abstract interface (KG-2.38) when the entity conforms.

    Two paths:

    * ``surface`` *names* an interface (e.g. "Locatable") -> return that
      interface name (callers expand it via ``resolve_target`` if they need the
      concrete implementers).
    * an ``object_dict`` of properties/links ``conforms`` to a registered
      interface's shared shape -> return that interface name, so cross-modal
      entities sharing a shape (lat/lon = ``Locatable``) unify.

    Best-effort: a missing interfaces module or any registry error -> None.
    """
    try:
        from ..ontology.interfaces import DEFAULT_INTERFACE_REGISTRY
    except Exception:  # noqa: BLE001
        return None
    reg = DEFAULT_INTERFACE_REGISTRY
    norm = _norm(surface)
    # Direct interface-name match (case-insensitive against registered names).
    for iface in reg.list_interfaces():
        if iface.name.lower() == norm:
            # resolve_target proves the name is registered and routable.
            try:
                reg.resolve_target(iface.name)
            except Exception:  # noqa: BLE001
                pass
            return iface.name
    # Shape conformance: does the supplied object dict satisfy any interface?
    if object_dict:
        for iface in reg.list_interfaces():
            try:
                if reg.conforms(dict(object_dict), iface.name):
                    return iface.name
            except Exception:  # noqa: BLE001
                continue
    return None


def ground_fact(
    subject: str,
    predicate: str,
    obj: str,
    *,
    subject_props: Mapping[str, Any] | None = None,
    object_props: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Ground one extracted triple onto the OWL ontology (CONCEPT:KG-2.74).

    Maps the surface ``subject``/``object`` onto canonical OWL types so
    cross-modal entities converge: a *vendor*, *supplier* and *company* all
    ground to ``organization``; an atomic object coerces to a value type
    (``"USD"`` -> ``ISOCurrencyCode``, an e-mail -> ``EmailAddress``).

    Resolution order per endpoint:
      1. interface shape/name (KG-2.38) -- the most abstract shared contract;
      2. node-class synonym lexicon -- the canonical concrete OWL class;
      3. (object only) value-type coercion (KG-2.39) for atomic literals.

    Args:
        subject: The triple subject surface form.
        predicate: The triple predicate (returned unchanged; kept for symmetry
            and future predicate-canonicalization).
        obj: The triple object surface form.
        subject_props: Optional declared properties/links of the subject (an
            object dict) used for interface conformance.
        object_props: Optional declared properties/links of the object.

    Returns:
        A dict with ``subject_type``, ``object_type`` (each a canonical OWL
        class / value-type / interface name, or ``None`` when unresolved),
        ``predicate`` (unchanged), and -- when the object grounded to a value
        type -- ``object_value`` (the coerced literal). Never raises.
    """
    result: dict[str, Any] = {
        "subject_type": None,
        "object_type": None,
        "predicate": predicate,
    }
    try:
        # --- subject: interface -> node class -----------------------------
        s_type = _ground_interface(subject, subject_props) or _ground_node_class(
            subject
        )
        result["subject_type"] = s_type

        # --- object: interface -> node class -> value type ----------------
        o_type = _ground_interface(obj, object_props) or _ground_node_class(obj)
        if o_type is None:
            vt = _ground_value_type(obj)
            if vt is not None:
                o_type, coerced = vt
                result["object_value"] = coerced
        result["object_type"] = o_type
    except Exception:  # noqa: BLE001 - grounding is best-effort, never fatal
        logger.debug("ground_fact failed for (%r, %r, %r)", subject, predicate, obj)
    return result


def ground_facts(facts: Iterable[Any]) -> list[Any]:
    """Annotate a batch of extracted facts with ontology types (best-effort).

    Accepts ``ExtractedFact`` instances (or any object/dict carrying
    ``subject``/``predicate``/``object``). For each fact it computes
    :func:`ground_fact` and attaches the grounding back onto the fact so the
    persistence layer can write the types as node/edge properties:

    * ``ExtractedFact`` (a pydantic model) -- the grounding is stored in the
      fact's ``tags`` is **not** mutated; instead the returned list contains
      ``(fact, grounding)`` only when called with the dict form. To keep the
      common case ergonomic, when a fact exposes a settable attribute namespace
      we stash the grounding under ``ontology`` via ``object.__setattr__`` on a
      shallow copy when possible; otherwise the fact is returned unchanged and
      the caller reads :func:`ground_fact` directly.

    The function never raises and always returns a list the same length as the
    input, each element being a ``(fact, grounding_dict)`` tuple so the wiring
    layer has both the original fact and its grounding without guessing where it
    was stashed.
    """
    out: list[Any] = []
    for fact in facts:
        grounding: dict[str, Any]
        try:
            subject = _attr(fact, "subject")
            predicate = _attr(fact, "predicate")
            obj = _attr(fact, "object")
            subject_props = _attr(fact, "subject_props", default=None)
            object_props = _attr(fact, "object_props", default=None)
            grounding = ground_fact(
                subject,
                predicate,
                obj,
                subject_props=subject_props,
                object_props=object_props,
            )
        except Exception:  # noqa: BLE001
            grounding = {
                "subject_type": None,
                "object_type": None,
                "predicate": _attr(fact, "predicate", default=""),
            }
        out.append((fact, grounding))
    return out


def _attr(obj: Any, name: str, *, default: Any = "") -> Any:
    """Read ``name`` from a pydantic model / object / mapping, defaulting safely."""
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


__all__ = [
    "ground_fact",
    "ground_facts",
    "CLASS_SYNONYMS",
    "SUBJECT_TYPE_KEY",
    "OBJECT_TYPE_KEY",
    "EDGE_SUBJECT_TYPE_KEY",
    "EDGE_OBJECT_TYPE_KEY",
]
