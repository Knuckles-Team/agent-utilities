#!/usr/bin/python
from __future__ import annotations

"""Ontology PropertyType registry — typed data primitives for ontology objects.

CONCEPT:KG-2.47 — Ontology Property Types.

Palantir Foundry doc matched: *object-link-types / type-reference* and the
ontology *data-types* page, whose property base/complex type set is documented
as "inspired by RDF, OWL and XSD". This module ports that full type vocabulary
into agent-utilities so that an ontology property declared as e.g. ``geohash``,
``timeseries`` or ``vector`` carries three coupled facts everywhere it is used:

1.  a **Python/Pydantic validator** (real :meth:`PropertyType.coerce` /
    :meth:`PropertyType.validate` logic — never a pass-through),
2.  an **XSD/OWL datatype IRI** so the property serialises correctly when the
    LPG is promoted to RDF/OWL (the ``owl_bridge`` reasoning substrate), and
3.  a **storage hint** that maps the type onto the existing
    ``agent_utilities.models.schema_definition`` column-type vocabulary
    (the Ladybug/epistemic-graph type strings such as ``STRING``,
    ``DOUBLE``, ``INT64``, ``TIMESTAMP``, ``STRING[]``, ``FLOAT[768]``).

Covered base types (Palantir base property types): ``string``, ``boolean``,
``byte``, ``short``, ``integer``, ``long``, ``float``, ``double``, ``decimal``,
``date``, ``timestamp``. Geo types: ``geohash``, ``geoshape``,
``geotimeseries``. Series: ``timeseries``. Complex/reference types:
``array<T>``, ``struct``, ``attachment``, ``media_reference``, ``marking``,
and the dimension-parameterized ``vector`` / ``embedding`` (default dim ties
to ``create_embedding_model``'s 768-dim default).

The module follows the import-populated-registry idiom: :data:`PROPERTY_TYPES`
is populated at import with the full real built-in type set (never an empty
shell), and lookup helpers (:func:`get_property_type`, :func:`parse_type_ref`)
resolve a Palantir-style type reference — including parameterized forms like
``array<string>`` and ``vector<1536>`` — back to a concrete
:class:`PropertyType`.
"""

import base64
import datetime as _dt
import json
import re
from collections.abc import Callable
from decimal import Decimal, InvalidOperation
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent_utilities.core.config import config

# XSD / OWL namespaces (match those bound in the ontology *.ttl files).
XSD = "http://www.w3.org/2001/XMLSchema#"
KG = "http://knuckles.team/kg#"
GEO = "http://www.opengis.net/ont/geosparql#"

# Default embedding dimensionality — ties to create_embedding_model()'s 768
# default and config.kg_embedding_dim (CONCEPT:KG-2.47).
try:
    DEFAULT_VECTOR_DIM: int = int(config.kg_embedding_dim or "768")
except (TypeError, ValueError):  # pragma: no cover - defensive
    DEFAULT_VECTOR_DIM = 768


# ---------------------------------------------------------------------------
# Storage-hint vocabulary (the existing schema_definition column types)
# ---------------------------------------------------------------------------
# These are the exact Ladybug/epistemic-graph column-type *strings* used in
# agent_utilities/models/schema_definition.py (e.g. "STRING", "DOUBLE",
# "INT64", "TIMESTAMP", "STRING[]", "FLOAT[768]"). Storing them as plain
# strings keeps PropertyType decoupled from any one backend while remaining
# byte-for-byte compatible with TableDefinition.columns.


class PropertyType(BaseModel):
    """A single ontology data-type with validation, OWL IRI and storage hint.

    CONCEPT:KG-2.47 — Ontology Property Types.

    Palantir doc matched: ontology *data-types* ("inspired by RDF, OWL and
    XSD"). Each instance binds a logical property type name to (a) a Python
    runtime type, (b) an XSD/OWL datatype IRI for RDF promotion, and (c) a
    storage hint string drawn from the ``schema_definition`` column vocabulary.

    Attributes:
        name: Logical type name (Palantir base/complex type, e.g. ``geohash``).
        xsd_iri: The XSD or OWL datatype IRI the type serialises to.
        python_type: The canonical Python type a coerced value takes.
        storage_hint: A ``schema_definition`` column-type string
            (e.g. ``STRING``, ``DOUBLE``, ``INT64``, ``TIMESTAMP``).
        is_complex: True for non-scalar types (array, struct, vector, …).
        element_type: For ``array<T>`` / ``vector`` the contained scalar type.
        dimension: For ``vector`` / ``embedding`` the fixed dimensionality.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=False)

    name: str
    xsd_iri: str
    python_type: type
    storage_hint: str
    is_complex: bool = False
    element_type: str | None = None
    dimension: int | None = None
    description: str = ""

    # The coercion function is attached out-of-band (not a pydantic field) so it
    # can be a real closure with backend-agnostic logic.
    def coerce(self, value: Any) -> Any:
        """Coerce ``value`` into this type's canonical Python representation.

        Raises:
            ValueError: if ``value`` cannot be coerced.
        """
        fn = _COERCERS.get(self.name)
        if fn is None:
            # Parameterized (array<T>/vector) types resolve by family.
            if self.is_complex and self.element_type is not None and self.name.startswith("array"):
                return _coerce_array(value, self.element_type)
            if self.name in ("vector", "embedding") or self.name.startswith("vector"):
                return _coerce_vector(value, self.dimension or DEFAULT_VECTOR_DIM)
            raise ValueError(f"No coercer registered for property type {self.name!r}")
        return fn(value)

    def validate(self, value: Any) -> bool:
        """Return True iff ``value`` is (or can be coerced to) this type."""
        try:
            self.coerce(value)
            return True
        except (ValueError, TypeError, InvalidOperation):
            return False


# ---------------------------------------------------------------------------
# Scalar coercers — real, total logic for every base type
# ---------------------------------------------------------------------------
def _coerce_string(value: Any) -> str:
    if value is None:
        raise ValueError("string property cannot be None")
    return value if isinstance(value, str) else str(value)


_TRUE = {"true", "1", "yes", "y", "t", "on"}
_FALSE = {"false", "0", "no", "n", "f", "off"}


def _coerce_boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        low = value.strip().lower()
        if low in _TRUE:
            return True
        if low in _FALSE:
            return False
    raise ValueError(f"cannot coerce {value!r} to boolean")


def _bounded_int(bits: int, signed: bool = True) -> Callable[[Any], int]:
    if signed:
        lo, hi = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    else:
        lo, hi = 0, 2**bits - 1

    def _coerce(value: Any) -> int:
        if isinstance(value, bool):
            raise ValueError("bool is not a valid integer property")
        if isinstance(value, int):
            iv = value
        elif isinstance(value, float):
            if not value.is_integer():
                raise ValueError(f"{value!r} is not an integer")
            iv = int(value)
        elif isinstance(value, str):
            iv = int(value.strip())
        else:
            raise ValueError(f"cannot coerce {value!r} to integer")
        if not (lo <= iv <= hi):
            raise ValueError(f"{iv} out of range for {bits}-bit integer [{lo},{hi}]")
        return iv

    return _coerce


def _coerce_float(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("bool is not a valid float property")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value.strip())
    raise ValueError(f"cannot coerce {value!r} to float")


def _coerce_decimal(value: Any) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool):
        raise ValueError("bool is not a valid decimal property")
    if isinstance(value, (int, float, str)):
        return Decimal(str(value).strip())
    raise ValueError(f"cannot coerce {value!r} to decimal")


def _coerce_date(value: Any) -> _dt.date:
    if isinstance(value, _dt.datetime):
        return value.date()
    if isinstance(value, _dt.date):
        return value
    if isinstance(value, str):
        return _dt.date.fromisoformat(value.strip())
    raise ValueError(f"cannot coerce {value!r} to date")


def _coerce_timestamp(value: Any) -> _dt.datetime:
    if isinstance(value, _dt.datetime):
        return value if value.tzinfo else value.replace(tzinfo=_dt.UTC)
    if isinstance(value, _dt.date):
        return _dt.datetime(value.year, value.month, value.day, tzinfo=_dt.UTC)
    if isinstance(value, (int, float)):
        return _dt.datetime.fromtimestamp(value, tz=_dt.UTC)
    if isinstance(value, str):
        ts = _dt.datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        return ts if ts.tzinfo else ts.replace(tzinfo=_dt.UTC)
    raise ValueError(f"cannot coerce {value!r} to timestamp")


# ---------------------------------------------------------------------------
# Geo coercers (Palantir geo property types)
# ---------------------------------------------------------------------------
_GEOHASH_RE = re.compile(r"^[0123456789bcdefghjkmnpqrstuvwxyz]+$")


def _coerce_geohash(value: Any) -> str:
    """A base-32 geohash string (Palantir ``geohash``)."""
    if not isinstance(value, str):
        raise ValueError(f"geohash must be a string, got {type(value).__name__}")
    s = value.strip().lower()
    if not s or not _GEOHASH_RE.match(s):
        raise ValueError(f"{value!r} is not a valid base-32 geohash")
    return s


def _coerce_geoshape(value: Any) -> dict[str, Any]:
    """A GeoJSON geometry object (Palantir ``geoshape``)."""
    geo = value
    if isinstance(value, str):
        geo = json.loads(value)
    if not isinstance(geo, dict):
        raise ValueError("geoshape must be a GeoJSON object/dict")
    gtype = geo.get("type")
    valid = {
        "Point",
        "LineString",
        "Polygon",
        "MultiPoint",
        "MultiLineString",
        "MultiPolygon",
        "GeometryCollection",
    }
    if gtype not in valid:
        raise ValueError(f"geoshape has invalid GeoJSON type {gtype!r}")
    if gtype == "GeometryCollection":
        if not isinstance(geo.get("geometries"), list):
            raise ValueError("GeometryCollection requires a 'geometries' list")
    elif "coordinates" not in geo:
        raise ValueError("geoshape GeoJSON requires 'coordinates'")
    return geo


def _coerce_geo_point(value: Any) -> dict[str, float]:
    """A {lat, lon} point (Palantir geo point under geoshape family)."""
    pt = value
    if isinstance(value, str):
        pt = json.loads(value)
    if isinstance(pt, (list, tuple)) and len(pt) == 2:
        lat, lon = float(pt[0]), float(pt[1])
    elif isinstance(pt, dict):
        lat = float(pt.get("lat", pt.get("latitude")))
        lon = float(pt.get("lon", pt.get("lng", pt.get("longitude"))))
    else:
        raise ValueError(f"cannot coerce {value!r} to geo point")
    if not (-90.0 <= lat <= 90.0):
        raise ValueError(f"latitude {lat} out of range [-90, 90]")
    if not (-180.0 <= lon <= 180.0):
        raise ValueError(f"longitude {lon} out of range [-180, 180]")
    return {"lat": lat, "lon": lon}


def _coerce_timeseries(value: Any) -> dict[str, Any]:
    """A numeric time-series reference/series (Palantir ``timeseries``).

    Accepts either a sync/reference id string (the Foundry series-reference
    form) or a concrete ``[[ts, value], …]`` / ``{"points": …}`` series; the
    points are validated to be (timestamp, number) pairs.
    """
    if isinstance(value, str):
        s = value.strip()
        if not s:
            raise ValueError("timeseries reference cannot be empty")
        return {"series_id": s, "points": []}
    points: Any
    if isinstance(value, dict):
        points = value.get("points", [])
        series_id = value.get("series_id")
    elif isinstance(value, (list, tuple)):
        points = value
        series_id = None
    else:
        raise ValueError(f"cannot coerce {value!r} to timeseries")
    norm: list[list[Any]] = []
    for p in points:
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            raise ValueError("timeseries point must be a (timestamp, value) pair")
        ts = _coerce_timestamp(p[0])
        val = _coerce_float(p[1])
        norm.append([ts.isoformat(), val])
    out: dict[str, Any] = {"points": norm}
    if series_id is not None:
        out["series_id"] = str(series_id)
    return out


def _coerce_geotimeseries(value: Any) -> dict[str, Any]:
    """A geo-tagged numeric series (Palantir ``geotimeseries``).

    Each point is ``(timestamp, geo-point)``; reuses the geo-point and
    timestamp coercers so range checks apply.
    """
    if isinstance(value, str):
        s = value.strip()
        if not s:
            raise ValueError("geotimeseries reference cannot be empty")
        return {"series_id": s, "points": []}
    if isinstance(value, dict):
        points = value.get("points", [])
        series_id = value.get("series_id")
    elif isinstance(value, (list, tuple)):
        points = value
        series_id = None
    else:
        raise ValueError(f"cannot coerce {value!r} to geotimeseries")
    norm: list[dict[str, Any]] = []
    for p in points:
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            raise ValueError("geotimeseries point must be (timestamp, geo-point)")
        ts = _coerce_timestamp(p[0])
        geo = _coerce_geo_point(p[1])
        norm.append({"t": ts.isoformat(), "geo": geo})
    out: dict[str, Any] = {"points": norm}
    if series_id is not None:
        out["series_id"] = str(series_id)
    return out


# ---------------------------------------------------------------------------
# Complex / reference coercers
# ---------------------------------------------------------------------------
def _coerce_struct(value: Any) -> dict[str, Any]:
    """A nested object (Palantir ``struct``)."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    if isinstance(value, BaseModel):
        return value.model_dump()
    raise ValueError(f"cannot coerce {value!r} to struct (object)")


def _coerce_attachment(value: Any) -> dict[str, Any]:
    """A file attachment reference (Palantir ``attachment``).

    Normalises to ``{rid, filename, media_type?, size?}``. ``rid`` (the
    Foundry resource id) is required.
    """
    att = value
    if isinstance(value, str):
        att = {"rid": value}
    if not isinstance(att, dict):
        raise ValueError(f"cannot coerce {value!r} to attachment")
    rid = att.get("rid") or att.get("attachment_rid") or att.get("id")
    if not rid:
        raise ValueError("attachment requires a resource id ('rid')")
    out: dict[str, Any] = {"rid": str(rid)}
    if att.get("filename"):
        out["filename"] = str(att["filename"])
    if att.get("media_type") or att.get("mediaType"):
        out["media_type"] = str(att.get("media_type") or att.get("mediaType"))
    if att.get("size") is not None:
        out["size"] = int(att["size"])
    return out


def _coerce_media_reference(value: Any) -> dict[str, Any]:
    """A media-set reference (Palantir ``media_reference``).

    Normalises to ``{media_set_rid, media_item_rid, media_type?}``.
    """
    ref = value
    if isinstance(value, str):
        try:
            ref = json.loads(value)
        except ValueError:
            ref = {"media_item_rid": value}
    if not isinstance(ref, dict):
        raise ValueError(f"cannot coerce {value!r} to media_reference")
    # Palantir nests under "reference"; accept flat or nested.
    inner = ref.get("reference", ref)
    item = (
        inner.get("media_item_rid")
        or inner.get("mediaItemRid")
        or inner.get("rid")
        or inner.get("id")
    )
    if not item:
        raise ValueError("media_reference requires a media item rid")
    out: dict[str, Any] = {"media_item_rid": str(item)}
    media_set = inner.get("media_set_rid") or inner.get("mediaSetRid")
    if media_set:
        out["media_set_rid"] = str(media_set)
    media_type = inner.get("media_type") or inner.get("mimeType") or inner.get("mediaType")
    if media_type:
        out["media_type"] = str(media_type)
    return out


def _coerce_marking(value: Any) -> dict[str, Any]:
    """A security marking / classification (Palantir ``marking``).

    A marking is a set of marking ids gating row/column access. Normalises to
    ``{marking_ids: [...]}`` — ties into the platform's security model and the
    ``permissions_kernel`` / ``secured_reads`` ACL fabric (CONCEPT:KG-2.47).
    """
    ids: list[str]
    if isinstance(value, str):
        ids = [value.strip()] if value.strip() else []
    elif isinstance(value, dict):
        raw = value.get("marking_ids", value.get("markings", []))
        ids = [str(x) for x in raw] if isinstance(raw, (list, tuple)) else []
        single = value.get("marking_id") or value.get("id")
        if single:
            ids.append(str(single))
    elif isinstance(value, (list, tuple, set)):
        ids = [str(x) for x in value]
    else:
        raise ValueError(f"cannot coerce {value!r} to marking")
    # Deduplicate, preserve order, drop empties.
    seen: dict[str, None] = {}
    for x in ids:
        if x:
            seen.setdefault(x, None)
    if not seen:
        raise ValueError("marking requires at least one marking id")
    return {"marking_ids": list(seen.keys())}


def _coerce_bytes(value: Any) -> bytes:
    """Raw binary content (used by the byte storage of attachments/markings)."""
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, str):
        try:
            return base64.b64decode(value, validate=True)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"bytes value must be base64: {exc}") from exc
    raise ValueError(f"cannot coerce {value!r} to bytes")


# ---------------------------------------------------------------------------
# Parameterized complex coercers (array<T>, vector<dim>)
# ---------------------------------------------------------------------------
def _coerce_array(value: Any, element_type: str) -> list[Any]:
    """Coerce a homogeneous array, validating each element as ``element_type``."""
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError as exc:
            raise ValueError(f"array string must be JSON: {exc}") from exc
        value = parsed
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"cannot coerce {value!r} to array")
    elem_pt = get_property_type(element_type)
    if elem_pt is None:
        raise ValueError(f"unknown array element type {element_type!r}")
    return [elem_pt.coerce(v) for v in value]


def _coerce_vector(value: Any, dim: int) -> list[float]:
    """Coerce a fixed-dim float vector / embedding (Palantir ``vector``).

    Validates the dimensionality against ``dim`` (default 768, matching
    ``create_embedding_model``). Accepts list/tuple, JSON string, or numpy
    array (duck-typed via ``tolist``).
    """
    if hasattr(value, "tolist") and not isinstance(value, (list, tuple)):
        value = value.tolist()
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"cannot coerce {value!r} to vector")
    vec = [_coerce_float(v) for v in value]
    if dim and len(vec) != dim:
        raise ValueError(f"vector has {len(vec)} dims, expected {dim}")
    return vec


# Registry of scalar/non-parameterized coercers keyed by type name.
_COERCERS: dict[str, Callable[[Any], Any]] = {
    "string": _coerce_string,
    "boolean": _coerce_boolean,
    "byte": _bounded_int(8),
    "short": _bounded_int(16),
    "integer": _bounded_int(32),
    "long": _bounded_int(64),
    "float": _coerce_float,
    "double": _coerce_float,
    "decimal": _coerce_decimal,
    "date": _coerce_date,
    "timestamp": _coerce_timestamp,
    "geohash": _coerce_geohash,
    "geoshape": _coerce_geoshape,
    "geo_point": _coerce_geo_point,
    "timeseries": _coerce_timeseries,
    "geotimeseries": _coerce_geotimeseries,
    "struct": _coerce_struct,
    "attachment": _coerce_attachment,
    "media_reference": _coerce_media_reference,
    "marking": _coerce_marking,
    "bytes": _coerce_bytes,
}


# ---------------------------------------------------------------------------
# Built-in PropertyType registry (populated at import — never an empty shell)
# ---------------------------------------------------------------------------
def _pt(
    name: str,
    xsd_iri: str,
    python_type: type,
    storage_hint: str,
    *,
    is_complex: bool = False,
    element_type: str | None = None,
    dimension: int | None = None,
    description: str = "",
) -> PropertyType:
    return PropertyType(
        name=name,
        xsd_iri=xsd_iri,
        python_type=python_type,
        storage_hint=storage_hint,
        is_complex=is_complex,
        element_type=element_type,
        dimension=dimension,
        description=description,
    )


_VECTOR_STORAGE = f"FLOAT[{DEFAULT_VECTOR_DIM}]"

PROPERTY_TYPES: dict[str, PropertyType] = {
    # --- Base scalar types (Palantir base property types) ---
    "string": _pt("string", XSD + "string", str, "STRING", description="UTF-8 text."),
    "boolean": _pt("boolean", XSD + "boolean", bool, "BOOLEAN"),
    "byte": _pt("byte", XSD + "byte", int, "INT64", description="8-bit signed integer."),
    "short": _pt("short", XSD + "short", int, "INT64", description="16-bit signed integer."),
    "integer": _pt("integer", XSD + "int", int, "INT64", description="32-bit signed integer."),
    "long": _pt("long", XSD + "long", int, "INT64", description="64-bit signed integer."),
    "float": _pt("float", XSD + "float", float, "FLOAT"),
    "double": _pt("double", XSD + "double", float, "DOUBLE"),
    "decimal": _pt("decimal", XSD + "decimal", Decimal, "STRING", description="Exact decimal; stored as string to avoid float drift."),
    "date": _pt("date", XSD + "date", _dt.date, "STRING", description="ISO-8601 calendar date."),
    "timestamp": _pt("timestamp", XSD + "dateTime", _dt.datetime, "TIMESTAMP"),
    # --- Geo types ---
    "geohash": _pt("geohash", GEO + "asGeoHash", str, "STRING", description="Base-32 geohash."),
    "geoshape": _pt("geoshape", GEO + "wktLiteral", dict, "STRING", is_complex=True, description="GeoJSON geometry serialised as JSON."),
    "geo_point": _pt("geo_point", GEO + "Point", dict, "STRING", is_complex=True, description="{lat, lon} point."),
    "timeseries": _pt("timeseries", KG + "TimeSeries", dict, "STRING", is_complex=True, description="Numeric time-series reference/series."),
    "geotimeseries": _pt("geotimeseries", KG + "GeoTimeSeries", dict, "STRING", is_complex=True, description="Geo-tagged numeric time-series."),
    # --- Complex / reference types ---
    "struct": _pt("struct", KG + "Struct", dict, "STRING", is_complex=True, description="Nested object serialised as JSON."),
    "attachment": _pt("attachment", KG + "Attachment", dict, "STRING", is_complex=True, description="File attachment reference (rid)."),
    "media_reference": _pt("media_reference", KG + "MediaReference", dict, "STRING", is_complex=True, description="Media-set item reference."),
    "marking": _pt("marking", KG + "Marking", dict, "STRING[]", is_complex=True, description="Security marking ids gating access."),
    "bytes": _pt("bytes", XSD + "base64Binary", bytes, "STRING", is_complex=True, description="Raw binary, base64-encoded for storage."),
    # --- Vector / embedding (dim-parameterized, default 768) ---
    "vector": _pt(
        "vector",
        KG + "Vector",
        list,
        _VECTOR_STORAGE,
        is_complex=True,
        element_type="float",
        dimension=DEFAULT_VECTOR_DIM,
        description="Fixed-dimension float vector.",
    ),
    "embedding": _pt(
        "embedding",
        KG + "Embedding",
        list,
        _VECTOR_STORAGE,
        is_complex=True,
        element_type="float",
        dimension=DEFAULT_VECTOR_DIM,
        description="Semantic embedding; ties to create_embedding_model (768-dim default).",
    ),
}

# Palantir spells several types differently; accept the aliases on lookup.
_ALIASES: dict[str, str] = {
    "str": "string",
    "text": "string",
    "bool": "boolean",
    "int": "integer",
    "int8": "byte",
    "int16": "short",
    "int32": "integer",
    "int64": "long",
    "datetime": "timestamp",
    "geopoint": "geo_point",
    "geo": "geoshape",
    "object": "struct",
    "json": "struct",
    "binary": "bytes",
    "blob": "bytes",
}


# ---------------------------------------------------------------------------
# Lookup / parsing helpers
# ---------------------------------------------------------------------------
_ARRAY_RE = re.compile(r"^(?:array|list|set)\s*<\s*(.+?)\s*>$", re.IGNORECASE)
_VECTOR_RE = re.compile(r"^(vector|embedding)\s*<\s*(\d+)\s*>$", re.IGNORECASE)


def parse_type_ref(type_ref: str) -> PropertyType:
    """Resolve a Palantir-style type reference to a :class:`PropertyType`.

    Handles parameterized forms: ``array<string>`` / ``list<long>`` and
    ``vector<1536>`` / ``embedding<768>``, in addition to the registered base
    and complex types and their aliases.

    Raises:
        ValueError: if the reference cannot be resolved.
    """
    if not isinstance(type_ref, str) or not type_ref.strip():
        raise ValueError("type reference must be a non-empty string")
    ref = type_ref.strip()
    low = ref.lower()

    m = _VECTOR_RE.match(ref)
    if m:
        family, dim_s = m.group(1).lower(), int(m.group(2))
        base = PROPERTY_TYPES[family]
        return base.model_copy(
            update={"dimension": dim_s, "storage_hint": f"FLOAT[{dim_s}]"}
        )

    m = _ARRAY_RE.match(ref)
    if m:
        inner_ref = m.group(1)
        inner = parse_type_ref(inner_ref)
        # Map element storage hint to its array form when the backend supports it
        # (e.g. STRING -> STRING[]); otherwise serialise the array as JSON STRING.
        elem_hint = inner.storage_hint
        arr_hint = f"{elem_hint}[]" if "[" not in elem_hint else "STRING"
        return PropertyType(
            name=f"array<{inner.name}>",
            xsd_iri=KG + "Array",
            python_type=list,
            storage_hint=arr_hint,
            is_complex=True,
            element_type=inner.name,
            description=f"Homogeneous array of {inner.name}.",
        )

    canonical = _ALIASES.get(low, low)
    pt = PROPERTY_TYPES.get(canonical)
    if pt is None:
        raise ValueError(f"unknown property type reference {type_ref!r}")
    return pt


def get_property_type(name: str) -> PropertyType | None:
    """Return the :class:`PropertyType` for ``name`` (alias/parameter aware).

    Returns ``None`` rather than raising for an unknown plain name, but still
    resolves parameterized forms (``array<…>`` / ``vector<…>``).
    """
    try:
        return parse_type_ref(name)
    except ValueError:
        return None


def list_property_types() -> list[str]:
    """Return all registered base/complex property type names."""
    return sorted(PROPERTY_TYPES.keys())


def column_type_for(type_ref: str | PropertyType) -> str:
    """Map a property type (ref or instance) to a ``schema_definition`` column type.

    CONCEPT:KG-2.47 — bridges the ontology type vocabulary onto the exact
    Ladybug/epistemic-graph column-type strings consumed by
    ``agent_utilities.models.knowledge_graph.TableDefinition.columns``. Pass the
    returned string straight into a ``TableDefinition`` column map.

    Examples:
        ``column_type_for("timestamp")`` -> ``"TIMESTAMP"``
        ``column_type_for("array<string>")`` -> ``"STRING[]"``
        ``column_type_for("vector")`` -> ``"FLOAT[768]"``
    """
    pt = type_ref if isinstance(type_ref, PropertyType) else parse_type_ref(type_ref)
    return pt.storage_hint


def coerce_value(type_ref: str, value: Any) -> Any:
    """Convenience: resolve ``type_ref`` and coerce ``value`` in one call."""
    return parse_type_ref(type_ref).coerce(value)


def validate_value(type_ref: str, value: Any) -> bool:
    """Convenience: resolve ``type_ref`` and validate ``value`` in one call."""
    try:
        return parse_type_ref(type_ref).validate(value)
    except ValueError:
        return False


__all__ = [
    "PropertyType",
    "PROPERTY_TYPES",
    "DEFAULT_VECTOR_DIM",
    "get_property_type",
    "parse_type_ref",
    "list_property_types",
    "column_type_for",
    "coerce_value",
    "validate_value",
]
