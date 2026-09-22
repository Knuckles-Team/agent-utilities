from __future__ import annotations

from typing import Any

import pytest

from agent_utilities.knowledge_graph.retrieval.capability_projection import (
    CapabilityProjectionUnavailable,
    load_capability_projection,
)

_KG = "http://knuckles.team/kg#"


class _Graph:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    def owl_reason(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.result


class _Engine:
    def __init__(self, result: Any) -> None:
        self.graph = _Graph(result)


def test_projection_uses_one_digest_bound_composed_schema_classification() -> None:
    engine = _Engine(
        {
            "consistent": True,
            "schema_digests": ["sha256:capability", "sha256:core"],
            "subclasses": [
                [f"<{_KG}DNSCapability>", f"<{_KG}NetworkCapability>"],
                [f"<{_KG}NetworkCapability>", f"<{_KG}ServiceCapability>"],
                [f"<{_KG}DNSCapability>", f"<{_KG}ServiceCapability>"],
                ["<http://foreign.example/Type>", f"<{_KG}ServiceCapability>"],
            ],
            "direct_subclasses": [
                [f"<{_KG}DNSCapability>", f"<{_KG}NetworkCapability>"],
                [f"<{_KG}NetworkCapability>", f"<{_KG}ServiceCapability>"],
            ],
        }
    )

    projection = load_capability_projection(engine)

    assert engine.graph.calls == [
        {
            "ontology": None,
            "target_class": None,
            "class_base": _KG,
        }
    ]
    assert projection.schema_digests == ("sha256:capability", "sha256:core")
    assert projection.is_subtype_of("DNSCapability", "ServiceCapability")
    assert projection.descendants("ServiceCapability") == frozenset(
        {"DNSCapability", "NetworkCapability"}
    )
    assert projection.subsumption_path("DNSCapability", "ServiceCapability") == [
        "DNSCapability",
        "NetworkCapability",
        "ServiceCapability",
    ]
    assert all("Type" not in relation for relation in projection.relations)


@pytest.mark.parametrize(
    "result, message",
    [
        ({"consistent": False, "schema_digests": ["sha256:x"]}, "inconsistent"),
        (
            {
                "consistent": True,
                "schema_digests": [],
                "subclasses": [],
                "direct_subclasses": [],
            },
            "source digests",
        ),
        (
            {"consistent": True, "schema_digests": ["sha256:x"]},
            "no subclasses",
        ),
        (
            {
                "consistent": True,
                "schema_digests": ["sha256:x"],
                "subclasses": [[f"<{_KG}Child>", f"<{_KG}Parent>"]],
                "direct_subclasses": [],
            },
            "no direct",
        ),
        (
            {
                "consistent": True,
                "schema_digests": ["sha256:x"],
                "subclasses": [[f"<{_KG}Child>", f"<{_KG}Parent>"]],
                "direct_subclasses": [[f"<{_KG}Child>", f"<{_KG}Other>"]],
            },
            "disagree",
        ),
    ],
)
def test_projection_fails_closed_without_accepted_engine_lineage(
    result: dict[str, Any], message: str
) -> None:
    with pytest.raises(CapabilityProjectionUnavailable, match=message):
        load_capability_projection(_Engine(result))


def test_designation_does_not_demote_missing_schema_authority_to_keyword_fallback() -> (
    None
):
    from agent_utilities.graph.routing.enrichers.capability_designation import (
        designate_specialists,
    )

    engine = type("Engine", (), {"graph": object()})()
    with pytest.raises(CapabilityProjectionUnavailable, match="OwlReason"):
        designate_specialists(
            engine,
            "route me",
            required_caps=["ServiceCapability"],
            embed_fn=lambda _query: [1.0],
        )
