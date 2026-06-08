"""Code → capability REALIZES resolution tests (CONCEPT:KG-2.8).

Covers the three unified modes: match-existing, bottom-up mint, and strict
curated-registry matching. Pure functions, no backend / no network.
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.enrichment.realizes import resolve_realizes


def _feat(fid, name, summary=""):
    return {"id": fid, "name": name, "summary": summary}


def _cap(cid, name, summary=""):
    return {"id": cid, "name": name, "summary": summary}


def test_match_existing_capability_top_down():
    features = [
        _feat("feature:1", "order fulfillment pipeline", "ships customer orders")
    ]
    caps = [
        _cap("capability:OF", "Order Fulfillment", "fulfill and ship orders"),
        _cap("capability:HR", "Human Resources", "payroll and people"),
    ]
    minted, edges = resolve_realizes(features, caps, mint_missing=False)

    assert minted == []
    assert len(edges) == 1
    assert edges[0].source == "feature:1"
    assert edges[0].target == "capability:OF"
    assert edges[0].rel_type == "REALIZES"


def test_mint_bottom_up_when_no_match():
    features = [_feat("feature:9", "quantum widget reticulation", "splines")]
    # No remotely-related capability -> mint a provisional one from the feature.
    minted, edges = resolve_realizes(features, capabilities=[], mint_missing=True)

    assert len(minted) == 1
    cap = minted[0]
    assert cap.type == "BusinessCapability"
    assert cap.id == "capability:derived:quantum-widget-reticulation"
    assert cap.props["provisional"] is True
    assert cap.props["derived_from"] == "code"

    assert len(edges) == 1
    assert edges[0].source == "feature:9"
    assert edges[0].target == cap.id


def test_strict_registry_no_mint():
    features = [
        _feat("feature:1", "billing engine", "invoices and charges"),
        _feat("feature:2", "obscure internal helper", "no capability"),
    ]
    registry = [_cap("capability:BILL", "Billing", "invoices and charges")]
    minted, edges = resolve_realizes(
        features, capabilities=[], registry=registry, mint_missing=False
    )

    assert minted == []
    targets = {e.target for e in edges}
    assert targets == {"capability:BILL"}  # only the matching feature links


def test_registry_preferred_over_existing_on_tie():
    features = [_feat("feature:1", "payments", "process payments")]
    registry = [_cap("capability:CURATED", "payments", "process payments")]
    existing = [_cap("capability:LEANIX", "payments", "process payments")]
    minted, edges = resolve_realizes(
        features, existing, registry=registry, mint_missing=False
    )
    assert edges[0].target == "capability:CURATED"


def test_embed_fn_used_when_supplied():
    # A trivial embedder: orthogonal vectors unless texts are identical-ish.
    def embed(text: str):
        t = text.lower()
        return [1.0, 0.0] if "alpha" in t else [0.0, 1.0]

    features = [_feat("feature:1", "alpha service", "")]
    caps = [
        _cap("capability:A", "alpha capability", ""),
        _cap("capability:B", "beta", ""),
    ]
    minted, edges = resolve_realizes(
        features, caps, mint_missing=False, embed_fn=embed, match_threshold=0.9
    )
    assert edges[0].target == "capability:A"
