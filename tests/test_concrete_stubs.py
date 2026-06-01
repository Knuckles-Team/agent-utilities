import os
import json
import pytest
import asyncio
from unittest.mock import MagicMock
from agent_utilities.knowledge_graph.orchestration.engine_query import QueryMixin
from agent_utilities.security.zero_day_immunity import ZeroDayImmunity
from agent_utilities.knowledge_graph.core.ontological_team_sharing import OntologicalTeamExporter
from agent_utilities.harness.trace_backend import OTelTraceBackend
from agent_utilities.messaging.backends.imessage import IMessageBackend
from agent_utilities.messaging.models import MessagingConfig
from agent_utilities.domains.finance.exchange_bridge import BinanceExchange, ExchangeBridge

class DummyEngine(QueryMixin):
    def _serialize_node(self, *args, **kwargs): pass
    def _upsert_node(self, *args, **kwargs): pass
    def _get_set_clause(self, *args, **kwargs): pass
    def _get_allowed_columns(self, *args, **kwargs): pass
    def link_nodes(self, *args, **kwargs): pass
    def add_node(self, *args, **kwargs): pass

# 1. Test MAGMA Place View
def test_magma_place_view():
    backend = MagicMock()
    # Mock return values for backend.execute
    backend.execute.return_value = [
        {"e": {"id": "entity_1", "type": "Entity"}, "p": {"id": "place_1", "type": "Place"}}
    ]
    
    query_engine = DummyEngine()
    query_engine.backend = backend


    
    # Test retrieve_place_view with place_ids
    res = query_engine.retrieve_place_view(query="test", place_ids=["place_1"])
    assert len(res) == 1
    assert res[0]["id"] == "entity_1"
    assert res[0]["_place"] == "place_1"
    assert backend.execute.called

    # Reset mock and test with phase_ids
    backend.execute.reset_mock()
    backend.execute.return_value = [
        {"e": {"id": "entity_2"}, "p": {"id": "phase_1"}}
    ]
    res2 = query_engine.retrieve_place_view(query="test", phase_ids=["phase_1"])
    assert len(res2) == 1
    assert res2[0]["id"] == "entity_2"
    assert res2[0]["_phase"] == "phase_1"

    # Reset mock and test general query search
    backend.execute.reset_mock()
    backend.execute.return_value = [
        {"e": {"id": "entity_3"}, "p": {"id": "place_3"}}
    ]
    res3 = query_engine.retrieve_place_view(query="test_q")
    assert len(res3) == 1
    assert res3[0]["id"] == "entity_3"
    assert res3[0]["_place"] == "place_3"


# 2. Test Zero-Day Immunity
def test_zero_day_immunity():
    analogy = MagicMock()
    analogy.find_isomorphism.return_value = False
    
    zdi = ZeroDayImmunity(analogy_engine=analogy, enabled=True)
    
    # Test scan on natural language prompt
    prompt = "This is a standard query without code."
    payload = {"param1": "val1"}
    result = zdi.scan_request(prompt, payload)
    assert result is True
    
    # Test manual parse subgraph extraction
    sub = zdi._parse_to_subgraph(prompt, payload)
    assert "payload:param1" in sub["nodes"]
    assert "value:val1" in sub["nodes"]
    assert ("payload:param1", "value:val1") in sub["edges"]
    assert "standard" in sub["nodes"]

    # Test with code in prompt (should invoke AST fallback parser)
    code_prompt = "def my_func(): pass"
    sub_code = zdi._parse_to_subgraph(code_prompt, {})
    assert len(sub_code["nodes"]) > 0


# 3. Test Ontological Turtle Import
def test_ontological_turtle_import():
    ttl = """
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix agent: <http://example.org/agent-ontology#> .

agent:test_team a agent:TeamComposition ;
    agent:hasSource "test_src" ;
    agent:hasTopologyTemplate "test_topo" ;
    agent:executionMode "sequential" ;
    agent:confidence 0.95 .

agent:test_team agent:hasSpecialist agent:test_team_spec_0 .
agent:test_team_spec_0 a agent:Specialist ;
    agent:hasRole "analyst" ;
    agent:hasAgentId "agent_0" ;
    agent:usesTool "tool_A" ;
    agent:usesTool "tool_B" .
"""
    # 3.1. Test manual regex fallback (rdflib might not be present or we can force fallback)
    res = OntologicalTeamExporter.import_from_turtle(ttl)
    assert res["team_id"] == "test_team"
    assert res["source"] == "test_src"
    assert res["topology_template_id"] == "test_topo"
    assert res["execution_mode"] == "sequential"
    assert res["confidence"] == 0.95
    assert len(res["adaptive_agent_router"]) == 1
    spec = res["adaptive_agent_router"][0]
    assert spec["role"] == "analyst"
    assert spec["agent_id"] == "agent_0"
    assert "tool_A" in spec["tools"]
    assert "tool_B" in spec["tools"]


# 4. Test OTel Trace Ingestion
@pytest.mark.asyncio
async def test_otel_trace_ingestion(tmp_path):
    # Setup test trace JSON file in temp dir
    trace_data = {
        "id": "trace_123",
        "name": "test_span",
        "status": "success",
        "duration_ms": 150,
        "usageDetails": {"input": 40, "output": 20},
        "score": 0.88
    }
    
    export_dir = str(tmp_path)
    file_path = os.path.join(export_dir, "round_test_trace.json")
    with open(file_path, "w") as f:
        json.dump(trace_data, f)
        
    backend = OTelTraceBackend(export_dir=export_dir)
    
    # Test get_traces
    traces = await backend.get_traces("round_test")
    assert len(traces) == 1
    assert traces[0]["id"] == "trace_123"
    
    # Test get_trace_summary
    summary = await backend.get_trace_summary("trace_123")
    assert summary["id"] == "trace_123"
    assert summary["name"] == "test_span"
    assert summary["status"] == "success"
    assert summary["duration_ms"] == 150
    assert summary["input_tokens"] == 40
    assert summary["output_tokens"] == 20
    assert summary["score"] == 0.88
    
    # Test get_trace_scores
    scores = await backend.get_trace_scores(["trace_123"])
    assert scores["trace_123"] == 0.88


# 5. Test iMessage Inbound Polling
@pytest.mark.asyncio
async def test_imessage_inbound_polling():
    config = MagicMock()
    backend = IMessageBackend(config=config)
    
    # Since we are on Linux/non-Darwin runtime, it should gracefully fall back to idle loop.
    # We can use asyncio.wait_for to ensure it doesn't block forever and behaves safely.
    gen = backend.listen()
    try:
        # Get first element with a short timeout. Should raise TimeoutError because fallback yields nothing
        # but keeps generator alive.
        await asyncio.wait_for(gen.__anext__(), timeout=0.5)
        pytest.fail("Should have timed out on fallback empty yield")
    except asyncio.TimeoutError:
        # Success: did not crash with NotImplementedError!
        pass
    except StopAsyncIteration:
        pass


# 6. Test Binance Exchange Execution Mock
def test_binance_exchange_mock():
    exchange = BinanceExchange()
    
    # Submit a market buy order
    res = exchange.submit_order(symbol="BTC/USDT", side="buy", qty=0.5, order_type="market")
    
    assert "binance-mock" in res.order_id
    assert res.status == "filled"
    assert res.filled_qty == 0.5
    assert res.average_price > 0.0
    assert res.fees > 0.0
    assert res.exchange == "binance"
    
    # Route via ExchangeBridge in paper mode
    bridge = ExchangeBridge(paper_mode=False)
    res2 = bridge.execute(symbol="BTC/USDT", side="sell", qty=0.2)
    assert res2.exchange == "binance"
    assert res2.filled_qty == 0.2
