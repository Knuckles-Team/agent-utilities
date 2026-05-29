#!/usr/bin/env python3
import asyncio
import os
import sys

from agent_utilities.knowledge_graph.core.event_backend import create_event_backend

async def main() -> None:
    print("Starting Kafka Sync Verification...")
    
    # Force Kafka Backend Configuration
    os.environ["KAFKA_BOOTSTRAP_SERVERS"] = "localhost:9092"
    os.environ["EVENT_BACKEND"] = "kafka"
    
    backend = create_event_backend()
    await backend.start()
    
    # 1. Publish a test mutation payload
    test_query = "INSERT DATA { <TestNode> <Status> \"SyncVerified\" }"
    payload = {
        "event_type": "TRIPLE_INSERT",
        "query": test_query,
        "source": "verify_kafka_sync"
    }
    
    print(f"Publishing mutation to kg.mutations over Kafka: {payload}")
    await backend.publish("kg.mutations", payload)
    
    # Wait slightly for Kafka delivery
    await asyncio.sleep(2)
    
    print("Event published.")
    await backend.stop()
    
    print("Verification script complete.")
    print("Check your epistemic-graph service logs to confirm the event was consumed!")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
