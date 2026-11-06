"""Run the guarded graph topology, pause at synthesis and resume in memory.

No LLM, seller data, database, tracing provider or marketplace writes are used.
Domain dependencies are intentionally not supplied: this exercises orchestration,
not business analysis. See domain packages for opt-in implementations.
"""
from __future__ import annotations

import json
from uuid import uuid4

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from commerce_agent.context import CommerceRuntimeContext
from commerce_agent.graph import build_graph
from commerce_agent.ingress import GuardedGraph
from commerce_agent.serialization import checkpoint_serializer


def run_demo() -> dict:
    graph = GuardedGraph(build_graph(
        checkpointer=InMemorySaver(serde=checkpoint_serializer()),
        interrupt_before=["synthesis"],
    ))
    context = CommerceRuntimeContext(
        tenant_id="demo-tenant", user_id="demo-user", roles=("viewer",), request_id="demo-request",
    )
    config = {"configurable": {"thread_id": str(uuid4())}}
    graph.invoke(
        {"schema_version": 1, "messages": [HumanMessage(content="Compare sales and finance", id="demo-question")]},
        config=config, context=context,
    )
    paused = graph.get_state(config)
    assert paused.next == ("synthesis",), paused.next
    result = graph.invoke(None, config=config, context=context)
    finished = graph.get_state(config)
    assert not finished.next
    assert not result["execution_receipts"]
    return {
        "graph": "commerce_agent",
        "paused_before": list(paused.next),
        "resumed": not bool(finished.next),
        "domain_statuses": {domain: result[f"{domain}_analysis"].status for domain in ("sales", "marketing", "supply", "price", "finance")},
        "external_writes": len(result["execution_receipts"]),
        "scope": "in-memory topology/recovery demo; domain services not connected",
    }


if __name__ == "__main__":
    print(json.dumps(run_demo(), ensure_ascii=False, indent=2))
