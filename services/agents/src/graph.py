from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, Sequence

from langgraph.graph import END, START, StateGraph

from .memory import MemoryStore
from .nodes import (
    VisionClient,
    memory_node,
    retrieval_node,
    router_node,
    synthesis_node,
    vision_node,
)
from .state import AgentState


def build_graph(
    *,
    model: Any | None = None,
    retriever: Callable[[str, int], Sequence[Any]] | None = None,
    memory_store: MemoryStore | None = None,
    vision_client: VisionClient | None = None,
):
    workflow = StateGraph(AgentState)
    workflow.add_node("router", router_node)
    workflow.add_node("vision", lambda state: vision_node(state, vision_client))
    workflow.add_node("memory", lambda state: memory_node(state, memory_store))
    workflow.add_node("retrieval", lambda state: retrieval_node(state, retriever))
    workflow.add_node("synthesis", lambda state: synthesis_node(state, model))
    workflow.add_edge(START, "router")
    workflow.add_conditional_edges(
        "router",
        lambda state: state["request_type"],
        {
            "new_inspection": "vision",
            "history_query": "memory",
            "general_advisory": "memory",
        },
    )
    workflow.add_edge("vision", "memory")
    workflow.add_edge("memory", "retrieval")
    workflow.add_edge("retrieval", "synthesis")
    workflow.add_edge("synthesis", END)
    return workflow.compile()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run or validate the GanodermaScout agent graph")
    parser.add_argument("--dry-run", action="store_true", help="validate graph topology only")
    parser.add_argument("--query")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--palm-id")
    parser.add_argument("--palm-code")
    parser.add_argument("--block-id")
    args = parser.parse_args(argv)
    graph = build_graph()

    if args.dry_run:
        nodes = set(graph.get_graph().nodes)
        required_nodes = {"router", "vision", "memory", "retrieval", "synthesis"}
        if not required_nodes.issubset(nodes):
            raise RuntimeError(f"Agent graph is missing nodes: {sorted(required_nodes - nodes)}")
        print(f"Agent graph valid: {', '.join(sorted(required_nodes))}")
        return 0
    if not args.query and args.image is None:
        parser.error("--query or --image is required unless --dry-run is used")

    state: AgentState = {}
    if args.query:
        state["user_query"] = args.query
    if args.image is not None:
        state["image_bytes"] = args.image.read_bytes()
    if args.palm_id:
        state["palm_id"] = args.palm_id
    if args.palm_code:
        state["palm_code"] = args.palm_code
    if args.block_id:
        state["block_id"] = args.block_id
    result = graph.invoke(state)
    print(result.get("drafted_advisory", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
