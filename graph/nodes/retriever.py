from __future__ import annotations

import time

from embeddings.encoder import embed_query
from graph.state import RAGState
from graph.telemetry import record_node
from vectordb.chroma_client import query_top_k

TOP_K = 5


def retrieve(state: RAGState) -> RAGState:
    start = time.perf_counter()
    query_embedding = embed_query(state["question"])
    matches = query_top_k(state["session_id"], query_embedding, top_k=TOP_K)
    duration_ms = round((time.perf_counter() - start) * 1000, 1)

    trace = [*state.get("trace", []), f"Retrieved {len(matches)} chunk(s)"]
    node_trace = record_node(state, "retrieve", duration_ms)
    return {**state, "retrieved_chunks": matches, "trace": trace, "node_trace": node_trace}
