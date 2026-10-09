from __future__ import annotations

import time

from graph.state import RAGState
from graph.telemetry import record_node
from services.llm import complete

GRADER_SYSTEM_PROMPT = (
    "You are a strict relevance grader. Given a user question and a document chunk, "
    "respond with exactly one word: 'yes' if the chunk contains information useful for "
    "answering the question, or 'no' if it does not."
)


def _is_relevant(question: str, chunk_text: str) -> tuple[bool, int, int, float]:
    result = complete(GRADER_SYSTEM_PROMPT, f"Question: {question}\n\nChunk:\n{chunk_text}")
    relevant = result.content.strip().lower().startswith("y")
    return relevant, result.input_tokens, result.output_tokens, result.estimated_cost_usd


def grade(state: RAGState) -> RAGState:
    start = time.perf_counter()
    chunks = state.get("retrieved_chunks", [])

    approved = []
    tokens_in = tokens_out = 0
    cost_usd = 0.0
    for chunk in chunks:
        relevant, t_in, t_out, chunk_cost = _is_relevant(state["question"], chunk["text"])
        tokens_in += t_in
        tokens_out += t_out
        cost_usd += chunk_cost
        if relevant:
            approved.append(chunk)

    duration_ms = round((time.perf_counter() - start) * 1000, 1)
    trace = [*state.get("trace", []), f"Graded: {len(approved)}/{len(chunks)} chunk(s) approved"]
    node_trace = record_node(state, "grade", duration_ms, tokens_in, tokens_out, cost_usd)
    return {**state, "graded_chunks": approved, "trace": trace, "node_trace": node_trace}
