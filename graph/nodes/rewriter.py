from __future__ import annotations

import time

from graph.state import RAGState
from graph.telemetry import record_node
from services.llm import complete

REWRITER_SYSTEM_PROMPT = (
    "You rewrite user questions to improve document retrieval. "
    "Return ONLY the rewritten question, with no explanation."
)


def rewrite(state: RAGState) -> RAGState:
    start = time.perf_counter()
    previous_question = state["question"]

    result = complete(
        REWRITER_SYSTEM_PROMPT,
        f"Original question: {state.get('original_question', previous_question)}\n"
        f"Previous attempt: {previous_question}\n"
        "Rewrite it to be clearer and more specific for document search.",
    )
    rewritten = result.content.strip()
    duration_ms = round((time.perf_counter() - start) * 1000, 1)

    retry_count = state.get("retry_count", 0) + 1
    trace = [*state.get("trace", []), f"Rewritten (retry {retry_count})"]
    node_trace = record_node(
        state, "rewrite", duration_ms, result.input_tokens, result.output_tokens, result.estimated_cost_usd
    )
    rewrite_history = [
        *state.get("rewrite_history", []),
        {"retry": retry_count, "from_": previous_question, "to": rewritten},
    ]
    return {
        **state,
        "question": rewritten,
        "retry_count": retry_count,
        "trace": trace,
        "node_trace": node_trace,
        "rewrite_history": rewrite_history,
    }
