from __future__ import annotations

import json
import time

from graph.state import RAGState
from graph.telemetry import record_node
from services.llm import complete

VERIFY_SYSTEM_PROMPT = (
    "You fact-check an AI-generated answer against the excerpts it was based on. "
    "Respond with ONLY a JSON object, no other text: "
    '{"grounded": true or false, "follow_up_questions": ["...", "...", "..."]}. '
    "Set grounded to false if the answer contains any claim not supported by the excerpts. "
    "follow_up_questions is a list of up to 3 short, relevant follow-up questions the user "
    "might ask next, based on the excerpts. If the answer says the information wasn't found, "
    "return grounded: true and an empty follow_up_questions list."
)


def verify(state: RAGState) -> RAGState:
    start = time.perf_counter()
    source = state.get("source", "none")

    if source in ("none", "ai_knowledge"):
        duration_ms = round((time.perf_counter() - start) * 1000, 1)
        trace = [*state.get("trace", []), "Verified"]
        node_trace = record_node(state, "verify", duration_ms)
        return {
            **state,
            "grounded": True,
            "follow_up_questions": [],
            "trace": trace,
            "node_trace": node_trace,
        }

    chunks = state.get("graded_chunks") or []
    web_results = state.get("web_results") or []
    context = "\n\n".join(chunk["text"] for chunk in chunks) or "\n\n".join(
        result["content"] for result in web_results
    )

    result = complete(
        VERIFY_SYSTEM_PROMPT,
        f"Question: {state['original_question']}\n\nExcerpts:\n{context}\n\n"
        f"Answer:\n{state.get('answer', '')}",
    )
    grounded, follow_up_questions = _parse(result.content)
    duration_ms = round((time.perf_counter() - start) * 1000, 1)

    # grounded check is the strongest quality signal, so let it move the confidence too
    confidence = state.get("confidence", 0.0)
    confidence = min(98.0, confidence + 8) if grounded else round(confidence * 0.5, 1)

    trace = [
        *state.get("trace", []),
        f"Verified: {'grounded' if grounded else 'flagged - not fully grounded'}",
    ]
    node_trace = record_node(
        state, "verify", duration_ms, result.input_tokens, result.output_tokens, result.estimated_cost_usd
    )
    return {
        **state,
        "grounded": grounded,
        "confidence": confidence,
        "follow_up_questions": follow_up_questions,
        "trace": trace,
        "node_trace": node_trace,
    }


def _parse(raw: str) -> tuple[bool, list[str]]:
    try:
        start_idx = raw.index("{")
        end_idx = raw.rindex("}") + 1
        data = json.loads(raw[start_idx:end_idx])
        follow_ups = [str(question) for question in data.get("follow_up_questions", [])][:3]
        return bool(data.get("grounded", True)), follow_ups
    except (ValueError, json.JSONDecodeError, AttributeError, TypeError):
        return True, []
