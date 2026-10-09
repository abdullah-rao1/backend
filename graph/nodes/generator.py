from __future__ import annotations

import time

from graph.state import RAGState
from graph.telemetry import record_node
from services.llm import complete

GENERATOR_SYSTEM_PROMPT = (
    "You answer questions using ONLY the information in the provided excerpts. Read all of "
    "them carefully and synthesize a helpful, thorough answer, combining or reasoning over "
    "multiple excerpts if needed. Do not add facts that aren't supported by the excerpts.\n\n"
    "Write like a knowledgeable assistant, not a search snippet: use **bold** for key terms, "
    "names, and entities, and bullet or numbered lists when listing multiple items. Even when "
    "the direct answer is 'no' or the excerpts don't contain exactly what was asked, don't stop "
    "at a one-line answer - briefly cover what the excerpts DO say that's related, so the reply "
    "is genuinely useful rather than just a rejection.\n\n"
    "Only respond with exactly \"I couldn't find this in the uploaded documents.\" if the "
    "excerpts are genuinely unrelated to the question and contain nothing useful for answering "
    "it or providing helpful context - not merely because the answer isn't stated word-for-word."
)

AI_KNOWLEDGE_SYSTEM_PROMPT = (
    "Answer the user's question directly and helpfully using your own general knowledge, "
    "formatted the same way you would for any question - use **bold** for key terms and "
    "lists when helpful. Do not mention documents, uploads, or search results; just answer "
    "the question as asked. If you're not confident in the answer, say so briefly, but still "
    "give your best answer."
)


# remaps raw cosine similarity (rarely near 1.0 even for good matches) to a % that reads right
_SIMILARITY_FLOOR = 0.25
_SIMILARITY_CEILING = 0.75


def _calibrate_similarity(similarity: float) -> float:
    similarity = max(0.0, min(similarity, 1.0))
    if similarity <= _SIMILARITY_FLOOR:
        return (similarity / _SIMILARITY_FLOOR) * 40
    if similarity >= _SIMILARITY_CEILING:
        excess = (similarity - _SIMILARITY_CEILING) / (1 - _SIMILARITY_CEILING)
        return 90 + min(8, excess * 8)
    progress = (similarity - _SIMILARITY_FLOOR) / (_SIMILARITY_CEILING - _SIMILARITY_FLOOR)
    return 40 + progress * 50


def generate(state: RAGState) -> RAGState:
    start = time.perf_counter()
    chunks = state.get("graded_chunks", [])
    web_results = state.get("web_results", [])

    if chunks:
        update = _answer_from_documents(state, chunks)
    elif web_results:
        update = _answer_from_web(state, web_results)
    else:
        update = _answer_from_ai_knowledge(state)

    duration_ms = round((time.perf_counter() - start) * 1000, 1)
    trace_label = (
        "Not found in documents or web - generated from AI knowledge"
        if update["source"] == "ai_knowledge"
        else "Generated"
    )
    trace = [*state.get("trace", []), trace_label]
    node_trace = record_node(
        state,
        "generate",
        duration_ms,
        update.pop("tokens_in"),
        update.pop("tokens_out"),
        update.pop("cost_usd"),
    )
    return {**state, **update, "trace": trace, "node_trace": node_trace}


def _answer_from_documents(state: RAGState, chunks: list[dict]) -> dict:
    context = "\n\n".join(
        f"[{chunk['document_name']} | page {chunk['page_number']} | chunk {chunk['chunk_index']}]\n"
        f"{chunk['text']}"
        for chunk in chunks
    )
    result = complete(
        GENERATOR_SYSTEM_PROMPT,
        f"Question: {state['original_question']}\n\nApproved excerpts:\n{context}",
    )
    citations = [
        {
            "source_type": "document",
            "document_id": chunk["document_id"],
            "document_name": chunk["document_name"],
            "page_number": chunk["page_number"],
            "chunk_index": chunk["chunk_index"],
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
        }
        for chunk in chunks
    ]

    retrieved_count = len(state.get("retrieved_chunks", [])) or len(chunks)
    grade_pass_rate = len(chunks) / retrieved_count
    avg_similarity = sum(max(chunk["score"], 0.0) for chunk in chunks) / len(chunks)
    # similarity of chunks actually used matters more than how many got filtered out
    similarity_score = _calibrate_similarity(avg_similarity)
    confidence = round(0.85 * similarity_score + 0.15 * (grade_pass_rate * 100), 1)

    return {
        "answer": result.content.strip(),
        "citations": citations,
        "source": "documents",
        "confidence": confidence,
        "tokens_in": result.input_tokens,
        "tokens_out": result.output_tokens,
        "cost_usd": result.estimated_cost_usd,
    }


def _answer_from_web(state: RAGState, web_results: list[dict]) -> dict:
    context = "\n\n".join(
        f"[{result['title']} | {result['url']}]\n{result['content']}" for result in web_results
    )
    result = complete(
        GENERATOR_SYSTEM_PROMPT,
        f"Question: {state['original_question']}\n\nApproved excerpts:\n{context}",
    )
    citations = [
        {
            "source_type": "web",
            "title": web_result["title"],
            "url": web_result["url"],
            "text": web_result["content"][:500],
        }
        for web_result in web_results
    ]
    confidence = round(min(100.0, 40 + 10 * len(web_results)), 1)

    return {
        "answer": result.content.strip(),
        "citations": citations,
        "source": "web",
        "confidence": confidence,
        "tokens_in": result.input_tokens,
        "tokens_out": result.output_tokens,
        "cost_usd": result.estimated_cost_usd,
    }


def _answer_from_ai_knowledge(state: RAGState) -> dict:
    result = complete(AI_KNOWLEDGE_SYSTEM_PROMPT, f"Question: {state['original_question']}")

    return {
        "answer": result.content.strip(),
        "citations": [],
        "source": "ai_knowledge",
        "confidence": 0.0,
        "tokens_in": result.input_tokens,
        "tokens_out": result.output_tokens,
        "cost_usd": result.estimated_cost_usd,
    }
