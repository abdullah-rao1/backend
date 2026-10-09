from __future__ import annotations

import time
from functools import lru_cache

from core.config import settings
from graph.state import RAGState
from graph.telemetry import record_node

MAX_WEB_RESULTS = 5


@lru_cache(maxsize=1)
def _tavily_client():
    from tavily import TavilyClient

    return TavilyClient(api_key=settings.tavily_api_key)


def web_search(state: RAGState) -> RAGState:
    start = time.perf_counter()
    trace = [*state.get("trace", []), "Web search"]

    if not settings.tavily_api_key:
        duration_ms = round((time.perf_counter() - start) * 1000, 1)
        node_trace = record_node(state, "web_search", duration_ms)
        return {**state, "web_results": [], "trace": trace, "node_trace": node_trace}

    response = _tavily_client().search(state["original_question"], max_results=MAX_WEB_RESULTS)
    results = [
        {
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "content": item.get("content", ""),
        }
        for item in response.get("results", [])
    ]
    duration_ms = round((time.perf_counter() - start) * 1000, 1)
    node_trace = record_node(state, "web_search", duration_ms)
    return {**state, "web_results": results, "trace": trace, "node_trace": node_trace}
