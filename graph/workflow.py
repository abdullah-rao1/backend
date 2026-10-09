from __future__ import annotations

from langgraph.graph import END, StateGraph

from graph.nodes.generator import generate
from graph.nodes.grader import grade
from graph.nodes.retriever import retrieve
from graph.nodes.rewriter import rewrite
from graph.nodes.verifier import verify
from graph.nodes.web_search import web_search
from graph.state import RAGState

MAX_RETRIES = 2


def _route_after_grading(state: RAGState) -> str:
    if state.get("graded_chunks"):
        return "generate"
    if state.get("retry_count", 0) < state.get("max_retries", MAX_RETRIES):
        return "rewrite"
    return "web_search"


def build_workflow():
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade", grade)
    graph.add_node("rewrite", rewrite)
    graph.add_node("web_search", web_search)
    graph.add_node("generate", generate)
    graph.add_node("verify", verify)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade",
        _route_after_grading,
        {"generate": "generate", "rewrite": "rewrite", "web_search": "web_search"},
    )
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("web_search", "generate")
    graph.add_edge("generate", "verify")
    graph.add_edge("verify", END)

    return graph.compile()


_workflow = None


def get_workflow():
    global _workflow
    if _workflow is None:
        _workflow = build_workflow()
    return _workflow


def run_workflow(session_id: str, question: str) -> RAGState:
    initial_state: RAGState = {
        "session_id": session_id,
        "question": question,
        "original_question": question,
        "retry_count": 0,
        "max_retries": MAX_RETRIES,
        "trace": [],
    }
    return get_workflow().invoke(initial_state)
