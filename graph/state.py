from __future__ import annotations

from typing import TypedDict


class ChunkMatch(TypedDict):
    chunk_id: str
    document_id: str
    text: str
    document_name: str
    page_number: int | None
    chunk_index: int
    score: float


class WebResult(TypedDict):
    title: str
    url: str
    content: str


class NodeTrace(TypedDict):
    node: str
    duration_ms: float
    tokens_in: int
    tokens_out: int
    cost_usd: float


class RewriteStep(TypedDict):
    retry: int
    from_: str
    to: str


class RAGState(TypedDict, total=False):
    session_id: str
    question: str
    original_question: str
    retrieved_chunks: list[ChunkMatch]
    graded_chunks: list[ChunkMatch]
    web_results: list[WebResult]
    retry_count: int
    max_retries: int
    source: str
    answer: str
    citations: list[dict]
    trace: list[str]
    node_trace: list[NodeTrace]
    rewrite_history: list[RewriteStep]
    confidence: float
    grounded: bool
    follow_up_questions: list[str]
