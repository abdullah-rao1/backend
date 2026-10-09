"""Question answering: runs the LangGraph RAG workflow and saves the result to Postgres."""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.deps import get_current_user
from core.database import get_db
from graph.workflow import run_workflow
from models.orm.user import User
from schemas.chat import ChatRequest, ChatResponse
from services.chat_history import save_message
from services.llm import LLMUnavailableError
from services.sessions_db import create_session, get_owned_session, maybe_set_title

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ChatResponse:
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    session = (
        get_owned_session(db, current_user, request.session_id)
        if request.session_id
        else create_session(db, current_user)
    )
    maybe_set_title(db, session, request.question)

    started = time.perf_counter()
    try:
        result = run_workflow(session.id, request.question)
    except LLMUnavailableError as exc:
        logger.warning("LLM call failed: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="The AI service (Groq) is temporarily unavailable. Please try again shortly.",
        ) from exc
    response_time_ms = round((time.perf_counter() - started) * 1000, 1)

    answer = result.get("answer", "")
    source = result.get("source", "none")
    citations = result.get("citations", [])
    trace = result.get("trace", [])
    retry_count = result.get("retry_count", 0)
    confidence = result.get("confidence", 0.0)
    grounded = result.get("grounded", True)
    follow_up_questions = result.get("follow_up_questions", [])
    rewrite_history = result.get("rewrite_history", [])
    node_trace = result.get("node_trace", [])
    total_tokens = sum(node["tokens_in"] + node["tokens_out"] for node in node_trace)
    estimated_cost_usd = sum(node["cost_usd"] for node in node_trace)

    save_message(
        db,
        session,
        request.question,
        answer,
        source,
        citations,
        trace,
        retry_count,
        confidence=confidence,
        grounded=grounded,
        follow_up_questions=follow_up_questions,
        rewrite_history=rewrite_history,
        node_trace=node_trace,
        response_time_ms=response_time_ms,
        total_tokens=total_tokens,
        estimated_cost_usd=estimated_cost_usd,
    )

    return ChatResponse(
        session_id=session.id,
        question=result.get("original_question", request.question),
        answer=answer,
        source=source,
        citations=citations,
        trace=trace,
        retry_count=retry_count,
        confidence=confidence,
        grounded=grounded,
        follow_up_questions=follow_up_questions,
        response_time_ms=response_time_ms,
    )
