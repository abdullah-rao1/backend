from __future__ import annotations

from fastapi import HTTPException
from sqlalchemy.orm import Session

from models.orm.chat_message import ChatMessage
from models.orm.chat_session import ChatSession


def save_message(
    db: Session,
    session: ChatSession,
    question: str,
    answer: str,
    source: str,
    citations: list[dict],
    trace: list[str],
    retry_count: int,
    confidence: float = 0.0,
    grounded: bool = True,
    follow_up_questions: list[str] | None = None,
    rewrite_history: list[dict] | None = None,
    node_trace: list[dict] | None = None,
    response_time_ms: float = 0.0,
    total_tokens: int = 0,
    estimated_cost_usd: float = 0.0,
) -> ChatMessage:
    message = ChatMessage(
        session_id=session.id,
        question=question,
        answer=answer,
        source=source,
        citations=citations,
        trace=trace,
        retry_count=retry_count,
        confidence=confidence,
        grounded=grounded,
        follow_up_questions=follow_up_questions or [],
        rewrite_history=rewrite_history or [],
        node_trace=node_trace or [],
        response_time_ms=response_time_ms,
        total_tokens=total_tokens,
        estimated_cost_usd=estimated_cost_usd,
    )
    db.add(message)
    db.commit()
    db.refresh(message)
    return message


def list_messages(db: Session, session: ChatSession) -> list[ChatMessage]:
    return (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session.id)
        .order_by(ChatMessage.created_at.asc())
        .all()
    )


def get_owned_message(db: Session, session: ChatSession, message_id: str) -> ChatMessage:
    message = db.get(ChatMessage, message_id)
    if not message or message.session_id != session.id:
        raise HTTPException(status_code=404, detail="Message not found")
    return message


def delete_message(db: Session, message: ChatMessage) -> None:
    db.delete(message)
    db.commit()


def clear_messages(db: Session, session: ChatSession) -> None:
    db.query(ChatMessage).filter(ChatMessage.session_id == session.id).delete()
    db.commit()
