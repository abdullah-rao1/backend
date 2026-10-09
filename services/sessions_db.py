from __future__ import annotations

import shutil

from fastapi import HTTPException
from sqlalchemy.orm import Session

from core.config import settings
from models.orm.chat_session import ChatSession
from models.orm.user import User
from vectordb.chroma_client import clear_session


def create_session(db: Session, user: User, title: str | None = None) -> ChatSession:
    session = ChatSession(user_id=user.id, title=title)
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def list_sessions(db: Session, user: User) -> list[ChatSession]:
    return (
        db.query(ChatSession)
        .filter(ChatSession.user_id == user.id)
        .order_by(ChatSession.created_at.desc())
        .all()
    )


def get_owned_session(db: Session, user: User, session_id: str) -> ChatSession:
    session = db.get(ChatSession, session_id)
    if not session or session.user_id != user.id:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


def update_title(db: Session, session: ChatSession, title: str) -> ChatSession:
    session.title = title
    db.commit()
    db.refresh(session)
    return session


def maybe_set_title(db: Session, session: ChatSession, title: str) -> None:
    """Auto-name a chat (from its first document or first question) unless it already has a name."""
    if session.title or not title.strip():
        return
    trimmed = title.strip()
    session.title = trimmed if len(trimmed) <= 60 else trimmed[:60].rstrip() + "..."
    db.commit()


def delete_session(db: Session, session: ChatSession) -> None:
    session_id = session.id
    db.delete(session)
    db.commit()
    _delete_session_artifacts(session_id)


def delete_all_for_user(db: Session, user: User) -> None:
    for session in list_sessions(db, user):
        delete_session(db, session)


def _delete_session_artifacts(session_id: str) -> None:
    clear_session(session_id)
    shutil.rmtree(settings.upload_dir / session_id, ignore_errors=True)
