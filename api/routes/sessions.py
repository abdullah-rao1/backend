"""Chats ("sessions"): each chat owns its uploaded documents and its messages."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session

from api.deps import get_current_user
from core.database import get_db
from models.orm.chat_session import ChatSession
from models.orm.user import User
from schemas.session import ChatMessageOut, DocumentOut, SessionCreate, SessionOut, SessionUpdate
from services import chat_history, documents_db, sessions_db

router = APIRouter(prefix="/sessions", tags=["sessions"])

FILE_MEDIA_TYPES = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "txt": "text/plain",
    "csv": "text/csv",
}


def _to_session_out(session: ChatSession) -> SessionOut:
    return SessionOut(
        id=session.id,
        title=session.title,
        created_at=session.created_at,
        document_count=len(session.documents),
        message_count=len(session.messages),
    )


@router.post("", response_model=SessionOut)
def create_session(
    request: SessionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> SessionOut:
    session = sessions_db.create_session(db, current_user, request.title)
    return _to_session_out(session)


@router.get("", response_model=list[SessionOut])
def list_sessions(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[SessionOut]:
    return [_to_session_out(s) for s in sessions_db.list_sessions(db, current_user)]


@router.get("/{session_id}", response_model=SessionOut)
def get_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> SessionOut:
    session = sessions_db.get_owned_session(db, current_user, session_id)
    return _to_session_out(session)


@router.patch("/{session_id}", response_model=SessionOut)
def rename_session(
    session_id: str,
    request: SessionUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> SessionOut:
    session = sessions_db.get_owned_session(db, current_user, session_id)
    session = sessions_db.update_title(db, session, request.title)
    return _to_session_out(session)


@router.delete("/{session_id}")
def delete_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, str]:
    session = sessions_db.get_owned_session(db, current_user, session_id)
    sessions_db.delete_session(db, session)
    return {"message": "Chat deleted."}


@router.get("/{session_id}/documents", response_model=list[DocumentOut])
def list_documents(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[DocumentOut]:
    session = sessions_db.get_owned_session(db, current_user, session_id)
    return documents_db.list_documents(db, session)


@router.delete("/{session_id}/documents/{document_id}")
def delete_document(
    session_id: str,
    document_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, str]:
    session = sessions_db.get_owned_session(db, current_user, session_id)
    document = documents_db.get_owned_document(db, session, document_id)
    documents_db.delete_document(db, session, document)
    return {"message": "Document deleted."}


@router.get("/{session_id}/documents/{document_id}/file")
def get_document_file(
    session_id: str,
    document_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Response:
    session = sessions_db.get_owned_session(db, current_user, session_id)
    document = documents_db.get_owned_document(db, session, document_id)
    path = Path(document.file_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found on disk")

    media_type = FILE_MEDIA_TYPES.get(document.file_type, "application/octet-stream")
    return Response(content=path.read_bytes(), media_type=media_type)


@router.get("/{session_id}/messages", response_model=list[ChatMessageOut])
def list_messages(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[ChatMessageOut]:
    session = sessions_db.get_owned_session(db, current_user, session_id)
    return chat_history.list_messages(db, session)
