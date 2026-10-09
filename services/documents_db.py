from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException
from sqlalchemy.orm import Session

from models.orm.chat_session import ChatSession
from models.orm.document import Document
from vectordb.chroma_client import delete_document_chunks


def create_document(
    db: Session,
    session: ChatSession,
    document_id: str,
    file_name: str,
    file_type: str,
    file_path: str,
    page_count: int | None,
    text_length: int,
) -> Document:
    document = Document(
        id=document_id,
        session_id=session.id,
        file_name=file_name,
        file_type=file_type,
        file_path=file_path,
        page_count=page_count,
        text_length=text_length,
    )
    db.add(document)
    db.commit()
    db.refresh(document)
    return document


def list_documents(db: Session, session: ChatSession) -> list[Document]:
    return (
        db.query(Document)
        .filter(Document.session_id == session.id)
        .order_by(Document.created_at.asc())
        .all()
    )


def get_owned_document(db: Session, session: ChatSession, document_id: str) -> Document:
    document = db.get(Document, document_id)
    if not document or document.session_id != session.id:
        raise HTTPException(status_code=404, detail="Document not found")
    return document


def delete_document(db: Session, session: ChatSession, document: Document) -> None:
    delete_document_chunks(session.id, document.id)
    Path(document.file_path).unlink(missing_ok=True)
    db.delete(document)
    db.commit()
