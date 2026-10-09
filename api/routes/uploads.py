"""Ingestion: file -> text -> chunks -> embeddings -> ChromaDB (+ document row in Postgres)."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from api.deps import get_current_user
from core.config import settings
from core.database import get_db
from embeddings.encoder import embed_texts
from models.document import DocumentRecord
from models.orm.user import User
from schemas.ingestion import UploadedChunk, UploadedDocument, UploadResponse
from services.documents_db import create_document
from services.extraction import extract_pages
from services.sessions_db import create_session, get_owned_session, maybe_set_title
from utils.files import (
    file_extension,
    is_supported_file,
    make_document_id,
    sanitize_filename,
    save_upload_file,
)
from utils.text import chunk_text
from vectordb.chroma_client import upsert_chunks

router = APIRouter(prefix="/uploads", tags=["uploads"])


def session_document_path(session_id: str, file_name: str) -> Path:
    directory = settings.upload_dir / session_id
    directory.mkdir(parents=True, exist_ok=True)
    return directory / file_name


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


@router.post("", response_model=UploadResponse)
async def upload_documents(
    files: list[UploadFile] = File(...),
    session_id: str | None = Form(default=None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> UploadResponse:
    session = (
        get_owned_session(db, current_user, session_id)
        if session_id
        else create_session(db, current_user)
    )

    records: list[DocumentRecord] = []
    documents: list[UploadedDocument] = []
    all_chunks: list[UploadedChunk] = []
    trace: list[str] = []

    for upload in files:
        if not upload.filename or not is_supported_file(upload.filename):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {upload.filename}",
            )

        # 1. Save the original file to disk (on Railway: the volume)
        file_name = sanitize_filename(upload.filename)
        destination = session_document_path(session.id, file_name)
        await save_upload_file(destination, upload)
        trace.append(f"Uploaded {file_name}")

        size_mb = destination.stat().st_size / (1024 * 1024)
        if size_mb > settings.max_file_size_mb:
            _safe_unlink(destination)
            raise HTTPException(
                status_code=400,
                detail=f"{file_name} exceeds max size of {settings.max_file_size_mb}MB",
            )

        # 2. Extract text (page by page for PDFs)
        extension = file_extension(file_name)
        try:
            pages = extract_pages(destination, extension)
        except Exception as exc:
            _safe_unlink(destination)
            raise HTTPException(
                status_code=400,
                detail=f"{file_name} could not be read. It may be corrupted or password-protected.",
            ) from exc

        full_text = "\n".join(page_text for page_text, _ in pages)
        if not full_text.strip():
            _safe_unlink(destination)
            raise HTTPException(
                status_code=400,
                detail=f"{file_name} appears to be empty or has no extractable text.",
            )
        page_count = len(pages) if extension == ".pdf" else None
        trace.append(f"Extracted text from {file_name} ({page_count or 1} page(s))")

        record = DocumentRecord(
            document_id=make_document_id(session.id, file_name),
            file_name=file_name,
            file_type=extension.lstrip("."),
            file_path=str(destination),
            session_id=session.id,
            page_count=page_count,
            text=full_text,
        )
        records.append(record)
        documents.append(
            UploadedDocument(
                document_id=record.document_id,
                file_name=record.file_name,
                file_type=record.file_type,
                page_count=record.page_count,
                text_length=len(record.text),
            )
        )
        # 3. Save document metadata in Postgres
        create_document(
            db,
            session,
            document_id=record.document_id,
            file_name=record.file_name,
            file_type=record.file_type,
            file_path=record.file_path,
            page_count=record.page_count,
            text_length=len(record.text),
        )

        # 4. Split into overlapping chunks (1000 chars, 150 overlap), keeping page numbers
        document_chunks: list[tuple[str, int | None]] = [
            (chunk, page_number) for page_text, page_number in pages for chunk in chunk_text(page_text)
        ]
        for index, (chunk, page_number) in enumerate(document_chunks):
            all_chunks.append(
                UploadedChunk(
                    chunk_id=f"{record.document_id}-{index}",
                    document_id=record.document_id,
                    document_name=record.file_name,
                    chunk_index=index,
                    page_number=page_number,
                    text=chunk,
                )
            )
        trace.append(f"Chunked {file_name} into {len(document_chunks)} chunk(s)")

    # 5. Embed every chunk and store the vectors in ChromaDB
    _embed_and_store(session.id, all_chunks)
    trace.append(f"Embedded {len(all_chunks)} chunk(s)")
    trace.append("Stored in ChromaDB")

    if documents:
        title_guess = Path(documents[0].file_name).stem
        maybe_set_title(db, session, title_guess)

    return UploadResponse(
        session_id=session.id,
        message=f"Stored {len(documents)} document(s).",
        documents=documents,
        chunk_count=len(all_chunks),
        trace=trace,
    )


def _embed_and_store(session_id: str, chunks: list[UploadedChunk]) -> None:
    if not chunks:
        return

    embeddings = embed_texts([chunk.text for chunk in chunks])
    upsert_chunks(
        session_id=session_id,
        ids=[chunk.chunk_id for chunk in chunks],
        embeddings=embeddings,
        documents=[chunk.text for chunk in chunks],
        metadatas=[
            {
                "document_id": chunk.document_id,
                "document_name": chunk.document_name,
                "chunk_index": chunk.chunk_index,
                "page_number": chunk.page_number if chunk.page_number is not None else -1,
            }
            for chunk in chunks
        ],
    )
