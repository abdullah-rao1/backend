from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".csv"}


def sanitize_filename(filename: str) -> str:
    return Path(filename).name.replace(" ", "_")


def file_extension(filename: str) -> str:
    return Path(filename).suffix.lower()


def is_supported_file(filename: str) -> bool:
    return file_extension(filename) in SUPPORTED_EXTENSIONS


def make_document_id(session_id: str, filename: str) -> str:
    return f"{session_id}-{Path(filename).stem}-{uuid4().hex[:12]}"


async def save_upload_file(destination: Path, upload: UploadFile) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = await upload.read()
    destination.write_bytes(content)
