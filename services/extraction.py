from __future__ import annotations

import csv
from pathlib import Path

import fitz
from docx import Document


def extract_pages(path: Path, extension: str) -> list[tuple[str, int | None]]:
    """Returns (page_text, page_number) pairs. page_number is None for formats without real pages."""
    if extension == ".pdf":
        return _extract_pdf_pages(path)
    if extension == ".docx":
        return [(_extract_docx(path), None)]
    if extension == ".txt":
        return [(_extract_txt(path), None)]
    if extension == ".csv":
        return [(_extract_csv(path), None)]
    raise ValueError(f"Unsupported extension: {extension}")


def _extract_pdf_pages(path: Path) -> list[tuple[str, int | None]]:
    with fitz.open(path) as document:
        return [(page.get_text(), index + 1) for index, page in enumerate(document)]


def _extract_docx(path: Path) -> str:
    document = Document(path)
    return "\n".join(paragraph.text for paragraph in document.paragraphs)


def _extract_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _extract_csv(path: Path) -> str:
    with path.open(encoding="utf-8", errors="ignore", newline="") as handle:
        rows = list(csv.reader(handle))

    if not rows:
        return ""

    header, data_rows = rows[0], rows[1:]
    lines = [
        ", ".join(f"{col}: {value}" for col, value in zip(header, row))
        for row in data_rows
    ]
    return "\n".join(lines)
