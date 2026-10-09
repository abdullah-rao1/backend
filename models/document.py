from dataclasses import dataclass


@dataclass(slots=True)
class DocumentRecord:
    document_id: str
    file_name: str
    file_type: str
    file_path: str
    session_id: str
    page_count: int | None = None
    text: str = ""
