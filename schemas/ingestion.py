from pydantic import BaseModel, Field


class UploadedChunk(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    page_number: int | None = None
    chunk_index: int
    text: str


class UploadedDocument(BaseModel):
    document_id: str
    file_name: str
    file_type: str
    page_count: int | None = None
    text_length: int = 0
    status: str = "stored"


class UploadResponse(BaseModel):
    session_id: str
    message: str
    documents: list[UploadedDocument] = Field(default_factory=list)
    chunk_count: int = 0
    trace: list[str] = Field(default_factory=list)
