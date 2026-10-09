from datetime import datetime

from pydantic import BaseModel, Field


class SessionCreate(BaseModel):
    title: str | None = None


class SessionUpdate(BaseModel):
    title: str = Field(min_length=1)


class SessionOut(BaseModel):
    id: str
    title: str | None = None
    created_at: datetime
    document_count: int = 0
    message_count: int = 0

    model_config = {"from_attributes": True}


class DocumentOut(BaseModel):
    id: str
    file_name: str
    file_type: str
    page_count: int | None = None
    text_length: int = 0
    created_at: datetime

    model_config = {"from_attributes": True}


class ChatMessageOut(BaseModel):
    id: str
    question: str
    answer: str
    source: str
    citations: list[dict] = Field(default_factory=list)
    trace: list[str] = Field(default_factory=list)
    retry_count: int = 0
    confidence: float = 0.0
    grounded: bool = True
    follow_up_questions: list[str] = Field(default_factory=list)
    response_time_ms: float = 0.0
    created_at: datetime

    model_config = {"from_attributes": True}
