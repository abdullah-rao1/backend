from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str | None = None
    question: str


class Citation(BaseModel):
    source_type: str
    document_id: str | None = None
    document_name: str | None = None
    page_number: int | None = None
    chunk_index: int | None = None
    chunk_id: str | None = None
    title: str | None = None
    url: str | None = None
    text: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    question: str
    answer: str
    source: str  # "documents" | "web" | "ai_knowledge"
    citations: list[Citation] = Field(default_factory=list)
    trace: list[str] = Field(default_factory=list)  # human-readable pipeline steps
    retry_count: int = 0
    confidence: float = 0.0
    grounded: bool = True  # The LLM's verdict from the verify step
    follow_up_questions: list[str] = Field(default_factory=list)
    response_time_ms: float = 0.0
