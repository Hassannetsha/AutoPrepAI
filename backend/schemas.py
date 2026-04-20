from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class ChatResponsePayload(BaseModel):
    shape: tuple[int, int] | None = None
    logs: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    data_preview: list[dict] = Field(default_factory=list)
    output_file: str | None = None
    download_url: str | None = None


class ChatResponse(BaseModel):
    conversation_id: UUID
    assistant_message: str
    result: ChatResponsePayload


class ConversationMessageOut(BaseModel):
    id: int
    role: str
    content: str
    payload: dict | None
    created_at: datetime

    class Config:
        from_attributes = True


class ConversationOut(BaseModel):
    id: UUID
    created_at: datetime
    messages: list[ConversationMessageOut]

    class Config:
        from_attributes = True
        
class ConversationRenameRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)
