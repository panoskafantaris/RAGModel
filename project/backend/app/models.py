from pydantic import BaseModel
from typing import List, Optional

class NewChatRequest(BaseModel):
    title: Optional[str] = None

class MessageRequest(BaseModel):
    role: str # "user" or "system"
    content: str

class ChatSummary(BaseModel):
    id: str
    title: str
    last_updated: str

class UploadResponse(BaseModel):
    filename: str
    status: str