from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field

class Message(BaseModel):
    """Schema for a single chat message."""
    role: str = Field(..., description="Role of the message sender (user/assistant)")
    content: str = Field(..., description="Content of the message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of the message")

class ChatHistory(BaseModel):
    """Schema for chat history."""
    user_id: str = Field(..., description="ID of the user")
    messages: List[Message] = Field(default_factory=list, description="List of chat messages")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")

class ChatRequest(BaseModel):
    """Schema for chat request."""
    query: str = Field(..., description="User's message")
    session_id: str = Field(..., description="ID of the user")

class ChatResponse(BaseModel):
    """Schema for chat response."""
    response: str = Field(..., description="Assistant's response")
    message_id: Optional[str] = Field(None, description="ID of the message if needed")

class PromptRequest(BaseModel):
    prompt: str 