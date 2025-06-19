from fastapi import APIRouter, HTTPException, Body
from typing import Dict, Any
import logging
from ..services.chatbot_service import Iris, set_system_prompt, get_system_prompt
from ..schemas.chat import ChatRequest, ChatResponse, ChatHistory
from ..config import get_settings
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/query",
    tags=["chatbot"],
)

settings = get_settings()

class PromptRequest(BaseModel):
    prompt: str

@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat message and return a response."""
    try:
        # Initialize chatbot for this user
        chatbot = Iris({"user_id": request.user_id})
        
        # Process message with chatbot
        response = await chatbot.chat(request.message)
        
        return ChatResponse(response=response)
        
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your request"
        )

@router.get("/history/{user_id}", response_model=ChatHistory)
async def get_chat_history(user_id: str):
    """Get chat history for a user."""
    try:
        chatbot = Iris({"user_id": user_id})
        history = await chatbot.get_history()
        return history
    except Exception as e:
        logger.error(f"Error retrieving chat history: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while retrieving chat history"
        )

@router.post("/prompt")
async def update_system_prompt(request: PromptRequest):
    """Update the system prompt for the chatbot agent."""
    set_system_prompt(request.prompt)
    return {"message": "System prompt updated successfully."}

@router.get("/prompt")
async def read_system_prompt():
    """Get the current system prompt for the chatbot agent."""
    return {"prompt": get_system_prompt()}

# Export the Iris service
__all__ = ['Iris']
