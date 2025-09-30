from fastapi import APIRouter, HTTPException, Body
from typing import Dict, Any
import logging
import time
from ..services.chatbot_service import Iris, set_system_prompt, get_system_prompt
from ..schemas.chat import ChatRequest, ChatResponse, ChatHistory
from ..config import get_settings
from ..middleware.analytics_middleware import get_reasoning_tracker, reset_reasoning_tracker
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
    print("request:", request)
    """Process a chat message and return a response."""

    # Reset reasoning tracker for this request
    reset_reasoning_tracker()
    start_time = time.time()

    try:
        # Initialize chatbot for this user
        chatbot = Iris({"user_id": request.session_id})

        # Add initial reasoning step
        reasoning_tracker = get_reasoning_tracker()
        reasoning_tracker.add_thought(f"Processing query from user {request.session_id}: {request.query}")

        # Process message with chatbot
        response = await chatbot.chat(request.query)

        # Add final reasoning step
        reasoning_tracker.add_final_answer(response)

        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000

        # Get reasoning steps for response
        reasoning_steps = reasoning_tracker.get_steps()

        # Enhanced response with metadata (middleware will capture this)
        enhanced_response = ChatResponse(
            response=response,
            message_id=f"{request.session_id}_{int(time.time())}"
        )

        # Add metadata to response for middleware to capture
        # Note: This will be captured by the analytics middleware
        enhanced_response.__dict__.update({
            "reasoning_steps": reasoning_steps,
            "processing_time_ms": processing_time,
            "model_used": settings.CHAT_OPENAI_MODEL_NAME,
            "tokens_used": None  # Could be enhanced to track actual token usage
        })

        return enhanced_response

    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")

        # Add error to reasoning tracker
        reasoning_tracker = get_reasoning_tracker()
        reasoning_tracker.add_step("error", f"Error occurred: {str(e)}")

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
