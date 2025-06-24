from fastapi import FastAPI
from app.routers import documents
from app.routers import chatbot
from app.config import get_settings

settings = get_settings()

app = FastAPI(
    title="Iris Document Management API",
    description="API for managing documents with AI-powered search capabilities",
    version="1.0.0"
)

# Include routers
app.include_router(documents.router)
app.include_router(chatbot.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)