from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import documents
from app.routers import chatbot
from app.config import get_settings

settings = get_settings()

app = FastAPI(
    title="Iris Document Management API",
    description="API for managing documents with AI-powered search capabilities",
    version="1.0.0"
)

origins = [
    "null",
    "https://thelandofcolor.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allow specific origins
    allow_credentials=True, # Allows cookies/authorization headers
    allow_methods=["POST", "OPTIONS"], # MUST include OPTIONS for preflight requests
    allow_headers=["Content-Type", "Authorization"], # MUST include headers your JS client sends
)

# Include routers
app.include_router(documents.router)
app.include_router(chatbot.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)