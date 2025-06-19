from pydantic_settings import BaseSettings
from functools import lru_cache
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # API Keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY")
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./iris.db")
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Pinecone
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME")
    PINECONE_DIMENSION: int = 1536  # OpenAI embedding dimension
    PINECONE_NAMESPACE: str = os.getenv("PINECONE_NAMESPACE")
    
    # File Upload
    MAX_UPLOAD_SIZE: int = 100 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: set = {"pdf", "csv", "docx", "txt"}

    # Embedding model
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME")

    CHAT_OPENAI_TEMPERATURE: float = os.getenv("CHAT_OPENAI_TEMPERATURE")
    CHAT_OPENAI_MODEL_NAME: str = os.getenv("CHAT_OPENAI_MODEL_NAME")
    
    class Config:
        case_sensitive = True

@lru_cache()
def get_settings():
    return Settings() 