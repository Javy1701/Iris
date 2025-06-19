# Iris - Document Management System

A FastAPI-based document management system with AI-powered search capabilities.

## Features

- Document upload and management (PDF, CSV, DOCX, TXT)
- Vector-based document search using Pinecone
- Secure document storage with encryption
- Admin interface for document management
- Integration with OpenAI for enhanced search capabilities

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with the following variables:
```
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
DATABASE_URL=your_database_url
SECRET_KEY=your_secret_key
```

4. Run the application:
```bash
uvicorn app.main:app --reload
```

## API Documentation

Once the application is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Project Structure

```
app/
├── main.py              # FastAPI application entry point
├── config.py           # Configuration settings
├── database.py         # Database connection and models
├── schemas/            # Pydantic models
├── routers/            # API routes
├── services/           # Business logic
└── utils/              # Utility functions
``` 