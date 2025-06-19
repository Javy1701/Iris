# Iris - AI Color Expert & Document Management System

Iris is an intelligent AI assistant with a PhD in Color, designed to provide expert color consultation and document management capabilities. Built with FastAPI and powered by OpenAI, Pinecone, and LangChain, Iris offers comprehensive color analysis, professional color consultation, and secure document storage with AI-powered search.

## 🌟 Features

### Color Expert Capabilities
- **Professional Color Analysis**: Get detailed color DNA attributes including LRV, hue, value, chroma, and Munsell notation
- **Color Comparison**: Systematic analysis of color attributes with visual perception insights
- **Color Bracket Logic**: Precise numerical data and determination methodology for color classifications
- **Professional Consultation**: Expert guidance on color selection and application
- **Smart Redirects**: Automatic redirection to specialized tools like Paint Color DNA Table and Camp Chroma

### Document Management
- **Multi-format Support**: Upload and manage PDF, CSV, DOCX, and TXT files
- **Vector-based Search**: AI-powered document search using Pinecone vector database
- **Secure Storage**: Encrypted document storage with user authentication
- **Admin Interface**: Comprehensive document management dashboard

### Technical Features
- **Conversation Memory**: Persistent chat history with user-specific sessions
- **RESTful API**: Complete FastAPI backend with Swagger documentation
- **Web Interface**: Modern Streamlit frontend for intuitive user interaction
- **Authentication**: JWT-based user authentication and authorization
- **Real-time Processing**: Asynchronous chat processing with context retrieval

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key
- Pinecone API key and index
- Google Search API (optional, for web search functionality)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd Iris
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**
Create a `.env` file in the root directory:
```env
# API Keys
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CSE_ID=your_google_cse_id

# Pinecone Configuration
PINECONE_INDEX_NAME=your_pinecone_index_name
PINECONE_NAMESPACE=your_pinecone_namespace

# Database
DATABASE_URL=sqlite:///./iris.db

# Security
SECRET_KEY=your_secret_key_here

# OpenAI Models
EMBEDDING_MODEL_NAME=text-embedding-ada-002
CHAT_OPENAI_MODEL_NAME=gpt-4
CHAT_OPENAI_TEMPERATURE=0.7

# Authentication
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

5. **Initialize the database**
```bash
python -c "from app.database import Base, engine; Base.metadata.create_all(engine)"
```

6. **Start the FastAPI backend**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

7. **Start the Streamlit frontend** (in a new terminal)
```bash
streamlit run app/streamlit_app.py
```

## 📖 Usage

### Web Interface
- Open your browser and navigate to `http://localhost:8501`
- Enter a User ID to start or continue a chat session
- Ask Iris about colors, upload documents, or request professional consultation

### API Endpoints

#### Authentication
- `POST /token` - Login and get access token
- `POST /users/` - Create new user account
- `GET /users/me/` - Get current user information

#### Chatbot
- `POST /query/` - Send a message to Iris
- `GET /query/history/{user_id}` - Get chat history for a user
- `POST /query/prompt` - Update system prompt
- `GET /query/prompt` - Get current system prompt

#### Documents
- `POST /documents/upload` - Upload a document
- `GET /documents/` - List all documents
- `GET /documents/{document_id}` - Get document details
- `DELETE /documents/{document_id}` - Delete a document

### API Documentation
Once the application is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🏗️ Project Structure

```
Iris/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application entry point
│   ├── config.py              # Configuration settings
│   ├── database.py            # Database models and connection
│   ├── streamlit_app.py       # Streamlit web interface
│   ├── chatbot_test.py        # Chatbot testing utilities
│   ├── routers/
│   │   ├── chatbot.py         # Chatbot API endpoints
│   │   ├── documents.py       # Document management endpoints
│   │   └── component.tsx      # Frontend components
│   ├── schemas/
│   │   ├── chat.py           # Chat-related Pydantic models
│   │   ├── document.py       # Document Pydantic models
│   │   └── user.py           # User Pydantic models
│   ├── services/
│   │   ├── chatbot_service.py # Core chatbot logic
│   │   └── document_service.py # Document processing logic
│   └── utils/
│       ├── auth.py           # Authentication utilities
│       └── encryption.py     # Encryption utilities
├── Additional Documents/      # Color data and documentation
├── uploads/                   # Document storage directory
├── iris.db                    # SQLite database
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## 🎨 Color Expert Features

Iris provides professional color consultation with:

- **Color DNA Analysis**: Scientific color measurements (L, C, h, LRV, Munsell)
- **Professional Summaries**: Comprehensive color information with application guidance
- **Comparison Tools**: Side-by-side color analysis with visual perception insights
- **Bracket Logic**: Precise color classification using numerical data
- **Expert Guidance**: Professional recommendations for color selection

### Example Color Analysis
```
Nantucket Gray – 2139-50 – Benjamin Moore
 • L: 80.5 | C: 6.2 | h: 92.3°
 • Munsell: 10 YR / 8.5 Value / 1 Chroma
 • LRV: 56
 • HEX: #AEAA93

This sophisticated low-chroma, warm green-yellow neutral exhibits exceptional 
versatility due to its balanced undertones and moderate light reflectance.
```

## 🔧 Configuration

### Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key for GPT and embedding models
- `PINECONE_API_KEY`: Pinecone API key for vector database
- `PINECONE_INDEX_NAME`: Name of your Pinecone index
- `PINECONE_NAMESPACE`: Namespace for document storage
- `SECRET_KEY`: Secret key for JWT token generation
- `DATABASE_URL`: Database connection string (defaults to SQLite)

### Model Configuration
- `EMBEDDING_MODEL_NAME`: OpenAI embedding model (default: text-embedding-ada-002)
- `CHAT_OPENAI_MODEL_NAME`: OpenAI chat model (default: gpt-4)
- `CHAT_OPENAI_TEMPERATURE`: Model temperature for response creativity

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Check the API documentation at http://localhost:8000/docs
- Review the color data in the `Additional Documents/` folder
- Contact the development team for technical issues

## 🔮 Future Enhancements

- Enhanced color visualization tools
- Integration with paint brand APIs
- Mobile application
- Advanced document processing capabilities
- Multi-language support
- Real-time collaboration features 