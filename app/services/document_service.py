import os
import uuid
import pinecone
from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy.orm import Session
from fastapi import UploadFile, HTTPException
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone as PineconeClient, ServerlessSpec  # Renamed to avoid conflict
from ..database import Document
from ..config import get_settings
from ..utils.encryption import encrypt_filename

settings = get_settings()

# Instantiate a Pinecone client
# Note: Using PineconeClient to avoid naming conflict with the pinecone module
pc = PineconeClient(api_key=settings.PINECONE_API_KEY)

# Check if the index exists and create it if it doesn't
if settings.PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=settings.PINECONE_INDEX_NAME,
        dimension=1536,  # Dimensionality of text-embedding-ada-002
        metric='dotproduct',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(
    model=settings.EMBEDDING_MODEL_NAME
)


class DocumentService:
    def __init__(self, db: Session):
        self.db = db
        self.upload_dir = "uploads"
        # Use the instantiated Pinecone client
        self.pinecone_client = pc
        # Get the Pinecone index object
        self.pinecone_index = self.pinecone_client.Index(settings.PINECONE_INDEX_NAME)
        os.makedirs(self.upload_dir, exist_ok=True)

    def _get_document_loader(self, file_path: str, file_type: str):
        if file_type == "pdf":
            return PyPDFLoader(file_path)
        elif file_type == "txt":
            return TextLoader(file_path)
        elif file_type == "csv":
            return CSVLoader(file_path)
        elif file_type == "docx":
            return Docx2txtLoader(file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")

    async def upload_document(self, file: UploadFile) -> Document:
        file_type = file.filename.split(".")[-1].lower()
        if file_type not in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        encrypted_name = encrypt_filename(file.filename)
        file_path = os.path.join(self.upload_dir, encrypted_name)

        try:
            # Save file content
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)

            # Process document for vectorization
            loader = self._get_document_loader(file_path, file_type)
            docs = loader.load()

            # Add metadata to each document before splitting
            for doc in docs:
                doc.metadata['document_id'] = encrypted_name

            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(docs)

            # Instantiate the PineconeVectorStore object once
            vector_store = PineconeVectorStore(
                index_name=settings.PINECONE_INDEX_NAME,
                embedding=embeddings,
                namespace=settings.PINECONE_NAMESPACE
            )

            # Define a safe batch size
            batch_size = 100  # Process 100 chunks at a time

            # Loop through the chunks in batches
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                print(f"Processing batch {i // batch_size + 1}/{(len(chunks) - 1) // batch_size + 1}...")
                # Use add_documents to upload the current batch
                vector_store.add_documents(batch_chunks)

            # Create document record in DB only after successful processing
            document = Document(
                original_name=file.filename,
                encrypted_name=encrypted_name,
                file_type=file_type
            )

            self.db.add(document)
            self.db.commit()
            self.db.refresh(document)

        except Exception as e:
            # Clean up file on failure
            if os.path.exists(file_path):
                os.remove(file_path)
            # Re-raise exception to be handled by the endpoint
            raise HTTPException(status_code=500, detail=f"Failed to process and embed document: {str(e)}")

        return document

    def get_documents(self, skip: int = 0, limit: int = 100) -> List[Document]:
        return self.db.query(Document).filter(
            Document.is_deleted == False
        ).offset(skip).limit(limit).all()

    def get_document(self, document_id: int) -> Optional[Document]:
        return self.db.query(Document).filter(
            Document.id == document_id,
            Document.is_deleted == False
        ).first()

    def delete_document(self, document_id: int) -> bool:
        document = self.get_document(document_id)
        if not document:
            return False

        # --- CORRECTED PINECONE DELETION LOGIC ---
        try:
            # Use the correct metadata filter key ('document_id')
            delete_filter = {'document_id': document.encrypted_name}

            # Delete from Pinecone using the correct filter and namespace
            self.pinecone_index.delete(
                filter=delete_filter,
                namespace=settings.PINECONE_NAMESPACE
            )
            print(f"Successfully deleted vectors from Pinecone for document_id: {document.encrypted_name}")

        except Exception as e:
            # Log the error but continue with cleanup
            print(f"Error deleting from Pinecone, but proceeding with cleanup: {str(e)}")

        # Mark as deleted in the database
        document.is_deleted = True
        document.deleted_date = datetime.now(timezone.utc)
        self.db.commit()

        # Delete the physical file
        file_path = os.path.join(self.upload_dir, document.encrypted_name)
        if os.path.exists(file_path):
            os.remove(file_path)

        return True

    async def update_document(self, document_id: int, file: UploadFile) -> Optional[Document]:
        # First, delete the old document and its associated data
        if not self.delete_document(document_id):
            raise HTTPException(status_code=404, detail="Document to update not found")

        # Then, upload the new document.
        # This re-uses the upload logic, including vectorization.
        # Note: This will create a new Document record.
        # If you need to maintain the same document_id, the logic would need to be more complex.
        return await self.upload_document(file)
