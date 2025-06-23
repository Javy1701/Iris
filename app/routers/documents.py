from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from ..database import get_db
from ..services.document_service import DocumentService
from ..schemas.document import DocumentResponse, DocumentList

router = APIRouter(
    prefix="/documents",
    tags=["documents"]
)

@router.post("/", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload a new document."""
    service = DocumentService(db)
    return await service.upload_document(file)

@router.get("/", response_model=DocumentList)
def list_documents(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """List all documents."""
    service = DocumentService(db)
    documents = service.get_documents(skip=skip, limit=limit)
    total = len(documents)
    return DocumentList(documents=documents, total=total)

@router.get("/{document_id}", response_model=DocumentResponse)
def get_document(
    document_id: int,
    db: Session = Depends(get_db)
):
    """Get a specific document."""
    service = DocumentService(db)
    document = service.get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return document

@router.delete("/{document_id}")
def delete_document(
    document_id: int,
    db: Session = Depends(get_db)
):
    """Delete a document."""
    service = DocumentService(db)
    if not service.delete_document(document_id):
        raise HTTPException(status_code=404, detail="Document not found")
    return {"message": "Document deleted successfully"}

@router.put("/{document_id}", response_model=DocumentResponse)
async def update_document(
    document_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Update a document."""
    service = DocumentService(db)
    document = await service.update_document(document_id, file)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return document 