from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class DocumentBase(BaseModel):
    original_name: str
    file_type: str

class DocumentCreate(DocumentBase):
    pass

class DocumentUpdate(DocumentBase):
    pass

class DocumentInDB(DocumentBase):
    id: int
    encrypted_name: str
    upload_date: datetime
    is_deleted: bool
    deleted_date: Optional[datetime] = None

    class Config:
        from_attributes = True

class DocumentResponse(DocumentInDB):
    pass

class DocumentList(BaseModel):
    documents: list[DocumentResponse]
    total: int 