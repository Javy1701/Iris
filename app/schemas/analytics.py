from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class UserBase(BaseModel):
    user_id: str = Field(..., description="Unique user identifier")
    username: Optional[str] = Field(None, description="Display name")
    email: Optional[str] = Field(None, description="User email")
    daily_query_limit: int = Field(100, description="Daily query limit")

class UserCreate(UserBase):
    is_admin: bool = Field(False, description="Admin privileges")

class UserUpdate(BaseModel):
    username: Optional[str] = None
    email: Optional[str] = None
    is_active: Optional[bool] = None
    is_admin: Optional[bool] = None
    daily_query_limit: Optional[int] = None

class UserResponse(UserBase):
    id: int
    is_active: bool
    is_admin: bool
    created_at: datetime
    last_active: datetime
    
    class Config:
        from_attributes = True

class QueryLogBase(BaseModel):
    user_id: str
    session_id: Optional[str] = None
    query_text: str
    response_text: Optional[str] = None
    reasoning_steps: Optional[Dict[str, Any]] = None
    response_time_ms: Optional[float] = None
    tokens_used: Optional[int] = None
    model_used: Optional[str] = None
    status: str = "success"
    error_message: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

class QueryLogCreate(QueryLogBase):
    pass

class QueryLogResponse(QueryLogBase):
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

class UserSessionBase(BaseModel):
    user_id: str
    session_id: str
    queries_count: int = 0

class UserSessionCreate(UserSessionBase):
    pass

class UserSessionResponse(UserSessionBase):
    id: int
    started_at: datetime
    last_activity: datetime
    is_active: bool
    
    class Config:
        from_attributes = True

class AccessControlBase(BaseModel):
    user_id: str
    endpoint: str = "/query"
    is_allowed: bool = True
    rate_limit_per_hour: int = 60
    rate_limit_per_day: int = 1000

class AccessControlCreate(AccessControlBase):
    pass

class AccessControlUpdate(BaseModel):
    is_allowed: Optional[bool] = None
    rate_limit_per_hour: Optional[int] = None
    rate_limit_per_day: Optional[int] = None

class AccessControlResponse(AccessControlBase):
    id: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

# Dashboard Analytics Schemas
class QueryStats(BaseModel):
    total_queries: int
    successful_queries: int
    failed_queries: int
    average_response_time: float
    total_tokens_used: int

class UserStats(BaseModel):
    user_id: str
    username: Optional[str]
    total_queries: int
    queries_today: int
    last_query: Optional[datetime]
    average_response_time: float
    success_rate: float

class DashboardSummary(BaseModel):
    total_users: int
    active_users_today: int
    total_queries_today: int
    total_queries_all_time: int
    average_response_time: float
    top_users: List[UserStats]
    recent_queries: List[QueryLogResponse]

class QueryAnalytics(BaseModel):
    date: str
    query_count: int
    unique_users: int
    average_response_time: float
    success_rate: float

class ReasoningStep(BaseModel):
    step_number: int
    step_type: str  # "thought", "action", "observation", "final_answer"
    content: str
    timestamp: Optional[datetime] = None
    tool_used: Optional[str] = None
    execution_time_ms: Optional[float] = None

class DetailedQueryLog(QueryLogResponse):
    reasoning_breakdown: Optional[List[ReasoningStep]] = None
    context_retrieved: Optional[List[str]] = None
    tools_used: Optional[List[str]] = None

# Rate Limiting Schemas
class RateLimitStatus(BaseModel):
    user_id: str
    queries_this_hour: int
    queries_today: int
    hourly_limit: int
    daily_limit: int
    is_rate_limited: bool
    reset_time_hour: datetime
    reset_time_day: datetime

class BulkUserOperation(BaseModel):
    user_ids: List[str]
    operation: str  # "activate", "deactivate", "set_limit", "delete"
    parameters: Optional[Dict[str, Any]] = None

class DashboardFilters(BaseModel):
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    user_ids: Optional[List[str]] = None
    status: Optional[str] = None
    min_response_time: Optional[float] = None
    max_response_time: Optional[float] = None
