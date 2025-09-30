from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
from ..database import get_db
from ..services.analytics_service import AnalyticsService
from ..schemas.analytics import (
    UserCreate, UserUpdate, UserResponse, QueryLogResponse, UserSessionResponse,
    AccessControlCreate, AccessControlUpdate, AccessControlResponse,
    DashboardSummary, QueryAnalytics, RateLimitStatus, DetailedQueryLog,
    DashboardFilters, BulkUserOperation
)

router = APIRouter(
    prefix="/dashboard",
    tags=["dashboard"],
)

def get_analytics_service(db: Session = Depends(get_db)) -> AnalyticsService:
    """Dependency to get analytics service."""
    return AnalyticsService(db)

# Dashboard Overview
@router.get("/summary", response_model=DashboardSummary)
async def get_dashboard_summary(
    analytics: AnalyticsService = Depends(get_analytics_service)
):
    """Get dashboard summary with key metrics."""
    return analytics.get_dashboard_summary()

@router.get("/analytics", response_model=List[QueryAnalytics])
async def get_query_analytics(
    days: int = Query(30, description="Number of days to analyze"),
    analytics: AnalyticsService = Depends(get_analytics_service)
):
    """Get query analytics over time."""
    return analytics.get_query_analytics(days)

# User Management
@router.get("/users", response_model=List[UserResponse])
async def get_users(
    skip: int = Query(0, description="Number of records to skip"),
    limit: int = Query(100, description="Maximum number of records to return"),
    analytics: AnalyticsService = Depends(get_analytics_service)
):
    """Get all users with pagination."""
    return analytics.get_users(skip, limit)

@router.get("/users/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: str,
    analytics: AnalyticsService = Depends(get_analytics_service)
):
    """Get specific user by ID."""
    user = analytics.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@router.post("/users", response_model=UserResponse)
async def create_user(
    user_data: UserCreate,
    analytics: AnalyticsService = Depends(get_analytics_service)
):
    """Create a new user."""
    # Check if user already exists
    existing_user = analytics.get_user(user_data.user_id)
    if existing_user:
        raise HTTPException(status_code=400, detail="User already exists")
    
    return analytics.create_user(user_data)

@router.put("/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: str,
    user_data: UserUpdate,
    analytics: AnalyticsService = Depends(get_analytics_service)
):
    """Update user information."""
    user = analytics.update_user(user_id, user_data)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@router.delete("/users/{user_id}")
async def delete_user(
    user_id: str,
    analytics: AnalyticsService = Depends(get_analytics_service)
):
    """Delete a user."""
    success = analytics.delete_user(user_id)
    if not success:
        raise HTTPException(status_code=404, detail="User not found")
    return {"message": "User deleted successfully"}

@router.post("/users/bulk")
async def bulk_user_operation(
    operation: BulkUserOperation,
    analytics: AnalyticsService = Depends(get_analytics_service)
):
    """Perform bulk operations on users."""
    results = []
    
    for user_id in operation.user_ids:
        try:
            if operation.operation == "activate":
                user = analytics.update_user(user_id, UserUpdate(is_active=True))
                results.append({"user_id": user_id, "status": "success", "action": "activated"})
            
            elif operation.operation == "deactivate":
                user = analytics.update_user(user_id, UserUpdate(is_active=False))
                results.append({"user_id": user_id, "status": "success", "action": "deactivated"})
            
            elif operation.operation == "set_limit":
                limit = operation.parameters.get("daily_query_limit", 100) if operation.parameters else 100
                user = analytics.update_user(user_id, UserUpdate(daily_query_limit=limit))
                results.append({"user_id": user_id, "status": "success", "action": f"limit set to {limit}"})
            
            elif operation.operation == "delete":
                success = analytics.delete_user(user_id)
                if success:
                    results.append({"user_id": user_id, "status": "success", "action": "deleted"})
                else:
                    results.append({"user_id": user_id, "status": "error", "action": "not found"})
            
            else:
                results.append({"user_id": user_id, "status": "error", "action": "unknown operation"})
        
        except Exception as e:
            results.append({"user_id": user_id, "status": "error", "action": str(e)})
    
    return {"results": results}

# Query Logs
@router.get("/queries", response_model=List[QueryLogResponse])
async def get_query_logs(
    skip: int = Query(0, description="Number of records to skip"),
    limit: int = Query(100, description="Maximum number of records to return"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    status: Optional[str] = Query(None, description="Filter by status"),
    start_date: Optional[datetime] = Query(None, description="Start date filter"),
    end_date: Optional[datetime] = Query(None, description="End date filter"),
    min_response_time: Optional[float] = Query(None, description="Minimum response time filter"),
    max_response_time: Optional[float] = Query(None, description="Maximum response time filter"),
    analytics: AnalyticsService = Depends(get_analytics_service)
):
    """Get query logs with filtering options."""
    filters = DashboardFilters(
        start_date=start_date,
        end_date=end_date,
        user_ids=[user_id] if user_id else None,
        status=status,
        min_response_time=min_response_time,
        max_response_time=max_response_time
    )
    return analytics.get_query_logs(filters, skip, limit)

@router.get("/queries/{query_id}", response_model=DetailedQueryLog)
async def get_detailed_query_log(
    query_id: int,
    analytics: AnalyticsService = Depends(get_analytics_service)
):
    """Get detailed query log with reasoning steps."""
    log = analytics.get_detailed_query_log(query_id)
    if not log:
        raise HTTPException(status_code=404, detail="Query log not found")
    return log

# Sessions
@router.get("/sessions", response_model=List[UserSessionResponse])
async def get_active_sessions(
    hours: int = Query(24, description="Hours to look back for active sessions"),
    analytics: AnalyticsService = Depends(get_analytics_service)
):
    """Get active user sessions."""
    return analytics.get_active_sessions(hours)

# Access Control
@router.get("/access-control/{user_id}", response_model=AccessControlResponse)
async def get_access_control(
    user_id: str,
    endpoint: str = Query("/query", description="Endpoint to check"),
    analytics: AnalyticsService = Depends(get_analytics_service)
):
    """Get access control settings for a user."""
    access_control = analytics.get_access_control(user_id, endpoint)
    if not access_control:
        raise HTTPException(status_code=404, detail="Access control not found")
    return access_control

@router.post("/access-control", response_model=AccessControlResponse)
async def set_access_control(
    access_data: AccessControlCreate,
    analytics: AnalyticsService = Depends(get_analytics_service)
):
    """Set access control for a user."""
    return analytics.set_access_control(access_data.user_id, access_data)

@router.put("/access-control/{user_id}", response_model=AccessControlResponse)
async def update_access_control(
    user_id: str,
    access_data: AccessControlUpdate,
    endpoint: str = Query("/query", description="Endpoint to update"),
    analytics: AnalyticsService = Depends(get_analytics_service)
):
    """Update access control settings."""
    # Get existing access control
    existing = analytics.get_access_control(user_id, endpoint)
    if not existing:
        # Create new access control with updated data
        create_data = AccessControlCreate(
            user_id=user_id,
            endpoint=endpoint,
            **access_data.dict(exclude_unset=True)
        )
        return analytics.set_access_control(user_id, create_data)
    
    # Update existing
    update_dict = access_data.dict(exclude_unset=True)
    create_data = AccessControlCreate(
        user_id=user_id,
        endpoint=endpoint,
        is_allowed=update_dict.get('is_allowed', existing.is_allowed),
        rate_limit_per_hour=update_dict.get('rate_limit_per_hour', existing.rate_limit_per_hour),
        rate_limit_per_day=update_dict.get('rate_limit_per_day', existing.rate_limit_per_day)
    )
    return analytics.set_access_control(user_id, create_data)

# Rate Limiting
@router.get("/rate-limit/{user_id}", response_model=RateLimitStatus)
async def check_rate_limit(
    user_id: str,
    analytics: AnalyticsService = Depends(get_analytics_service)
):
    """Check rate limit status for a user."""
    return analytics.check_rate_limit(user_id)

# Health Check
@router.get("/health")
async def health_check():
    """Health check endpoint for the dashboard."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "dashboard"
    }
