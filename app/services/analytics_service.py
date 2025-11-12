import json
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, desc
from ..database import User, QueryLog, UserSession, AccessControl
from ..schemas.analytics import (
    UserCreate, UserUpdate, UserResponse, QueryLogCreate, QueryLogResponse,
    UserSessionCreate, UserSessionResponse, AccessControlCreate, AccessControlUpdate,
    AccessControlResponse, DashboardSummary, QueryStats, UserStats, QueryAnalytics,
    RateLimitStatus, DetailedQueryLog, ReasoningStep, DashboardFilters
)

class AnalyticsService:
    def __init__(self, db: Session):
        self.db = db

    # User Management
    def create_user(self, user_data: UserCreate) -> UserResponse:
        """Create a new user."""
        db_user = User(**user_data.dict())
        self.db.add(db_user)
        self.db.commit()
        self.db.refresh(db_user)
        return UserResponse.model_validate(db_user)

    def get_user(self, user_id: str) -> Optional[UserResponse]:
        """Get user by user_id."""
        user = self.db.query(User).filter(User.user_id == user_id).first()
        return UserResponse.model_validate(user) if user else None

    def get_users(self, skip: int = 0, limit: int = 100) -> List[UserResponse]:
        """Get all users with pagination."""
        users = self.db.query(User).offset(skip).limit(limit).all()
        return [UserResponse.model_validate(user) for user in users]

    def update_user(self, user_id: str, user_data: UserUpdate) -> Optional[UserResponse]:
        """Update user information."""
        user = self.db.query(User).filter(User.user_id == user_id).first()
        if not user:
            return None
        
        for field, value in user_data.dict(exclude_unset=True).items():
            setattr(user, field, value)
        
        self.db.commit()
        self.db.refresh(user)
        return UserResponse.model_validate(user)

    def delete_user(self, user_id: str) -> bool:
        """Delete a user."""
        user = self.db.query(User).filter(User.user_id == user_id).first()
        if not user:
            return False
        
        self.db.delete(user)
        self.db.commit()
        return True

    # Query Logging
    def log_query(self, query_data: QueryLogCreate) -> QueryLogResponse:
        """Log a query with all metadata."""
        db_query = QueryLog(**query_data.dict())
        self.db.add(db_query)
        self.db.commit()
        self.db.refresh(db_query)
        return QueryLogResponse.model_validate(db_query)

    def get_query_logs(self, filters: DashboardFilters, skip: int = 0, limit: int = 100) -> List[QueryLogResponse]:
        """Get query logs with filtering."""
        query = self.db.query(QueryLog)
        
        if filters.start_date:
            query = query.filter(QueryLog.created_at >= filters.start_date)
        if filters.end_date:
            query = query.filter(QueryLog.created_at <= filters.end_date)
        if filters.user_ids:
            query = query.filter(QueryLog.user_id.in_(filters.user_ids))
        if filters.status:
            query = query.filter(QueryLog.status == filters.status)
        if filters.min_response_time:
            query = query.filter(QueryLog.response_time_ms >= filters.min_response_time)
        if filters.max_response_time:
            query = query.filter(QueryLog.response_time_ms <= filters.max_response_time)
        
        logs = query.order_by(desc(QueryLog.created_at)).offset(skip).limit(limit).all()
        return [QueryLogResponse.model_validate(log) for log in logs]

    def get_detailed_query_log(self, query_id: int) -> Optional[DetailedQueryLog]:
        """Get detailed query log with reasoning breakdown."""
        log = self.db.query(QueryLog).filter(QueryLog.id == query_id).first()
        if not log:
            return None
        
        detailed_log = DetailedQueryLog.model_validate(log)
        
        # Parse reasoning steps if available
        if log.reasoning_steps:
            try:
                reasoning_data = json.loads(log.reasoning_steps) if isinstance(log.reasoning_steps, str) else log.reasoning_steps
                detailed_log.reasoning_breakdown = self._parse_reasoning_steps(reasoning_data)
            except (json.JSONDecodeError, TypeError):
                pass
        
        return detailed_log

    def _parse_reasoning_steps(self, reasoning_data: Dict[str, Any]) -> List[ReasoningStep]:
        """Parse reasoning steps from LLM response."""
        steps = []
        if isinstance(reasoning_data, dict):
            # Handle different formats of reasoning data
            if 'steps' in reasoning_data:
                for i, step in enumerate(reasoning_data['steps']):
                    steps.append(ReasoningStep(
                        step_number=i + 1,
                        step_type=step.get('type', 'thought'),
                        content=step.get('content', ''),
                        tool_used=step.get('tool'),
                        execution_time_ms=step.get('execution_time')
                    ))
        return steps

    # Session Management
    def create_or_update_session(self, user_id: str, session_id: str) -> UserSessionResponse:
        """Create or update user session."""
        session = self.db.query(UserSession).filter(UserSession.session_id == session_id).first()
        
        if session:
            session.last_activity = datetime.now(timezone.utc)
            session.queries_count += 1
        else:
            session = UserSession(
                user_id=user_id,
                session_id=session_id,
                queries_count=1
            )
            self.db.add(session)
        
        self.db.commit()
        self.db.refresh(session)
        return UserSessionResponse.model_validate(session)

    def get_active_sessions(self, hours: int = 24) -> List[UserSessionResponse]:
        """Get active sessions within specified hours."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        sessions = self.db.query(UserSession).filter(
            and_(UserSession.last_activity >= cutoff, UserSession.is_active == True)
        ).all()
        return [UserSessionResponse.model_validate(session) for session in sessions]

    # Access Control
    def get_access_control(self, user_id: str, endpoint: str = "/query") -> Optional[AccessControlResponse]:
        """Get access control settings for user."""
        access = self.db.query(AccessControl).filter(
            and_(AccessControl.user_id == user_id, AccessControl.endpoint == endpoint)
        ).first()
        return AccessControlResponse.model_validate(access) if access else None

    def set_access_control(self, user_id: str, access_data: AccessControlCreate) -> AccessControlResponse:
        """Set access control for user."""
        existing = self.db.query(AccessControl).filter(
            and_(AccessControl.user_id == user_id, AccessControl.endpoint == access_data.endpoint)
        ).first()
        
        if existing:
            for field, value in access_data.dict().items():
                setattr(existing, field, value)
            existing.updated_at = datetime.now(timezone.utc)
            db_access = existing
        else:
            db_access = AccessControl(**access_data.dict())
            self.db.add(db_access)
        
        self.db.commit()
        self.db.refresh(db_access)
        return AccessControlResponse.model_validate(db_access)

    # Rate Limiting
    def check_rate_limit(self, user_id: str) -> RateLimitStatus:
        """Check if user has exceeded rate limits."""
        now = datetime.now(timezone.utc)
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(days=1)
        
        # Get access control settings
        access_control = self.get_access_control(user_id)
        hourly_limit = access_control.rate_limit_per_hour if access_control else 60
        daily_limit = access_control.rate_limit_per_day if access_control else 1000
        
        # Count queries in last hour and day
        queries_this_hour = self.db.query(QueryLog).filter(
            and_(QueryLog.user_id == user_id, QueryLog.created_at >= hour_ago)
        ).count()
        
        queries_today = self.db.query(QueryLog).filter(
            and_(QueryLog.user_id == user_id, QueryLog.created_at >= day_ago)
        ).count()
        
        is_rate_limited = queries_this_hour >= hourly_limit or queries_today >= daily_limit
        
        return RateLimitStatus(
            user_id=user_id,
            queries_this_hour=queries_this_hour,
            queries_today=queries_today,
            hourly_limit=hourly_limit,
            daily_limit=daily_limit,
            is_rate_limited=is_rate_limited,
            reset_time_hour=now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1),
            reset_time_day=now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        )

    # Analytics and Dashboard
    def get_dashboard_summary(self) -> DashboardSummary:
        """Get dashboard summary statistics."""
        now = datetime.now(timezone.utc)
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Basic counts
        total_users = self.db.query(User).count()
        active_users_today = self.db.query(func.count(func.distinct(QueryLog.user_id))).filter(
            QueryLog.created_at >= today
        ).scalar()
        
        total_queries_today = self.db.query(QueryLog).filter(QueryLog.created_at >= today).count()
        total_queries_all_time = self.db.query(QueryLog).count()
        
        # Average response time
        avg_response_time = self.db.query(func.avg(QueryLog.response_time_ms)).filter(
            QueryLog.response_time_ms.isnot(None)
        ).scalar() or 0
        
        # Top users
        top_users = self._get_top_users(limit=5)
        
        # Recent queries
        recent_queries = self.get_query_logs(DashboardFilters(), limit=10)
        
        return DashboardSummary(
            total_users=total_users,
            active_users_today=active_users_today,
            total_queries_today=total_queries_today,
            total_queries_all_time=total_queries_all_time,
            average_response_time=avg_response_time,
            top_users=top_users,
            recent_queries=recent_queries
        )

    def _get_top_users(self, limit: int = 10) -> List[UserStats]:
        """Get top users by query count."""
        # This is a simplified version - you might want to make this more sophisticated
        user_query_counts = self.db.query(
            QueryLog.user_id,
            func.count(QueryLog.id).label('total_queries'),
            func.avg(QueryLog.response_time_ms).label('avg_response_time'),
            func.max(QueryLog.created_at).label('last_query')
        ).group_by(QueryLog.user_id).order_by(desc('total_queries')).limit(limit).all()
        
        top_users = []
        for user_data in user_query_counts:
            user = self.db.query(User).filter(User.user_id == user_data.user_id).first()
            
            # Calculate success rate
            total_queries = user_data.total_queries
            successful_queries = self.db.query(QueryLog).filter(
                and_(QueryLog.user_id == user_data.user_id, QueryLog.status == 'success')
            ).count()
            success_rate = (successful_queries / total_queries * 100) if total_queries > 0 else 0
            
            # Queries today
            today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
            queries_today = self.db.query(QueryLog).filter(
                and_(QueryLog.user_id == user_data.user_id, QueryLog.created_at >= today)
            ).count()
            
            top_users.append(UserStats(
                user_id=user_data.user_id,
                username=user.username if user else None,
                total_queries=total_queries,
                queries_today=queries_today,
                last_query=user_data.last_query,
                average_response_time=user_data.avg_response_time or 0,
                success_rate=success_rate
            ))
        
        return top_users

    def get_query_analytics(self, days: int = 30) -> List[QueryAnalytics]:
        """Get query analytics over time."""
        from sqlalchemy import case

        start_date = datetime.now(timezone.utc) - timedelta(days=days)

        # Group by date
        daily_stats = self.db.query(
            func.date(QueryLog.created_at).label('date'),
            func.count(QueryLog.id).label('query_count'),
            func.count(func.distinct(QueryLog.user_id)).label('unique_users'),
            func.avg(QueryLog.response_time_ms).label('avg_response_time'),
            func.sum(case((QueryLog.status == 'success', 1), else_=0)).label('successful_queries')
        ).filter(QueryLog.created_at >= start_date).group_by(func.date(QueryLog.created_at)).all()
        
        analytics = []
        for stat in daily_stats:
            success_rate = (stat.successful_queries / stat.query_count * 100) if stat.query_count > 0 else 0
            analytics.append(QueryAnalytics(
                date=stat.date.isoformat(),
                query_count=stat.query_count,
                unique_users=stat.unique_users,
                average_response_time=stat.avg_response_time or 0,
                success_rate=success_rate
            ))
        
        return analytics
