import time
import json
from typing import Callable, Dict, Any, Optional
from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from sqlalchemy.orm import Session
from ..database import get_db, User
from ..services.analytics_service import AnalyticsService
from ..schemas.analytics import QueryLogCreate, UserCreate

class QueryAnalyticsMiddleware(BaseHTTPMiddleware):
    """Middleware to track all queries to /query endpoint with analytics."""
    
    def __init__(self, app, track_endpoints: list = None):
        super().__init__(app)
        self.track_endpoints = track_endpoints or ["/query"]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Only track specified endpoints
        if not any(request.url.path.startswith(endpoint) for endpoint in self.track_endpoints):
            return await call_next(request)

        # Skip tracking for non-POST requests to /query
        if request.url.path.startswith("/query") and request.method != "POST":
            return await call_next(request)

        # Skip tracking for dashboard endpoints
        if request.url.path.startswith("/dashboard"):
            return await call_next(request)
        
        start_time = time.time()
        
        # Get request data
        user_id = None
        session_id = None
        query_text = ""
        
        try:
            # Read request body
            body = await request.body()
            if body:
                try:
                    request_data = json.loads(body.decode())
                    user_id = request_data.get("session_id") or request_data.get("user_id")
                    session_id = request_data.get("session_id")
                    query_text = request_data.get("query", "")
                except json.JSONDecodeError:
                    pass
            
            # Recreate request with body for downstream processing
            async def receive():
                return {"type": "http.request", "body": body}
            
            request._receive = receive
            
        except Exception as e:
            # If we can't read the request, continue without tracking
            return await call_next(request)
        
        # Check rate limiting before processing
        if user_id:
            db = next(get_db())
            try:
                analytics_service = AnalyticsService(db)
                
                # Ensure user exists
                await self._ensure_user_exists(analytics_service, user_id, request)
                
                # Check rate limits
                rate_limit_status = analytics_service.check_rate_limit(user_id)
                if rate_limit_status.is_rate_limited:
                    # Log the rate-limited attempt
                    await self._log_query(
                        analytics_service, user_id, session_id, query_text,
                        None, None, time.time() - start_time, "rate_limited",
                        "Rate limit exceeded", request
                    )
                    
                    return JSONResponse(
                        status_code=429,
                        content={
                            "detail": "Rate limit exceeded",
                            "queries_this_hour": rate_limit_status.queries_this_hour,
                            "hourly_limit": rate_limit_status.hourly_limit,
                            "reset_time": rate_limit_status.reset_time_hour.isoformat()
                        }
                    )
                
                # Update session
                if session_id:
                    analytics_service.create_or_update_session(user_id, session_id)
                
            finally:
                db.close()
        
        # Process the request
        response = None
        error_message = None
        status = "success"
        
        try:
            response = await call_next(request)
            
            # Check if response indicates an error
            if response.status_code >= 400:
                status = "error"
                if hasattr(response, 'body'):
                    try:
                        error_data = json.loads(response.body.decode())
                        error_message = error_data.get('detail', f'HTTP {response.status_code}')
                    except:
                        error_message = f'HTTP {response.status_code}'
                else:
                    error_message = f'HTTP {response.status_code}'
        
        except Exception as e:
            status = "error"
            error_message = str(e)
            response = JSONResponse(
                status_code=500,
                content={"detail": "Internal server error"}
            )
        
        # Calculate response time
        response_time_ms = (time.time() - start_time) * 1000
        
        # Extract response data
        response_text = None
        reasoning_steps = None
        tokens_used = None
        model_used = None
        
        if response and status == "success":
            try:
                # Read response body if available
                if hasattr(response, 'body'):
                    response_data = json.loads(response.body.decode())
                    response_text = response_data.get('response', '')
                    
                    # Extract reasoning steps if available
                    reasoning_steps = response_data.get('reasoning_steps')
                    tokens_used = response_data.get('tokens_used')
                    model_used = response_data.get('model_used')
            except:
                pass
        
        # Log the query
        if user_id:
            db = next(get_db())
            try:
                analytics_service = AnalyticsService(db)
                await self._log_query(
                    analytics_service, user_id, session_id, query_text,
                    response_text, reasoning_steps, response_time_ms, status,
                    error_message, request, tokens_used, model_used
                )
            finally:
                db.close()
        
        return response
    
    async def _ensure_user_exists(self, analytics_service: AnalyticsService, user_id: str, request: Request):
        """Ensure user exists in database, create if not."""
        existing_user = analytics_service.get_user(user_id)
        if not existing_user:
            # Create user with default settings
            user_create = UserCreate(
                user_id=user_id,
                username=user_id,  # Use user_id as default username
                daily_query_limit=100
            )
            analytics_service.create_user(user_create)
    
    async def _log_query(
        self, 
        analytics_service: AnalyticsService,
        user_id: str,
        session_id: Optional[str],
        query_text: str,
        response_text: Optional[str],
        reasoning_steps: Optional[Dict[str, Any]],
        response_time_ms: float,
        status: str,
        error_message: Optional[str],
        request: Request,
        tokens_used: Optional[int] = None,
        model_used: Optional[str] = None
    ):
        """Log query to database."""
        try:
            query_log = QueryLogCreate(
                user_id=user_id,
                session_id=session_id,
                query_text=query_text,
                response_text=response_text,
                reasoning_steps=reasoning_steps,
                response_time_ms=response_time_ms,
                tokens_used=tokens_used,
                model_used=model_used,
                status=status,
                error_message=error_message,
                ip_address=self._get_client_ip(request),
                user_agent=request.headers.get("user-agent")
            )
            analytics_service.log_query(query_log)
        except Exception as e:
            # Don't let logging errors break the main request
            print(f"Error logging query: {e}")
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check for forwarded headers first
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fall back to direct client IP
        if hasattr(request, 'client') and request.client:
            return request.client.host
        
        return "unknown"

class ReasoningStepTracker:
    """Helper class to track reasoning steps during LLM processing."""
    
    def __init__(self):
        self.steps = []
        self.current_step = 0
    
    def add_step(self, step_type: str, content: str, tool_used: Optional[str] = None):
        """Add a reasoning step."""
        self.current_step += 1
        step = {
            "step_number": self.current_step,
            "step_type": step_type,
            "content": content,
            "timestamp": time.time(),
            "tool_used": tool_used
        }
        self.steps.append(step)
        return step
    
    def get_steps(self) -> Dict[str, Any]:
        """Get all reasoning steps."""
        return {
            "steps": self.steps,
            "total_steps": len(self.steps),
            "total_time": self.steps[-1]["timestamp"] - self.steps[0]["timestamp"] if self.steps else 0
        }
    
    def add_thought(self, content: str):
        """Add a thought step."""
        return self.add_step("thought", content)
    
    def add_action(self, content: str, tool_used: str):
        """Add an action step."""
        return self.add_step("action", content, tool_used)
    
    def add_observation(self, content: str):
        """Add an observation step."""
        return self.add_step("observation", content)
    
    def add_final_answer(self, content: str):
        """Add the final answer step."""
        return self.add_step("final_answer", content)

# Global reasoning tracker for the current request
reasoning_tracker = None

def get_reasoning_tracker() -> ReasoningStepTracker:
    """Get the current reasoning tracker."""
    global reasoning_tracker
    if reasoning_tracker is None:
        reasoning_tracker = ReasoningStepTracker()
    return reasoning_tracker

def reset_reasoning_tracker():
    """Reset the reasoning tracker for a new request."""
    global reasoning_tracker
    reasoning_tracker = ReasoningStepTracker()

def add_reasoning_step(step_type: str, content: str, tool_used: Optional[str] = None):
    """Add a reasoning step to the current tracker."""
    tracker = get_reasoning_tracker()
    return tracker.add_step(step_type, content, tool_used)
