import time
from typing import Dict, Optional
from fastapi import HTTPException, Request
from datetime import datetime, timedelta
from collections import defaultdict, deque
import asyncio

class RateLimiter:
    """In-memory rate limiter with sliding window algorithm."""
    
    def __init__(self):
        # Store request timestamps for each user
        self.user_requests: Dict[str, deque] = defaultdict(lambda: deque())
        self.lock = asyncio.Lock()
    
    async def is_allowed(
        self, 
        user_id: str, 
        requests_per_hour: int = 60, 
        requests_per_day: int = 1000
    ) -> tuple[bool, Dict[str, any]]:
        """
        Check if user is allowed to make a request based on rate limits.
        
        Returns:
            tuple: (is_allowed, rate_limit_info)
        """
        async with self.lock:
            now = time.time()
            hour_ago = now - 3600  # 1 hour ago
            day_ago = now - 86400  # 24 hours ago
            
            # Get user's request history
            user_queue = self.user_requests[user_id]
            
            # Remove old requests (older than 24 hours)
            while user_queue and user_queue[0] < day_ago:
                user_queue.popleft()
            
            # Count requests in the last hour and day
            requests_last_hour = sum(1 for timestamp in user_queue if timestamp >= hour_ago)
            requests_last_day = len(user_queue)
            
            # Check limits
            hour_exceeded = requests_last_hour >= requests_per_hour
            day_exceeded = requests_last_day >= requests_per_day
            
            is_allowed = not (hour_exceeded or day_exceeded)
            
            # If allowed, add current request to queue
            if is_allowed:
                user_queue.append(now)
            
            # Calculate reset times
            next_hour = datetime.fromtimestamp(now).replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            next_day = datetime.fromtimestamp(now).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            
            rate_limit_info = {
                "requests_last_hour": requests_last_hour,
                "requests_last_day": requests_last_day,
                "hourly_limit": requests_per_hour,
                "daily_limit": requests_per_day,
                "reset_time_hour": next_hour.isoformat(),
                "reset_time_day": next_day.isoformat(),
                "is_rate_limited": not is_allowed,
                "limit_type": "hour" if hour_exceeded else "day" if day_exceeded else None
            }
            
            return is_allowed, rate_limit_info
    
    async def get_user_stats(self, user_id: str) -> Dict[str, any]:
        """Get current rate limit stats for a user."""
        async with self.lock:
            now = time.time()
            hour_ago = now - 3600
            day_ago = now - 86400
            
            user_queue = self.user_requests[user_id]
            
            # Clean old requests
            while user_queue and user_queue[0] < day_ago:
                user_queue.popleft()
            
            requests_last_hour = sum(1 for timestamp in user_queue if timestamp >= hour_ago)
            requests_last_day = len(user_queue)
            
            return {
                "user_id": user_id,
                "requests_last_hour": requests_last_hour,
                "requests_last_day": requests_last_day,
                "last_request": datetime.fromtimestamp(user_queue[-1]).isoformat() if user_queue else None,
                "total_requests": len(user_queue)
            }
    
    async def reset_user_limits(self, user_id: str):
        """Reset rate limits for a specific user."""
        async with self.lock:
            if user_id in self.user_requests:
                self.user_requests[user_id].clear()
    
    async def cleanup_old_data(self):
        """Clean up old request data to prevent memory leaks."""
        async with self.lock:
            now = time.time()
            day_ago = now - 86400
            
            users_to_remove = []
            for user_id, user_queue in self.user_requests.items():
                # Remove old requests
                while user_queue and user_queue[0] < day_ago:
                    user_queue.popleft()
                
                # If no recent requests, remove user entirely
                if not user_queue:
                    users_to_remove.append(user_id)
            
            for user_id in users_to_remove:
                del self.user_requests[user_id]

class AccessController:
    """Manages user access permissions and restrictions."""
    
    def __init__(self):
        # In-memory storage for access rules
        # In production, this should be backed by a database
        self.access_rules: Dict[str, Dict[str, any]] = {}
        self.blocked_users: set = set()
        self.blocked_ips: set = set()
        self.lock = asyncio.Lock()
    
    async def is_user_allowed(self, user_id: str, endpoint: str = "/query") -> tuple[bool, Optional[str]]:
        """
        Check if user has access to the endpoint.
        
        Returns:
            tuple: (is_allowed, reason_if_blocked)
        """
        async with self.lock:
            # Check if user is globally blocked
            if user_id in self.blocked_users:
                return False, "User is blocked"
            
            # Check endpoint-specific rules
            user_rules = self.access_rules.get(user_id, {})
            endpoint_rules = user_rules.get(endpoint, {})
            
            # Default to allowed if no specific rules
            is_allowed = endpoint_rules.get("allowed", True)
            
            if not is_allowed:
                return False, "Access denied for this endpoint"
            
            return True, None
    
    async def is_ip_allowed(self, ip_address: str) -> tuple[bool, Optional[str]]:
        """Check if IP address is allowed."""
        async with self.lock:
            if ip_address in self.blocked_ips:
                return False, "IP address is blocked"
            return True, None
    
    async def block_user(self, user_id: str, reason: str = "Manual block"):
        """Block a user from accessing the system."""
        async with self.lock:
            self.blocked_users.add(user_id)
            # Log the blocking action
            print(f"User {user_id} blocked: {reason}")
    
    async def unblock_user(self, user_id: str):
        """Unblock a user."""
        async with self.lock:
            self.blocked_users.discard(user_id)
            print(f"User {user_id} unblocked")
    
    async def block_ip(self, ip_address: str, reason: str = "Manual block"):
        """Block an IP address."""
        async with self.lock:
            self.blocked_ips.add(ip_address)
            print(f"IP {ip_address} blocked: {reason}")
    
    async def unblock_ip(self, ip_address: str):
        """Unblock an IP address."""
        async with self.lock:
            self.blocked_ips.discard(ip_address)
            print(f"IP {ip_address} unblocked")
    
    async def set_user_access_rule(
        self, 
        user_id: str, 
        endpoint: str, 
        allowed: bool, 
        rate_limit_hour: int = 60,
        rate_limit_day: int = 1000
    ):
        """Set access rules for a user and endpoint."""
        async with self.lock:
            if user_id not in self.access_rules:
                self.access_rules[user_id] = {}
            
            self.access_rules[user_id][endpoint] = {
                "allowed": allowed,
                "rate_limit_hour": rate_limit_hour,
                "rate_limit_day": rate_limit_day,
                "updated_at": datetime.utcnow().isoformat()
            }
    
    async def get_user_access_rules(self, user_id: str) -> Dict[str, any]:
        """Get access rules for a user."""
        async with self.lock:
            return self.access_rules.get(user_id, {})
    
    async def get_rate_limits_for_user(self, user_id: str, endpoint: str = "/query") -> tuple[int, int]:
        """Get rate limits for a user and endpoint."""
        async with self.lock:
            user_rules = self.access_rules.get(user_id, {})
            endpoint_rules = user_rules.get(endpoint, {})
            
            hour_limit = endpoint_rules.get("rate_limit_hour", 60)
            day_limit = endpoint_rules.get("rate_limit_day", 1000)
            
            return hour_limit, day_limit

# Global instances
rate_limiter = RateLimiter()
access_controller = AccessController()

async def check_access_and_rate_limit(
    user_id: str, 
    ip_address: str, 
    endpoint: str = "/query"
) -> tuple[bool, Dict[str, any]]:
    """
    Comprehensive access and rate limit check.
    
    Returns:
        tuple: (is_allowed, info_dict)
    """
    # Check IP access
    ip_allowed, ip_reason = await access_controller.is_ip_allowed(ip_address)
    if not ip_allowed:
        return False, {
            "error": "access_denied",
            "reason": ip_reason,
            "type": "ip_blocked"
        }
    
    # Check user access
    user_allowed, user_reason = await access_controller.is_user_allowed(user_id, endpoint)
    if not user_allowed:
        return False, {
            "error": "access_denied",
            "reason": user_reason,
            "type": "user_blocked"
        }
    
    # Check rate limits
    hour_limit, day_limit = await access_controller.get_rate_limits_for_user(user_id, endpoint)
    rate_allowed, rate_info = await rate_limiter.is_allowed(user_id, hour_limit, day_limit)
    
    if not rate_allowed:
        return False, {
            "error": "rate_limit_exceeded",
            "reason": f"Rate limit exceeded ({rate_info['limit_type']})",
            "type": "rate_limited",
            **rate_info
        }
    
    return True, {
        "status": "allowed",
        "rate_limit_info": rate_info
    }

# Cleanup task to run periodically
async def cleanup_task():
    """Periodic cleanup task to prevent memory leaks."""
    while True:
        await asyncio.sleep(3600)  # Run every hour
        await rate_limiter.cleanup_old_data()
        print("Rate limiter cleanup completed")
