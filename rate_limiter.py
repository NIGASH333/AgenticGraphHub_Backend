"""
Rate Limiting Module for OpenAI API Protection

Implements multiple rate limiting strategies to protect against:
- API quota exhaustion
- Cost overruns
- Abuse and spam
"""

import time
import asyncio
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    # Per-minute limits
    requests_per_minute: int = 10
    tokens_per_minute: int = 50000
    
    # Per-hour limits
    requests_per_hour: int = 100
    tokens_per_hour: int = 200000
    
    # Per-day limits
    requests_per_day: int = 1000
    tokens_per_day: int = 1000000
    
    # Cost limits (in USD)
    cost_per_day: float = 10.0
    
    # Burst protection
    max_burst_requests: int = 5
    burst_window_seconds: int = 10


@dataclass
class RateLimitStatus:
    """Current rate limit status for a user/IP."""
    requests_minute: deque = field(default_factory=lambda: deque())
    requests_hour: deque = field(default_factory=lambda: deque())
    requests_day: deque = field(default_factory=lambda: deque())
    
    tokens_minute: int = 0
    tokens_hour: int = 0
    tokens_day: int = 0
    
    cost_day: float = 0.0
    
    burst_requests: deque = field(default_factory=lambda: deque())
    last_request_time: float = 0.0


class RateLimiter:
    """Advanced rate limiter with multiple strategies."""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.user_limits: Dict[str, RateLimitStatus] = defaultdict(RateLimitStatus)
        self.lock = asyncio.Lock()
        
        # Token cost estimates (per 1K tokens)
        self.token_costs = {
            'gpt-3.5-turbo': 0.0015,  # $0.0015 per 1K tokens
            'gpt-4': 0.03,             # $0.03 per 1K tokens
            'gpt-4-turbo': 0.01,       # $0.01 per 1K tokens
        }
    
    async def check_rate_limit(self, 
                             user_id: str, 
                             estimated_tokens: int = 1000,
                             model: str = 'gpt-3.5-turbo') -> Tuple[bool, str, Dict]:
        """
        Check if request is within rate limits.
        
        Returns:
            (allowed, reason, status_info)
        """
        async with self.lock:
            current_time = time.time()
            user_status = self.user_limits[user_id]
            
            # Clean old entries
            self._cleanup_old_entries(user_status, current_time)
            
            # Check burst protection
            if not self._check_burst_limit(user_status, current_time):
                return False, "Burst limit exceeded. Please slow down.", self._get_status_info(user_status)
            
            # Check per-minute limits
            if not self._check_minute_limit(user_status, estimated_tokens):
                return False, "Rate limit exceeded: too many requests or tokens per minute.", self._get_status_info(user_status)
            
            # Check per-hour limits
            if not self._check_hour_limit(user_status, estimated_tokens):
                return False, "Rate limit exceeded: too many requests or tokens per hour.", self._get_status_info(user_status)
            
            # Check per-day limits
            if not self._check_day_limit(user_status, estimated_tokens):
                return False, "Rate limit exceeded: too many requests or tokens per day.", self._get_status_info(user_status)
            
            # Check cost limits
            estimated_cost = self._calculate_cost(estimated_tokens, model)
            if not self._check_cost_limit(user_status, estimated_cost):
                return False, "Daily cost limit exceeded.", self._get_status_info(user_status)
            
            # Record the request
            self._record_request(user_status, estimated_tokens, estimated_cost, current_time)
            
            return True, "Request allowed", self._get_status_info(user_status)
    
    def _cleanup_old_entries(self, user_status: RateLimitStatus, current_time: float):
        """Remove old entries from tracking deques."""
        # Clean minute window (60 seconds)
        while (user_status.requests_minute and 
               current_time - user_status.requests_minute[0] > 60):
            user_status.requests_minute.popleft()
        
        # Clean hour window (3600 seconds)
        while (user_status.requests_hour and 
               current_time - user_status.requests_hour[0] > 3600):
            user_status.requests_hour.popleft()
        
        # Clean day window (86400 seconds)
        while (user_status.requests_day and 
               current_time - user_status.requests_day[0] > 86400):
            user_status.requests_day.popleft()
        
        # Clean burst window
        while (user_status.burst_requests and 
               current_time - user_status.burst_requests[0] > self.config.burst_window_seconds):
            user_status.burst_requests.popleft()
    
    def _check_burst_limit(self, user_status: RateLimitStatus, current_time: float) -> bool:
        """Check burst protection."""
        if len(user_status.burst_requests) >= self.config.max_burst_requests:
            return False
        return True
    
    def _check_minute_limit(self, user_status: RateLimitStatus, tokens: int) -> bool:
        """Check per-minute limits."""
        if len(user_status.requests_minute) >= self.config.requests_per_minute:
            return False
        if user_status.tokens_minute + tokens > self.config.tokens_per_minute:
            return False
        return True
    
    def _check_hour_limit(self, user_status: RateLimitStatus, tokens: int) -> bool:
        """Check per-hour limits."""
        if len(user_status.requests_hour) >= self.config.requests_per_hour:
            return False
        if user_status.tokens_hour + tokens > self.config.tokens_per_hour:
            return False
        return True
    
    def _check_day_limit(self, user_status: RateLimitStatus, tokens: int) -> bool:
        """Check per-day limits."""
        if len(user_status.requests_day) >= self.config.requests_per_day:
            return False
        if user_status.tokens_day + tokens > self.config.tokens_per_day:
            return False
        return True
    
    def _check_cost_limit(self, user_status: RateLimitStatus, cost: float) -> bool:
        """Check daily cost limit."""
        return user_status.cost_day + cost <= self.config.cost_per_day
    
    def _calculate_cost(self, tokens: int, model: str) -> float:
        """Calculate estimated cost for tokens."""
        cost_per_1k = self.token_costs.get(model, 0.0015)  # Default to gpt-3.5-turbo
        return (tokens / 1000) * cost_per_1k
    
    def _record_request(self, user_status: RateLimitStatus, tokens: int, cost: float, current_time: float):
        """Record the request in all tracking structures."""
        user_status.requests_minute.append(current_time)
        user_status.requests_hour.append(current_time)
        user_status.requests_day.append(current_time)
        user_status.burst_requests.append(current_time)
        
        user_status.tokens_minute += tokens
        user_status.tokens_hour += tokens
        user_status.tokens_day += tokens
        user_status.cost_day += cost
        user_status.last_request_time = current_time
    
    def _get_status_info(self, user_status: RateLimitStatus) -> Dict:
        """Get current status information."""
        current_time = time.time()
        
        return {
            "requests_this_minute": len(user_status.requests_minute),
            "requests_this_hour": len(user_status.requests_hour),
            "requests_this_day": len(user_status.requests_day),
            "tokens_this_minute": user_status.tokens_minute,
            "tokens_this_hour": user_status.tokens_hour,
            "tokens_this_day": user_status.tokens_day,
            "cost_this_day": round(user_status.cost_day, 4),
            "last_request": user_status.last_request_time,
            "limits": {
                "requests_per_minute": self.config.requests_per_minute,
                "requests_per_hour": self.config.requests_per_hour,
                "requests_per_day": self.config.requests_per_day,
                "tokens_per_minute": self.config.tokens_per_minute,
                "tokens_per_hour": self.config.tokens_per_hour,
                "tokens_per_day": self.config.tokens_per_day,
                "cost_per_day": self.config.cost_per_day,
            }
        }
    
    async def get_user_status(self, user_id: str) -> Dict:
        """Get current status for a user."""
        async with self.lock:
            if user_id not in self.user_limits:
                return {"error": "User not found"}
            
            user_status = self.user_limits[user_id]
            current_time = time.time()
            self._cleanup_old_entries(user_status, current_time)
            
            return self._get_status_info(user_status)
    
    async def reset_user_limits(self, user_id: str):
        """Reset limits for a specific user (admin function)."""
        async with self.lock:
            if user_id in self.user_limits:
                del self.user_limits[user_id]
                logger.info(f"Reset rate limits for user: {user_id}")
    
    async def get_global_stats(self) -> Dict:
        """Get global statistics across all users."""
        async with self.lock:
            total_users = len(self.user_limits)
            current_time = time.time()
            
            active_users = 0
            total_requests_today = 0
            total_tokens_today = 0
            total_cost_today = 0.0
            
            for user_status in self.user_limits.values():
                self._cleanup_old_entries(user_status, current_time)
                
                if user_status.last_request_time > current_time - 3600:  # Active in last hour
                    active_users += 1
                
                total_requests_today += len(user_status.requests_day)
                total_tokens_today += user_status.tokens_day
                total_cost_today += user_status.cost_day
            
            return {
                "total_users": total_users,
                "active_users": active_users,
                "total_requests_today": total_requests_today,
                "total_tokens_today": total_tokens_today,
                "total_cost_today": round(total_cost_today, 4),
                "config": {
                    "requests_per_minute": self.config.requests_per_minute,
                    "requests_per_hour": self.config.requests_per_hour,
                    "requests_per_day": self.config.requests_per_day,
                    "tokens_per_minute": self.config.tokens_per_minute,
                    "tokens_per_hour": self.config.tokens_hour,
                    "tokens_per_day": self.config.tokens_per_day,
                    "cost_per_day": self.config.cost_per_day,
                }
            }


# Global rate limiter instance
rate_limiter = RateLimiter(RateLimitConfig())
