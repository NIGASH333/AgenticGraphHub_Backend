"""
Rate Limiting Middleware for FastAPI

Provides middleware to automatically apply rate limiting to API endpoints.
"""

import time
import uuid
from typing import Callable, Optional
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
import logging

from rate_limiter import rate_limiter

logger = logging.getLogger(__name__)


class RateLimitMiddleware:
    """Middleware to apply rate limiting to FastAPI requests."""
    
    def __init__(self, 
                 app,
                 get_user_id: Optional[Callable[[Request], str]] = None,
                 exempt_paths: list = None):
        """
        Initialize rate limiting middleware.
        
        Args:
            app: FastAPI application instance
            get_user_id: Function to extract user ID from request (default: use IP)
            exempt_paths: List of paths to exempt from rate limiting
        """
        self.app = app
        self.get_user_id = get_user_id or self._default_get_user_id
        self.exempt_paths = exempt_paths or ['/health', '/docs', '/openapi.json']
    
    def _default_get_user_id(self, request: Request) -> str:
        """Default function to get user ID from request IP."""
        # Get client IP
        client_ip = request.client.host
        
        # Check for forwarded headers (for reverse proxy setups)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            client_ip = real_ip
        
        return f"ip:{client_ip}"
    
    async def __call__(self, request: Request, call_next):
        """Apply rate limiting to the request."""
        # Skip rate limiting for exempt paths
        if request.url.path in self.exempt_paths:
            return await call_next(request)
        
        # Get user identifier
        try:
            user_id = self.get_user_id(request)
        except Exception as e:
            logger.error(f"Error getting user ID: {e}")
            user_id = f"unknown:{uuid.uuid4()}"
        
        # Estimate tokens based on request
        estimated_tokens = self._estimate_tokens(request)
        
        # Check rate limits
        try:
            allowed, reason, status_info = await rate_limiter.check_rate_limit(
                user_id=user_id,
                estimated_tokens=estimated_tokens,
                model='gpt-3.5-turbo'  # Default model
            )
            
            if not allowed:
                logger.warning(f"Rate limit exceeded for user {user_id}: {reason}")
                
                # Add rate limit headers
                headers = {
                    "X-RateLimit-Limit": str(rate_limiter.config.requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(time.time() + 60)),
                    "X-RateLimit-Status": "exceeded"
                }
                
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "error": "Rate limit exceeded",
                        "message": reason,
                        "retry_after": 60,
                        "status_info": status_info
                    },
                    headers=headers
                )
            
            # Add rate limit headers for successful requests
            response = await call_next(request)
            
            headers = {
                "X-RateLimit-Limit": str(rate_limiter.config.requests_per_minute),
                "X-RateLimit-Remaining": str(max(0, rate_limiter.config.requests_per_minute - status_info["requests_this_minute"])),
                "X-RateLimit-Reset": str(int(time.time() + 60)),
                "X-RateLimit-Status": "ok"
            }
            
            for key, value in headers.items():
                response.headers[key] = value
            
            return response
            
        except Exception as e:
            logger.error(f"Rate limiting error: {e}")
            # If rate limiting fails, allow the request but log the error
            return await call_next(request)
    
    def _estimate_tokens(self, request: Request) -> int:
        """Estimate token count for the request."""
        # Base estimation on request size and type
        content_length = int(request.headers.get("content-length", 0))
        
        # Rough estimation: 1 token ≈ 4 characters
        estimated_tokens = max(100, content_length // 4)
        
        # Cap at reasonable maximum
        return min(estimated_tokens, 4000)


def get_user_id_from_api_key(request: Request) -> str:
    """Extract user ID from API key header."""
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"api_key:{api_key}"
    
    # Fallback to IP-based identification
    client_ip = request.client.host
    return f"ip:{client_ip}"


def get_user_id_from_auth(request: Request) -> str:
    """Extract user ID from Authorization header."""
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header[7:]  # Remove "Bearer " prefix
        return f"auth:{token}"
    
    # Fallback to IP-based identification
    client_ip = request.client.host
    return f"ip:{client_ip}"
