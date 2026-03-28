"""
Rate Limiting for API Requests

This module provides rate limiting functionality to prevent abuse
and ensure fair usage of the API resources.
"""

import time
from typing import Dict, Optional
from collections import defaultdict, deque
from fastapi import Request, HTTPException
import threading

from ..utils.logging_config import get_logger
from ..utils.config import get_config
from ..models.data_models import ErrorResponse


logger = get_logger(__name__)

# Rate limiting configuration - more generous for testing and development
RATE_LIMITS = {
    'requests_per_minute': 120,  # Increased from 60 to handle test scenarios
    'requests_per_hour': 2000,   # Increased from 1000
    'requests_per_day': 20000    # Increased from 10000
}

# In-memory storage for rate limiting (in production, use Redis or similar)
class RateLimiter:
    """Thread-safe rate limiter using sliding window approach with automatic cleanup."""
    
    def __init__(self):
        self._requests = defaultdict(lambda: {
            'minute': deque(),
            'hour': deque(),
            'day': deque()
        })
        self._lock = threading.Lock()
        self._last_cleanup = time.time()
        self._cleanup_interval = 300  # Clean up every 5 minutes
    
    def _cleanup_old_entries(self):
        """Clean up old entries to prevent memory leaks during sustained load."""
        current_time = time.time()
        
        # Only cleanup if enough time has passed
        if current_time - self._last_cleanup < self._cleanup_interval:
            return
        
        self._last_cleanup = current_time
        
        # Remove entries older than 1 day for all clients
        cutoff_time = current_time - 86400  # 1 day
        clients_to_remove = []
        
        for client_id, client_requests in self._requests.items():
            # Clean all windows for this client
            for window in ['minute', 'hour', 'day']:
                window_seconds = {'minute': 60, 'hour': 3600, 'day': 86400}[window]
                window_cutoff = current_time - window_seconds
                
                # Remove old requests
                while client_requests[window] and client_requests[window][0] < window_cutoff:
                    client_requests[window].popleft()
            
            # If client has no recent requests, mark for removal
            if all(len(client_requests[window]) == 0 for window in ['minute', 'hour', 'day']):
                clients_to_remove.append(client_id)
        
        # Remove inactive clients to free memory
        for client_id in clients_to_remove:
            del self._requests[client_id]
        
        if clients_to_remove:
            logger.debug(f"Cleaned up {len(clients_to_remove)} inactive rate limit entries")
    
    def is_allowed(self, client_id: str) -> tuple[bool, Optional[str], Optional[int]]:
        """
        Check if request is allowed based on rate limits.
        
        Args:
            client_id: Unique identifier for the client (IP address)
            
        Returns:
            Tuple of (is_allowed, error_message, retry_after_seconds)
        """
        current_time = time.time()
        
        with self._lock:
            # Perform periodic cleanup to prevent memory leaks
            self._cleanup_old_entries()
            
            client_requests = self._requests[client_id]
            
            # Clean old requests and check limits
            for window, limit in [
                ('minute', RATE_LIMITS['requests_per_minute']),
                ('hour', RATE_LIMITS['requests_per_hour']),
                ('day', RATE_LIMITS['requests_per_day'])
            ]:
                window_seconds = {'minute': 60, 'hour': 3600, 'day': 86400}[window]
                cutoff_time = current_time - window_seconds
                
                # Remove old requests
                while client_requests[window] and client_requests[window][0] < cutoff_time:
                    client_requests[window].popleft()
                
                # Check if limit exceeded
                if len(client_requests[window]) >= limit:
                    retry_after = int(client_requests[window][0] + window_seconds - current_time) + 1
                    return False, f"Rate limit exceeded: {limit} requests per {window}", retry_after
            
            # Add current request to all windows
            for window in ['minute', 'hour', 'day']:
                client_requests[window].append(current_time)
            
            return True, None, None
    
    def get_usage_stats(self, client_id: str) -> Dict[str, int]:
        """Get current usage statistics for a client."""
        current_time = time.time()
        
        with self._lock:
            client_requests = self._requests[client_id]
            stats = {}
            
            for window in ['minute', 'hour', 'day']:
                window_seconds = {'minute': 60, 'hour': 3600, 'day': 86400}[window]
                cutoff_time = current_time - window_seconds
                
                # Count requests in current window
                count = sum(1 for req_time in client_requests[window] if req_time >= cutoff_time)
                stats[f'requests_in_last_{window}'] = count
                stats[f'limit_per_{window}'] = RATE_LIMITS[f'requests_per_{window}']
                stats[f'remaining_in_{window}'] = max(0, RATE_LIMITS[f'requests_per_{window}'] - count)
            
            return stats


# Global rate limiter instance
rate_limiter = RateLimiter()


def get_client_id(request: Request) -> str:
    """
    Extract client identifier from request.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Client identifier string
    """
    # Try to get real IP from headers (for reverse proxy setups)
    forwarded_for = request.headers.get('X-Forwarded-For')
    if forwarded_for:
        # Take the first IP in the chain
        client_ip = forwarded_for.split(',')[0].strip()
    else:
        # Fall back to direct client IP
        client_ip = request.client.host if request.client else 'unknown'
    
    # Sanitize IP address
    if not client_ip or client_ip == 'unknown':
        client_ip = 'unknown'
    
    return client_ip


async def check_rate_limit(request: Request) -> bool:
    """
    FastAPI dependency to check rate limits.
    
    Args:
        request: FastAPI request object
        
    Returns:
        True if request is allowed
        
    Raises:
        HTTPException: If rate limit is exceeded
    """
    # Check if rate limiting is disabled via environment variable (for testing)
    import os
    if os.getenv("RATE_LIMIT_ENABLED", "").lower() in ("false", "0", "no", "off"):
        logger.debug("Rate limiting is disabled via environment variable")
        return True
    
    # Check if rate limiting is enabled in configuration
    config = get_config()
    if not config.rate_limit.enabled:
        logger.debug("Rate limiting is disabled in configuration")
        return True
    
    client_id = get_client_id(request)
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    try:
        is_allowed, error_message, retry_after = rate_limiter.is_allowed(client_id)
        
        if not is_allowed:
            logger.warning(
                f"Rate limit exceeded",
                extra={
                    "request_id": request_id,
                    "client_id": client_id,
                    "error": error_message,
                    "retry_after": retry_after
                }
            )
            
            error_response = ErrorResponse(
                error="RATE_LIMIT_EXCEEDED",
                message=error_message
            )
            
            # Add rate limit headers
            headers = {
                "X-RateLimit-Limit": str(RATE_LIMITS['requests_per_minute']),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(int(time.time()) + retry_after),
                "Retry-After": str(retry_after)
            }
            
            raise HTTPException(
                status_code=429,
                detail=error_response.to_dict(),
                headers=headers
            )
        
        # Add rate limit headers for successful requests
        usage_stats = rate_limiter.get_usage_stats(client_id)
        
        # Add headers to response (will be handled by middleware)
        if hasattr(request.state, 'rate_limit_headers'):
            request.state.rate_limit_headers = {}
        else:
            request.state.rate_limit_headers = {}
            
        request.state.rate_limit_headers.update({
            "X-RateLimit-Limit-Minute": str(RATE_LIMITS['requests_per_minute']),
            "X-RateLimit-Remaining-Minute": str(usage_stats['remaining_in_minute']),
            "X-RateLimit-Limit-Hour": str(RATE_LIMITS['requests_per_hour']),
            "X-RateLimit-Remaining-Hour": str(usage_stats['remaining_in_hour']),
            "X-RateLimit-Limit-Day": str(RATE_LIMITS['requests_per_day']),
            "X-RateLimit-Remaining-Day": str(usage_stats['remaining_in_day'])
        })
        
        logger.debug(
            f"Rate limit check passed",
            extra={
                "request_id": request_id,
                "client_id": client_id,
                "usage_stats": usage_stats
            }
        )
        
        return True
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
        
    except Exception as e:
        logger.error(
            f"Error during rate limit check",
            extra={
                "request_id": request_id,
                "client_id": client_id,
                "error": str(e)
            },
            exc_info=True
        )
        
        # In case of error, allow the request but log the issue
        return True


def get_rate_limit_status(client_id: str) -> Dict[str, any]:
    """
    Get current rate limit status for a client.
    
    Args:
        client_id: Client identifier
        
    Returns:
        Dictionary with rate limit status information
    """
    try:
        usage_stats = rate_limiter.get_usage_stats(client_id)
        
        return {
            "client_id": client_id,
            "limits": RATE_LIMITS,
            "usage": usage_stats,
            "timestamp": int(time.time())
        }
        
    except Exception as e:
        logger.error(f"Error getting rate limit status: {e}", exc_info=True)
        return {
            "client_id": client_id,
            "error": "Unable to retrieve rate limit status"
        }


def reset_rate_limit(client_id: str) -> bool:
    """
    Reset rate limit for a specific client (admin function).
    
    Args:
        client_id: Client identifier to reset
        
    Returns:
        True if reset was successful
    """
    try:
        with rate_limiter._lock:
            if client_id in rate_limiter._requests:
                del rate_limiter._requests[client_id]
                logger.info(f"Rate limit reset for client: {client_id}")
                return True
            return False
            
    except Exception as e:
        logger.error(f"Error resetting rate limit: {e}", exc_info=True)
        return False
