"""Helper utilities for AI Voice Detection System"""

import base64
import hashlib
import time
from typing import Any, Dict, Optional
from functools import wraps

from .logging_config import get_logger

logger = get_logger(__name__)


def validate_base64(data: str) -> bool:
    """
    Validates if a string is valid Base64 encoded data.
    
    Args:
        data: String to validate
        
    Returns:
        True if valid Base64, False otherwise
    """
    try:
        if isinstance(data, str) and data.strip():
            # Check if string contains only valid Base64 characters
            base64.b64decode(data, validate=True)
            return True
    except Exception:
        pass
    return False


def decode_base64_safe(data: str) -> Optional[bytes]:
    """
    Safely decodes Base64 data with error handling.
    
    Args:
        data: Base64 encoded string
        
    Returns:
        Decoded bytes or None if decoding fails
    """
    try:
        return base64.b64decode(data)
    except Exception as e:
        logger.error(f"Base64 decoding failed: {e}")
        return None


def calculate_file_hash(data: bytes) -> str:
    """
    Calculates SHA256 hash of binary data.
    
    Args:
        data: Binary data to hash
        
    Returns:
        Hexadecimal hash string
    """
    return hashlib.sha256(data).hexdigest()


def timing_decorator(func):
    """
    Decorator to measure function execution time.
    
    Args:
        func: Function to measure
        
    Returns:
        Wrapped function that logs execution time
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} executed in {execution_time:.3f} seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.3f} seconds: {e}")
            raise
    return wrapper


def sanitize_error_message(error: Exception) -> str:
    """
    Sanitizes error messages for safe logging and user display.
    
    Args:
        error: Exception to sanitize
        
    Returns:
        Sanitized error message string
    """
    error_str = str(error)
    # Remove potentially sensitive information
    sanitized = error_str.replace('"', "'").replace('\n', ' ').replace('\r', '')
    # Truncate very long error messages
    if len(sanitized) > 200:
        sanitized = sanitized[:197] + "..."
    return sanitized


def create_error_response(error_code: str, message: str) -> Dict[str, Any]:
    """
    Creates standardized error response dictionary.
    
    Args:
        error_code: Error code identifier
        message: Human-readable error message
        
    Returns:
        Standardized error response dictionary
    """
    return {
        "error": error_code,
        "message": message
    }


def validate_audio_size(data: bytes, max_size_mb: int = 50) -> bool:
    """
    Validates audio file size against maximum allowed size.
    
    Args:
        data: Audio data bytes
        max_size_mb: Maximum allowed size in megabytes
        
    Returns:
        True if size is within limits, False otherwise
    """
    max_size_bytes = max_size_mb * 1024 * 1024
    return len(data) <= max_size_bytes