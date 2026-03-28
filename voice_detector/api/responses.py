"""
Response Formatting and HTTP Status Code Management

This module provides consistent response formatting, error handling,
and HTTP status code management for the API endpoints.
"""

from typing import Dict, Any, Optional, Union
from fastapi import Response
from fastapi.responses import JSONResponse
import time

from ..utils.logging_config import get_logger
from ..models.data_models import DetectionResponse, ErrorResponse


logger = get_logger(__name__)

# HTTP Status Code Constants
class StatusCodes:
    """HTTP status code constants for consistent usage."""
    OK = 200
    CREATED = 201
    ACCEPTED = 202
    NO_CONTENT = 204
    
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    METHOD_NOT_ALLOWED = 405
    NOT_ACCEPTABLE = 406
    REQUEST_TIMEOUT = 408
    CONFLICT = 409
    GONE = 410
    LENGTH_REQUIRED = 411
    PRECONDITION_FAILED = 412
    PAYLOAD_TOO_LARGE = 413
    URI_TOO_LONG = 414
    UNSUPPORTED_MEDIA_TYPE = 415
    RANGE_NOT_SATISFIABLE = 416
    EXPECTATION_FAILED = 417
    UNPROCESSABLE_ENTITY = 422
    TOO_MANY_REQUESTS = 429
    
    INTERNAL_SERVER_ERROR = 500
    NOT_IMPLEMENTED = 501
    BAD_GATEWAY = 502
    SERVICE_UNAVAILABLE = 503
    GATEWAY_TIMEOUT = 504
    HTTP_VERSION_NOT_SUPPORTED = 505


class ResponseFormatter:
    """Utility class for formatting API responses consistently."""
    
    @staticmethod
    def success_response(
        data: Union[DetectionResponse, Dict[str, Any]],
        status_code: int = StatusCodes.OK,
        headers: Optional[Dict[str, str]] = None
    ) -> JSONResponse:
        """
        Format successful response.
        
        Args:
            data: Response data (DetectionResponse or dict)
            status_code: HTTP status code
            headers: Additional headers
            
        Returns:
            JSONResponse with formatted success data
        """
        if isinstance(data, DetectionResponse):
            response_data = data.to_dict()
        else:
            response_data = data
        
        # Add metadata
        response_data["status"] = "success"
        response_data["timestamp"] = int(time.time())
        
        # Prepare headers
        response_headers = {
            "Content-Type": "application/json",
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
        
        if headers:
            response_headers.update(headers)
        
        return JSONResponse(
            content=response_data,
            status_code=status_code,
            headers=response_headers
        )
    
    @staticmethod
    def error_response(
        error: Union[ErrorResponse, str],
        message: Optional[str] = None,
        status_code: int = StatusCodes.INTERNAL_SERVER_ERROR,
        headers: Optional[Dict[str, str]] = None,
        request_id: Optional[str] = None
    ) -> JSONResponse:
        """
        Format error response.
        
        Args:
            error: Error code or ErrorResponse object
            message: Error message (if error is string)
            status_code: HTTP status code
            headers: Additional headers
            request_id: Request ID for tracking
            
        Returns:
            JSONResponse with formatted error data
        """
        if isinstance(error, ErrorResponse):
            error_data = error.to_dict()
        else:
            error_data = {
                "error": error,
                "message": message or "An error occurred"
            }
        
        # API specification compliance: return ONLY the required fields
        # Metadata goes in headers, not response body
        
        # Prepare headers
        response_headers = {
            "Content-Type": "application/json",
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
        
        if request_id:
            response_headers["X-Request-ID"] = request_id
        
        if headers:
            response_headers.update(headers)
        
        # Log error response
        logger.warning(
            f"Error response generated",
            extra={
                "status_code": status_code,
                "error_code": error_data.get("error"),
                "error_message": error_data.get("message"),
                "request_id": request_id
            }
        )
        
        return JSONResponse(
            content=error_data,
            status_code=status_code,
            headers=response_headers
        )
    
    @staticmethod
    def validation_error_response(
        field_errors: Dict[str, str],
        request_id: Optional[str] = None
    ) -> JSONResponse:
        """
        Format validation error response.
        
        Args:
            field_errors: Dictionary of field names to error messages
            request_id: Request ID for tracking
            
        Returns:
            JSONResponse with validation error details
        """
        # API specification compliance: only error and message fields
        error_data = {
            "error": "VALIDATION_ERROR",
            "message": "Request validation failed"
        }
        
        # Additional details can go in headers if needed
        headers = {
            "Content-Type": "application/json"
        }
        
        if request_id:
            headers["X-Request-ID"] = request_id
        
        return JSONResponse(
            content=error_data,
            status_code=StatusCodes.BAD_REQUEST,
            headers=headers
        )


def create_detection_response(
    classification: str,
    confidence_score: float,
    processing_time: Optional[float] = None,
    request_id: Optional[str] = None,
    extra_headers: Optional[Dict[str, str]] = None,
) -> JSONResponse:
    """
    Create a formatted detection response.
    
    Args:
        classification: "AI_GENERATED" or "HUMAN"
        confidence_score: Confidence score between 0.0 and 1.0
        processing_time: Processing time in seconds
        request_id: Request ID for tracking
        
    Returns:
        JSONResponse with detection results
    """
    try:
        # Create and validate detection response
        detection_response = DetectionResponse(
            classification=classification,
            confidence_score=confidence_score
        )
        
        # API specification compliance: return ONLY the required fields
        response_data = detection_response.to_dict()
        
        # Prepare headers (metadata goes in headers, not response body)
        headers = {
            "Content-Type": "application/json",
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
        
        if request_id:
            headers["X-Request-ID"] = request_id
        
        if processing_time is not None:
            headers["X-Process-Time"] = str(round(processing_time, 4))

        if extra_headers:
            headers.update(extra_headers)
        
        # Return specification-compliant response with only required fields
        return JSONResponse(
            content=response_data,
            status_code=StatusCodes.OK,
            headers=headers
        )
        
    except ValueError as e:
        logger.error(f"Invalid detection response data: {e}")
        return ResponseFormatter.error_response(
            error="INVALID_RESPONSE_DATA",
            message=f"Invalid response data: {str(e)}",
            status_code=StatusCodes.INTERNAL_SERVER_ERROR,
            request_id=request_id
        )


def create_error_response_from_exception(
    exception: Exception,
    request_id: Optional[str] = None,
    include_details: bool = False
) -> JSONResponse:
    """
    Create error response from exception.
    
    Args:
        exception: Exception that occurred
        request_id: Request ID for tracking
        include_details: Whether to include exception details (debug mode)
        
    Returns:
        JSONResponse with error information
    """
    # Map common exceptions to appropriate error responses
    if isinstance(exception, ValueError):
        return ResponseFormatter.error_response(
            error="INVALID_INPUT",
            message="Invalid input provided",
            status_code=StatusCodes.BAD_REQUEST,
            request_id=request_id
        )
    elif isinstance(exception, FileNotFoundError):
        return ResponseFormatter.error_response(
            error="RESOURCE_NOT_FOUND",
            message="Required resource not found",
            status_code=StatusCodes.NOT_FOUND,
            request_id=request_id
        )
    elif isinstance(exception, PermissionError):
        return ResponseFormatter.error_response(
            error="PERMISSION_DENIED",
            message="Permission denied",
            status_code=StatusCodes.FORBIDDEN,
            request_id=request_id
        )
    elif isinstance(exception, TimeoutError):
        return ResponseFormatter.error_response(
            error="REQUEST_TIMEOUT",
            message="Request processing timed out",
            status_code=StatusCodes.REQUEST_TIMEOUT,
            request_id=request_id
        )
    elif isinstance(exception, MemoryError):
        return ResponseFormatter.error_response(
            error="INSUFFICIENT_MEMORY",
            message="Insufficient memory to process request",
            status_code=StatusCodes.SERVICE_UNAVAILABLE,
            request_id=request_id
        )
    else:
        # Generic error response
        error_message = "An unexpected error occurred"
        if include_details:
            error_message += f": {str(exception)}"
        
        return ResponseFormatter.error_response(
            error="INTERNAL_SERVER_ERROR",
            message=error_message,
            status_code=StatusCodes.INTERNAL_SERVER_ERROR,
            request_id=request_id
        )


def add_cors_headers(response: Response) -> Response:
    """
    Add CORS headers to response.
    
    Args:
        response: Response object to modify
        
    Returns:
        Response with CORS headers added
    """
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
    response.headers["Access-Control-Max-Age"] = "86400"
    
    return response


def add_security_headers(response: Response) -> Response:
    """
    Add security headers to response.
    
    Args:
        response: Response object to modify
        
    Returns:
        Response with security headers added
    """
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    
    return response


# Error code to HTTP status code mapping
ERROR_CODE_STATUS_MAP = {
    # JSON validation errors (400 Bad Request)
    "INVALID_REQUEST": StatusCodes.BAD_REQUEST,
    "VALIDATION_ERROR": StatusCodes.BAD_REQUEST,
    "MISSING_FIELD": StatusCodes.BAD_REQUEST,
    "SUSPICIOUS_INPUT": StatusCodes.BAD_REQUEST,
    "INVALID_INPUT_TYPE": StatusCodes.BAD_REQUEST,
    
    # Audio processing errors (422 Unprocessable Entity)
    "INVALID_BASE64": StatusCodes.UNPROCESSABLE_ENTITY,
    "AUDIO_TOO_SMALL": StatusCodes.UNPROCESSABLE_ENTITY,
    "AUDIO_TOO_SHORT": StatusCodes.UNPROCESSABLE_ENTITY,
    "UNSUPPORTED_FORMAT": StatusCodes.UNPROCESSABLE_ENTITY,
    "INVALID_AUDIO_FORMAT": StatusCodes.UNPROCESSABLE_ENTITY,
    "CORRUPTED_AUDIO": StatusCodes.UNPROCESSABLE_ENTITY,
    "LOW_ENTROPY_INPUT": StatusCodes.UNPROCESSABLE_ENTITY,
    
    # Size-related errors
    "PAYLOAD_TOO_LARGE": StatusCodes.PAYLOAD_TOO_LARGE,
    "CONTENT_TOO_LARGE": StatusCodes.PAYLOAD_TOO_LARGE,
    "REQUEST_TOO_LARGE": StatusCodes.PAYLOAD_TOO_LARGE,
    
    # Rate limiting
    "RATE_LIMIT_EXCEEDED": StatusCodes.TOO_MANY_REQUESTS,
    "VALIDATION_TIMEOUT": StatusCodes.TOO_MANY_REQUESTS,
    "DUPLICATE_REQUEST": StatusCodes.CONFLICT,
    
    # Server errors (500 Internal Server Error)
    "PROCESSING_ERROR": StatusCodes.INTERNAL_SERVER_ERROR,
    "MODEL_ERROR": StatusCodes.INTERNAL_SERVER_ERROR,
    "FEATURE_EXTRACTION_ERROR": StatusCodes.INTERNAL_SERVER_ERROR,
    "AUDIO_PROCESSING_ERROR": StatusCodes.INTERNAL_SERVER_ERROR,
    "INTERNAL_SERVER_ERROR": StatusCodes.INTERNAL_SERVER_ERROR,
    
    # Service availability errors (503 Service Unavailable)
    "SERVICE_UNAVAILABLE": StatusCodes.SERVICE_UNAVAILABLE,
    "INSUFFICIENT_MEMORY": StatusCodes.SERVICE_UNAVAILABLE,
    
    # Timeout errors
    "REQUEST_TIMEOUT": StatusCodes.REQUEST_TIMEOUT,
}


def get_status_code_for_error(error_code: str) -> int:
    """
    Get appropriate HTTP status code for error code.
    
    Args:
        error_code: Error code string
        
    Returns:
        HTTP status code integer
    """
    return ERROR_CODE_STATUS_MAP.get(error_code, StatusCodes.INTERNAL_SERVER_ERROR)
