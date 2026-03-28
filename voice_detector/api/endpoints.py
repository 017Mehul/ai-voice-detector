"""
API Endpoints for AI Voice Detection System

This module contains the main API endpoints for voice detection,
including request validation, processing coordination, and response formatting.
"""

from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, field_validator
from typing import Dict, Any, Optional
import asyncio
import time

from ..utils.logging_config import get_logger
from ..utils.reporting import (
    ScanRecord,
    build_report_pdf,
    create_audio_fingerprint,
    create_scan_id,
    default_summary,
    get_scan_history_store,
)
from ..models.data_models import (
    DetectionRequest, 
    DetectionResponse, 
    ErrorResponse,
    ValidationResult
)
from .validation import (
    validate_detection_request,
    validate_audio_size,
    sanitize_base64_input,
    sanitize_and_validate_comprehensive,
    validate_request_rate_and_duplicates
)
from .responses import (
    create_detection_response,
    create_error_response_from_exception,
    ResponseFormatter,
    get_status_code_for_error
)
from .rate_limiting import check_rate_limit, get_client_id


logger = get_logger(__name__)
router = APIRouter()


class DetectionRequestModel(BaseModel):
    """Pydantic model for request validation."""
    audio_base64: str = Field(
        ...,
        description="Base64-encoded MP3, WAV, or FLAC audio data",
        max_length=50_000_000  # ~37MB limit for Base64 data
    )
    filename: Optional[str] = Field(default=None, max_length=255)
    file_format: Optional[str] = Field(default=None, max_length=16)
    duration_seconds: Optional[float] = Field(default=None, ge=0.0, le=600.0)
    sample_rate_hz: Optional[int] = Field(default=None, ge=8000, le=192000)
    channels: Optional[str] = Field(default=None, max_length=32)
    bitrate_kbps: Optional[str] = Field(default=None, max_length=32)
    waveform_bars: Optional[list[float]] = Field(default=None, min_length=16, max_length=256)
    
    @field_validator('audio_base64')
    @classmethod
    def validate_base64_format(cls, v):
        """Basic string validation only - detailed base64 validation happens later."""
        if not isinstance(v, str):
            raise ValueError("audio_base64 must be a string")
        
        # Only check for obviously invalid characters that would never be in base64
        # Leave detailed base64 validation (including empty string check) for the custom validation pipeline
        if any(c in v for c in ['<', '>', '{', '}', '[', ']', '|', '\\', '^', '`']):
            raise ValueError("audio_base64 contains obviously invalid characters")
        
        return v

    @field_validator("filename", "file_format", "channels", "bitrate_kbps")
    @classmethod
    def validate_optional_strings(cls, value):
        """Normalize empty optional strings to None."""
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @field_validator("waveform_bars")
    @classmethod
    def validate_waveform_bars(cls, value):
        """Ensure waveform values are normalized floats."""
        if value is None:
            return None
        normalized = [max(0.0, min(1.0, float(item))) for item in value]
        return normalized


def _build_scan_record(
    request_id: str,
    request_data: DetectionRequestModel,
    detection_response: DetectionResponse,
) -> ScanRecord:
    """Create a persisted history record for a successful scan."""
    file_format = (request_data.file_format or request_data.filename or "WAV").split(".")[-1].upper()
    filename = request_data.filename or f"{create_scan_id()}.{file_format.lower()}"
    scan_id = create_scan_id()
    fingerprint = create_audio_fingerprint(request_data.audio_base64)

    return ScanRecord(
        scan_id=scan_id,
        request_id=request_id,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        classification=detection_response.classification,
        confidence_score=float(detection_response.confidence_score),
        filename=filename,
        file_format=file_format,
        duration_seconds=request_data.duration_seconds,
        sample_rate_hz=request_data.sample_rate_hz,
        channels=request_data.channels,
        bitrate_kbps=request_data.bitrate_kbps,
        fingerprint=fingerprint,
        summary=default_summary(
            detection_response.classification,
            float(detection_response.confidence_score),
        ),
        waveform_bars=request_data.waveform_bars,
    )


def _record_scan_history(
    request_id: str,
    request_data: DetectionRequestModel,
    detection_response: DetectionResponse,
) -> ScanRecord:
    """Persist a scan to the shared history store."""
    history_store = get_scan_history_store()
    scan_record = _build_scan_record(request_id, request_data, detection_response)
    return history_store.add_record(scan_record)


@router.post(
    "/detect-voice",
    response_model=None,  # Using JSONResponse directly
    responses={
        200: {
            "description": "Successful voice detection",
            "content": {
                "application/json": {
                    "example": {
                        "classification": "AI_GENERATED",
                        "confidence_score": 0.85
                    }
                }
            }
        },
        400: {
            "description": "Bad request - invalid input format",
            "content": {
                "application/json": {
                    "example": {
                        "error": "INVALID_BASE64",
                        "message": "The provided audio_base64 is not valid Base64 data"
                    }
                }
            }
        },
        413: {
            "description": "Payload too large",
            "content": {
                "application/json": {
                    "example": {
                        "error": "PAYLOAD_TOO_LARGE",
                        "message": "Audio file size exceeds maximum allowed limit"
                    }
                }
            }
        },
        422: {
            "description": "Unprocessable entity - audio processing error",
            "content": {
                "application/json": {
                    "example": {
                        "error": "AUDIO_TOO_SHORT",
                        "message": "Audio duration must be at least 2 seconds"
                    }
                }
            }
        },
        409: {
            "description": "Duplicate request submitted within the duplicate protection window",
            "content": {
                "application/json": {
                    "example": {
                        "error": "DUPLICATE_REQUEST",
                        "message": "Duplicate request detected for the same audio payload. Please wait before retrying."
                    }
                }
            }
        },
        429: {
            "description": "Too many requests",
            "content": {
                "application/json": {
                    "example": {
                        "error": "RATE_LIMIT_EXCEEDED",
                        "message": "Too many requests. Please try again later."
                    }
                }
            }
        },
        500: {
            "description": "Internal server error",
            "content": {
                "application/json": {
                    "example": {
                        "error": "INTERNAL_SERVER_ERROR",
                        "message": "An unexpected error occurred during processing"
                    }
                }
            }
        }
    },
    summary="Detect AI-generated voice",
    description="Classify an audio sample as either AI-generated or human-spoken voice"
)
async def detect_voice(
    request_data: DetectionRequestModel,
    request: Request,
    rate_limit_check: bool = Depends(check_rate_limit)
) -> JSONResponse:
    """
    Main endpoint for voice detection.
    
        This endpoint accepts Base64-encoded MP3, WAV, or FLAC audio data and returns a classification
    result indicating whether the voice is AI-generated or human, along with a
    confidence score.
    
    Args:
        request_data: Validated request containing audio_base64
        request: FastAPI request object for context
        rate_limit_check: Rate limiting dependency
        
    Returns:
        Dict containing classification and confidence_score
        
    Raises:
        HTTPException: For various error conditions with appropriate status codes
    """
    request_id = getattr(request.state, 'request_id', 'unknown')
    start_time = time.time()
    
    logger.info(
        f"Voice detection request received",
        extra={
            "request_id": request_id,
            "audio_size": len(request_data.audio_base64)
        }
    )
    
    try:
        # Convert Pydantic model to internal data model
        detection_request = DetectionRequest(audio_base64=request_data.audio_base64)
        
        # Get client ID for validation
        client_id = get_client_id(request)
        
        # Step 1: Validate and sanitize input
        validation_result = await _validate_and_sanitize_input(detection_request, request_id, client_id)
        if not validation_result.is_valid:
            error_response = validation_result.to_error_response()
            logger.warning(
                f"Input validation failed",
                extra={
                    "request_id": request_id,
                    "error": error_response.error,
                    "error_message": error_response.message
                }
            )
            status_code = get_status_code_for_error(error_response.error)
            return ResponseFormatter.error_response(
                error=error_response,
                status_code=status_code,
                request_id=request_id
            )
        
        # Step 2: Process audio and extract features
        # Get the voice detection service from service manager
        from ..services import get_service_manager
        service_manager = get_service_manager()
        
        if not service_manager.is_initialized():
            logger.error(f"Service manager not initialized", extra={"request_id": request_id})
            error_response = ErrorResponse(
                error="SERVICE_UNAVAILABLE",
                message="System services are not properly initialized"
            )
            return ResponseFormatter.error_response(
                error=error_response,
                status_code=503,
                request_id=request_id
            )
        
        # Get the voice detection service
        try:
            voice_detection_service = service_manager.get_service('voice_detection_service')
        except ValueError as e:
            logger.error(f"Failed to get voice detection service", extra={"request_id": request_id, "error": str(e)})
            error_response = ErrorResponse(
                error="SERVICE_UNAVAILABLE",
                message="Voice detection service is not available"
            )
            return ResponseFormatter.error_response(
                error=error_response,
                status_code=503,
                request_id=request_id
            )
        
        # Process the request through the complete pipeline
        logger.info(f"Starting voice detection pipeline", extra={"request_id": request_id})
        
        detection_response, pipeline_error = voice_detection_service.detect_voice(detection_request, request_id)
        
        if pipeline_error:
            # Pipeline returned an error
            error_response = pipeline_error.to_error_response()
            logger.warning(
                f"Voice detection pipeline failed",
                extra={
                    "request_id": request_id,
                    "error": error_response.error,
                    "error_message": error_response.message
                }
            )
            status_code = get_status_code_for_error(error_response.error)
            return ResponseFormatter.error_response(
                error=error_response,
                status_code=status_code,
                request_id=request_id
            )
        
        # Success - return the detection response
        processing_time = time.time() - start_time
        logger.info(
            f"Voice detection completed successfully",
            extra={
                "request_id": request_id,
                "classification": detection_response.classification,
                "confidence": detection_response.confidence_score,
                "processing_time": round(processing_time, 4)
            }
        )
        
        scan_record = _record_scan_history(request_id, request_data, detection_response)

        return create_detection_response(
            classification=detection_response.classification,
            confidence_score=detection_response.confidence_score,
            processing_time=processing_time,
            request_id=request_id,
            extra_headers={
                "X-Scan-ID": scan_record.scan_id,
                "X-Report-URL": f"/api/v1/reports/{scan_record.scan_id}.pdf",
            },
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(
            f"Unexpected error during voice detection",
            extra={
                "request_id": request_id,
                "error": str(e),
                "processing_time": round(processing_time, 4)
            },
            exc_info=True
        )
        
        return create_error_response_from_exception(
            exception=e,
            request_id=request_id,
            include_details=False  # Don't expose internal details in production
        )


async def _validate_and_sanitize_input(
    detection_request: DetectionRequest, 
    request_id: str,
    client_id: str = "unknown"
) -> ValidationResult:
    """
    Validate and sanitize the detection request input.
    
    Args:
        detection_request: The request to validate
        request_id: Request ID for logging
        client_id: Client ID for rate limiting
        
    Returns:
        ValidationResult indicating success or failure with error details
    """
    try:
        # Step 1: Comprehensive validation using the new pipeline
        request_dict = {"audio_base64": detection_request.audio_base64}
        comprehensive_result = sanitize_and_validate_comprehensive(request_dict)
        if not comprehensive_result.is_valid:
            return comprehensive_result
        
        # Step 2: Rate limiting and duplicate detection
        rate_result = validate_request_rate_and_duplicates(request_dict, client_id)
        if not rate_result.is_valid:
            return rate_result
        
        # Step 3: Sanitize Base64 input
        sanitized_audio = sanitize_base64_input(detection_request.audio_base64)
        if not sanitized_audio:
            return ValidationResult(
                is_valid=False,
                error_code="INVALID_BASE64",
                error_message="Base64 data could not be sanitized or is empty"
            )
        
        # Update the request with sanitized data
        detection_request.audio_base64 = sanitized_audio
        
        logger.debug(
            f"Input validation successful",
            extra={
                "request_id": request_id,
                "client_id": client_id,
                "sanitized_size": len(sanitized_audio)
            }
        )
        
        return ValidationResult(is_valid=True)
        
    except Exception as e:
        logger.error(
            f"Error during input validation",
            extra={
                "request_id": request_id,
                "client_id": client_id,
                "error": str(e)
            },
            exc_info=True
        )
        
        return ValidationResult(
            is_valid=False,
            error_code="VALIDATION_ERROR",
            error_message="An error occurred during input validation"
        )


@router.get("/history")
async def get_scan_history(limit: int = 50):
    """Return recent completed scan history entries."""
    history_store = get_scan_history_store()
    safe_limit = max(1, min(limit, 250))
    records = [record.to_dict() for record in history_store.list_records(limit=safe_limit)]
    return {
        "items": records,
        "count": len(records),
    }


@router.get("/history/{scan_id}")
async def get_scan_history_item(scan_id: str):
    """Return a specific persisted scan entry."""
    history_store = get_scan_history_store()
    record = history_store.get_record(scan_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Scan report not found")

    return record.to_dict()


@router.get("/reports/{scan_id}.pdf")
async def download_scan_report(scan_id: str):
    """Return a generated PDF report for a persisted scan."""
    history_store = get_scan_history_store()
    record = history_store.get_record(scan_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Scan report not found")

    pdf_bytes = build_report_pdf(record)
    safe_name = record.filename.rsplit(".", 1)[0] if "." in record.filename else record.filename
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{safe_name}-report.pdf"',
            "Cache-Control": "no-cache, no-store, must-revalidate",
        },
    )


# Additional utility endpoints for monitoring and debugging

@router.get("/status")
async def get_status():
    """Get API status and basic system information."""
    from ..services import get_service_manager
    
    try:
        service_manager = get_service_manager()
        
        if not service_manager.is_initialized():
            return {
                "status": "degraded",
                "api_version": "1.0.0",
                "services": "not_initialized",
                "message": "System services are not properly initialized"
            }
        
        # Get detailed service status
        service_status = service_manager.get_service_status()
        
        # Get voice detection service status
        try:
            voice_service = service_manager.get_service('voice_detection_service')
            pipeline_status = voice_service.get_service_status()
            system_readiness = voice_service.validate_system_readiness()
        except Exception as e:
            pipeline_status = {"error": str(e)}
            system_readiness = ValidationResult(
                is_valid=False,
                error_code="VALIDATION_ERROR",
                error_message=str(e)
            )
        
        # Determine overall status
        overall_status = "operational"
        if not system_readiness.is_valid:
            overall_status = "degraded"
        
        return {
            "status": overall_status,
            "api_version": "1.0.0",
            "endpoints": {
                "detect_voice": "/api/v1/detect-voice",
                "history": "/api/v1/history",
                "report_pdf": "/api/v1/reports/{scan_id}.pdf",
                "web_app": "/app",
            },
            "supported_formats": ["MP3", "WAV", "FLAC"],
            "supported_languages": {
                "languages": ["Tamil", "English", "Hindi", "Malayalam", "Telugu"],
                "note": "Performance characteristics may vary by language - see /api/v1/performance-metrics for detailed language-specific metrics"
            },
            "services": service_status,
            "pipeline": pipeline_status,
            "system_readiness": {
                "is_valid": system_readiness.is_valid,
                "error_code": system_readiness.error_code,
                "error_message": system_readiness.error_message
            }
        }
        
    except Exception as e:
        logger.error(f"Status endpoint error: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "api_version": "1.0.0",
            "error": str(e)
        }


@router.get("/performance-metrics")
async def get_performance_metrics():
    """Get detailed language-specific performance metrics."""
    from ..services import get_service_manager

    calibration_status = "unavailable"
    calibration_note = "Confidence scores fall back to raw model output if calibration artifacts are missing."

    try:
        service_manager = get_service_manager()
        if service_manager.is_initialized():
            calibrator = service_manager.get_service('confidence_calibrator')
            if calibrator.is_calibrated():
                calibration_status = "active"
                calibration_note = "Confidence scores are calibrated using the loaded confidence calibrator."
    except Exception:
        pass

    return {
        "language_performance": {
            "note": "Performance characteristics vary by language based on measured data",
            "languages": {
                "Tamil": {
                    "processing_time_range": "0.107-1.369s",
                    "feature_extraction_efficiency": "78,574 samples/sec",
                    "classification_timing": "~0.107s",
                    "notes": "Variable processing time depending on audio complexity"
                },
                "Hindi": {
                    "processing_time_range": "0.025-0.110s", 
                    "feature_extraction_efficiency": "78,065 samples/sec",
                    "classification_timing": "~0.110s",
                    "notes": "Generally faster processing"
                },
                "Malayalam": {
                    "processing_time_range": "0.022-0.109s",
                    "feature_extraction_efficiency": "80,557 samples/sec", 
                    "classification_timing": "~0.109s",
                    "notes": "Highest feature extraction efficiency"
                },
                "Telugu": {
                    "processing_time_range": "0.022-0.109s",
                    "feature_extraction_efficiency": "68,544 samples/sec",
                    "classification_timing": "~0.109s", 
                    "notes": "Consistent mid-range performance"
                },
                "English": {
                    "processing_time_range": "Similar to other languages",
                    "notes": "Performance characteristics similar to other supported languages"
                }
            }
        },
        "feature_consistency": {
            "vector_size": 1976,
            "consistent_across_languages": True,
            "note": "All languages produce same-sized feature vectors"
        },
        "accuracy_notes": {
            "current_status": "Runtime model artifacts are loaded; quality still depends on the training datasets behind these benchmark numbers",
            "confidence_calibration": calibration_note,
            "recommendation": "Performance metrics will be more meaningful after proper model training"
        },
        "runtime_status": {
            "confidence_calibration": calibration_status
        },
        "measurement_date": "2026-01-25",
        "data_source": "language_wise_classification_metrics.json and language_wise_feature_metrics.json"
    }


@router.get("/limits")
async def get_limits():
    """Get API limits and constraints."""
    from ..utils.config import get_config
    from .rate_limiting import RATE_LIMITS

    config = get_config()

    return {
        "max_audio_size_mb": config.audio.max_size_mb,
        "min_audio_duration_seconds": config.audio.min_duration_seconds,
        "max_audio_duration_seconds": config.audio.max_duration_seconds,
        "supported_formats": ["MP3", "WAV", "FLAC"],
        "supported_sample_rates": [16000, 22050, 44100, 48000],
        "rate_limits": {
            "enabled": config.rate_limit.enabled,
            "requests_per_minute": RATE_LIMITS['requests_per_minute'],
            "requests_per_hour": RATE_LIMITS['requests_per_hour'],
            "requests_per_day": RATE_LIMITS['requests_per_day']
        }
    }


@router.get("/pipeline/health")
async def get_pipeline_health():
    """Get detailed pipeline health information for monitoring."""
    from ..services import get_service_manager
    
    try:
        service_manager = get_service_manager()
        
        if not service_manager.is_initialized():
            return {
                "status": "unhealthy",
                "message": "Service manager not initialized",
                "components": {}
            }
        
        # Test each pipeline component
        health_status = {
            "status": "healthy",
            "components": {},
            "pipeline_ready": False
        }
        
        # Check audio processor
        try:
            audio_processor = service_manager.get_service('audio_processor')
            health_status["components"]["audio_processor"] = {
                "status": "healthy",
                "type": type(audio_processor).__name__,
                "config": {
                    "target_sample_rate": audio_processor.target_sample_rate,
                    "min_duration": audio_processor.min_duration,
                    "max_duration": audio_processor.max_duration
                }
            }
        except Exception as e:
            health_status["components"]["audio_processor"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["status"] = "degraded"
        
        # Check feature extractor
        try:
            feature_extractor = service_manager.get_service('feature_extractor')
            health_status["components"]["feature_extractor"] = {
                "status": "healthy",
                "type": type(feature_extractor).__name__,
                "dimensions": feature_extractor.get_feature_dimensions()
            }
        except Exception as e:
            health_status["components"]["feature_extractor"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["status"] = "degraded"
        
        # Check classifier
        try:
            classifier = service_manager.get_service('classifier')
            is_trained = classifier.is_trained()
            health_status["components"]["classifier"] = {
                "status": "healthy" if is_trained else "degraded",
                "type": type(classifier).__name__,
                "trained": is_trained,
                "warning": "Model not trained - detection requests will fail" if not is_trained else None
            }
            if not is_trained:
                health_status["status"] = "degraded"
        except Exception as e:
            health_status["components"]["classifier"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["status"] = "degraded"
        
        # Check confidence calibrator
        try:
            calibrator = service_manager.get_service('confidence_calibrator')
            is_calibrated = calibrator.is_calibrated()
            health_status["components"]["confidence_calibrator"] = {
                "status": "healthy" if is_calibrated else "degraded",
                "type": type(calibrator).__name__,
                "calibrated": is_calibrated,
                "warning": "Raw model confidence will be returned until a usable calibrator is loaded" if not is_calibrated else None
            }
            if not is_calibrated:
                health_status["status"] = "degraded"
        except Exception as e:
            health_status["components"]["confidence_calibrator"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["status"] = "degraded"
        
        # Check voice detection service
        try:
            voice_service = service_manager.get_service('voice_detection_service')
            readiness = voice_service.validate_system_readiness()
            health_status["components"]["voice_detection_service"] = {
                "status": "healthy" if readiness.is_valid else "degraded",
                "ready": readiness.is_valid,
                "error": readiness.error_message if not readiness.is_valid else None
            }
            health_status["pipeline_ready"] = readiness.is_valid
        except Exception as e:
            health_status["components"]["voice_detection_service"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["status"] = "degraded"
        
        return health_status
        
    except Exception as e:
        logger.error(f"Pipeline health check error: {str(e)}", exc_info=True)
        return {
            "status": "unhealthy",
            "error": str(e),
            "components": {}
        }
