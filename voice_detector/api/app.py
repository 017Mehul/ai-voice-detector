"""
FastAPI Application Factory and Configuration

This module creates and configures the FastAPI application with proper middleware,
error handling, and endpoint routing for the AI Voice Detection System.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
import time
import uuid
import json
from typing import Dict, Any

from ..utils.logging_config import get_logger
from ..utils.config import get_config
from ..utils.prometheus_metrics import get_prometheus_metrics, initialize_system_metrics, start_metrics_server
from ..models.data_models import ErrorResponse
from .endpoints import router


logger = get_logger(__name__)


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.
    
    Returns:
        FastAPI: Configured application instance
    """
    config = get_config()
    
    app = FastAPI(
        title="AI Voice Detection System",
        description="REST API service for classifying audio samples as AI-generated or human-spoken voices",
        version="1.0.0",
        docs_url="/docs" if config.debug else None,
        redoc_url="/redoc" if config.debug else None,
        lifespan=_create_lifespan(),
    )
    
    # Add middleware
    _add_middleware(app)
    
    # Add exception handlers
    _add_exception_handlers(app)
    
    # Include routers
    app.include_router(router, prefix="/api/v1")

    # Add frontend routes
    _add_frontend_routes(app)
    
    # Add health check endpoints
    _add_health_endpoints(app)
    
    return app


def _add_middleware(app: FastAPI) -> None:
    """Add middleware to the FastAPI application."""
    config = get_config()
    
    # Prometheus metrics middleware
    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        """Collect Prometheus metrics for requests."""
        metrics = get_prometheus_metrics()
        start_time = time.time()
        
        try:
            response = await call_next(request)
            duration = time.time() - start_time
            
            # Record successful request metrics
            metrics.record_request(
                method=request.method,
                endpoint=request.url.path,
                status=response.status_code,
                duration=duration
            )
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            
            # Record failed request metrics
            metrics.record_request(
                method=request.method,
                endpoint=request.url.path,
                status=500,
                duration=duration
            )
            
            # Record error
            metrics.record_error(
                error_type=type(e).__name__,
                component="api"
            )
            
            raise
    
    # Request size limiting middleware
    @app.middleware("http")
    async def limit_request_size(request: Request, call_next):
        """Limit request size to prevent memory exhaustion attacks."""
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                content_length = int(content_length)
                if content_length > config.security.max_request_size:
                    logger.warning(
                        f"Request size too large: {content_length} bytes",
                        extra={
                            "client_ip": request.client.host if request.client else "unknown",
                            "content_length": content_length,
                            "max_allowed": config.security.max_request_size
                        }
                    )
                    error_response = ErrorResponse(
                        error="PAYLOAD_TOO_LARGE",
                        message=f"Request size exceeds maximum allowed limit of {config.security.max_request_size // (1024*1024)}MB"
                    )
                    return JSONResponse(
                        status_code=413,
                        content=error_response.to_dict()
                    )
            except ValueError:
                # Invalid content-length header
                pass
        
        return await call_next(request)
    
    # Request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Log all incoming requests with timing and request ID."""
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Add request ID to request state
        request.state.request_id = request_id
        
        # Log request start
        logger.info(
            f"Request started",
            extra={
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "client_ip": request.client.host if request.client else "unknown",
                "user_agent": request.headers.get("user-agent", "unknown")
            }
        )
        
        try:
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log request completion
            logger.info(
                f"Request completed",
                extra={
                    "request_id": request_id,
                    "status_code": response.status_code,
                    "process_time": round(process_time, 4)
                }
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(round(process_time, 4))
            
            # Add rate limit headers if available
            if hasattr(request.state, 'rate_limit_headers'):
                for header_name, header_value in request.state.rate_limit_headers.items():
                    response.headers[header_name] = header_value
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"Request failed",
                extra={
                    "request_id": request_id,
                    "error": str(e),
                    "process_time": round(process_time, 4)
                },
                exc_info=True
            )
            raise
    
    # CORS middleware
    if config.security.enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.api.cors_origins,
            allow_credentials=True,
            allow_methods=config.api.cors_methods,
            allow_headers=config.api.cors_headers,
        )


def _add_exception_handlers(app: FastAPI) -> None:
    """Add custom exception handlers for consistent error responses."""
    
    @app.exception_handler(json.JSONDecodeError)
    async def json_decode_error_handler(request: Request, exc: json.JSONDecodeError):
        """Handle JSON parsing errors."""
        request_id = getattr(request.state, 'request_id', 'unknown')
        
        logger.warning(
            f"JSON parsing failed",
            extra={
                "request_id": request_id,
                "error": str(exc),
                "doc": exc.doc[:100] if hasattr(exc, 'doc') and exc.doc else "unknown"
            }
        )
        
        error_response = ErrorResponse(
            error="JSON_PARSE_ERROR",
            message="Invalid JSON format. Please check your request body."
        )
        
        return JSONResponse(
            status_code=400,
            content=error_response.to_dict(),
            headers={"X-Request-ID": request_id}
        )
    
    @app.exception_handler(UnicodeDecodeError)
    async def unicode_decode_error_handler(request: Request, exc: UnicodeDecodeError):
        """Handle Unicode decoding errors in request body."""
        request_id = getattr(request.state, 'request_id', 'unknown')
        
        logger.warning(
            f"Unicode decoding failed",
            extra={
                "request_id": request_id,
                "error": str(exc),
                "encoding": exc.encoding if hasattr(exc, 'encoding') else "unknown"
            }
        )
        
        error_response = ErrorResponse(
            error="INVALID_ENCODING",
            message="Request body contains invalid character encoding."
        )
        
        return JSONResponse(
            status_code=400,
            content=error_response.to_dict(),
            headers={"X-Request-ID": request_id}
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle request validation errors."""
        request_id = getattr(request.state, 'request_id', 'unknown')
        
        logger.warning(
            f"Request validation failed",
            extra={
                "request_id": request_id,
                "errors": exc.errors(),
                "body": exc.body
            }
        )
        
        error_response = ErrorResponse(
            error="VALIDATION_ERROR",
            message="Request validation failed. Please check your input format."
        )
        
        return JSONResponse(
            status_code=400,
            content=error_response.to_dict(),
            headers={"X-Request-ID": request_id}
        )
    
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        """Handle ValueError exceptions, including JSON parsing errors."""
        request_id = getattr(request.state, 'request_id', 'unknown')
        
        # Check if this looks like a JSON parsing error
        error_str = str(exc).lower()
        is_json_error = any(keyword in error_str for keyword in [
            'json', 'parse', 'decode', 'invalid', 'malformed', 'syntax'
        ])
        
        if is_json_error:
            logger.warning(
                f"JSON-related ValueError",
                extra={
                    "request_id": request_id,
                    "error": str(exc),
                    "error_type": type(exc).__name__
                }
            )
            
            error_response = ErrorResponse(
                error="JSON_PARSE_ERROR",
                message="Invalid JSON format. Please check your request body."
            )
            
            return JSONResponse(
                status_code=400,
                content=error_response.to_dict(),
                headers={"X-Request-ID": request_id}
            )
        else:
            # Generic ValueError handling
            logger.warning(
                f"ValueError occurred",
                extra={
                    "request_id": request_id,
                    "error": str(exc),
                    "error_type": type(exc).__name__
                }
            )
            
            error_response = ErrorResponse(
                error="BAD_REQUEST",
                message="Invalid request data."
            )
            
            return JSONResponse(
                status_code=400,
                content=error_response.to_dict(),
                headers={"X-Request-ID": request_id}
            )
    
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        """Handle HTTP exceptions."""
        request_id = getattr(request.state, 'request_id', 'unknown')
        
        logger.warning(
            f"HTTP exception occurred",
            extra={
                "request_id": request_id,
                "status_code": exc.status_code,
                "detail": exc.detail
            }
        )
        
        # Map status codes to error codes
        error_code_map = {
            400: "BAD_REQUEST",
            401: "UNAUTHORIZED",
            403: "FORBIDDEN",
            404: "NOT_FOUND",
            405: "METHOD_NOT_ALLOWED",
            413: "PAYLOAD_TOO_LARGE",
            422: "UNPROCESSABLE_ENTITY",
            429: "TOO_MANY_REQUESTS",
            500: "INTERNAL_SERVER_ERROR",
            503: "SERVICE_UNAVAILABLE"
        }
        
        error_code = error_code_map.get(exc.status_code, "HTTP_ERROR")
        error_response = ErrorResponse(
            error=error_code,
            message=str(exc.detail)
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response.to_dict(),
            headers={"X-Request-ID": request_id}
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle unexpected exceptions."""
        request_id = getattr(request.state, 'request_id', 'unknown')
        
        logger.error(
            f"Unexpected error occurred",
            extra={
                "request_id": request_id,
                "error_type": type(exc).__name__,
                "error_message": str(exc)
            },
            exc_info=True
        )
        
        error_response = ErrorResponse(
            error="INTERNAL_SERVER_ERROR",
            message="An unexpected error occurred. Please try again later."
        )
        
        return JSONResponse(
            status_code=500,
            content=error_response.to_dict(),
            headers={"X-Request-ID": request_id}
        )


def _add_health_endpoints(app: FastAPI) -> None:
    """Add health check and monitoring endpoints."""
    
    @app.get("/health")
    async def health_check():
        """Basic health check endpoint for load balancers."""
        return {
            "status": "healthy",
            "service": "ai-voice-detection",
            "version": "1.0.0",
            "timestamp": time.time()
        }
    
    @app.get("/health/detailed")
    async def detailed_health_check():
        """Detailed health check with service status."""
        try:
            from ..services import get_service_manager
            service_manager = get_service_manager()
            
            # Check service status
            service_status = service_manager.get_service_status()
            is_healthy = service_manager.is_initialized()
            classifier_status = service_status.get("classifier", {})
            model_loaded = bool(classifier_status.get("trained"))
            
            return {
                "status": "healthy" if is_healthy else "unhealthy",
                "service": "ai-voice-detection",
                "version": "1.0.0",
                "timestamp": time.time(),
                "services": service_status,
                "checks": {
                    "services_initialized": is_healthy,
                    "model_loaded": model_loaded
                }
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "service": "ai-voice-detection",
                    "version": "1.0.0",
                    "timestamp": time.time(),
                    "error": str(e)
                }
            )
    
    @app.get("/metrics")
    async def metrics_endpoint():
        """Prometheus metrics endpoint."""
        config = get_config()
        metrics = get_prometheus_metrics()
        if not config.monitoring.metrics_enabled or not metrics.enabled:
            raise HTTPException(status_code=404, detail="Metrics endpoint disabled")
        
        # Return Prometheus metrics
        content = metrics.get_metrics()
        content_type = metrics.get_content_type()
        
        return Response(content=content, media_type=content_type)
    
    @app.get("/")
    async def root():
        """Root endpoint with basic service information."""
        config = get_config()
        return {
            "service": "AI Voice Detection System",
            "version": "1.0.0",
            "description": "REST API service for classifying audio samples as AI-generated or human-spoken voices",
            "endpoints": {
                "detect_voice": "/api/v1/detect-voice",
                "history": "/api/v1/history",
                "report_pdf": "/api/v1/reports/{scan_id}.pdf",
                "web_app": "/app",
                "health": "/health",
                "detailed_health": "/health/detailed",
                "metrics": "/metrics" if config.monitoring.metrics_enabled and get_prometheus_metrics().enabled else "disabled",
                "docs": "/docs" if config.debug else "disabled"
            }
        }


def _add_frontend_routes(app: FastAPI) -> None:
    """Serve the static web UI without affecting the existing API root."""
    ui_dir = Path(__file__).resolve().parent.parent / "webui"
    assets_dir = ui_dir / "assets"
    index_file = ui_dir / "index.html"

    if not assets_dir.exists() or not index_file.exists():
        logger.warning("Web UI assets not found - frontend routes will not be mounted")
        return

    app.mount("/app-assets", StaticFiles(directory=assets_dir), name="app-assets")

    @app.get("/app", include_in_schema=False)
    async def serve_web_app():
        return FileResponse(index_file)

    @app.get("/app/{full_path:path}", include_in_schema=False)
    async def serve_web_app_routes(full_path: str):
        return FileResponse(index_file)


async def _startup_services() -> None:
    """Handle application startup tasks."""
    config = get_config()
    logger.info("AI Voice Detection System starting up")
    logger.info(f"Debug mode: {config.debug}")
    logger.info(f"Environment: {config.environment}")
    logger.info(f"CORS origins: {config.api.cors_origins}")

    if config.monitoring.metrics_enabled:
        logger.info("Initializing Prometheus metrics")
        initialize_system_metrics()

        if config.monitoring.prometheus_port and config.monitoring.prometheus_port != 8000:
            success = start_metrics_server(config.monitoring.prometheus_port)
            if success:
                logger.info(
                    f"Prometheus metrics server started on port {config.monitoring.prometheus_port}"
                )
            else:
                logger.warning("Failed to start Prometheus metrics server")
    else:
        logger.info("Prometheus metrics disabled")

    try:
        from ..services import initialize_services, get_service_manager

        logger.info("Initializing services...")
        initialize_services()

        service_manager = get_service_manager()
        if service_manager.is_initialized():
            logger.info("Services are properly initialized")

            status = service_manager.get_service_status()
            for service_name, service_status in status.items():
                logger.info(f"Service {service_name}: {service_status}")
        else:
            logger.error("Services failed to initialize properly")
    except Exception as e:
        logger.error(f"Failed to initialize services during startup: {e}", exc_info=True)


async def _shutdown_services() -> None:
    """Handle application shutdown tasks."""
    logger.info("AI Voice Detection System shutting down")

    try:
        from ..services import shutdown_services

        logger.info("Shutting down services...")
        shutdown_services()
        logger.info("Services shutdown complete")
    except Exception as e:
        logger.error(f"Error during service shutdown: {e}", exc_info=True)


def _create_lifespan():
    """Create the FastAPI lifespan handler."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await _startup_services()
        try:
            yield
        finally:
            await _shutdown_services()

    return lifespan


# Create the application instance
app = create_app()
