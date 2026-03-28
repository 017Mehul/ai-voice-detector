"""
Prometheus Metrics Integration

This module provides Prometheus metrics collection for the AI Voice Detection System.
It tracks request counts, processing times, error rates, and system resource usage.
"""

from __future__ import annotations

import time
import logging
from typing import Dict, Any, Optional
from contextlib import contextmanager
from functools import wraps

try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Info, CollectorRegistry, 
        generate_latest, CONTENT_TYPE_LATEST, start_http_server
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)


class _NoopMetric:
    """No-op replacement used when Prometheus is unavailable."""

    def labels(self, **kwargs):
        return self

    def inc(self, amount: float = 1.0):
        return None

    def observe(self, value):
        return None

    def set(self, value):
        return None

    def info(self, info):
        return None


class PrometheusMetrics:
    """
    Prometheus metrics collector for AI Voice Detection System.
    
    Provides comprehensive metrics collection including:
    - Request counts and response times
    - Error rates and types
    - System resource usage
    - Model performance metrics
    - Cache performance
    """
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """Initialize Prometheus metrics."""
        if not PROMETHEUS_AVAILABLE:
            logger.warning("prometheus_client not available - metrics collection disabled")
            self.enabled = False
            self.registry = None
            self.request_count = _NoopMetric()
            self.request_duration = _NoopMetric()
            self.audio_processing_duration = _NoopMetric()
            self.feature_extraction_duration = _NoopMetric()
            self.model_inference_duration = _NoopMetric()
            self.error_count = _NoopMetric()
            self.memory_usage = _NoopMetric()
            self.cpu_usage = _NoopMetric()
            self.cache_hits = _NoopMetric()
            self.cache_misses = _NoopMetric()
            self.classification_count = _NoopMetric()
            self.confidence_score = _NoopMetric()
            self.audio_duration = _NoopMetric()
            self.audio_sample_rate = _NoopMetric()
            self.system_info = _NoopMetric()
            return
        
        self.enabled = True
        self.registry = registry or CollectorRegistry()
        
        # Request metrics
        self.request_count = Counter(
            'ai_voice_detector_requests_total',
            'Total number of requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            'ai_voice_detector_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint'],
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=self.registry
        )
        
        # Processing metrics
        self.audio_processing_duration = Histogram(
            'ai_voice_detector_audio_processing_seconds',
            'Audio processing duration in seconds',
            ['stage'],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
            registry=self.registry
        )
        
        self.feature_extraction_duration = Histogram(
            'ai_voice_detector_feature_extraction_seconds',
            'Feature extraction duration in seconds',
            ['feature_type'],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
            registry=self.registry
        )
        
        self.model_inference_duration = Histogram(
            'ai_voice_detector_model_inference_seconds',
            'Model inference duration in seconds',
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5],
            registry=self.registry
        )
        
        # Error metrics
        self.error_count = Counter(
            'ai_voice_detector_errors_total',
            'Total number of errors',
            ['error_type', 'component'],
            registry=self.registry
        )
        
        # System metrics
        self.memory_usage = Gauge(
            'ai_voice_detector_memory_usage_bytes',
            'Current memory usage in bytes',
            registry=self.registry
        )
        
        self.cpu_usage = Gauge(
            'ai_voice_detector_cpu_usage_percent',
            'Current CPU usage percentage',
            registry=self.registry
        )
        
        # Cache metrics
        self.cache_hits = Counter(
            'ai_voice_detector_cache_hits_total',
            'Total cache hits',
            ['cache_type'],
            registry=self.registry
        )
        
        self.cache_misses = Counter(
            'ai_voice_detector_cache_misses_total',
            'Total cache misses',
            ['cache_type'],
            registry=self.registry
        )
        
        # Model performance metrics
        self.classification_count = Counter(
            'ai_voice_detector_classifications_total',
            'Total classifications performed',
            ['classification', 'language'],
            registry=self.registry
        )
        
        self.confidence_score = Histogram(
            'ai_voice_detector_confidence_score',
            'Classification confidence scores',
            ['classification'],
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            registry=self.registry
        )
        
        # Audio quality metrics
        self.audio_duration = Histogram(
            'ai_voice_detector_audio_duration_seconds',
            'Audio sample duration in seconds',
            buckets=[1, 2, 5, 10, 30, 60, 120],
            registry=self.registry
        )
        
        self.audio_sample_rate = Histogram(
            'ai_voice_detector_audio_sample_rate_hz',
            'Audio sample rate in Hz',
            buckets=[8000, 16000, 22050, 44100, 48000, 96000],
            registry=self.registry
        )
        
        # System info
        self.system_info = Info(
            'ai_voice_detector_system_info',
            'System information',
            registry=self.registry
        )
        
        logger.info("Prometheus metrics initialized successfully")
    
    def record_request(self, method: str, endpoint: str, status: int, duration: float):
        """Record HTTP request metrics."""
        if not self.enabled:
            return
        
        self.request_count.labels(method=method, endpoint=endpoint, status=str(status)).inc()
        self.request_duration.labels(method=method, endpoint=endpoint).observe(duration)
    
    def record_audio_processing(self, stage: str, duration: float):
        """Record audio processing metrics."""
        if not self.enabled:
            return
        
        self.audio_processing_duration.labels(stage=stage).observe(duration)
    
    def record_feature_extraction(self, feature_type: str, duration: float):
        """Record feature extraction metrics."""
        if not self.enabled:
            return
        
        self.feature_extraction_duration.labels(feature_type=feature_type).observe(duration)
    
    def record_model_inference(self, duration: float):
        """Record model inference metrics."""
        if not self.enabled:
            return
        
        self.model_inference_duration.observe(duration)
    
    def record_error(self, error_type: str, component: str):
        """Record error metrics."""
        if not self.enabled:
            return
        
        self.error_count.labels(error_type=error_type, component=component).inc()
    
    def update_system_metrics(self, memory_bytes: float, cpu_percent: float):
        """Update system resource metrics."""
        if not self.enabled:
            return
        
        self.memory_usage.set(memory_bytes)
        self.cpu_usage.set(cpu_percent)
    
    def record_cache_hit(self, cache_type: str):
        """Record cache hit."""
        if not self.enabled:
            return
        
        self.cache_hits.labels(cache_type=cache_type).inc()
    
    def record_cache_miss(self, cache_type: str):
        """Record cache miss."""
        if not self.enabled:
            return
        
        self.cache_misses.labels(cache_type=cache_type).inc()
    
    def record_classification(self, classification: str, confidence: float, language: str = "unknown"):
        """Record classification metrics."""
        if not self.enabled:
            return
        
        self.classification_count.labels(classification=classification, language=language).inc()
        self.confidence_score.labels(classification=classification).observe(confidence)
    
    def record_audio_properties(self, duration: float, sample_rate: int):
        """Record audio properties metrics."""
        if not self.enabled:
            return
        
        self.audio_duration.observe(duration)
        self.audio_sample_rate.observe(sample_rate)
    
    def set_system_info(self, info: Dict[str, str]):
        """Set system information."""
        if not self.enabled:
            return
        
        self.system_info.info(info)
    
    def get_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        if not self.enabled:
            return "# Prometheus metrics not available\n"
        
        return generate_latest(self.registry).decode('utf-8')
    
    def get_content_type(self) -> str:
        """Get Prometheus content type."""
        if not self.enabled:
            return "text/plain"
        
        return CONTENT_TYPE_LATEST
    
    @contextmanager
    def time_operation(self, operation_type: str, **labels):
        """Context manager for timing operations."""
        if not self.enabled:
            yield
            return
        
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            
            if operation_type == "request":
                self.record_request(duration=duration, **labels)
            elif operation_type == "audio_processing":
                self.record_audio_processing(duration=duration, **labels)
            elif operation_type == "feature_extraction":
                self.record_feature_extraction(duration=duration, **labels)
            elif operation_type == "model_inference":
                self.record_model_inference(duration)


class MetricsMiddleware:
    """
    Middleware for automatic metrics collection.
    """
    
    def __init__(self, metrics: PrometheusMetrics):
        """Initialize metrics middleware."""
        self.metrics = metrics
    
    def __call__(self, request, call_next):
        """Process request and collect metrics."""
        start_time = time.time()
        
        try:
            response = call_next(request)
            duration = time.time() - start_time
            
            # Record successful request
            self.metrics.record_request(
                method=request.method,
                endpoint=request.url.path,
                status=response.status_code,
                duration=duration
            )
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            
            # Record failed request
            self.metrics.record_request(
                method=request.method,
                endpoint=request.url.path,
                status=500,
                duration=duration
            )
            
            # Record error
            self.metrics.record_error(
                error_type=type(e).__name__,
                component="api"
            )
            
            raise


def metrics_decorator(operation_type: str, **labels):
    """
    Decorator for automatic metrics collection.
    
    Args:
        operation_type: Type of operation being measured
        **labels: Additional labels for the metric
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            metrics = get_prometheus_metrics()
            
            with metrics.time_operation(operation_type, **labels):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    metrics.record_error(
                        error_type=type(e).__name__,
                        component=operation_type
                    )
                    raise
        
        return wrapper
    return decorator


# Global metrics instance
_prometheus_metrics: Optional[PrometheusMetrics] = None


def get_prometheus_metrics() -> PrometheusMetrics:
    """Get the global Prometheus metrics instance."""
    global _prometheus_metrics
    
    if _prometheus_metrics is None:
        _prometheus_metrics = PrometheusMetrics()
    
    return _prometheus_metrics


def start_metrics_server(port: int = 9090) -> bool:
    """
    Start Prometheus metrics HTTP server.
    
    Args:
        port: Port to serve metrics on
        
    Returns:
        bool: True if server started successfully
    """
    if not PROMETHEUS_AVAILABLE:
        logger.warning("prometheus_client not available - metrics server not started")
        return False
    
    try:
        start_http_server(port)
        logger.info(f"Prometheus metrics server started on port {port}")
        return True
    except Exception as e:
        logger.error(f"Failed to start metrics server: {str(e)}")
        return False


def initialize_system_metrics():
    """Initialize system-level metrics."""
    metrics = get_prometheus_metrics()
    
    if not metrics.enabled:
        return
    
    # Set system information
    import platform
    import sys
    
    system_info = {
        "version": "1.0.0",
        "python_version": sys.version.split()[0],
        "platform": platform.system(),
        "architecture": platform.machine(),
        "hostname": platform.node()
    }
    
    metrics.set_system_info(system_info)
    logger.info("System metrics initialized")
