"""
Performance Monitoring and Optimization Utilities

This module provides utilities for monitoring and optimizing system performance,
including memory management, timing analysis, and resource usage tracking.
"""

import time
import gc
import threading
from typing import Dict, Any, Optional, Callable
from contextlib import contextmanager
from dataclasses import dataclass
import logging

# Optional psutil import for enhanced monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    processing_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    peak_memory_mb: Optional[float] = None
    cache_hits: int = 0
    cache_misses: int = 0


class PerformanceMonitor:
    """
    Performance monitoring utility for tracking system resource usage.
    """
    
    def __init__(self):
        """Initialize performance monitor."""
        if PSUTIL_AVAILABLE:
            self.process = psutil.Process()
            self.baseline_memory = self._get_memory_usage()
        else:
            self.process = None
            self.baseline_memory = 0.0
            logger.warning("psutil not available - memory monitoring will be limited")
        
        self.metrics_history = []
        self.cache_stats = {"hits": 0, "misses": 0}
        
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if PSUTIL_AVAILABLE and self.process:
            try:
                return self.process.memory_info().rss / 1024 / 1024
            except Exception:
                return 0.0
        return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        if PSUTIL_AVAILABLE and self.process:
            try:
                return self.process.cpu_percent()
            except Exception:
                return 0.0
        return 0.0
    
    @contextmanager
    def measure_performance(self, operation_name: str = "operation"):
        """
        Context manager for measuring performance of operations.
        
        Args:
            operation_name: Name of the operation being measured
            
        Yields:
            PerformanceMetrics: Metrics object that gets populated during execution
        """
        start_time = time.time()
        start_memory = self._get_memory_usage()
        peak_memory = start_memory
        
        # Create metrics object
        metrics = PerformanceMetrics(
            processing_time=0.0,
            memory_usage_mb=start_memory,
            cpu_usage_percent=0.0
        )
        
        # Start memory monitoring in background
        monitoring = True
        
        def monitor_memory():
            nonlocal peak_memory, monitoring
            while monitoring:
                current_memory = self._get_memory_usage()
                peak_memory = max(peak_memory, current_memory)
                time.sleep(0.1)  # Check every 100ms
        
        monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
        monitor_thread.start()
        
        try:
            yield metrics
        finally:
            # Stop monitoring
            monitoring = False
            monitor_thread.join(timeout=0.5)
            
            # Calculate final metrics
            end_time = time.time()
            end_memory = self._get_memory_usage()
            cpu_usage = self._get_cpu_usage()
            
            metrics.processing_time = end_time - start_time
            metrics.memory_usage_mb = end_memory
            metrics.peak_memory_mb = peak_memory
            metrics.cpu_usage_percent = cpu_usage
            metrics.cache_hits = self.cache_stats["hits"]
            metrics.cache_misses = self.cache_stats["misses"]
            
            # Store in history
            self.metrics_history.append({
                "operation": operation_name,
                "timestamp": end_time,
                "metrics": metrics
            })
            
            # Log performance info
            logger.info(
                f"Performance [{operation_name}]: "
                f"time={metrics.processing_time:.3f}s, "
                f"memory={metrics.memory_usage_mb:.1f}MB, "
                f"peak_memory={metrics.peak_memory_mb:.1f}MB, "
                f"cpu={metrics.cpu_usage_percent:.1f}%"
            )
    
    def record_cache_hit(self):
        """Record a cache hit."""
        self.cache_stats["hits"] += 1
    
    def record_cache_miss(self):
        """Record a cache miss."""
        self.cache_stats["misses"] += 1
    
    def get_cache_hit_rate(self) -> float:
        """Get cache hit rate as a percentage."""
        total = self.cache_stats["hits"] + self.cache_stats["misses"]
        if total == 0:
            return 0.0
        return (self.cache_stats["hits"] / total) * 100
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get current system resource information."""
        if not PSUTIL_AVAILABLE or not self.process:
            return {
                "psutil_available": False,
                "cache": {
                    "hits": self.cache_stats["hits"],
                    "misses": self.cache_stats["misses"],
                    "hit_rate_percent": self.get_cache_hit_rate()
                },
                "metrics_history_count": len(self.metrics_history)
            }
        
        try:
            memory_info = self.process.memory_info()
            cpu_times = self.process.cpu_times()
            
            return {
                "psutil_available": True,
                "memory": {
                    "rss_mb": memory_info.rss / 1024 / 1024,
                    "vms_mb": memory_info.vms / 1024 / 1024,
                    "baseline_mb": self.baseline_memory,
                    "increase_mb": (memory_info.rss / 1024 / 1024) - self.baseline_memory
                },
                "cpu": {
                    "percent": self.process.cpu_percent(),
                    "user_time": cpu_times.user,
                    "system_time": cpu_times.system
                },
                "cache": {
                    "hits": self.cache_stats["hits"],
                    "misses": self.cache_stats["misses"],
                    "hit_rate_percent": self.get_cache_hit_rate()
                },
                "metrics_history_count": len(self.metrics_history)
            }
        except Exception as e:
            logger.warning(f"Failed to get system info: {str(e)}")
            return {"error": str(e)}
    
    def clear_history(self):
        """Clear performance metrics history."""
        self.metrics_history.clear()
        logger.debug("Performance metrics history cleared")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of recent performance metrics."""
        if not self.metrics_history:
            return {"message": "No performance data available"}
        
        recent_metrics = self.metrics_history[-10:]  # Last 10 operations
        
        processing_times = [m["metrics"].processing_time for m in recent_metrics]
        memory_usages = [m["metrics"].memory_usage_mb for m in recent_metrics]
        
        return {
            "recent_operations": len(recent_metrics),
            "avg_processing_time": sum(processing_times) / len(processing_times),
            "max_processing_time": max(processing_times),
            "min_processing_time": min(processing_times),
            "avg_memory_usage": sum(memory_usages) / len(memory_usages),
            "max_memory_usage": max(memory_usages),
            "cache_hit_rate": self.get_cache_hit_rate(),
            "total_operations": len(self.metrics_history)
        }


class MemoryOptimizer:
    """
    Memory optimization utility for managing memory usage.
    """
    
    def __init__(self, max_memory_mb: float = 1000.0):
        """
        Initialize memory optimizer.
        
        Args:
            max_memory_mb: Maximum memory usage threshold in MB
        """
        self.max_memory_mb = max_memory_mb
        if PSUTIL_AVAILABLE:
            self.process = psutil.Process()
        else:
            self.process = None
            logger.warning("psutil not available - memory optimization will be limited")
        
    def get_current_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        if PSUTIL_AVAILABLE and self.process:
            try:
                return self.process.memory_info().rss / 1024 / 1024
            except Exception:
                return 0.0
        return 0.0
    
    def is_memory_pressure(self) -> bool:
        """Check if system is under memory pressure."""
        current_memory = self.get_current_memory_mb()
        return current_memory > self.max_memory_mb
    
    def force_garbage_collection(self) -> Dict[str, int]:
        """Force garbage collection and return statistics."""
        logger.debug("Forcing garbage collection")
        
        # Get initial counts
        initial_objects = len(gc.get_objects())
        
        # Force collection
        collected = gc.collect()
        
        # Get final counts
        final_objects = len(gc.get_objects())
        
        stats = {
            "collected_objects": collected,
            "initial_objects": initial_objects,
            "final_objects": final_objects,
            "objects_freed": initial_objects - final_objects
        }
        
        logger.info(f"Garbage collection completed: {stats}")
        return stats
    
    def optimize_memory_if_needed(self, service_manager=None) -> bool:
        """
        Optimize memory usage if under pressure.
        
        Args:
            service_manager: Optional service manager to clear caches
            
        Returns:
            bool: True if optimization was performed
        """
        if not self.is_memory_pressure():
            return False
        
        logger.warning(f"Memory pressure detected: {self.get_current_memory_mb():.1f}MB > {self.max_memory_mb}MB")
        
        # Clear service caches if available
        if service_manager and hasattr(service_manager, 'clear_caches'):
            service_manager.clear_caches()
        
        # Force garbage collection
        self.force_garbage_collection()
        
        # Check if pressure is relieved
        new_memory = self.get_current_memory_mb()
        logger.info(f"Memory after optimization: {new_memory:.1f}MB")
        
        return True


class PerformanceOptimizer:
    """
    Performance optimization coordinator.
    """
    
    def __init__(self, target_response_time: float = 1.0):
        """
        Initialize performance optimizer.
        
        Args:
            target_response_time: Target response time in seconds
        """
        self.target_response_time = target_response_time
        self.monitor = PerformanceMonitor()
        self.memory_optimizer = MemoryOptimizer()
        self.optimization_history = []
        
    def should_optimize_for_speed(self, recent_times: list[float]) -> bool:
        """
        Determine if speed optimizations should be enabled.
        
        Args:
            recent_times: List of recent processing times
            
        Returns:
            bool: True if speed optimization is needed
        """
        if not recent_times:
            return False
        
        avg_time = sum(recent_times) / len(recent_times)
        return avg_time > self.target_response_time
    
    def get_optimization_recommendations(self) -> Dict[str, Any]:
        """
        Get recommendations for performance optimization.
        
        Returns:
            Dictionary containing optimization recommendations
        """
        system_info = self.monitor.get_system_info()
        performance_summary = self.monitor.get_performance_summary()
        
        recommendations = {
            "enable_caching": True,
            "optimize_for_speed": False,
            "clear_caches": False,
            "force_gc": False,
            "reasons": []
        }
        
        # Check processing time
        if "avg_processing_time" in performance_summary:
            avg_time = performance_summary["avg_processing_time"]
            if avg_time > self.target_response_time:
                recommendations["optimize_for_speed"] = True
                recommendations["reasons"].append(f"Average processing time ({avg_time:.3f}s) exceeds target ({self.target_response_time}s)")
        
        # Check memory usage
        if "memory" in system_info:
            memory_increase = system_info["memory"].get("increase_mb", 0)
            if memory_increase > 500:  # More than 500MB increase
                recommendations["clear_caches"] = True
                recommendations["force_gc"] = True
                recommendations["reasons"].append(f"Memory usage increased by {memory_increase:.1f}MB")
        
        # Check cache hit rate
        cache_hit_rate = system_info.get("cache", {}).get("hit_rate_percent", 0)
        if cache_hit_rate < 50:  # Less than 50% hit rate
            recommendations["enable_caching"] = True
            recommendations["reasons"].append(f"Low cache hit rate ({cache_hit_rate:.1f}%)")
        
        return recommendations
    
    def apply_optimizations(self, service_manager=None) -> Dict[str, Any]:
        """
        Apply performance optimizations based on current system state.
        
        Args:
            service_manager: Optional service manager for cache management
            
        Returns:
            Dictionary containing applied optimizations
        """
        recommendations = self.get_optimization_recommendations()
        applied = []
        
        # Apply memory optimizations
        if recommendations["clear_caches"] and service_manager:
            service_manager.clear_caches()
            applied.append("cleared_caches")
        
        if recommendations["force_gc"]:
            self.memory_optimizer.force_garbage_collection()
            applied.append("forced_gc")
        
        # Record optimization
        optimization_record = {
            "timestamp": time.time(),
            "recommendations": recommendations,
            "applied": applied
        }
        self.optimization_history.append(optimization_record)
        
        logger.info(f"Applied optimizations: {applied}")
        return optimization_record


# Global performance monitor instance
_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    global _performance_monitor
    
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    
    return _performance_monitor


def measure_performance(operation_name: str = "operation"):
    """
    Decorator for measuring performance of functions.
    
    Args:
        operation_name: Name of the operation being measured
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            monitor = get_performance_monitor()
            with monitor.measure_performance(operation_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator