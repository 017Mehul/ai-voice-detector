#!/usr/bin/env python3
"""
Performance Benchmark Script for AI Voice Detection System

This script demonstrates the performance improvements achieved through optimization.
"""

import time
import numpy as np
from voice_detector.features.extractor import FeatureExtractor
from voice_detector.models.data_models import AudioData
from voice_detector.utils.performance import get_performance_monitor


def create_test_audio(duration: float = 3.0, sample_rate: int = 22050) -> AudioData:
    """Create test audio data."""
    t = np.linspace(0, duration, int(sample_rate * duration))
    # Create a more complex signal with multiple frequencies
    samples = (
        0.5 * np.sin(2 * np.pi * 440 * t) +  # A4 note
        0.3 * np.sin(2 * np.pi * 880 * t) +  # A5 note
        0.2 * np.sin(2 * np.pi * 220 * t)    # A3 note
    ).astype(np.float32)
    
    return AudioData(
        samples=samples,
        sample_rate=sample_rate,
        duration=duration,
        channels=1
    )


def benchmark_feature_extraction():
    """Benchmark feature extraction performance."""
    print("AI Voice Detection System - Performance Benchmark")
    print("=" * 60)
    
    # Create test audio
    audio_data = create_test_audio(duration=5.0)  # 5 seconds of audio
    print(f"Test audio: {audio_data.duration}s, {audio_data.sample_rate}Hz")
    print()
    
    # Test standard feature extractor
    print("Testing Standard Feature Extractor...")
    standard_extractor = FeatureExtractor(
        enable_caching=False,
        optimize_for_speed=False
    )
    
    start_time = time.time()
    standard_features = standard_extractor.extract_all_features(audio_data)
    standard_time = time.time() - start_time
    
    print(f"  Processing time: {standard_time:.3f}s")
    print(f"  Feature vector size: {len(standard_features.to_array())}")
    print()
    
    # Test optimized feature extractor
    print("Testing Optimized Feature Extractor...")
    optimized_extractor = FeatureExtractor(
        enable_caching=True,
        optimize_for_speed=True
    )
    
    performance_monitor = get_performance_monitor()
    
    with performance_monitor.measure_performance("optimized_extraction") as metrics:
        optimized_features = optimized_extractor.extract_all_features(audio_data)
    
    print(f"  Processing time: {metrics.processing_time:.3f}s")
    print(f"  Feature vector size: {len(optimized_features.to_array())}")
    print(f"  Memory usage: {metrics.memory_usage_mb:.1f}MB")
    print(f"  Peak memory: {metrics.peak_memory_mb:.1f}MB")
    print()
    
    # Performance comparison
    speedup = standard_time / metrics.processing_time if metrics.processing_time > 0 else 1.0
    print("Performance Comparison:")
    print(f"  Standard time:  {standard_time:.3f}s")
    print(f"  Optimized time: {metrics.processing_time:.3f}s")
    print(f"  Speedup:        {speedup:.2f}x")
    print(f"  Improvement:    {((standard_time - metrics.processing_time) / standard_time * 100):.1f}%")
    print()
    
    # Cache effectiveness test
    print("Testing Cache Effectiveness...")
    
    # Second run (should benefit from cache)
    start_time = time.time()
    cached_features = optimized_extractor.extract_all_features(audio_data)
    cached_time = time.time() - start_time
    
    cache_info = optimized_extractor.get_cache_info()
    print(f"  Second run time: {cached_time:.3f}s")
    print(f"  Cache enabled: {cache_info['cache_enabled']}")
    print(f"  Cache size: {cache_info['cache_size']}")
    print(f"  Cache speedup: {metrics.processing_time / cached_time:.2f}x")
    print()
    
    # Memory optimization test
    print("Testing Memory Optimization...")
    
    # Process multiple samples to test memory management
    total_time = 0
    num_samples = 10
    
    for i in range(num_samples):
        start_time = time.time()
        features = optimized_extractor.extract_all_features(audio_data)
        total_time += time.time() - start_time
        
        # Clear cache every 5 samples to test memory management
        if i % 5 == 4:
            optimized_extractor.clear_cache()
    
    avg_time = total_time / num_samples
    print(f"  Average time per sample ({num_samples} samples): {avg_time:.3f}s")
    print(f"  Total processing time: {total_time:.3f}s")
    print(f"  Throughput: {num_samples / total_time:.1f} samples/second")
    print()
    
    # System information
    system_info = performance_monitor.get_system_info()
    performance_summary = performance_monitor.get_performance_summary()
    
    print("System Performance Summary:")
    print(f"  Total operations: {performance_summary.get('total_operations', 0)}")
    print(f"  Average processing time: {performance_summary.get('avg_processing_time', 0):.3f}s")
    print(f"  Max processing time: {performance_summary.get('max_processing_time', 0):.3f}s")
    print(f"  Min processing time: {performance_summary.get('min_processing_time', 0):.3f}s")
    
    if system_info.get("psutil_available", False):
        memory_info = system_info.get("memory", {})
        print(f"  Memory usage: {memory_info.get('rss_mb', 0):.1f}MB")
        print(f"  Memory increase: {memory_info.get('increase_mb', 0):.1f}MB")
    
    cache_stats = system_info.get("cache", {})
    print(f"  Cache hit rate: {cache_stats.get('hit_rate_percent', 0):.1f}%")
    print()
    
    # Performance targets
    print("Performance Targets:")
    target_time = 1.0  # Target: sub-second processing
    if metrics.processing_time <= target_time:
        print(f"  ✅ Sub-second processing: {metrics.processing_time:.3f}s <= {target_time}s")
    else:
        print(f"  ❌ Sub-second processing: {metrics.processing_time:.3f}s > {target_time}s")
    
    if speedup >= 1.2:
        print(f"  ✅ Performance improvement: {speedup:.2f}x speedup")
    else:
        print(f"  ⚠️  Performance improvement: {speedup:.2f}x speedup (target: 1.2x)")
    
    if cache_info['cache_enabled']:
        print(f"  ✅ Caching enabled: {cache_info['cache_size']} entries")
    else:
        print(f"  ❌ Caching disabled")
    
    print()
    print("Benchmark completed successfully!")


if __name__ == "__main__":
    benchmark_feature_extraction()
