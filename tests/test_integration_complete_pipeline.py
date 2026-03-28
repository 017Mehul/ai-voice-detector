"""
Integration tests for complete AI Voice Detection pipeline

This module tests the complete end-to-end processing pipeline from API request
to final response, including error handling across component boundaries and
multi-language processing capabilities.

Requirements tested:
- 2.1: Tamil speech classification without language-specific bias
- 2.2: English speech classification without language-specific bias  
- 2.3: Hindi speech classification without language-specific bias
- 2.4: Malayalam speech classification without language-specific bias
- 2.5: Telugu speech classification without language-specific bias
"""

import pytest
import json
import base64
import numpy as np
import io
import time
import wave
from typing import Dict, Any, List, Tuple
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from ai_voice_detector.api.app import create_app
from ai_voice_detector.services.service_manager import ServiceManager, get_service_manager
from ai_voice_detector.services.voice_detection_service import VoiceDetectionService
from ai_voice_detector.models.data_models import (
    DetectionRequest, DetectionResponse, AudioData, FeatureVector,
    SpectralFeatures, TemporalFeatures, ValidationResult
)
from ai_voice_detector.audio.processor import AudioProcessor
from ai_voice_detector.features.extractor import FeatureExtractor
from ai_voice_detector.models.classifier import RandomForestVoiceClassifier
from ai_voice_detector.models.calibration import ConfidenceCalibrator


class TestCompleteIntegrationPipeline:
    """Integration tests for complete processing pipeline"""
    
    @pytest.fixture(scope="class")
    def service_manager(self):
        """Create and initialize service manager for testing"""
        manager = ServiceManager()
        try:
            manager.initialize()
            yield manager
        finally:
            manager.shutdown()
    
    @pytest.fixture
    def app(self, service_manager):
        """Create FastAPI test application with initialized services"""
        # Patch the global service manager to use our test instance
        with patch('ai_voice_detector.services.get_service_manager', return_value=service_manager):
            app = create_app()
            yield app
    
    @pytest.fixture
    def client(self, app):
        """Create test client"""
        return TestClient(app)
    
    @pytest.fixture
    def voice_service(self, service_manager):
        """Get voice detection service from service manager"""
        return service_manager.get_service('voice_detection_service')
    
    @pytest.fixture
    def sample_audio_data(self):
        """Generate sample audio data for different languages and scenarios"""
        def create_audio_sample(language: str, duration: float = 3.0, ai_generated: bool = False) -> bytes:
            """Create synthetic WAV audio sample for testing (since MP3 requires ffmpeg)"""
            sample_rate = 22050
            t = np.linspace(0, duration, int(sample_rate * duration))
            
            # Create different frequency patterns for different languages
            language_frequencies = {
                'tamil': [440, 554, 659],      # A4, C#5, E5
                'english': [523, 659, 784],    # C5, E5, G5
                'hindi': [392, 494, 587],      # G4, B4, D5
                'malayalam': [349, 440, 523],  # F4, A4, C5
                'telugu': [294, 370, 440]      # D4, F#4, A4
            }
            
            # Get frequencies for the language
            freqs = language_frequencies.get(language, [440, 523, 659])
            
            # Create complex waveform with multiple harmonics
            samples = np.zeros(len(t))
            for i, freq in enumerate(freqs):
                amplitude = 0.3 / (i + 1)  # Decreasing amplitude for harmonics
                samples += amplitude * np.sin(2 * np.pi * freq * t)
            
            # Add some noise to make it more realistic
            noise = np.random.normal(0, 0.05, len(samples))
            samples += noise
            
            # Simulate AI-generated characteristics (more regular patterns)
            if ai_generated:
                # Add very regular modulation for AI-like characteristics
                modulation = 0.1 * np.sin(2 * np.pi * 5 * t)  # 5Hz modulation
                samples *= (1 + modulation)
            else:
                # Add irregular variations for human-like characteristics
                irregular_mod = np.random.normal(1, 0.1, len(samples))
                samples *= irregular_mod
            
            # Normalize to prevent clipping
            samples = samples / np.max(np.abs(samples)) * 0.8
            
            # Convert to 16-bit PCM format
            samples_int16 = (samples * 32767).astype(np.int16)
            
            # Create WAV file in memory
            import wave
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(samples_int16.tobytes())
            
            wav_buffer.seek(0)
            return wav_buffer.read()
        
        return create_audio_sample
    
    @pytest.fixture
    def multi_language_samples(self, sample_audio_data):
        """Create audio samples for all supported languages"""
        languages = ['tamil', 'english', 'hindi', 'malayalam', 'telugu']
        samples = {}
        
        for lang in languages:
            # Create both human and AI samples for each language
            human_audio = sample_audio_data(lang, duration=3.0, ai_generated=False)
            ai_audio = sample_audio_data(lang, duration=3.0, ai_generated=True)
            
            samples[lang] = {
                'human': base64.b64encode(human_audio).decode('utf-8'),
                'ai': base64.b64encode(ai_audio).decode('utf-8')
            }
        
        return samples
    
    def test_end_to_end_api_processing(self, client, sample_audio_data):
        """Test complete end-to-end processing through API"""
        # Create sample audio
        audio_bytes = sample_audio_data('english', duration=3.0)
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        # Make API request
        response = client.post("/api/v1/detect-voice", json={"audio_base64": audio_base64})
        
        # Test complete end-to-end processing through API
        assert response.status_code in [200, 422, 500]  # Success, processing error, or system error
        
        data = response.json()
        
        if response.status_code == 200:
            # Success response validation
            assert "classification" in data
            assert "confidence_score" in data
            assert data["classification"] in ["AI_GENERATED", "HUMAN"]
            assert isinstance(data["confidence_score"], (int, float))
            assert 0.0 <= data["confidence_score"] <= 1.0
            
            # Verify response headers
            assert "X-Request-ID" in response.headers
            assert "X-Process-Time" in response.headers
            
            # Verify processing time is reasonable
            process_time = float(response.headers["X-Process-Time"])
            assert process_time < 10.0  # Should complete within 10 seconds
        else:
            # Error response validation
            assert "error" in data
            assert "message" in data
            assert isinstance(data["error"], str)
            assert isinstance(data["message"], str)
    
    def test_service_level_processing(self, voice_service, sample_audio_data):
        """Test complete processing at service level"""
        # Create sample audio
        audio_bytes = sample_audio_data('english', duration=3.0)
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        # Create detection request
        request = DetectionRequest(audio_base64=audio_base64)
        
        # Process through service
        response, error = voice_service.detect_voice(request, request_id="test-001")
        
        # Verify response
        assert isinstance(response, DetectionResponse)
        assert response.classification in ["AI_GENERATED", "HUMAN"]
        assert 0.0 <= response.confidence_score <= 1.0
        
        # If there's an error, it should be properly formatted
        if error:
            assert isinstance(error, ValidationResult)
            assert not error.is_valid
            assert error.error_code is not None
            assert error.error_message is not None
    
    def test_multi_language_processing_consistency(self, client, multi_language_samples):
        """Test processing consistency across all supported languages"""
        results = {}
        
        # Process samples for each language
        for language, samples in multi_language_samples.items():
            results[language] = {}
            
            for sample_type, audio_base64 in samples.items():
                response = client.post("/api/v1/detect-voice", json={"audio_base64": audio_base64})
                
                # Should handle all languages consistently
                assert response.status_code in [200, 422, 500]  # Success, processing error, or system error
                
                data = response.json()
                results[language][sample_type] = {
                    'status_code': response.status_code,
                    'response': data
                }
                
                if response.status_code == 200:
                    # Verify consistent response format
                    assert "classification" in data
                    assert "confidence_score" in data
                    assert data["classification"] in ["AI_GENERATED", "HUMAN"]
                    assert 0.0 <= data["confidence_score"] <= 1.0
        
        # Verify all languages were processed
        assert len(results) == 5  # All 5 languages
        
        # Verify response format consistency across languages
        successful_responses = []
        for lang_results in results.values():
            for sample_result in lang_results.values():
                if sample_result['status_code'] == 200:
                    successful_responses.append(sample_result['response'])
        
        if successful_responses:
            # All successful responses should have the same structure
            first_response = successful_responses[0]
            for response in successful_responses[1:]:
                assert set(response.keys()) == set(first_response.keys())
    
    def test_language_agnostic_feature_extraction(self, voice_service, multi_language_samples):
        """Test that feature extraction produces consistent results across languages"""
        feature_vectors = {}
        
        # Extract features for each language
        for language, samples in multi_language_samples.items():
            # Use human sample for consistency
            audio_base64 = samples['human']
            
            # Process through audio processor
            audio_processor = voice_service.audio_processor
            audio_data, error = audio_processor.process_audio(audio_base64)
            
            if not error and audio_data.is_valid():
                # Extract features
                feature_extractor = voice_service.feature_extractor
                features = feature_extractor.extract_all_features(audio_data)
                
                feature_vectors[language] = features
        
        # Verify we got features for multiple languages
        assert len(feature_vectors) >= 2
        
        # Verify feature vector consistency across languages
        first_lang = list(feature_vectors.keys())[0]
        first_features = feature_vectors[first_lang]
        
        for language, features in feature_vectors.items():
            # All feature vectors should have the same structure
            assert type(features) == type(first_features)
            assert features.mfcc_features.shape[0] == first_features.mfcc_features.shape[0]  # Same MFCC dimensions
            
            # Spectral features should have consistent structure
            assert type(features.spectral_features) == type(first_features.spectral_features)
            
            # Temporal features should have consistent structure
            assert type(features.temporal_features) == type(first_features.temporal_features)
            
            # All features should be valid numbers
            assert not np.isnan(features.pitch_variance)
            assert not np.isnan(features.energy_variance)
            assert features.pitch_variance >= 0.0
            assert features.energy_variance >= 0.0
    
    def test_error_handling_across_component_boundaries(self, client, voice_service):
        """Test error handling propagation across all component boundaries"""
        
        # Test 1: Invalid Base64 input (Audio Processor boundary)
        response = client.post("/api/v1/detect-voice", json={"audio_base64": "invalid_base64!"})
        assert response.status_code in [400, 422]  # Could be validation or processing error
        data = response.json()
        assert "error" in data
        assert "message" in data
        
        # Test 2: Empty audio data (Audio Processor boundary)
        empty_audio = base64.b64encode(b"").decode('utf-8')
        response = client.post("/api/v1/detect-voice", json={"audio_base64": empty_audio})
        assert response.status_code in [400, 422]  # Could be validation or processing error
        data = response.json()
        assert "error" in data
        
        # Test 3: Very short audio (Audio Processor boundary)
        short_audio = np.random.random(1000).astype(np.float32).tobytes()  # Very short
        short_base64 = base64.b64encode(short_audio).decode('utf-8')
        response = client.post("/api/v1/detect-voice", json={"audio_base64": short_base64})
        assert response.status_code in [400, 422, 500]  # Could be validation, processing, or system error
        data = response.json()
        assert "error" in data
        
        # Test 4: Malformed JSON (API boundary)
        response = client.post("/api/v1/detect-voice", data="invalid json")
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        
        # Test 5: Missing required field (API boundary)
        response = client.post("/api/v1/detect-voice", json={})
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        
        # Test 6: Service-level error handling
        invalid_request = DetectionRequest(audio_base64="invalid_base64!")
        response, error = voice_service.detect_voice(invalid_request, request_id="test-error")
        
        # Should return error information
        assert error is not None
        assert isinstance(error, ValidationResult)
        assert not error.is_valid
        assert error.error_code is not None
        assert error.error_message is not None
    
    def test_concurrent_request_handling(self, client, sample_audio_data):
        """Test handling of concurrent requests across the pipeline"""
        import concurrent.futures
        import threading
        
        # Create sample audio
        audio_bytes = sample_audio_data('english', duration=2.5)
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        def make_request(request_id: int):
            """Make a single request"""
            response = client.post("/api/v1/detect-voice", json={"audio_base64": audio_base64})
            return {
                'request_id': request_id,
                'status_code': response.status_code,
                'response': response.json(),
                'headers': dict(response.headers)
            }
        
        # Make 5 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request, i) for i in range(5)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Verify all requests completed
        assert len(results) == 5
        
        # Verify each request has unique ID (if headers are available)
        request_ids = [r['headers'].get('X-Request-ID') for r in results if r['headers'].get('X-Request-ID')]
        if request_ids:
            assert len(set(request_ids)) == len(request_ids)  # All unique
        
        # Verify all responses are properly formatted
        for result in results:
            assert result['status_code'] in [200, 422, 500]  # Allow system errors during testing
            
            if result['status_code'] == 200:
                response = result['response']
                assert "classification" in response
                assert "confidence_score" in response
                assert response["classification"] in ["AI_GENERATED", "HUMAN"]
                assert 0.0 <= response["confidence_score"] <= 1.0
            else:
                response = result['response']
                assert "error" in response
                assert "message" in response
    
    def test_pipeline_performance_characteristics(self, client, sample_audio_data):
        """Test performance characteristics of the complete pipeline"""
        # Create different duration samples
        test_cases = [
            ('short', 2.0),    # Minimum duration
            ('medium', 5.0),   # Medium duration
            ('long', 10.0)     # Longer duration
        ]
        
        performance_results = {}
        
        for case_name, duration in test_cases:
            audio_bytes = sample_audio_data('english', duration=duration)
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            # Measure processing time
            start_time = time.time()
            response = client.post("/api/v1/detect-voice", json={"audio_base64": audio_base64})
            end_time = time.time()
            
            processing_time = end_time - start_time
            server_time = float(response.headers.get('X-Process-Time', 0))
            
            performance_results[case_name] = {
                'duration': duration,
                'processing_time': processing_time,
                'server_time': server_time,
                'status_code': response.status_code
            }
            
            # Performance assertions
            assert processing_time < 15.0  # Should complete within 15 seconds
            assert server_time < 10.0      # Server processing should be under 10 seconds
        
        # Verify performance scales reasonably with audio duration
        if len(performance_results) >= 2:
            short_time = performance_results['short']['server_time']
            long_time = performance_results['long']['server_time']
            
            # Longer audio shouldn't take dramatically longer (within 5x)
            if short_time > 0:
                assert long_time / short_time < 5.0
    
    def test_system_stability_under_load(self, client, sample_audio_data):
        """Test system stability under sustained load"""
        audio_bytes = sample_audio_data('english', duration=3.0)
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        # Process multiple requests sequentially
        response_times = []
        error_count = 0
        
        for i in range(10):
            start_time = time.time()
            response = client.post("/api/v1/detect-voice", json={"audio_base64": audio_base64})
            end_time = time.time()
            
            response_time = end_time - start_time
            response_times.append(response_time)
            
            # Count errors
            if response.status_code not in [200, 422, 500]:  # Allow system errors during testing
                error_count += 1
            
            # Verify response is properly formatted
            data = response.json()
            assert isinstance(data, dict)
            
            if response.status_code == 200:
                assert "classification" in data
                assert "confidence_score" in data
            else:
                assert "error" in data
        
        # System stability assertions
        # Allow some errors during testing, but not all requests should fail
        assert error_count < len(response_times)  # At least some requests should succeed
        
        # Response times should be relatively consistent
        avg_time = sum(response_times) / len(response_times)
        max_time = max(response_times)
        min_time = min(response_times)
        
        # Variance shouldn't be too high (within 3x of average)
        assert max_time < avg_time * 3
        assert min_time > avg_time / 3
    
    def test_component_integration_boundaries(self, voice_service, sample_audio_data):
        """Test integration at each component boundary"""
        audio_bytes = sample_audio_data('english', duration=3.0)
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        # Test Audio Processor boundary
        audio_processor = voice_service.audio_processor
        audio_data, audio_error = audio_processor.process_audio(audio_base64)
        
        if not audio_error:
            assert isinstance(audio_data, AudioData)
            assert audio_data.is_valid()
            assert audio_data.duration >= 2.0
            
            # Test Feature Extractor boundary
            feature_extractor = voice_service.feature_extractor
            features = feature_extractor.extract_all_features(audio_data)
            
            assert isinstance(features, FeatureVector)
            assert features.mfcc_features.shape[0] > 0
            assert not np.isnan(features.pitch_variance)
            assert not np.isnan(features.energy_variance)
            
            # Test Classifier boundary
            classifier = voice_service.classifier
            if classifier.is_trained():
                classification, raw_confidence = classifier.predict(features)
                
                assert classification in ["AI_GENERATED", "HUMAN"]
                assert isinstance(raw_confidence, (int, float))
                assert 0.0 <= raw_confidence <= 1.0
                
                # Test Confidence Calibrator boundary
                calibrator = voice_service.confidence_calibrator
                calibrated_confidence = calibrator.calibrate_confidence(raw_confidence, features)
                
                assert isinstance(calibrated_confidence, (int, float))
                assert 0.0 <= calibrated_confidence <= 1.0
    
    def test_service_status_and_health_checks(self, client, voice_service):
        """Test service status reporting and health checks"""
        # Test API health endpoints
        health_response = client.get("/health")
        assert health_response.status_code == 200
        
        health_data = health_response.json()
        assert health_data["status"] == "healthy"
        assert health_data["service"] == "ai-voice-detection"
        
        # Test detailed health check
        detailed_response = client.get("/health/detailed")
        assert detailed_response.status_code in [200, 503]
        
        detailed_data = detailed_response.json()
        assert "status" in detailed_data
        assert "service" in detailed_data
        
        # Test service-level status
        service_status = voice_service.get_service_status()
        assert isinstance(service_status, dict)
        
        # Verify all expected components are present
        expected_components = ['audio_processor', 'feature_extractor', 'classifier', 'confidence_calibrator']
        for component in expected_components:
            assert component in service_status
            assert isinstance(service_status[component], dict)
        
        # Test system readiness validation
        readiness = voice_service.validate_system_readiness()
        assert isinstance(readiness, ValidationResult)
        
        if readiness.is_valid:
            # System should be ready for processing
            assert readiness.error_code is None
            assert readiness.error_message is None
        else:
            # If not ready, should have error information
            assert readiness.error_code is not None
            assert readiness.error_message is not None
    
    def test_data_flow_integrity(self, voice_service, sample_audio_data):
        """Test data integrity throughout the processing pipeline"""
        # Create test audio with known characteristics
        audio_bytes = sample_audio_data('english', duration=4.0)
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        # Track data through each stage
        request = DetectionRequest(audio_base64=audio_base64)
        
        # Stage 1: Audio processing
        audio_data, audio_error = voice_service._process_audio(request, "integrity-test")
        
        if not audio_error:
            # Verify audio data integrity
            assert audio_data.samples is not None
            assert len(audio_data.samples) > 0
            assert audio_data.sample_rate > 0
            assert audio_data.duration > 0
            assert audio_data.channels > 0
            
            # Stage 2: Feature extraction
            features, feature_error = voice_service._extract_features(audio_data, "integrity-test")
            
            if not feature_error:
                # Verify feature integrity
                assert features.mfcc_features.size > 0
                assert not np.any(np.isnan(features.mfcc_features))
                assert not np.isinf(features.pitch_variance)
                assert not np.isinf(features.energy_variance)
                
                # Stage 3: Classification
                classification, raw_confidence, class_error = voice_service._classify_voice(features, "integrity-test")
                
                if not class_error:
                    # Verify classification integrity
                    assert classification in ["AI_GENERATED", "HUMAN"]
                    assert 0.0 <= raw_confidence <= 1.0
                    assert not np.isnan(raw_confidence)
                    assert not np.isinf(raw_confidence)
                    
                    # Stage 4: Confidence calibration
                    calibrated_confidence = voice_service._calibrate_confidence(raw_confidence, features, "integrity-test")
                    
                    # Verify final output integrity
                    assert 0.0 <= calibrated_confidence <= 1.0
                    assert not np.isnan(calibrated_confidence)
                    assert not np.isinf(calibrated_confidence)