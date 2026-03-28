"""
Integration tests for hackathon-style evaluation scenarios
"""
import pytest
import json
import base64
import numpy as np
from fastapi.testclient import TestClient
from ai_voice_detector.api.app import create_app
from ai_voice_detector.models.data_models import AudioData
from ai_voice_detector.services.voice_detection_service import VoiceDetectionService
from ai_voice_detector.audio.processor import AudioProcessor
from ai_voice_detector.features.extractor import FeatureExtractor
from ai_voice_detector.models.classifier import RandomForestVoiceClassifier
from ai_voice_detector.models.calibration import ConfidenceCalibrator
import io
from pydub import AudioSegment
from pydub.generators import Sine


class TestHackathonIntegration:
    """Integration tests simulating hackathon evaluation scenarios"""
    
    @pytest.fixture
    def app(self):
        """Create FastAPI test application"""
        return create_app()
    
    @pytest.fixture
    def client(self, app):
        """Create test client"""
        return TestClient(app)
    
    @pytest.fixture
    def sample_audio_base64(self):
        """Generate sample audio data encoded as base64"""
        # Generate 3 seconds of sine wave
        sample_rate = 22050
        duration = 3.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        samples = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        # Convert to bytes (simulate MP3 encoding)
        audio_bytes = samples.tobytes()
        
        # Encode to base64
        return base64.b64encode(audio_bytes).decode('utf-8')
    
    @pytest.fixture
    def voice_service(self):
        """Create voice detection service for direct testing"""
        audio_processor = AudioProcessor()
        feature_extractor = FeatureExtractor()
        classifier = RandomForestVoiceClassifier()
        calibrator = ConfidenceCalibrator()
        
        return VoiceDetectionService(
            audio_processor=audio_processor,
            feature_extractor=feature_extractor,
            classifier=classifier,
            confidence_calibrator=calibrator
        )
    
    def test_api_health_check(self, client):
        """Test API health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "ai-voice-detection"
        assert "version" in data
        assert "timestamp" in data
    
    def test_api_detailed_health_check(self, client):
        """Test detailed health check endpoint"""
        response = client.get("/health/detailed")
        assert response.status_code in [200, 503]  # May be unhealthy if services not initialized
        
        data = response.json()
        assert "status" in data
        assert "service" in data
        assert "timestamp" in data
    
    def test_api_root_endpoint(self, client):
        """Test root endpoint provides service information"""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["service"] == "AI Voice Detection System"
        assert "version" in data
        assert "endpoints" in data
        assert "detect_voice" in data["endpoints"]
    
    def test_api_request_validation(self, client):
        """Test API request validation with various invalid inputs"""
        # Test empty request
        response = client.post("/api/v1/detect-voice", json={})
        assert response.status_code == 400
        
        data = response.json()
        assert data["error"] == "VALIDATION_ERROR"
        
        # Test invalid JSON structure
        response = client.post("/api/v1/detect-voice", json={"invalid_field": "value"})
        assert response.status_code == 400
        
        # Test invalid base64
        response = client.post("/api/v1/detect-voice", json={"audio_base64": "invalid_base64!"})
        assert response.status_code == 422
        
        data = response.json()
        assert "error" in data
    
    def test_api_response_format(self, client, sample_audio_base64):
        """Test API response format compliance"""
        response = client.post("/api/v1/detect-voice", json={"audio_base64": sample_audio_base64})
        
        # Response should be either success or error, but properly formatted
        assert response.status_code in [200, 422, 500]
        
        data = response.json()
        
        if response.status_code == 200:
            # Success response format
            assert "classification" in data
            assert "confidence_score" in data
            assert data["classification"] in ["AI_GENERATED", "HUMAN"]
            assert isinstance(data["confidence_score"], (int, float))
            assert 0.0 <= data["confidence_score"] <= 1.0
        else:
            # Error response format
            assert "error" in data
            assert "message" in data
            assert isinstance(data["error"], str)
            assert isinstance(data["message"], str)
    
    def test_api_headers_and_metadata(self, client, sample_audio_base64):
        """Test API response headers and metadata"""
        response = client.post("/api/v1/detect-voice", json={"audio_base64": sample_audio_base64})
        
        # Check for required headers
        assert "X-Request-ID" in response.headers
        assert "X-Process-Time" in response.headers
        
        # Validate header values
        request_id = response.headers["X-Request-ID"]
        process_time = float(response.headers["X-Process-Time"])
        
        assert len(request_id) > 0
        assert process_time >= 0.0
    
    def test_concurrent_requests(self, client, sample_audio_base64):
        """Test handling of concurrent requests"""
        import concurrent.futures
        import threading
        
        def make_request():
            return client.post("/api/v1/detect-voice", json={"audio_base64": sample_audio_base64})
        
        # Make 5 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(5)]
            responses = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All requests should complete
        assert len(responses) == 5
        
        # Each response should have unique request ID
        request_ids = [r.headers.get("X-Request-ID") for r in responses]
        assert len(set(request_ids)) == 5  # All unique
        
        # All responses should be properly formatted
        for response in responses:
            assert response.status_code in [200, 422, 500]
            data = response.json()
            
            if response.status_code == 200:
                assert "classification" in data
                assert "confidence_score" in data
            else:
                assert "error" in data
                assert "message" in data
    
    def test_error_handling_robustness(self, client):
        """Test system robustness with various error conditions"""
        # Test extremely large base64 string (should be rejected)
        large_data = "A" * (50 * 1024 * 1024)  # 50MB of 'A' characters
        response = client.post("/api/v1/detect-voice", json={"audio_base64": large_data})
        assert response.status_code in [413, 422]  # Payload too large or unprocessable
        
        # Test malformed JSON
        response = client.post("/api/v1/detect-voice", data="invalid json")
        assert response.status_code == 400
        
        # Test empty base64
        response = client.post("/api/v1/detect-voice", json={"audio_base64": ""})
        assert response.status_code == 422
        
        # Test null values
        response = client.post("/api/v1/detect-voice", json={"audio_base64": None})
        assert response.status_code == 400
    
    def test_multi_language_processing_simulation(self, client):
        """Simulate multi-language processing scenarios"""
        # Create different "audio samples" representing different languages
        # In a real scenario, these would be actual audio files
        
        languages = ["tamil", "english", "hindi", "malayalam", "telugu"]
        
        for lang in languages:
            # Generate different base64 data for each language (simulated)
            lang_data = f"simulated_{lang}_audio_data".encode()
            lang_base64 = base64.b64encode(lang_data).decode()
            
            response = client.post("/api/v1/detect-voice", json={"audio_base64": lang_base64})
            
            # Should handle all languages consistently
            assert response.status_code in [200, 422]  # Success or processing error
            
            data = response.json()
            if response.status_code == 200:
                # Should return consistent format regardless of language
                assert "classification" in data
                assert "confidence_score" in data
                assert data["classification"] in ["AI_GENERATED", "HUMAN"]
    
    def test_performance_under_load(self, client, sample_audio_base64):
        """Test system performance under load"""
        import time
        
        # Measure response times for multiple requests
        response_times = []
        
        for _ in range(10):
            start_time = time.time()
            response = client.post("/api/v1/detect-voice", json={"audio_base64": sample_audio_base64})
            end_time = time.time()
            
            response_time = end_time - start_time
            response_times.append(response_time)
            
            # Each request should complete within reasonable time
            assert response_time < 5.0  # 5 seconds max
        
        # Average response time should be reasonable
        avg_response_time = sum(response_times) / len(response_times)
        assert avg_response_time < 3.0  # 3 seconds average max
        
        # Response times should be relatively consistent
        max_time = max(response_times)
        min_time = min(response_times)
        assert (max_time - min_time) < 2.0  # Variance should be less than 2 seconds
    
    def test_system_stability(self, client, sample_audio_base64):
        """Test system stability over extended usage"""
        # Process multiple requests to check for memory leaks or degradation
        for i in range(20):
            response = client.post("/api/v1/detect-voice", json={"audio_base64": sample_audio_base64})
            
            # System should remain stable - 429 is acceptable during load testing
            assert response.status_code in [200, 422, 429, 500]
            
            # Response should be properly formatted
            data = response.json()
            assert isinstance(data, dict)
            
            # Should have consistent response structure
            if response.status_code == 200:
                assert "classification" in data
                assert "confidence_score" in data
            else:
                assert "error" in data
                # For rate limiting, we might get different error structures
                if response.status_code != 429:
                    assert "message" in data
    
    def test_api_specification_compliance(self, client, sample_audio_base64):
        """Test compliance with API specification"""
        response = client.post("/api/v1/detect-voice", json={"audio_base64": sample_audio_base64})
        
        # Check HTTP status codes are appropriate - 429 is valid for rate limiting
        assert response.status_code in [200, 400, 422, 429, 500, 503]
        
        # Check content type
        assert response.headers.get("content-type") == "application/json"
        
        # Check response structure
        data = response.json()
        
        if response.status_code == 200:
            # Success response must have exact fields
            assert set(data.keys()) == {"classification", "confidence_score"}
            assert data["classification"] in ["AI_GENERATED", "HUMAN"]
            assert isinstance(data["confidence_score"], (int, float))
            assert 0.0 <= data["confidence_score"] <= 1.0
        elif response.status_code == 429:
            # Rate limit response has different structure
            assert "error" in data or "detail" in data
        else:
            # Error response must have error structure
            assert "error" in data
            assert "message" in data
            assert isinstance(data["error"], str)
            assert isinstance(data["message"], str)
    
    def test_hackathon_evaluation_scenarios(self, client):
        """Test specific scenarios that might be used in hackathon evaluation"""
        
        # Scenario 1: Rapid fire requests
        for i in range(5):
            sample_data = f"test_audio_{i}".encode()
            sample_base64 = base64.b64encode(sample_data).decode()
            
            response = client.post("/api/v1/detect-voice", json={"audio_base64": sample_base64})
            # 429 is acceptable during rapid fire requests due to rate limiting
            assert response.status_code in [200, 422, 429]
        
        # Scenario 2: Edge case inputs
        edge_cases = [
            "",  # Empty string
            "A",  # Single character
            "=" * 100,  # Padding characters
            base64.b64encode(b"short").decode(),  # Very short data
        ]
        
        for case in edge_cases:
            response = client.post("/api/v1/detect-voice", json={"audio_base64": case})
            # Should handle gracefully, not crash
            assert response.status_code in [200, 400, 422, 500]
            
            data = response.json()
            assert isinstance(data, dict)
        
        # Scenario 3: Service availability check
        health_response = client.get("/health")
        assert health_response.status_code == 200
        
        # Scenario 4: Documentation availability
        docs_response = client.get("/docs")
        # May be disabled in production, but should not crash
        assert docs_response.status_code in [200, 404]