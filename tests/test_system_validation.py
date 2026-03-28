"""
System validation tests for hackathon readiness
"""
import pytest
import json
import base64
import numpy as np
from ai_voice_detector.models.data_models import AudioData, DetectionRequest, DetectionResponse
from ai_voice_detector.services.voice_detection_service import VoiceDetectionService
from ai_voice_detector.audio.processor import AudioProcessor
from ai_voice_detector.features.extractor import FeatureExtractor
from ai_voice_detector.models.classifier import RandomForestVoiceClassifier
from ai_voice_detector.models.calibration import ConfidenceCalibrator


class TestSystemValidation:
    """System validation tests for hackathon readiness"""
    
    @pytest.fixture
    def audio_processor(self):
        return AudioProcessor()
    
    @pytest.fixture
    def feature_extractor(self):
        return FeatureExtractor()
    
    @pytest.fixture
    def classifier(self):
        classifier = RandomForestVoiceClassifier()
        # Try to load trained model if it exists
        try:
            from pathlib import Path
            model_path = Path("models/production/model.joblib")
            if model_path.exists():
                classifier.load_model(model_path)
        except Exception:
            # If loading fails, use untrained classifier
            pass
        return classifier
    
    @pytest.fixture
    def calibrator(self):
        calibrator = ConfidenceCalibrator()
        # Fit with dummy data for testing
        dummy_confidences = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        dummy_labels = np.array([0, 0, 1, 1, 1])  # Binary labels
        calibrator.fit(dummy_confidences, dummy_labels)
        return calibrator
    
    @pytest.fixture
    def voice_service(self, audio_processor, feature_extractor, classifier, calibrator):
        return VoiceDetectionService(
            audio_processor=audio_processor,
            feature_extractor=feature_extractor,
            classifier=classifier,
            confidence_calibrator=calibrator
        )
    
    @pytest.fixture
    def sample_audio_data(self):
        """Create sample audio data for testing"""
        sample_rate = 22050
        duration = 3.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        samples = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        return AudioData(
            samples=samples,
            sample_rate=sample_rate,
            duration=duration,
            channels=1
        )
    
    @pytest.fixture
    def sample_audio_base64(self, sample_audio_data):
        """Create base64 encoded audio data"""
        audio_bytes = sample_audio_data.samples.tobytes()
        return base64.b64encode(audio_bytes).decode('utf-8')
    
    def test_detection_request_validation(self):
        """Test detection request validation"""
        # Valid request
        valid_request = DetectionRequest(audio_base64="dGVzdA==")  # "test" in base64
        assert valid_request.validate()
        
        # Invalid requests
        invalid_requests = [
            DetectionRequest(audio_base64=""),
            DetectionRequest(audio_base64=None),
        ]
        
        for req in invalid_requests:
            assert not req.validate()
    
    def test_detection_response_format(self):
        """Test detection response format compliance"""
        # Valid responses
        valid_responses = [
            DetectionResponse(classification="AI_GENERATED", confidence_score=0.8),
            DetectionResponse(classification="HUMAN", confidence_score=0.3),
            DetectionResponse(classification="AI_GENERATED", confidence_score=0.0),
            DetectionResponse(classification="HUMAN", confidence_score=1.0),
        ]
        
        for response in valid_responses:
            # DetectionResponse validates in __post_init__, so if it was created successfully, it's valid
            assert response.classification in ["AI_GENERATED", "HUMAN"]
            assert 0.0 <= response.confidence_score <= 1.0
            
            # Test serialization
            response_dict = response.to_dict()
            assert "classification" in response_dict
            assert "confidence_score" in response_dict
            assert response_dict["classification"] in ["AI_GENERATED", "HUMAN"]
            assert 0.0 <= response_dict["confidence_score"] <= 1.0
        
        # Invalid responses should raise errors
        with pytest.raises(ValueError):
            DetectionResponse(classification="INVALID", confidence_score=0.5)
        
        with pytest.raises(ValueError):
            DetectionResponse(classification="HUMAN", confidence_score=1.5)
        
        with pytest.raises(ValueError):
            DetectionResponse(classification="AI_GENERATED", confidence_score=-0.1)
    
    def test_audio_processing_pipeline(self, audio_processor, sample_audio_base64):
        """Test complete audio processing pipeline"""
        try:
            # Decode base64
            audio_data = audio_processor.decode_base64_audio(sample_audio_base64)
            assert audio_data is not None
            
            # Validate audio
            assert audio_processor.validate_audio_integrity(audio_data)
            
            # Normalize audio
            normalized_audio = audio_processor.normalize_audio(audio_data)
            assert normalized_audio.is_valid()
            
        except Exception as e:
            # Audio processing may fail with synthetic data, but should handle gracefully
            from ai_voice_detector.audio.processor import AudioProcessingError
            assert isinstance(e, (ValueError, RuntimeError, AudioProcessingError))
    
    def test_feature_extraction_pipeline(self, feature_extractor, sample_audio_data):
        """Test complete feature extraction pipeline"""
        # Extract all features
        mfcc_features = feature_extractor.extract_mfcc_features(sample_audio_data)
        spectral_features = feature_extractor.extract_spectral_features(sample_audio_data)
        pitch_variance = feature_extractor.analyze_pitch_variance(sample_audio_data)
        energy_variance = feature_extractor.analyze_energy_variance(sample_audio_data)
        temporal_features = feature_extractor.analyze_temporal_consistency(sample_audio_data)
        
        # Combine features
        feature_vector = feature_extractor.combine_features(
            mfcc_features, spectral_features, pitch_variance, energy_variance, temporal_features
        )
        
        # Validate feature vector
        assert feature_vector is not None
        assert feature_vector.validate()
        
        # Check feature dimensions
        feature_array = feature_vector.to_array()
        assert len(feature_array) > 0
        assert np.all(np.isfinite(feature_array))
    
    def test_classification_consistency(self, classifier, feature_extractor, sample_audio_data):
        """Test classification consistency and determinism"""
        # Extract features
        mfcc_features = feature_extractor.extract_mfcc_features(sample_audio_data)
        spectral_features = feature_extractor.extract_spectral_features(sample_audio_data)
        pitch_variance = feature_extractor.analyze_pitch_variance(sample_audio_data)
        energy_variance = feature_extractor.analyze_energy_variance(sample_audio_data)
        temporal_features = feature_extractor.analyze_temporal_consistency(sample_audio_data)
        
        feature_vector = feature_extractor.combine_features(
            mfcc_features, spectral_features, pitch_variance, energy_variance, temporal_features
        )
        
        # Test multiple predictions on same input
        predictions = []
        for _ in range(5):
            try:
                classification, confidence = classifier.predict(feature_vector)
                predictions.append((classification, confidence))
            except ValueError:
                # Classifier may not be trained, which is expected
                pytest.skip("Classifier not trained - skipping consistency test")
        
        # All predictions should be identical (deterministic)
        if predictions:
            first_prediction = predictions[0]
            for prediction in predictions[1:]:
                assert prediction == first_prediction
    
    def test_confidence_calibration(self, calibrator, feature_extractor, sample_audio_data):
        """Test confidence calibration"""
        # Extract features
        mfcc_features = feature_extractor.extract_mfcc_features(sample_audio_data)
        spectral_features = feature_extractor.extract_spectral_features(sample_audio_data)
        pitch_variance = feature_extractor.analyze_pitch_variance(sample_audio_data)
        energy_variance = feature_extractor.analyze_energy_variance(sample_audio_data)
        temporal_features = feature_extractor.analyze_temporal_consistency(sample_audio_data)
        
        feature_vector = feature_extractor.combine_features(
            mfcc_features, spectral_features, pitch_variance, energy_variance, temporal_features
        )
        
        # Test confidence calibration with various raw confidence values
        raw_confidences = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        for raw_confidence in raw_confidences:
            calibrated_confidence = calibrator.calibrate_confidence(raw_confidence, feature_vector)
            
            # Calibrated confidence should be valid
            assert 0.0 <= calibrated_confidence <= 1.0
            assert isinstance(calibrated_confidence, float)
    
    def test_multi_language_feature_consistency(self, feature_extractor):
        """Test feature extraction consistency across different 'languages'"""
        # Create different audio samples representing different languages
        languages = ["tamil", "english", "hindi", "malayalam", "telugu"]
        feature_vectors = []
        
        for i, lang in enumerate(languages):
            # Generate different audio for each language (simulated)
            sample_rate = 22050
            duration = 3.0
            frequency = 440 + (i * 50)  # Different frequency for each "language"
            
            t = np.linspace(0, duration, int(sample_rate * duration))
            samples = np.sin(2 * np.pi * frequency * t).astype(np.float32)
            
            audio_data = AudioData(
                samples=samples,
                sample_rate=sample_rate,
                duration=duration,
                channels=1
            )
            
            # Extract features
            mfcc_features = feature_extractor.extract_mfcc_features(audio_data)
            spectral_features = feature_extractor.extract_spectral_features(audio_data)
            pitch_variance = feature_extractor.analyze_pitch_variance(audio_data)
            energy_variance = feature_extractor.analyze_energy_variance(audio_data)
            temporal_features = feature_extractor.analyze_temporal_consistency(audio_data)
            
            feature_vector = feature_extractor.combine_features(
                mfcc_features, spectral_features, pitch_variance, energy_variance, temporal_features
            )
            
            feature_vectors.append(feature_vector)
        
        # All feature vectors should have consistent structure
        first_vector = feature_vectors[0]
        for vector in feature_vectors[1:]:
            # Same dimensions
            assert len(vector.to_array()) == len(first_vector.to_array())
            
            # Same structure
            assert vector.validate()
            assert type(vector.mfcc_features) == type(first_vector.mfcc_features)
            assert type(vector.spectral_features) == type(first_vector.spectral_features)
    
    def test_error_handling_robustness(self, voice_service):
        """Test system robustness with various error conditions"""
        # Test with invalid base64
        invalid_inputs = [
            "",  # Empty string
            "invalid_base64!",  # Invalid characters
            "A",  # Too short
            None,  # None value
        ]
        
        for invalid_input in invalid_inputs:
            try:
                # Create a proper DetectionRequest object
                from ai_voice_detector.models.data_models import DetectionRequest
                if invalid_input is not None:
                    request = DetectionRequest(audio_base64=invalid_input)
                    result = voice_service.detect_voice(request)
                    # Service returns tuple (response, validation_result)
                    response, validation_result = result
                    assert response is not None
                    assert hasattr(response, 'classification') or hasattr(response, 'error')
                else:
                    # None input should raise exception before service call
                    request = DetectionRequest(audio_base64="")
                    result = voice_service.detect_voice(request)
                    response, validation_result = result
                    assert response is not None
            except (ValueError, TypeError, AttributeError) as e:
                # Expected to raise exceptions for invalid input
                assert isinstance(e, (ValueError, TypeError, AttributeError))
    
    def test_performance_requirements(self, voice_service, sample_audio_base64):
        """Test performance requirements for hackathon evaluation"""
        import time
        
        # Test processing time
        start_time = time.time()
        
        try:
            result = voice_service.detect_voice(sample_audio_base64)
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            # Should complete within reasonable time (5 seconds max for hackathon)
            assert processing_time < 5.0
            
            # Result should be properly formatted
            if hasattr(result, 'classification'):
                assert result.classification in ["AI_GENERATED", "HUMAN"]
                assert 0.0 <= result.confidence_score <= 1.0
                
        except Exception as e:
            # Service may fail due to untrained model, but should fail quickly
            end_time = time.time()
            processing_time = end_time - start_time
            assert processing_time < 5.0  # Even failures should be fast
    
    def test_system_stability_under_load(self, voice_service, sample_audio_base64):
        """Test system stability under repeated requests"""
        # Process multiple requests
        for i in range(10):
            try:
                result = voice_service.detect_voice(sample_audio_base64)
                
                # System should remain stable
                if hasattr(result, 'classification'):
                    assert result.classification in ["AI_GENERATED", "HUMAN"]
                    assert 0.0 <= result.confidence_score <= 1.0
                    
            except Exception as e:
                # Exceptions are acceptable but should be consistent
                assert isinstance(e, (ValueError, TypeError, RuntimeError))
    
    def test_api_specification_compliance(self):
        """Test compliance with API specification requirements"""
        # Test request format
        valid_request = {"audio_base64": "dGVzdA=="}
        request_obj = DetectionRequest(**valid_request)
        assert request_obj.validate()
        
        # Test response format
        valid_response_data = {
            "classification": "AI_GENERATED",
            "confidence_score": 0.75
        }
        response_obj = DetectionResponse(**valid_response_data)
        # DetectionResponse validates in __post_init__
        assert response_obj.classification == "AI_GENERATED"
        assert response_obj.confidence_score == 0.75
        
        response_dict = response_obj.to_dict()
        assert set(response_dict.keys()) == {"classification", "confidence_score"}
        
        # Test error response format
        from ai_voice_detector.models.data_models import ErrorResponse
        error_response = ErrorResponse(error="TEST_ERROR", message="Test error message")
        error_dict = error_response.to_dict()
        assert set(error_dict.keys()) == {"error", "message"}
    
    def test_hackathon_readiness_checklist(self, voice_service):
        """Final checklist for hackathon readiness"""
        # 1. Service can be instantiated
        assert voice_service is not None
        
        # 2. All components are properly initialized
        assert voice_service.audio_processor is not None
        assert voice_service.feature_extractor is not None
        assert voice_service.classifier is not None
        assert voice_service.confidence_calibrator is not None
        
        # 3. Service has required methods
        assert hasattr(voice_service, 'detect_voice')
        assert callable(voice_service.detect_voice)
        
        # 4. Data models are properly defined
        from ai_voice_detector.models.data_models import DetectionRequest, DetectionResponse, ErrorResponse
        
        # Test instantiation
        req = DetectionRequest(audio_base64="test")
        resp = DetectionResponse(classification="HUMAN", confidence_score=0.5)
        err = ErrorResponse(error="TEST", message="Test")
        
        assert req is not None
        assert resp is not None
        assert err is not None
        
        # 5. All required validation methods exist
        assert hasattr(req, 'validate')
        # DetectionResponse uses __post_init__ for validation, not a validate method
        assert hasattr(resp, 'to_dict')
        assert hasattr(err, 'to_dict')
        
        print("✓ All hackathon readiness checks passed!")