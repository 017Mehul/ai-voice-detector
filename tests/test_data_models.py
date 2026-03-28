"""
Tests for data model classes

Tests the core data models for requests, responses, and internal data structures.
Focuses on validation methods and serialization support.
"""

import pytest
import numpy as np
import json
import base64
from ai_voice_detector.models.data_models import (
    DetectionRequest,
    DetectionResponse,
    ErrorResponse,
    AudioData,
    SpectralFeatures,
    TemporalFeatures,
    FeatureVector,
    ValidationResult
)


class TestDetectionRequest:
    """Test DetectionRequest data model"""
    
    def test_valid_request_creation(self):
        """Test creating valid DetectionRequest"""
        valid_b64 = base64.b64encode(b"test audio data").decode('utf-8')
        request = DetectionRequest(audio_base64=valid_b64)
        
        assert request.audio_base64 == valid_b64
        assert request.validate() is True
    
    def test_invalid_request_validation(self):
        """Test validation of invalid requests"""
        # Empty string
        request = DetectionRequest(audio_base64="")
        assert request.validate() is False
        
        # Invalid base64
        request = DetectionRequest(audio_base64="invalid!@#")
        assert request.validate() is False
        
        # None value
        request = DetectionRequest(audio_base64=None)
        assert request.validate() is False
    
    def test_from_dict_creation(self):
        """Test creating request from dictionary"""
        valid_b64 = base64.b64encode(b"test").decode('utf-8')
        data = {"audio_base64": valid_b64}
        request = DetectionRequest.from_dict(data)
        
        assert request.audio_base64 == valid_b64
        assert request.validate() is True
    
    def test_to_dict_serialization(self):
        """Test converting request to dictionary"""
        valid_b64 = base64.b64encode(b"test").decode('utf-8')
        request = DetectionRequest(audio_base64=valid_b64)
        result = request.to_dict()
        
        assert result == {"audio_base64": valid_b64}


class TestDetectionResponse:
    """Test DetectionResponse data model"""
    
    def test_valid_response_creation(self):
        """Test creating valid DetectionResponse"""
        response = DetectionResponse(
            classification="AI_GENERATED",
            confidence_score=0.85
        )
        
        assert response.classification == "AI_GENERATED"
        assert response.confidence_score == 0.85
    
    def test_invalid_classification_raises_error(self):
        """Test that invalid classification raises ValueError"""
        with pytest.raises(ValueError, match="Invalid classification"):
            DetectionResponse(
                classification="INVALID",
                confidence_score=0.5
            )
    
    def test_invalid_confidence_score_raises_error(self):
        """Test that invalid confidence scores raise ValueError"""
        # Out of range
        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            DetectionResponse(
                classification="HUMAN",
                confidence_score=1.5
            )
        
        # Negative
        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            DetectionResponse(
                classification="HUMAN",
                confidence_score=-0.1
            )
    
    def test_response_serialization(self):
        """Test response serialization methods"""
        response = DetectionResponse(
            classification="HUMAN",
            confidence_score=0.75
        )
        
        # Test to_dict
        result_dict = response.to_dict()
        expected = {
            "classification": "HUMAN",
            "confidence_score": 0.75
        }
        assert result_dict == expected
        
        # Test to_json
        result_json = response.to_json()
        parsed = json.loads(result_json)
        assert parsed == expected


class TestErrorResponse:
    """Test ErrorResponse data model"""
    
    def test_error_response_creation(self):
        """Test creating ErrorResponse"""
        error = ErrorResponse(
            error="INVALID_INPUT",
            message="The provided input is invalid"
        )
        
        assert error.error == "INVALID_INPUT"
        assert error.message == "The provided input is invalid"
    
    def test_error_response_serialization(self):
        """Test error response serialization"""
        error = ErrorResponse(
            error="TEST_ERROR",
            message="Test message"
        )
        
        # Test to_dict
        result_dict = error.to_dict()
        expected = {
            "error": "TEST_ERROR",
            "message": "Test message"
        }
        assert result_dict == expected
        
        # Test to_json
        result_json = error.to_json()
        parsed = json.loads(result_json)
        assert parsed == expected


class TestAudioData:
    """Test AudioData data model"""
    
    def test_valid_audio_data(self):
        """Test creating valid AudioData"""
        samples = np.random.randn(44100 * 3)  # 3 seconds at 44.1kHz
        audio = AudioData(
            samples=samples,
            sample_rate=44100,
            duration=3.0,
            channels=1
        )
        
        assert audio.is_valid() is True
        assert len(audio.samples) == len(samples)
    
    def test_invalid_audio_data(self):
        """Test validation of invalid audio data"""
        # Too short duration
        samples = np.random.randn(1000)
        audio = AudioData(
            samples=samples,
            sample_rate=44100,
            duration=1.0,  # Less than 2 seconds
            channels=1
        )
        assert audio.is_valid() is False
        
        # Empty samples
        audio = AudioData(
            samples=np.array([]),
            sample_rate=44100,
            duration=3.0,
            channels=1
        )
        assert audio.is_valid() is False
    
    def test_audio_info(self):
        """Test getting audio information"""
        samples = np.random.randn(88200)  # 2 seconds at 44.1kHz
        audio = AudioData(
            samples=samples,
            sample_rate=44100,
            duration=2.0,
            channels=1
        )
        
        info = audio.get_info()
        expected = {
            "sample_rate": 44100,
            "duration": 2.0,
            "channels": 1,
            "samples_count": 88200
        }
        assert info == expected


class TestSpectralFeatures:
    """Test SpectralFeatures data model"""
    
    def test_valid_spectral_features(self):
        """Test creating valid SpectralFeatures"""
        length = 100
        features = SpectralFeatures(
            centroid=np.random.randn(length),
            bandwidth=np.random.randn(length),
            rolloff=np.random.randn(length),
            zero_crossing_rate=np.random.randn(length)
        )
        
        assert features.validate() is True
    
    def test_invalid_spectral_features(self):
        """Test validation of invalid spectral features"""
        # Mismatched lengths
        features = SpectralFeatures(
            centroid=np.random.randn(100),
            bandwidth=np.random.randn(50),  # Different length
            rolloff=np.random.randn(100),
            zero_crossing_rate=np.random.randn(100)
        )
        assert features.validate() is False
        
        # Empty arrays
        features = SpectralFeatures(
            centroid=np.array([]),
            bandwidth=np.array([]),
            rolloff=np.array([]),
            zero_crossing_rate=np.array([])
        )
        assert features.validate() is False
    
    def test_spectral_statistics(self):
        """Test getting spectral feature statistics"""
        features = SpectralFeatures(
            centroid=np.array([1.0, 2.0, 3.0]),
            bandwidth=np.array([0.5, 1.0, 1.5]),
            rolloff=np.array([2.0, 3.0, 4.0]),
            zero_crossing_rate=np.array([0.1, 0.2, 0.3])
        )
        
        stats = features.get_statistics()
        
        # Check structure
        assert "centroid" in stats
        assert "bandwidth" in stats
        assert "rolloff" in stats
        assert "zero_crossing_rate" in stats
        
        # Check centroid stats
        assert stats["centroid"]["mean"] == 2.0
        assert stats["centroid"]["min"] == 1.0
        assert stats["centroid"]["max"] == 3.0


class TestTemporalFeatures:
    """Test TemporalFeatures data model"""
    
    def test_valid_temporal_features(self):
        """Test creating valid TemporalFeatures"""
        features = TemporalFeatures(
            frame_consistency=0.8,
            rhythm_regularity=0.6,
            pause_patterns=np.array([0.1, 0.2, 0.3])
        )
        
        assert features.validate() is True
    
    def test_invalid_temporal_features(self):
        """Test validation of invalid temporal features"""
        # Out of range values
        features = TemporalFeatures(
            frame_consistency=1.5,  # > 1.0
            rhythm_regularity=0.6,
            pause_patterns=np.array([0.1, 0.2])
        )
        assert features.validate() is False
        
        # Empty pause patterns
        features = TemporalFeatures(
            frame_consistency=0.8,
            rhythm_regularity=0.6,
            pause_patterns=np.array([])
        )
        assert features.validate() is False
    
    def test_temporal_summary(self):
        """Test getting temporal feature summary"""
        pause_patterns = np.array([0.1, 0.2, 0.3])
        features = TemporalFeatures(
            frame_consistency=0.8,
            rhythm_regularity=0.6,
            pause_patterns=pause_patterns
        )
        
        summary = features.get_summary()
        
        assert summary["frame_consistency"] == 0.8
        assert summary["rhythm_regularity"] == 0.6
        assert summary["pause_patterns_count"] == 3
        assert abs(summary["pause_patterns_mean"] - 0.2) < 1e-10


class TestFeatureVector:
    """Test FeatureVector data model"""
    
    def create_valid_feature_vector(self):
        """Helper to create valid FeatureVector"""
        mfcc = np.random.randn(13, 100)  # 13 MFCC coefficients, 100 frames
        
        spectral = SpectralFeatures(
            centroid=np.random.randn(100),
            bandwidth=np.random.randn(100),
            rolloff=np.random.randn(100),
            zero_crossing_rate=np.random.randn(100)
        )
        
        temporal = TemporalFeatures(
            frame_consistency=0.8,
            rhythm_regularity=0.6,
            pause_patterns=np.array([0.1, 0.2, 0.3])
        )
        
        return FeatureVector(
            mfcc_features=mfcc,
            spectral_features=spectral,
            pitch_variance=0.5,
            energy_variance=0.3,
            temporal_features=temporal
        )
    
    def test_valid_feature_vector(self):
        """Test creating valid FeatureVector"""
        features = self.create_valid_feature_vector()
        assert features.validate() is True
    
    def test_invalid_feature_vector(self):
        """Test validation of invalid FeatureVector"""
        features = self.create_valid_feature_vector()
        
        # Negative variance
        features.pitch_variance = -0.1
        assert features.validate() is False
    
    def test_feature_vector_to_array(self):
        """Test converting FeatureVector to array"""
        features = self.create_valid_feature_vector()
        array = features.to_array()
        
        assert isinstance(array, np.ndarray)
        assert len(array) > 0
        
        # Should contain flattened MFCC + spectral stats + variances + temporal
        expected_size = (
            features.mfcc_features.size +  # Flattened MFCC
            8 +  # Spectral stats (mean, std for 4 features)
            2 +  # Pitch and energy variance
            3    # Temporal features (consistency, regularity, pause mean)
        )
        assert len(array) == expected_size
    
    def test_feature_info(self):
        """Test getting feature vector information"""
        features = self.create_valid_feature_vector()
        info = features.get_feature_info()
        
        assert "mfcc_shape" in info
        assert "mfcc_size" in info
        assert "spectral_valid" in info
        assert "temporal_valid" in info
        assert "total_features" in info
        
        assert info["spectral_valid"] is True
        assert info["temporal_valid"] is True


class TestValidationResult:
    """Test ValidationResult helper class"""
    
    def test_valid_result(self):
        """Test creating valid ValidationResult"""
        result = ValidationResult(is_valid=True)
        assert result.is_valid is True
        assert result.error_message is None
        assert result.to_error_response() is None
    
    def test_invalid_result_with_error(self):
        """Test creating invalid ValidationResult with error"""
        result = ValidationResult(
            is_valid=False,
            error_message="Invalid input",
            error_code="INVALID_INPUT"
        )
        
        assert result.is_valid is False
        
        error_response = result.to_error_response()
        assert error_response is not None
        assert error_response.error == "INVALID_INPUT"
        assert error_response.message == "Invalid input"