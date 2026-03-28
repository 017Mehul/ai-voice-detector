"""
Tests for Feature Extraction Module

Tests the FeatureExtractor class functionality including MFCC extraction,
spectral features, variance analysis, and temporal consistency.
"""

import pytest
import numpy as np
from ai_voice_detector.features import FeatureExtractor
from ai_voice_detector.models.data_models import AudioData, SpectralFeatures, TemporalFeatures, FeatureVector


class TestFeatureExtractor:
    """Test FeatureExtractor functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.extractor = FeatureExtractor()
    
    def create_test_audio(self, duration_seconds=3.0, sample_rate=22050, frequency=440):
        """Create test audio data"""
        t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds))
        # Create a simple sine wave with some variation
        samples = 0.5 * np.sin(2 * np.pi * frequency * t)
        # Add some noise for more realistic audio
        samples += 0.1 * np.random.randn(len(samples))
        
        return AudioData(
            samples=samples,
            sample_rate=sample_rate,
            duration=duration_seconds,
            channels=1
        )
    
    def test_extract_mfcc_features_valid_audio(self):
        """Test MFCC extraction with valid audio"""
        audio_data = self.create_test_audio(duration_seconds=3.0)
        
        mfcc_features = self.extractor.extract_mfcc_features(audio_data)
        
        # Verify MFCC output
        assert isinstance(mfcc_features, np.ndarray)
        assert mfcc_features.shape[0] == self.extractor.n_mfcc  # 13 coefficients
        assert mfcc_features.shape[1] > 0  # Should have frames
        
        # Check for valid values (no NaN or infinite)
        assert not np.any(np.isnan(mfcc_features))
        assert not np.any(np.isinf(mfcc_features))
    
    def test_extract_mfcc_features_invalid_audio(self):
        """Test MFCC extraction with invalid audio"""
        # Create invalid audio (too short)
        invalid_audio = AudioData(
            samples=np.array([0.1, 0.2]),
            sample_rate=22050,
            duration=0.5,  # Too short
            channels=1
        )
        
        with pytest.raises(ValueError) as exc_info:
            self.extractor.extract_mfcc_features(invalid_audio)
        
        assert "invalid audio" in str(exc_info.value).lower()
    
    def test_extract_spectral_features_valid_audio(self):
        """Test spectral feature extraction with valid audio"""
        audio_data = self.create_test_audio(duration_seconds=3.0)
        
        spectral_features = self.extractor.extract_spectral_features(audio_data)
        
        # Verify spectral features
        assert isinstance(spectral_features, SpectralFeatures)
        assert spectral_features.validate()
        
        # Check all features have same length
        lengths = [
            len(spectral_features.centroid),
            len(spectral_features.bandwidth),
            len(spectral_features.rolloff),
            len(spectral_features.zero_crossing_rate)
        ]
        assert len(set(lengths)) == 1  # All same length
        
        # Check for valid values
        for feature_array in [spectral_features.centroid, spectral_features.bandwidth, 
                             spectral_features.rolloff, spectral_features.zero_crossing_rate]:
            assert not np.any(np.isnan(feature_array))
            assert not np.any(np.isinf(feature_array))
            assert np.all(feature_array >= 0)  # Spectral features should be non-negative
    
    def test_analyze_pitch_variance_valid_audio(self):
        """Test pitch variance analysis with valid audio"""
        audio_data = self.create_test_audio(duration_seconds=3.0, frequency=440)
        
        pitch_variance = self.extractor.analyze_pitch_variance(audio_data)
        
        # Verify pitch variance
        assert isinstance(pitch_variance, float)
        assert pitch_variance >= 0.0  # Should be non-negative
        assert not np.isnan(pitch_variance)
        assert not np.isinf(pitch_variance)
    
    def test_analyze_energy_variance_valid_audio(self):
        """Test energy variance analysis with valid audio"""
        audio_data = self.create_test_audio(duration_seconds=3.0)
        
        energy_variance = self.extractor.analyze_energy_variance(audio_data)
        
        # Verify energy variance
        assert isinstance(energy_variance, float)
        assert energy_variance >= 0.0  # Should be non-negative
        assert not np.isnan(energy_variance)
        assert not np.isinf(energy_variance)
    
    def test_analyze_temporal_consistency_valid_audio(self):
        """Test temporal consistency analysis with valid audio"""
        audio_data = self.create_test_audio(duration_seconds=3.0)
        
        temporal_features = self.extractor.analyze_temporal_consistency(audio_data)
        
        # Verify temporal features
        assert isinstance(temporal_features, TemporalFeatures)
        assert temporal_features.validate()
        
        # Check individual components
        assert 0.0 <= temporal_features.frame_consistency <= 1.0
        assert 0.0 <= temporal_features.rhythm_regularity <= 1.0
        assert isinstance(temporal_features.pause_patterns, np.ndarray)
        assert len(temporal_features.pause_patterns) > 0
    
    def test_combine_features_valid_inputs(self):
        """Test feature combination with valid inputs"""
        audio_data = self.create_test_audio(duration_seconds=3.0)
        
        # Extract individual features
        mfcc_features = self.extractor.extract_mfcc_features(audio_data)
        spectral_features = self.extractor.extract_spectral_features(audio_data)
        pitch_variance = self.extractor.analyze_pitch_variance(audio_data)
        energy_variance = self.extractor.analyze_energy_variance(audio_data)
        temporal_features = self.extractor.analyze_temporal_consistency(audio_data)
        
        # Combine features
        feature_vector = self.extractor.combine_features(
            mfcc_features, spectral_features, pitch_variance, energy_variance, temporal_features
        )
        
        # Verify combined features
        assert isinstance(feature_vector, FeatureVector)
        assert feature_vector.validate()
        
        # Check feature vector can be converted to array
        feature_array = feature_vector.to_array()
        assert isinstance(feature_array, np.ndarray)
        assert len(feature_array) > 0
        assert not np.any(np.isnan(feature_array))
        assert not np.any(np.isinf(feature_array))
    
    def test_extract_all_features_complete_pipeline(self):
        """Test complete feature extraction pipeline"""
        audio_data = self.create_test_audio(duration_seconds=4.0)
        
        feature_vector = self.extractor.extract_all_features(audio_data)
        
        # Verify complete feature vector
        assert isinstance(feature_vector, FeatureVector)
        assert feature_vector.validate()
        
        # Check all components are present
        assert isinstance(feature_vector.mfcc_features, np.ndarray)
        assert isinstance(feature_vector.spectral_features, SpectralFeatures)
        assert isinstance(feature_vector.pitch_variance, float)
        assert isinstance(feature_vector.energy_variance, float)
        assert isinstance(feature_vector.temporal_features, TemporalFeatures)
        
        # Check feature info
        info = feature_vector.get_feature_info()
        assert info['total_features'] > 0
        assert info['mfcc_size'] > 0
    
    def test_get_feature_dimensions(self):
        """Test feature dimension information"""
        dimensions = self.extractor.get_feature_dimensions()
        
        assert isinstance(dimensions, dict)
        assert 'mfcc_coefficients' in dimensions
        assert 'spectral_features' in dimensions
        assert 'variance_features' in dimensions
        assert 'temporal_features' in dimensions
        assert 'processing_params' in dimensions
        
        # Check expected values
        assert dimensions['mfcc_coefficients'] == 13
        assert dimensions['spectral_features'] == 4
        assert dimensions['variance_features'] == 2
        assert dimensions['temporal_features'] == 3