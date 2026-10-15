"""
Property-Based Tests for Spectral Feature Extraction

Tests Property 15: Spectral Feature Extraction
Validates: Requirements 6.2
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, assume, settings, HealthCheck
from ai_voice_detector.features import FeatureExtractor
from ai_voice_detector.models.data_models import AudioData, SpectralFeatures


# Test data generators
@st.composite
def valid_audio_data(draw):
    """Generate valid AudioData for testing"""
    # Generate reasonable parameters with smaller sizes
    duration = draw(st.floats(min_value=2.0, max_value=4.0))  # Reduced max duration
    sample_rate = draw(st.sampled_from([16000, 22050]))  # Removed 44100 to reduce size
    
    # Generate audio samples
    num_samples = int(duration * sample_rate)
    
    # Create realistic audio signal (sine wave + noise)
    t = np.linspace(0, duration, num_samples)
    frequency = draw(st.floats(min_value=100, max_value=1000))  # Reduced max frequency
    amplitude = draw(st.floats(min_value=0.1, max_value=0.8))
    
    # Base signal
    samples = amplitude * np.sin(2 * np.pi * frequency * t)
    
    # Add some noise for realism
    noise_level = draw(st.floats(min_value=0.01, max_value=0.05))  # Reduced noise
    noise = noise_level * np.random.randn(num_samples)
    samples = samples + noise
    
    # Ensure samples are in valid range
    samples = np.clip(samples, -1.0, 1.0)
    
    return AudioData(
        samples=samples,
        sample_rate=sample_rate,
        duration=duration,
        channels=1
    )


class TestSpectralFeatureProperties:
    """Property-based tests for spectral feature extraction"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.extractor = FeatureExtractor()
    
    @given(audio_data=valid_audio_data())
    @settings(max_examples=8, deadline=3000, suppress_health_check=[HealthCheck.data_too_large])
    def test_spectral_feature_extraction_property(self, audio_data):
        """
        **Feature: ai-voice-detector, Property 15: Spectral Feature Extraction**
        
        For any audio sample, the Feature_Extractor should extract spectral features 
        (centroid, bandwidth) with non-negative values and reasonable magnitudes.
        
        **Validates: Requirements 6.2**
        """
        # Ensure audio is valid
        assume(audio_data.is_valid())
        assume(len(audio_data.samples) > 1024)  # Minimum samples for spectral analysis
        
        # Extract spectral features
        spectral_features = self.extractor.extract_spectral_features(audio_data)
        
        # Property 1: Output should be SpectralFeatures object
        assert isinstance(spectral_features, SpectralFeatures), "Output must be SpectralFeatures object"
        
        # Property 2: SpectralFeatures should validate successfully
        assert spectral_features.validate(), "SpectralFeatures should pass validation"
        
        # Property 3: All feature arrays should be non-negative
        assert np.all(spectral_features.centroid >= 0), "Spectral centroid must be non-negative"
        assert np.all(spectral_features.bandwidth >= 0), "Spectral bandwidth must be non-negative"
        assert np.all(spectral_features.rolloff >= 0), "Spectral rolloff must be non-negative"
        assert np.all(spectral_features.zero_crossing_rate >= 0), "Zero crossing rate must be non-negative"
        
        # Property 4: All feature arrays should be finite
        assert np.all(np.isfinite(spectral_features.centroid)), "Spectral centroid must be finite"
        assert np.all(np.isfinite(spectral_features.bandwidth)), "Spectral bandwidth must be finite"
        assert np.all(np.isfinite(spectral_features.rolloff)), "Spectral rolloff must be finite"
        assert np.all(np.isfinite(spectral_features.zero_crossing_rate)), "Zero crossing rate must be finite"
        
        # Property 5: All feature arrays should have same length
        lengths = [
            len(spectral_features.centroid),
            len(spectral_features.bandwidth),
            len(spectral_features.rolloff),
            len(spectral_features.zero_crossing_rate)
        ]
        assert len(set(lengths)) == 1, "All spectral feature arrays must have same length"
        
        # Property 6: Feature arrays should have at least one frame
        assert len(spectral_features.centroid) > 0, "Spectral features should have at least one frame"
        
        # Property 7: Spectral centroid should be within reasonable frequency range
        # Should be between 0 and Nyquist frequency
        nyquist_freq = audio_data.sample_rate / 2
        assert np.all(spectral_features.centroid <= nyquist_freq), \
            f"Spectral centroid should be <= Nyquist frequency ({nyquist_freq} Hz)"
        
        # Property 8: Zero crossing rate should be between 0 and 1
        assert np.all(spectral_features.zero_crossing_rate <= 1.0), \
            "Zero crossing rate should be <= 1.0"
        
        # Property 9: Spectral rolloff is a frequency and must stay within
        # the representable range. It is not mathematically guaranteed to be
        # close to the centroid for every valid spectrum, so do not assert a
        # fixed relationship between their means.
        assert np.all(spectral_features.rolloff <= nyquist_freq), \
            f"Spectral rolloff should be <= Nyquist frequency ({nyquist_freq} Hz)"
    
    @given(audio_data=valid_audio_data())
    @settings(max_examples=8, deadline=2000, suppress_health_check=[HealthCheck.data_too_large])
    def test_spectral_features_determinism_property(self, audio_data):
        """
        Property: Spectral feature extraction should be deterministic for same input
        """
        assume(audio_data.is_valid())
        assume(len(audio_data.samples) > 1024)
        
        # Extract features twice
        features_1 = self.extractor.extract_spectral_features(audio_data)
        features_2 = self.extractor.extract_spectral_features(audio_data)
        
        # Should be identical
        np.testing.assert_array_almost_equal(features_1.centroid, features_2.centroid, decimal=10)
        np.testing.assert_array_almost_equal(features_1.bandwidth, features_2.bandwidth, decimal=10)
        np.testing.assert_array_almost_equal(features_1.rolloff, features_2.rolloff, decimal=10)
        np.testing.assert_array_almost_equal(features_1.zero_crossing_rate, features_2.zero_crossing_rate, decimal=10)
    
    @given(
        frequency=st.floats(min_value=200, max_value=1000),
        sample_rate=st.sampled_from([16000, 22050, 44100])
    )
    @settings(max_examples=5, deadline=2000)
    def test_spectral_centroid_frequency_relationship(self, frequency, sample_rate):
        """
        Property: Spectral centroid should be related to the dominant frequency in pure tones
        """
        # Create pure tone
        duration = 3.0
        num_samples = int(duration * sample_rate)
        t = np.linspace(0, duration, num_samples)
        samples = 0.5 * np.sin(2 * np.pi * frequency * t)
        
        audio_data = AudioData(
            samples=samples,
            sample_rate=sample_rate,
            duration=duration,
            channels=1
        )
        
        # Extract spectral features
        spectral_features = self.extractor.extract_spectral_features(audio_data)
        
        # For a pure tone, spectral centroid should be close to the fundamental frequency
        mean_centroid = np.mean(spectral_features.centroid)
        
        # Allow reasonable tolerance (pure tones may have harmonics affecting centroid)
        frequency_tolerance = frequency * 0.5  # 50% tolerance
        assert abs(mean_centroid - frequency) <= frequency_tolerance, \
            f"Spectral centroid {mean_centroid:.1f} should be close to frequency {frequency:.1f}"
    
    @given(audio_data=valid_audio_data())
    @settings(max_examples=5, deadline=2000)
    def test_spectral_features_statistics_property(self, audio_data):
        """
        Property: Spectral features statistics should be reasonable
        """
        assume(audio_data.is_valid())
        assume(len(audio_data.samples) > 1024)
        
        # Extract spectral features
        spectral_features = self.extractor.extract_spectral_features(audio_data)
        
        # Get statistics
        stats = spectral_features.get_statistics()
        
        # Property: Statistics should be valid
        for feature_name, feature_stats in stats.items():
            assert feature_stats['mean'] >= 0, f"{feature_name} mean should be non-negative"
            assert feature_stats['std'] >= 0, f"{feature_name} std should be non-negative"
            assert feature_stats['min'] >= 0, f"{feature_name} min should be non-negative"
            assert feature_stats['max'] >= feature_stats['min'], f"{feature_name} max should be >= min"
            
            # All statistics should be finite
            for stat_name, stat_value in feature_stats.items():
                assert np.isfinite(stat_value), f"{feature_name}.{stat_name} should be finite"