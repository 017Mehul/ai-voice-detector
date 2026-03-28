"""
Property-Based Tests for Temporal Analysis

Tests Property 17: Temporal Analysis
Validates: Requirements 6.4
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, assume, settings, HealthCheck
from ai_voice_detector.features import FeatureExtractor
from ai_voice_detector.models.data_models import AudioData, TemporalFeatures


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


class TestTemporalAnalysisProperties:
    """Property-based tests for temporal analysis"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.extractor = FeatureExtractor()
    
    @given(audio_data=valid_audio_data())
    @settings(max_examples=8, deadline=3000, suppress_health_check=[HealthCheck.data_too_large])
    def test_temporal_analysis_property(self, audio_data):
        """
        **Feature: ai-voice-detector, Property 17: Temporal Analysis**
        
        For any audio input, the Feature_Extractor should analyze frame-to-frame 
        variations and produce temporal consistency features.
        
        **Validates: Requirements 6.4**
        """
        # Ensure audio is valid
        assume(audio_data.is_valid())
        assume(len(audio_data.samples) > 1024)  # Minimum samples for analysis
        
        # Analyze temporal consistency
        temporal_features = self.extractor.analyze_temporal_consistency(audio_data)
        
        # Property 1: Output should be TemporalFeatures object
        assert isinstance(temporal_features, TemporalFeatures), "Output must be TemporalFeatures object"
        
        # Property 2: TemporalFeatures should validate successfully
        assert temporal_features.validate(), "TemporalFeatures should pass validation"
        
        # Property 3: Frame consistency should be between 0 and 1
        assert 0.0 <= temporal_features.frame_consistency <= 1.0, \
            f"Frame consistency must be in [0,1], got {temporal_features.frame_consistency}"
        
        # Property 4: Rhythm regularity should be between 0 and 1
        assert 0.0 <= temporal_features.rhythm_regularity <= 1.0, \
            f"Rhythm regularity must be in [0,1], got {temporal_features.rhythm_regularity}"
        
        # Property 5: Pause patterns should be numpy array with at least one element
        assert isinstance(temporal_features.pause_patterns, np.ndarray), \
            "Pause patterns must be numpy array"
        assert len(temporal_features.pause_patterns) > 0, \
            "Pause patterns should have at least one element"
        
        # Property 6: All pause pattern values should be non-negative (durations)
        assert np.all(temporal_features.pause_patterns >= 0), \
            "Pause pattern durations must be non-negative"
        
        # Property 7: All values should be finite
        assert np.isfinite(temporal_features.frame_consistency), \
            "Frame consistency must be finite"
        assert np.isfinite(temporal_features.rhythm_regularity), \
            "Rhythm regularity must be finite"
        assert np.all(np.isfinite(temporal_features.pause_patterns)), \
            "Pause patterns must be finite"
        
        # Property 8: Pause patterns should be reasonable (not longer than audio duration)
        max_pause = np.max(temporal_features.pause_patterns)
        assert max_pause <= audio_data.duration, \
            f"Maximum pause {max_pause:.2f}s should not exceed audio duration {audio_data.duration:.2f}s"
    
    @given(audio_data=valid_audio_data())
    @settings(max_examples=5, deadline=2000, suppress_health_check=[HealthCheck.data_too_large])
    def test_temporal_analysis_determinism_property(self, audio_data):
        """
        Property: Temporal analysis should be deterministic for same input
        """
        assume(audio_data.is_valid())
        assume(len(audio_data.samples) > 1024)
        
        # Analyze temporal features twice
        features_1 = self.extractor.analyze_temporal_consistency(audio_data)
        features_2 = self.extractor.analyze_temporal_consistency(audio_data)
        
        # Should be identical
        assert abs(features_1.frame_consistency - features_2.frame_consistency) < 1e-10, \
            "Frame consistency should be deterministic"
        assert abs(features_1.rhythm_regularity - features_2.rhythm_regularity) < 1e-10, \
            "Rhythm regularity should be deterministic"
        np.testing.assert_array_almost_equal(features_1.pause_patterns, features_2.pause_patterns, 
                                           decimal=10, err_msg="Pause patterns should be deterministic")
    
    @given(
        frequency=st.floats(min_value=200, max_value=800),
        sample_rate=st.sampled_from([16000, 22050])
    )
    @settings(max_examples=5, deadline=2000)
    def test_pure_tone_high_consistency_property(self, frequency, sample_rate):
        """
        Property: Pure tones should have high frame consistency
        """
        duration = 4.0
        num_samples = int(duration * sample_rate)
        t = np.linspace(0, duration, num_samples)
        
        # Create pure tone (should have high consistency)
        samples = 0.5 * np.sin(2 * np.pi * frequency * t)
        
        audio_data = AudioData(
            samples=samples,
            sample_rate=sample_rate,
            duration=duration,
            channels=1
        )
        
        # Analyze temporal features
        temporal_features = self.extractor.analyze_temporal_consistency(audio_data)
        
        # Pure tone should have relatively high frame consistency
        assert temporal_features.frame_consistency >= 0.3, \
            f"Pure tone should have reasonable frame consistency, got {temporal_features.frame_consistency:.3f}"
    
    @given(audio_data=valid_audio_data())
    @settings(max_examples=5, deadline=2000)
    def test_temporal_features_summary_property(self, audio_data):
        """
        Property: Temporal features summary should be valid
        """
        assume(audio_data.is_valid())
        assume(len(audio_data.samples) > 1024)
        
        # Analyze temporal features
        temporal_features = self.extractor.analyze_temporal_consistency(audio_data)
        
        # Get summary
        summary = temporal_features.get_summary()
        
        # Property: Summary should contain expected keys
        expected_keys = ['frame_consistency', 'rhythm_regularity', 'pause_patterns_count', 'pause_patterns_mean']
        for key in expected_keys:
            assert key in summary, f"Summary should contain key: {key}"
        
        # Property: Summary values should be valid
        assert 0.0 <= summary['frame_consistency'] <= 1.0, "Summary frame consistency should be in [0,1]"
        assert 0.0 <= summary['rhythm_regularity'] <= 1.0, "Summary rhythm regularity should be in [0,1]"
        assert summary['pause_patterns_count'] > 0, "Summary should have positive pause pattern count"
        assert summary['pause_patterns_mean'] >= 0.0, "Summary pause patterns mean should be non-negative"
        
        # All summary values should be finite
        for key, value in summary.items():
            assert np.isfinite(value), f"Summary {key} should be finite"
    
    @given(
        sample_rate=st.sampled_from([16000, 22050])
    )
    @settings(max_examples=5, deadline=1500)
    def test_silent_audio_temporal_properties(self, sample_rate):
        """
        Property: Silent audio should have specific temporal characteristics
        """
        duration = 3.0
        num_samples = int(duration * sample_rate)
        
        # Create silent audio
        samples = np.zeros(num_samples)
        audio_data = AudioData(
            samples=samples,
            sample_rate=sample_rate,
            duration=duration,
            channels=1
        )
        
        # Analyze temporal features
        temporal_features = self.extractor.analyze_temporal_consistency(audio_data)
        
        # Silent audio should have specific characteristics
        # Frame consistency might be high (no variation) or default value
        assert 0.0 <= temporal_features.frame_consistency <= 1.0, \
            "Silent audio frame consistency should be in valid range"
        
        # Rhythm regularity should be in valid range (might be default)
        assert 0.0 <= temporal_features.rhythm_regularity <= 1.0, \
            "Silent audio rhythm regularity should be in valid range"
        
        # Should have pause patterns (likely one long pause)
        assert len(temporal_features.pause_patterns) > 0, \
            "Silent audio should have pause patterns"
    
    @given(audio_data=valid_audio_data())
    @settings(max_examples=5, deadline=1500)
    def test_temporal_features_bounds_property(self, audio_data):
        """
        Property: All temporal feature components should be within expected bounds
        """
        assume(audio_data.is_valid())
        assume(len(audio_data.samples) > 1024)
        
        # Analyze temporal features
        temporal_features = self.extractor.analyze_temporal_consistency(audio_data)
        
        # Strict bounds checking
        assert isinstance(temporal_features.frame_consistency, (int, float)), \
            "Frame consistency should be numeric"
        assert isinstance(temporal_features.rhythm_regularity, (int, float)), \
            "Rhythm regularity should be numeric"
        
        # Values should not be extreme
        assert temporal_features.frame_consistency <= 1.0, "Frame consistency should not exceed 1.0"
        assert temporal_features.rhythm_regularity <= 1.0, "Rhythm regularity should not exceed 1.0"
        
        # Pause patterns should be reasonable
        if len(temporal_features.pause_patterns) > 1:
            total_pause_time = np.sum(temporal_features.pause_patterns)
            assert total_pause_time <= audio_data.duration * 2, \
                "Total pause time should not be excessive compared to audio duration"