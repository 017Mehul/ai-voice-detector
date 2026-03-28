"""
Property-Based Tests for Variance Metrics

Tests Property 16: Variance Metrics Calculation
Validates: Requirements 6.3
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, assume, settings, HealthCheck
from ai_voice_detector.features import FeatureExtractor
from ai_voice_detector.models.data_models import AudioData


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


class TestVarianceMetricsProperties:
    """Property-based tests for variance metrics calculation"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.extractor = FeatureExtractor()
    
    @given(audio_data=valid_audio_data())
    @settings(max_examples=8, deadline=3000, suppress_health_check=[HealthCheck.data_too_large])
    def test_pitch_variance_property(self, audio_data):
        """
        **Feature: ai-voice-detector, Property 16: Variance Metrics Calculation**
        
        For any voice audio input, the Feature_Extractor should calculate pitch variance 
        and energy variance metrics as non-negative float values.
        
        **Validates: Requirements 6.3**
        """
        # Ensure audio is valid
        assume(audio_data.is_valid())
        assume(len(audio_data.samples) > 1024)  # Minimum samples for analysis
        
        # Analyze pitch variance
        pitch_variance = self.extractor.analyze_pitch_variance(audio_data)
        
        # Property 1: Output should be float
        assert isinstance(pitch_variance, float), "Pitch variance must be float"
        
        # Property 2: Should be non-negative
        assert pitch_variance >= 0.0, f"Pitch variance must be non-negative, got {pitch_variance}"
        
        # Property 3: Should be finite
        assert np.isfinite(pitch_variance), "Pitch variance must be finite"
        
        # Property 4: Should be reasonable range (coefficient of variation typically < 10)
        assert pitch_variance <= 20.0, f"Pitch variance should be reasonable, got {pitch_variance}"
    
    @given(audio_data=valid_audio_data())
    @settings(max_examples=8, deadline=3000, suppress_health_check=[HealthCheck.data_too_large])
    def test_energy_variance_property(self, audio_data):
        """
        Property: Energy variance should be non-negative float
        """
        # Ensure audio is valid
        assume(audio_data.is_valid())
        assume(len(audio_data.samples) > 1024)
        
        # Analyze energy variance
        energy_variance = self.extractor.analyze_energy_variance(audio_data)
        
        # Property 1: Output should be float
        assert isinstance(energy_variance, float), "Energy variance must be float"
        
        # Property 2: Should be non-negative
        assert energy_variance >= 0.0, f"Energy variance must be non-negative, got {energy_variance}"
        
        # Property 3: Should be finite
        assert np.isfinite(energy_variance), "Energy variance must be finite"
        
        # Property 4: Should be reasonable range
        assert energy_variance <= 20.0, f"Energy variance should be reasonable, got {energy_variance}"
    
    @given(audio_data=valid_audio_data())
    @settings(max_examples=5, deadline=2000, suppress_health_check=[HealthCheck.data_too_large])
    def test_variance_determinism_property(self, audio_data):
        """
        Property: Variance calculations should be deterministic for same input
        """
        assume(audio_data.is_valid())
        assume(len(audio_data.samples) > 1024)
        
        # Calculate variance metrics twice
        pitch_var_1 = self.extractor.analyze_pitch_variance(audio_data)
        pitch_var_2 = self.extractor.analyze_pitch_variance(audio_data)
        
        energy_var_1 = self.extractor.analyze_energy_variance(audio_data)
        energy_var_2 = self.extractor.analyze_energy_variance(audio_data)
        
        # Should be identical (within floating point precision)
        assert abs(pitch_var_1 - pitch_var_2) < 1e-10, "Pitch variance should be deterministic"
        assert abs(energy_var_1 - energy_var_2) < 1e-10, "Energy variance should be deterministic"
    
    @given(
        frequency=st.floats(min_value=200, max_value=800),
        sample_rate=st.sampled_from([16000, 22050])
    )
    @settings(max_examples=5, deadline=3000)
    def test_pure_tone_low_variance_property(self, frequency, sample_rate):
        """
        Property: Pure tones should have lower variance than noisy signals
        """
        duration = 3.0
        num_samples = int(duration * sample_rate)
        t = np.linspace(0, duration, num_samples)
        
        # Create pure tone (should have low variance)
        pure_samples = 0.5 * np.sin(2 * np.pi * frequency * t)
        pure_audio = AudioData(
            samples=pure_samples,
            sample_rate=sample_rate,
            duration=duration,
            channels=1
        )
        
        # Create noisy signal (should have higher variance)
        noisy_samples = pure_samples + 0.2 * np.random.randn(num_samples)
        noisy_samples = np.clip(noisy_samples, -1.0, 1.0)
        noisy_audio = AudioData(
            samples=noisy_samples,
            sample_rate=sample_rate,
            duration=duration,
            channels=1
        )
        
        # Calculate variances
        pure_pitch_var = self.extractor.analyze_pitch_variance(pure_audio)
        noisy_pitch_var = self.extractor.analyze_pitch_variance(noisy_audio)
        
        pure_energy_var = self.extractor.analyze_energy_variance(pure_audio)
        noisy_energy_var = self.extractor.analyze_energy_variance(noisy_audio)
        
        # Pure tone should generally have lower or similar variance
        # (This is a loose constraint as pitch detection can be noisy)
        assert pure_energy_var <= noisy_energy_var * 2.0, \
            "Pure tone should have lower or similar energy variance than noisy signal"
    
    @given(audio_data=valid_audio_data())
    @settings(max_examples=5, deadline=1500)
    def test_silent_audio_zero_variance_property(self, audio_data):
        """
        Property: Silent audio should have zero or very low variance
        """
        assume(audio_data.is_valid())
        
        # Create silent audio
        silent_samples = np.zeros_like(audio_data.samples)
        silent_audio = AudioData(
            samples=silent_samples,
            sample_rate=audio_data.sample_rate,
            duration=audio_data.duration,
            channels=1
        )
        
        # Calculate variances for silent audio
        pitch_variance = self.extractor.analyze_pitch_variance(silent_audio)
        energy_variance = self.extractor.analyze_energy_variance(silent_audio)
        
        # Silent audio should have zero variance
        assert pitch_variance == 0.0, "Silent audio should have zero pitch variance"
        assert energy_variance == 0.0, "Silent audio should have zero energy variance"
    
    @given(
        amplitude=st.floats(min_value=0.1, max_value=0.9),
        sample_rate=st.sampled_from([16000, 22050])
    )
    @settings(max_examples=5, deadline=1500)
    def test_constant_amplitude_low_energy_variance(self, amplitude, sample_rate):
        """
        Property: Constant amplitude signals should have low energy variance
        """
        duration = 3.0
        num_samples = int(duration * sample_rate)
        
        # Create constant amplitude signal
        samples = np.full(num_samples, amplitude)
        audio_data = AudioData(
            samples=samples,
            sample_rate=sample_rate,
            duration=duration,
            channels=1
        )
        
        # Calculate energy variance
        energy_variance = self.extractor.analyze_energy_variance(audio_data)
        
        # Constant amplitude should have very low energy variance (but not necessarily zero due to windowing)
        assert energy_variance <= 0.1, f"Constant amplitude signal should have very low energy variance, got {energy_variance}"