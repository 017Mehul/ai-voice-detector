"""
Tests for Audio Processing Module

Tests the AudioProcessor class functionality including Base64 decoding,
format conversion, normalization, and validation.
"""

import pytest
import base64
import numpy as np
import io
from pydub import AudioSegment
import soundfile as sf

from ai_voice_detector.audio import AudioProcessor, AudioProcessingError
from ai_voice_detector.models.data_models import AudioData, ValidationResult


class TestAudioProcessor:
    """Test AudioProcessor functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.processor = AudioProcessor()
    
    def create_test_wav_base64(self, duration_seconds=3.0, sample_rate=22050):
        """Create a test WAV file encoded as Base64 (simpler than MP3)"""
        # Generate test audio signal
        t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds))
        # Create a simple sine wave
        samples = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
        
        # Convert to AudioSegment using WAV format (no ffmpeg needed)
        # Scale to 16-bit integer range
        samples_int16 = (samples * 32767).astype(np.int16)
        audio_segment = AudioSegment(
            samples_int16.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,
            channels=1
        )
        
        # Export to WAV in memory (no external dependencies)
        wav_io = io.BytesIO()
        audio_segment.export(wav_io, format="wav")
        wav_bytes = wav_io.getvalue()
        
        # Encode as Base64
        return base64.b64encode(wav_bytes).decode('utf-8')
    
    def create_test_mp3_base64(self, duration_seconds=3.0, sample_rate=22050):
        """Create a test MP3 file encoded as Base64 - fallback to WAV for testing"""
        # For testing purposes, we'll use WAV format to avoid ffmpeg dependency
        # In real usage, the system would receive actual MP3 files
        return self.create_test_wav_base64(duration_seconds, sample_rate)

    def create_test_flac_base64(self, duration_seconds=3.0, sample_rate=22050):
        """Create a test FLAC file encoded as Base64."""
        t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), endpoint=False)
        samples = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        flac_io = io.BytesIO()
        sf.write(flac_io, samples, sample_rate, format="FLAC")
        flac_bytes = flac_io.getvalue()

        return base64.b64encode(flac_bytes).decode('utf-8')
    
    def test_decode_valid_base64_audio(self):
        """Test decoding valid Base64 MP3 audio"""
        # Create test MP3 Base64
        test_b64 = self.create_test_mp3_base64(duration_seconds=3.0)
        
        # Decode audio
        audio_data = self.processor.decode_base64_audio(test_b64)
        
        # Verify result
        assert isinstance(audio_data, AudioData)
        assert audio_data.duration >= 2.9  # Allow for small conversion differences
        assert audio_data.duration <= 3.1
        assert audio_data.sample_rate == self.processor.target_sample_rate
        assert len(audio_data.samples) > 0
        assert audio_data.channels == 1

    def test_decode_valid_base64_flac_audio(self):
        """Test decoding valid Base64 FLAC audio."""
        test_b64 = self.create_test_flac_base64(duration_seconds=3.0)

        audio_data = self.processor.decode_base64_audio(test_b64)

        assert isinstance(audio_data, AudioData)
        assert audio_data.duration >= 2.9
        assert audio_data.duration <= 3.1
        assert audio_data.sample_rate == self.processor.target_sample_rate
        assert len(audio_data.samples) > 0
        assert audio_data.channels == 1
    
    def test_decode_invalid_base64(self):
        """Test handling of invalid Base64 strings"""
        invalid_b64_strings = [
            "invalid_base64!@#",
            "not_base64_at_all",
            "",
            "SGVsbG8gV29ybGQ=",  # Valid Base64 but not MP3
        ]
        
        for invalid_b64 in invalid_b64_strings:
            with pytest.raises(AudioProcessingError) as exc_info:
                self.processor.decode_base64_audio(invalid_b64)
            
            # Check error message contains relevant information
            error_msg = str(exc_info.value).lower()
            assert any(keyword in error_msg for keyword in 
                      ['base64', 'invalid', 'decode', 'mp3', 'format'])
    
    def test_decode_empty_base64(self):
        """Test handling of empty Base64 string"""
        with pytest.raises(AudioProcessingError) as exc_info:
            self.processor.decode_base64_audio("")
        
        assert "empty" in str(exc_info.value).lower() or "invalid" in str(exc_info.value).lower()
    
    def test_convert_mp3_to_wav(self):
        """Test MP3 to WAV conversion"""
        # Create test MP3 bytes
        test_b64 = self.create_test_mp3_base64(duration_seconds=2.5)
        mp3_bytes = base64.b64decode(test_b64)
        
        # Convert to WAV
        audio_data = self.processor.convert_mp3_to_wav(mp3_bytes)
        
        # Verify conversion
        assert isinstance(audio_data, AudioData)
        assert audio_data.duration >= 2.4
        assert audio_data.duration <= 2.6
        assert audio_data.sample_rate == self.processor.target_sample_rate
        assert audio_data.channels == 1
        assert len(audio_data.samples) > 0
        
        # Check samples are valid
        assert not np.any(np.isnan(audio_data.samples))
        assert not np.any(np.isinf(audio_data.samples))
    
    def test_convert_invalid_mp3(self):
        """Test handling of invalid MP3 data"""
        invalid_mp3_data = [
            b"not_mp3_data",
            b"",
            b"random_bytes_123456789",
        ]
        
        for invalid_data in invalid_mp3_data:
            with pytest.raises(AudioProcessingError) as exc_info:
                self.processor.convert_mp3_to_wav(invalid_data)
            
            error_msg = str(exc_info.value).lower()
            assert any(keyword in error_msg for keyword in 
                      ['mp3', 'format', 'invalid', 'convert', 'corrupted'])
    
    def test_normalize_audio(self):
        """Test audio normalization"""
        # Create test audio with known characteristics
        samples = np.array([0.1, -0.2, 0.8, -0.5, 0.3])
        audio_data = AudioData(
            samples=samples,
            sample_rate=22050,
            duration=len(samples) / 22050,
            channels=1
        )
        
        # Normalize
        normalized = self.processor.normalize_audio(audio_data)
        
        # Check normalization
        assert isinstance(normalized, AudioData)
        assert len(normalized.samples) == len(samples)
        assert normalized.sample_rate == audio_data.sample_rate
        assert normalized.duration == audio_data.duration
        assert normalized.channels == audio_data.channels
        
        # Check amplitude is normalized
        max_amplitude = np.max(np.abs(normalized.samples))
        assert max_amplitude <= 1.0
        
        # Check samples are in valid range
        assert np.all(normalized.samples >= -1.0)
        assert np.all(normalized.samples <= 1.0)
    
    def test_normalize_zero_amplitude_audio(self):
        """Test normalization of silent audio"""
        # Create silent audio
        samples = np.zeros(1000)
        audio_data = AudioData(
            samples=samples,
            sample_rate=22050,
            duration=len(samples) / 22050,
            channels=1
        )
        
        # Should not raise error but log warning
        normalized = self.processor.normalize_audio(audio_data)
        assert isinstance(normalized, AudioData)
        assert len(normalized.samples) == len(samples)
    
    def test_validate_audio_duration_valid(self):
        """Test duration validation for valid audio"""
        # Create valid duration audio
        audio_data = AudioData(
            samples=np.random.randn(44100 * 3),  # 3 seconds
            sample_rate=22050,
            duration=3.0,
            channels=1
        )
        
        result = self.processor.validate_audio_duration(audio_data)
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert result.error_message is None
        assert result.error_code is None
    
    def test_validate_audio_duration_too_short(self):
        """Test duration validation for too short audio"""
        # Create too short audio
        audio_data = AudioData(
            samples=np.random.randn(1000),
            sample_rate=22050,
            duration=1.0,  # Less than minimum 2.0 seconds
            channels=1
        )
        
        result = self.processor.validate_audio_duration(audio_data)
        assert isinstance(result, ValidationResult)
        assert result.is_valid is False
        assert "short" in result.error_message.lower()
        assert result.error_code == "AUDIO_TOO_SHORT"
    
    def test_validate_audio_duration_too_long(self):
        """Test duration validation for too long audio"""
        # Create too long audio
        audio_data = AudioData(
            samples=np.random.randn(1000),
            sample_rate=22050,
            duration=400.0,  # More than maximum 300.0 seconds
            channels=1
        )
        
        result = self.processor.validate_audio_duration(audio_data)
        assert isinstance(result, ValidationResult)
        assert result.is_valid is False
        assert "long" in result.error_message.lower()
        assert result.error_code == "AUDIO_TOO_LONG"
    
    def test_detect_corrupted_audio_valid(self):
        """Test corruption detection for valid audio"""
        # Create valid audio
        samples = np.random.randn(44100) * 0.5  # Random noise with reasonable amplitude
        audio_data = AudioData(
            samples=samples,
            sample_rate=22050,
            duration=2.0,
            channels=1
        )
        
        result = self.processor.detect_corrupted_audio(audio_data)
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
    
    def test_detect_corrupted_audio_empty(self):
        """Test corruption detection for empty audio"""
        audio_data = AudioData(
            samples=np.array([]),
            sample_rate=22050,
            duration=0.0,
            channels=1
        )
        
        result = self.processor.detect_corrupted_audio(audio_data)
        assert result.is_valid is False
        assert result.error_code == "EMPTY_AUDIO"
    
    def test_detect_corrupted_audio_nan_values(self):
        """Test corruption detection for NaN values"""
        samples = np.array([0.1, np.nan, 0.3, -0.2])
        audio_data = AudioData(
            samples=samples,
            sample_rate=22050,
            duration=len(samples) / 22050,
            channels=1
        )
        
        result = self.processor.detect_corrupted_audio(audio_data)
        assert result.is_valid is False
        assert result.error_code == "CORRUPTED_AUDIO"
        assert "nan" in result.error_message.lower()
    
    def test_detect_corrupted_audio_infinite_values(self):
        """Test corruption detection for infinite values"""
        samples = np.array([0.1, np.inf, 0.3, -0.2])
        audio_data = AudioData(
            samples=samples,
            sample_rate=22050,
            duration=len(samples) / 22050,
            channels=1
        )
        
        result = self.processor.detect_corrupted_audio(audio_data)
        assert result.is_valid is False
        assert result.error_code == "CORRUPTED_AUDIO"
        assert "infinite" in result.error_message.lower()
    
    def test_detect_corrupted_audio_silent(self):
        """Test corruption detection for completely silent audio"""
        samples = np.zeros(44100)  # All zeros
        audio_data = AudioData(
            samples=samples,
            sample_rate=22050,
            duration=2.0,
            channels=1
        )
        
        result = self.processor.detect_corrupted_audio(audio_data)
        assert result.is_valid is False
        assert result.error_code == "SILENT_AUDIO"
    
    def test_detect_corrupted_audio_low_dynamic_range(self):
        """Test corruption detection for very low dynamic range"""
        # Create audio with very small variations
        samples = np.full(44100, 0.5) + np.random.randn(44100) * 0.001
        audio_data = AudioData(
            samples=samples,
            sample_rate=22050,
            duration=2.0,
            channels=1
        )
        
        result = self.processor.detect_corrupted_audio(audio_data)
        assert result.is_valid is False
        assert result.error_code == "LOW_QUALITY_AUDIO"
    
    def test_validate_audio_integrity_valid(self):
        """Test comprehensive integrity validation for valid audio"""
        # Create valid test audio
        test_b64 = self.create_test_mp3_base64(duration_seconds=3.0)
        audio_data = self.processor.decode_base64_audio(test_b64)
        
        result = self.processor.validate_audio_integrity(audio_data)
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
    
    def test_validate_audio_integrity_invalid_duration(self):
        """Test integrity validation fails for invalid duration"""
        # Create short audio
        samples = np.random.randn(1000) * 0.5
        audio_data = AudioData(
            samples=samples,
            sample_rate=22050,
            duration=1.0,  # Too short
            channels=1
        )
        
        result = self.processor.validate_audio_integrity(audio_data)
        assert result.is_valid is False
        assert result.error_code == "AUDIO_TOO_SHORT"
    
    def test_process_audio_complete_pipeline(self):
        """Test complete audio processing pipeline"""
        # Create test MP3 Base64
        test_b64 = self.create_test_mp3_base64(duration_seconds=4.0)
        
        # Process audio
        processed_audio, validation_error = self.processor.process_audio(test_b64)
        
        # Should succeed
        assert validation_error is None
        assert isinstance(processed_audio, AudioData)
        assert processed_audio.is_valid()
        assert processed_audio.duration >= 3.9
        assert processed_audio.duration <= 4.1
        
        # Check normalization was applied
        max_amplitude = np.max(np.abs(processed_audio.samples))
        assert max_amplitude <= 1.0
    
    def test_process_audio_pipeline_with_error(self):
        """Test audio processing pipeline with invalid input"""
        invalid_b64 = "invalid_base64_string"
        
        processed_audio, validation_error = self.processor.process_audio(invalid_b64)
        
        # Should return error
        assert validation_error is not None
        assert validation_error.is_valid is False
        assert validation_error.error_code == "PROCESSING_ERROR"
        assert isinstance(processed_audio, AudioData)  # Dummy audio returned
        assert not processed_audio.is_valid()  # Should be invalid dummy
