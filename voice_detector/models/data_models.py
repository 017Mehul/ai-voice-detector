"""
Data models for AI Voice Detection System

This module contains all data model classes for requests, responses, and
internal data structures. Implements validation methods and serialization
support as specified in the design document.
"""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import numpy as np
import json
import base64
import re


@dataclass
class DetectionRequest:
    """
    Request model for voice detection API endpoint.

    Attributes:
        audio_base64: Base64-encoded MP3, WAV, or FLAC audio data
    """

    audio_base64: str

    def validate(self) -> bool:
        """
        Validates request format and content.

        Returns:
            bool: True if request is valid, False otherwise
        """
        if not self.audio_base64 or not isinstance(self.audio_base64, str):
            return False

        try:
            cleaned_b64 = re.sub(r"\s+", "", self.audio_base64)
            base64.b64decode(cleaned_b64, validate=True)
            return len(cleaned_b64) > 0
        except Exception:
            return False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DetectionRequest":
        """Create DetectionRequest from dictionary."""
        return cls(audio_base64=data.get("audio_base64", ""))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class DetectionResponse:
    """
    Response model for successful voice detection.

    Attributes:
        classification: Either "AI_GENERATED" or "HUMAN"
        confidence_score: Float between 0.0 and 1.0
    """

    classification: str
    confidence_score: float

    def __post_init__(self):
        """Validate response data after initialization."""
        if self.classification not in ["AI_GENERATED", "HUMAN"]:
            raise ValueError(f"Invalid classification: {self.classification}")

        if not isinstance(self.confidence_score, (int, float)):
            raise ValueError(f"Confidence score must be numeric: {self.confidence_score}")

        if not (0.0 <= self.confidence_score <= 1.0):
            raise ValueError(
                f"Confidence score must be between 0.0 and 1.0: {self.confidence_score}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "classification": self.classification,
            "confidence_score": float(self.confidence_score),
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


@dataclass
class ErrorResponse:
    """
    Response model for error cases.

    Attributes:
        error: Error code identifier
        message: Human-readable error description
    """

    error: str
    message: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "error": self.error,
            "message": self.message,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


@dataclass
class AudioData:
    """
    Internal data model for processed audio.

    Attributes:
        samples: Audio sample data as numpy array
        sample_rate: Sample rate in Hz
        duration: Duration in seconds
        channels: Number of audio channels
    """

    samples: np.ndarray
    sample_rate: int
    duration: float
    channels: int

    def is_valid(self) -> bool:
        """
        Check if audio data meets minimum requirements.

        Returns:
            bool: True if audio is valid for processing
        """
        return (
            self.duration >= 2.0
            and len(self.samples) > 0
            and self.sample_rate > 0
            and self.channels > 0
        )

    def get_info(self) -> Dict[str, Any]:
        """Get audio information as dictionary."""
        return {
            "sample_rate": self.sample_rate,
            "duration": self.duration,
            "channels": self.channels,
            "samples_count": len(self.samples),
        }


@dataclass
class SpectralFeatures:
    """
    Spectral feature data extracted from audio.

    Attributes:
        centroid: Spectral centroid values
        bandwidth: Spectral bandwidth values
        rolloff: Spectral rolloff values
        zero_crossing_rate: Zero crossing rate values
    """

    centroid: np.ndarray
    bandwidth: np.ndarray
    rolloff: np.ndarray
    zero_crossing_rate: np.ndarray

    def validate(self) -> bool:
        """Validate that all spectral features have consistent dimensions."""
        arrays = [self.centroid, self.bandwidth, self.rolloff, self.zero_crossing_rate]

        if not all(isinstance(arr, np.ndarray) and len(arr) > 0 for arr in arrays):
            return False

        lengths = [len(arr) for arr in arrays]
        return len(set(lengths)) == 1

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistical summary of spectral features."""
        return {
            "centroid": {
                "mean": float(np.mean(self.centroid)),
                "std": float(np.std(self.centroid)),
                "min": float(np.min(self.centroid)),
                "max": float(np.max(self.centroid)),
            },
            "bandwidth": {
                "mean": float(np.mean(self.bandwidth)),
                "std": float(np.std(self.bandwidth)),
                "min": float(np.min(self.bandwidth)),
                "max": float(np.max(self.bandwidth)),
            },
            "rolloff": {
                "mean": float(np.mean(self.rolloff)),
                "std": float(np.std(self.rolloff)),
                "min": float(np.min(self.rolloff)),
                "max": float(np.max(self.rolloff)),
            },
            "zero_crossing_rate": {
                "mean": float(np.mean(self.zero_crossing_rate)),
                "std": float(np.std(self.zero_crossing_rate)),
                "min": float(np.min(self.zero_crossing_rate)),
                "max": float(np.max(self.zero_crossing_rate)),
            },
        }


@dataclass
class TemporalFeatures:
    """
    Temporal feature data for voice consistency analysis.

    Attributes:
        frame_consistency: Frame-to-frame consistency score
        rhythm_regularity: Rhythm regularity measure
        pause_patterns: Pause pattern analysis results
    """

    frame_consistency: float
    rhythm_regularity: float
    pause_patterns: np.ndarray

    def validate(self) -> bool:
        """Validate temporal features are within expected ranges."""
        return (
            isinstance(self.frame_consistency, (int, float))
            and isinstance(self.rhythm_regularity, (int, float))
            and isinstance(self.pause_patterns, np.ndarray)
            and len(self.pause_patterns) > 0
            and 0.0 <= self.frame_consistency <= 1.0
            and 0.0 <= self.rhythm_regularity <= 1.0
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of temporal features."""
        return {
            "frame_consistency": float(self.frame_consistency),
            "rhythm_regularity": float(self.rhythm_regularity),
            "pause_patterns_count": len(self.pause_patterns),
            "pause_patterns_mean": float(np.mean(self.pause_patterns)),
        }


@dataclass
class FeatureVector:
    """
    Combined feature vector containing all extracted features.

    Attributes:
        mfcc_features: MFCC coefficient array
        spectral_features: Spectral analysis features
        pitch_variance: Pitch variation measure
        energy_variance: Energy variation measure
        temporal_features: Temporal consistency features
    """

    mfcc_features: np.ndarray
    spectral_features: SpectralFeatures
    pitch_variance: float
    energy_variance: float
    temporal_features: TemporalFeatures

    def validate(self) -> bool:
        """
        Validate that all feature components are present and valid.

        Returns:
            bool: True if feature vector is complete and valid
        """
        return (
            isinstance(self.mfcc_features, np.ndarray)
            and self.mfcc_features.size > 0
            and isinstance(self.spectral_features, SpectralFeatures)
            and self.spectral_features.validate()
            and isinstance(self.pitch_variance, (int, float))
            and isinstance(self.energy_variance, (int, float))
            and isinstance(self.temporal_features, TemporalFeatures)
            and self.temporal_features.validate()
            and self.pitch_variance >= 0.0
            and self.energy_variance >= 0.0
        )

    def _get_mfcc_matrix(self) -> np.ndarray:
        """
        Normalize MFCC input into a coefficient x frame matrix.

        Property-based tests sometimes provide a 1D coefficient vector instead
        of a 2D matrix, so those inputs are reshaped into one frame per
        coefficient.
        """
        mfcc_array = np.asarray(self.mfcc_features, dtype=float)

        if mfcc_array.ndim == 0:
            return mfcc_array.reshape(1, 1)

        if mfcc_array.ndim == 1:
            return mfcc_array.reshape(-1, 1)

        if mfcc_array.ndim > 2:
            return mfcc_array.reshape(mfcc_array.shape[0], -1)

        return mfcc_array

    def _get_spectral_summary_array(self) -> np.ndarray:
        """Return mean/std summaries for each spectral feature."""
        spectral_stats = self.spectral_features.get_statistics()
        spectral_values = []

        for stats in spectral_stats.values():
            spectral_values.extend([stats["mean"], stats["std"]])

        return np.asarray(spectral_values, dtype=float)

    def _get_temporal_summary_array(self) -> np.ndarray:
        """Return compact temporal summary values."""
        return np.asarray(
            [
                self.temporal_features.frame_consistency,
                self.temporal_features.rhythm_regularity,
                np.mean(self.temporal_features.pause_patterns),
            ],
            dtype=float,
        )

    def to_array(self) -> np.ndarray:
        """
        Convert all features to a detailed flat array for analysis and testing.

        Returns:
            np.ndarray: Flattened MFCC features plus summary features
        """
        mfcc_matrix = self._get_mfcc_matrix()
        return np.concatenate(
            [
                mfcc_matrix.reshape(-1),
                self._get_spectral_summary_array(),
                np.asarray([self.pitch_variance, self.energy_variance], dtype=float),
                self._get_temporal_summary_array(),
            ]
        )

    def to_model_array(self, target_mfcc_coefficients: int = 13) -> np.ndarray:
        """
        Convert features to the fixed-size model input expected by the classifier.

        The bundled RandomForest model was trained on 13 MFCC coefficients
        summarized with mean/std/min/max, so coefficients are padded or
        truncated to keep the feature contract stable at 65 values.
        """
        mfcc_matrix = self._get_mfcc_matrix()

        if mfcc_matrix.shape[0] < target_mfcc_coefficients:
            padding = np.zeros(
                (target_mfcc_coefficients - mfcc_matrix.shape[0], mfcc_matrix.shape[1]),
                dtype=float,
            )
            mfcc_matrix = np.vstack([mfcc_matrix, padding])
        elif mfcc_matrix.shape[0] > target_mfcc_coefficients:
            mfcc_matrix = mfcc_matrix[:target_mfcc_coefficients, :]

        mfcc_stats = []
        for coefficient_values in mfcc_matrix:
            mfcc_stats.extend(
                [
                    np.mean(coefficient_values),
                    np.std(coefficient_values),
                    np.min(coefficient_values),
                    np.max(coefficient_values),
                ]
            )

        return np.concatenate(
            [
                np.asarray(mfcc_stats, dtype=float),
                self._get_spectral_summary_array(),
                np.asarray([self.pitch_variance, self.energy_variance], dtype=float),
                self._get_temporal_summary_array(),
            ]
        )

    def get_feature_info(self) -> Dict[str, Any]:
        """Get information about the feature vector structure."""
        mfcc_matrix = self._get_mfcc_matrix()
        return {
            "mfcc_shape": self.mfcc_features.shape,
            "mfcc_size": int(mfcc_matrix.size),
            "mfcc_stats_features": int(min(mfcc_matrix.shape[0], 13) * 4),
            "spectral_valid": self.spectral_features.validate(),
            "pitch_variance": float(self.pitch_variance),
            "energy_variance": float(self.energy_variance),
            "temporal_valid": self.temporal_features.validate(),
            "total_features": len(self.to_array()),
            "model_features": len(self.to_model_array()),
        }


@dataclass
class ValidationResult:
    """
    Result of input validation operations.

    Attributes:
        is_valid: Whether validation passed
        error_message: Error description if validation failed
        error_code: Error code for API responses
    """

    is_valid: bool
    error_message: Optional[str] = None
    error_code: Optional[str] = None

    def to_error_response(self) -> Optional[ErrorResponse]:
        """Convert to ErrorResponse if validation failed."""
        if not self.is_valid and self.error_message and self.error_code:
            return ErrorResponse(error=self.error_code, message=self.error_message)
        return None
