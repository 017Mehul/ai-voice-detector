"""
Regression tests for backend completion fixes.
"""

import base64
import io
import os
import time
import uuid
import wave
from pathlib import Path

import joblib
import numpy as np
import pytest
import soundfile as sf

from ai_voice_detector.api.validation import validate_audio_header, validate_request_rate_and_duplicates
from ai_voice_detector.models.calibration import ConfidenceCalibrator
from ai_voice_detector.models.data_models import FeatureVector, SpectralFeatures, TemporalFeatures
from ai_voice_detector.models.dataset_preparation import TrainingDatasetPreparator
from ai_voice_detector.models.training import ModelTrainingPipeline
from ai_voice_detector.services.voice_detection_service import VoiceDetectionService


def create_feature_vector(seed: int = 0) -> FeatureVector:
    """Create a compact valid feature vector for training and service tests."""
    rng = np.random.default_rng(seed)
    spectral_length = 6

    return FeatureVector(
        mfcc_features=rng.normal(0.0, 1.0, size=(13, 8)).astype(np.float32),
        spectral_features=SpectralFeatures(
            centroid=np.abs(rng.normal(1000.0, 50.0, size=spectral_length)).astype(np.float32),
            bandwidth=np.abs(rng.normal(500.0, 25.0, size=spectral_length)).astype(np.float32),
            rolloff=np.abs(rng.normal(3000.0, 100.0, size=spectral_length)).astype(np.float32),
            zero_crossing_rate=np.clip(rng.normal(0.2, 0.05, size=spectral_length), 0.0, 1.0).astype(np.float32),
        ),
        pitch_variance=0.5,
        energy_variance=0.4,
        temporal_features=TemporalFeatures(
            frame_consistency=0.6,
            rhythm_regularity=0.55,
            pause_patterns=np.array([0.1, 0.2, 0.15], dtype=np.float32)
        )
    )


def write_wav_file(path: Path, frequency: float, duration_seconds: float = 2.1, sample_rate: int = 22050) -> None:
    """Write a small mono WAV file for dataset preparation tests."""
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), endpoint=False)
    samples = (0.4 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
    pcm_samples = (samples * 32767).astype(np.int16)

    with wave.open(str(path), 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_samples.tobytes())


def create_flac_base64(duration_seconds: float = 2.1, sample_rate: int = 22050, frequency: float = 440.0) -> str:
    """Create Base64-encoded FLAC audio for validation tests."""
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), endpoint=False)
    samples = (0.4 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)

    flac_io = io.BytesIO()
    sf.write(flac_io, samples, sample_rate, format="FLAC")
    return base64.b64encode(flac_io.getvalue()).decode("utf-8")


class DummyClassifier:
    """Classifier stub used to verify service behavior when the model is unavailable."""

    def is_trained(self) -> bool:
        return False

    def predict(self, features):  # pragma: no cover - should never be reached in this test
        raise AssertionError("predict should not be called for an untrained classifier")

    def get_model_info(self) -> dict:
        return {}


class DummyAudioProcessor:
    def process_audio(self, *_args, **_kwargs):
        raise NotImplementedError


class DummyFeatureExtractor:
    def extract_all_features(self, *_args, **_kwargs):
        raise NotImplementedError

    def get_feature_dimensions(self):
        return {}


def test_legacy_nested_calibrator_artifact_loads_successfully(tmp_path):
    inner_calibrator = ConfidenceCalibrator()
    inner_calibrator.fit(
        np.array([0.1, 0.3, 0.7, 0.9], dtype=float),
        np.array([0, 0, 1, 1], dtype=int)
    )

    artifact_path = tmp_path / "legacy_calibrator.joblib"
    joblib.dump({"calibrator": inner_calibrator, "is_calibrated": True}, artifact_path)

    restored = ConfidenceCalibrator()
    restored.load_calibrator(artifact_path)

    calibrated = restored.calibrate_confidence(0.8)
    assert restored.is_calibrated()
    assert 0.0 <= calibrated <= 1.0


def test_invalid_nested_calibrator_artifact_is_rejected(tmp_path):
    artifact_path = tmp_path / "broken_calibrator.joblib"
    joblib.dump({"calibrator": ConfidenceCalibrator(), "is_calibrated": True}, artifact_path)

    restored = ConfidenceCalibrator()
    with pytest.raises(ValueError):
        restored.load_calibrator(artifact_path)


def test_flac_audio_header_is_accepted():
    validation = validate_audio_header(create_flac_base64())
    assert validation.is_valid


def test_duplicate_detection_blocks_repeated_requests_when_enabled(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "true")

    request_data = {"audio_base64": "dGVzdA=="}
    client_id = f"duplicate-test-{uuid.uuid4()}"

    first_result = validate_request_rate_and_duplicates(request_data, client_id, duplicate_window_seconds=30)
    second_result = validate_request_rate_and_duplicates(request_data, client_id, duplicate_window_seconds=30)

    assert first_result.is_valid
    assert not second_result.is_valid
    assert second_result.error_code == "DUPLICATE_REQUEST"


def test_duplicate_detection_is_skipped_when_rate_limiting_disabled(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")

    request_data = {"audio_base64": "dGVzdA=="}
    client_id = f"duplicate-disabled-{uuid.uuid4()}"

    first_result = validate_request_rate_and_duplicates(request_data, client_id, duplicate_window_seconds=30)
    second_result = validate_request_rate_and_duplicates(request_data, client_id, duplicate_window_seconds=30)

    assert first_result.is_valid
    assert second_result.is_valid


def test_dataset_preparator_supports_dictionary_sources_and_zero_splits(tmp_path):
    human_dir = tmp_path / "human"
    ai_dir = tmp_path / "ai_generated"
    human_dir.mkdir()
    ai_dir.mkdir()

    write_wav_file(human_dir / "human_1.wav", frequency=220.0)
    write_wav_file(human_dir / "human_2.wav", frequency=240.0)
    write_wav_file(ai_dir / "ai_1.wav", frequency=440.0)
    write_wav_file(ai_dir / "ai_2.wav", frequency=480.0)

    preparator = TrainingDatasetPreparator(random_state=7)
    result = preparator.prepare_dataset(
        dataset_source={"type": "directory", "source": str(tmp_path)},
        extract_features=False,
        test_size=0.0,
        validation_size=0.0
    )

    assert result["original_analysis"]["total_samples"] == 4
    assert result["dataset_splits"]["train_size"] == 4
    assert result["dataset_splits"]["validation_size"] == 0
    assert result["dataset_splits"]["test_size"] == 0


def test_prepare_training_data_supports_zero_splits():
    features = [create_feature_vector(seed) for seed in range(6)]
    labels = ["HUMAN", "AI_GENERATED", "HUMAN", "AI_GENERATED", "HUMAN", "AI_GENERATED"]

    pipeline = ModelTrainingPipeline(random_state=11)
    data_info = pipeline.prepare_training_data(
        features_list=features,
        labels=labels,
        test_size=0.0,
        validation_size=0.0
    )

    assert data_info["train_size"] == 6
    assert data_info["val_size"] == 0
    assert data_info["test_size"] == 0
    assert data_info["X_val"].shape[0] == 0
    assert data_info["X_test"].shape[0] == 0


def test_untrained_classifier_returns_service_unavailable():
    service = VoiceDetectionService(
        audio_processor=DummyAudioProcessor(),
        feature_extractor=DummyFeatureExtractor(),
        classifier=DummyClassifier(),
        confidence_calibrator=ConfidenceCalibrator()
    )

    classification, raw_confidence, error = service._classify_voice(create_feature_vector(), "untrained-model")

    assert classification == "HUMAN"
    assert raw_confidence == 0.0
    assert error is not None
    assert error.error_code == "SERVICE_UNAVAILABLE"
