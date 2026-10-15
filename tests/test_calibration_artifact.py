"""Regression tests for intentionally disabled calibration artifacts."""

import joblib

from voice_detector.models.calibration import ConfidenceCalibrator


def test_unfitted_calibrator_artifact_is_treated_as_disabled(tmp_path):
    artifact_path = tmp_path / "calibration_params.joblib"
    joblib.dump(ConfidenceCalibrator(), artifact_path)

    calibrator = ConfidenceCalibrator()
    calibrator.load_calibrator(artifact_path)

    assert not calibrator.has_usable_calibrator()
    assert calibrator.calibrate_confidence(0.73) == 0.73


def test_unfitted_calibrator_artifact_preserves_method_and_weights(tmp_path):
    artifact_path = tmp_path / "calibration_params.joblib"
    saved = ConfidenceCalibrator(method="isotonic")
    saved.feature_quality_weights = {"mfcc_quality": 1.0}
    joblib.dump(saved, artifact_path)

    calibrator = ConfidenceCalibrator()
    calibrator.load_calibrator(artifact_path)

    assert calibrator.method == "isotonic"
    assert calibrator.feature_quality_weights == {"mfcc_quality": 1.0}
    assert not calibrator.is_calibrated()


def test_nested_unfitted_calibrator_dict_artifact_is_treated_as_disabled(tmp_path):
    artifact_path = tmp_path / "calibration_params.joblib"
    nested = ConfidenceCalibrator()
    saved = {
        "calibrator": nested,
        "method": "platt",
        "is_fitted": False,
        "is_calibrated": False,
        "feature_quality_weights": {
            "mfcc_quality": 0.3,
            "spectral_quality": 0.25,
            "temporal_quality": 0.25,
            "variance_quality": 0.2,
        },
    }
    joblib.dump(saved, artifact_path)

    calibrator = ConfidenceCalibrator()
    calibrator.load_calibrator(artifact_path)

    assert not calibrator.has_usable_calibrator()
    assert calibrator.calibrate_confidence(0.73) == 0.73
    assert calibrator.get_calibrator_info()["is_usable"] is False
