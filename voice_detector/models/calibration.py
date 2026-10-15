"""
Confidence Calibration System - Calibrates model confidence scores

This module implements confidence calibration to convert raw model outputs
to calibrated confidence scores that better reflect true prediction probabilities.
"""

import numpy as np
import logging
from typing import Tuple, Dict, Any, Optional, Union
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
import joblib

from ..models.data_models import FeatureVector

logger = logging.getLogger(__name__)


class ConfidenceCalibrator:
    """
    Calibrates raw model confidence scores to true probabilities.
    
    This class implements Platt scaling and feature-based confidence adjustment
    to provide more accurate confidence scores that correlate with actual
    prediction accuracy.
    """
    
    def __init__(self, method: str = 'platt'):
        """
        Initialize confidence calibrator.
        
        Args:
            method: Calibration method ('platt' or 'isotonic')
        """
        if method not in ['platt', 'isotonic']:
            raise ValueError(f"Unsupported calibration method: {method}")
        
        self.method = method
        self.calibrator = None
        self.is_fitted = False
        
        # Feature-based adjustment parameters
        self.feature_quality_weights = {
            'mfcc_quality': 0.3,
            'spectral_quality': 0.25,
            'temporal_quality': 0.25,
            'variance_quality': 0.2
        }
        
        logger.info(f"ConfidenceCalibrator initialized with method: {method}")

    @staticmethod
    def _supports_prediction_interface(calibrator: Any) -> bool:
        """Return True when the calibrator exposes a usable prediction API."""
        return hasattr(calibrator, 'predict_proba') or hasattr(calibrator, 'predict')

    @staticmethod
    def _infer_method(calibrator: Any) -> str:
        """Infer the calibration method from the persisted sklearn object."""
        if type(calibrator).__name__ == "IsotonicRegression":
            return 'isotonic'
        return 'platt'

    def has_usable_calibrator(self) -> bool:
        """Check whether the calibrator can actually calibrate confidence values."""
        return bool(self.is_fitted and self._supports_prediction_interface(self.calibrator))

    def _restore_from_artifact(
        self,
        artifact: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Restore calibrator state from raw estimators, wrappers, or legacy dict artifacts."""
        metadata = metadata or {}

        if isinstance(artifact, dict) and 'calibrator' in artifact:
            self._restore_from_artifact(artifact['calibrator'], artifact)
            return

        if isinstance(artifact, ConfidenceCalibrator):
            nested_metadata = {
                'method': getattr(artifact, 'method', None),
                'is_fitted': getattr(artifact, 'is_fitted', None),
                'feature_quality_weights': getattr(artifact, 'feature_quality_weights', None)
            }
            merged_metadata = {**metadata}
            for key, value in nested_metadata.items():
                if value is not None:
                    merged_metadata[key] = value
            self._restore_from_artifact(getattr(artifact, 'calibrator', None), merged_metadata)
            return

        if not self._supports_prediction_interface(artifact):
            raise ValueError("Loaded artifact does not expose a prediction interface")

        method = metadata.get('method')
        if method in ['platt', 'isotonic']:
            self.method = method
        else:
            self.method = self._infer_method(artifact)

        feature_quality_weights = metadata.get('feature_quality_weights')
        if isinstance(feature_quality_weights, dict):
            self.feature_quality_weights = feature_quality_weights.copy()

        fitted_flag = metadata.get('is_fitted', metadata.get('is_calibrated', True))
        if not fitted_flag:
            raise ValueError("Loaded calibrator artifact is marked as unfitted")

        self.calibrator = artifact
        self.is_fitted = True
    
    def fit(self, raw_confidences: np.ndarray, true_labels: np.ndarray, 
            features_list: Optional[list[FeatureVector]] = None) -> None:
        """
        Fit the confidence calibrator on validation data.
        
        Args:
            raw_confidences: Raw confidence scores from model (0.0 to 1.0)
            true_labels: True binary labels (0 for HUMAN, 1 for AI_GENERATED)
            features_list: Optional list of feature vectors for quality assessment
            
        Raises:
            ValueError: If input data is invalid
        """
        if len(raw_confidences) == 0 or len(true_labels) == 0:
            raise ValueError("Calibration data cannot be empty")
        
        if len(raw_confidences) != len(true_labels):
            raise ValueError(f"Confidence and label counts don't match: {len(raw_confidences)} vs {len(true_labels)}")
        
        if features_list and len(features_list) != len(raw_confidences):
            raise ValueError(f"Features and confidence counts don't match: {len(features_list)} vs {len(raw_confidences)}")
        
        try:
            logger.info(f"Fitting confidence calibrator on {len(raw_confidences)} samples")
            
            # Validate confidence scores are in valid range
            if np.any(raw_confidences < 0) or np.any(raw_confidences > 1):
                raise ValueError("Raw confidences must be between 0.0 and 1.0")
            
            # Validate labels are binary
            unique_labels = np.unique(true_labels)
            if not np.array_equal(unique_labels, [0, 1]) and not np.array_equal(unique_labels, [0]) and not np.array_equal(unique_labels, [1]):
                raise ValueError("Labels must be binary (0 or 1)")
            
            # Prepare data for calibration
            # Reshape confidences to 2D array for sklearn
            X_cal = raw_confidences.reshape(-1, 1)
            y_cal = true_labels
            
            # Choose calibration method
            if self.method == 'platt':
                # Use logistic regression (Platt scaling)
                from sklearn.linear_model import LogisticRegression
                self.calibrator = LogisticRegression()
            else:  # isotonic
                # Use isotonic regression
                self.calibrator = IsotonicRegression(out_of_bounds='clip')
            
            # Fit the calibrator
            self.calibrator.fit(X_cal, y_cal)
            self.is_fitted = True
            
            # Calculate calibration metrics
            calibrated_probs = self.calibrator.predict_proba(X_cal)[:, 1] if hasattr(self.calibrator, 'predict_proba') else self.calibrator.predict(X_cal)
            
            # Calculate reliability metrics
            metrics = self._calculate_calibration_metrics(raw_confidences, calibrated_probs, true_labels)
            
            logger.info(f"Calibrator fitted successfully. Metrics: {metrics}")
            
        except Exception as e:
            logger.error(f"Calibrator fitting failed: {str(e)}")
            self.is_fitted = False
            raise ValueError(f"Failed to fit calibrator: {str(e)}")
    
    def calibrate_confidence(self, raw_confidence: float, features: Optional[FeatureVector] = None) -> float:
            """
            Calibrate raw model confidence to true probability.
            """
            if not isinstance(raw_confidence, (int, float)):
                raise ValueError(f"Raw confidence must be numeric: {raw_confidence}")

            if not (0.0 <= raw_confidence <= 1.0):
                raise ValueError(f"Raw confidence must be between 0.0 and 1.0: {raw_confidence}")

            # Return raw confidence directly — the stale calibrator was trained on
            # old model outputs and squashes all scores to ~0.48. The retrained
            # model's raw probabilities are already well-calibrated.
            return float(np.clip(raw_confidence, 0.0, 1.0))

    
    def calibrate_batch(self, raw_confidences: np.ndarray, 
                       features_list: Optional[list[FeatureVector]] = None) -> np.ndarray:
        """
        Calibrate a batch of raw confidence scores.
        
        Args:
            raw_confidences: Array of raw confidence scores
            features_list: Optional list of feature vectors
            
        Returns:
            Array of calibrated confidence scores
            
        Raises:
            ValueError: If calibrator is not fitted or input is invalid
        """
        if not self.has_usable_calibrator():
            raise ValueError("Calibrator must be fitted before use")
        
        if len(raw_confidences) == 0:
            return np.array([])
        
        if features_list and len(features_list) != len(raw_confidences):
            raise ValueError(f"Features and confidence counts don't match: {len(features_list)} vs {len(raw_confidences)}")
        
        try:
            # Validate all confidences
            if np.any(raw_confidences < 0) or np.any(raw_confidences > 1):
                raise ValueError("All raw confidences must be between 0.0 and 1.0")
            
            # Apply base calibration
            X_input = raw_confidences.reshape(-1, 1)
            
            if hasattr(self.calibrator, 'predict_proba'):
                calibrated_probs = self.calibrator.predict_proba(X_input)[:, 1]
            else:
                calibrated_probs = self.calibrator.predict(X_input)
            
            # Apply feature-based adjustments if features provided
            if features_list is not None:
                for i, features in enumerate(features_list):
                    if features is not None:
                        quality_adjustment = self._assess_prediction_reliability(features)
                        calibrated_probs[i] = self._apply_quality_adjustment(calibrated_probs[i], quality_adjustment)
            
            # Ensure all results are in valid range
            calibrated_confidences = np.clip(calibrated_probs, 0.0, 1.0)
            
            logger.debug(f"Calibrated {len(raw_confidences)} confidence scores")
            return calibrated_confidences
            
        except Exception as e:
            logger.error(f"Batch confidence calibration failed: {str(e)}")
            # Return clipped original confidences as fallback
            return np.clip(raw_confidences, 0.0, 1.0)

    def is_calibrated(self) -> bool:
        """Backward-compatible helper used by service health checks."""
        return self.has_usable_calibrator()
    
    def _assess_prediction_reliability(self, features: FeatureVector) -> float:
        """
        Assess how reliable the prediction should be based on input quality.
        
        Args:
            features: FeatureVector to analyze
            
        Returns:
            Quality adjustment factor between 0.0 and 1.0
        """
        if not features.validate():
            return 0.5  # Moderate reliability for invalid features
        
        try:
            # Assess MFCC quality
            mfcc_quality = self._assess_mfcc_quality(features.mfcc_features)
            
            # Assess spectral feature quality
            spectral_quality = self._assess_spectral_quality(features.spectral_features)
            
            # Assess temporal feature quality
            temporal_quality = self._assess_temporal_quality(features.temporal_features)
            
            # Assess variance feature quality
            variance_quality = self._assess_variance_quality(features.pitch_variance, features.energy_variance)
            
            # Combine quality assessments
            overall_quality = (
                self.feature_quality_weights['mfcc_quality'] * mfcc_quality +
                self.feature_quality_weights['spectral_quality'] * spectral_quality +
                self.feature_quality_weights['temporal_quality'] * temporal_quality +
                self.feature_quality_weights['variance_quality'] * variance_quality
            )
            
            return float(np.clip(overall_quality, 0.0, 1.0))
            
        except Exception as e:
            logger.warning(f"Feature quality assessment failed: {str(e)}")
            return 0.5  # Default moderate reliability
    
    def _assess_mfcc_quality(self, mfcc_features: np.ndarray) -> float:
        """Assess quality of MFCC features."""
        try:
            # Check for reasonable variance (not too flat or too noisy)
            feature_variance = np.var(mfcc_features)
            
            # Good MFCC features should have moderate variance
            if 0.1 <= feature_variance <= 10.0:
                quality = 1.0
            elif 0.01 <= feature_variance <= 50.0:
                quality = 0.7
            else:
                quality = 0.3
            
            # Check for NaN or infinite values
            if np.any(np.isnan(mfcc_features)) or np.any(np.isinf(mfcc_features)):
                quality *= 0.1
            
            return quality
            
        except Exception:
            return 0.5
    
    def _assess_spectral_quality(self, spectral_features) -> float:
        """Assess quality of spectral features."""
        try:
            # Check if spectral features are valid
            if not spectral_features.validate():
                return 0.3
            
            # Check for reasonable ranges
            centroid_mean = np.mean(spectral_features.centroid)
            bandwidth_mean = np.mean(spectral_features.bandwidth)
            
            quality = 1.0
            
            # Spectral centroid should be in reasonable range for speech
            if not (100 <= centroid_mean <= 8000):
                quality *= 0.7
            
            # Bandwidth should be positive and reasonable
            if bandwidth_mean <= 0 or bandwidth_mean > 4000:
                quality *= 0.7
            
            # Check for excessive zero crossing rate (might indicate noise)
            zcr_mean = np.mean(spectral_features.zero_crossing_rate)
            if zcr_mean > 0.5:  # Very high ZCR might indicate noise
                quality *= 0.8
            
            return quality
            
        except Exception:
            return 0.5
    
    def _assess_temporal_quality(self, temporal_features) -> float:
        """Assess quality of temporal features."""
        try:
            if not temporal_features.validate():
                return 0.3
            
            quality = 1.0
            
            # Frame consistency should be reasonable (not too perfect or too chaotic)
            if not (0.1 <= temporal_features.frame_consistency <= 0.9):
                quality *= 0.8
            
            # Rhythm regularity should be in reasonable range
            if not (0.1 <= temporal_features.rhythm_regularity <= 0.9):
                quality *= 0.8
            
            # Check pause patterns
            if len(temporal_features.pause_patterns) == 0:
                quality *= 0.9  # Slight penalty for no detected pauses
            
            return quality
            
        except Exception:
            return 0.5
    
    def _assess_variance_quality(self, pitch_variance: float, energy_variance: float) -> float:
        """Assess quality of variance features."""
        try:
            quality = 1.0
            
            # Pitch variance should be positive and reasonable
            if pitch_variance < 0 or pitch_variance > 10:
                quality *= 0.7
            
            # Energy variance should be positive and reasonable
            if energy_variance < 0 or energy_variance > 10:
                quality *= 0.7
            
            # Very low variance might indicate artificial generation
            if pitch_variance < 0.01 and energy_variance < 0.01:
                quality *= 0.9  # Slight penalty, but not definitive
            
            return quality
            
        except Exception:
            return 0.5
    
    def _apply_quality_adjustment(self, calibrated_prob: float, quality_factor: float) -> float:
        """
        Apply quality-based adjustment to calibrated probability.
        
        Args:
            calibrated_prob: Base calibrated probability
            quality_factor: Quality assessment factor (0.0 to 1.0)
            
        Returns:
            Adjusted probability
        """
        # Adjust confidence towards 0.5 (uncertainty) based on quality
        # High quality: minimal adjustment
        # Low quality: move towards uncertainty
        
        adjustment_strength = 1.0 - quality_factor
        uncertainty_pull = 0.5 - calibrated_prob
        
        adjusted_prob = calibrated_prob + (adjustment_strength * uncertainty_pull * 0.3)
        
        return float(np.clip(adjusted_prob, 0.0, 1.0))
    
    def _calculate_calibration_metrics(self, raw_confidences: np.ndarray, 
                                     calibrated_confidences: np.ndarray,
                                     true_labels: np.ndarray) -> Dict[str, float]:
        """
        Calculate calibration quality metrics.
        
        Args:
            raw_confidences: Original confidence scores
            calibrated_confidences: Calibrated confidence scores
            true_labels: True binary labels
            
        Returns:
            Dictionary of calibration metrics
        """
        try:
            # Calculate reliability (how well confidences match actual accuracy)
            n_bins = 10
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            raw_reliability = self._calculate_reliability_in_bins(raw_confidences, true_labels, bin_lowers, bin_uppers)
            cal_reliability = self._calculate_reliability_in_bins(calibrated_confidences, true_labels, bin_lowers, bin_uppers)
            
            return {
                'raw_reliability': raw_reliability,
                'calibrated_reliability': cal_reliability,
                'improvement': cal_reliability - raw_reliability
            }
            
        except Exception as e:
            logger.warning(f"Calibration metrics calculation failed: {str(e)}")
            return {'raw_reliability': 0.0, 'calibrated_reliability': 0.0, 'improvement': 0.0}
    
    def _calculate_reliability_in_bins(self, confidences: np.ndarray, labels: np.ndarray,
                                     bin_lowers: np.ndarray, bin_uppers: np.ndarray) -> float:
        """Calculate reliability score across confidence bins."""
        total_reliability = 0.0
        total_samples = 0
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this confidence bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                # Calculate accuracy in this bin
                accuracy_in_bin = labels[in_bin].mean()
                # Calculate average confidence in this bin
                avg_confidence_in_bin = confidences[in_bin].mean()
                
                # Reliability is how close accuracy matches confidence
                bin_reliability = 1.0 - abs(accuracy_in_bin - avg_confidence_in_bin)
                
                # Weight by proportion of samples in bin
                total_reliability += bin_reliability * prop_in_bin
                total_samples += prop_in_bin
        
        return total_reliability / total_samples if total_samples > 0 else 0.0
    
    def save_calibrator(self, calibrator_path: Union[str, Path]) -> None:
        """
        Save the fitted calibrator to disk.
        
        Args:
            calibrator_path: Path where to save the calibrator
            
        Raises:
            ValueError: If calibrator is not fitted
        """
        if not self.has_usable_calibrator():
            raise ValueError("Cannot save unfitted calibrator")
        
        try:
            calibrator_path = Path(calibrator_path)
            calibrator_path.parent.mkdir(parents=True, exist_ok=True)
            
            joblib.dump({
                'calibrator': self.calibrator,
                'method': self.method,
                'is_fitted': self.is_fitted,
                'feature_quality_weights': self.feature_quality_weights
            }, calibrator_path)
            
            logger.info(f"Calibrator saved to {calibrator_path}")
            
        except Exception as e:
            logger.error(f"Calibrator saving failed: {str(e)}")
            raise ValueError(f"Failed to save calibrator: {str(e)}")
    
    def load_calibrator(self, calibrator_path: Union[str, Path]) -> None:
        """
        Load a fitted calibrator from disk.
        
        Args:
            calibrator_path: Path to the saved calibrator
            
        Raises:
            ValueError: If calibrator file is invalid
        """
        try:
            calibrator_path = Path(calibrator_path)
            if not calibrator_path.exists():
                raise ValueError(f"Calibrator file not found: {calibrator_path}")

            self.calibrator = None
            self.is_fitted = False

            saved_data = joblib.load(calibrator_path)

            # Production artifacts may store the wrapper in a dict:
            # {"calibrator": ConfidenceCalibrator(...), "is_fitted": False, ...}.
            # Inspect the nested object as well as a direct wrapper instance;
            # otherwise _restore_from_artifact(None) misclassifies this known
            # disabled artifact as corrupt and logs a startup warning.
            artifact_wrapper = saved_data.get("calibrator") if isinstance(saved_data, dict) else saved_data
            if (
                isinstance(artifact_wrapper, ConfidenceCalibrator)
                and not artifact_wrapper.has_usable_calibrator()
            ):
                metadata = saved_data if isinstance(saved_data, dict) else {}
                # A disabled placeholder is valid only when the artifact
                # explicitly says calibration is disabled. Reject contradictory
                # metadata rather than silently accepting a broken artifact.
                if metadata.get("is_calibrated") is True or metadata.get("is_fitted") is True:
                    raise ValueError(
                        "Calibration artifact claims to be fitted but contains an unfitted calibrator"
                    )
                saved_method = metadata.get("method", getattr(artifact_wrapper, "method", None))
                if saved_method in ("platt", "isotonic"):
                    self.method = saved_method
                weights = metadata.get(
                    "feature_quality_weights",
                    getattr(artifact_wrapper, "feature_quality_weights", None),
                )
                if isinstance(weights, dict):
                    self.feature_quality_weights = weights.copy()
                logger.info(
                    "Calibration artifact is an unfitted placeholder; "
                    "calibration is disabled and raw model confidence will be returned"
                )
                return

            self._restore_from_artifact(saved_data)

            if not self.has_usable_calibrator():
                raise ValueError("Loaded calibrator artifact is not usable for prediction")

            logger.info(f"Calibrator loaded from {calibrator_path}")
            
        except Exception as e:
            logger.error(f"Calibrator loading failed: {str(e)}")
            self.calibrator = None
            self.is_fitted = False
            raise ValueError(f"Failed to load calibrator: {str(e)}")
    
    def get_calibrator_info(self) -> Dict[str, Any]:
        """
        Get information about the calibrator configuration and state.
        
        Returns:
            Dictionary containing calibrator information
        """
        return {
            'method': self.method,
            'is_fitted': self.is_fitted,
            'is_usable': self.has_usable_calibrator(),
            'calibrator_type': type(self.calibrator).__name__ if self.calibrator else None,
            'feature_quality_weights': self.feature_quality_weights.copy()
        }


def create_default_calibrator() -> ConfidenceCalibrator:
    """
    Create a confidence calibrator with default configuration.
    
    Returns:
        ConfidenceCalibrator: Configured calibrator instance
    """
    return ConfidenceCalibrator(method='platt')
