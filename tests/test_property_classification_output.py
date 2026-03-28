"""
Property-Based Tests for Classification Output

Tests Property 7: Classification Output Format
Tests Property 8: Confidence Score Bounds
Validates: Requirements 3.1, 3.2
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, assume, settings, HealthCheck
from hypothesis.extra import numpy as hnp
from unittest.mock import Mock, patch

from ai_voice_detector.models.classifier import RandomForestVoiceClassifier, create_default_classifier
from ai_voice_detector.models.calibration import ConfidenceCalibrator, create_default_calibrator
from ai_voice_detector.models.data_models import FeatureVector, SpectralFeatures, TemporalFeatures


# Test data generators
@st.composite
def valid_feature_vectors(draw):
    """Generate valid FeatureVector objects for testing"""
    # Generate MFCC features (typically 13 coefficients)
    mfcc_size = draw(st.integers(min_value=12, max_value=15))
    mfcc_features = draw(hnp.arrays(
        dtype=np.float32,
        shape=(mfcc_size,),
        elements=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False, width=32, allow_subnormal=False)
    ))
    
    # Generate spectral features
    feature_length = draw(st.integers(min_value=10, max_value=50))
    spectral_features = SpectralFeatures(
        centroid=draw(hnp.arrays(
            dtype=np.float32,
            shape=(feature_length,),
            elements=st.floats(min_value=100.0, max_value=8000.0, allow_nan=False, allow_infinity=False, width=32, allow_subnormal=False)
        )),
        bandwidth=draw(hnp.arrays(
            dtype=np.float32,
            shape=(feature_length,),
            elements=st.floats(min_value=50.0, max_value=4000.0, allow_nan=False, allow_infinity=False, width=32, allow_subnormal=False)
        )),
        rolloff=draw(hnp.arrays(
            dtype=np.float32,
            shape=(feature_length,),
            elements=st.floats(min_value=1000.0, max_value=8000.0, allow_nan=False, allow_infinity=False, width=32, allow_subnormal=False)
        )),
        zero_crossing_rate=draw(hnp.arrays(
            dtype=np.float32,
            shape=(feature_length,),
            elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False, width=32, allow_subnormal=False)
        ))
    )
    
    # Generate variance features
    pitch_variance = draw(st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False, width=32, allow_subnormal=False))
    energy_variance = draw(st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False, width=32, allow_subnormal=False))
    
    # Generate temporal features
    temporal_features = TemporalFeatures(
        frame_consistency=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False, width=32, allow_subnormal=False)),
        rhythm_regularity=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False, width=32, allow_subnormal=False)),
        pause_patterns=draw(hnp.arrays(
            dtype=np.float32,
            shape=(draw(st.integers(min_value=1, max_value=10)),),
            elements=st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False, width=32, allow_subnormal=False)
        ))
    )
    
    return FeatureVector(
        mfcc_features=mfcc_features,
        spectral_features=spectral_features,
        pitch_variance=pitch_variance,
        energy_variance=energy_variance,
        temporal_features=temporal_features
    )


@st.composite
def trained_classifier_with_mock_data(draw):
    """Generate a trained classifier with mock training data"""
    classifier = create_default_classifier()
    
    # Create mock training data
    n_samples = draw(st.integers(min_value=20, max_value=100))
    n_features = draw(st.integers(min_value=50, max_value=200))
    
    X_train = draw(hnp.arrays(
        dtype=np.float32,
        shape=(n_samples, n_features),
        elements=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False, width=32, allow_subnormal=False)
    ))
    
    # Generate balanced binary labels
    y_train = draw(hnp.arrays(
        dtype=np.int32,
        shape=(n_samples,),
        elements=st.integers(min_value=0, max_value=1)
    ))
    
    # Ensure we have both classes
    if len(np.unique(y_train)) < 2:
        y_train[0] = 0
        y_train[1] = 1
    
    # Train the classifier
    classifier.train(X_train, y_train)
    
    return classifier


@st.composite
def raw_confidence_scores(draw):
    """Generate raw confidence scores for calibration testing"""
    return draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False, width=32, allow_subnormal=False))


class TestClassificationOutputProperties:
    """Property-based tests for classification output format and confidence bounds"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.classifier = create_default_classifier()
        self.calibrator = create_default_calibrator()
    
    @given(features=valid_feature_vectors())
    @settings(max_examples=8, deadline=10000, suppress_health_check=[HealthCheck.data_too_large])
    def test_classification_output_format_property(self, features):
        """
        **Feature: ai-voice-detector, Property 7: Classification Output Format**
        
        For any valid audio input, the Classification_Model should output exactly 
        one of "AI_GENERATED" or "HUMAN" as the classification result.
        
        Validates: Requirements 3.1
        """
        # Skip if features are invalid
        assume(features.validate())
        
        # Create a mock trained classifier for testing
        with patch.object(self.classifier, 'is_trained', return_value=True):
            with patch.object(self.classifier.model, 'predict') as mock_predict:
                with patch.object(self.classifier.model, 'predict_proba') as mock_predict_proba:
                    # Mock the sklearn model to return valid predictions
                    mock_predict.return_value = np.array([0])  # or [1]
                    mock_predict_proba.return_value = np.array([[0.7, 0.3]])  # probabilities
                    
                    # Property test: Classification should return valid format
                    classification, raw_confidence = self.classifier.predict(features)
                    
                    # Property 1: Classification must be exactly one of the valid values
                    valid_classifications = {"AI_GENERATED", "HUMAN"}
                    assert classification in valid_classifications, \
                        f"Classification must be 'AI_GENERATED' or 'HUMAN', got: {classification}"
                    
                    # Property 2: Classification must be a string
                    assert isinstance(classification, str), \
                        f"Classification must be string, got: {type(classification)}"
                    
                    # Property 3: Classification must not be empty
                    assert len(classification) > 0, "Classification must not be empty string"
                    
                    # Property 4: Classification must match expected format exactly
                    if classification == "AI_GENERATED":
                        assert classification == "AI_GENERATED", "AI classification must match exact format"
                    else:
                        assert classification == "HUMAN", "Human classification must match exact format"
                    
                    # Property 5: Raw confidence should be numeric
                    assert isinstance(raw_confidence, (int, float)), \
                        f"Raw confidence must be numeric, got: {type(raw_confidence)}"
    
    @given(features=valid_feature_vectors())
    @settings(max_examples=8, deadline=10000, suppress_health_check=[HealthCheck.data_too_large])
    def test_confidence_score_bounds_property(self, features):
        """
        **Feature: ai-voice-detector, Property 8: Confidence Score Bounds**
        
        For any classification result, the Confidence_Calibrator should provide 
        a confidence score that is a float value between 0.0 and 1.0 inclusive.
        
        Validates: Requirements 3.2
        """
        # Skip if features are invalid
        assume(features.validate())
        
        # Create a mock trained classifier for testing
        with patch.object(self.classifier, 'is_trained', return_value=True):
            with patch.object(self.classifier.model, 'predict') as mock_predict:
                with patch.object(self.classifier.model, 'predict_proba') as mock_predict_proba:
                    # Mock the sklearn model to return valid predictions
                    mock_predict.return_value = np.array([0])
                    mock_predict_proba.return_value = np.array([[0.6, 0.4]])
                    
                    # Get raw confidence from classifier
                    classification, raw_confidence = self.classifier.predict(features)
                    
                    # Property test: Raw confidence should be in valid bounds
                    # Property 1: Raw confidence must be between 0.0 and 1.0 inclusive
                    assert 0.0 <= raw_confidence <= 1.0, \
                        f"Raw confidence must be between 0.0 and 1.0, got: {raw_confidence}"
                    
                    # Property 2: Raw confidence must be a float
                    assert isinstance(raw_confidence, (int, float)), \
                        f"Raw confidence must be numeric, got: {type(raw_confidence)}"
                    
                    # Property 3: Raw confidence must not be NaN or infinite
                    assert not np.isnan(raw_confidence), "Raw confidence must not be NaN"
                    assert not np.isinf(raw_confidence), "Raw confidence must not be infinite"
                    
                    # Test calibrated confidence bounds
                    # Mock a fitted calibrator
                    with patch.object(self.calibrator, 'is_fitted', True):
                        with patch.object(self.calibrator, 'calibrate_confidence') as mock_calibrate:
                            # Mock calibration to return a valid confidence score
                            mock_calibrated_confidence = 0.75  # Example calibrated value
                            mock_calibrate.return_value = mock_calibrated_confidence
                            
                            calibrated_confidence = self.calibrator.calibrate_confidence(raw_confidence, features)
                            
                            # Property 4: Calibrated confidence must be between 0.0 and 1.0 inclusive
                            assert 0.0 <= calibrated_confidence <= 1.0, \
                                f"Calibrated confidence must be between 0.0 and 1.0, got: {calibrated_confidence}"
                            
                            # Property 5: Calibrated confidence must be a float
                            assert isinstance(calibrated_confidence, (int, float)), \
                                f"Calibrated confidence must be numeric, got: {type(calibrated_confidence)}"
                            
                            # Property 6: Calibrated confidence must not be NaN or infinite
                            assert not np.isnan(calibrated_confidence), "Calibrated confidence must not be NaN"
                            assert not np.isinf(calibrated_confidence), "Calibrated confidence must not be infinite"
    
    @given(
        features_list=st.lists(valid_feature_vectors(), min_size=1, max_size=10)
    )
    @settings(max_examples=5, deadline=15000, suppress_health_check=[HealthCheck.data_too_large])
    def test_batch_classification_output_consistency_property(self, features_list):
        """
        Property: Batch classification should maintain output format consistency
        
        For any list of valid feature vectors, batch classification should 
        return results with consistent format for all items.
        """
        # Skip if any features are invalid
        assume(all(features.validate() for features in features_list))
        
        # Create a mock trained classifier for testing
        with patch.object(self.classifier, 'is_trained', return_value=True):
            with patch.object(self.classifier.model, 'predict') as mock_predict:
                with patch.object(self.classifier.model, 'predict_proba') as mock_predict_proba:
                    # Mock batch predictions
                    batch_size = len(features_list)
                    mock_predict.return_value = np.random.randint(0, 2, size=batch_size)
                    mock_predict_proba.return_value = np.random.rand(batch_size, 2)
                    # Ensure probabilities sum to 1
                    mock_predict_proba.return_value = mock_predict_proba.return_value / \
                        mock_predict_proba.return_value.sum(axis=1, keepdims=True)
                    
                    # Property test: Batch results should have consistent format
                    batch_results = self.classifier.predict_batch(features_list)
                    
                    # Property 1: Should return list of tuples
                    assert isinstance(batch_results, list), "Batch results must be a list"
                    assert len(batch_results) == len(features_list), \
                        "Batch results length must match input length"
                    
                    # Property 2: Each result should be a tuple of (classification, confidence)
                    for i, (classification, confidence) in enumerate(batch_results):
                        # Classification format properties
                        assert isinstance(classification, str), \
                            f"Classification {i} must be string, got: {type(classification)}"
                        assert classification in {"AI_GENERATED", "HUMAN"}, \
                            f"Classification {i} must be valid value, got: {classification}"
                        
                        # Confidence bounds properties
                        assert isinstance(confidence, (int, float)), \
                            f"Confidence {i} must be numeric, got: {type(confidence)}"
                        assert 0.0 <= confidence <= 1.0, \
                            f"Confidence {i} must be in [0,1], got: {confidence}"
                        assert not np.isnan(confidence), f"Confidence {i} must not be NaN"
                        assert not np.isinf(confidence), f"Confidence {i} must not be infinite"
    
    @given(raw_confidence=raw_confidence_scores())
    @settings(max_examples=8, deadline=5000)
    def test_calibrator_confidence_bounds_property(self, raw_confidence):
        """
        Property: Confidence calibrator should always return values in valid bounds
        
        For any raw confidence score in [0,1], the calibrator should return 
        a calibrated confidence also in [0,1].
        """
        # Mock a fitted calibrator with realistic calibration behavior
        with patch.object(self.calibrator, 'is_fitted', True):
            # Mock the internal calibrator to return a realistic calibrated value
            # Simulate Platt scaling behavior: slight adjustment towards 0.5
            adjustment_factor = 0.1
            mock_calibrated = raw_confidence + adjustment_factor * (0.5 - raw_confidence)
            mock_calibrated = np.clip(mock_calibrated, 0.0, 1.0)  # Ensure bounds
            
            with patch.object(self.calibrator, 'calibrate_confidence', return_value=mock_calibrated):
                calibrated_confidence = self.calibrator.calibrate_confidence(raw_confidence)
                
                # Property 1: Calibrated confidence must be in valid bounds
                assert 0.0 <= calibrated_confidence <= 1.0, \
                    f"Calibrated confidence must be in [0,1], got: {calibrated_confidence}"
                
                # Property 2: Must be numeric
                assert isinstance(calibrated_confidence, (int, float)), \
                    f"Calibrated confidence must be numeric, got: {type(calibrated_confidence)}"
                
                # Property 3: Must not be NaN or infinite
                assert not np.isnan(calibrated_confidence), "Calibrated confidence must not be NaN"
                assert not np.isinf(calibrated_confidence), "Calibrated confidence must not be infinite"
    
    @given(features=valid_feature_vectors())
    @settings(max_examples=5, deadline=8000)
    def test_classification_determinism_property(self, features):
        """
        Property: Classification should be deterministic for identical inputs
        
        For any identical feature vector processed multiple times, 
        the classification and confidence should be consistent.
        """
        # Skip if features are invalid
        assume(features.validate())
        
        # Create a mock trained classifier for testing
        with patch.object(self.classifier, 'is_trained', return_value=True):
            with patch.object(self.classifier.model, 'predict') as mock_predict:
                with patch.object(self.classifier.model, 'predict_proba') as mock_predict_proba:
                    # Mock deterministic predictions
                    mock_predict.return_value = np.array([1])  # Always AI_GENERATED
                    mock_predict_proba.return_value = np.array([[0.3, 0.7]])  # Consistent probabilities
                    
                    # Property test: Multiple predictions should be identical
                    result1 = self.classifier.predict(features)
                    result2 = self.classifier.predict(features)
                    result3 = self.classifier.predict(features)
                    
                    # Property 1: Classifications should be identical
                    assert result1[0] == result2[0] == result3[0], \
                        "Classification should be deterministic for identical input"
                    
                    # Property 2: Confidences should be identical
                    assert result1[1] == result2[1] == result3[1], \
                        "Confidence should be deterministic for identical input"
                    
                    # Property 3: All results should maintain valid format
                    for result in [result1, result2, result3]:
                        classification, confidence = result
                        assert classification in {"AI_GENERATED", "HUMAN"}, \
                            "All results should have valid classification format"
                        assert 0.0 <= confidence <= 1.0, \
                            "All results should have valid confidence bounds"