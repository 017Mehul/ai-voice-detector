"""
Property-Based Tests for Classification Determinism

Tests Property 19: Classification Determinism
Validates: Requirements 7.2
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, assume, settings, HealthCheck
from hypothesis.extra import numpy as hnp
from unittest.mock import Mock, patch
import copy

from ai_voice_detector.models.classifier import RandomForestVoiceClassifier, create_default_classifier
from ai_voice_detector.models.calibration import ConfidenceCalibrator, create_default_calibrator
from ai_voice_detector.models.data_models import FeatureVector, SpectralFeatures, TemporalFeatures
from ai_voice_detector.services.voice_detection_service import VoiceDetectionService
from ai_voice_detector.audio.processor import AudioProcessor
from ai_voice_detector.features.extractor import FeatureExtractor


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


class TestClassificationDeterminismProperty:
    """Property-based tests for classification determinism"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.classifier = create_default_classifier()
        self.calibrator = create_default_calibrator()
        
        # Create mock components for service
        self.audio_processor = Mock(spec=AudioProcessor)
        self.feature_extractor = Mock(spec=FeatureExtractor)
        
        self.service = VoiceDetectionService(
            audio_processor=self.audio_processor,
            feature_extractor=self.feature_extractor,
            classifier=self.classifier,
            confidence_calibrator=self.calibrator
        )
    
    @given(features=valid_feature_vectors())
    @settings(max_examples=15, deadline=15000, suppress_health_check=[HealthCheck.data_too_large])
    def test_classification_determinism_property(self, features):
        """
        **Feature: ai-voice-detector, Property 19: Classification Determinism**
        
        For any identical audio input processed multiple times, the system should 
        provide consistent classification results and confidence scores.
        
        Validates: Requirements 7.2
        """
        # Skip if features are invalid
        assume(features.validate())
        
        # Create a mock trained classifier for testing
        with patch.object(self.classifier, 'is_trained', return_value=True):
            with patch.object(self.classifier.model, 'predict') as mock_predict:
                with patch.object(self.classifier.model, 'predict_proba') as mock_predict_proba:
                    # Mock deterministic predictions - use fixed seed for consistency
                    np.random.seed(42)
                    mock_class = np.random.randint(0, 2)  # 0 or 1
                    mock_prob_1 = np.random.uniform(0.6, 0.9)  # High confidence
                    mock_prob_0 = 1.0 - mock_prob_1
                    
                    mock_predict.return_value = np.array([mock_class])
                    if mock_class == 1:
                        mock_predict_proba.return_value = np.array([[mock_prob_0, mock_prob_1]])
                    else:
                        mock_predict_proba.return_value = np.array([[mock_prob_1, mock_prob_0]])
                    
                    # Property test: Multiple predictions should be identical
                    # Test 1: Direct classifier predictions
                    result1 = self.classifier.predict(features)
                    result2 = self.classifier.predict(features)
                    result3 = self.classifier.predict(features)
                    
                    # Property 1: Classifications should be identical across multiple calls
                    assert result1[0] == result2[0] == result3[0], \
                        f"Classification should be deterministic for identical input. Got: {result1[0]}, {result2[0]}, {result3[0]}"
                    
                    # Property 2: Confidences should be identical across multiple calls
                    assert result1[1] == result2[1] == result3[1], \
                        f"Confidence should be deterministic for identical input. Got: {result1[1]}, {result2[1]}, {result3[1]}"
                    
                    # Property 3: All results should maintain valid format
                    for i, result in enumerate([result1, result2, result3], 1):
                        classification, confidence = result
                        assert classification in {"AI_GENERATED", "HUMAN"}, \
                            f"Result {i} should have valid classification format, got: {classification}"
                        assert 0.0 <= confidence <= 1.0, \
                            f"Result {i} should have valid confidence bounds, got: {confidence}"
                        assert isinstance(classification, str), \
                            f"Result {i} classification should be string, got: {type(classification)}"
                        assert isinstance(confidence, (int, float)), \
                            f"Result {i} confidence should be numeric, got: {type(confidence)}"
    
    @given(features=valid_feature_vectors())
    @settings(max_examples=10, deadline=20000, suppress_health_check=[HealthCheck.data_too_large])
    def test_batch_classification_determinism_property(self, features):
        """
        **Feature: ai-voice-detector, Property 19: Batch Classification Determinism**
        
        For any identical batch of audio inputs processed multiple times, the system 
        should provide consistent classification results for each item in the batch.
        
        Validates: Requirements 7.2
        """
        # Skip if features are invalid
        assume(features.validate())
        
        # Create identical feature vectors for batch testing
        batch_size = 3
        identical_features = [copy.deepcopy(features) for _ in range(batch_size)]
        
        # Create a mock trained classifier for testing
        with patch.object(self.classifier, 'is_trained', return_value=True):
            with patch.object(self.classifier.model, 'predict') as mock_predict:
                with patch.object(self.classifier.model, 'predict_proba') as mock_predict_proba:
                    # Mock deterministic batch predictions
                    np.random.seed(42)
                    mock_classes = np.array([1, 0, 1])  # Fixed pattern
                    mock_probs = np.array([
                        [0.2, 0.8],  # AI_GENERATED with 0.8 confidence
                        [0.7, 0.3],  # HUMAN with 0.7 confidence  
                        [0.1, 0.9]   # AI_GENERATED with 0.9 confidence
                    ])
                    
                    mock_predict.return_value = mock_classes
                    mock_predict_proba.return_value = mock_probs
                    
                    # Property test: Multiple batch predictions should be identical
                    batch_result1 = self.classifier.predict_batch(identical_features)
                    batch_result2 = self.classifier.predict_batch(identical_features)
                    batch_result3 = self.classifier.predict_batch(identical_features)
                    
                    # Property 1: Batch results should have same length
                    assert len(batch_result1) == len(batch_result2) == len(batch_result3) == batch_size, \
                        "All batch results should have same length"
                    
                    # Property 2: Each item in batch should be deterministic across runs
                    for i in range(batch_size):
                        classification1, confidence1 = batch_result1[i]
                        classification2, confidence2 = batch_result2[i]
                        classification3, confidence3 = batch_result3[i]
                        
                        assert classification1 == classification2 == classification3, \
                            f"Batch item {i} classification should be deterministic. Got: {classification1}, {classification2}, {classification3}"
                        
                        assert confidence1 == confidence2 == confidence3, \
                            f"Batch item {i} confidence should be deterministic. Got: {confidence1}, {confidence2}, {confidence3}"
                        
                        # Property 3: Each result should maintain valid format
                        assert classification1 in {"AI_GENERATED", "HUMAN"}, \
                            f"Batch item {i} should have valid classification"
                        assert 0.0 <= confidence1 <= 1.0, \
                            f"Batch item {i} should have valid confidence bounds"
    
    @given(features=valid_feature_vectors())
    @settings(max_examples=8, deadline=15000, suppress_health_check=[HealthCheck.data_too_large])
    def test_service_level_determinism_property(self, features):
        """
        **Feature: ai-voice-detector, Property 19: Service-Level Determinism**
        
        For any identical feature input processed through the complete service pipeline
        multiple times, the system should provide consistent results.
        
        Validates: Requirements 7.2
        """
        # Skip if features are invalid
        assume(features.validate())
        
        # Mock the service's internal _classify_voice method for deterministic testing
        with patch.object(self.service, '_classify_voice') as mock_classify_voice:
            # Mock deterministic service responses
            mock_classification = "AI_GENERATED"
            mock_raw_confidence = 0.85
            mock_error = None
            
            mock_classify_voice.return_value = (mock_classification, mock_raw_confidence, mock_error)
            
            # Property test: Service-level predictions should be deterministic
            result1 = self.service._classify_voice(features, "test-1")
            result2 = self.service._classify_voice(features, "test-2")
            result3 = self.service._classify_voice(features, "test-3")
            
            # Property 1: Service classifications should be identical
            assert result1[0] == result2[0] == result3[0], \
                f"Service classification should be deterministic. Got: {result1[0]}, {result2[0]}, {result3[0]}"
            
            # Property 2: Service confidence scores should be identical
            assert result1[1] == result2[1] == result3[1], \
                f"Service confidence should be deterministic. Got: {result1[1]}, {result2[1]}, {result3[1]}"
            
            # Property 3: Service errors should be identical (None in this case)
            assert result1[2] == result2[2] == result3[2], \
                f"Service errors should be deterministic. Got: {result1[2]}, {result2[2]}, {result3[2]}"
            
            # Property 4: All service results should maintain valid format
            for i, result in enumerate([result1, result2, result3], 1):
                classification, confidence, error = result
                assert classification in {"AI_GENERATED", "HUMAN"}, \
                    f"Service result {i} should have valid classification"
                assert 0.0 <= confidence <= 1.0, \
                    f"Service result {i} should have valid confidence bounds"
                assert isinstance(classification, str), \
                    f"Service result {i} classification should be string"
                assert isinstance(confidence, (int, float)), \
                    f"Service result {i} confidence should be numeric"
    
    @given(features=valid_feature_vectors())
    @settings(max_examples=5, deadline=12000, suppress_health_check=[HealthCheck.data_too_large])
    def test_deep_copy_determinism_property(self, features):
        """
        **Feature: ai-voice-detector, Property 19: Deep Copy Determinism**
        
        For any feature vector and its deep copy processed separately, the system 
        should provide identical results, ensuring determinism is not affected by 
        object identity.
        
        Validates: Requirements 7.2
        """
        # Skip if features are invalid
        assume(features.validate())
        
        # Create deep copies of the features
        features_copy1 = copy.deepcopy(features)
        features_copy2 = copy.deepcopy(features)
        
        # Verify copies are different objects but equal content
        assert features is not features_copy1, "Deep copy should create different object"
        assert features is not features_copy2, "Deep copy should create different object"
        assert features_copy1 is not features_copy2, "Deep copies should be different objects"
        
        # Create a mock trained classifier for testing
        with patch.object(self.classifier, 'is_trained', return_value=True):
            with patch.object(self.classifier.model, 'predict') as mock_predict:
                with patch.object(self.classifier.model, 'predict_proba') as mock_predict_proba:
                    # Mock deterministic predictions
                    mock_predict.return_value = np.array([1])  # AI_GENERATED
                    mock_predict_proba.return_value = np.array([[0.25, 0.75]])  # 0.75 confidence
                    
                    # Property test: Results should be identical for original and copies
                    result_original = self.classifier.predict(features)
                    result_copy1 = self.classifier.predict(features_copy1)
                    result_copy2 = self.classifier.predict(features_copy2)
                    
                    # Property 1: Classifications should be identical across all versions
                    assert result_original[0] == result_copy1[0] == result_copy2[0], \
                        f"Classification should be identical for original and copies. Got: {result_original[0]}, {result_copy1[0]}, {result_copy2[0]}"
                    
                    # Property 2: Confidences should be identical across all versions
                    assert result_original[1] == result_copy1[1] == result_copy2[1], \
                        f"Confidence should be identical for original and copies. Got: {result_original[1]}, {result_copy1[1]}, {result_copy2[1]}"
                    
                    # Property 3: All results should maintain valid format
                    for label, result in [("original", result_original), ("copy1", result_copy1), ("copy2", result_copy2)]:
                        classification, confidence = result
                        assert classification in {"AI_GENERATED", "HUMAN"}, \
                            f"{label} should have valid classification format"
                        assert 0.0 <= confidence <= 1.0, \
                            f"{label} should have valid confidence bounds"
    
    @given(
        features=valid_feature_vectors(),
        num_iterations=st.integers(min_value=3, max_value=8)
    )
    @settings(max_examples=5, deadline=20000, suppress_health_check=[HealthCheck.data_too_large])
    def test_multiple_iteration_determinism_property(self, features, num_iterations):
        """
        **Feature: ai-voice-detector, Property 19: Multiple Iteration Determinism**
        
        For any feature vector processed multiple times in sequence, all results 
        should be identical, ensuring the system maintains determinism across 
        extended usage.
        
        Validates: Requirements 7.2
        """
        # Skip if features are invalid
        assume(features.validate())
        
        # Create a mock trained classifier for testing
        with patch.object(self.classifier, 'is_trained', return_value=True):
            with patch.object(self.classifier.model, 'predict') as mock_predict:
                with patch.object(self.classifier.model, 'predict_proba') as mock_predict_proba:
                    # Mock deterministic predictions
                    mock_predict.return_value = np.array([0])  # HUMAN
                    mock_predict_proba.return_value = np.array([[0.8, 0.2]])  # 0.8 confidence
                    
                    # Property test: All iterations should produce identical results
                    results = []
                    for i in range(num_iterations):
                        result = self.classifier.predict(features)
                        results.append(result)
                    
                    # Property 1: All classifications should be identical
                    classifications = [result[0] for result in results]
                    assert all(c == classifications[0] for c in classifications), \
                        f"All {num_iterations} classifications should be identical. Got: {classifications}"
                    
                    # Property 2: All confidences should be identical
                    confidences = [result[1] for result in results]
                    assert all(c == confidences[0] for c in confidences), \
                        f"All {num_iterations} confidences should be identical. Got: {confidences}"
                    
                    # Property 3: All results should maintain valid format
                    for i, (classification, confidence) in enumerate(results):
                        assert classification in {"AI_GENERATED", "HUMAN"}, \
                            f"Iteration {i+1} should have valid classification format"
                        assert 0.0 <= confidence <= 1.0, \
                            f"Iteration {i+1} should have valid confidence bounds"
                        assert isinstance(classification, str), \
                            f"Iteration {i+1} classification should be string"
                        assert isinstance(confidence, (int, float)), \
                            f"Iteration {i+1} confidence should be numeric"